# 第7章：通过 QNN + ExecuTorch 跑 BitNet 推理

## 1. 概述

本章使用 Qualcomm 的 QNN 后端 + ExecuTorch 框架，在 Snapdragon 8 Gen 3 上跑 Microsoft 的 BitNet b1.58-2B-4T 模型。核心加速来自 T-MAN（Table-lookup MAtrix multiplicatioN）自定义算子包，它利用 HVX 的 VLUT16 指令把三值权重的矩阵乘法变成查表操作。

| 项目 | 值 |
|------|-----|
| 模型 | microsoft/bitnet-b1.58-2B-4T（20 亿参数，1.58-bit 权重）|
| 量化 | 16a4w（16-bit activation, 4-bit weight）|
| 框架 | ExecuTorch (kaleid-liner fork) + QNN 2.44 |
| 自定义算子 | TMAN Op Package（HVX VLUT16 查表 GEMV）|
| 设备 | Snapdragon 8 Gen 3, Hexagon v75 DSP |

### 实测结果

```
Prompt: "What is 1+1?"
Output: "1+1 is a mathematical operation that represents the addition
        of two numbers together. In this case, it is the sum of 1 and 1,
        which equals 2. This is a fundamental concept in arithmetic..."

生成速度:  39.3 tokens/sec (decode)
首token延迟: 0.47s
模型加载:   2.3s
.pte 大小:  1.3 GB
```

## 2. 架构

整个流程分两阶段：**x86 主机编译** 和 **设备端推理**。

```
x86 主机（AOT 编译）                    设备端（推理）
┌────────────────────────┐             ┌────────────────────────┐
│ bitnet.py              │             │ qnn_llama_runner       │
│   ↓ torch.export       │             │   ↓ 加载 .pte         │
│ FX Graph               │             │ ExecuTorch Runtime     │
│   ↓ QNN partitioner    │  adb push   │   ↓ QNN HTP delegate  │
│ QNN context binary     │ ─────────→  │ libQnnHtp.so           │
│   ↓ serialize          │             │   ↓ 调用自定义算子      │
│ kv_llama_qnn.pte       │             │ libQnnTMANOpPackage.so │
└────────────────────────┘             │   ↓ HVX VLUT16        │
                                       │ Hexagon v75 DSP        │
                                       └────────────────────────┘
```

**为什么不在 C 里手写 QNN 图？**

一开始我们试过在设备端用 C 代码调 `QnnGraph_addNode` 手动构建 QNN 图——这相当于跳过整个 Python 编译链，从零重写 executorch 的图构建逻辑。权重 packing 极其复杂（VTCM-aware 分块、bit-plane 拆分、nibble 交织），Python 端的 `hvx_preprocess_weights` 有几百行逻辑，不可能在 C 里手写。正确做法是走 executorch 的 AOT 导出流程。

## 3. 运行步骤

### 前置条件

- executorch 编译环境已搭好（参考 `tools/executorch/` 的 README）
- `tools/executorch/.venv/` Python 虚拟环境已创建（含 torch 2.7.0）
- `tools/executorch/build-x86/` 和 `build-android/` 已编译
- x86 TMAN op package 已编译：`tools/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackage/build/x86_64-linux-clang/libQnnTMANOpPackage.so`
- `qnn_llama_runner` 已用 `-DSUPPORT_REGEX_LOOKAHEAD=ON` 重新编译（HF tokenizer 的正则需要 PCRE2）

### 步骤 1：编译 TMAN Op Package（Hexagon DSP 端）

```bash
./build.sh
```

产物：
- `build/hexagon-v75/libTMANOpPackage_htp.so` — DSP 端 HVX 内核
- `build/aarch64/libTMANOpPackage_cpu.so` — ARM 端 fallback

### 步骤 2：导出 .pte 模型（x86 主机）

```bash
./run_export.sh
```

这一步耗时约 4 分钟，做了以下事情：
1. 加载 HuggingFace 的 `microsoft/bitnet-b1.58-2B-4T` checkpoint
2. 用 `--use_tman` 把 Linear 层替换为 TMANLinear 自定义算子
3. 用 `--ptq 16a4w` 做 16-bit activation / 4-bit weight 量化
4. 通过 QNN partitioner 把算子图下沉到 HTP 后端
5. 序列化为 `.pte` 文件

产物：
- `artifacts/kv_llama_qnn.pte`（1.3 GB）— ExecuTorch 模型文件
- `artifacts/kv_llama_qnn_quant_attrs.txt` — 输出层量化参数（scale/zero_point）

### 步骤 3：推送到设备并推理

```bash
./run_device.sh "What is the meaning of life?"
./run_device.sh "用中文回答：1+1等于几？" 256   # 第二个参数是 seq_len
```

推送到设备的文件清单（全部放在 `/data/local/tmp/llama/`）：

| 文件 | 来源 | 说明 |
|------|------|------|
| `qnn_llama_runner` | executorch build-android | 推理主程序（aarch64）|
| `kv_llama_qnn.pte` | run_export.sh 产物 | 编译后的模型 |
| `tokenizer.json` | HuggingFace 模型目录 | HF 格式 tokenizer |
| `libQnnTMANOpPackage.so` | build.sh 产物 | TMAN HVX 内核（**必须用这个名字**）|
| `libQnnHtp.so` | QNN SDK | HTP 后端运行时 |
| `libQnnHtpPrepare.so` | QNN SDK | HTP 图准备 |
| `libQnnSystem.so` | QNN SDK | QNN 系统后端 |
| `libQnnHtpV75Stub.so` | QNN SDK | HTP v75 通信桩 |
| `libQnnHtpV75Skel.so` | QNN SDK | Hexagon DSP 侧 skel |
| `libqnn_executorch_backend.so` | executorch build-android | ExecuTorch QNN 后端 |

## 4. 踩坑记录

调通这个流程遇到了不少问题，记录如下供参考。

### 4.1 flatc 不在 PATH

导出脚本内部调用 FlatBuffers 编译器（`flatc`）来序列化 QNN compiler spec。它在 `build-x86/third-party/flatbuffers/flatc`，需要手动加到 PATH。

**修复**：`run_export.sh` 中添加 `export PATH=$(pwd)/build-x86/third-party/flatbuffers:$PATH`

### 4.2 x86 端需要 TMAN Op Package

导出过程中 QNN 在 x86 主机上编译图，需要注册 TMAN 自定义算子包。硬编码路径在 `QnnBackendCommon.cpp:77` 是设备端路径，x86 上需要通过环境变量覆盖：

```bash
export QNN_OP_PACKAGE_PATHS="$TMAN_X86_SO:TMANOpPackageInterfaceProvider"
```

注意：**不能**加 `:HTP` 后缀！x86 上的 RouterX86 不支持 HTP target，加了会报 error 4003。

### 4.3 quant_attrs_map KeyError

`bitnet.py` 第 469 行访问 `n.meta[QCOM_QUANT_ATTRS_MAP]` 时，某些 output 节点没有这个 metadata。

**修复**：加了 `and QCOM_QUANT_ATTRS_MAP in n.meta` 条件检查（改动在 `tools/executorch` 中）。

### 4.4 必须加 --ptq 16a4w

不加量化时，模型导出的 KV cache 是 float32，但 `qnn_llama_runner` 的 IoManager 假设是 uint8，运行时会 segfault。加 `--ptq 16a4w` 后 KV cache 也被量化，类型匹配，同时 .pte 从 1.8GB 降到 1.3GB。

### 4.5 gflags 定义 bug：kv_updater

`qnn_llama_runner.cpp` 第 56-59 行的 `DEFINE_string` 把 default 值和 description 写反了。必须显式传 `-kv_updater SmartMask`，否则 KV cache 更新逻辑不正确。

### 4.6 HF Tokenizer 正则不兼容 RE2

HuggingFace tokenizer 用了 `(?!` 负向前瞻（negative lookahead），RE2 不支持。需要用 `-DSUPPORT_REGEX_LOOKAHEAD=ON` 重编 `qnn_llama_runner`，启用 PCRE2 回退。

## 5. T-MAN 优化细节（供 direct-bitnet 实验参考）

本节拆解 T-MAN 的 HVX 内核实现，为 ch10-direct-bitnet 手写实验提供技术参考。

### 5.1 核心思路：用 VLUT16 替代乘法

三值权重 w ∈ {-1, 0, +1}，传统做法是反量化再做浮点乘加。T-MAN 的洞察是：

> 把 4 个 activation 分为一组（g=4），4 个三值权重的所有组合只有 3⁴=81 种。
> 预先算好所有可能的点积结果，存成查找表，运行时用权重做索引直接查表。

实际上 VLUT16 指令只支持 16 条目的表，所以用 4-bit 索引（2⁴=16），每个权重用 1 bit 编码（忽略 0，用 bit-plane 分别处理）。

### 5.2 三个算子的数据流

一个 PyTorch Linear 层被拆成 5 个 QNN 节点：

```
输入 X (fp16)
  ↓
Convert (fp16)           ← QNN 内置算子
  ↓
TMANPrecompute           ← 自定义算子 #1
  ├→ L (int16 LUT)       查找表，每 4 个 activation 一张 16 条目表
  ├→ ls (float)           activation 的量化 scale
  └→ lb (float)           对称调整的 bias
  ↓
TMANLinear               ← 自定义算子 #2（核心计算）
  输入: L, packed_weight, scales
  输出: C (float32)       累加结果
  ↓
TMANFinalize             ← 自定义算子 #3
  输入: C (多个 bit-plane 的部分和)
  输出: Y (fp16)          bit-serial 求和 + scale + 转 fp16
  ↓
Convert (UFIXED_POINT_16) ← QNN 内置算子
```

### 5.3 Precompute：构建查找表

`hvx_lut_ctor()` 的逻辑（`hvx_funcs.h`）：

```
对每 4 个 activation (x0, x1, x2, x3)：
  LUT[0000] = 0
  LUT[0001] = x0
  LUT[0010] = x1
  LUT[0011] = x0 + x1
  LUT[0100] = x2
  ...
  LUT[1111] = x0 + x1 + x2 + x3
```

- 输入 fp16 activation → 转 int16（乘以 scale 取整）
- 生成 16 条目 int16 查找表
- **交织存储**：8 组 LUT（来自 8 个不同的 activation 四元组）交织到一个 128 字节 HVX 向量中
- 这样一次 `vlut16` 指令可以并行查 128 个字节（64 个 even + 64 个 odd）

### 5.4 Linear：VLUT16 查表 GEMV

`hvx_tbl()` 的核心循环（`TMANLinear.cpp`）：

```cpp
// 伪代码
for tile_p in range(0, M, TileP):        // 沿输出维度分块
  for q in range(0, K/g, TileQ):         // 沿输入维度分块
    w_vec = load_weight(tile_p, q);      // 加载 packed 权重（4-bit nibble）
    l_vec = load_lut(q);                 // 加载预计算的 LUT
    result = Q6_Wh_vlut16_VbVhR_nomatch(w_vec, l_vec, ...);
    acc = Q6_Ww_vaddacc_WwVhVh(acc, result_hi, result_lo);  // int32 累加
  // 累加完一个 tile 后，转 float 并乘 scale
  c_vec = Q6_Vsf_equals_Vw(acc);                    // int32 → float
  c_vec = Q6_Vqf32_vmpy_VsfVsf(c_vec, ls_vec);     // × activation scale
  c_vec = Q6_Vqf32_vmpy_Vqf32Vqf32(c_vec, s_vec);  // × weight scale
```

**关键细节**：
- `vlut16` 一条指令完成 128 字节的并行查表，替代了 128 次乘法
- 累加用 int32（`vaddacc`），避免 int16 溢出
- 最终用 qf32（quasi-float，v75 不支持原生 f32 HVX 运算）做 scale 乘法
- 用 `l2fetch` 和 `Q6_dcfetch_A` 预取下一轮的权重和 LUT，与当前计算重叠

### 5.5 Finalize：bit-serial 求和

权重被拆成多个 bit-plane（BitNet 是 2 bit），每个 bit-plane 独立走 Precompute → Linear 产生一个部分和。Finalize 负责：

```
Y = C_bit0 × 1 + C_bit1 × 2 + ... + C_bitN × 2^N
```

然后转回 fp16 输出。

### 5.6 权重 packing（Python 端预处理）

`hvx_preprocess_weights()`（`utils.py`）在导出时执行，产物直接嵌入 .pte：

```python
# 1. 三值权重 {-1, 0, +1} → 映射到 {1, 2, 3}（+2 偏移）
# 2. 拆 bit-plane：2-bit 权重拆成 bit0 和 bit1
# 3. 每 4 个权重打包成 1 个 nibble：
#    nibble = w[0] | (w[1]<<1) | (w[2]<<2) | (w[3]<<3)
# 4. 多层 reshape + transpose 适配 HVX 向量布局：
#    目标形状: (M // vec_p, 2, 2, vec_c, K//g)
#    确保 HVX 加载一个 128B 向量时，字节 i 和 i+64 索引同一张 LUT
```

这是最复杂的部分——手写实现时需要精确复刻这个布局，否则 `vlut16` 的索引会错位。

### 5.7 分块策略与 VTCM 管理

| 参数 | 含义 | 取值 |
|------|------|------|
| TileP | 输出维度分块（M 方向）| 由 `_decide_tile_size` 根据 VTCM 大小和线程数决定 |
| TileQ | 输入维度分块（K/g 方向）| `TileK / g`，例如 256/4 = 64 |
| g | 每组 activation 数 | 4（VLUT16 用 4-bit 索引）|
| bits | 权重 bit-plane 数 | 2（BitNet 三值）|
| VTCM 大小 | 片上高速缓存 | 8 MB，6 线程共享 |

**VTCM 分配**（通过 QNN Op Package 的 `Tcm()` 声明）：
- `l`（LUT）→ VTCM
- `qweight`（打包权重）→ VTCM
- `scales`（缩放因子）→ VTCM
- 输出 `y` → DDR（因为后续 Add 等算子在 MainMemory 工作）

TileP 的选择确保每个线程的工作集（权重 tile + LUT + 累加器）不超过 `VTCM_SIZE / N_THREADS ≈ 1.3 MB`。

### 5.8 QNN 图级优化

QNN HTP 编译器在 TMAN 算子之外还做了以下优化：

| 优化 | 说明 |
|------|------|
| AUTOSPLIT | 大 tensor 自动沿 M 维度切分，适配 VTCM |
| 图调度 | Precompute/Linear/Finalize 流水化，层间重叠 |
| 6 线程并行 | TileP 按线程数等分，每个 HVX 线程独立处理一个 tile |
| Spill/Fill DMA | VTCM 装不下时自动 DMA 换入换出（本次编译：spill=8.2MB, fill=8.2MB）|
| 算子融合 | fp16↔fp32 Convert 与 Precompute/Finalize 融合 |

### 5.9 性能对比：论文 vs 我们

| | T-MAN 论文 | ch07 实测 |
|---|---|---|
| Decode 速度 | **49.1 tok/s** | **39.3 tok/s**（慢 20%）|
| 设备 | Snapdragon 8 Gen 3 | 同款 |
| 量化 | INT2 weight + INT16 activation | 16a4w（可能更保守）|

慢 20% 的可能原因：量化方式差异（论文用原生 2-bit，我们用 16a4w PTQ）、executorch fork 版本差异、编译选项未完全对齐。

### 5.10 手写实现的关键 takeaway

给 ch10-direct-bitnet 的建议：

1. **必须复刻权重布局**：`hvx_preprocess_weights` 的 reshape/transpose 决定了 `vlut16` 能否正确工作
2. **LUT 交织是性能关键**：8 组 LUT 交织到一个 HVX 向量，一条 `vlut16` 并行查 128 字节
3. **累加用 int32**：`vaddacc` 避免 int16 溢出，最后才转 float
4. **qf32 而非 f32**：v75 没有原生 f32 HVX 运算，必须用 `Q6_Vqf32_vmpy_VsfVsf`
5. **预取覆盖延迟**：`l2fetch` + `dcfetch` 让下一轮的 DMA 和当前计算重叠
6. **分块适配 VTCM**：每个线程的工作集（权重 + LUT + 累加器）必须 < 1.3 MB

## 6. 文件说明

```
ch07-qnn-bitnet/
├── README.md              ← 本文件
├── build.sh               ← 编译 TMAN Op Package（hexagon-v75 + aarch64）
├── run_export.sh          ← AOT 导出 .pte（x86 主机上运行）
├── run_device.sh          ← 推送文件到设备 + 运行推理
├── config/
│   └── TMANOpPackageHtp.xml   ← QNN 自定义算子包定义
├── src/dsp/
│   ├── TMANOpPackageInterface.cpp  ← 算子包注册入口
│   ├── fp_extend.cpp / fp_trunc.cpp ← fp16↔fp32 转换
│   ├── include/hvx_funcs.h         ← HVX VLUT16 查表核心函数
│   └── ops/
│       ├── TMANPrecompute.cpp      ← 权重预处理（构建查找表）
│       ├── TMANLinear.cpp          ← 查表 GEMV 主逻辑
│       └── TMANFinalize.cpp        ← 输出后处理（累加、缩放）
├── artifacts/              ← 导出产物（.pte + quant_attrs）
└── build/                  ← 编译产物
    ├── hexagon-v75/libTMANOpPackage_htp.so
    └── aarch64/libTMANOpPackage_cpu.so
```

## 7. 与 ch10-direct-bitnet 的对比

ch10 直接在 HVX 上手写 VLUT16 内核（不经过 QNN/ExecuTorch），本章走 QNN 官方路径。

| | ch07（本章）| ch10 |
|---|---|---|
| 框架 | ExecuTorch + QNN | 纯 FastRPC + HVX |
| 编译方式 | Python AOT 导出 .pte | C 手写算子 |
| 权重 packing | Python 端自动完成 | 手动 bit-plane 拆分 |
| 算子注册 | QNN Op Package 机制 | 直接调 HVX intrinsics |
| 灵活性 | 低（必须走 executorch 流程）| 高（完全控制）|
| 工程复杂度 | 高（搭环境难）| 中（理解 HVX 即可）|
| 性能 | 39.3 tok/s | 待测 |

## 8. 依赖

- **executorch**（kaleid-liner fork with TMAN）：`tools/executorch/`
- **QNN SDK 2.44**：`tools/qnn-sdk/` → `/home/taowen/qnn_sdk/qairt/2.44.0.260225`
- **模型**：`microsoft/bitnet-b1.58-2B-4T`（从 HuggingFace 下载，约 2GB）
- **Python 环境**：`tools/executorch/.venv/`（torch 2.7.0 + executorch）
