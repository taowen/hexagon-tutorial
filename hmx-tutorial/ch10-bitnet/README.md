# 第10章：BitNet — 三值权重 LLM 的 HVX VLUT16 推理

## 1. 概述

本章研究 BitNet（1.58-bit LLM）在 Hexagon DSP 上的高效推理。BitNet 的权重只有 {-1, 0, +1} 三个值，传统的 "反量化再乘加" 方案在 NPU 上反而比 CPU 慢 3.8 倍。T-MAN 方案（Table-lookup MAtrix multiplicatioN）的核心洞察是：**用查表替代乘法**——利用 HVX 的 VLUT16 指令，把矩阵乘法变成表查找。

| 方案 | 核心操作 | 瓶颈 |
|------|---------|------|
| 传统反量化 | dequant → fmul → fadd | 反量化开销大，NPU 无原生 2-bit 支持 |
| T-MAN 查表 | precompute LUT → VLUT16 gather → accumulate | 内存带宽（DMA 59 GB/s 可缓解） |

**参考论文**: [T-MAN: Enabling End-to-End Low-Bit LLM Inference on NPUs via Unified Table Lookup](https://arxiv.org/abs/2511.11248)
- 论文基于 **Snapdragon 8 Gen 3**（OnePlus 12）和 **Snapdragon 8 Elite**（OnePlus 13T）
- Hexagon v75 架构，与我们的实验设备一致

**参考实现**: `tools/executorch` (kaleid-liner/executorch fork)，核心代码：
- HVX 内核：`backends/qualcomm/runtime/op_packages/TMANOpPackage/include/hvx_funcs.h`
- 自定义算子：`backends/qualcomm/builders/custom_ops.py`
- 权重打包：`backends/qualcomm/utils/utils.py` → `BitLinear`, `unpack_weights()`
- 模型定义：`examples/qualcomm/oss_scripts/bitnet/model/static_bitnet.py`

**前置章节**：
- ch04 已覆盖 VTCM 分配、UDMA、double-buffering（本章直接复用，不重复）
- ch05 已覆盖 HMX 矩阵乘法、权重 AH/WH 布局打包
- ch08 已覆盖 HVX f32 GEMM 和 qf32 用法

## 2. 核心原理：为什么查表能替代乘法

### 2.1 三值权重的数学性质

对于一组 g=4 个连续的激活值 (x₀, x₁, x₂, x₃) 和对应的三值权重 (w₀, w₁, w₂, w₃)，点积为：

```
dot = w₀·x₀ + w₁·x₁ + w₂·x₂ + w₃·x₃
```

因为每个 wᵢ ∈ {-1, +1}（忽略 0 的情况，后面处理），4 个权重共有 2⁴ = 16 种组合。这 16 个可能的点积结果可以**预计算**成一张查找表（LUT），然后用权重的 bit pattern 作为索引直接查出结果。

### 2.2 镜像优化

LUT 有对称性：`LUT[i] = -LUT[15 - i]`。因此只需计算 8 个奇数索引，偶数索引通过取反得到，计算量减半。

### 2.3 三值编码处理 {0} 的情况

实际三值权重包含 0。编码方式：将 {-1, 0, +1} 映射为 2-bit 值 {0, 1, 2}（加 1 偏移）。用 bit-serial 分解：
- bit 0：决定是否包含该激活（类似 mask）
- bit 1：决定符号方向

最终结果通过 `hvx_bit_serial` 合并：`output = 0.5 × partial_bit0 + partial_bit1`，并加上偏置校正 `lb`（补偿编码偏移）。

## 3. HVX VLUT16 指令详解

```
Q6_Wh_vlut16_VbVhR_nomatch(indices, lut, segment)
```

| 参数 | 含义 |
|------|------|
| `indices` | 128 字节，每字节低 4 bit 是 LUT 索引（0-15） |
| `lut` | 16 条目 × int16 的查找表（需要特殊交错布局） |
| `segment` | 0-3，选择 128B 向量中 4 组 16 条目 LUT 中的一组 |
| 返回值 | HVX_VectorPair：lo = 偶数字节的查表结果，hi = 奇数字节的查表结果 |

**关键特性**：
- 一条指令处理 128 个索引的并行查表
- 每次查表等效于 128 次乘加操作
- 对于 int16 激活 + 16 条目 LUT = 1024 等效 MADDs

### 3.1 实验 1 验证结果

在 8 Gen 3 真机上验证了 VLUT16 的三个核心行为：

**LUT 必须经过 vshuff 交错排列**：直接把 16 个 int16 线性放进向量是不行的。需要经过 4 级 `Q6_W_vshuff_VVR`（步长 -4, -8, -16, -32）将条目交错排列到硬件期望的位置。128B 向量容纳 64 个 int16 = 4 组 × 16 条目，segment 参数选择用哪一组。

```c
// 构建 VLUT16 LUT 的正确方式：
// 1. 每个 LUT 条目 splat 成一个完整向量
HVX_Vector l_tmp[16];
for (int i = 0; i < 16; i++) {
    l_tmp[i] = Q6_Vh_vsplat_R(values[i]);
}

// 2. 四级 vshuff 蝶形交错（参考 executorch hvx_lut_ctor）
//    步骤 1: vshuff(l[i+1], l[i], -4)   // 32-bit 交错
//    步骤 2: vshuff 结果, -8             // 64-bit 交错
//    步骤 3: vshuff 结果, -16            // 128-bit 交错
//    步骤 4: vshuff 结果, -32            // 256-bit 交错

// 3. 取第一个输出向量作为 VLUT16 的 lut 参数
```

**结果分 even/odd 字节**：VLUT16 返回 VectorPair，lo 向量包含偶数字节位置（0, 2, 4, ...）的查表结果，hi 向量包含奇数字节位置（1, 3, 5, ...）的查表结果。验证方法：偶数字节放索引 5（期望 500），奇数字节放索引 10（期望 1000），lo 全是 500，hi 全是 1000。64/64 正确。

**查表即乘法**：给定 4 个激活值 (1, 2, 3, 4)，预计算 16 个二值权重组合的点积作为 LUT，用 VLUT16 查出结果——与直接计算完全一致。例如 index=0x0F（所有权重 +1），VLUT16 返回 2560（= 10.0 × 256 scale），即 1+2+3+4=10。

## 4. 完整数据流

```
┌─────────────────────────────────────────────────────────────┐
│  Activation x (fp16, shape: [1, K])                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  TMANPrecompute (hvx_lut_ctor)                              │
│                                                             │
│  1. 每 4 个激活分一组: (x₀, x₁, x₂, x₃)                    │
│  2. 计算 16 个可能的点积 → LUT[0..15] (int16)               │
│  3. 计算激活量化 scale (ls = absmax / 32767)                 │
│  4. 计算偏置校正 lb = -0.5 × Σ(所有激活)                    │
│  5. shuffle LUT 到 VLUT16 所需的交错布局                     │
│                                                             │
│  输出: lvec[] (LUT向量), ls (scale), lb (bias)               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  TMANLinear (hvx_tbl, GroupSize==0 特化)                     │
│                                                             │
│  对每个权重字节:                                              │
│    w_lo = w & 0x0F          // 低 4 bit 索引                 │
│    w_hi = w >> 4            // 高 4 bit 索引                 │
│    result = VLUT16(w_lo, LUT, seg)  // 128 路并行查表        │
│                                                             │
│  累加: int16 → widen 到 int32                                │
│                                                             │
│  块边界处:                                                    │
│    int32 → float → ×ls → +lb → ×weight_scale → qf32 累加    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  TMANFinalize (hvx_bit_serial)                              │
│                                                             │
│  合并 bit planes: 0.5 × partial_bit0 + partial_bit1         │
│  qf32 → fp16 输出                                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
              Output y (fp16, shape: [1, M])
```

## 5. 权重预处理与内存布局

### 5.1 权重打包（ARM 侧，离线完成）

```python
# 三值 {-1, 0, +1} → 2-bit {0, 1, 2}（加1偏移）
# 4 个值打包到 1 个 uint8
packed[i] = (w0+1) | ((w1+1)<<2) | ((w2+1)<<4) | ((w3+1)<<6)
```

### 5.2 Bit 分解与分组

```python
# 每个 uint8 → 2 bit planes
# 每 bit plane 中，每 4 个连续 K 维度的 bit 打包成 4-bit 索引
# 这个 4-bit 索引直接用于 VLUT16 查表
```

### 5.3 Tile 布局

权重被重排为 `(P_tiles, Q_tiles, tile_p//vec_p, tile_q//vec_q, vec_q, vec_p)` 的分块格式：
- **P 维度**（输出维度）：对应 VLUT16 的 128 路并行
- **Q 维度**（输入维度 / g）：对应 LUT 分组
- Even/odd 交错以匹配 VLUT16 的 even/odd byte 输出模式
- 两个相邻 Q 索引打包到一个字节的高低 4 bit

## 6. 实验计划

**目标模型**：`microsoft/bitnet-b1.58-2B-4T`
- hidden=2560, heads=20, kv_heads=5, intermediate=6912, layers=30
- 词表 128256（Llama-3 tokenizer）
- 权重 ~600MB（1.58-bit packed）

### 实验 1：VLUT16 指令探索 ✅

真机验证 3/3 测试全部通过。

- LUT 必须经过 4 级 vshuff 交错排列（直接线性放入会得到错误结果）
- 结果 VectorPair 的 lo/hi 分别对应偶数/奇数字节位置的查表结果
- VLUT16 可以正确实现 "查表即乘法"：预计算 16 种二值权重组合的点积，用权重 bit pattern 做索引

**产出**：`vlut16_shuffle()` 和 `vlut16_build_lut()` 两个可复用 helper 函数。

### 实验 2：BitNet GEMV 完整内核 ✅

**状态**：完成。M=256, K=256 真机验证三个版本全部通过（0 mismatches，max relative error < 0.01%）。

实现了完整的三阶段 BitNet GEMV，并逐步优化：

**阶段 1 — LUT 构建**：每 4 个激活分组，预计算 16 种 ±x 组合的点积，量化到 int16，vshuff 交错排列。

**阶段 2 — VLUT16 查表**：packed 权重的 4-bit nibble 作为索引，VLUT16 并行查 128 个输出维度的结果。

**阶段 3 — bit-serial 合并**：三值权重编码为 2 bit planes（enc = w + 2），各做一次查表累加，合并公式：`output = 0.5 × partial_bit0 + partial_bit1 - 0.5 × sum(x)`。

**关键推导**：ternary w ∈ {-1,0,+1} 编码为 enc = w+2 ∈ {1,2,3}，拆成 bit0/bit1 后 w = bit0 + 2×bit1 - 2。VLUT16 的 LUT 计算 ±x 的所有组合，偏置项 lb = -0.5×sum(x) 补偿编码偏移。

**三个版本的性能对比**（M=256, K=256）：

| 版本 | 耗时 | 加速比 | 关键优化 |
|------|------|--------|----------|
| `bitnet_gemv` | 293 us | 1.0x | 基线：per-Q scale + 标量 float 累加 |
| `bitnet_gemv_opt` | 64 us | **4.6x** | 全局 scale + int32 向量累加（`vaddacc`） |
| `bitnet_gemv_4q` | 93 us | 3.2x | 4Q 多段 VLUT16 + int32 向量累加 |

**优化 1 — 全局 scale + int32 向量累加**（4.6x 加速）：
- 用一个全局 `ls = 4×max|x| / 32767` 替代 per-Q 的 max_abs 搜索
- 用 `Q6_Ww_vaddacc_WwVhVh` 将 VLUT16 的 int16 结果直接累加到 int32 向量，跨所有 Q 位置
- 只在最后转一次 float，消除了 64×64 次标量 float 乘加

**优化 2 — 4Q 多段 VLUT16**（概念验证）：
- 将 4 个 Q 位置的 nibble 打包到 2 字节（每字节高低 4 bit 各一个 Q），权重体积减半
- 用 `vlut16_build_lut_4q` 将 4 个 LUT 交错到一个向量，VLUT16 的 segment 0-3 各读一个
- 4 次 VLUT16 + `vaddacc` 处理 4 个 Q 位置

**4Q 的发现与教训**：
- **VLUT16 segment 映射在 v75 上不对称**：seg0→pos0, seg1→pos2, seg2→pos1, seg3→pos3（seg1 和 seg2 交换了）。必须运行时验证，不能凭文档假设。
- **int16 溢出**：不能先把 4 个 segment 的 int16 结果相加再 widen 到 int32（4×32767 > 32767），必须每对结果都用 `vaddacc` 直接 widen。
- **字节级移位**：提取高 nibble 必须用 `Q6_Vub_vlsr_VubR(v, 4)`（字节级），不能用 `Q6_Vuh_vlsr_VuhR`（半字级），否则跨字节污染。
- **当前 4Q 比 opt 慢**：标量 LUT 构建开销（4 个 LUT + 交错 shuffle）抵消了 VLUT16 节省。下一步需要 HVX 向量化 LUT 构建（如 executorch 的 `hvx_lut_ctor` 用 `vdeal` 转置 + qf32 并行计算）才能让 4Q 真正快于 opt。

**产出**：`bitnet_gemv.h` 包含 6 个函数：
- `vlut16_build_lut()` / `vlut16_build_lut_4q()` — 单 LUT / 4-LUT 向量构建
- `bitnet_pack_weights()` / `bitnet_pack_weights_4q()` — 原始 / 4Q 权重打包
- `bitnet_gemv()` / `bitnet_gemv_opt()` / `bitnet_gemv_4q()` — 三个 GEMV 内核
- `bitnet_gemv_reference()` — 标量参考实现

### 实验 3：Decoder Layer 完整组件 ✅

**状态**：完成。9 个 HVX f32 算子全部在真机验证通过。

BitNet decoder 与标准 Llama 的区别：
```
标准 Llama:              BitNet b1.58:
RMSNorm                  RMSNorm
Attention                Attention + attn_sub_norm（注意力输出额外归一化）
RMSNorm                  RMSNorm
MLP (SiLU gate)          MLP (ReLU² gate) + ffn_sub_norm
```

实现的 9 个 HVX 算子（全部在 `bitnet_ops.h` 中）：

| 算子 | 函数 | 验证结果 |
|------|------|----------|
| RMSNorm | `hvx_rmsnorm_f32` | max_err=0/100000（qf32 乘 + 标量 sqrt） |
| ReLU² | `hvx_relu2_f32` | 精确匹配（整数符号位比较 + qf32 平方） |
| 逐元素乘法 | `hvx_mul_f32` | PASS |
| 逐元素加法 | `hvx_add_f32` | PASS |
| RoPE | `hvx_rope_f32` | max_err=1/100000（two-half split，cos/sin 查表） |
| 点积 | `hvx_dot_f32` | PASS（qf32 乘加 + 标量归约） |
| Softmax | `hvx_softmax_f32` | [0.0321, 0.0871, 0.2369, 0.6439] 全部匹配（标量 expf + HVX 归一化） |
| 单头注意力 | `hvx_attention_decode_f32` | 4/4 输出精确匹配（dot→softmax→weighted V sum） |
| 多头 GQA 注意力 | `hvx_mha_decode_f32` | 8/8 输出精确匹配（GQA ratio=4, head→kv_head 映射） |

**BitNet 特有算子**：
- `attn_sub_norm`：注意力头输出拼接后、O projection 之前的 RMSNorm（复用 `hvx_rmsnorm_f32`）
- `ffn_sub_norm`：gate×up 乘积之后、down projection 之前的 RMSNorm（复用 `hvx_rmsnorm_f32`）
- `ReLU²`：`max(0, x)²`，替代 Llama 的 SiLU（更适合低精度训练）

**产出**：`bitnet_ops.h` 包含全部 9 个函数，可直接用于组装 decoder layer。

### 实验 4：单层 Decoder 端到端 ✅

**状态**：完成。真实 BitNet 权重加载 + 完整 decoder layer，DSP vs PyTorch 相对误差 **0.14%**。

完整数据流（decode 一个 token）：
```
input [1, 2560]
  → RMSNorm
  → Q/K/V projection (3 个 BitNet GEMV)
  → RoPE(Q), RoPE(K)
  → KV Cache append
  → Attention: Q×K^T → softmax → ×V
  → attn_sub_norm → O projection (BitNet GEMV)
  → residual add
  → RMSNorm
  → gate/up projection (2 个 BitNet GEMV)
  → ReLU² + element-wise mul
  → ffn_sub_norm → down projection (BitNet GEMV)
  → residual add
output [1, 2560]
```

一层包含 **7 个 BitNet GEMV** + 若干 HVX 向量算子。

**权重准备**：`prepare_weights.py` 从 HuggingFace 下载 `microsoft/bitnet-b1.58-2B-4T`，解包三值权重（{-1,0,1}），重新打包为 DSP 格式，生成 33.2 MB 的 `decoder_layer.bin`（包含 WeightLayout 头 + 所有权重 + 测试输入/参考输出 + RoPE 表 + KV Cache）。

**HF 权重格式**：packed `[M/4, K]` uint8，每字节 4 个 2-bit 值，块排列（不是交错排列）。三值分布约 25-31% 各 ±1，38-50% 为 0。

**验证结果**（pos=0, seq_len=1 decode）：
```
Output[0..3]:    -407.316  -121.177  -20.016  -43.398
Reference[0..3]: -407.287  -121.323  -19.897  -43.763
Max abs error:   0.73
Relative error:  0.14%
Mismatches:      0 / 2560
```

**性能**：单层 decoder = **98.6 ms**（未优化，7 个 GEMV 占主导）。
- 7 个 GEMV（尺寸 2560×2560 到 6912×2560），每个约 10-14 ms
- 30 层 × 98.6 ms ≈ 3 秒/token（~0.3 tok/s），需要优化

**产出**：
- `bitnet_decoder.h`：完整 decoder layer 函数
- `prepare_weights.py`：权重下载、打包、参考计算
- ARM 共享内存传输：`rpcmem_alloc` + `fastrpc_mmap` + dspqueue buffer ref

### 实验 5：完整推理 — 文本输入到文本输出

**目标**：输入一句话，BitNet-2B 在 DSP 上自回归生成文本。

```
$ echo "The future of AI is" | ./bitnet_infer --model bitnet-2b-4t.bin
The future of AI is ...
```

需要补全的模块：

| 模块 | 位置 | 说明 |
|------|------|------|
| Tokenizer | ARM | Llama-3 tokenizer（sentencepiece 或 tiktoken），文本 → token IDs |
| Embedding | ARM/DSP | token ID → fp16 向量 [2560]，查表（128256 × 2560 × 2B ≈ 625MB） |
| 30 层 Decoder | DSP | 实验 4 的 layer × 30，逐层串行 |
| LM Head | DSP | hidden [2560] → logits [128256]，全精度线性层（非 BitNet） |
| Sampling | ARM | greedy argmax 或 top-k/top-p |
| Detokenizer | ARM | token ID → 文本 |

**内存规划**：
- 权重（1.58-bit packed）：~600MB DDR
- Embedding + LM Head（fp16）：~1.2GB DDR
- KV Cache（fp16，per layer）：30 × 2 × ctx_len × 5 × 128 × 2B
- VTCM（8MB）：当前层的 LUT + 权重 tile + 中间激活

**生成循环**：
```
tokens = tokenize(prompt)
for each token in prefill:
    run_all_layers(token)  // 填充 KV cache
for i in range(max_new_tokens):
    logits = run_all_layers(last_token)
    next_token = sample(logits)
    print(detokenize(next_token))
    if next_token == eos: break
```

## 7. 当前目录结构

```
ch10-bitnet/
├── README.md
├── build.sh
├── run_device.sh
├── prepare_weights.py             # 权重下载、打包、参考输出生成
├── weights/                       # (gitignored) 二进制权重文件
│   └── decoder_layer.bin          # 33.2 MB: WeightLayout头 + 所有权重 + 测试数据
└── src/
    ├── bitnet_test.idl            # FastRPC 接口（dspqueue bootstrap）
    ├── common/
    │   └── protocol.h             # 消息协议 + WeightLayout 共享内存头
    ├── arm/
    │   └── main.c                 # ARM 驱动（rpcmem 共享内存 + dspqueue）
    └── dsp/
        ├── skel.c                 # DSP 端（测试调度 + 全部实验测试）
        ├── bitnet_gemv.h          # 实验 2：BitNet GEMV 内核
        ├── bitnet_ops.h           # 实验 3：9 个 HVX f32 算子
        └── bitnet_decoder.h       # 实验 4：完整 decoder layer
```

## 8. 关键问题与风险

**已验证**：
- ✅ `Q6_Wh_vlut16_VbVhR_nomatch` 在 v75 / 8 Gen 3 上可用
- ✅ LUT 交错布局：4 级 vshuff（-4, -8, -16, -32）
- ✅ 所有 4 个 segment（0-3）行为一致（对 splatted LUT 而言）
- ✅ even/odd byte 分流：lo = 偶数字节结果, hi = 奇数字节结果

**待验证**：
- Embedding + LM Head 的内存（~1.8GB）是否放得下——可能需要 mmap 或分块加载
- int16 LUT 累加到 int32 的溢出边界（K=2560 时累加 640 组）
- LM Head 是全精度线性层还是也做了量化——决定用 HVX qf16 matmul 还是 VLUT16
- 30 层 × 7 GEMV 的总延迟能否做到可用的 token/s（目标 >5 tok/s）
- KV Cache 管理：prefill 批量写入 vs decode 逐 token 追加
