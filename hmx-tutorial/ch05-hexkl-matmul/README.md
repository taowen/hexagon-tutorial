# Chapter 8: HexKL — HMX Matrix Multiply from Scratch

本章使用 Qualcomm 的 **HexKL (Hexagon Kernel Library)** 直接调用 HMX 硬件做矩阵乘法。
HexKL 提供三个层级的 API：

| 层级 | API | 运行位置 | 控制粒度 | 本章实验 |
|------|-----|----------|----------|----------|
| **高层** | SDKL (`sdkl_npu_mm_*`) | ARM 侧 | 一个函数搞定 | Experiment 2a |
| **中层** | SDKL + 手动 layout | ARM 侧 | 控制数据布局 | Experiment 2b |
| **底层** | Micro (`hexkl_micro_*`) | DSP 侧 | 控制每个 tile | Experiment 3 |
| **组合** | HVX intrinsics + Micro | DSP 侧 | HVX + HMX 混合 | **Experiment 1** |

> HexKL 还有一个 Macro API (`hexkl_macro_mm_f16`)，但它在真机上不工作（返回 success 但输出全零），
> 仅限模拟器使用。hexagon-mlir 项目也只使用 SDKL 和 Micro 两个层级。

## 环境搭建

```bash
# 1. 安装 HexKL addon（如果还没装）
bash install_hexkl.sh

# 2. 编译
bash build.sh

# 3. 推送到设备并运行
bash run_device.sh
```

`install_hexkl.sh` 会从 Qualcomm 下载 `Hexagon_KL.Core.1.0.0.Linux-Any.zip`，
解压到 `tools/hexkl-addon/`。如果 `/home/taowen/hexagon-mlir/downloads` 有缓存则直接使用。

## 文件结构

```
ch08-hexkl-matmul/
├── install_hexkl.sh      # 安装 HexKL addon
├── build.sh              # 编译 DSP 和 ARM 目标
├── run_device.sh         # 推送到手机并运行
├── README.md
└── src/
    ├── demo_hvx_hmx.c    # DSP 侧：HVX dequant + HMX matmul + HVX bias（重点）
    ├── demo_sdkl.c       # ARM 侧：SDKL 自动 matmul + 手动 layout matmul
    └── demo_micro.c      # DSP 侧：Micro API 手动 tile 操作
```

## Part 1: HVX + HMX 组合流水线（核心实验）

**代码**: `src/demo_hvx_hmx.c`

这是本章最重要的实验。LLM 推理需要在**一次 FastRPC 调用**中同时使用 HVX 和 HMX：

```
┌─────────────────────────── 一次 FastRPC 调用 ───────────────────────────┐
│                                                                         │
│  Step 1 (HVX)          Step 2 (HMX)           Step 3 (HVX)             │
│  ┌──────────────┐      ┌──────────────┐       ┌──────────────┐         │
│  │ Dequant      │      │ Tiled        │       │ Bias Add     │         │
│  │ int8 → f16   │ ──→  │ MatMul       │ ──→   │ out += bias  │         │
│  │ (Q8_0 格式)  │      │ (hexkl_micro)│       │ (f32 vadd)   │         │
│  └──────────────┘      └──────────────┘       └──────────────┘         │
│                                                                         │
│  HVX 向量指令           HMX 矩阵指令           HVX 向量指令             │
└─────────────────────────────────────────────────────────────────────────┘
```

**为什么这很重要？** 如果 dequant 和 matmul 是两次 FastRPC 调用，
每次 FastRPC 的开销约 50-100μs。一个 transformer 层有 4 次 matmul，
32 层模型就是 128 次 FastRPC = 6-12ms 纯开销。把所有操作合并到一次调用里，
这个开销降到几乎为零。

### Q8_0 量化格式

模拟 llama.cpp 的 `block_q8_0`：每 32 个 int8 值共享一个 f16 scale。

```c
typedef struct {
    _Float16 scale;              // 这组 32 个值的缩放因子
    int8_t   quants[GROUP_SIZE]; // 量化值：real = quants[i] * scale
} block_q8;
```

### Step 1: HVX 反量化

用 HVX 向量指令把 int8 权重转为 f16，每次处理 64 个值（2 个 Q8_0 block）：

```c
// 1. 加载 64 个 int8 到 HVX 向量
HVX_Vector v_i8 = *(HVX_Vector *)tmp;

// 2. int8 → int16 符号扩展
HVX_VectorPair wp = Q6_Wh_vunpack_Vb(v_i8);
HVX_Vector v_i16 = Q6_V_lo_W(wp);

// 3. int16 → f16 转换
HVX_Vector v_f16 = Q6_Vhf_equals_Vh(v_i16);

// 4. 广播 scale 到所有 lane
HVX_Vector v_scale = Q6_V_vmux_QVV(pred, v_s0, v_s1);

// 5. f16 乘法（通过 quasi-float16 中间格式）
HVX_Vector v_prod = Q6_Vhf_equals_Vqf16(
    Q6_Vqf16_vmpy_VhfVhf(v_f16, v_scale));
```

关键 HVX 指令：
| 指令 | 功能 |
|------|------|
| `Q6_Wh_vunpack_Vb` | int8 → int16 符号扩展（128 → 2×64 值） |
| `Q6_Vhf_equals_Vh` | int16 → float16 类型转换 |
| `Q6_Vqf16_vmpy_VhfVhf` | f16 向量乘法（结果为 quasi-f16） |
| `Q6_Vhf_equals_Vqf16` | quasi-f16 → f16 转换 |
| `Q6_V_vmux_QVV` | 按 predicate 选择两个向量的 lane |

### Step 2: HMX 矩阵乘法

使用 hexkl_micro API 做 tiled matmul（与 demo_micro.c 相同模式）。
HVX dequant 输出的 f16 权重直接喂给 HMX，无需额外拷贝。

### Step 3: HVX Bias 加法

HMX 输出是 f32，用 HVX 向量加法：

```c
// v75 没有直接的 Vsf_vadd，需要 quasi-float32 路径
HVX_Vector v_qf32 = Q6_Vqf32_vadd_VsfVsf(v_out, v_bias);
HVX_Vector v_sum  = Q6_Vsf_equals_Vqf32(v_qf32);
```

> **v75 的 quasi-float**: Hexagon v75 的 HVX 浮点运算使用"准浮点"（quasi-float）
> 中间格式。加法和乘法的结果是 qf16/qf32，需要额外一步转回标准 f16/f32。
> 这是硬件设计的权衡——用一个额外指令换取更简单的 FPU 流水线。

### 真机数据（32 × 128 × 64）

```
Step 1: HVX dequant int8 → f16 (256 blocks)
  Dequant check: max_diff=0.000078, errors=0/8192  ✓
Step 2: HMX matmul [32 x 64] * [64 x 128]          ✓
Step 3: HVX bias add (32 rows x 128 cols)            ✓
Max diff vs reference: 0.025215
Verification PASSED (4096 elements)
```

### 与 llama.cpp 的对应

| 本章 demo | llama.cpp Hexagon 后端 |
|-----------|----------------------|
| `block_q8` + HVX dequant | `block_q4_0`/`block_q8_0` + `ggml_vec_dot_*` |
| `hexkl_micro_hmx_mm_f16` | `ggml_compute_forward_mul_mat` (HMX 路径) |
| HVX bias add | RMSNorm / RoPE / Softmax (全部用 HVX) |
| 一次 `run_main_on_hexagon` | 一次 `dspqueue` 回调 |

## Part 2: SDKL matmul（ARM 侧）

**代码**: `src/demo_sdkl.c`

### Experiment 2a: 自动 matmul

最简单的 HMX 使用方式——从 ARM 调用一个函数：

```c
sdkl_npu_initialize(CDSP_DOMAIN_ID, NULL, NULL);
sdkl_npu_alloc(W_bytes, (void **)&W_f16_npu);
sdkl_cpu_rm_to_wh_f16_inplace(N_COL, N_INNER, W_f16_npu);
sdkl_npu_mm_f32f16_f32(CDSP_DOMAIN_ID,
    N_ROW, N_COL, N_INNER, A_npu, X_f32, W_f16_npu);
```

**内部流程**: SDKL 通过 FastRPC 把数据发到 DSP → DSP 内部做 AH layout 转换 →
分配 VTCM → 调 HMX → 结果写回 DDR → FastRPC 返回 ARM。

### Experiment 2b: 手动 Layout 控制（LLM 优化路径）

权重在模型加载时一次性转换为 WH 布局，之后每次推理只需转换激活值：

```c
// 模型加载时（一次性）
sdkl_cpu_rm_to_wh_f16_inplace(N_COL, N_INNER, W2_f16);

// 推理热路径
sdkl_cpu_rm_to_ah_f16_inplace(N_ROW, N_INNER, X2_f16);
sdkl_npu_mm_f16(CDSP_DOMAIN_ID, N_ROW, N_COL, N_INNER, A2_f16, X2_f16, W2_f16);
sdkl_cpu_ah_to_rm_f16_inplace(N_ROW, N_COL, A2_f16);
```

### 真机数据（256 × 1024 × 4096）

| 步骤 | 耗时 | 备注 |
|------|------|------|
| CPU 参考实现（单核） | 85.00 ms | |
| **NPU auto (f32f16_f32)** | **0.86 ms** | **98.7x** 加速 |
| W→WH 转换 | 14.38 ms | 一次性 |
| X→AH 转换 | 2.73 ms | 每次推理 |
| **NPU manual (mm_f16)** | **0.51 ms** | **166.0x** 加速 |
| A AH→RM 转换 | 0.68 ms | |
| 推理热路径合计 | 3.92 ms | |

### 为什么 mm_f16 比 f32f16_f32 快？

- `f32f16_f32`: 每次调用都要内部做 X 的 AH 转换 + f32↔f16 类型转换
- `mm_f16`: 数据已经是 f16 AH/WH 布局，直接喂 HMX，零开销

## Part 3: Micro API（DSP 侧，手动 Tile 操作）

**代码**: `src/demo_micro.c`

最底层的 HMX 编程。代码直接运行在 DSP 上（通过 `run_main_on_hexagon`），
手动管理 VTCM 布局、tile 拷贝、布局转换和 HMX 指令。

### VTCM 内存布局

```
  offset 0          K-1     K      K+1                  end
  +-------+-----+-------+-------+-------+-----+----------+
  | act 0 | ... | act   | wt    | scratch     |hmx_config|
  |  (AH) |     | (K-1) | (WH)  | (readback)  |          |
  +-------+-----+-------+-------+-------+-----+----------+
```

每个 slot = `HEXKL_HMX_ACTIVATION_ALIGNMENT` = 2048 字节 = 一个 32×32 f16 tile。

### 三层嵌套 Tiling 循环

```
for row in 0..N_ROW step 32:           // 外层：输出行
    load act tiles[0..K-1] to VTCM     // DDR→VTCM, RM→AH
    for col in 0..N_COL step 32:       // 中层：输出列
        hmx_acc_clear()                 // 清零累加器
        for k in 0..K-1:               // 内层：K 维度
            rm_to_wh(wt_tile)           // 权重 DDR→VTCM→WH
            hmx_mm(act[k], wt_tile)     // HMX 32×32 matmul
        hmx_acc_read()                  // 读出累加器
        ah_to_rm()                      // 恢复 row-major
        copy_f16_to_f32()               // VTCM→DDR, f16→f32
```

### 真机数据（32 × 128 × 64）

- VTCM: 8192 KB at 0xFF000000
- HexKL version: 1.0.0-beta1 (Hexagon V75)
- Verification PASSED — all 4096 elements match

## Part 4: HMX 数据布局详解

HMX 不接受 row-major 数据。它需要两种特殊布局：

### AH (Activation-Hexagon) 布局

用于激活值（X）和输出（A）。数据按 32×32 tile 重排，
tile 内部的元素顺序针对 HMX 的 MAC 流水线优化。

### WH (Weight-Hexagon) 布局

用于权重（W）。与 AH 不同的重排方式，因为 HMX 对
activation 和 weight 的访问模式不同。

### 为什么需要这些布局？

HMX 的 32×32 MAC 阵列有固定的数据读取模式。
如果数据不在正确的物理位置，HMX 会读到错误的值。
布局转换本质上是把 row-major 数据"预排"到 HMX 期望的位置。

## Part 5: 四个实验的对比

| | HVX+HMX 组合 | SDKL (auto) | SDKL (manual) | Micro |
|---|---|---|---|---|
| **运行位置** | DSP | ARM | ARM | DSP |
| **HVX 使用** | dequant + bias | 无 | 无 | 无 |
| **HMX 使用** | hexkl_micro | SDKL 黑盒 | SDKL 黑盒 | hexkl_micro |
| **代码量** | ~150 行 | ~10 行 | ~30 行 | ~100 行 |
| **Layout 转换** | 手动（VTCM 内） | 自动 | 手动（ARM CPU） | 手动（VTCM 内） |
| **适用场景** | LLM 推理内核 | 快速原型 | LLM 权重预处理 | 自定义算子 |
| **典型用户** | llama.cpp | 应用开发者 | 框架开发者 | 编译器 Runtime |

### 为什么 LLM 需要 HVX+HMX 组合？

一个 transformer 层的计算流程：

```
Token 输入
  ↓
RMSNorm (HVX)          ← 向量运算
  ↓
Q/K/V 投影 (HMX)       ← 矩阵乘法，权重需要 HVX dequant
  ↓
RoPE (HVX)             ← 位置编码，向量运算
  ↓
Attention (HMX+HVX)    ← matmul + softmax
  ↓
Output 投影 (HMX)      ← 矩阵乘法
  ↓
RMSNorm (HVX)
  ↓
FFN up/gate (HMX)      ← 矩阵乘法 × 2
  ↓
SiLU (HVX)             ← 激活函数
  ↓
FFN down (HMX)         ← 矩阵乘法
  ↓
输出
```

HVX 和 HMX 交替出现。如果每次切换都需要一次 FastRPC 往返，性能会很差。
把整个 transformer 层（或至少一个大的子图）放在一次 DSP 调用里，
HVX 和 HMX 之间的切换就是零成本的函数调用。

## Part 6: 与 ch06 VTCM 的联系

ch06 讲了 VTCM 的分配和管理。本章展示了 VTCM 的实际用途：

| ch06 概念 | ch08 对应 |
|-----------|-----------|
| `HAP_compute_res_acquire` VTCM | `hexkl_micro_hw_init` 内部调用 |
| Bump allocator | Micro API 手动分配 tile slot |
| DMA DDR→VTCM | `hexkl_micro_hmx_copy_submatrix_to_f16` |
| VTCM 延迟优势 | HMX **必须**从 VTCM 读 tile |

## 章节总结

- HMX 是 Hexagon 的矩阵乘加速器，32×32 tile 粒度
- 数据必须先转换为 AH/WH 布局才能喂给 HMX
- **HVX 和 HMX 可以在一次 FastRPC 调用中混合使用**——这是 LLM 推理的关键
- HVX 负责 dequant、norm、activation；HMX 负责 matmul
- SDKL 提供一键 matmul（99x 加速），适合快速上手
- 手动 layout 控制可以进一步优化（166x），适合 LLM 推理
- Micro API + HVX intrinsics 提供最底层控制，适合编译器 Runtime
- 权重布局转换是一次性成本，在 LLM 场景下完全 amortized
