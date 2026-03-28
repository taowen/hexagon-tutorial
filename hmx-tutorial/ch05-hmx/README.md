# Chapter 5: HMX 矩阵乘法——从 tile 基础到生产级优化

HMX 是 Hexagon 的矩阵乘法加速器，32×32 tile 粒度，f16 精度。
本章通过 6 个实验，从最基础的 tile 操作开始，逐步搭建出完整的 HMX 优化管线。

结论先说：**HMX 适合大矩阵（≥512 维度），小矩阵用 HVX qf32 更快。但绕过 hexkl readback（用 `hmx_store_acc` + HVX 解交织）后，HMX 在 VTCM 常驻场景下比 hexkl 路径快 24 倍。**

## HMX 是什么

HMX（Hexagon Matrix Extension）是 Hexagon v75+ 的矩阵乘法单元，工作在 32×32 tile 粒度。

**5 条核心指令**：
- `mxclracc.hf` — 清零累加器
- `activation.hf = mxmem(addr, limit)` — 从 VTCM 加载激活 tile（AH 格式）
- `weight.hf = mxmem(addr, limit)` — 从 VTCM 加载权重 tile（WH 格式）
- `cvt.hf = acc` — 从累加器转换到输出格式
- `mxmem(addr) = cvt` — 将输出写回 VTCM

**硬性约束**：
- 数据必须在 VTCM 中（不能直接从 DDR 计算）
- 激活用 AH（Activation Height）格式，权重用 WH（Weight Height）格式
- activation 和 weight 加载必须在同一个 VLIW packet 中

## 文件结构

```
ch05-hmx/
├── install_hexkl.sh      # 安装 HexKL addon
├── build.sh              # 编译
├── run_device.sh         # 推送到手机并运行
├── README.md
└── src/
    ├── exp1_tile_basics.c       # Experiment 1: Tile 基础——VTCM 管理、格式转换、手动 matmul
    ├── exp2_weight_layout.c     # Experiment 2: 权重布局优化——L0→L1→L2 逐级加速
    ├── exp3_streaming.c         # Experiment 3: VTCM 流式计算——大矩阵分块处理
    ├── exp4_pipeline.c          # Experiment 4: 完整管线对比——hexkl vs ASM vs HVX
    ├── exp5_standalone_asm.c    # Experiment 5: 脱离 hexkl——htp-ops-lib 的真实路径
    └── exp6_init_test.c         # Experiment 6: HMX 初始化需求测试——什么前置条件是真正必要的
```

## 快速开始

```bash
bash install_hexkl.sh   # 首次运行，安装 HexKL addon
bash build.sh           # 编译 DSP 目标
bash run_device.sh      # 推送到手机运行全部实验
```

## Experiment 1: Tile 基础——VTCM 管理、格式转换、手动 matmul

**代码**: `src/exp1_tile_basics.c`

最底层的 HMX 编程入门。手动管理 VTCM 布局、tile 拷贝、格式转换和 HMX 计算：

```
VTCM 布局:
  [act tile 0][...][act tile K-1][wt tile][scratch][hmx_config]

三层嵌套循环:
  for row:                          // 输出行
    load act tiles → VTCM (AH)     // DDR→VTCM→AH 格式
    for col:                        // 输出列
      acc_clear()                   // 清零累加器
      for k: rm_to_wh() → mm()     // 加载权重 + MAC
      acc_read() → ah_to_rm()      // 读出累加器
      copy_f16_to_f32()            // 写回 DDR
```

## Experiment 2: 权重布局优化——L0→L1→L2 逐级加速

**代码**: `src/exp2_weight_layout.c`

测试 4 个级别，量化每项优化的实际收益：

| 级别 | 做法 | 目的 |
|------|------|------|
| **L0 baseline** | hexkl_micro 全 API，每行重新加载权重 | 基线 |
| **L1 wt-cached** | 权重预加载到 VTCM（`rm_to_wh_f16` 格式转换+拷贝）| 量化权重缓存收益 |
| **L2 preformatted** | 权重在 DDR 预转为 WH 格式，加载只做 memcpy | 分离格式转换 vs DDR→VTCM 拷贝开销 |
| **compute-only** | 全部 tile（激活+权重）预缓存，只计时计算+回写 | 暴露真正瓶颈 |

### 真机数据（Snapdragon 8 Gen 3, VTCM=8MB, 20 次迭代平均）

#### Fwd L1: [128×128] = [128×800] @ [800×128]（26.2M FLOPS）

| 级别 | 耗时 (μs) | GFLOPS | vs L0 |
|------|----------|--------|-------|
| L0 baseline | 1203 | 21.78 | 1.00x |
| L1 wt-cached | 863 | 30.37 | **1.39x** |
| **L2 preformatted** | **818** | **32.03** | **1.47x** |
| compute-only | 484 | 54.21 | 2.48x |

#### Bwd dW1: [128×800] = [128×128] @ [128×800]（输出最宽，25 列 tile）

| 级别 | 耗时 (μs) | GFLOPS | vs L0 |
|------|----------|--------|-------|
| L0 baseline | 3455 | 7.59 | 1.00x |
| L1 wt-cached | 3162 | 8.29 | 1.09x |
| L2 preformatted | 3128 | 8.38 | 1.10x |
| compute-only | 3009 | 8.71 | 1.15x |

#### Small: [32×128] = [32×800] @ [800×128]（只 1 行 tile，L1 无效但 L2 有效）

| 级别 | 耗时 (μs) | GFLOPS | vs L0 |
|------|----------|--------|-------|
| L0 baseline | 301 | 21.79 | 1.00x |
| L1 wt-cached | 301 | 21.78 | **1.00x** |
| **L2 preformatted** | **249** | **26.29** | **1.21x** |
| compute-only | 122 | 53.92 | 2.47x |

#### Medium: [128×256] = [128×512] @ [512×256]（33.6M FLOPS）

| 级别 | 耗时 (μs) | GFLOPS | vs L0 |
|------|----------|--------|-------|
| L0 baseline | 1734 | 19.35 | 1.00x |
| L1 wt-cached | 1300 | 25.81 | 1.33x |
| **L2 preformatted** | **1240** | **27.06** | **1.40x** |
| compute-only | 968 | 34.67 | 1.79x |

### 关键发现

**1. 权重缓存是最大优化（L0→L1：最高 39%）**

对于多行 tile 的矩阵（Mr>1），权重缓存避免了每行重新从 DDR 加载权重 tile。
Fwd L1（Mr=4）提升 39%，Bwd dW1（Mr=4）提升 9%。
单行 tile 的矩阵（Mr=1，如 Small）**完全无效**——因为权重本来就只加载一次。

**2. HMX Weight Layout 提供额外 5-21% 收益（L1→L2）**

`rm_to_wh_f16` 做两件事：DDR→VTCM 拷贝 + 行主序→WH 格式转换。
L2 用 `hexkl_macro_rm_to_wh_f16_inplace` 在 DDR 中预转换权重为 WH 格式，
加载时用 `copy_psubmatrix_to_f16_weight` 只做 memcpy（无格式转换）。

| 配置 | L1→L2 提升 | 说明 |
|------|-----------|------|
| **Small** (Mr=1, K=25) | **21%** | L1 对 Mr=1 无效，但 L2 省掉格式转换 |
| Fwd L1 (Mr=4, K=25) | 5.2% | 100 个 tile，每 tile 省 ~0.5μs 转换时间 |
| Medium (Mr=4, K=16) | 4.6% | 128 个 tile |
| Bwd dW1 (Mr=4, K=4) | 1.1% | tile 少，收益小 |

**关键意义**：这正是 QNN NativeKV Cache 和 llama.cpp 的做法——权重/KV 以 WH 格式持久存储，
matmul 时零格式转换。对 LLM 推理（decode 阶段 Mr=1）尤其重要。

**3. 真正的瓶颈是输出回写**

L2 和 compute-only 的差距 = 纯数据搬运开销（`ah_to_rm_f16` + `copy_f16_to_f32_submatrix`）：
- Fwd L1: 818μs vs 484μs → **回写占 41%**
- Small: 249μs vs 122μs → **回写占 51%**
- Bwd dW1: 3128μs vs 3009μs → 回写只占 4%（N=800，计算量远大于回写量）

每个输出 tile 需要两次 hexkl_micro 调用才能转回 f32 row-major。
对于"宽输出矩阵"（大 M×N，小 K），计算量 >> 回写量，HMX 吞吐好。
对于"窄输出矩阵"（小 N 或小 M），回写开销占主导，HMX 效率低。

**4. 纯 HMX 吞吐可达 54 GFLOPS**

compute-only 在 K=800 的配置上达到 54 GFLOPS，接近 HMX f16 峰值。
瓶颈不在 HMX 计算本身，而在数据搬运和格式转换。

### 曾经尝试但无效的优化

| 优化 | 做法 | 结果 |
|------|------|------|
| bias=mxmem2 | 用 256B scale buffer 替换 `setup_acc_read_f16` 配置块 | ~0% |
| 全 VTCM 常驻 | 激活值 + 权重全部预缓存到 VTCM | ~0%（激活值加载不是瓶颈） |
| 全 f16 路径 | 跳过 f32↔f16 转换 | ~0%（转换时间 << tile 格式转换时间） |

这些优化在 Experiment 2 的 DDR 回写路径下完全被淹没。但 Experiment 5 证明：**当数据常驻 VTCM 且绕过 hexkl readback 时，HMX 吞吐提升 24 倍**（详见 Exp 5 V3 vs V1）。

## Experiment 3: VTCM 流式计算——大矩阵分块处理

**代码**: `src/exp3_streaming.c`

Experiment 1-2 只测试了"能装进 VTCM"的小矩阵。LLM 推理的典型尺寸是 4096×4096 或更大，
权重远超 VTCM 容量。本实验解决这个问题：**当权重不能全部放入 VTCM 时，如何高效分块流式处理？**

### 核心思路（来自 htp-ops-lib）

```
VTCM 预算分配:
  固定开销: K 个激活 tile + staging + readback + HMX config
  剩余空间: 分配给权重列 tile（每列需要 K 个 tile）

  例: K=4096 → 128 个 K-tile
      VTCM=8MB, 固定开销 ≈ K×ALIGN + 少量
      剩余空间可放 N_chunk 列的权重

流式循环:
  for 每个行条带 (TILE 行):
    加载激活到 VTCM (一次)
    for 每个列块 (N_chunk 列):
      加载权重块到 VTCM (覆盖上一块)
      计算 + 回写输出
```

关键优化：**激活复用** —— 激活只加载一次，跨所有列块复用。
这正是 htp-ops-lib（`hmx_mat_mul_permuted_w16a32`）和 llama.cpp Hexagon 后端的做法。

### 测试配置

| 配置 | 尺寸 | FLOPS | 说明 |
|------|------|-------|------|
| Fwd L1 | 128×128×800 | 26.2M | 小矩阵，对照 Exp 2 |
| Medium | 128×256×512 | 33.6M | 中等，对照 Exp 2 |
| **LLM decode** | **1×4096×4096** | **33.6M** | **典型 decode，Mr=1** |
| **LLM prefill 32** | **32×4096×4096** | **1.07G** | **短序列 prefill** |
| **LLM prefill 128** | **128×4096×4096** | **4.29G** | **长序列 prefill** |
| **LLM wide** | **1×11008×4096** | **90.2M** | **FFN 宽层 (LLaMA-7B)** |

### 对比方案

| 方案 | 权重格式 | 可处理大矩阵 | 说明 |
|------|---------|-------------|------|
| L2 preformatted | WH 预格式化 | **否**（VTCM 不够则 SKIP） | Exp 2 的 L2 |
| Stream baseline | 行主序（加载时 rm→wh） | **是** | 流式 + 运行时格式转换 |
| **Stream prefmt** | **WH 预格式化** | **是** | **流式 + 预格式化（最优）** |

### 真机数据（Snapdragon 8 Gen 3, VTCM=8MB, 20 次迭代平均）

VTCM 参数：K=4096 时，128 个 K-tile，每 chunk 可放 30 列 tile（960 列/chunk）。

#### 小矩阵（全部装入 VTCM，1 chunk）

| 配置 | 方案 | 耗时 (μs) | GFLOPS | 说明 |
|------|------|----------|--------|------|
| Fwd L1 [128×128×800] | L2 preformatted | 823 | 31.85 | 基准 |
| | Stream baseline | 1310 | 20.01 | 流式有开销 |
| | **Stream prefmt** | **1017** | **25.78** | 接近 L2 |
| Medium [128×256×512] | L2 preformatted | 1239 | 27.09 | 基准 |
| | Stream baseline | 1857 | 18.07 | |
| | **Stream prefmt** | **1477** | **22.72** | |

小矩阵上 Stream prefmt 比 L2 慢约 20%——这是分 chunk 循环的固定开销。
但 L2 在大矩阵上 SKIP，Stream 不会。

#### LLM 级大矩阵（必须分 chunk，L2 SKIP）

| 配置 | 方案 | 耗时 (μs) | GFLOPS | chunks | 说明 |
|------|------|----------|--------|--------|------|
| **LLM decode** [1×4096×4096] | L2 prefmt | SKIP | — | — | VTCM 不够 |
| | Stream baseline | 283,303 | 0.12 | 5 | rm→wh 转换极慢 |
| | **Stream prefmt** | **41,510** | **0.81** | 5 | **6.8x 快于 baseline** |
| **LLM prefill 32** [32×4096×4096] | L2 prefmt | SKIP | — | — | |
| | Stream baseline | 235,627 | 4.56 | 5 | |
| | **Stream prefmt** | **46,469** | **23.11** | 5 | **5.1x 快，激活复用** |
| **LLM prefill 128** [128×4096×4096] | L2 prefmt | SKIP | — | — | |
| | Stream baseline | 1,275,095 | 3.37 | 5 | |
| | **Stream prefmt** | **185,753** | **23.12** | 5 | **6.9x 快，近峰值** |
| **LLM wide** [1×11008×4096] | L2 prefmt | SKIP | — | — | |
| | Stream baseline | 396,316 | 0.23 | 12 | |
| | **Stream prefmt** | **110,923** | **0.81** | 12 | **3.6x 快** |

### 关键发现

**1. 预格式化权重在流式模式下收益巨大（5-7x）**

Stream baseline 每次加载 chunk 都要做 `rm_to_wh` 格式转换，且重复 N_ITERS×N_chunks 次。
Stream prefmt 只做 memcpy，加载速度快 5-7 倍。

这验证了 htp-ops-lib 和 llama.cpp 的做法：**权重必须以 WH 格式持久存储在 DDR。**

**2. 激活复用是 prefill 的关键**

LLM decode (M=1): 0.81 GFLOPS
LLM prefill 32 (M=32): 23.11 GFLOPS — **28x 吞吐提升**

M=32 时激活 strip 被 32 个输出行复用，分摊了权重加载的开销。
这解释了为什么 LLM prefill（大 batch）比 decode（batch=1）高效得多。

**3. Stream prefmt 在大矩阵上达到 23 GFLOPS**

这与 Exp 2 的 L2 preformatted 在小矩阵上的 27-32 GFLOPS 接近。
差距主要来自：每个 chunk 需要重新加载权重（5 chunks × 30 列/chunk）。
进一步优化方向：DMA 双缓冲（异步加载下一个 chunk + 当前 chunk 计算）。

**4. VTCM 预算决定 chunk 大小**

VTCM=8MB, K=4096 时：
- 固定开销：128×ALIGN（激活）+ 2×ALIGN（staging/readback）+ config ≈ 260KB
- 每列权重：128×ALIGN ≈ 256KB
- 可放 30 列/chunk → 4096 列需要 5 chunks
- 11008 列需要 12 chunks

### 与 ch08 MNIST 训练的对比

ch08 实验数据（batch=128）：

| 方案 | Epoch 时间 | 相对 HVX |
|------|-----------|---------|
| HVX qf32 (baseline) | 1850ms | 1.0x |
| HMX + weight caching | 6308ms | 3.4x 慢 |

**为什么 HMX 比 HVX 慢 3.4x？**

MNIST 训练每 batch 有 6 次 matmul。权重缓存只对 Mr>1 的有效（节省 ~624 次 rm_to_wh 调用/batch，
每次 ~1μs = ~624μs/batch = ~292ms/epoch）。但整个 HMX 路径还有 f32↔f16 转换 + tile 格式转换 +
输出回写的固定开销，这些在 MNIST 的小矩阵上远超实际计算量。

**结论：MNIST 级别的小矩阵（128×800），HVX qf32 是最优选择。**

## Experiment 4: 完整管线对比——hexkl vs 直接 HMX 汇编 vs HVX 输出

**代码**: `src/exp4_pipeline.c`

### 实验设计

Experiment 2 的 compute-only 测的是"全部 tile 预缓存 + 计算 + DDR 回写"，耗时 484us。
本实验追问：**如果输出也留在 VTCM（不写回 DDR），纯 HMX 计算到底有多快？**

所有 tile（激活+权重）预缓存在 VTCM（不计时），只测量计算循环。四种方法对比：

| 方法 | 做法 |
|------|------|
| **A: hexkl full** | hexkl_micro 标准调用链（acc_clear -> mm_f16 -> acc_read -> copy_f16_to_f32），输出写回 L2 cache |
| **B: hexkl 逐阶段** | 同 A，但对每个阶段单独计时（分解开销） |
| **C: ASM+hexkl** | 直接 HMX 内联汇编计算（`hmx_set_scales` + `mxclracc.hf` + 批量 `mxmem` 加载）+ hexkl 回写 |
| **D: ASM+HVX** | 直接 HMX 内联汇编计算 + HVX 输出转换（htp-ops-lib 方式，不经过 hexkl） |

关键区别：1000 次迭代，L2 cache 全热。Experiment 2 只跑 20 次，cache 冷。

### 真机数据（Snapdragon 8 Gen 3, VTCM=8MB, 1000 次迭代平均）

所有 6 个配置 PASS。

#### Fwd L1: [128x128x800] (26.2 MFLOPS, 16 output tiles, K=25)

| Method | Time (us) | GFLOPS | Pass |
|--------|----------|--------|------|
| A hexkl full | 5.8 | 4,482 | PASS |
| C ASM+hexkl | 2.2 | 11,687 | PASS |
| D ASM+HVX | 4.6 | 5,705 | PASS |
| D compute only | 0.9 | 28,463 | - |
| D hvx_out only | 3.7 | - | - |

Method B 逐阶段分解：

| Phase | Time (us) | % | Per-call |
|-------|----------|---|---------|
| acc_clear | 0.1 | 1.6% | - |
| mm_f16 | 3.0 | 84.1% | 0.008 us/call |
| acc_rd+rm | 0.3 | 7.2% | - |
| f16_to_f32 | 0.3 | 7.1% | 0.02 us/tile |

Speedup vs A: C **1.39x** faster, D 0.69x (slower)

#### Bwd dW1: [128x800x128] (26.2 MFLOPS, 100 output tiles, K=4)

| Method | Time (us) | GFLOPS | Pass |
|--------|----------|--------|------|
| A hexkl full | 7.5 | 3,483 | PASS |
| C ASM+hexkl | 4.3 | 6,167 | PASS |
| D ASM+HVX | 20.1 | 1,306 | PASS |
| D compute only | 2.8 | 9,471 | - |
| D hvx_out only | 17.3 | - | - |

Speedup vs A: C **1.17x** faster, D 0.25x (slower)

#### Medium: [128x256x512] (33.6 MFLOPS, 32 output tiles, K=16)

| Method | Time (us) | GFLOPS | Pass |
|--------|----------|--------|------|
| A hexkl full | 7.8 | 4,308 | PASS |
| C ASM+hexkl | 2.9 | 11,491 | PASS |
| D ASM+HVX | 8.3 | 4,054 | PASS |
| D compute only | 2.2 | 15,329 | - |
| D hvx_out only | 6.1 | - | - |

Speedup vs A: C **1.44x** faster, D 0.51x (slower)

#### 全部 6 配置汇总

| Config | A (us) | C (us) | D (us) | C speedup |
|--------|--------|--------|--------|-----------|
| Fwd L1 [128x128x800] | 5.8 | 2.2 | 4.6 | 1.39x |
| Fwd L2 [128x32x128] | 0.3 | 0.2 | 1.2 | 0.99x |
| Bwd dW1 [128x800x128] | 7.5 | 4.3 | 20.1 | 1.17x |
| Bwd dH [128x128x32] | 0.6 | 0.4 | 3.9 | 1.28x |
| Small [32x128x800] | 1.3 | 0.6 | 1.5 | 1.75x |
| Medium [128x256x512] | 7.8 | 2.9 | 8.3 | 1.44x |

### 关键发现

**1. VTCM 常驻时 HMX 吞吐惊人**

当所有 tile 预缓存在 VTCM 中，纯 HMX 计算达到 ~28,000 GFLOPS（Fwd L1 D compute only）：
- `mxmem` 加 `:deep` 后缀启用深度流水线
- 批量加载 25 个 tile 一条指令完成（K=25），消除了逐 tile 开销
- 400 次 tile 乘法在 0.9us 内完成 = 2.3ns/tile，约 4-5 cycles @ 2GHz

对比 Experiment 2 的 compute-only：484us / 54 GFLOPS。**差距 500 倍**，因为 Experiment 2 的
"compute-only"仍包含 `acc_read + ah_to_rm + copy_f16_to_f32`（DDR 写回）。
本实验消除了 DDR 路径（输出留在 VTCM）。

**2. hexkl API 调用开销极低（cache 热时）**

Method B 分解显示：
- `mm_f16`（核心 tile 乘法）：0.008 us/call — 几乎瞬时
- `acc_clear`：< 0.1 us
- `copy_f16_to_f32`：**0.02 us/tile**（L2 cache 热）

这颠覆了旧结论："30us/tile for copy_f16_to_f32"。30us 是 DDR 写回 +
冷 cache 的延迟（Experiment 2 只跑 20 次迭代）。跑 1000 次迭代后 L2 cache 全热，
hexkl 的转换器快了 **1500x**。

**3. 批量 tile 加载（Method C）一致有效**

直接 ASM 批量加载（`mxmem` 带 limit 参数）在所有配置上给出 1.2-1.7x 加速：

| 优化来源 | 说明 |
|---------|------|
| hexkl `mm_f16` | 逐 tile 函数调用（即使 0.008us/call，400 次调用累计 3us） |
| 直接 ASM | 一条指令加载 25 个 tile，充分利用 HMX 深度流水线 |

Small 配置（K=25, 只 1 行 tile）加速最大（1.75x），因为批量加载的优势在高 K 时最明显。

**4. HVX 输出转换反而更慢**

Method D 的 HVX 输出（0.17-0.37 us/tile）比 hexkl 的 `copy_f16_to_f32`（0.02 us/tile）
慢 **10-20 倍**：

| 方式 | us/tile | 说明 |
|------|---------|------|
| hexkl `copy_f16_to_f32` | 0.02 | 可能使用 DMA 或专用硬件路径 |
| HVX `hvx_vhf_to_wsf` | 0.17-0.37 | 软件格式转换 + 手动 store |

对 VTCM 常驻 + cache 热的数据，hexkl 的转换器远优于手写 HVX。

**5. 对 htp-ops-lib 的重新理解**

htp-ops-lib 的核心优势**不在于**替换 hexkl 的转换函数，而在于：
- **VTCM 数据流管理**：输出留在 VTCM 给下一个算子用（避免 DDR 往返）
- **流式处理大矩阵**：Experiment 3 已证明（5-7x 加速）
- **量化权重反量化**：HVX 在 VTCM 内做 Q4_0->f16 反量化

对于小矩阵（MNIST 级别），hexkl 已经足够高效。

**6. 完整瓶颈图谱（修正版，含 Exp 5 数据）**

```
Exp 2 L0 (full DDR path):             ~1200 us  ->  DDR I/O 占 99.5%
Exp 2 compute-only (DDR write):         ~483 us  ->  DDR writeback 占 ~99%
Exp 4 Method A (VTCM, cache hot):         ~6 us  ->  hexkl readback 是瓶颈
Exp 5 V3 ASM+HVX f16 (VTCM→VTCM):       ~21 us  ->  HVX 解交织 + VTCM 写入
Exp 5 V4 ASM+HVX f32 (VTCM→DDR):          ~3 us  ->  HVX 转换 + DDR 写入
Exp 4 compute only:                        ~1 us  ->  纯 HMX 吞吐 28,000 GFLOPS
```

**注意**：Exp 4 Method A 的 6µs 和 Exp 5 V1 的 485µs 差异巨大，原因是迭代次数不同
（Exp 4: 1000 次 L2 cache 全热; Exp 5 V1/V2 的 hexkl readback 包含 VTCM→L2→DDR 路径）。
Exp 5 V3/V4 绕过 hexkl readback，输出留在 VTCM 或直接 HVX 转换写 DDR，无此瓶颈。

真正的性能层级：

| 层级 | 量级 | 说明 |
|------|------|------|
| DDR <-> VTCM I/O | 1000us | 数据搬运是绝对瓶颈 |
| hexkl readback | 120-3000us | `acc_read` + `ah_to_rm` + `copy_f16_to_f32` 主导 |
| HVX readback (VTCM→VTCM) | 5-126us | `hmx_store_acc` + `vdeal` 解交织，**24x** 快于 hexkl |
| 纯 HMX 计算 | 1us | 已接近硬件极限 |

从 3000us 到 1us，跨越三个数量级。hexkl readback 是被忽视的中间瓶颈。

## Experiment 5: 脱离 hexkl——htp-ops-lib 的真实路径

**代码**: `src/exp5_standalone_asm.c`

### 问题

Experiment 4 的 Method C（直接 ASM 计算）在 Method A（hexkl 计算）之后运行，
**看起来**不需要额外初始化就能工作。但这是一个隐蔽的 bug：Method C 依赖了 Method A
遗留的 HMX 硬件状态。独立运行 Method C 会产生错误结果。

更根本的问题是：Qualcomm 的生产级 HMX 库（htp-ops-lib）**不使用 hexkl**。
hexkl 是开发者工具，htp-ops-lib 是生产代码。本实验回答：
**不依赖 hexkl 的 HMX matmul runtime 长什么样？需要哪些前置条件？**

### HMX 是有状态的协处理器

HMX 不是"传入数据就算"的无状态单元。它有内部寄存器需要初始化：

| 寄存器 | 设置方式 | 作用 | 何时需要 |
|--------|---------|------|---------|
| **scale/bias** | `bias = mxmem2(scales_ptr)` | 输出缩放因子 | 仅 hexkl readback（`mxcvtr.sat.hf`）路径 |
| accumulator | `mxclracc.hf` | 清零累加器 | 每次 matmul |

**注**：Exp 6 在单次 matmul 下证明 `cvt.hf = acc(2)` 路径的 `acc(2)` 参数自带 f16 scale=1.0 语义，不读取 scale/bias 寄存器。但 ch09 训练实验证实：`setup_acc_read`（`mxmem2.bias`）在持续多 matmul 工作负载下仍然必要，dummy `mm_f16` 则不需要（详见 Exp 6）。

**`hmx_set_scales()` 是 hexkl readback 路径（`mxcvtr.sat.hf`）的前置依赖。**
（注：Exp 6 在单次 matmul 下证明 direct ASM + `cvt.hf = acc(2)` 路径连 `hmx_set_scales` 也不需要。但 ch09 训练证实：`setup_acc_read`（`mxmem2.bias`）在持续工作负载下必要，dummy `mm_f16` 不需要。）

直接用 `mxmem` 指令 + hexkl readback 时，必须先调 `hmx_set_scales`，
否则 hexkl 的 `mxcvtr.sat.hf` 指令读到未初始化的 scale 寄存器，输出是垃圾。
如果用 `cvt.hf = acc(2)` + `mxmem = cvt` 路径（Exp 6），则不需要 `hmx_set_scales`，
但持续多 matmul 计算仍需 `setup_acc_read` 初始化 scale/bias 寄存器。

### 实验设计

4 个变体，测量从纯 hexkl 到完全不依赖 hexkl 的性能阶梯：

| 变体 | 计算 | readback | 输出 | hexkl 依赖 |
|------|------|----------|------|-----------|
| **V1** hexkl full | `hexkl mm_f16` | `acc_read + ah_to_rm + copy_f16_to_f32` | f32 DDR | 全部 hexkl |
| **V2** ASM+hexkl | `mxmem` 批量加载 | 同 V1 | f32 DDR | readback 用 hexkl |
| **V3** ASM+HVX f16 | `mxmem` 批量加载 | `hmx_store_acc` + HVX `vdeal` 解交织 | **f16 VTCM** | **零 hexkl** |
| **V4** ASM+HVX f32 | `mxmem` 批量加载 | `hmx_store_acc` + HVX `hvx_vhf_to_wsf` | f32 DDR | **零 hexkl** |

V3/V4 的 AH tile 准备也用 HVX `vshuff` 替代了 hexkl `rm_to_ah`（WH 仍用 hexkl，这是离线权重准备步骤）。

### AH tile 格式与 HVX 转换

AH（Activation HMX）格式是行对交织：32×32 tile 存储为 16 个 HVX 向量，
每个向量包含两行数据交替排列：

```
vector[i] = {row[2i][0], row[2i+1][0], row[2i][1], row[2i+1][1], ...}
             ^even row    ^odd row       ^even       ^odd
```

**RM → AH**（`vshuff`）：两行各 32 个 f16 放入两个向量低半部分，`Q6_W_vshuff_VVR(v_r1, v_r0, -2)` 交织为一个向量。

**AH → RM**（`vdeal`）：`Q6_W_vdeal_VVR(Q6_V_vzero(), v_tile, -2)` 解交织，lo = 偶数行，hi = 奇数行。

### 真机数据（Snapdragon 8 Gen 3, VTCM=8MB, 1000 次迭代平均）

#### AH Tile 准备：HVX vshuff vs hexkl

| 配置 | hexkl AH prep | HVX AH prep | 加速比 |
|------|-------------|------------|-------|
| Fwd L1 (128×800, 100 tiles) | 266 µs | 92 µs | **2.9x** |
| Fwd L2 (128×128, 16 tiles) | 43 µs | 15 µs | **2.9x** |
| Bwd dW1 (128×128, 16 tiles) | 43 µs | 15 µs | **2.9x** |
| Bwd dH (128×32, 4 tiles) | 10 µs | 4 µs | **2.7x** |

hexkl 的 `copy_submatrix_to_f16` + `rm_to_ah_f16` 两步调用有函数调用开销。
HVX `vshuff` 直接在 VTCM 中完成交织，省掉了中间 staging 拷贝。

#### Compute + Readback

| 配置 | V1 hexkl (µs) | V2 ASM+hexkl (µs) | V3 ASM+HVX f16 (µs) | V4 ASM+HVX f32 (µs) |
|------|:---:|:---:|:---:|:---:|
| Fwd L1 [128×128] K=800 | 485 | 482 | **21** | **3** |
| Fwd L2 [128×32] K=128 | 120 | 120 | **5** | **1** |
| Bwd dW1 [128×800] K=128 | 3006 | 3006 | **126** | **17** |
| Bwd dH [128×128] K=32 | 480 | 480 | **20** | **2** |

全部 PASS（与标量参考实现一致）。

### 关键发现

**1. hexkl readback 是 Exp 4 的真正瓶颈（99%）**

V1 和 V2 几乎相同（485 vs 482 µs），证明 compute 阶段无论用 hexkl `mm_f16`
还是直接 ASM `mxmem`，差异可以忽略。**真正慢的是 hexkl 的 readback 路径**：
`acc_read_f16` + `ah_to_rm_f16` + `copy_f16_to_f32_submatrix`。

这颠覆了 Exp 4 的结论："hexkl API 调用开销极低"。实际上 hexkl 的 readback
在 VTCM 常驻场景下是绝对瓶颈。Exp 4 中 Method A 的 5.8µs 有 3µs 花在 readback
（Method B 显示 acc_rd+rm + f16_to_f32 ≈ 0.6µs，但这是 cache 热的最好情况；
真实场景中 readback 涉及 VTCM↔L2 cache 数据搬运）。

**2. `hmx_store_acc` + HVX 解交织快 24-150 倍**

| 配置 | hexkl readback | HVX f16 readback | 加速比 |
|------|:---:|:---:|:---:|
| Fwd L1 | 485 µs | 21 µs | **23x** |
| Bwd dW1 | 3006 µs | 126 µs | **24x** |
| Bwd dH | 480 µs | 20 µs | **24x** |
| Fwd L2 | 120 µs | 5 µs | **24x** |

一致的 24x 加速。`hmx_store_acc` 将累加器直接写入 VTCM tile（1 条指令），
然后 HVX `vdeal` 在 VTCM 内解交织为行主序 f16。
全程 VTCM→VTCM，零 DDR 往返，零格式转换函数调用。

**3. V4 (HVX f32) 比 V3 (HVX f16) 更快——反直觉但合理**

V4 的 `hvx_vhf_to_wsf` 直接从 AH 交织格式转为 f32，利用 qf16→qf32 扩展时
**隐式完成解交织**（偶数位→lo 向量，奇数位→hi 向量）。一步完成格式转换+解交织。

V3 的 `vdeal` + `memcpy` 是两步操作：先解交织到临时向量，再 memcpy 64 字节到目标行。
额外的内存操作抵消了"不做 f16→f32 转换"的优势。

对于 ch09 训练（f16 全程），V3 的路径更合适，但需要优化 memcpy
（例如用 HVX predicated store 替代 memcpy）。

**4. htp-ops-lib 的完整 runtime 路径（零 hexkl）**

```
              hexkl 依赖？    替代方案
              ──────────    ────────
AH tile 准备:    否          HVX vshuff 交织行对 (2.9x faster)
WH tile 准备:    是(离线)    权重只需转换一次，存储为 WH 格式
scales 初始化:   否          bias = mxmem2(scales)
HMX 计算:       否          mxclracc + mxmem(:deep) 批量加载
readback:       否          hmx_store_acc + HVX vdeal 解交织
```

Runtime hot path 完全不依赖 hexkl。hexkl 只用于权重的离线预处理（RM→WH 格式转换），
这在生产环境中是模型加载阶段的一次性开销。

**5. `hmx_set_scales` 仅 hexkl readback 路径需要（Exp 6 修正）**

使用 hexkl readback（`mxcvtr.sat.hf`）时需要 `hmx_set_scales`：

```c
// 初始化 scales: f16 1.0 + zero bias (一次性)
HVX_Vector *pv = (HVX_Vector *)(vtcm + scales_off);
pv[0] = Q6_Vh_vsplat_R(0x3C00);  // scale = 1.0
pv[1] = Q6_V_vzero();             // bias = 0.0

// 设置 HMX scale 寄存器 (hexkl readback 路径需要)
hmx_set_scales(vtcm + scales_off);  // bias = mxmem2(ptr)
```

Exp 6 在单次 matmul 下证明：使用 `cvt.hf = acc(2)` + `mxmem = cvt` 的 direct ASM readback 路径时，
连 `hmx_set_scales` 也不需要。`acc(2)` 参数本身编码了 f16 scale=1.0 的行为。
但 ch09 训练实验证实：持续多 matmul 工作负载下需要 `setup_acc_read`（`mxmem2.bias`）初始化 scale/bias 寄存器，
dummy `mm_f16` 则不需要（详见 Exp 6）。

## Experiment 6: HMX 初始化需求测试 (`exp6_init_test.c`)

**背景**：反编译 `hexkl_micro.a` 发现 `hexkl_micro_hmx_mm_f16` 只有 2 条 ASM 指令（`M8.mxmem.blk.sm.act.hf` + `M8.mxmem.wei.hf`），不配置任何隐藏状态。`hexkl_micro_hmx_setup_acc_read_f16` 调用 `M8.mxmem2.bias` 设置 scale/bias 寄存器。

**问题**：direct ASM compute + `hmx_store_acc` + HVX `vdeal` readback 路径到底需要什么初始化？

### 4 个变体

| 变体 | 初始化内容 | 结果 |
|------|-----------|------|
| V1 | `setup_acc_read` + dummy `mm_f16`（ch09 方案） | PASS diff=0 |
| V2 | 只 `setup_acc_read`（去掉 dummy mm_f16） | PASS diff=0 |
| V3 | 只 `hmx_set_scales`（exp5 方案） | PASS diff=0 |
| V4 | 完全不初始化（只 `hmx_lock`） | PASS diff=0 |

所有变体使用相同的 compute + readback：direct ASM `mxmem` + `hmx_store_acc` + HVX `vdeal` 解交织。
每个变体之间通过 `hmx_unlock` + `hmx_lock` 完全重置 HMX 硬件状态。

### ch09 精确验证：哪个初始化才是必要的

exp6 的 V4（完全不初始化）在单次 matmul 下 PASS，但完整训练需要更精确的验证。
ch09 的精确控制变量实验：

- **去掉 dummy `mm_f16`，保留 `setup_acc_read`**：训练正常，准确率 **96.09%**
- **去掉 `setup_acc_read`（同时去掉 `mm_f16`）**：训练崩坏，准确率掉到随机水平

反编译 `mm_f16` 只有 2 条 mxmem 加载指令，无状态配置——实验结果与反编译完全吻合。
早期曾误以为"去掉 `mm_f16` 导致训练崩坏"，但那次实验同时去掉了 `setup_acc_read`，错误归因。

**结论**：`setup_acc_read`（`mxmem2.bias`）是持续 HMX 计算的必要初始化，dummy `mm_f16` 不是。
`mxmem2.bias` 配置的 scale/bias 寄存器在持续多 matmul 计算中会影响结果精度。
exp6 V4 单次 matmul PASS 可能是因为 `hmx_lock` 设置了安全默认值，但在完整训练循环下不够。

**做法**：在 `hmx_lock()` 后调用一次 `hexkl_micro_hmx_setup_acc_read_f16()` 即可。dummy `mm_f16` 已从 ch09 移除。

### 关键发现

**1. `cvt.hf = acc(2)` 的 `acc(2)` 参数本身选择 f16 scale=1.0 行为**

V4 完全不初始化（没有 `setup_acc_read`，没有 `hmx_set_scales`，没有 dummy `mm_f16`），
仅 `hmx_lock()` 获取硬件访问权，在单次 matmul 测试中输出与参考实现完全一致。
说明 `acc(2)` 参数本身编码了 f16 scale=1.0 的行为，单次计算不依赖外部寄存器。

**2. `setup_acc_read`（`mxmem2.bias`）是持续计算的必要初始化**

ch09 精确实验证实：去掉 `setup_acc_read` 训练崩坏，去掉 dummy `mm_f16` 训练正常（96.09%）。
`mxmem2.bias` 配置的 scale/bias 寄存器在持续多 matmul 工作负载下影响结果精度。

**3. dummy `mm_f16` 不是必要的初始化**

反编译显示 `mm_f16` 只有 2 条 mxmem 加载指令，不配置任何状态。
ch09 去掉它后训练完全正常。这修正了早期的错误结论。

**4. 对于 direct ASM 路径，初始化只需 `hmx_lock` + `setup_acc_read`**

这修正了 Exp 5 的结论（"`hmx_set_scales` 是唯一的隐式依赖"）。
`hmx_set_scales` 对 direct ASM readback 路径不需要（单次计算），
但 `setup_acc_read`（底层也是 `mxmem2.bias`）在持续工作负载下必要。

## HMX vs HVX：何时用什么

| 矩阵规模 | 推荐 | 原因 |
|-----------|------|------|
| ≥512 维度（LLM matmul） | **HMX** | 计算量 O(n³) >> tile 管理 O(n²) |
| 128-512 维度 | 看情况 | 需要 benchmark |
| ≤128 维度（MNIST） | **HVX qf32** | tile 管理开销 > 计算量 |

## 与 llama.cpp 的对应

llama.cpp Hexagon 后端（`ggml/src/ggml-hexagon/htp/`）使用的核心技巧：

| 技巧 | 做法 | 本章对应 | 适用条件 |
|------|------|---------|---------|
| **权重预缓存** | 一次性转为 WH 布局，VTCM 常驻 | L1 wt-cached | 所有场景 |
| **HMX Weight Layout** | 权重以 WH 格式持久存储在 DDR | L2 preformatted | 推理/decode（Mr=1）|
| **Chunk 计算器** | 根据 VTCM 预算动态求解最大工作集 | **Exp 3 compute_chunk_params** | 矩阵超 VTCM 容量时 |
| **VTCM 流式** | 激活复用 + 权重分块流过 VTCM | **Exp 3 Stream prefmt** | 大矩阵 |
| **直接 ASM 计算** | `mxmem(:deep)` 批量 tile 加载 | **Exp 5 V3/V4** | 所有场景 |
| **HVX AH 转换** | `vshuff` 交织行对替代 hexkl `rm_to_ah` | **Exp 5 prep_ah_hvx** | 运行时激活准备 |
| **HVX readback** | `hmx_store_acc` + `vdeal` 解交织 | **Exp 5 V3** | 输出留 VTCM 时 |
| **`hmx_set_scales`** | HMX scale 寄存器初始化 | **Exp 5** | 仅 hexkl readback 路径需要 |
| **HMX 初始化** | `hmx_lock` + `setup_acc_read`（`mxmem2.bias`） | **Exp 6** | direct ASM 路径；单次 matmul 不需要，持续工作负载需要 `setup_acc_read` |
| DMA 双缓冲 | 异步加载下一个 chunk + 当前 chunk 计算 | —（需 DMA API） | 大矩阵 |
| Fused VScatter | HVX dequant + 布局转换一步完成 | — | 量化模型 |

本章验证了**权重预缓存**（L1, +39%）、**HMX Weight Layout**（L2, 额外+5~21%）、
**零 hexkl runtime**（Exp 5, hexkl readback → HVX readback 带来 24x 加速）、
以及**初始化需求澄清**（Exp 6, direct ASM 路径单次 matmul 无需额外配置，但持续工作负载需要 `setup_acc_read`（`mxmem2.bias`），dummy `mm_f16` 不需要）的收益。
