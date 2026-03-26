# Chapter 9: NativeKV — 为什么高性能 KV Cache 需要新款 NPU

LLM 推理的核心瓶颈不是矩阵乘法本身，而是 **KV Cache 的布局转换**。本章用真机 HMX 实验揭示：

> Genie 引擎的 NativeKV 技术将 KV Cache 直接存储为 HMX Weight Layout，
> 跳过每步 attention 都要做的 `rm_to_wh` 转换。这个优化**只有 v75+ NPU** 才能支持。

## 文件结构

```
ch09-kv-cache/
├── README.md                  # 本文
├── build.sh                   # 编译 DSP .so
├── run_device.sh              # 推送到真机运行
└── src/
    └── demo_native_kv.c       # NativeKV vs SmartMask 基准测试
```

## 环境准备

需要第五章的 HexKL addon（`tools/hexkl-addon/`）。如未安装：

```bash
cd ../ch05-hexkl-matmul && bash install_hexkl.sh
```

## 构建和运行

```bash
bash build.sh          # 编译 DSP 端 .so
bash run_device.sh     # 推送到手机并运行
```

---

## Part 1: 问题 — 为什么 KV Cache 是性能瓶颈？

### Attention 计算中 K 的角色

在 transformer 的 attention 层：

```
scores = Q × K^T    # Q:[n_query, head_dim] × K:[head_dim, ctx_size] → [n_query, ctx_size]
output = softmax(scores) × V
```

K 在 HMX matmul 中充当 **weight**。HMX 要求 weight 必须是 WH (Weight HMX) 布局——一种 32×32 tile 的特殊排列。

### 两种策略

| | SmartMask (旧) | NativeKV (新) |
|---|---|---|
| **K 存储格式** | Row-major `[head_dim, ctx_size]` | HMX WH 布局 |
| **每步 attention** | `rm_to_wh_f16()` 转换整个 K | `copy_psubmatrix()` 拷贝 tile |
| **转换开销** | O(head_dim × ctx_size) | 0 |
| **追加新 KV** | 直接 memcpy | 需要 32-对齐写入 |
| **硬件要求** | 任意 Hexagon | v75+ (HMX Weight Layout 支持) |

SmartMask 每次 attention 都要把 K 从 row-major 转成 WH——对于 4K 上下文、128 维 head，这意味着**每步转换 512KB** 数据。NativeKV 把 K 永久存储为 WH 格式，彻底跳过转换。

---

## Part 2: 实验 — SmartMask vs NativeKV

### 实验设置

```
Q:  [32 × 64]   (n_query=32, head_dim=64)
K:  [64 × 256]  (head_dim=64, ctx_size=256)
→ scores: [32 × 256]
```

### 代码对比

**SmartMask** 内循环（每步都转换）：

```c
// 每次 attention 都要执行：DDR row-major → VTCM WH layout
hexkl_micro_hmx_rm_to_wh_f16(vtcm_base, weight_offset,
    K_rm, row_tile, col_tile, ctx_size);  // ← 转换！
hexkl_micro_hmx_mm_f16(vtcm_base, act_offset, weight_offset);
```

**NativeKV** 内循环（只拷贝）：

```c
// K 已经是 WH 格式，直接拷贝 tile 到 VTCM
hexkl_micro_hmx_copy_psubmatrix_to_f16_weight(vtcm_base, weight_offset,
    K_wh, row_tile, col_tile, head_dim, ctx_size);  // ← 只拷贝！
hexkl_micro_hmx_mm_f16(vtcm_base, act_offset, weight_offset);
```

### 真机结果

```
Part 1: Pre-format K → WH layout (one-time)
  Preformat: 2299 ticks (row_tiles=2, col_tiles=8)

Part 2: SmartMask (converts K every time)
  SmartMask verify: PASS
  SmartMask: 4937 ticks/iter

Part 3: NativeKV (K already in WH)
  NativeKV verify: PASS
  NativeKV: 4822 ticks/iter

Summary:
  Speedup: 1.02x
  Break-even: preformat cost recovered after 20 steps
```

在 64×256 这个小尺寸上，speedup 只有 1.02x——因为 HMX 的 tile 操作本身很快，`rm_to_wh` 和 `copy_psubmatrix` 在单个 32×32 tile 上的差距不大。

### 真正的收益在大上下文

| ctx_size | rm_to_wh tiles | 转换开销占比 |
|----------|---------------|------------|
| 256 | 2×8 = 16 tiles | ~2% |
| 4096 | 2×128 = 256 tiles | **~30%** |
| 32768 | 2×1024 = 2048 tiles | **~60%** |

随着上下文变长，`rm_to_wh` 的开销线性增长，最终主导整个 attention 计算。这就是 Genie 引擎在长上下文场景必须使用 NativeKV 的原因。

---

## Part 3: WH Layout 详解 — fromFlatOffset

### HMX Weight Layout 变换公式

Genie 引擎的核心布局函数 `fromFlatOffset`：

```
flat [din, dout]
  → [dout/N_TILE, din/32, (dout%N_TILE)/32, [(din%32)/4, dout%32, din%4]]
```

```c
static int fromFlatOffset(int DIN, int DOUT, int N_TILE, int din, int dout) {
    int tile_size   = min(DOUT, N_TILE);
    int tile_stride = DIN * tile_size;
    int tile_idx    = dout / tile_size;

    int dout_0 = (dout % tile_size) >> 5;   // tile 内 32-col 块
    int dout_1 = dout & 0x1f;               // 块内 col

    int din_0 = din >> 5;                    // 32-row 块
    int din_1 = (din & 0x1f) >> 2;           // 块内 4-row 组
    int din_2 = din & 0x3;                   // 组内 row

    int din_0_stride = tile_size << 5;       // tile_size * 32

    return tile_idx * tile_stride + din_0 * din_0_stride +
           (dout_0 << 10 | din_1 << 7 | dout_1 << 2 | din_2);
}
```

### 最内层 1024B block

每个 32×32 的 weight tile 占 1024 字节（对 int8）：

```
block[1024] = [8 组][32 列][4 行]
            = din_1 * 128 + dout_1 * 4 + din_2
```

这个排列让 HMX 可以一次加载 128 字节（一个 cache line），恰好是 32 个连续的 dout 元素——与 HMX 的 32-wide SIMD 对齐。

### Worked Example

```
Key Cache: embed_dim=128, ctx_size=512, K_TILE=256
要找 din=65, dout=300 的物理偏移：

tile_size   = min(512, 256) = 256
tile_stride = 128 × 256 = 32768

tile_idx = 300 / 256 = 1        → 第 2 个 tile
dout_0   = (300 % 256) / 32 = 1 → tile 内第 2 个 32-col 块
dout_1   = 300 % 32 = 12        → 块内第 12 列

din_0 = 65 / 32 = 2             → 第 3 个 32-row 块
din_1 = (65 % 32) / 4 = 0       → 第 0 组
din_2 = 65 % 4 = 1              → 组内第 1 行

offset = 1×32768 + 2×8192 + (1×1024 | 0×128 | 12×4 | 1)
       = 32768 + 16384 + 1024 + 48 + 1
       = 50225
```

真机验证：`[KV] Key [128,512] din=65 dout=300 K_TILE=256 -> offset=50225` ✓

---

## Part 4: K 和 V 的 Tile 差异

### Key: K_TILE = 256

```
K 在 attention 中: scores = Q × K^T
K 的 dout 维度 = ctx_size（上下文长度）
K_TILE=256: 每个 tile 覆盖 256 个 context position
→ 减少 VTCM 搬运次数（4096 上下文只需 16 个 tile）
```

### Value: V_TILE = 64

```
V 在 attention 中: output = softmax(scores) × V
V 的 dout 维度 = embed_dim（嵌入维度）
V_TILE=64: 典型 head_dim=128 → 2 个 tile
→ 与 HMX 的 VTCM 容量匹配
```

### 为什么 tile 大小不同？

K 的 ctx_size 维度很长（数千到数万），用大 tile 减少 VTCM 搬入次数。V 的 embed_dim 维度较短（64~128），小 tile 就够了。

---

## Part 5: 32-对齐约束

NativeKV 要求新 KV 写入位置必须对齐到 32：

```c
int new_idx = ceil(n_valid_kv / 32.0) * 32;
```

| n_valid_kv | new_idx | 跳过 |
|-----------|---------|------|
| 0 | 0 | 0 |
| 1 | 32 | 31 |
| 33 | 64 | 31 |

被跳过的位置在 attention 中被 **MaskedSoftmax** 屏蔽——这也是为什么 Genie 需要 MaskedSoftmax 优化。

### MaskedSoftmax

传统做法：`Softmax(Add(scores, mask))` — 先加 mask 再 softmax。
Genie 优化：将 mask 操作融入 softmax 内部，当有效 token 只占 ctx_size 的一小部分时（比如 decode 阶段），性能和精度都更好。

---

## Part 6: 为什么需要新款 NPU？

### HMX Weight Layout 是 v75 新特性

| 特性 | v68/v69 | v73 | v75+ |
|------|---------|-----|------|
| HVX (向量) | ✓ | ✓ | ✓ |
| HMX (矩阵) | ✗ | ✓ | ✓ |
| HMX Weight Layout 格式 | ✗ | ✗ | ✓ |
| QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT | ✗ | ✗ | ✓ |
| NativeKV Cache | ✗ | ✗ | ✓ |

NativeKV 依赖 NPU 能**直接消费** WH 布局的 tensor。v73 虽然有 HMX，但 QNN 编译器不支持将 KV I/O tensor 标记为 `HMX_WEIGHT_LAYOUT` 格式。只有 v75+（Snapdragon 8 Gen 3 为 v75，Snapdragon 8 Elite 为 v79）才完整支持。

### Genie 的自动检测

```cpp
bool isKvOutputHMXFormat =
    QNN_TENSOR_GET_DATA_FORMAT(key_out->tensor) ==
    QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT;

if (isKvOutputHMXFormat)
    manager = std::make_unique<NativeKV>(...);   // v75+
else
    manager = std::make_unique<SmartMask>(...);   // 回退
```

QNN 编译器根据目标硬件决定是否输出 HMX 格式。旧硬件自动回退到 SmartMask。

### 完整的 Genie KV Cache 架构

```
KVManager (全局调度器)
  └── CacheGroup ("past_")
        ├── CacheManager → NativeKV (v75+) 或 SmartMask (旧硬件)
        │     ├── updateKV(): 对齐 block memcpy (NativeKV) 或逐元素复制 (SmartMask)
        │     ├── reshapeCache(): variant/ctx_size 切换时重排内存
        │     └── fromFlatOffset(): WH 布局地址计算
        ├── ContextManager → SlidingWindow 或 KeyDiff
        │     ├── SlidingWindow: FIFO 驱逐 (sink tokens 保留)
        │     └── KeyDiff: 评分网络选择性驱逐 (per-head)
        └── KVTensor[] → 每层一个, 持有 key_buf/val_buf
```

---

## Part 7: 对 LLM 推理引擎的设计启示

### 分阶段优化路径

| 阶段 | KV 格式 | Attention 策略 | 适用场景 |
|------|---------|--------------|---------|
| Phase 1 | Row-major (SimpleKV) | SDKL 自动转换 | 快速原型 |
| Phase 2 | HMX WH (NativeKV) | 零转换 matmul | 性能优化 |
| Phase 3 | + SlidingWindow | FIFO 驱逐 | 长上下文 |
| Phase 4 | + INT8 量化 | 内存减半 | 大模型 |

### 内存预算 (Qwen3-0.6B, 28 层, 2 KV heads, head_dim=64)

| ctx_size | F16 KV 总量 | INT8 KV 总量 |
|----------|-----------|------------|
| 4096 | 56 MB | 28 MB |
| 32768 | 448 MB | 224 MB |

长上下文时 KV Cache 可能超过模型权重本身——这就是为什么 NativeKV 的零转换开销和 INT8 量化如此重要。
