# 第9章：VTCM + HMX 训练 -- 从 HVX 到矩阵加速器

## 1. 概述

ch08 的 HVX f32 训练基于 DDR 共享内存（rpcmem）。本章在此基础上叠加三个优化方向：

1. **VTCM 片上存储**：全部网络数据驻留在 VTCM（8MB，1 cycle 访问），消除 L2 cache miss
2. **f16 半精度**：数据量减半，HVX 每次处理 64 个元素（vs f32 的 32 个）
3. **HMX 矩阵加速**：用 Hexagon Matrix eXtension 硬件替代 HVX widening multiply 做 matmul

最终结论：**ch09 HMX + NativeKV 比 ch08 HVX 快 1.10x epoch / 1.24x DSP 时间**（1960ms vs 2150ms/epoch）。精度持平（96.09% vs 96.02%）。

## 2. 性能对比

| 配置 | Epoch 耗时 | DSP 总计 (5ep) | 测试精度 |
|------|-----------|----------------|----------|
| ch08 f32 HVX (DDR) | 2150ms | 8857ms | 96.02% |
| ch09 HMX (hexkl WH) | 1992ms | 7376ms | 96.09% |
| ch09 HMX + NativeKV | **1960ms** | **7154ms** | 96.09% |

加速比（vs ch08）：epoch **1.10x**，DSP 时间 **1.24x**。NativeKV 进一步省 3% DSP 时间。

## 3. HMX matmul 实现

### 3.1 四阶段流水线

每个 matmul C[m,n] = A[m,k] @ B[k,n] 分四个阶段：

```
1. AH prep  — HVX vshuff(-2) 将 A 的行对交织为 AH 格式（2行交织）
2. WH prep  — HVX vshuff(-2) 将 B 转换为 WH 格式（v75 f16: WH = AH = 2行交织）
3. Compute  — ASM mxclracc.hf + mxmem:deep 批量加载 tile 对
4. Readback — ASM hmx_store_acc + HVX vdeal(-2) 解交织回行优先
```

### 3.2 Direct ASM — 完全去 hexkl 热路径

**热路径（每 batch 执行）完全不依赖 hexkl**：

| 操作 | 实现 | hexkl? |
|------|------|--------|
| AH 转换 | HVX `vshuff(-2)` 交织行对 | 否 |
| WH 转换 | HVX `vshuff(-2)` 交织行对（v75 f16 WH = AH） | 否 |
| Compute | `mxclracc.hf` + `mxmem:deep` | 否 |
| Readback | `cvt.hf = acc` + `mxmem = cvt` + HVX `vdeal(-2)` | 否 |
| 初始化 | dummy `hexkl_micro_hmx_mm_f16` 一次 | 是（仅 setup） |

**关键发现：v75 f16 的 WH 格式 = AH 格式**（2 行交织），通过与 hexkl 输出逐字节对比验证。Genie 的 `fromFlatOffset` 4 行交织公式是 int8 专用，不适用于 f16。

### 3.3 HMX 有状态协处理器

**关键发现：HMX 是一个有隐式状态的协处理器。**

- `hexkl_micro_hmx_mm_f16` 在 setup 阶段必须执行**一次**，配置 HMX 内部的数据类型、累加器格式、scale/bias 寄存器
- 之后 direct ASM（`mxclracc.hf` + `mxmem:deep` + `hmx_store_acc`）复用这些配置
- **不能调 `hmx_set_scales()`（`bias = mxmem2`）**——它会覆盖 `mm_f16` 设置的配置，导致输出全零

这个发现来自 ch05/exp5 的独立实验：当 `hmx_set_scales` 被显式调用后，即使参数正确（scale=1.0, bias=0.0），HMX 输出也变成垃圾。根因是 `mm_f16` 和 `set_scales` 设置的寄存器格式不兼容。

### 3.4 NativeKV 策略 — 权重常驻 tile 格式

借鉴 Genie 引擎的 NativeKV 思路：**数据永久存储在 HMX 原生 tile 格式中**，消除每 batch 的格式转换开销。

NativeKV 在推理中的含义：KV cache 永久以 WH/AH 格式存储在 VTCM，新 token 转换一次后追加，消除 O(ctx_size) 的 `rm_to_wh` 瓶颈。

在训练中，我们将同样的思路应用于权重：

```
传统做法（每 batch）：
  Forward: RM(W) → WH(W) → HMX compute → AH(out) → RM(out) → bias/ReLU
  SGD:     W_rm -= lr * dW_rm
  下一 batch: 又要 RM→WH(W)  ← 重复 112 tiles 的转换！

NativeKV 做法：
  权重永久存储为 WH tiles（W1^T_wh: 104 tiles, W2^T_wh: 8 tiles）
  Forward: W_wh already ready → HMX compute → readback → bias/ReLU
  Backward dW: HMX output → 直接存为 WH tiles（AH = WH for f16!）
  SGD: W_wh -= lr * dW_wh  ← element-wise 对任何 layout 都成立
  下一 batch: W_wh already ready → 零转换！
```

**核心洞见：SGD 是 element-wise 操作**。`W[i] -= lr * dW[i]` 不关心数据布局，只要 W 和 dW 在同一格式下逐元素对应即可。由于 HMX 输出（AH 格式）= WH 格式（f16），梯度 dW 可以直接和权重 W 做 SGD。

#### VTCM WH tile 分区

```
wh_base + 0:           W1^T 永久存储 (104 tiles, 208KB)
wh_base + 104×2048:    W2^T 永久存储 (8 tiles, 16KB)
wh_base + 112×2048:    W2 永久存储   (8 tiles, 16KB) — 用于 backward dH
wh_base + 120×2048:    dW1 输出       (104 tiles, 208KB)
wh_base + 224×2048:    dW2 输出       (8 tiles, 16KB)
wh_base + 232×2048:    临时 WH        (32 tiles, 64KB) — backward B 操作数
```

#### 每 batch 操作

```
Forward:
  Fwd L1: AH(inp) + cached W1^T_wh → HMX → readback → bias+ReLU
  Fwd L2: AH(hidden) + cached W2^T_wh → HMX → readback → softmax

Backward:
  dW2: transpose(H) → AH + WH(dlogits) → HMX → 输出到 dW2_wh tiles
  dH:  AH(dlogits) + cached W2_wh → HMX → readback → ReLU backward
  dW1: transpose(X) → AH + WH(dhidden) → HMX → 输出到 dW1_wh tiles

SGD (tile 格式):
  W1^T_wh -= lr × dW1_wh    (104 tiles × 1024 f16)
  W2^T_wh -= lr × dW2_wh    (8 tiles × 1024 f16)
  b1, b2: RM 格式 SGD（不变）

Post-SGD 维护:
  W2^T_wh → RM (vdeal) → transpose → WH → W2_wh   (16 tiles)
  保持 v_w2_t RM 副本同步（16KB，用于下次 backward transpose）
```

#### 节省

每 batch 省掉：
- `convert_rm_to_wh(W1^T)` = 104 tiles（最大开销！）
- `convert_rm_to_wh(W2^T)` = 8 tiles
- `readback_acc_to_rm` for dW1 = 104 tiles 的 vdeal + memcpy
- `readback_acc_to_rm` for dW2 = 8 tiles
- `blocked_transpose(W2^T)` for backward dH

新增：post-SGD W2 维护 = 16 tiles。净省 ~208 tiles/batch。

### 3.5 训练中的转置处理

HMX 只支持 NN matmul（C = A @ B），不直接支持 TN/NT。反向传播需要：

```
dW2 = H^T @ dL   → transpose(H) to scratch, then NN matmul → WH tiles
dH  = dL @ W2    → 直接用 cached W2_wh（NativeKV 省掉了 transpose）
dW1 = X^T @ dH   → transpose(X) to scratch, then NN matmul → WH tiles
```

每个 batch 需要 2 次 `blocked_transpose_f16_vtcm`（NativeKV 前是 3 次）。

## 4. 其他优化

### 4.1 HVX 多项式 softmax

从 QNN SDK 的 `softmax_hf_approx` 移植，替换标量 `fast_expf()`。softmax 本身加速约 10x。

### 4.2 对齐 matmul

`NET_OUTPUT_DIM_PAD=64`，`NET_INPUT_DIM_PAD=832`。所有矩阵行步长 128B 对齐，消除非对齐 HVX 加载。

### 4.3 train-all 模式

跳过中间 eval，训练完所有 epoch 后再统一评估。每 epoch 节省约 80ms。

## 5. 历史发现

### 5.1 VTCM 数据腐败

**根因：DSP 空闲时 VTCM 被系统回收。** 解决：DSP 侧执行 eval（OP_EVAL），保持 DSP 忙碌。

### 5.2 VTCM 标量访问代价

VTCM 标量读取 10+ cycles（vs L2 的 2 cycles）。解决：全 HVX 向量化访问。

### 5.3 HVX f16 widening multiply

v75 无直接 f16 乘法。`Wqf32 = Vhf * Vhf` widening multiply + qf32 累加是标准做法。

### 5.4 VTCM 非对齐访问

`vmem(Rx)` 静默向下对齐到 128B。非对齐地址返回错误数据。通过维度 padding 彻底消除。

### 5.5 HMX 小矩阵开销

MNIST 维度（128x832, 128x128）下，tile 格式转换（RM↔AH/WH）开销显著。对于 LLM 级别的大矩阵（4096x4096），HMX compute 优势才能覆盖转换开销。

## 6. 架构

### 数据流

```
OP_REGISTER_NET:
  1. hexkl_micro_hw_init 获取 VTCM + hmx_lock
  2. Bump-allocate f16 缓冲区（128B 对齐）
  3. setup_hmx_workspace: AH/WH tile 区域 + dummy mm_f16 初始化
  4. f32 权重 DDR -> f16 转置 VTCM

OP_TRAIN_BATCH:
  1. f16 输入 DDR -> VTCM（HVX bulk copy）
  2. Forward: AH prep → HMX with cached W_wh → readback → bias+ReLU
  3. Backward dW: transpose → AH/WH prep → HMX → 输出到 dW WH tiles
  4. Backward dH: AH prep → HMX with cached W2_wh → readback
  5. SGD: HVX element-wise 更新（直接操作 WH tile 格式权重）
  6. Post-SGD: 同步 W2_wh 和 v_w2_t RM 副本

OP_EVAL:
  1. Forward only（复用 HMX matmul）
  2. 保持 DSP 忙碌，防止 VTCM 回收
```

### VTCM 内存布局

```
[0 .. ~600KB]     f16 数据缓冲区（b1/b2, v_w2_t, 激活, scratch）
                  NativeKV 省掉了 v_w1_t(200KB) + v_dw1_t(200KB) + v_dw2_t(16KB)
[~600KB .. ~1MB]  HMX AH tiles (208 × 2048B = 416KB)
[~1MB .. ~1.5MB]  HMX WH tiles (264 × 2048B = 528KB)
                  W1^T(104) + W2^T(8) + W2(8) + dW1(104) + dW2(8) + temp(32)
[+staging]        2 个 staging tiles (readback 用)
[end-cfg_size]    hexkl config (acc_read 配置)
```

VTCM 8MB，总使用约 2MB，空间充裕。

## 7. 文件

```
ch09-vtcm-train/
├── build.sh
├── run_device.sh
├── README.md
└── src/
    ├── arm/
    │   └── train_vtcm_dspq.c         # ARM 侧训练循环
    ├── common/
    │   ├── protocol.h                 # dspqueue 消息协议
    │   └── hvx_ops_f16.h             # HVX f16 ops（自包含）
    └── dsp/
        ├── skel_vtcm.c                # 训练 skel：forward/backward/SGD
        ├── hmx_matmul_f16_vtcm.h      # HMX matmul：ASM compute + HVX tile 转换
        └── hvx_matmul_f16_vtcm.h      # HVX matmul（transpose 辅助 + DDR scratch）
```

## 8. Build & Run

```bash
bash build.sh
bash run_device.sh          # 5 epochs, batch=128
bash run_device.sh 3 64     # custom
```

## 9. 总结

1. **HMX 替代 HVX matmul 有效。** direct ASM compute + HVX tile 转换，DSP 时间快 1.24x。
2. **HMX 是有状态协处理器。** `hexkl_micro_hmx_mm_f16` 必须在 setup 执行一次配置隐式寄存器；之后 pure ASM 复用。`hmx_set_scales` 会破坏这个配置。
3. **AH 用 HVX vshuff，readback 用 HVX vdeal。** 2 行交织/解交织，ch05/exp5 证明比 hexkl readback 快 24x。
4. **v75 f16: WH = AH（2 行交织）。** 通过逐字节对比 hexkl 输出验证。Genie `fromFlatOffset` 的 4 行交织是 int8 专用。WH 转换完全用 HVX vshuff 实现，热路径零 hexkl 依赖。
5. **NativeKV 思路适用于训练。** 权重永久 WH 格式，梯度直接输出 WH 格式，SGD element-wise 在 tile 格式上执行。每 batch 省 ~208 tiles 的格式转换。
6. **训练特有挑战：反向传播需要转置。** HMX 只支持 NN matmul，每 batch 2 次 transpose（NativeKV 省掉了 W2^T 的转置）。
7. **VTCM 常驻 + HMX 是自然组合。** HMX 要求操作数在 VTCM 中，数据已在 VTCM 时切换到 HMX 几乎零额外成本。
8. **小矩阵下 HMX 优势有限。** MNIST 维度的 tile 转换开销相对显著，LLM 级大矩阵才能充分发挥 HMX 吞吐优势。
