# 第9章：VTCM + HMX 训练 -- 从 HVX 到矩阵加速器

## 1. 概述

ch08 的 HVX f32 训练基于 DDR 共享内存（rpcmem）。本章在此基础上叠加三个优化方向：

1. **VTCM 片上存储**：全部网络数据驻留在 VTCM（8MB，1 cycle 访问），消除 L2 cache miss
2. **f16 半精度**：数据量减半，HVX 每次处理 64 个元素（vs f32 的 32 个）
3. **HMX 矩阵加速**：用 Hexagon Matrix eXtension 硬件替代 HVX widening multiply 做 matmul

最终结论：**ch09 HMX 比 ch08 HVX 快 1.08x epoch / 1.21x DSP 时间**（1992ms vs 2147ms/epoch）。精度持平（96.09% vs 96.02%）。

## 2. 性能对比

| 配置 | Epoch 耗时 | DSP 总计 (5ep) | 测试精度 |
|------|-----------|----------------|----------|
| ch08 f32 HVX (DDR) | 2147ms | 8906ms | 96.02% |
| ch09 f16 HMX (VTCM) | **1992ms** | **7376ms** | 96.09% |

加速比：epoch **1.08x**，DSP 时间 **1.21x**。

## 3. HMX matmul 实现

### 3.1 四阶段流水线

每个 matmul C[m,n] = A[m,k] @ B[k,n] 分四个阶段：

```
1. AH prep  — HVX vshuff(-2) 将 A 的行对交织为 AH 格式（2行交织）
2. WH prep  — hexkl rm_to_wh 将 B 转换为 WH 格式（4行×32列交织）
3. Compute  — ASM mxclracc.hf + mxmem:deep 批量加载 tile 对
4. Readback — ASM hmx_store_acc + HVX vdeal(-2) 解交织回行优先
```

### 3.2 Direct ASM vs hexkl

**热路径（每 batch 执行）**用 direct ASM + HVX，**不依赖 hexkl**：

| 操作 | 实现 | hexkl? |
|------|------|--------|
| AH 转换 | HVX `vshuff(-2)` 交织行对 | 否 |
| WH 转换 | `hexkl_micro_hmx_rm_to_wh_f16` | 是（唯一剩余） |
| Compute | `mxclracc.hf` + `mxmem:deep` | 否 |
| Readback | `cvt.hf = acc` + `mxmem = cvt` + HVX `vdeal(-2)` | 否 |
| 初始化 | dummy `hexkl_micro_hmx_mm_f16` 一次 | 是（仅 setup） |

WH 转换仍用 hexkl 是因为 `rm_to_wh_f16` 有 `restrict` 限定符，VTCM 源指针会导致错误结果，必须先拷贝到 DDR。

### 3.3 HMX 有状态协处理器

**关键发现：HMX 是一个有隐式状态的协处理器。**

- `hexkl_micro_hmx_mm_f16` 在 setup 阶段必须执行**一次**，配置 HMX 内部的数据类型、累加器格式、scale/bias 寄存器
- 之后 direct ASM（`mxclracc.hf` + `mxmem:deep` + `hmx_store_acc`）复用这些配置
- **不能调 `hmx_set_scales()`（`bias = mxmem2`）**——它会覆盖 `mm_f16` 设置的配置，导致输出全零

这个发现来自 ch05/exp5 的独立实验：当 `hmx_set_scales` 被显式调用后，即使参数正确（scale=1.0, bias=0.0），HMX 输出也变成垃圾。根因是 `mm_f16` 和 `set_scales` 设置的寄存器格式不兼容。

### 3.4 WH 缓存策略

前向传播中 W1^T 和 W2^T 不变，WH tiles 只转换一次：

```
Forward:
  convert_wh(W1^T) → cache at wh_w1t_off (104 tiles)
  convert_wh(W2^T) → cache at wh_w2t_off (8 tiles)
  Fwd L1: hmx_matmul_cached_wh(inp, wh_w1t_off)    ← 复用 WH
  Fwd L2: hmx_matmul_cached_wh(hidden, wh_w2t_off)  ← 复用 WH

Backward:
  dW2, dH, dW1: hmx_matmul_nn (每次重新转 WH，因为操作数不同)
```

### 3.5 训练中的转置处理

HMX 只支持 NN matmul（C = A @ B），不直接支持 TN/NT。反向传播需要：

```
dW2 = H^T @ dL   → transpose(H) to scratch, then NN matmul
dH  = dL @ W2    → transpose(W2^T) to scratch, then NN matmul
dW1 = X^T @ dH   → transpose(X) to scratch, then NN matmul
```

每个 batch 需要 3 次 `blocked_transpose_f16_vtcm`，这是当前实现的主要开销来源之一。

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
  2. Forward: AH prep → HMX compute → readback → bias+ReLU
  3. Backward: transpose → AH/WH prep → HMX compute → readback
  4. SGD: HVX element-wise 更新（直接操作 RM 格式权重）

OP_EVAL:
  1. Forward only（复用 HMX matmul）
  2. 保持 DSP 忙碌，防止 VTCM 回收
```

### VTCM 内存布局

```
[0 .. ~1MB]     f16 数据缓冲区（权重、激活、梯度、scratch）
[~1MB .. ~2MB]  HMX AH tiles (208 × 2048B = 416KB)
[~2MB .. ~2.2MB] HMX WH tiles (112 × 2048B = 224KB)
[+staging]       2 个 staging tiles (readback 用)
[end-cfg_size]   hexkl config (acc_read 配置)
```

VTCM 8MB，总使用约 2.3MB，空间充裕。

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

1. **HMX 替代 HVX matmul 有效。** direct ASM compute + HVX tile 转换，DSP 时间快 1.21x。
2. **HMX 是有状态协处理器。** `hexkl_micro_hmx_mm_f16` 必须在 setup 执行一次配置隐式寄存器；之后 pure ASM 复用。`hmx_set_scales` 会破坏这个配置。
3. **AH 用 HVX vshuff，readback 用 HVX vdeal。** 2 行交织/解交织，ch05/exp5 证明比 hexkl readback 快 24x。
4. **WH 转换仍需 hexkl。** `rm_to_wh` 的 `restrict` 限定符要求 DDR 源，VTCM 指针会别名错误。
5. **训练特有挑战：反向传播需要转置。** HMX 只支持 NN matmul，每 batch 3 次 transpose 是主要开销。
6. **VTCM 常驻 + HMX 是自然组合。** HMX 要求操作数在 VTCM 中，数据已在 VTCM 时切换到 HMX 几乎零额外成本。
7. **小矩阵下 HMX 优势有限。** MNIST 维度的 tile 转换开销相对显著，LLM 级大矩阵才能充分发挥 HMX 吞吐优势。
