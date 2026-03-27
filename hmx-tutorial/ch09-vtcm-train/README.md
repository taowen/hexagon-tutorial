# 第9章：VTCM + f16 训练 -- 从 DDR/L2 到片上存储的探索

## 1. 概述

ch08 的 HVX 训练基于 DDR 共享内存（rpcmem），通过 L2 cache 访问。本章尝试两个优化方向：

1. **VTCM 片上存储**：将全部网络数据（权重、激活、梯度）驻留在 VTCM（8MB，1 cycle 访问），消除 L2 cache miss
2. **f16 半精度**：数据量减半，HVX 每次处理 64 个元素（vs f32 的 32 个），理论 2x 吞吐

最终结论：**对于 MNIST 规模（690KB 工作集），ch08 的 f32 + L2 cache 反而是最优方案。** VTCM 和 f16 各有适用场景，但不在这里。

## 2. 性能对比

| 配置 | Epoch 耗时 | DSP 总计 (5ep) | 测试精度 |
|------|-----------|----------------|----------|
| ch08 f32 DDR (L2 cache) | 1.85s | 7.5s | 95.6% |
| ch09 f16 VTCM (标量转换) | 5.0s | 23s | 96.9% |
| ch09 f16 VTCM (Wqf32 widening) | 3.25s | 13.5s | 96.9% |

f16 精度略高（96.9% vs 95.6%），但速度慢了 1.75x（最优 f16 配置 vs f32 baseline）。

## 3. 关键发现

### 3.1 VTCM 数据腐败的根因

调试过程排除了 L2 cache 一致性、memcpy 错误、VTCM coherence 等假设。

**根因：DSP 空闲时，调度器会将 VTCM 分配给其他系统客户端（camera、audio、NN 服务）。**

关键证据：
- 仅 OP_SYNC（不做 eval）：连续 10 epoch **正常**
- OP_SYNC + ARM 侧 `cpu_evaluate`（DSP 空闲）：Epoch 2 即**腐败**

**解决方案：让 DSP 保持忙碌。** 将 evaluation 也搬到 DSP 侧执行（OP_EVAL），DSP 始终持有 VTCM，不被回收。

### 3.2 VTCM 标量访问代价

VTCM 专为 HVX 宽端口设计（1024-bit，1 cycle）。标量访问 VTCM 约 10+ cycles，而 L2 cache 仅约 2 cycles。

matmul 内循环的瓶颈：`a_val = a_row[p]` 标量读取。即使 A 矩阵在 VTCM 中，标量读取的高延迟导致整体变慢。

**对于 690KB 的工作集完全装入 L2 的场景，DDR + L2 cache 反而比 VTCM 更快。**

解决方案：将 A 行从 VTCM 预加载到栈上（L2），标量读取走 L2 快速路径：

```c
// 从 VTCM 预加载到栈（HVX 宽端口，1 cycle）
preload_row_f16(a_local, a_row, k);
// 标量读取走 L2（2 cycles）
_Float16 a_val = a_local[p];
```

### 3.3 HVX f16 widening multiply（v75 正确做法）

v75 没有直接的 f16 乘法指令。正确方式是 **widening multiply**：

```c
// f16 输入，qf32 输出对（64 个 f16 元素 -> 2x32 qf32）
HVX_VectorPair prod = Q6_Wqf32_vmpy_VhfVhf(a_f16, b_f16);

// qf32 累加（f32 精度）
acc_lo = Q6_Vqf32_vadd_Vqf32Vqf32(acc_lo, Q6_V_lo_W(prod));
acc_hi = Q6_Vqf32_vadd_Vqf32Vqf32(acc_hi, Q6_V_hi_W(prod));

// qf32 对转回 f16 向量
HVX_Vector result = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(acc_hi, acc_lo));
```

特点：
- 每次迭代处理 64 个 f16 元素（vs f32 的 32 个）
- 内部 f32 精度累加（和 HMX 内部行为一致）
- 零显式 f16<->f32 转换开销
- 这也是 htp-ops-lib / QNN SDK 示例的标准做法

早期版本使用标量 `f16->f32` 转换 + f32 HVX 运算，5.0s/epoch。改用 widening multiply 后降至 3.25s/epoch（1.54x 加速）。

### 3.4 VTCM 非对齐访问陷阱

`vmem(Rx)` 会静默地将 Rx 向下对齐到 128 字节边界。当矩阵行步长不是 128 字节对齐时（如 n=32 步长=64B，n=800 步长=1600B），HVX load 返回**错误数据**。

修复方法：使用 `valign` 处理非对齐加载：

```c
static inline HVX_Vector hvx_load_unaligned(const void *ptr) {
    uintptr_t addr = (uintptr_t)ptr;
    int offset = (int)(addr & 127);
    const HVX_Vector *base = (const HVX_Vector *)(addr & ~(uintptr_t)127);
    if (offset == 0) return base[0];
    return Q6_V_valign_VVR(base[1], base[0], offset);
}
```

这是 VTCM 编程的关键陷阱 -- 不会报错，只会返回错误结果。

### 3.5 VTCM 何时有用 vs 无用

| 场景 | VTCM 优势 | VTCM 劣势 |
|------|----------|----------|
| 工作集 > L2（大模型） | 消除 L2 miss，1-cycle 访问 | -- |
| HMX 操作 | HMX 要求数据在 VTCM | -- |
| 纯 HVX 流式处理 | 宽端口 1-cycle | -- |
| 小工作集（装入 L2） | -- | 标量访问慢，额外 copy 开销 |
| 含标量访问模式的代码 | -- | 10+ cycles vs L2 的 2 cycles |

**对于 MNIST（690KB），ch08 f32 + L2 cache 是最优选择。**

### 3.6 HMX 小矩阵开销

HMX 处理小矩阵（如 128x800）时，tile 管理开销过大：
- 格式转换：RM -> AH/WH（行优先 -> HMX 内部布局）
- 每个 tile：acc_clear + 乘累加 + acc_read
- AH -> RM 转回行优先

对于 MNIST 维度，HVX matmul 比 HMX 更快。HMX 的优势在大矩阵（如 LLM 的 4096x4096）。

## 4. 架构

### 数据流

```
OP_REGISTER_NET:
  1. hexkl_micro_hw_init 获取 VTCM
  2. Bump-allocate 所有 f16 缓冲区（128B 对齐）
  3. f32 权重 DDR -> f16 VTCM（一次性）
  4. 预转置 W1_T, W2_T（前向传播用转置权重）

OP_TRAIN_BATCH:
  1. f16 输入 DDR -> VTCM（HVX bulk copy）
  2. 前向/反向/SGD 全部在 VTCM f16 上执行
  3. 重新转置更新后的权重
  4. 返回 loss + accuracy

OP_EVAL:（取代 OP_SYNC + ARM cpu_evaluate）
  1. f16 测试输入 DDR -> VTCM
  2. 前向传播（使用 VTCM 权重）
  3. 返回正确数量
  4. 保持 DSP 忙碌，防止 VTCM 被回收

OP_SYNC:（仅在需要时同步权重回 DDR）
  1. f16 VTCM -> f32 DDR 权重转换
```

### 内存布局（batch=128, f16）

| 缓冲区 | 形状 | 大小 |
|--------|------|------|
| W1, W1_T, dW1 | 128x800 x3 | 600 KB |
| W2, W2_T, dW2 | 32x128 x3 | 24 KB |
| B1, B2 | 128 + 32 | 0.3 KB |
| hidden + hidden_pre | 128x128 x2 | 64 KB |
| logits + probs + dlogits | 128x32 x3 | 24 KB |
| dhidden | 128x128 | 32 KB |
| input (per batch) | 128x800 | 200 KB |
| **总计** | | **~690 KB** |

VTCM 有 8MB，空间充裕。

## 5. 文件

```
ch09-vtcm-train/
├── build.sh
├── run_device.sh
├── README.md
└── src/
    ├── arm/
    │   └── train_vtcm_dspq.c     # ARM 侧训练循环 + DSP 评估
    └── dsp/
        ├── skel_vtcm.c            # VTCM f16 训练 skel（dspqueue callback）
        └── hvx_matmul_f16_vtcm.h  # f16 HVX matmul（widening multiply + valign）
```

## 6. Build & Run

```bash
bash build.sh
bash run_device.sh          # 5 epochs, batch=128
bash run_device.sh 3 64     # custom
```

## 7. 总结

本章的核心教训：

1. **VTCM 不是万能的。** 对于装入 L2 的小工作集，VTCM 的标量访问代价反而拖慢速度。
2. **VTCM 在 DSP 空闲时会被回收。** 必须通过保持 DSP 忙碌（如 DSP 侧 eval）来防止数据腐败。
3. **f16 widening multiply 是 v75 的正确 f16 做法。** 不需要标量 f16<->f32 转换，直接 f16 输入、qf32 累加、f16 输出。
4. **VTCM 的 vmem 对齐行为是静默陷阱。** 非 128B 对齐的地址会返回错误数据，必须用 valign 处理。
5. **HMX 对小矩阵开销过大。** MNIST 维度下 HVX matmul 更快。

VTCM 的真正用武之地：大模型（工作集超出 L2）、HMX 操作、纯 HVX 流式处理。
