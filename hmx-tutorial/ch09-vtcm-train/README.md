# 第9章：VTCM + f16 训练 -- 从 DDR/L2 到片上存储的探索

## 1. 概述

ch08 的 HVX 训练基于 DDR 共享内存（rpcmem），通过 L2 cache 访问。本章在此基础上叠加两个优化方向：

1. **VTCM 片上存储**：将全部网络数据（权重、激活、梯度）驻留在 VTCM（8MB，1 cycle 访问），消除 L2 cache miss
2. **f16 半精度**：数据量减半，HVX 每次处理 64 个元素（vs f32 的 32 个），理论 2x 吞吐

最终结论：**ch09 比 ch08 快 1.37x**（1588ms vs 2168ms/epoch）。f16（2x 向量宽度）+ VTCM（1-cycle HVX 访问）+ 多项式 softmax + 零转置反向传播的组合，克服了早期 VTCM 标量访问的性能惩罚。

## 2. 性能对比

| 配置 | Epoch 耗时 | DSP 总计 (5ep) | 测试精度 |
|------|-----------|----------------|----------|
| ch08 f32 DDR baseline | 2168ms | 8953ms | 96.02% |
| ch09 f16 VTCM per-batch | 1668ms | 5759ms | 96.09% |
| ch09 f16 VTCM train-all | 1588ms | 5790ms | 96.09% |

ch09 最优配置（train-all）比 ch08 f32 baseline 快 **1.37x**（1588ms vs 2168ms/epoch）。f16 精度与 f32 持平（96.09% vs 96.02%）。

## 3. 关键优化

### 3.1 HVX 多项式 softmax

从 QNN SDK 的 `softmax_hf_approx` 移植了多项式近似 softmax，替换原来的标量 `fast_expf()` 实现。softmax 本身加速约 10x。核心思路是用 HVX f16 多项式逼近 exp()，全向量化执行，避免逐元素标量运算。

### 3.2 对齐 matmul

将 `NET_OUTPUT_DIM_PAD` 从 32 pad 到 64，`NET_INPUT_DIM_PAD` 从 800 pad 到 832。所有矩阵行步长变为 128 字节的倍数，消除非对齐 HVX 加载。例如 832*2=1664=13*128，所有路径走快速对齐加载。

### 3.3 消除权重重新转置

新增 `matmul_nt_f16`（A @ B^T），直接以转置形式计算梯度（dW_T），避免每个 batch 执行 2 次 `blocked_transpose`。反向传播中：
- dhidden = dlogits @ W2，当只存储 W2_T 时，用 `matmul_nt_f16(dhidden, dlogits, W2_T, bs, 128, 64)` 直接计算
- 梯度 dW 直接以转置形式累加，无需转回再转置

### 3.4 移除标量 VTCM wrapper

早期版本为了处理 VTCM 标量访问慢的问题，包装了一层标量访问函数。现在改为直接使用 `hvx_ops_f16.h` 中的向量化函数操作对齐的 VTCM 数据，避免额外间接层。

### 3.5 train-all 模式

跳过中间 eval，训练完所有 epoch 后再统一评估一次。每 epoch 节省约 80ms（省去 DSP->ARM 切换、eval forward pass、结果回传的开销）。

### 3.6 与 htp-ops-lib / QNN SDK 的对比

matmul 核心的 widening multiply + qf32 accumulation 模式是 v75 HVX 的标准做法，htp-ops-lib 和 QNN SDK 示例中都采用。

**[htp-ops-lib](https://github.com/haozixu/htp-ops-lib)**（开源，haozixu 开发，被 llama.cpp Hexagon 后端使用）：
- **多线程**：worker pool 实现多 HVX 硬件线程并行（本章尚未实现）
- **HMX matmul**：权重 WH 预格式化 + ASM wrapper（ch05 的 HMX matmul 参考了此模式）
- **Taylor 级数 exp**：`1+x+x²/2!+...+x⁷/7!`（与本章的多项式 softmax 不同）
- **DMA 数据搬运**：基于 DMA 的显式数据移动

**QNN SDK**（Qualcomm 私有 SDK）：
- **多项式 softmax**：本章的 HVX 多项式 softmax 移植自 QNN SDK 的 `ExampleOpPackageSoftmax.cpp`（f16 多项式 exp + repeated squaring）
- **Crouton 布局**：HMX 友好的 tile 数据布局，对大矩阵更高效
- **AUTOSPLIT tiling / DEF_PACKAGE_OP**：QNN 算子包的标准模式

## 4. 历史发现（调试过程中的关键经验）

### 4.1 VTCM 数据腐败的根因

调试过程排除了 L2 cache 一致性、memcpy 错误、VTCM coherence 等假设。

**根因：DSP 空闲时，调度器会将 VTCM 分配给其他系统客户端（camera、audio、NN 服务）。**

关键证据：
- 仅 OP_SYNC（不做 eval）：连续 10 epoch **正常**
- OP_SYNC + ARM 侧 `cpu_evaluate`（DSP 空闲）：Epoch 2 即**腐败**

**解决方案：让 DSP 保持忙碌。** 将 evaluation 也搬到 DSP 侧执行（OP_EVAL），DSP 始终持有 VTCM，不被回收。

### 4.2 VTCM 标量访问代价

VTCM 专为 HVX 宽端口设计（1024-bit，1 cycle）。标量访问 VTCM 约 10+ cycles，而 L2 cache 仅约 2 cycles。

早期版本 matmul 内循环的瓶颈：`a_val = a_row[p]` 标量读取。即使 A 矩阵在 VTCM 中，标量读取的高延迟导致整体变慢。

解决方案：将 A 行从 VTCM 预加载到栈上（L2），标量读取走 L2 快速路径：

```c
// 从 VTCM 预加载到栈（HVX 宽端口，1 cycle）
preload_row_f16(a_local, a_row, k);
// 标量读取走 L2（2 cycles）
_Float16 a_val = a_local[p];
```

### 4.3 HVX f16 widening multiply（v75 正确做法）

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
- 这是 v75 HVX 的标准做法，htp-ops-lib 和 QNN SDK 示例中都采用

### 4.4 VTCM 非对齐访问陷阱

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

这是 VTCM 编程的关键陷阱 -- 不会报错，只会返回错误结果。通过对齐优化（3.2 节），当前版本已消除所有非对齐访问。

### 4.5 VTCM 何时有用 vs 无用

| 场景 | VTCM 优势 | VTCM 劣势 |
|------|----------|----------|
| 工作集 > L2（大模型） | 消除 L2 miss，1-cycle 访问 | -- |
| HMX 操作 | HMX 要求数据在 VTCM | -- |
| 纯 HVX 流式处理 | 宽端口 1-cycle | -- |
| 小工作集（装入 L2） | -- | 标量访问慢，额外 copy 开销 |
| 含标量访问模式的代码 | -- | 10+ cycles vs L2 的 2 cycles |

经过对齐优化和消除标量访问后，VTCM 在本场景下也能发挥优势。

### 4.6 HMX 小矩阵开销

HMX 处理小矩阵（如 128x800）时，tile 管理开销过大：
- 格式转换：RM -> AH/WH（行优先 -> HMX 内部布局）
- 每个 tile：acc_clear + 乘累加 + acc_read
- AH -> RM 转回行优先

对于 MNIST 维度，HVX matmul 比 HMX 更快。HMX 的优势在大矩阵（如 LLM 的 4096x4096）。

## 5. 架构

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
| W1, W1_T, dW1 | 128x832 x3 | 624 KB |
| W2, W2_T, dW2 | 64x128 x3 | 48 KB |
| B1, B2 | 128 + 64 | 0.4 KB |
| hidden + hidden_pre | 128x128 x2 | 64 KB |
| logits + probs + dlogits | 128x64 x3 | 48 KB |
| dhidden | 128x128 | 32 KB |
| input (per batch) | 128x832 | 208 KB |
| **总计** | | **~1 MB** |

VTCM 有 8MB，空间充裕。

## 6. 文件

```
ch09-vtcm-train/
├── build.sh
├── run_device.sh
├── README.md
└── src/
    ├── arm/
    │   └── train_vtcm_dspq.c       # ARM 侧训练循环 + DSP 评估
    ├── common/
    │   └── ...                      # 共享协议/常量定义
    └── dsp/
        ├── skel_vtcm.c              # VTCM f16 训练 skel（dspqueue callback）
        ├── hvx_matmul_f16_vtcm.h    # f16 HVX matmul（widening multiply + matmul_nt）
        └── hvx_matmul_vtcm.h        # f32 HVX matmul（参考实现）
```

## 7. Build & Run

```bash
bash build.sh
bash run_device.sh          # 5 epochs, batch=128
bash run_device.sh 3 64     # custom
```

## 8. 总结

本章的核心经验：

1. **f16 + VTCM 组合有效。** 经过充分优化后，ch09 比 ch08 快 1.37x（1588ms vs 2168ms/epoch），精度持平（96.09% vs 96.02%）。
2. **多项式 softmax 是关键优化。** 从 QNN SDK 移植的 HVX 多项式近似 softmax 替换标量 fast_expf()，softmax 本身加速约 10x。
3. **对齐消除是基础。** 将维度 pad 到 64 的倍数（f16 HVX 宽度），确保所有 HVX load/store 走对齐快速路径。
4. **消除转置开销。** matmul_nt_f16 直接以转置形式计算梯度，省去每 batch 2 次 blocked_transpose。
5. **VTCM 在 DSP 空闲时会被回收。** 必须通过保持 DSP 忙碌（如 DSP 侧 eval）来防止数据腐败。
6. **f16 widening multiply 是 v75 的标准做法。** htp-ops-lib 和 QNN SDK 示例中都采用此模式。
7. **VTCM 的 vmem 对齐行为是静默陷阱。** 非 128B 对齐的地址会返回错误数据，对齐优化后已完全消除。
8. **与生产级实现的差距。** 多线程（htp-ops-lib 的 worker pool）和 Crouton 布局（QNN SDK）是主要剩余差距。
