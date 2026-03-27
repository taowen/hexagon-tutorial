# 第12章：HVX+dspqueue 训练 MNIST

## 1. 概述

在骁龙 8 Gen 3 的 Hexagon DSP 上真实训练 MNIST（不只是推理），使用 2 层 MLP 网络（800->128->10），
对比 4 种训练模式的性能：

| 模式 | 通信方式 | matmul 执行 |
|------|----------|-------------|
| **CPU** | 无（纯 ARM 大核） | f32 标量 matmul |
| **FastRPC + HVX** | ARM->DSP 内核态切换（每次 matmul） | DSP HVX 向量化 f32 matmul |
| **dspqueue + HVX (5次通信)** | ARM->DSP 用户态通信，每 batch 5 次 | DSP HVX 向量化 f32 matmul |
| **dspqueue fused (1次通信)** | ARM->DSP 用户态通信，每 batch 1 次 | DSP 上完成全部前向+反向+SGD |

核心优化思路：将每 batch 的 5 次 ARM<->DSP 通信压缩为 1 次，把 bias、relu、softmax、loss、relu_backward、SGD 全部搬到 DSP 侧执行。

## 2. 网络结构

```
Input[batch x 800] -> W1[128x800]^T -> ReLU -> W2[32x128]^T -> Softmax -> 10 classes
```

- **800** = 784 像素 padding 到 32 对齐（HVX 要求）
- **32** = 10 个输出类别 padding 到 32 对齐
- **参数量**: 128x800 + 128 + 32x128 + 32 = **106,272**
- **优化器**: SGD，学习率 0.1
- **损失函数**: Softmax + Cross-Entropy

## 3. 训练模式详解

### CPU (ARM Cortex-A)

纯 f32 标量 matmul，所有操作（前向、反向、SGD 更新）在 ARM 大核上完成。作为性能基准。

### FastRPC + HVX

每次 matmul 通过 FastRPC 发送到 DSP 执行（涉及内核态切换）。DSP 上使用 HVX 1024-bit SIMD 向量化 f32 matmul。每 batch 5 次 FastRPC 调用（前向 2 次 + 反向 3 次）。bias、relu、softmax、SGD 仍在 ARM 上执行。

### dspqueue + HVX (5次通信)

与 FastRPC 相同的计算拆分，但用 dspqueue 替代 FastRPC（用户态通信，无内核切换）。仍然每 batch 5 次通信。

### dspqueue fused (1次通信)

最终优化版。每 batch 只发 1 条 OP_TRAIN_BATCH 消息到 DSP。DSP 在单次调用中完成全部计算：

- 前向：2 次 HVX matmul + bias + relu + softmax
- 损失计算：cross-entropy loss
- 反向：3 次 HVX matmul + relu_backward
- 更新：SGD 权重更新

ARM 只负责打乱数据和组装 batch。权重留在 DSP 缓存中，不需要每 batch 刷新。

## 4. 真机测试结果

**Snapdragon 8 Gen 3，Epoch 3 稳态计时（ms/epoch）：**

| batch | CPU | FastRPC+HVX | dspqueue fused + HVX | fused vs CPU | 测试准确率 |
|-------|-----|-------------|----------------------|-------------|-----------|
| 32 | 7694 | — | **2733** | **2.82x** | 96.9% |
| 128 | 7842 | 3941 | **1826** | **4.29x** | 94.3% |
| 256 | 9384 | — | **1749** | **5.37x** | 92.8% |

**CPU 时间分解（batch=128, epoch 3）：** forward 3742ms, backward 4551ms, update 7ms

**FastRPC 时间分解（batch=128, epoch 3）：** forward 1490ms, backward 2105ms, update 139ms

**DSP 实际计算时间（fused, HAP_perf 计时）：**

| batch | DSP 计算/epoch | 总时间/epoch | ARM+通信开销 |
|-------|---------------|-------------|-------------|
| 32 | 1964ms | 2733ms | 769ms |
| 128 | 1484ms | 1826ms | 342ms |
| 256 | 1498ms | 1749ms | 251ms |

## 5. 优化分析

### 通信开销对比 (每 epoch)

| batch | batches/epoch | 旧方案调用次数 | 新方案调用次数 | 节省 |
|-------|--------------|--------------|--------------|------|
| 32 | 1875 | 9375 | 1875 | 7500 次 |
| 128 | 469 | 2345 | 469 | 1876 次 |
| 256 | 234 | 1170 | 234 | 936 次 |

### 每次通信的开销

- **旧方案**每次需要：3 个 buffer 的 cache flush + dspqueue 系统调用 + 3 个 buffer 的 DEREF 响应
- **新方案**每次只需：1 个 input buffer 的 cache flush + dspqueue 调用 + 1 个 buffer 的 DEREF

### 新方案额外优势

1. **权重缓存命中**：权重 buffer 只在注册时 flush 一次，之后留在 DSP 缓存，省去每 batch 约 400KB 的 cache flush
2. **SGD 在 DSP 本地执行**：避免 ARM 写 rpcmem 的 23x 慢速惩罚
3. **梯度无需 memset**：用非累积 matmul (C = A^T@B) 替代累积 (C += A^T@B)，梯度 buffer 不需要每 batch 清零

### batch=32 为什么从 "DSP 更慢" 变成 "DSP 快 2.68x"

这是 fused 优化最显著的效果。在旧方案中 batch=32 时 DSP 反而比 CPU 慢，因为通信开销吞掉了 HVX 的计算优势：

- **旧方案**：9375 次通信 x ~500us/次 = 4.7s 通信开销（超过 HVX 计算节省）
- **新方案**：1875 次通信 x ~410us/次 = 0.77s 通信开销
- **HVX 计算**：1964ms DSP vs 7694ms ARM，HVX 4x unrolled + 全 HVX 向量化带来 3.9x 加速

三层优化叠加：操作融合（5→1 次通信）+ matmul 4x unrolling + 标量操作 HVX 向量化。

## 6. HVX 向量化策略 (DSP 侧)

DSP 端实现了三个 matmul 内核，核心是 `matmul_nn`，使用 **4x accumulator unrolling + broadcast+accumulate** 模式：

```c
// matmul_nn: C[m×n] = A[m×k] @ B[k×n]  (4x unrolled)
// matmul_nt: transpose B -> scratch, then call matmul_nn
// matmul_tn: transpose A -> scratch, then call matmul_nn
//
// 4x unrolled core: process 128 output columns per iteration
for (row i) {
    for (col j, step 128) {              // 4 HVX vectors = 128 floats
        HVX_Vector acc0..acc3 = vzero();  // 4 independent accumulators
        for (p = 0..k-1) {
            a_splat = Q6_V_vsplat_R(A[i,p]);
            b0..b3 = load 4 contiguous vectors from B[p*n + j];
            acc0 += a_splat * b0;         // independent: pipeline can overlap
            acc1 += a_splat * b1;
            acc2 += a_splat * b2;
            acc3 += a_splat * b3;
        }
        store 4 vectors to C[i*n + j];
    }
    // cleanup: 1x loop for remaining 32-float blocks, then scalar tail
}
```

### 为什么 4x unrolling 有效

单 accumulator 的内循环有数据依赖链：`acc = qadd(acc, qmul(splat, b))`，每次迭代要等前一次的 `acc` 写回才能继续。在 Hexagon v75 上，`vmpy` + `vadd` 的延迟约 8 个周期，但吞吐量为 1-2 个周期。4 个独立累加器让流水线始终有指令可发射，将执行单元利用率从约 25% 提升到接近 100%。

### matmul_nt/matmul_tn 的策略

不再各自维护独立的 HVX 内循环，而是先将需要转置的矩阵写入 scratch buffer（1MB），然后复用优化过的 `matmul_nn`。转置的开销远小于未优化的 HVX 内循环节省。

### 标量操作 HVX 向量化

除 matmul 外，fused 路径中的所有数值操作也使用 HVX 向量化：

| 操作 | HVX 实现 | 注意事项 |
|------|----------|----------|
| bias add | `Q6_Vqf32_vadd_VsfVsf` + 转换 | v75 不支持 `Q6_Vsf_vadd`，需走 qf32 |
| ReLU forward | `Q6_Q_vcmp_gt_VwVw` + `Q6_V_vmux_QVV` | 利用 IEEE f32 正数的整数序一致性 |
| ReLU backward | 同上（mask = pre_relu > 0） | |
| SGD update | `Q6_Vqf32_vmpy` + `Q6_Vqf32_vsub` + 转换 | w -= lr * grad，全 qf32 路径 |

v75 架构的 HVX 不支持直接的 IEEE f32 `vadd.sf`/`vsub.sf`/`vmpy.sf`（这些是 v79+ 指令），所有 f32 向量运算必须经过 qf32（quasi-float）中间格式。

## 7. HMX 实验

### 7.1 初始 HMX 实现

通过 HexKL micro API（`hexkl_micro_hmx_mm_f16`，32x32x32 tile 乘法）使用 HMX。

**HMX vs HVX（fused dspqueue, batch=128, ms/epoch）：**

| 版本 | Batch=32 | Batch=128 | Batch=256 |
|------|----------|-----------|-----------|
| **HVX f32** (baseline) | 2737 | **1834** | 1761 |
| HMX f32 (f32→f16 转换) | — | 6278 | — |
| HMX f16 (全 f16, 无转换) | 20100 | 6308 | 4682 |
| HMX / HVX 比值 | 7.4x 慢 | 3.4x 慢 | 2.7x 慢 |

HMX 在 MNIST 规模的矩阵（最大 128x800）上比 HVX 慢 2.7-7.4x。瓶颈分析：

1. **标量转置**（cache-unfriendly 列访问）：每次 matmul 调用要转置权重/激活矩阵
2. **标量 f16 操作**：bias_add、SGD、softmax 全部用标量 `(_Float16)((float)x + ...)` 循环
3. **libm expf/logf**：DSP 上的 `expf()` 调用非常昂贵（~100 cycles/call）
4. **per-tile API 开销**：每个 32x32 tile 需要 ~6 次 `hexkl_micro` 函数调用

### 7.2 优化后的 HMX f16 训练

针对上述瓶颈逐一优化：

| 优化 | 做法 | 目标 |
|------|------|------|
| **HVX qf16 向量化** | SGD、bias_add、bias_backward 用 `Q6_Vqf16_vmpy_VhfVhf` + `Q6_Vqf16_vadd_Vqf16Vqf16` | 64 个 f16/vector，消除标量循环 |
| **快速 exp 近似** | Schraudolph 算法替代 libm `expf()`/`logf()` | softmax 中的 exp 从 ~100cy 降到 ~5cy |
| **缓存友好转置** | 32×32 块转置替代朴素列访问 | 消除 cache thrashing |
| **预存转置权重** | 在 DSP 上维护 W1^T、W2^T 副本 | 前向传播零转置开销 |

**优化后性能（ms/epoch）：**

| 版本 | Batch=32 | Batch=128 | Batch=256 |
|------|----------|-----------|-----------|
| **HVX f32** (baseline) | 2737 | **1834** | 1761 |
| HMX f16 (优化前) | 20100 | 6308 | 4682 |
| **HMX f16 (优化后)** | **13282** | **4394** | **2944** |
| 优化加速比 | **1.51x** | **1.44x** | **1.59x** |
| vs HVX f32 | 4.9x 慢 | 2.4x 慢 | 1.67x 慢 |

精度一致（test_acc: HVX f32 94.3% vs HMX f16 93.9% at batch=128）。

### 7.3 优化后 DSP 时间分解（batch=128, 1 epoch）

```
Forward matmul:   637ms  (15.9%)  ← HMX 32x32 tile 乘法
Backward matmul: 3021ms  (75.4%)  ← dW1 最大 (128×800 output, 100 个 tile)
Forward other:    177ms  ( 4.4%)  ← bias + relu + softmax (HVX qf16)
Backward other:    76ms  ( 1.9%)  ← dlogits + relu_bwd + bias_bwd (HVX qf16)
SGD + re-transpose: 92ms ( 2.3%)  ← 权重更新 + 重建转置副本
────────────────────────────────────
Total DSP:       4004ms (100%)
```

**关键发现**：非 matmul 操作现在只占 **8.6%**（优化前约 25-30%）。HMX matmul 占 **91.4%**，说明标量操作瓶颈已基本消除。剩余瓶颈是 HMX per-tile API 开销——对于 MNIST 的 128×800 矩阵，100 个 output tile 的 acc_read + ah_to_rm + copy 操作仍然占据大部分时间。

### 7.4 HVX qf16 向量化详解

v75 ISA 支持 qf16（quasi float 16），每个 HVX 向量处理 **64 个 f16 元素**（vs qf32 的 32 个 f32）：

```c
// SGD: w -= lr * grad (64 elements per iteration)
HVX_Vector neg_lr_vec = Q6_Vh_vsplat_R(neg_lr_bits);  // broadcast -lr
HVX_Vector one_vec = Q6_Vh_vsplat_R(0x3C00);           // f16 1.0

HVX_Vector delta_q = Q6_Vqf16_vmpy_VhfVhf(neg_lr_vec, grad);  // (-lr) * grad
HVX_Vector w_q = Q6_Vqf16_vmpy_VhfVhf(w, one_vec);            // w * 1.0 → qf16
HVX_Vector result = Q6_Vqf16_vadd_Vqf16Vqf16(w_q, delta_q);   // w + (-lr*grad)
w_out = Q6_Vhf_equals_Vqf16(result);                            // qf16 → f16
```

注意：qf16 没有直接的 `vadd_VhfVhf`，加法需要先通过 `vmpy(x, 1.0)` 提升到 qf16 格式。

### 7.5 结论

| 矩阵规模 | 推荐 | 原因 |
|----------|------|------|
| **≥512 维度** (LLM) | HMX | O(n³) 计算 >> O(n²/1024) tile 开销 |
| **128-512** (MNIST) | HVX f32/qf16 | HMX per-tile overhead 占比过高 |
| **≤128** | HVX | HMX 无优势 |

对于 MNIST 规模的训练，HMX 的 batch=256 结果（2944ms）接近 HVX f32（1761ms）的 1.67x，但仍不如纯 HVX。HMX 的真正价值在于 LLM 规模的矩阵运算——见第 5 章的 4096×4096 实验数据。

## 8. 构建和运行

```bash
bash build.sh              # 编译 4 个 step 二进制 + DSP skel
bash run_device.sh 3 128   # 依次运行 4 个 step，3 epochs，batch=128
```

build.sh 会构建 4 个独立的 ARM 二进制和 3 个 DSP skel：

| 二进制 | 源文件 | 说明 |
|--------|--------|------|
| `step1_cpu` | `step1_cpu.c` | 静态编译，无 SDK 依赖 |
| `step2_fastrpc` | `step2_fastrpc.c` | 链接 cdsprpc + rpcmem + stub |
| `step2_skel.so` | `step2_dsp.c` | DSP: FastRPC matmul only |
| `step3_dspqueue` | `step3_dspqueue.c` | 链接 cdsprpc + rpcmem + stub |
| `step3_skel.so` | `step3_dsp.c` | DSP: dspqueue OP_MATMUL handler |
| `step4_fused` | `step4_fused.c` | 链接 cdsprpc + rpcmem + stub |
| `step4_skel.so` | `step4_dsp.c` | DSP: fused training (forward+backward+SGD) |

如果没有 Hexagon SDK，只构建 step1_cpu。run_device.sh 依次运行所有已构建的 step。

## 9. 文件结构

```
ch12-mnist-train/
├── src/
│   ├── step1_cpu.c             # Step 1: CPU baseline (静态编译，无 SDK 依赖)
│   ├── step2_fastrpc.c         # Step 2: FastRPC matmul offload (5 次内核态切换/batch)
│   ├── step3_dspqueue.c        # Step 3: dspqueue matmul offload (5 次用户态调用/batch)
│   ├── step4_fused.c           # Step 4: dspqueue fused training (1 次调用/batch)
│   ├── dsp_common.h            # DSP lifecycle: open/close/power config
│   ├── step2_dsp.c             # DSP: FastRPC matmul only
│   ├── step3_dsp.c             # DSP: dspqueue OP_MATMUL handler
│   ├── step4_dsp.c             # DSP: fused training (forward+backward+SGD)
│   ├── mnist_common.h          # 常量、类型、RNG、计时工具
│   ├── mnist_data.h            # MNIST IDX 文件加载
│   ├── mnist_cpu_matmul.h      # CPU f32 matmul 参考实现
│   ├── mnist_network.h         # Forward/backward/SGD/训练循环
│   ├── mnist_train_shared.h    # ARM↔DSP 消息协议
│   ├── mnist_train.idl         # FastRPC 接口定义
│   ├── hvx_matmul.h            # HVX matmul 内核 (4x accumulator unrolling)
│   ├── hvx_ops.h               # HVX 训练操作 (bias, relu, softmax)
│   ├── hmx_matmul.h            # HMX 参考实现 (#ifdef USE_HMX)
│   ├── dspqueue_mgr.h          # dspqueue 共享内存和生命周期管理
│   └── fastrpc_mgr.h           # FastRPC 初始化和调度
├── build.sh                    # 交叉编译 4 个 step + DSP skel
├── run_device.sh               # 部署和依次运行 4 个 step
└── README.md
```

## 10. 关键发现

1. **dspqueue fused 比 CPU 快 4.29x**（batch=128）：ARM 7842ms → DSP 1826ms
2. **batch=256 达到 5.37x 加速**：ARM 9384ms → DSP 1749ms，大 batch 更能发挥 HVX 优势
3. **操作融合是最大优化**：5 次通信压缩为 1 次，batch=32 从 "DSP 更慢" 变为 2.82x 加速
4. **全 HVX 向量化**：matmul 4x accumulator unrolling + bias/relu/SGD 全部 HVX 化，标量操作零残留
5. **权重缓存命中**：fused 模式下权重留在 DSP 缓存，省去每 batch 约 400KB 的 cache flush
6. **SGD 在 DSP 侧执行**：避免 ARM 写 rpcmem 的 23x 慢速惩罚
7. **精度一致**：CPU 和 DSP 训练结果（准确率）基本一致（97.0% vs 96.9% at batch=32）
