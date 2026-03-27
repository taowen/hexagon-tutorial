# 第8章：HVX f32 训练 MNIST

## 1. 概述

ch08 是 Hexagon HVX f32 训练基线。在骁龙 8 Gen 3 上用 2 层 MLP 训练 MNIST 手写数字识别，提供两种模式：

| 模式 | 说明 |
|------|------|
| **CPU baseline** | 纯 ARM 大核 f32 标量训练，支持 `--test-synthetic` 合成数字验证 |
| **HVX fused dspqueue** | 每 batch 1 次 ARM->DSP 通信，DSP 上完成全部前向+反向+SGD |

ch09 在此基础上加入 VTCM 和 f16 优化。

## 2. 网络结构

```
Input[batch x 832] -> W1[128x832]^T + b1 -> ReLU -> W2[64x128]^T + b2 -> Softmax -> 10 classes
```

| 维度 | 原始 | Padded | 对齐原因 |
|------|------|--------|----------|
| 输入 | 784 (28x28 像素) | 832 (INPUT_DIM_PAD) | 832 = 13x64，HVX 128B 对齐 |
| 隐藏层 | 128 | 128 (HIDDEN_DIM) | 已对齐 |
| 输出 | 10 (数字类别) | 64 (OUTPUT_DIM_PAD) | 64x4 = 256B = 2 个 HVX vector |

- **参数量**: 128x832 + 128 + 64x128 + 64 = **114,688**
- **优化器**: SGD，学习率 0.1
- **损失函数**: Softmax + Cross-Entropy

## 3. 性能对比

**Snapdragon 8 Gen 3，batch=128，5 epochs：**

| 模式 | Epoch 时间 | DSP 总时间 (5ep) | 测试准确率 | 加速比 |
|------|-----------|-----------------|-----------|--------|
| CPU baseline | 8750ms | -- | 96.03% | 1.0x |
| HVX fused dspqueue | 2130ms | 8859ms | 96.02% | **4.1x** |

精度一致：两种模式的测试准确率均达 ~96%。

## 4. HVX 向量化策略 (DSP 侧)

DSP 端实现了三个 matmul 内核，核心是 `matmul_nn`，使用 **4x accumulator unrolling + broadcast+accumulate** 模式：

```c
// matmul_nn: C[m x n] = A[m x k] @ B[k x n]  (4x unrolled)
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

不再各自维护独立的 HVX 内循环，而是先将需要转置的矩阵写入 scratch buffer，然后复用优化过的 `matmul_nn`。转置的开销远小于未优化的 HVX 内循环节省。

### 标量操作 HVX 向量化

除 matmul 外，fused 路径中的所有数值操作也使用 HVX 向量化：

| 操作 | HVX 实现 | 注意事项 |
|------|----------|----------|
| bias add | `Q6_Vqf32_vadd_VsfVsf` + 转换 | v75 不支持 `Q6_Vsf_vadd`，需走 qf32 |
| ReLU forward | `Q6_Q_vcmp_gt_VwVw` + `Q6_V_vmux_QVV` | 利用 IEEE f32 正数的整数序一致性 |
| ReLU backward | 同上（mask = pre_relu > 0） | |
| SGD update | `Q6_Vqf32_vmpy` + `Q6_Vqf32_vsub` + 转换 | w -= lr * grad，全 qf32 路径 |

v75 架构的 HVX 不支持直接的 IEEE f32 `vadd.sf`/`vsub.sf`/`vmpy.sf`（这些是 v79+ 指令），所有 f32 向量运算必须经过 qf32（quasi-float）中间格式。

## 5. dspqueue fused 设计

ARM 每 batch 只发 1 条 `OP_TRAIN_BATCH` 消息，DSP 在单次调用中完成全部计算：

```
ARM                                DSP
 |                                  |
 |-- OP_REGISTER_NET (12 buffers) ->|  注册 w1,b1,w2,b2,dw1,dw2,hidden,logits...
 |                                  |
 |  for each epoch:                 |
 |    shuffle data                  |
 |    for each batch:               |
 |      fill input buffer           |
 |-- OP_TRAIN_BATCH (1 buffer) ---->|  forward + backward + SGD
 |<---- loss, correct count --------|
 |                                  |
 |-- OP_SYNC (4 buffers) ---------->|  flush w1,b1,w2,b2 回 ARM
```

关键优势：

1. **权重缓存命中**：权重 buffer 只在注册时 flush 一次，之后留在 DSP L2 缓存，不需要每 batch 重新加载
2. **SGD 在 DSP 本地执行**：避免 ARM 写 rpcmem 的慢速惩罚（ARM 写共享内存比写普通 malloc 慢约 23x）
3. **梯度无需 memset**：用非累积 matmul (C = A^T@B) 替代累积 (C += A^T@B)，梯度 buffer 不需要每 batch 清零

## 6. Synthetic digit 验证

`train_cpu --test-synthetic` 在训练完成后生成 10 个合成手写数字（0-9），用训练好的网络进行分类，验证网络确实学会了数字识别而不只是过拟合数据集：

```
$ ./train_cpu 5 128 --test-synthetic

Digit    Predicted    Result
------   ---------    ------
  0        0          CORRECT
  1        1          CORRECT
  ...
  6        8          WRONG       <- 合成 6 与 8 太像
  7        3          WRONG       <- 合成 7 笔画偏斜
  ...

Summary: 8 / 10 correct (80%)
```

8/10 正确率说明网络学会了数字的结构特征。错误的 2 个（6 和 7）是因为合成图案的笔画风格与 MNIST 训练集差异较大。

## 7. 构建和运行

```bash
bash build.sh              # 编译 train_cpu + train_fused + skel_fused.so
bash run_device.sh 5 128   # 推送到设备，运行 5 epochs，batch=128
```

build.sh 产出：

| 产物 | 说明 |
|------|------|
| `train_cpu` | 静态编译，无 SDK 依赖 |
| `train_fused` | 链接 cdsprpc + rpcmem + stub |
| `skel_fused.so` | DSP: fused forward+backward+SGD |

如果没有 Hexagon SDK，只构建 `train_cpu`。

```bash
# CPU baseline
./train_cpu 5 128

# CPU + 合成数字测试
./train_cpu 5 128 --test-synthetic

# HVX fused dspqueue
LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./train_fused 5 128
```

## 8. 文件结构

```
ch08-mnist-train/
├── src/
│   ├── arm/
│   │   ├── train_cpu.c          # CPU baseline + --test-synthetic
│   │   ├── train_fused.c        # dspqueue fused HVX 训练
│   │   ├── cpu_matmul.h         # CPU f32 matmul 参考实现
│   │   ├── network.h            # 网络初始化、训练循环、评估
│   │   ├── data.h               # MNIST IDX 数据加载
│   │   ├── dspqueue_mgr.h       # dspqueue + rpcmem 生命周期管理
│   │   └── synthetic_test.h     # 合成数字生成 + 推理测试
│   ├── dsp/
│   │   ├── skel_fused.c         # DSP: fused forward+backward+SGD 处理
│   │   ├── hvx_matmul.h         # HVX matmul (4x accumulator unrolling)
│   │   ├── hvx_ops.h            # HVX 训练操作 (bias, relu, softmax, SGD)
│   │   └── skel_common.h        # DSP 生命周期辅助函数
│   └── common/
│       ├── protocol.h           # OP_REGISTER_NET, OP_TRAIN_BATCH, OP_SYNC
│       ├── common.h             # 常量、类型、RNG、计时
│       └── mnist_train.idl      # FastRPC 接口 (dspqueue stub 依赖)
├── build.sh
├── run_device.sh
└── README.md
```

## 9. 关键发现

1. **HVX fused 比 CPU 快 4.1x**（batch=128）：8750ms -> 2130ms/epoch
2. **全 HVX 向量化是关键**：Hexagon DSP 标量 f32 比 ARM 大核慢约 8x，必须用 HVX SIMD
3. **4x accumulator unrolling**：隐藏流水线延迟，执行单元利用率从 ~25% 提升到接近 100%
4. **操作融合 (5->1 次通信)**：权重留在 DSP 缓存 + SGD 本地执行，消除通信和共享内存写入瓶颈
5. **qf32 是 v75 的必经之路**：所有 f32 向量加法/乘法必须走 qf32 中间格式（v79+ 才有直接 f32 指令）
6. **精度一致**：CPU 和 DSP 训练结果（~96% 准确率）基本一致
