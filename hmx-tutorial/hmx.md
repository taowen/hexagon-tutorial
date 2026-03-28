# 用手机 NPU 的矩阵加速器训练神经网络

## 要做什么

我要在骁龙 8 Gen 3 手机的 NPU 上训练一个识别手写数字的神经网络。

具体来说：给网络看 6 万张手写数字图片（0-9），让它自己学会识别。训练完后，用 1 万张它没见过的图片测试，正确率达到 96%。

为什么选这个任务？因为 MNIST 手写数字识别是最经典的机器学习入门问题——数据小、网络简单、几分钟就能训练完，但它包含了训练神经网络的所有核心步骤。在 NPU 上跑通这个，就证明了 NPU 能做训练，不只是推理。

## 什么是"训练"

先说推理：你有一个已经学好的模型，给它一张图片，它告诉你这是数字几。模型里的参数（权重）是固定的。

训练是反过来：你从随机的权重开始，一遍遍给模型看图片，每次看完根据它猜对还是猜错来调整权重。猜错了就往对的方向调一点。看够多的图片后，模型就学会了。

一次训练迭代的计算过程：

```
1. 前向传播：把图片输入网络，算出预测结果
     hidden = input × W1 + b1 → ReLU（把负数变 0，让网络能学非线性关系）
     output = hidden × W2 + b2 → softmax（把 10 个输出值变成加起来等于 1 的概率分布）

2. 算损失：预测结果和正确答案差多远

3. 反向传播：算每个权重对损失的贡献（梯度）
     这需要一系列矩阵乘法和逐元素运算

4. 更新权重：W = W - 学习率 × 梯度
     往减小损失的方向微调
     学习率控制每次调整的步幅——太大会震荡，太小收敛慢
```

整个过程的计算核心是**矩阵乘法**——前向要乘，反向也要乘，占了总计算量的 90% 以上。

训练时不是一张张喂图片，而是一组一起喂，称为一个 batch（比如 128 张）。一组一起算的好处是矩阵乘法可以并行处理多张图片。

而骁龙 8 Gen 3 里恰好有一个专门做矩阵乘法的硬件：**HMX（Hexagon Matrix eXtension）**。

## 为什么在手机 NPU 上做

直接说：没有人真的需要在手机上训练 MNIST。这是一个教学项目。

它的价值在于：MNIST 足够小（整个网络的参数才几百 KB），能完全装进 NPU 的片上内存。这让我能专注于搞清楚 HMX 的工作原理，而不用操心数据搬运的复杂性。搞清楚了原理，将来要做 on-device fine-tuning（比如在手机上根据用户习惯微调模型）就有基础了。

高通的 QNN 框架用 HMX 跑推理，但把所有细节包装在黑盒里。我想打开这个黑盒：HMX 到底怎么编程？数据要什么格式？内存怎么管理？从底层搞明白这些，花了好几个月。下面是完整的过程。

## 我们的网络

一个 2 层 MLP（多层感知机），最简单的神经网络之一：

```
输入层: 28×28 灰度图片 → 展平为 784 维向量（padding 到 832 维（832 = 26 × 32，HMX tile 的 32 行对齐））
隐藏层: 128 个神经元，ReLU 激活
输出层: 10 个神经元（对应数字 0-9），padding 到 64 维（32 的倍数，HMX tile 对齐），softmax 归一化

参数量:
  W1: 832 × 128 = 106,496 个权重
  W2: 128 × 64  = 8,192 个权重（其中只有 10 列有效，其余 padding 为 0）
  加上偏置，总共约 108,000 个有效参数
```

权重的形状直接匹配矩阵乘法：input[batch, 832] x W1[832, 128] = hidden[batch, 128]。

这不是什么深度学习——只有两层。但它包含了训练所需的所有操作：矩阵乘法、偏置加法、ReLU、softmax、交叉熵损失、反向传播、梯度下降。

## 第一步：让代码跑到 DSP 上

HMX 住在 Hexagon DSP 里——手机 SoC 中一颗独立的处理器，有自己的指令集。先写一个最简单的程序，让它在 DSP 上跑起来：

```c
// hello_dsp.c —— 跑在 Hexagon DSP 上的第一个程序
#include "HAP_farf.h"

int main() {
    int a = 3, b = 4;
    FARF(ALWAYS, "DSP says: %d + %d = %d", a, b, a + b);
    return 0;
}
```

用高通 Hexagon SDK 的编译器编译，推到手机，运行：

```bash
# 编译（-mv75 = 骁龙 8 Gen 3 的 DSP 架构）
hexagon-clang -mv75 -O2 hello_dsp.c -o hello_dsp

# 推到手机，用 run_main_on_hexagon 加载到 DSP 执行
adb push hello_dsp /data/local/tmp/
adb shell "cd /data/local/tmp && run_main_on_hexagon 3 hello_dsp"
```

```
DSP says: 3 + 4 = 7
```

代码跑在 DSP 上了。但要用 HVX 和 HMX，还需要申请硬件资源——它们默认是关着的：

```c
// 上电 HVX 和 HMX（省略重复的 memset，每个都是 HAP_power_set 一次调用）
power_on_hvx();   // HAP_power_set(ctx, &req)  req.type = HAP_power_set_HVX
power_on_hmx();   // HAP_power_set(ctx, &req)  req.type = HAP_power_set_HMX

// 申请 VTCM（片上高速内存）和 HMX 访问权
compute_res_attr_t attr;
HAP_compute_res_attr_init(&attr);
HAP_compute_res_attr_set_vtcm_param(&attr, 8*1024*1024, 1);  // 8MB VTCM
HAP_compute_res_attr_set_hmx_param(&attr, 1);
unsigned int ctx_id = HAP_compute_res_acquire(&attr, 100000);
void *vtcm_base = HAP_compute_res_attr_get_vtcm_ptr(&attr);

// 锁定 HMX（必须在 acquire 之后，否则 HMX 指令会崩）
HAP_compute_res_hmx_lock(ctx_id);
```

这里出现了一个关键名词：**VTCM（Vector Tightly Coupled Memory）**。它是 DSP 的片上高速内存，8MB，访问延迟只有 1 个时钟周期（DDR 主存大约 100 周期）。**HMX 的所有输入和输出数据必须在 VTCM 里**——这是硬件限制，不申请就用会直接崩溃。

到这里，开发环境搭好了：代码能跑在 DSP 上，VTCM 和 HMX 硬件就绪。但这只是 DSP 侧。训练还需要 ARM 侧加载数据、发任务给 DSP——这就是下一步要解决的通信问题。

## 第二步：解决 ARM 和 DSP 之间的通信

标准的 ARM-DSP 通信方式是 FastRPC：ARM 调一个函数，内核态序列化参数发给 DSP，DSP 返回结果。每次调用约 **364 微秒**。

训练时一个 batch 包含大约 15 个操作（5 个矩阵乘 + bias add + ReLU + softmax + 反向传播 + SGD 更新）。如果每个都走 FastRPC：

```
15 次 × 364μs = 5460μs 通信开销
实际计算只要 ~3000μs
通信比计算还慢
```

解决方案来自 llama.cpp 的 Hexagon 后端：**dspqueue**——共享内存消息队列。只用一次 FastRPC 启动 DSP 侧的消息循环，之后所有通信走共享内存，延迟降到 **61 微秒**。

更关键的是，我把整个 batch 的计算**融合成一条消息**：ARM 发一条 `OP_TRAIN_BATCH`，DSP 内部跑完前向、反向、SGD 更新后才回复。通信开销从 5460μs 变成 61μs。

```c
// DSP 侧：一条消息完成一个 batch 的完整训练
static void packet_callback(dspqueue_t queue, int error, void *context) {
    while (1) {
        err = dspqueue_read_noblock(queue, &flags, ...);
        if (err == AEE_EWOULDBLOCK) return;
        switch (msg.op) {
            case OP_TRAIN_BATCH:
                // 前向传播 + 反向传播 + SGD 更新，全在 DSP 上完成
                do_train_batch(ctx, labels, batch_size, lr, &loss, &correct);
                break;
        }
        dspqueue_write(queue, 0, ...);  // 回复 ARM
    }
}
```

### ARM 侧的训练循环

ARM 侧的工作很简单：组装 batch 数据，发消息，等结果。所有计算都在 DSP 上完成。

```c
// ARM 侧：每个 epoch 遍历所有 batch
for (int epoch = 0; epoch < epochs; epoch++) {
    shuffle_indices(indices, train_count);   // 打乱训练数据顺序

    for (int bi = 0; bi < n_batches; bi++) {
        // 1. 组装 batch：从训练集取 128 张图片，f32->f16
        for (int b = 0; b < batch_size; b++) {
            int idx = indices[bi * batch_size + b];
            memcpy(batch_f32 + b * 832, train_images + idx * 832, 832 * sizeof(float));
            labels[b] = train_labels[idx];
        }
        arm_f32_to_f16(batch_f16, batch_f32, batch_size * 832);

        // 2. 发一条消息给 DSP，等它完成整个 batch 的训练
        struct train_batch_req req = { .op = OP_TRAIN_BATCH, .batch_size = batch_size,
                                       .learning_rate = lr };
        memcpy(req.labels, labels, batch_size);
        dspqueue_write(queue, 0, 1, &input_buf, sizeof(req), &req, timeout);
        sem_wait(&done);  // 等 DSP 回复

        epoch_loss += last_loss;
        epoch_correct += last_correct;
    }

    // 3. 评估也在 DSP 上做（保持 DSP 忙碌，防止 VTCM 被抢占）
    float test_acc = dspqueue_evaluate(test_images, test_labels, test_count);
    printf("Epoch %d: loss=%.4f train_acc=%.4f test_acc=%.4f\n", ...);
}
```

## 第三步：管理 8MB 片上内存

我们的 MNIST 网络虽小，训练时需要存储的数据不少：

```
训练时需要在 VTCM 中存储：
  权重 W1, W2                  ~224KB
  激活值 hidden, logits, probs  ~96KB (batch=128)
  梯度 dW1, dW2, dhidden       ~240KB
  输入 input                    416KB (batch=256, 832 dims)
  临时缓冲                      416KB
  ─────────────────────────────
  约 1.4MB / 8MB VTCM（还有大量空间）
```

后面会看到，最终优化版本把权重和梯度以 HMX 原生 tile 格式存储，整体布局会更复杂——但初始的行优先版本就是这么简单。

VTCM 没有 `malloc`/`free`。我们用最简单的 bump allocator——一个指针从头往后移：

```c
static _Float16 *bump_alloc_f16(uint8_t **bump, uint32_t bytes) {
    uintptr_t addr = ((uintptr_t)*bump + 127) & ~(uintptr_t)127;
    _Float16 *ptr = (_Float16 *)addr;
    *bump = (uint8_t *)(addr + bytes);
    return ptr;
}

uint8_t *bump = vtcm_base;
ctx->v_b1     = bump_alloc_f16(&bump, 256);          // 偏置
ctx->v_b2     = bump_alloc_f16(&bump, 128);
ctx->v_w2_t   = bump_alloc_f16(&bump, 16*1024);      // W2 RM 副本
ctx->v_hidden = bump_alloc_f16(&bump, 64*1024);       // 隐藏层激活
// ... logits, probs, hidden_pre, dhidden ...
ctx->v_input  = bump_alloc_f16(&bump, 416*1024);      // batch 输入
ctx->v_scratch = bump_alloc_f16(&bump, 416*1024);     // 转置临时缓冲
```

llama.cpp 用的也是这个模式。简单、O(1) 分配、零碎片。

### VTCM 数据会"消失"

训练到一半遇到了一个诡异的 bug：前几个 epoch（训练完整个数据集一遍叫一个 epoch）准确率 96%，epoch 之间让 DSP 空闲几秒（ARM 侧做评估），准确率突然掉到 9.8%——和随机猜一样。

根因：**VTCM 不是独占的**。DSP 空闲时，系统可以把 VTCM 分配给 camera、audio 等高优先级客户端，我们存在里面的权重就被覆盖了。

解决方案：不让 DSP 空闲。评估也在 DSP 上做，保持 DSP 持续处理消息。

## 建立基线：先用 HVX 做矩阵乘法

在上 HMX 之前，先用 DSP 的向量单元 HVX 实现训练，建立性能基线。

HVX 是 1024 位 SIMD——每次处理 32 个 f32 或 64 个 f16。先用 f32 存储数据建立训练基线，但 v75 的 HVX **没有**直接的 f32 运算指令——`Q6_Vsf_vadd_VsfVsf` 之类是 v79+ 才有的，v75 上必须用 qf32（准浮点）做乘加。qf32 是 Hexagon 特有的浮点格式，和 f32 有相同的尾数位数但指数位更少，用范围换速度。在 v75 上所有 HVX 浮点运算都经过 qf32：先算出 Vqf32 结果，累加完再转回 Vsf/Vhf。

后来改用 f16 存储数据，计算用 **widening multiply**——两个 f16 向量相乘，得到 qf32 精度的中间结果，累加完再转回 f16：

```c
// HVX f16 矩阵乘法：widening multiply 获得 f32 精度的中间累加
HVX_Vector a_splat = Q6_Vh_vsplat_R(a_bits);        // 广播 A 的一个 f16 元素
HVX_Vector b_f16 = *(HVX_Vector *)(B + p * n);       // 加载 B 的一行 f16
HVX_VectorPair prod = Q6_Wqf32_vmpy_VhfVhf(a_splat, b_f16);  // f16×f16 → qf32 对
acc_lo = Q6_Vqf32_vadd_Vqf32Vqf32(acc_lo, Q6_V_lo_W(prod));  // 累加低 32 元素
acc_hi = Q6_Vqf32_vadd_Vqf32Vqf32(acc_hi, Q6_V_hi_W(prod));  // 累加高 32 元素
// 最终: Q6_Vhf_equals_Wqf32(combine(acc_hi, acc_lo)) 转回 f16
```

`Q6_Wqf32_vmpy_VhfVhf` 接收两个 64 元素的 f16 向量，输出一个 qf32 VectorPair（lo 和 hi 各 32 个 qf32 值）。这样从 f16 输入获得了 f32 精度的点积，最终再用 `Q6_Vhf_equals_Wqf32` 转回 f16。

结果：

| 版本 | 每 epoch 时间 | 测试精度 |
|------|-------------|---------|
| ARM CPU | 8750ms | 96.03% |
| **HVX f32 存储 + qf32 运算 (DSP)** | **2150ms** | 96.02% |

**4.1x 加速**，精度一致。纯靠向量化就快了 4 倍。但矩阵乘法仍然是瓶颈——HVX 一次只能做一行的点积，而 HMX 一次处理一整个 32×32 的矩阵块。

### 除了 matmul，其他算子也全用 HVX

训练不只是矩阵乘法。每个 batch 还有 bias 加法、ReLU、softmax、梯度计算、SGD 更新——这些如果用标量代码，DSP 会极慢（Hexagon 标量 f32 比 ARM 大核慢 8 倍）。全部用 HVX 向量化：

```c
// ReLU：把负数变 0。HVX 没有 f16 比较指令，但 f16 的符号位和整数一样
// 所以直接用整数比较 + vmux 选择
void hvx_relu_forward_f16(_Float16 *out, const _Float16 *in, uint32_t n) {
    HVX_Vector zero = Q6_V_vzero();
    for (uint32_t i = 0; i < n; i += 64) {
        HVX_Vector v = *(HVX_Vector *)(in + i);
        HVX_VectorPred pos = Q6_Q_vcmp_gt_VhVh(v, zero);  // 哪些 > 0？
        *(HVX_Vector *)(out + i) = Q6_V_vmux_QVV(pos, v, zero);  // 正的保留，负的变 0
    }
}

// SGD 更新：w -= lr * grad。用 qf16 乘法（v75 没有直接 f16 乘法）
void hvx_sgd_update_f16(_Float16 *w, const _Float16 *grad, float lr, uint32_t n) {
    HVX_Vector v_lr = Q6_Vh_vsplat_R(f32_to_f16_bits(-lr));  // 广播 -lr
    for (uint32_t i = 0; i < n; i += 64) {
        HVX_Vector vw = *(HVX_Vector *)(w + i);
        HVX_Vector vg = *(HVX_Vector *)(grad + i);
        HVX_Vector update = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vg, v_lr));
        *(HVX_Vector *)(w + i) = Q6_Vhf_equals_Vqf16(
            Q6_Vqf16_vadd_Vqf16Vqf16(Q6_Vqf16_vmpy_VhfVhf(vw, v_one), update));
    }
}
```

完整的 HVX 算子清单：

| 算子 | 作用 | 关键 HVX 操作 |
|------|------|-------------|
| `hvx_add_bias_f16` | 每行加偏置向量 | qf16 广播 + 逐元素加法 |
| `hvx_relu_forward_f16` | 前向 ReLU | 整数比较 + vmux |
| `hvx_relu_backward_f16` | 反向 ReLU 梯度 | 同上，mask 乘梯度 |
| `hvx_softmax_cross_entropy_f16` | softmax + 交叉熵损失 | 多项式 exp2 近似 + 反复平方 |
| `hvx_compute_dlogits_f16` | 交叉熵梯度 | (probs - one_hot) / batch |
| `hvx_sgd_update_f16` | SGD 权重更新 | qf16 乘加 |
| `hvx_bias_backward_f16` | 偏置梯度（列求和） | qf16 累加 |
| `blocked_transpose_f16_vtcm` | 矩阵转置 | 按 32×32 块 HVX 搬运 |

其中 softmax 最复杂（200+ 行）：HVX 没有 exp 指令，用多项式近似 exp2，再通过反复平方实现任意指数。其他算子都很短（10-20 行），因为 HVX 的 64 路 f16 并行天然适合这些逐元素操作。

## 搞清楚 HMX 的矩阵乘法

HMX 做矩阵乘法的核心只有三条 ASM 指令——但它们处理的是**一个 32×32 的 output tile**：

```c
// 三个 HMX 操作对应三个 inline 函数

static inline void hmx_clear_acc(void) {
    asm volatile("mxclracc.hf" ::: "memory");   // 清零 32x32 累加器
}

static inline void hmx_load_tiles(
    const void *act_tiles, const void *wt_tiles, uint32_t n_tiles)
{
    uint32_t limit = n_tiles * 2048 - 1;         // 字节数 - 1
    asm volatile(
        "{ activation.hf = mxmem(%0, %1):deep\n" // 加载 n_tiles 个激活 tile
        "  weight.hf = mxmem(%2, %3) }\n"         // 同时加载对应的权重 tile
        :: "r"(act_tiles), "r"(limit),
           "r"(wt_tiles), "r"(limit) : "memory");
}

static inline void hmx_store_acc_tile(_Float16 *out) {
    asm volatile(
        "cvt.hf = acc(%0)\n"                      // 累加器 -> f16（参数 2 = f16 格式）
        "mxmem(%1, %2) = cvt\n"                   // 写到 VTCM
        :: "r"(2), "r"(out), "r"(0) : "memory");
}
```

`:deep` 修饰符让 HMX 连续加载多个 tile 对并逐个累加到累加器中。不加 `:deep` 时只加载一对 tile。上限是 32 个 tile 对（= 32 x 2048 = 64KB，刚好是 VTCM 的一个 bank）。

一个完整的矩阵乘 C[m,n] = A[m,k] @ B[k,n] 需要遍历所有 output tile。以 [128,128] @ K=832 为例：128/32 = 4 行 x 4 列 = **16 个 output tile**，每个 tile 要清零累加器、加载 K/32 = 26 批 tile 对、写出结果：

```c
for (rt = 0; rt < Mr; rt++) {          // Mr=4 行
    for (ct = 0; ct < Nc; ct++) {      // Nc=4 列
        hmx_clear_acc();               // mxclracc.hf
        for (kt = 0; kt < K; kt += 32) {
            batch = min(K - kt, 32);   // :deep 最多批量加载 32 个 tile 对
            hmx_load_tiles(act + kt*2048, wt + kt*2048, batch);
        }
        hmx_store_acc_tile(out);       // cvt.hf + mxmem store
    }
}
```

指令虽少，但实际用起来难点不在计算本身，而在**数据准备**。

### HMX 要求特殊的数据格式

HMX 不接受普通的行优先（row-major）矩阵。它要求数据按 **tile 格式**排列——32×32 的 f16 矩阵块内部，行是两两交织的：

```
行优先存储:                   tile 格式 (2-row interleave):
row0: a0, a1, ..., a31       vec[0]:  a0, b0, a1, b1, ..., a31, b31
row1: b0, b1, ..., b31       vec[1]:  c0, d0, c1, d1, ..., c31, d31
row2: c0, c1, ..., c31       ...
row3: d0, d1, ..., d31       vec[15]: y0, z0, y1, z1, ..., y31, z31
```

相邻两行的元素交替排列。一个 32×32 tile 有 16 个这样的向量，共 2048 字节。

为什么要这样排列？HMX 硬件做矩阵乘法时，需要同时读取两行来做外积（outer product）并累加到结果矩阵。2 行交织让硬件可以用一次内存读取（一个 128 字节的 HVX 向量）同时获得两行的对应元素，减少访问次数。

高通的文档没有说明这个格式。我们是通过实验——写入已知数据、用 HMX 计算、逐字节检查输出——才搞清楚的。还发现了一个重要事实：**在 v75 上，f16 的 activation 格式（AH）和 weight 格式（WH）是一样的**，都是 2 行交织。这和 Genie 引擎代码里的 `fromFlatOffset` 函数不同——那个是 4 行交织，只适用于 int8。

### 用 HVX 做格式转换

知道了格式，用 HVX 的 `vshuff` 指令可以高效完成转换：

```c
// 行优先 → AH tile 格式：把 m×k 矩阵切成 32×32 tile 并交织
// 把 m*k 的行优先矩阵转为 ceil(m/32) * ceil(k/32) 个 tile
static void convert_rm_to_ah(struct hmx_workspace *ws,
                              uint32_t ah_offset,
                              const _Float16 *src, uint32_t m, uint32_t k)
{
    uint32_t Mr = (m + 31) / 32;   // 行方向 tile 数
    uint32_t K  = (k + 31) / 32;   // 列方向 tile 数

    for (uint32_t rt = 0; rt < Mr; rt++) {
        for (uint32_t kt = 0; kt < K; kt++) {
            uint32_t tile_idx = rt * K + kt;       // 行优先 tile 索引
            HVX_Vector *out = (HVX_Vector *)(ws->vtcm_base + ah_offset
                                              + tile_idx * 2048);
            uint32_t row0 = rt * 32;
            uint32_t col0 = kt * 32;
            uint32_t cols = (col0 + 32 <= k) ? 32 : k - col0;

            for (uint32_t rr = 0; rr < 16; rr++) {   // 16 对行 = 32 行
                uint32_t r_even = row0 + rr * 2;
                uint32_t r_odd  = r_even + 1;
                HVX_Vector v_r0 = Q6_V_vzero();       // 越界行用零填充
                HVX_Vector v_r1 = Q6_V_vzero();

                if (r_even < m)
                    memcpy(&v_r0, src + r_even * k + col0, cols * sizeof(_Float16));
                if (r_odd < m)
                    memcpy(&v_r1, src + r_odd * k + col0, cols * sizeof(_Float16));

                HVX_VectorPair vp = Q6_W_vshuff_VVR(v_r1, v_r0, -2);
                out[rr] = Q6_V_lo_W(vp);              // 交织后写入 tile
            }
        }
    }
}
```

WH 格式的转换函数 `convert_rm_to_wh` 结构完全相同，唯一区别是 tile 索引用**列优先**：`tile_idx = ct * Mr + rt`（HMX 的 `mxmem:deep` 要求同一列的 tile 在内存中连续）。

反向的解交织用 `vdeal` 指令——`vshuff(v1, v0, -2)` 以 2 字节粒度交织两行，`vdeal(zero, v, -2)` 做反操作解交织：

```c
// tile 格式 -> 行优先：用 vdeal 解交织
HVX_VectorPair dealt = Q6_W_vdeal_VVR(Q6_V_vzero(), tile[rr], -2);
HVX_Vector v_even = Q6_V_lo_W(dealt);   // 偶数行
HVX_Vector v_odd  = Q6_V_hi_W(dealt);   // 奇数行
```

### 从累加器读回结果

HMX 计算完后，结果在累加器里。`hmx_store_acc_tile` 一条指令就能写到 VTCM，但写出的是 tile 格式（2 行交织）。要变回行优先，用 HVX `vdeal` 解交织：

```c
// 累加器 → VTCM tile → 行优先 f16
static void readback_acc_to_rm(_Float16 *dst, uint32_t m, uint32_t n,
                                uint32_t rt, uint32_t ct, _Float16 *staging_tile)
{
    hmx_store_acc_tile(staging_tile);               // 累加器直接写 VTCM（1 条指令）

    uint32_t row0 = rt * 32, col0 = ct * 32;
    uint32_t cols = (col0 + 32 <= n) ? 32 : n - col0;
    const HVX_Vector *tv = (const HVX_Vector *)staging_tile;

    for (uint32_t rr = 0; rr < 16; rr++) {          // 16 个向量 = 32 行
        HVX_VectorPair dealt = Q6_W_vdeal_VVR(Q6_V_vzero(), tv[rr], -2);
        uint32_t r_even = row0 + rr * 2;
        if (r_even < m)
            memcpy(dst + r_even * n + col0, &Q6_V_lo_W(dealt), cols * 2);
        if (r_even + 1 < m)
            memcpy(dst + (r_even + 1) * n + col0, &Q6_V_hi_W(dealt), cols * 2);
    }
}
```

全程 f16，不做精度转换，不碰 DDR。`hmx_store_acc_tile` + `vdeal` 两步完成。

### HMX 的初始化

使用 HMX 之前，需要通过 `mxmem2.bias` 指令设置 scale/bias 寄存器（scale=1.0, bias=0.0）。这个配置告诉 HMX 累加器写出时不做额外的缩放和偏移：

```c
// 获取 HMX 硬件访问权后，配置一次 scale/bias 寄存器
HAP_compute_res_attr_set_hmx_param(&attr, 1);  // 申请 HMX
// ... acquire ...

// 设置 scale=1.0, bias=0.0（通过 mxmem2.bias 指令）
// 之后所有 mxclracc + mxmem:deep + cvt.hf 调用都能正确工作
```

不需要其他初始化。配置一次，整个训练过程都有效。

## 把 HMX 用到训练中

现在有了所有积木：dspqueue 通信、VTCM 内存管理、HMX tile 格式转换、直接 ASM 计算。把它们组合起来实现训练。

每个 matmul 的流程：

```
1. AH prep  — HVX vshuff 把激活矩阵的行对交织成 tile 格式
2. WH prep  — HVX vshuff 把权重矩阵的行对交织成 tile 格式
3. Compute  — ASM mxclracc + mxmem:deep 批量加载 tile 对
4. Readback — ASM hmx_store_acc + HVX vdeal 解交织回行优先
```

训练一个 batch 需要 5 个 matmul：前向 2 个（hidden 和 logits），反向 3 个（dW2、dH、dW1）。加上 bias add、ReLU、softmax、SGD 更新，全在 DSP 的一条消息里完成。

### 完整的训练循环

下面是真实的训练函数（简化了计时和变量声明）。5 个 HMX matmul 加粗标注了矩阵维度：

```c
static void do_train_batch(struct mnist_context *ctx,
                           const uint8_t *labels, uint32_t bs, float lr,
                           float *out_loss, uint32_t *out_correct)
{
    /* === 前向传播 === */
    // hidden[bs,128] = input[bs,832] x W1[832,128]    <- HMX matmul，W1 已在 WH tiles 中
    hmx_matmul_nn_f16_cached_wh(ws, hidden, input, wh_w1_off, bs, 128, 832);
    hvx_add_bias_f16(hidden, b1, bs, 128);              // + b1
    hvx_relu_forward_f16(hidden, hidden, bs * 128);     // ReLU

    // logits[bs,64] = hidden[bs,128] x W2[128,64]     <- HMX matmul
    hmx_matmul_nn_f16_cached_wh(ws, logits, hidden, wh_w2_off, bs, 64, 128);
    hvx_add_bias_f16(logits, b2, bs, 64);               // + b2
    hvx_softmax_cross_entropy_f16(logits, probs, labels, bs, &loss, &correct);

    /* === 反向传播 === */
    // dlogits = (probs - one_hot(labels)) / bs         <- 交叉熵梯度
    hvx_cross_entropy_backward(dlogits, probs, labels, bs);

    // dW2[128,64] = hidden^T[128,bs] x dlogits[bs,64] <- 梯度直接输出到 WH tiles
    blocked_transpose_f16_vtcm(scratch, hidden, bs, 128);
    hmx_matmul_nn_f16_to_wh(ws, wh_dw2_off, scratch, dlogits, 128, 64, bs);

    // dhidden[bs,128] = dlogits[bs,64] x W2^T[64,128] <- 用缓存的 W2 转置
    hmx_matmul_nn_f16_cached_wh(ws, dhidden, dlogits, wh_w2t_off, bs, 128, 64);
    hvx_relu_backward_f16(dhidden, dhidden, hidden_pre, bs * 128);

    // dW1[832,128] = input^T[832,bs] x dhidden[bs,128] <- 最大的 matmul
    blocked_transpose_f16_vtcm(scratch, input, bs, 832);
    hmx_matmul_nn_f16_to_wh(ws, wh_dw1_off, scratch, dhidden, 832, 128, bs);

    /* === SGD 更新（直接在 WH tile 格式上）=== */
    hvx_sgd_update_f16(w1_wh, dw1_wh, lr, 104 * 1024);   // W1: 104 tiles
    hvx_sgd_update_f16(w2_wh, dw2_wh, lr, 8 * 1024);     // W2: 8 tiles
    hvx_sgd_update_f16(b1, db1, lr, 128);                  // 偏置
    hvx_sgd_update_f16(b2, db2, lr, 64);

    // 同步 W2 的转置（反向传播 dH 需要）
    convert_wh_to_rm(ws, w2_rm, wh_w2_off, 128, 64);
    blocked_transpose_f16_vtcm(scratch, w2_rm, 128, 64);
    convert_rm_to_wh(ws, wh_w2t_off, scratch, 64, 128);
}
```

注意两种 matmul 函数的区别：`hmx_matmul_nn_f16_cached_wh` 权重已在 WH tiles 中，直接计算，结果输出为行优先格式；`hmx_matmul_nn_f16_to_wh` 把结果直接写入 WH tiles（用于梯度，SGD 直接读取）。

### 转置的问题

HMX 只支持 C = A × B 这一种矩阵乘法。但反向传播需要转置，比如 dW1 = input^T x dhidden：

```
dW2 = hidden^T × dlogits      ← A 要转置
dH  = dlogits × W2^T          ← 需要 W2 的转置
dW1 = input^T × dhidden       ← A 要转置
```

解决方案很朴素：先在 VTCM 上做显式转置，再调 HMX matmul。转置用 HVX 实现——对每个 32x32 的块，用 vdeal/vshuff 做行列互换；对于非方阵（如 128x256），按 32x32 块遍历并转置每块到目标位置。每个 batch 需要 3 次转置（后来优化到 2 次），这是当前实现的主要开销之一。

## 消除格式转换的开销

到这里 HMX 训练已经跑通了，但性能只比 HVX 基线快一点。原因是**格式转换吃掉了 HMX 的计算收益**：

```
每 batch 的格式转换开销：
  W1 → WH tiles: **104 个 tile** 的 vshuff（W1 是 [832, 128]，832/32 = 26 行 tile x 128/32 = 4 列 tile = 104 tiles，最大的单项开销）
  W2 → WH tiles: **8 个 tile** 的 vshuff（128/32 x 64/32 = 4 x 2 = 8 tiles）
  dW1 readback:    104 个 tile 的 vdeal + memcpy
  dW2 readback:    8 个 tile 的 vdeal + memcpy
  加上激活矩阵的 AH 转换...
```

权重每个 batch 都变（SGD 更新），所以每次都得重新转换。

### 让权重永远留在 tile 格式里

灵感来自 Genie 引擎的 **NativeKV** 技术。在 LLM 推理中，KV cache 永久存储为 HMX 原生 tile 格式，新 token 只转换一次后追加，避免每步重复转换整个 cache。

同样的思路用到训练：**让权重和梯度都永久存储为 tile 格式。**

```
之前的做法（每 batch）:
  1. 权重从行优先 → 转换成 tile 格式 → HMX 计算
  2. 梯度从 HMX → 转换回行优先
  3. SGD: 在行优先格式上更新权重
  4. 下一 batch: 又要把权重转成 tile 格式  ← 浪费！

NativeKV 做法:
  1. 权重本来就是 tile 格式 → 直接 HMX 计算
  2. 梯度直接留在 tile 格式
  3. SGD: 在 tile 格式上更新权重  ← 关键！
  4. 下一 batch: 权重已经是 tile 格式 → 零转换
```

这能行是因为两个关键事实：

**第一，SGD 是 element-wise 操作。** `W[i] -= lr × dW[i]`，逐元素更新。它不关心数据的排列方式——不管是行优先还是 tile 格式，只要 W 和 dW 按同样的方式排列，逐元素减就是对的。

**第二，v75 f16 的 AH = WH。** HMX 的输出（`hmx_store_acc` 写出的 AH 格式）和输入要求的 WH 格式是同一种 tile 格式。所以梯度的输出可以直接作为权重 SGD 的输入，不需要任何格式转换。

```c
// 梯度直接输出到 WH tile 位置
uint32_t wh_tile_idx = ct * Mr + rt;  // 列优先索引——HMX 的 mxmem:deep 要求同一列的 tile 在内存中连续
_Float16 *out = (_Float16 *)(vtcm + out_wh_off + wh_tile_idx * 2048);
hmx_store_acc_tile(out);

// SGD 在 tile 格式上做——和行优先格式用完全一样的函数
hvx_sgd_update_f16(w1t_wh_ptr, dw1_wh_ptr, lr, 104 * 1024);
```

VTCM 里 WH 区域的布局：

```
W1 永久存储     104 tiles  208KB   ← 前向直接用
W2 永久存储       8 tiles   16KB   ← 前向直接用
W2 的转置        8 tiles   16KB   ← 反向 dH 用
dW1 输出        104 tiles  208KB   ← 反向写入，SGD 读取
dW2 输出          8 tiles   16KB
临时区           32 tiles   64KB   ← 反向时给 dlogits/dhidden 用
```

唯一的额外开销：必须同时存储 W2 和 W2 的转置两份权重。原因是前向传播用 W2（计算 hidden x W2 = logits），反向传播算 dH 用 W2 的转置（计算 dlogits x W2^T = dhidden）。SGD 更新 W2 后，需要同步 W2 的转置——把 W2 从 tile 转回行优先，做一次矩阵转置，再转回 tile。但 W2 只有 8 个 tile，开销很小。

### NativeKV 后的完整 VTCM 布局

经过 NativeKV 优化，权重和梯度以 WH tile 格式永久存储在 HMX 工作区里，行优先（RM）区域只保留必要的缓冲区：

```
行优先数据区（bump allocator 分配）:
  b1, b2 偏置                    ~0.4KB
  W2 RM 副本（反向同步用）         16KB
  hidden, logits, probs 等        ~320KB（MAX_BATCH=256）
  input（每 batch 输入）           416KB
  scratch（转置临时缓冲）          416KB
  ─────────────────────────────
  数据区小计                     ~1.1MB

HMX 工作区（2048 字节对齐）:
  AH activation tiles（208 tiles） 416KB
  WH weight/gradient tiles:
    W1 永久存储    104 tiles       208KB
    W2 永久存储      8 tiles        16KB
    W2 永久存储      8 tiles        16KB
    dW1 输出       104 tiles       208KB
    dW2 输出         8 tiles        16KB
    临时区          32 tiles        64KB
  staging + config                 ~8KB
  ─────────────────────────────
  HMX 区小计                     ~0.95MB

总计 ~2.1MB / 8MB VTCM
```

## 最终结果

| 版本 | 每 epoch 时间 | DSP 总时间(5 epoch) | 测试精度 | vs CPU |
|------|-------------|-------------------|---------|--------|
| ARM CPU 基线 | 8750ms | — | 96.03% | 1x |
| HVX f32 + qf32 向量化 | 2150ms | 8857ms | 96.02% | 4.1x |
| **HMX + NativeKV** | **1960ms** | **7154ms** | 96.09% | **4.5x** |

从 ARM CPU 到 HMX 训练，**4.5 倍加速**，精度一致。

加速来源的分解：
- HVX 向量化 vs CPU：4.1x（1024 位 SIMD 吞吐）
- HMX vs HVX：额外 1.1x（矩阵加速器 + VTCM 常驻 + f16 半精度）
- NativeKV：额外 3%（消除权重格式转换）

HMX 相对 HVX 的加速只有 1.1x，不大。原因是 MNIST 的矩阵太小（最大 832x128），tile 格式转换的开销相对于计算量很显著。对于 LLM 级别的大矩阵（4096x4096），HMX 的优势才会真正体现——转换开销被计算吞吐覆盖。

回头看，MNIST 的矩阵规模太小了，HMX 的优势发挥不出来。但这个项目的价值不在最终加速比，而在搞清楚了 HMX 的完整工作流程——tile 格式、ASM 指令、初始化、格式转换优化。这些知识在大模型推理和 on-device fine-tuning 场景下才是真正有用的。

## 总结：学会用 HMX 需要搞清楚什么

HMX 的文档几乎为零。高通没有公开任何 HMX 编程手册。我能搞清楚 tile 格式和 ASM 指令，主要靠 [htp-ops-lib](https://github.com/haozixu/htp-ops-lib) 这个开源项目——感谢作者把 HMX 的底层操作用 C 封装了出来，没有这个项目，学习难度会大很多。

要真正用好 HMX，需要自己搞清楚这几件事：

**硬件接口只有三条指令。** `mxclracc.hf` 清零累加器，`mxmem:deep` 批量加载 tile 对并累乘累加，`cvt.hf = acc(2); mxmem = cvt` 写出结果。整个 HMX 编程就围绕这三条指令展开——剩下的工作全是数据准备。

**tile 格式是最大的障碍。** HMX 不接受行优先矩阵，必须按 32x32 tile 切块，每块内部 2 行交织。这个格式没有文档，只能通过写入已知数据、计算、逐字节比对输出来逆向。搞清楚格式后，用 HVX 的 `vshuff(-2)` 和 `vdeal(-2)` 就能高效完成正反转换。

**初始化需要 `mxmem2.bias`。** HMX 的 scale/bias 寄存器必须配置一次（scale=1.0, bias=0.0）。不配置的话单次 matmul 可能碰巧正确，但持续计算会出错。

**数据常驻 tile 格式能消除转换开销。** 权重和梯度永久存储为 tile 格式，SGD 是逐元素操作不关心数据排列，AH 和 WH 在 v75 f16 上是同一种格式——三个条件凑齐，就能实现零格式转换的训练循环。

**小矩阵上 HMX 优势不大。** MNIST 最大的矩阵是 832x128，HMX 相对 HVX 只快 1.1x。格式转换的固定开销相对于计算量太大。HMX 真正的战场是 LLM 级别的大矩阵——4096x4096 时，转换开销被计算吞吐覆盖。