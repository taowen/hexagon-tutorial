# 第二章：格式墙——NPU 只吃一种菜

上一章我们看到了一个矛盾：34 TOPS 的算力，实际利用率不到 0.1%。从这一章开始，我们自底向上拆解原因。第一道墙，从硬件本身说起。

---

## 一、HMX：一台高速印刷机

很多人第一次听说 NPU，会以为它像 GPU 一样有上千个通用小核心，只是数量更多、功耗更低。这个理解是错的。

高通 NPU 的核心计算单元叫 **HMX**（Hexagon Matrix eXtension）。它不是一堆小核心，而是一块专用的矩阵乘法电路——一个 32x32 的乘累加（MAC）阵列。流水线填满后，每个周期可以发射一次 32x32 的瓦片（tile）乘法。

原理上，HMX 和 GPU 的 Tensor Core 是同一类东西——脉动阵列（systolic array），专做矩阵乘累加。区别在于规模和功耗预算：GPU 有上百个 SM，每个 SM 里嵌着多个小的 Tensor Core（4x4 到 16x16），靠 300W 的功耗喂饱它们；HMX 是一个大的 32x32 阵列，靠 1-5W 的功耗在手机上跑。但不管阵列大小，**它们都只接受特定规格的数据排列**——放错了就算出垃圾。

这个"特定规格"，就是本章要讲的格式墙。

顺便提一句 Hexagon 的另一个特点：它是 VLIW（超长指令字）架构，一条"指令包"里可以同时塞 HMX 矩阵操作、HVX 向量操作和标量操作。编译器把多个操作打包进一个 cycle 执行。这意味着 HMX 做矩阵乘的同时，HVX 可以在旁边做数据预处理——后面会讲到这个流水线。

---

## 二、WH Layout：印刷机要求的"纸张规格"

### 从直觉开始

假设你有一个 4x4 的矩阵，在普通内存里是行优先排列：

```
Row-major（你在 PyTorch 里看到的样子）：
[ a00  a01  a02  a03 ]
[ a10  a11  a12  a13 ]
[ a20  a21  a22  a23 ]
[ a30  a31  a32  a33 ]

内存中连续存放：a00, a01, a02, a03, a10, a11, ...
```

这对 CPU 很友好——按行遍历时 cache line 命中率高。但 HMX 不吃这个。HMX 的 MAC 阵列有固定的数据读取模式：它需要在一个时钟周期内同时喂入 activation 的一行和 weight 的一列的数据。如果数据不在正确的物理位置，硬件读到的就是错误的值。

所以权重矩阵必须重排为 **WH Layout**（Weight HMX Layout），激活矩阵重排为 **AH Layout**（Activation HMX Layout）。

### 真正的 WH Layout 公式

对于一个 `[din, dout]` 的权重矩阵，WH Layout 的地址变换是：

```
flat [din, dout]
  → [dout/N_TILE, din/32, (dout%N_TILE)/32, [(din%32)/4, dout%32, din%4]]
```

最内层是一个 1024 字节的 block（对 int8 而言），结构是 `[8组][32列][4行]`，也就是 `din_1 * 128 + dout_1 * 4 + din_2`。

为什么是 `[8:din][32:dout][4:din]` 这种奇怪的交错？因为 HMX 的硬件数据通路一次加载 128 字节——恰好是 32 个连续的 dout 元素，对齐 HMX 的 32-wide SIMD 宽度。而 din 维度被拆成 8 组、每组 4 行，是为了匹配 MAC 阵列的流水线深度。这不是软件工程师拍脑袋想出来的格式，是硬件电路的物理布线决定的。

来看一个真实的地址计算。Key Cache 的参数：`embed_dim=128, ctx_size=512, K_TILE=256`，要找 `din=65, dout=300` 的物理偏移：

```
tile_idx = 300 / 256 = 1        → 第 2 个 tile
dout_0   = (300 % 256) / 32 = 1 → tile 内第 2 个 32-col 块
dout_1   = 300 % 32 = 12        → 块内第 12 列
din_0    = 65 / 32 = 2          → 第 3 个 32-row 块
din_1    = (65 % 32) / 4 = 0    → 第 0 组
din_2    = 65 % 4 = 1           → 组内第 1 行

offset = 1×32768 + 2×8192 + (1×1024 + 0×128 + 12×4 + 1) = 50225
```

真机验证输出：`offset=50225`。分毫不差。你放错一个字节，HMX 算出来的就是垃圾。

---

## 三、格式转换有多贵？

看起来只是一次数据重排，能有多贵？

我们在骁龙 8 Gen 3 真机上测了：一个 `[64x256]` 的矩阵（LLM 中 attention 的 K 矩阵，head_dim=64，上下文 256 tokens），从 row-major 转成 WH Layout，需要 **2299 个硬件时钟周期**。对应 16 个 32x32 的 tile（2 行 x 8 列）。

这个数字看起来不大。现在算 LLM 的真实场景。

一个典型的模型有 32 个 attention head。每生成一个 token，每个 head 都要做一次 `Q x K^T` 矩阵乘法——而 K 是不断增长的 KV Cache。如果 K 以普通行优先格式存储（Genie 引擎中称为 SmartMask 方案，第五章会详细对比），每次 attention 都要把整个 K 从 row-major 转成 WH Layout。

关键在于：K 的大小和上下文长度成正比。

| ctx_size | rm_to_wh tiles | 转换开销占 attention 总时间 |
|----------|---------------|------------------------|
| 256 | 2x8 = 16 | ~2% |
| 4096 | 2x128 = 256 | ~30% |
| 32768 | 2x1024 = 2048 | **~60%** |

当上下文到 32K tokens 时，32 个 head 每个都要转换一个 `[64x32768]` 的矩阵。格式转换吃掉了 attention 总时间的 60%。你的 HMX 印刷机有一多半时间在等人把纸裁成正确的尺寸，而不是在印东西。

这就是为什么 Genie 引擎在 v75+ 架构上引入了 NativeKV——让 K 从一开始就以 WH 格式存储，彻底跳过转换。但这是第五章的故事了。

---

## 四、HVX：印刷机旁边的万能工人

NPU 上不只有 HMX。还有一个向量单元叫 **HVX**（Hexagon Vector eXtension），每个周期处理 128 字节——也就是 64 个 f16 或 128 个 int8。

LLM 推理里不是所有操作都是矩阵乘法。反量化（int8 转 f16）、softmax、加偏置、RMSNorm、RoPE 位置编码、SiLU 激活函数——这些全部由 HVX 来做。

在骁龙 8 Gen 3 的真机实验中，我们让 HVX 和 HMX 在**同一次 DSP 调用内**接力工作，数据全程留在 VTCM（片上 SRAM），不回 DDR：

```
Step 1 (HVX):  int8 权重反量化 → f16
               加载 64 个 int8 值，符号扩展为 int16，转为 f16，乘以 scale
               用到 Q6_Wh_vunpack_Vb, Q6_Vhf_equals_Vh, Q6_Vqf16_vmpy_VhfVhf

Step 2 (HMX):  [32x64] x [64x128] 矩阵乘法
               f16 权重直接从 VTCM 喂给 HMX，零拷贝

Step 3 (HVX):  f32 偏置加法
               HMX 输出 f32，用 HVX 向量加法叠加 bias
```

真机验证：4096 个元素全部通过，最大误差 0.025215。

这就是 llama.cpp 在 Hexagon 后端的核心模式。一个 transformer 层里 HVX 和 HMX 交替出现十几次——RMSNorm(HVX) → QKV投影(HMX) → RoPE(HVX) → Attention(HMX+HVX) → FFN(HMX) → SiLU(HVX)。如果每次切换都要一次 FastRPC 往返（364us，详见第四章），32 层模型光通信就要几十毫秒。把整个 transformer 层打包进一次 DSP 调用，HVX 和 HMX 之间的切换就是零成本的函数调用。

---

## 五、quasi-float：一个正在被淘汰的硬件妥协

当你第一次在 Hexagon v75 上跑浮点运算，可能会遇到两个让人困惑的现象。

第一个：HMX 做 f16 矩阵乘法，act=1.0, wt=1.0 的 32x32 matmul 连续累加两次（共 64 次乘累加），结果不是 64.0，而是 **65.0**。这不是 bug，这是 HMX F16 累加的精度限制——硬件特性，大约 1-3% 偏差。

第二个：HVX 的 **quasi-float**（准浮点）格式。

在 v75 及更早的架构上，HVX 的浮点加法和乘法不直接输出标准 IEEE 754 结果，而是输出一种叫 qf16 / qf32 的中间格式，需要额外一条指令转回标准浮点：

```c
// v75 及更早：f32 加法需要两步
HVX_Vector v_qf32 = Q6_Vqf32_vadd_VsfVsf(v_out, v_bias);  // 结果是 qf32
HVX_Vector v_sum  = Q6_Vsf_equals_Vqf32(v_qf32);           // 转回标准 f32

// v75 及更早：f16 乘法同理
HVX_Vector v_prod = Q6_Vhf_equals_Vqf16(
    Q6_Vqf16_vmpy_VhfVhf(v_f16, v_scale));
```

为什么要这样？用一条额外的转换指令，换取更简单的 FPU 流水线——更小的芯片面积、更低的功耗、更高的频率。在手机这种功耗敏感的场景下，这曾经是合理的权衡。

但**从 v79（骁龙 8 Elite）开始，HVX 加入了标准 IEEE 754 浮点指令**，不再需要 quasi-float 中转。llama.cpp 的 Hexagon 后端里能看到这个演进：

```c
#if __HVX_ARCH__ < 79
    // v75: 必须走 quasi-float
    HVX_Vector qf = Q6_Vqf32_vadd_VsfVsf(a, b);
    HVX_Vector result = Q6_Vsf_equals_Vqf32(qf);
#else
    // v79+: 直接用标准浮点指令
    HVX_Vector result = Q6_Vsf_vadd_VsfVsf(a, b);
#endif
```

这说明 quasi-float 不是什么精妙的硬件设计哲学，而是早期面积/功耗约束下的工程妥协，随着工艺进步正在被淘汰。如果你在 v75 上开发，需要注意这个额外的转换步骤；如果你的目标平台是 v79+，可以直接忽略它。

---

## 下一章预告

格式墙告诉我们：数据必须严格按照硬件要求排列，否则再强的算力也用不上。但即使格式对了，还有一个更根本的问题——HMX 只能从 8MB 的 VTCM 片上内存读数据，而一个 7B 模型的权重有 14GB。差了 1750 倍。

第三章，我们讲内存墙：如何把 14GB 的数据，一口一口喂进 8MB 的胃里。
