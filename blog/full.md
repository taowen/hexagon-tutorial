# 骁龙 NPU 编程：34 TOPS 去哪了？

## 一个矛盾

骁龙 8 Gen 3 的 NPU（HTP Gen 3）INT8 算力为 34 TOPS。这个数字听起来很唬人，但它到底意味着什么？我们先做一个简单的实验。

取一个 256x1024x4096 的矩阵乘法，分别在 CPU 和 NPU 上跑。CPU 用了 84ms，NPU 只用了 0.5ms。**166 倍加速**。如果你只看这个数字，你会觉得 NPU 是一台性能怪兽，端侧 AI 的未来一片光明。

但换一个场景。同一颗芯片跑 LLM 推理，实际吐字速度大约 30 tokens/s。我们来算一笔账：一个 0.6B 参数的模型，每生成一个 token 大约需要 1.2 GFLOP 的计算量。30 tokens/s 就是 36 GFLOP/s 的实际吞吐。而 34 TOPS 意味着 34,000 GFLOP/s 的理论峰值。

**实际利用率不到 0.1%。**

166 倍加速和 0.1% 利用率，来自同一颗芯片。这不是高通的问题，苹果的 ANE、联发科的 APU、甚至 Google 的 TPU 在端侧场景下都面临类似的困境。矛盾的根源藏在芯片的物理结构里。

NPU 不是一个"更快的 CPU"。把它想象成一台高速印刷机。一台海德堡胶印机每小时能印 18000 张，速度惊人。但前提是：纸张必须裁切成精确的尺寸，油墨必须调配好颜色并注入指定墨斗，印版必须提前制好并安装到位，纸张传送带必须和印刷滚筒精确同步。如果你只想印一张名片，光是上版、调墨、校准的时间就够你手写一百张了。

NPU 就是这台印刷机。它的"印刷速度"——也就是矩阵乘法的吞吐——确实是 34 TOPS。但"上版调墨"的时间——数据格式转换、片上内存搬运、CPU 与 NPU 之间的通信——这些才是真正决定端到端性能的因素。矩阵乘法本身往往只占总耗时的一小部分，剩下的时间全花在了"喂料"上。

这篇系列文章的目的，就是把这个"喂料"过程彻底讲清楚。

---

## 先建立全景：你的模型从 PyTorch 到 NPU 要经过什么？

很多教程把模型部署画成一条线：`PyTorch -> ONNX -> QNN -> NPU`。三个箭头，好像"转换一下"就完事了。这给人一种危险的错觉，好像部署和训练一样，跑一行命令就能搞定。

真实的技术栈有七层。大多数性能问题都出在中间那几层——恰恰是教程们一笔带过的地方：

```
你在这里 --> PyTorch / TensorFlow.js      "我有一个模型"
              |
              |  torch.onnx.export()   <-- 一行代码，大多数教程到这里结束
              v
            ONNX / TFLite               "通用的计算图描述"
              |
              |  qnn-converter          <-- 黑箱开始
              v
            QNN 推理引擎                 "把计算图变成执行计划"
              |                          (算子融合、量化、内存调度)
              v
            HexKL / QHPI               "硬件抽象：帮我调一个高性能矩阵乘"
              |
              v
            FastRPC / dspqueue          "CPU 和 NPU 之间传数据"
              |
              v
            DMA + VTCM 管理             "把数据搬到正确的片上地址"
              |
本文要打开的 -> HMX 矩阵指令 / HVX 向量指令  "一条指令算完 32x32 矩阵乘法"
```

让我逐层解释这张图。

**PyTorch / TensorFlow** 是你训练模型的地方。这一层你很熟悉，不用多说。但要注意，训练框架的设计目标是灵活性和可调试性，不是硬件效率。你在 PyTorch 里写的 `torch.matmul` 和最终 NPU 上执行的矩阵乘法，除了数学含义相同之外，几乎没有任何共同之处。

**ONNX / TFLite** 是中间表示层。它把模型从 Python 代码变成一张静态计算图——每个节点是一个算子（MatMul、Softmax、LayerNorm），每条边是一个 tensor。这一层的意义在于"去 Python 化"：去掉控制流、去掉动态分配、把所有形状信息固定下来。这是硬件优化的前提条件。

**QNN 推理引擎** 是高通的核心软件栈。它接收 ONNX 计算图，输出一份完整的"执行计划"：哪些算子可以融合成一个（比如 MatMul + BiasAdd + ReLU 三合一）、每个 tensor 用什么量化方案（int8、int16、还是混合精度）、8MB 的片上内存怎么分配给各层。这是一个 NP-hard 级别的调度问题，QNN 用启发式算法来近似求解。`Finalize` 阶段动辄跑几十秒，就是在解这个优化问题。

**HexKL / QHPI** 是硬件抽象层。上层说"帮我算一个 [M, K] x [K, N] 的矩阵乘法"，这一层负责把它翻译成具体的硬件指令序列：数据怎么切成 32x32 的瓦片、瓦片之间怎么累加、中间结果存在 VTCM 的哪个地址。你可以把它理解为 NPU 版的 cuBLAS。

**FastRPC / dspqueue** 是通信层。NPU（高通叫它 DSP / HTP）是一颗独立的处理器，有自己的指令集、自己的操作系统（QuRT）。CPU 想让它干活，不能直接调函数——要通过进程间通信。FastRPC 是高通提供的专有 RPC 机制，走内核态，每次调用约 364 微秒。dspqueue 是高通提供的替代方案，用共享内存队列绕过内核，开销降到 61 微秒。选择哪个，直接影响你每秒能向 NPU 发多少条指令。

**DMA + VTCM 管理** 是内存搬运层。VTCM 是 NPU 的片上 SRAM，只有 8MB，但这是唯一能直接喂给矩阵计算单元的内存。所有权重、激活值都必须先通过 DMA（Direct Memory Access）引擎从主存（DDR）搬到 VTCM，才能参与计算。这一层决定了"搬什么、什么时候搬、搬到哪里"，是性能优化的核心战场。

**HMX / HVX** 是最底层的计算指令。HMX（Hexagon Matrix eXtension）是矩阵乘法专用硬件，一个周期完成 32x32 的矩阵乘法。HVX（Hexagon Vector eXtension）是 128 字节宽的向量处理单元，负责所有非矩阵运算：softmax、LayerNorm、反量化、加偏置。两者协作完成一个完整的 transformer 层。

**关键洞察**：从 PyTorch 到 ONNX 只是一步；从 ONNX 到真正在 NPU 上高效执行——那是五步。每一步都在解决同一个核心问题的不同侧面：**怎么把数据以正确的格式、在正确的时间、搬到正确的地方。**

---

## 打破迷信：HMX 不是黑箱，QNN 也不是唯一的路

看完这张七层图，很多人的第一反应是：我老老实实走 QNN 那条路就行了，下面那几层是高通内部的事，普通开发者碰不到。

**这是一个流传甚广的误解。**

事实上，高通 NPU 的编程开放程度远超大多数人的想象。从最底层的 HMX 矩阵指令到最上层的 QNN 推理引擎，每一层都有公开的 API，而且可以自由组合。条条大路通罗马：

**路线一：纯手写 HVX + HMX 内核。** llama.cpp 的 Hexagon 后端就是这么做的。19000 行 C 代码，不用 QNN，不用 HexKL，不用任何高通 ML 框架。直接用 HVX 内联函数做反量化和激活函数，用 HMX 内联汇编 `mxmem` 指令做矩阵乘法，用 dspqueue 和 CPU 通信。一切自己控制。这是最"硬核"的路，但也是自由度最高的路。文末会详细拆解这条路线。

**路线二：用 HexKL 做矩阵乘，其余自己写。** HexKL 提供了一个 Micro API，你调一行 `hexkl_micro_matmul()`，它帮你处理 32x32 瓦片切分和 VTCM 排布。而 HVX 部分——反量化、softmax、加偏置——你仍然用自己的内联函数来写。实际上完全可以在一次 DSP 调用里组合两者：HVX 做 int8 到 f16 的反量化，HMX 做矩阵乘，HVX 再做偏置加法，三步流水线一气呵成。

**路线三：在 QNN 上挂自定义算子。** QNN 的 QHPI 接口允许你注册自己的 C 函数作为算子实现。标准的卷积、矩阵乘让 QNN 的优化器去排课表；你自己发明的特殊算子（比如自定义的注意力变体），用 QHPI 写一个 HVX/HMX 内核注册进去就行。这条路兼顾了自动调度的便利和手写内核的灵活。

**路线四：纯 QNN 全自动。** 这是多数教程默认推荐的路线。`torch.onnx.export()` 导出计算图，`qnn-converter` 转成 QNN 格式，`Finalize` 生成执行计划，推理时一行 `execute()` 搞定。你不需要知道 VTCM 怎么分配、DMA 怎么调度、HMX 怎么吃数据——QNN 全包了。代价是你也无法控制这些，遇到性能问题只能调参数碰运气。

四条路线从下往上，自由度递减，易用性递增。但关键是：**它们不是互斥的。** 你可以在 QNN 图里混用自己的 HVX 算子（路线三 + 四），也可以在 HexKL 矩阵乘前后拼接 HVX 流水线（路线二 + 一）。这不是"选 A 还是选 B"的问题，而是"哪一层需要你亲自下场"的问题。

而且你甚至不需要一台骁龙手机就能开始。高通 Hexagon SDK 自带一个 **x86 模拟器**，在你的 Linux 开发机上就能跑 HVX 和 HMX 指令——矩阵乘法的结果和真机一致，只是没有真实的时序数据。想测 QNN 全流程？SDK 里还有一个 **QNN CPU Simulator**（`libQnnHtpNetRunExtensions.so`），用 `libnative` 在 x86 上模拟整个 HTP 后端，`Finalize`、`execute` 全部可以跑通。调试完再推到手机上量性能，开发效率比盲目刷机高一个量级。

在真机上开发测试的门槛也比想象中低。骁龙 8 Gen 3 的 CDSP 支持 **Unsigned PD（无签名进程域）**——你编译出的 `.so` 不需要高通的代码签名，开发阶段用 `run_main_on_hexagon` 就能直接加载到 DSP 上运行和调试。HVX、HMX、VTCM、DMA 全部可用。正式部署时，应用通过 FastRPC 或 dspqueue 加载 DSP 侧的 `.so`（比如 llama.cpp 就是这么做的），流程和普通 Android NDK 开发类似。这意味着任何拿到 Hexagon SDK 的开发者，都能在自己的手机上跑自己写的 NPU 内核，不需要找高通申请任何权限。

说了这么多，不如看一个真正能跑的 Hello World。下面这段 C 代码在骁龙 8 Gen 3 的 DSP 上完成一次 32x32 的 HMX 矩阵乘法——从申请 VTCM、锁定 HMX、填充数据、执行乘法到读回结果，总共不到 30 行核心代码：

```c
// 1. 申请 VTCM（片上 8MB SRAM）和 HMX 硬件锁
compute_res_attr_t attr;
HAP_compute_res_attr_init(&attr);
HAP_compute_res_attr_set_vtcm_param(&attr, vtcm_size, 1);  // 要 VTCM
HAP_compute_res_attr_set_hmx_param(&attr, 1);               // 要 HMX
unsigned int ctx_id = HAP_compute_res_acquire(&attr, 100000);
void *vtcm = HAP_compute_res_attr_get_vtcm_ptr(&attr);
HAP_compute_res_hmx_lock(ctx_id);

// 2. 在 VTCM 里划出三块：激活矩阵、权重矩阵、输出矩阵
unsigned short *act = (unsigned short *)(vtcm + 0x0000);  // 32x32 f16
unsigned short *wt  = (unsigned short *)(vtcm + 0x1000);  // 32x32 f16
unsigned short *out = (unsigned short *)(vtcm + 0x2000);  // 32x32 f16

// 3. 用 HVX 向量指令填充数据（act 全 1.0，wt 全 2.0）
hvx_fill_f16(act, 0x3C00/*f16 的 1.0*/, 1024);
hvx_fill_f16(wt,  0x4000/*f16 的 2.0*/, 1024);

// 4. 四条 HMX 指令 = 一次 32x32 矩阵乘法
asm("bias = mxmem2(%0)"  :: "r"(scales));           // 设置 scale
asm("mxclracc.hf");                                  // 清空累加器
asm("{ activation.hf = mxmem(%0, %1)\n"             // 喂入 act 和 wt，
    "  weight.hf     = mxmem(%2, %3) }"             //   硬件自动做 32x32 乘加
    :: "r"(act), "r"(2047), "r"(wt), "r"(2047));
asm("mxmem(%0, %1):after.hf = acc" :: "r"(out), "r"(0));  // 结果写回 VTCM

// 5. 读结果：每个元素 = 1.0 * 2.0 * 32（内积长度）≈ 64.0
printf("result = %.1f\n", f16_to_f32(out[0]));       // 输出: 64.0
```

真机上的运行输出：

```
[Init] VTCM=0xd8400000  HMX locked
  act: 0x3C00 0x3C00 0x3C00 0x3C00
  wt : 0x4000 0x4000 0x4000 0x4000
  out: 0x5408 0x5408 0x5408 0x5408
  result=65.0  expected~=64.0
[PASS] Test 1
```

注意结果是 65.0 而不是精确的 64.0——这是 HMX F16 累加的硬件精度特性（约 1-3% 偏差），不是 bug。在 x86 模拟器上跑同样的代码，结果完全一致。

这就是 HMX 编程的全部核心模式：**申请 VTCM → 摆好数据 → 四条指令 → 读回结果。** 所有后续的复杂性——DMA 流水线、WH Layout、多 tile 累加——都是在这个基础骨架上叠加的优化。

高通 NPU 的真正门槛不在 API 的开放度——API 全是公开的，工具链也是免费的。门槛在于理解硬件：32x32 瓦片格式、8MB VTCM 约束、DMA 流水线时序。**一旦你理解了硬件在想什么，用哪条路线都能喂饱它。**

下面我们从最底层看起，自底向上，用真机实验数据带你看清每一层为什么存在、每一道墙到底有多厚。

---

## 格式墙——NPU 只吃一种菜

### HMX：一台高速印刷机

很多人第一次听说 NPU，会以为它像 GPU 一样有上千个通用小核心，只是数量更多、功耗更低。这个理解是错的。

高通 NPU 的核心计算单元叫 **HMX**（Hexagon Matrix eXtension）。它不是一堆小核心，而是一块专用的矩阵乘法电路——一个 32x32 的乘累加（MAC）阵列。流水线填满后，每个周期可以发射一次 32x32 的瓦片（tile）乘法。

原理上，HMX 和 GPU 的 Tensor Core 是同一类东西——脉动阵列（systolic array），专做矩阵乘累加。区别在于规模和功耗预算：GPU 有上百个 SM，每个 SM 里嵌着多个小的 Tensor Core（4x4 到 16x16），靠 300W 的功耗喂饱它们；HMX 是一个大的 32x32 阵列，靠 1-5W 的功耗在手机上跑。但不管阵列大小，**它们都只接受特定规格的数据排列**——放错了就算出垃圾。

这个"特定规格"，就是格式墙的由来。

顺便提一句 Hexagon 的另一个特点：它是 VLIW（超长指令字）架构，一条"指令包"里可以同时塞 HMX 矩阵操作、HVX 向量操作和标量操作。编译器把多个操作打包进一个 cycle 执行。这意味着 HMX 做矩阵乘的同时，HVX 可以在旁边做数据预处理——后面会讲到这个流水线。

---

### WH Layout：印刷机要求的"纸张规格"

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

### 格式转换有多贵？

看起来只是一次数据重排，能有多贵？

我们在骁龙 8 Gen 3 真机上测了：一个 `[64x256]` 的矩阵（LLM 中 attention 的 K 矩阵，head_dim=64，上下文 256 tokens），从 row-major 转成 WH Layout，需要 **2299 个硬件时钟周期**。对应 16 个 32x32 的 tile（2 行 x 8 列）。

这个数字看起来不大。现在算 LLM 的真实场景。

一个典型的模型有 32 个 attention head。每生成一个 token，每个 head 都要做一次 `Q x K^T` 矩阵乘法——而 K 是不断增长的 KV Cache。如果 K 以普通行优先格式存储（Genie 引擎中称为 SmartMask 方案，后文会详细对比），每次 attention 都要把整个 K 从 row-major 转成 WH Layout。

关键在于：K 的大小和上下文长度成正比。

| ctx_size | rm_to_wh tiles | 转换开销占 attention 总时间 |
|----------|---------------|------------------------|
| 256 | 2x8 = 16 | ~2% |
| 4096 | 2x128 = 256 | ~30% |
| 32768 | 2x1024 = 2048 | **~60%** |

当上下文到 32K tokens 时，32 个 head 每个都要转换一个 `[64x32768]` 的矩阵。格式转换吃掉了 attention 总时间的 60%。你的 HMX 印刷机有一多半时间在等人把纸裁成正确的尺寸，而不是在印东西。

这就是为什么 Genie 引擎在 v75+ 架构上引入了 NativeKV——让 K 从一开始就以 WH 格式存储，彻底跳过转换。后文会详细展开。

---

### HVX：印刷机旁边的万能工人

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

这就是 llama.cpp 在 Hexagon 后端的核心模式。一个 transformer 层里 HVX 和 HMX 交替出现十几次——RMSNorm(HVX) → QKV投影(HMX) → RoPE(HVX) → Attention(HMX+HVX) → FFN(HMX) → SiLU(HVX)。如果每次切换都要一次 FastRPC 往返（364us），32 层模型光通信就要几十毫秒。把整个 transformer 层打包进一次 DSP 调用，HVX 和 HMX 之间的切换就是零成本的函数调用。

---

### quasi-float：一个正在被淘汰的硬件妥协

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

## 内存墙——8MB 要装 14GB

即使数据格式完全正确，还有一个更基本的物理事实：**数据根本不在 NPU 能够触及的地方。**

### 1750 倍的鸿沟

一个 7B 参数的 LLM，FP16 格式下权重约 14GB。NPU 的片上高速存储 VTCM（Vector Tightly Coupled Memory）只有 8MB。

14GB 对 8MB，差了 **1750 倍**。

这意味着任何一次推理，模型权重都不可能一次性放进 VTCM。所有 NPU 推理都被迫遵循同一个循环：

```
while 还有层没算完:
    DMA 把一块权重从 DDR 搬到 VTCM
    HMX 在 VTCM 上做矩阵乘法
    DMA 把下一块权重搬进来
```

这里的 DMA（Direct Memory Access）不是一条软件指令，而是一个**独立的硬件引擎**。程序向 DMA 引擎提交一个描述符——源地址、目标地址、传输长度——然后 DMA 引擎自己去搬数据，CPU/HVX/HMX 不需要参与。这个"独立"的特性，是后面一切优化的基础。

Hexagon DSP 上的 UDMA 引擎通过三个 intrinsic 控制：`Q6_dmstart_A` 启动传输，`Q6_R_dmwait` 阻塞等待完成，`Q6_R_dmpoll` 非阻塞查询状态。数据从 DDR 到 VTCM 的搬运绕过 L2 cache，直接走物理总线——所以启动 DMA 前必须手动 flush cache，否则 DMA 读到的是过期数据。

---

### 双缓冲：17 倍加速的来源

最朴素的做法是串行执行：搬完一块，算完一块，再搬下一块。DMA 工作时 HMX 空闲，HMX 工作时 DMA 空闲。两个硬件单元互相等待，利用率各只有一半。

既然 DMA 是独立硬件，一个自然的想法是：**让搬运和计算重叠**。当 HMX 正在计算第 N 块数据时，DMA 同时把第 N+1 块数据搬进来。这就是双缓冲（ping-pong）流水线。

具体实现需要在 VTCM 中开辟两块 scratch buffer（llama.cpp 中叫 `vtcm_scratch0` 和 `vtcm_scratch1`）：

```
第 0 轮: DMA → scratch0,  HMX 空闲
第 1 轮: DMA → scratch1,  HMX 算 scratch0 的数据
第 2 轮: DMA → scratch0,  HMX 算 scratch1 的数据
...交替进行...
```

我们在骁龙 8 Gen 3 上实测了 64KB 数据块、200 次迭代的流水线 benchmark：

| 策略 | 耗时 | 相对耗时 |
|------|------|---------|
| 串行（memcpy + HVX 计算） | 3701 us | 17x（慢） |
| 流水线（DMA + HVX 重叠） | 217 us | **1x（基准）** |

**17 倍加速。** 这个数字不来自更快的芯片或更强的算力，纯粹来自调度。

为什么是 17 倍？串行模式下每个 tile 的耗时 = DMA 时间 + 计算时间。流水线模式下每个 tile 的耗时 = max(DMA 时间, 计算时间)。当 DMA 时间远小于计算时间时，搬运延迟几乎被完全隐藏。实测中 memcpy 64KB 约需 17us，HVX 计算约需 18us，而 DMA 异步传输可以和计算完全重叠，相当于每轮只花计算时间。200 轮累计下来，串行模式花了 200 x (17+18) = 7000us 量级，流水线模式花了约 200 x 1 = 200us 量级，差距就是这么来的。

---

### 一个反直觉的发现：VTCM 并不比 DDR 快

你可能以为 VTCM 的价值在于"访问速度更快"。我们做了一个对照实验：同样的 HVX 向量加法（256KB 数据，500 次迭代），分别在 DDR 和 VTCM 上跑：

| 存储位置 | HVX vadd 耗时 | 加速比 |
|---------|--------------|--------|
| DDR | 2118 us | - |
| VTCM | 2115 us | 1.00x |

**完全一样。** 这不是测量误差。HVX 的向量加法是纯顺序访问，L2 cache 的硬件预取机制对顺序读写极其高效，能维持接近峰值带宽。VTCM 标称 1 周期延迟，L2 cache 约 10 周期，但在流式访问模式下，预取完全掩盖了延迟差异。

那 VTCM 到底有什么用？答案是两点，都和"速度"无关：

**第一，HMX 只能从 VTCM 读写。** 这是硬件连线决定的。HMX 矩阵单元的数据通路只连接到 VTCM，不连接 L2 cache，更不连接 DDR。没有 VTCM，HMX 根本无法工作。这就是 llama.cpp 使用 VTCM 的根本原因——不是为了让 HVX 更快，而是因为 HMX 矩阵乘别无选择。

**第二，DMA 可以异步写入 VTCM。** VTCM 作为 DMA 的目标地址，使得上一节的双缓冲流水线成为可能。DMA 不能写入 L2 cache，只能写入 VTCM 或 DDR。

所以 VTCM 不是"快内存"，而是"HMX 和 DMA 都能寻址的内存"。它是矩阵计算和异步搬运的交汇点。

---

### Bump Allocator：最简单的内存管理

8MB 的 VTCM 要在权重、激活值、输出、scratch buffer 之间分配。llama.cpp 用了可能是世界上最简单的分配器——bump allocator：

```c
static inline uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    return p;
}
```

每个矩阵乘法 op 开始时，把指针重置到 VTCM 起始地址，然后依次划出 weight 区域、activation 区域、output 区域、两块 scratch buffer。O(1) 分配，不需要 free，零碎片。

为什么这么简单的方案就够用？因为 LLM 推理是一个**流式工作负载**：每一层的计算模式完全相同，VTCM 的使用模式每层都一样——分配、计算、重置、下一层再来。没有长生命周期的对象，没有交错的分配释放，自然就没有碎片问题。bump allocator 是流式场景下的最优解。

---

### QNN Finalize：排一张 8MB 约束下的课表

如果你用 QNN 框架而不是 llama.cpp 手动管理，VTCM 的调度由框架自动完成。这发生在 `graphFinalize` 阶段——很多开发者以为这一步是在"编译"模型，实际上它是在**排课表**。

Finalize 分为三个核心阶段：

**Optimization（优化）：** 算子融合、布局转换。把多个小 op 合并成大 op，减少中间结果的 VTCM 占用。

**Sequencing（排序）：** 按依赖关系对 op 排序，分析每个 tensor 的生命周期——它什么时候被创建、什么时候最后一次被读取。生命周期不重叠的 tensor 可以复用同一块 VTCM 地址，类似编译器的寄存器分配。

**VTCM Allocation（分配）：** 给每个 tensor 分配 VTCM 偏移地址。如果所有同时存活的 tensor 总大小超过可用 VTCM，调度器必须插入 **spill 节点**（把数据从 VTCM 暂存回 DDR）和 **fill 节点**（把数据从 DDR 搬回 VTCM）。每一对 spill/fill 都意味着一次额外的 DMA 传输。

真机 profiling 揭示了 VTCM 大小对调度复杂度的影响。同样是 4 个算子串联的计算图：

| VTCM 大小 | VTCM Allocation 耗时 | 倍数 |
|----------|---------------------|------|
| MAX (8MB) | 122 us | 1x |
| 1MB | 563 us | **4.6x** |

可用 VTCM 从 8MB 缩小到 1MB，仅分配阶段的耗时就暴涨 4.6 倍。原因很直接：空间越小，生命周期分析越精细，spill/fill 决策越复杂，"课表"越难排。

在实际部署中，VTCM 经常需要在多个模型之间共享。QNN 提供了分区机制：两个模型可以各用一半 VTCM，Graph A 使用 [0, 4MB)，Graph B 使用 [4MB, 8MB)。不分区的话，两个模型交替执行时需要保存/恢复 VTCM 内容，类似操作系统的上下文切换。分区消除了这个开销，但每个模型能用的空间更小，调度更紧张。

---

## 通信墙——CPU 和 NPU 隔着一道远程调用

格式对了，搬运也能流水线化了。但还有一个更基本的问题：CPU 怎么把活儿交给 NPU？NPU 不是 CPU 的一个协处理器寄存器，不是写个地址就能触发计算的。它是一颗独立的处理器。

### NPU 是另一台机器

骁龙 8 Gen 3 的 CDSP（Compute DSP）运行自己的实时操作系统 QuRT，有自己的内存空间、自己的线程调度器、自己的异常处理。CPU 和它之间的关系，更像是两台通过网线连接的服务器，而不是一颗 CPU 和它的浮点单元。

CPU 想让 DSP 干活，走的是 **FastRPC**——Fast Remote Procedure Call。名字里有"Fast"，但它做的事情一点都不轻量：

1. 用户态把参数序列化成 RPC 消息
2. 切换到内核态
3. 通过 SMMU（系统内存管理单元）建立共享内存映射
4. 通知 DSP 端的 QuRT 有新请求
5. DSP 反序列化参数，执行计算
6. 结果原路返回

实测每次 FastRPC 调用的固定开销：**364 微秒**。

364 微秒听起来不多。但一次 [256x1024x4096] 的 HMX 矩阵乘法只要 510 微秒。也就是说，如果你每做一次矩阵乘法都单独发一次 RPC，通信开销占总时间的 **42%**。将近一半的时间花在"告诉 NPU 该干活"而不是"干活"上。

### 算一笔 LLM 的账

LLM 推理不是一次矩阵乘法的事。以 Qwen3-0.6B 为例，每生成一个 token 需要经过 28 层 transformer，每层包含约 7 个算子（QKV 投影、attention、输出投影、FFN up/gate/down、RMSNorm 等），合计约 **196 次 DSP 调用**。

如果每次都走 FastRPC：

```
196 次 x 364 微秒/次 = 71 毫秒
```

71 毫秒，光是通信开销。如果 DSP 上的实际计算需要 50 毫秒，那通信时间比计算还长。总耗时 121 毫秒，通信占 59%。

这显然不可接受。

### 解法一：把多个算子打包进一次调用

最直接的思路：既然每次调用有固定开销，那就减少调用次数。把多个算子合并到一次 DSP 调用里执行。

这里要先区分两个层次的 API。高通提供了 **SDKL**（SDK Layer），在 ARM 侧调用，一行 `sdkl_npu_mm_f16()` 完成一次矩阵乘法——内部走 FastRPC 把数据发到 DSP、做 layout 转换、分配 VTCM、调 HMX、结果写回，全部自动。方便是方便，但每次调用都是一次完整的 FastRPC 往返（364 微秒）。如果你的模型每层有 7 个矩阵乘法，28 层就是 196 次 RPC。

另一个选择是 **HexKL Micro API**，它运行在 DSP 侧。你调 `hexkl_micro_matmul()`，它帮你处理 32x32 tile 的切分和 VTCM 排布，但你的代码本身就跑在 DSP 上——多次 HexKL Micro 调用之间没有 RPC 开销，只是普通的函数调用。你甚至可以在同一次 DSP 调用里混合 HexKL Micro（矩阵乘）和 HVX 内联函数（反量化、激活函数），把整个 transformer 层的计算打包进去。

下面这个真机实验展示了这个模式。在一次 DSP 调用内，HVX 和 HMX（通过 HexKL Micro）接力完成三步操作：

```
一次 DSP 调用内部（HVX + HexKL Micro）：

Step 1 (HVX): 反量化 int8 -> f16     128B 向量指令，处理 Q8_0 格式
         |
         v
Step 2 (HMX): 矩阵乘法 [32x64]x[64x128]   32x32 tile 矩阵指令
         |
         v
Step 3 (HVX): 加偏置 f32             128B 向量加法
```

全程数据留在 VTCM，不回 DDR。HVX 和 HMX 之间的切换只是函数调用，零通信开销。

一个完整的 transformer 层其实就是 HVX 和 HMX 的交替：RMSNorm（HVX）、QKV 投影（HMX，权重需要 HVX 反量化）、RoPE（HVX）、Attention（HMX+HVX）、FFN（HMX）、SiLU（HVX）。如果把整个 transformer 层（或一个大的子图）放进一次 DSP 调用，196 次调用可以压缩到十几次甚至几次。

### 解法二：绕过内核

FastRPC 慢在哪？慢在内核态切换和 SMMU 映射。如果能绕过内核，直接在用户态通信呢？

Qualcomm 提供了另一种通信机制：**dspqueue**。它的原理是共享内存消息队列——ARM 和 DSP 共享一块物理内存，ARM 往队列里写消息，DSP 轮询读取，不经过内核。

骁龙 8 Gen 3 上的实测数据：

| 通信方式 | 单次开销 | 196 次/token |
|----------|---------|-------------|
| FastRPC  | 364 微秒 | 71 毫秒     |
| dspqueue | 61 微秒  | 12 毫秒     |

dspqueue 比 FastRPC 快约 **6 倍**。每 token 省下 59 毫秒。

但 dspqueue 不是免费的午餐。用 dspqueue 意味着你要自己管理一切：共享内存的分配（`rpcmem_alloc` + `fastrpc_mmap`）、消息协议的定义（op code、张量元数据、buffer 列表）、DSP 端的消息循环和算子分发。QNN 框架提供的自动调度、VTCM 分配、DMA 流水线——全部要自己来。

### 解法三：异步流水线——发了就走，批量等结果

前两个解法分别减少调用次数和单次开销。还有第三个维度：**重叠通信和计算**。

FastRPC 是同步的——CPU 发一个请求，等 DSP 做完才能发下一个。CPU 和 DSP 之间始终有一方在闲等。而 dspqueue 天然支持异步：ARM 侧的 `enqueue()` 把请求写进共享内存队列后立即返回，不等 DSP 回复。你可以连续 enqueue 多个算子，最后调一次 `flush()` 阻塞等待全部完成。

```
同步模式（FastRPC）：
CPU: [发送]----[等待]----[发送]----[等待]----[发送]----[等待]
DSP:       [计算]              [计算]              [计算]

异步模式（dspqueue）：
CPU: [发送][发送][发送][等 flush]
DSP:       [计算][计算][计算]
```

DSP 端的消息循环是一个 `while(1)` 紧循环，用 `dspqueue_read_noblock()` 轮询队列。请求到了立刻执行，不需要唤醒线程、不需要内核参与。ARM 侧用一个原子计数器 `op_pending` 跟踪有多少请求还没完成，`flush()` 只需等这个计数器归零。

llama.cpp 的 Hexagon 后端就是这么做的。每个 token 的 196 个算子，ARM 侧 enqueue 后不等回复，DSP 侧一个接一个地从队列里取出执行。通信和计算完全重叠。

### 解法四：零拷贝共享内存

还有一个容易被忽视的通信开销：**数据传输**。

FastRPC 需要通过内核做 SMMU 映射，确保 DSP 能访问 CPU 侧的内存。每次调用都可能涉及映射/解映射，这也是 364 微秒里的一部分。

dspqueue 的做法是一次性分配好共享内存，后续完全零拷贝：

```c
// 一次性：分配 ARM 和 DSP 都能直接访问的物理内存
void *buf = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
int fd = rpcmem_to_fd(buf);
fastrpc_mmap(fd, ...);  // 建立 SMMU 映射

// 之后：ARM 直接写，DSP 直接读，不经过内核
memcpy(buf, weights, size);  // ARM 写入权重
// DSP 通过同一个物理地址直接读取，只需 cache flush/invalidate
```

模型权重、激活值、KV cache——全部放在这种共享内存里。ARM 侧填数据，DSP 侧直接计算，中间没有任何拷贝或序列化。llama.cpp 用 `rpcmem_alloc` 分配所有 tensor buffer，DSP 端通过 buffer 的文件描述符直接访问同一块物理内存。

---

## KV Cache——三墙汇合之处

前面三道墙各有成熟的解法：格式转换可以在模型加载时预处理，内存搬运可以用 DMA 流水线隐藏，通信可以打包或走共享内存。但这些解法有一个共同前提：**数据是静态的**。

LLM 的 Attention 层打破了这个前提。KV Cache 是推理流程中唯一的"活数据"——它随着每个 token 的生成而增长，让三道墙同时被击中。

### 三墙同时命中

Attention 的核心运算是 `scores = Q x K^T`。Q 是当前 token 的查询向量（固定大小），K 是之前所有 token 的 Key 向量拼成的矩阵（KV Cache 的一部分）。每生成一个新 token，K 就多一行。

这件事为什么致命？因为它同时触发三道墙：

**格式墙**：HMX 要求 weight 必须是 WH Layout。K 每步都变了，WH Layout 要重新转换——而且 K 越来越大，转换成本线性增长。

**内存墙**：K 的大小在推理过程中不断变化，VTCM 的搬运计划（哪些 tile 先搬、放在哪个地址）无法在 Finalize 阶段提前排定。

**通信墙**：K 的更新（写入新 token）和使用（做矩阵乘法）跨越多次 DSP 调用，每次调用都需要同步状态。

每道墙单独看都有解法。KV Cache 的问题在于，三条绕行路线在这里汇合成了一个死胡同。

---

### 旧方案：SmartMask——每步都重新转换

SmartMask 是最直观的做法：K 以普通的行优先格式（row-major）存储在 DDR 中。每次执行 attention 时，先把整个 K 从 row-major 转成 WH Layout，然后交给 HMX 做矩阵乘法。

```
每次 attention:
    rm_to_wh(K_rm)        ← 把整个 K 从行优先转成 WH Layout
    scores = Q x K_wh     ← HMX 矩阵乘
```

我们在骁龙 8 Gen 3 真机上做了基准测试。实验配置：Q 为 [32x64]，K 为 [64x256]（head_dim=64，ctx_size=256），100 次迭代取平均。结果：

- SmartMask：4937 ticks/iter（包含 rm_to_wh 转换 + HMX matmul）
- 其中一次性预转换 K 到 WH 格式的开销：2299 ticks（2 x 8 = 16 个 32x32 tile）

在 256 tokens 这个短上下文下，rm_to_wh 只涉及 16 个 tile 的转换，开销尚可接受。但 rm_to_wh 的成本与 tile 数量成正比，而 tile 数量随上下文长度线性增长：

| ctx_size | tile 数量 | 转换开销占 attention 比例 |
|----------|----------|------------------------|
| 256 | 2 x 8 = 16 | ~2% |
| 4096 | 2 x 128 = 256 | **~30%** |
| 32768 | 2 x 1024 = 2048 | **~60%** |

到 32K 上下文时，你的 NPU 有六成时间不是在算矩阵乘法，而是在做格式转换。HMX 的算力在空转，等数据排好队列。

---

### 新方案：NativeKV——K 生来就是瓦片格式

NativeKV 的核心思想极其简单：既然 HMX 只接受 WH Layout，那就从一开始就把 K 存成 WH Layout，永远不做转换。

```
模型加载时:
    K_cache 按 WH Layout 格式分配

每个新 token:
    把新的 Key 向量直接写入 K_cache 的正确 tile 位置

每次 attention:
    直接用 K_cache    ← 零转换
```

真机结果验证了这个想法。同样的 Q[32x64] x K[64x256] 实验：

- SmartMask：4937 ticks/iter
- NativeKV：4822 ticks/iter
- 加速比：1.02x
- 预转换成本 2299 ticks，在 20 步后回本

在 64x256 这个小尺寸上差距不大。但关键在于：NativeKV 的开销不随上下文增长而增长，而 SmartMask 的 rm_to_wh 开销是 O(head_dim x ctx_size)。上下文越长，NativeKV 的优势越大。

**难点在于"写入正确的 tile 位置"。** WH Layout 不是简单的行优先或列优先，而是一种多级 tile 交错排列。新 token 的 Key 向量要写入 K_cache，必须精确计算每个元素的物理地址。这就是 Genie 引擎的 `fromFlatOffset` 公式：

```c
static int fromFlatOffset(int DIN, int DOUT, int N_TILE, int din, int dout) {
    int tile_size   = min(DOUT, N_TILE);
    int tile_stride = DIN * tile_size;
    int tile_idx    = dout / tile_size;

    int dout_0 = (dout % tile_size) >> 5;   // tile 内 32-col 块
    int dout_1 = dout & 0x1f;               // 块内列
    int din_0  = din >> 5;                   // 32-row 块
    int din_1  = (din & 0x1f) >> 2;          // 块内 4-row 组
    int din_2  = din & 0x3;                  // 组内行

    return tile_idx * tile_stride + din_0 * (tile_size << 5) +
           (dout_0 << 10 | din_1 << 7 | dout_1 << 2 | din_2);
}
```

逻辑分层是：先定位到哪个大 tile（tile_idx），再定位到 tile 内哪个 32x32 block（din_0, dout_0），最后在 1024 字节的 block 内部按 [8:din_1][32:dout_1][4:din_2] 交错排列。最内层的排列保证 HMX 一次 128 字节的加载恰好取到 32 个连续的 dout 元素，与 32-wide SIMD 对齐。

另一个细节是 K 和 V 使用不同的 tile 大小：**K_TILE=256，V_TILE=64**。原因是 K 参与的矩阵乘法 `Q x K^T` 中 dout 维度是 ctx_size（可能很长），大 tile 减少搬入 VTCM 的次数。V 参与的矩阵乘法 `softmax(scores) x V` 中 dout 维度是 embed_dim（通常 64 或 128），小 tile 即可覆盖。

---

### 32 对齐约束与 MaskedSoftmax

NativeKV 引入了一个新约束：WH Layout 以 32x32 为基本单元，新 token 的写入位置必须 32 对齐。

```c
int new_idx = ceil(n_valid_kv / 32.0) * 32;
```

| 已有 token 数 | 新 token 写入位置 | 浪费的 slot |
|-------------|----------------|------------|
| 0 | 0 | 0 |
| 1 | 32 | 31 |
| 33 | 64 | 31 |

33 个有效 token 必须分配 64 个 slot，其中 31 个是 padding。这些 padding 位置在 K 矩阵中有随机数据，如果不处理，会污染 attention 分数。

传统做法是 `Softmax(Add(scores, mask))`——先给 padding 位置加一个极大的负数（比如 -10000），让 softmax 输出趋近于零。但这有两个问题：一是多了一步 Add 操作，二是当有效 token 只占总 slot 的一小部分时（比如 decode 阶段，33 个有效 token 占 64 个 slot），大量计算花在处理 padding 上。

Genie 的 MaskedSoftmax 将 mask 操作融入 softmax 内部：先只对有效位置做 softmax，然后直接将 padding 位置的输出设为 0。当有效 token 占比低时，这比 Add+Softmax 更快，精度也更好——因为避免了 -10000 这类极端值对浮点计算的干扰。

---

### 为什么需要新硬件（v75+）

NativeKV 不是一个纯软件优化。在 v73 架构（骁龙 8 Gen 2）上，HMX 已经存在，WH Layout 也已经存在——但 QNN 的 tensor I/O 接口只接受 row-major 格式。哪怕你在应用层把 K 存成了 WH Layout，传给 QNN 图时仍然要声明为 row-major，QNN 内部会再转一次。

v75+ 架构（骁龙 8 Gen 3 / 8 Elite）引入了 `QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT` 标志。骁龙 8 Gen 3 对应 v75，8 Elite 对应 v79，两者都支持此特性。有了这个标志，K tensor 可以直接以 WH 格式作为 QNN 图的 I/O，NPU 硬件的数据通路能直接消费 WH 格式的数据，不需要内部转换。

```cpp
bool isKvOutputHMXFormat =
    QNN_TENSOR_GET_DATA_FORMAT(key_out->tensor) ==
    QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT;

if (isKvOutputHMXFormat)
    manager = std::make_unique<NativeKV>(...);   // v75+: 零转换
else
    manager = std::make_unique<SmartMask>(...);   // 旧硬件: 回退
```

这不是算力的提升——v73 和 v75 的 HMX 矩阵乘法速度几乎一样。差别在于数据通路。v73 的 NPU 像一台只接受 A4 纸的打印机，哪怕你的文件本来就是 A4 的，进入打印机前也要过一道"格式检查+重新排版"。v75 取消了这道关卡。

---

### Genie 的长上下文策略：多图变体 + 分块预填充

NativeKV 解决了"每步 attention 不用重新转格式"的问题。但 LLM 推理还有另一个挑战：**上下文长度是动态的**——从用户输入的第一个 token 到 32K 的长对话，KV cache 的大小跨越三个数量级。

QNN 的核心限制是图的 shape 必须在编译时确定。Genie 的解决方案是**预编译多个图变体**，每个变体对应一个固定的 `(AR-n, CL-m)` 组合——AR 是一步处理的 token 数，CL 是上下文窗口大小：

```
预编译的图变体（示例）：
AR-1,   CL-2048    ← decode 阶段，短上下文
AR-32,  CL-2048    ← 小批量 prefill
AR-128, CL-2048    ← 大批量 prefill
AR-1,   CL-4096    ← decode 阶段，长上下文
AR-128, CL-4096    ← 长上下文 prefill
```

**分块预填充（Chunked Prefill）**：用户输入 200 个 token，但最大的 AR 变体只有 128。Genie 的策略是贪心切块——先用 AR-128 处理前 128 个 token，更新 KV cache，再用 AR-128 处理剩下的 72 个（padding 到 128）。每个 chunk 之间 KV cache 在 DDR 中原地增长。

**上下文窗口升级**：当累积的 KV 数量超过当前 CL 容量时，Genie 自动切换到更大的 CL 变体，触发 `reshapeCache()` 在 DDR 中原地重排 KV 布局（反向拷贝避免覆盖未读数据）。

**超长上下文的淘汰策略**：当 KV cache 超出最大 CL 时，Genie 提供两种淘汰机制：

- **SlidingWindow（滑动窗口）**：保留最近的 token + 固定的"锚点 token"（如 BOS），FIFO 淘汰最旧的。所有 attention head 共享同一淘汰决策。简单但粗暴。
- **KeyDiff（评分淘汰）**：在 HTP 上跑一个**评分网络**，给每个 cache 条目打分，按重要性淘汰。不同 head 可以保留不同的 token 位置——某个 head 觉得第 50 个 token 重要，另一个 head 可能不这么认为。精细但有额外计算开销。

整个 KV cache 始终在 **DDR** 中（通过共享内存分配），不在 VTCM 里——一个 28 层、2 KV head、head_dim=64 的模型，4K 上下文就需要 56MB（F16），32K 上下文需要 448MB，远超 8MB VTCM。VTCM 只在 attention 计算时临时使用，DMA 把需要的 K/V tile 搬进来，算完就释放。

---

### llama.cpp 的另一条路：不转格式，DMA 流式计算

llama.cpp 的 Hexagon 后端选择了完全不同的 KV cache 策略：**不用 WH Layout，不用 NativeKV，K 和 V 就以标准行优先格式存在 DDR 共享内存里。**

这意味着它不需要 v75+ 硬件，从 v68 到 v81 全部兼容。代价是什么？每次 attention 都要处理非 WH 格式的 K——但 llama.cpp 绕过了格式转换，用了一个更直接的方案：**Flash Attention + DMA 流式处理**。

```
llama.cpp 的 Attention 路径：

Q (当前 token)   ← 在 VTCM 中
K (全部历史)     ← 在 DDR 中，行优先格式
V (全部历史)     ← 在 DDR 中，行优先格式

for each block of 64 tokens:
    DMA: K_block[64] DDR → VTCM     ← 后台搬运
    HVX: scores = Q · K_block^T      ← F16 点积
    online_softmax(scores)            ← 流式更新 max 和 sum
    DMA: V_block[64] DDR → VTCM
    HVX: output += softmax(scores) · V_block

（双缓冲：搬第 i+1 块的同时算第 i 块）
```

核心技巧有两个：

**第一，Online Softmax**。标准 softmax 需要先算完所有 `Q·K^T` 分数，找到全局 max，再做归一化。这要求整个 K 同时在内存里。Online softmax 边算边更新 max 和 sum，每个 block 处理完就可以丢弃，不需要保存完整的分数矩阵。内存占用从 O(ctx_size) 降到 O(block_size)。

**第二，DMA 双缓冲**。搬第 i+1 块的数据和算第 i 块的结果并行进行。KV cache 虽然在 DDR 里，但每个 block 只有 64 个 token × head_dim × 2 bytes，DMA 搬运速度远快于计算速度，搬运延迟被完全隐藏。

**两种方案的取舍**：

| | Genie (NativeKV) | llama.cpp (Flash Attention) |
|---|---|---|
| K 存储格式 | WH Layout（DDR 中） | 行优先（DDR 中） |
| 每步 attention 开销 | 零格式转换 | 零格式转换（根本不转） |
| 新 token 写入 | `fromFlatOffset` 计算 tile 地址 | 简单 memcpy 追加 |
| Attention 算法 | 标准 Q·K^T matmul（HMX） | Flash Attention（HVX F16 点积） |
| 矩阵乘硬件 | HMX（32x32 tile） | HVX（128B 向量点积） |
| 硬件要求 | v75+（需要 WH I/O 支持） | v68+（所有 Hexagon） |
| 上下文长度限制 | 受 CL 变体和淘汰策略限制 | 受 DDR 共享内存大小限制 |

Genie 的优势在于用 HMX 做 attention matmul（吞吐更高），但需要新硬件和复杂的多图管理。llama.cpp 的优势在于简单、兼容性好，Flash Attention 的 DMA 流水线在长上下文下也能保持稳定性能——因为它的开销始终是 O(ctx_size) 的线性 DMA，不存在随上下文增长而爆炸的格式转换。

---

### 更大的图景：NPU 演进的方向

NPU 每一代的核心进步都不是"更多 TOPS"，而是减少一道墙：

| 代际 | 关键改进 | 解决的墙 |
|------|---------|---------|
| v68 → v73 | 加入 HMX 矩阵单元 | 算力墙（从向量到矩阵） |
| v73 → v75 | VTCM 扩大到 8MB | 内存墙 |
| v73 → v75 | dspqueue 替代 FastRPC（61us vs 364us） | 通信墙 |
| v73 → v75 | HMX Weight Layout 作为 I/O 格式 | **格式墙（KV Cache 专项）** |

34 TOPS 的算力从第一天起就在那里。真正的工程战场从来不在乘法器上，而在乘法器前面那条越来越短、越来越灵活的数据通路上。理解了这一点，你就能判断下一代芯片的宣传里，哪些参数是真正的进步，哪些只是数字游戏。

---

## llama.cpp：19000 行代码如何喂饱 NPU

llama.cpp 的 Hexagon 后端大约 19000 行 C 代码，从零实现了 30 个算子——matmul、RMS norm、SiLU、softmax、RoPE、flash attention，全部手写 HVX intrinsics 和 HMX 内联汇编。它兼容 v68 到 v81 全部六代 Hexagon 架构，每代编译独立的 .so。要理解它的设计，从 ARM 与 DSP 之间的通信链路看起最直观。

ARM 和 DSP 是两颗独立的处理器。常规的跨处理器调用走 FastRPC，每次经过内核态，实测延迟约 364 微秒。llama.cpp 的做法是把 FastRPC 的使用压缩到只有两次：一次 `start()`，一次 `stop()`。整个 IDL 接口就这两个函数。所有推理期间的算子调度走 dspqueue——一块 ARM 和 DSP 共享的环形缓冲区，单次写入延迟约 61 微秒。ARM 侧的 `enqueue()` 往队列写入 `htp_general_req` 消息（约 200 字节，按 64 字节缓存行对齐），DSP 侧的 `htp_packet_callback()` 在 `while(1)` 死循环里用 `dspqueue_read_noblock()` 不断轮询。`enqueue()` 只递增原子计数器 `op_pending`，不等 DSP 回复；攒够一批后调 `flush()` 阻塞等待全部完成。整个流水线绕过内核，用共享内存直接对话。QNN 的通信机制也走 FastRPC/dspqueue，区别在于 llama.cpp 直接操作这一层，没有中间的图编译和运行时调度。

消息到达 DSP 之后，首先要解决的问题是数据格式。HMX 的矩阵乘法单元以 32x32 FP16 tile 为基本操作粒度，每个 tile 2048 字节。QNN 为此设计了 WH Layout，在离线阶段把整个权重矩阵预先排成瓦片交错格式，存成 context binary。llama.cpp 没有离线编译阶段。它设计了自己的 "x4x2 repack" 格式：4 个连续量化块交错排列，quants 在前、scales 在后。Repack 在 ARM 侧加载权重时一次性完成，之后权重布局就固定了。这个格式不是为 HMX 的瓦片结构设计的，而是为 HVX 的 128 字节向量加载优化的——先用 HVX 做反量化和数据重排，再喂给 HMX 的 32x32 tile。支持的量化格式包括 F32、F16、Q4_0、Q8_0、IQ4_NL、MXFP4，反量化在 DSP 侧计算时完成，利用 HVX 的 LUT 指令做查表转换。HMX 矩阵乘的核心代码来自高通开源的 htp-ops-lib 项目，用内联汇编操作 tile：`activation.hf = mxmem(ptr, limit):deep` 加载激活，`weight.hf = mxmem(ptr, limit)` 加载权重，`mxmem(out, 0):after.hf = acc` 写回结果。

数据格式对了，还要解决往哪里放。HMX 唯一能直接读写的内存是 VTCM，一块 8MB 的片上 SRAM，而模型权重动辄数 GB。QNN 在 Finalize 阶段静态规划好每一步计算用 VTCM 的哪一段。llama.cpp 选择运行时动态分配：启动时通过 `HAP_compute_res_acquire()` 请求全部 8MB VTCM，内部用顺序分配器 `vtcm_salloc` 把 VTCM 切成四块——weight tiles、activation tiles、output tiles、scratch。`hmx_compute_chunks()` 函数计算最优的 mc x nc 分块尺寸，目标是在 VTCM 容量内最大化数据复用：weight tile 驻留不动，activation tile 流式换入，output tile 就地累加。工作池最多 10 个 HVX 线程，每个线程有独立的 DMA 队列，实现双缓冲流水线——一块 tile 在计算，下一块已经在搬运。llama.cpp 还注册了 VTCM release callback，推理空闲时如果另一个进程需要 VTCM，它会主动释放，实现协作式共享。

KV cache 的处理体现了另一个设计取舍。llama.cpp 的 KV cache 存在 `rpcmem_alloc2()` + `fastrpc_mmap()` 分配的共享内存缓冲区中，不在 VTCM 里常驻。`SET_ROWS` 算子写入新的 K/V 条目，`GET_ROWS` 算子读取。`FLASH_ATTN_EXT` 实现融合注意力——从共享内存按需 DMA 搬运 K/V 到 VTCM，在 VTCM 中用 F16 做点积，算完释放空间给下一批。QNN 的 NativeKV 方案让 KV cache 始终以 WH Layout 存储来消除格式转换，但需要 v75+ 硬件支持。llama.cpp 的方案代价是每次注意力计算都要做一次 DMA 搬运，换来的是全架构兼容。

回头看这 19000 行代码，做的事情本质上就是三件：把数据排成硬件要的格式（x4x2 repack + HVX 重排）、在 8MB 里编排搬运时刻表（vtcm_salloc + DMA 双缓冲）、用共享内存绕过内核（dspqueue + rpcmem）。没有 Finalize，没有 context binary，没有 QNN 图编译。全部手动，全部透明。

34 TOPS 确实在那里。瓶颈从来不是算不动，而是喂不饱。

---

*全文数据来自骁龙 8 Gen 3（Hexagon v75 架构，8MB VTCM）真机测试与 llama.cpp Hexagon 后端源码分析。*
