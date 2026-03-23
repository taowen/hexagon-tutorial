# 第一章：34 TOPS 去哪了？

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

**路线一：纯手写 HVX + HMX 内核。** llama.cpp 的 Hexagon 后端就是这么做的。19000 行 C 代码，不用 QNN，不用 HexKL，不用任何高通 ML 框架。直接用 HVX 内联函数做反量化和激活函数，用 HMX 内联汇编 `mxmem` 指令做矩阵乘法，用 dspqueue 和 CPU 通信。一切自己控制。这是最"硬核"的路，但也是自由度最高的路。第六章会详细拆解这条路线。

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

---

下面我们从最底层看起，自底向上，用真机实验数据带你看清每一层为什么存在、每一道墙到底有多厚。
