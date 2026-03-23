# 第四章：通信墙——CPU 和 NPU 隔着一道远程调用

前两章讲的是 NPU 内部的事：格式墙是数据怎么排列，内存墙是数据怎么搬运。这一章要退一步，看一个更基本的问题：CPU 怎么把活儿交给 NPU？

答案比大多数人想象的要复杂。NPU 不是 CPU 的一个协处理器寄存器，不是写个地址就能触发计算的。它是一颗独立的处理器。

## NPU 是另一台机器

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

## 算一笔 LLM 的账

LLM 推理不是一次矩阵乘法的事。以 Qwen3-0.6B 为例，每生成一个 token 需要经过 28 层 transformer，每层包含约 7 个算子（QKV 投影、attention、输出投影、FFN up/gate/down、RMSNorm 等），合计约 **196 次 DSP 调用**。

如果每次都走 FastRPC：

```
196 次 x 364 微秒/次 = 71 毫秒
```

71 毫秒，光是通信开销。如果 DSP 上的实际计算需要 50 毫秒，那通信时间比计算还长。总耗时 121 毫秒，通信占 59%。

这显然不可接受。

## 解法一：把多个算子打包进一次调用

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

## 解法二：绕过内核

FastRPC 慢在哪？慢在内核态切换和 SMMU 映射。如果能绕过内核，直接在用户态通信呢？

Qualcomm 提供了另一种通信机制：**dspqueue**。它的原理是共享内存消息队列——ARM 和 DSP 共享一块物理内存，ARM 往队列里写消息，DSP 轮询读取，不经过内核。

骁龙 8 Gen 3 上的实测数据：

| 通信方式 | 单次开销 | 196 次/token |
|----------|---------|-------------|
| FastRPC  | 364 微秒 | 71 毫秒     |
| dspqueue | 61 微秒  | 12 毫秒     |

dspqueue 比 FastRPC 快约 **6 倍**。每 token 省下 59 毫秒。

但 dspqueue 不是免费的午餐。用 dspqueue 意味着你要自己管理一切：共享内存的分配（`rpcmem_alloc` + `fastrpc_mmap`）、消息协议的定义（op code、张量元数据、buffer 列表）、DSP 端的消息循环和算子分发。QNN 框架提供的自动调度、VTCM 分配、DMA 流水线——全部要自己来。

## 解法三：异步流水线——发了就走，批量等结果

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

## 解法四：零拷贝共享内存

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

## 小结

通信墙是三道墙中最"隐形"的一道。格式转换和内存搬运至少还是数据层面的问题，通信开销则完全是架构层面的——NPU 是独立处理器这个事实，决定了每次交互都有不可压缩的固定成本。

优化通信的两个方向——减少调用次数（算子融合）和降低单次开销（dspqueue）——都不是免费的。前者需要把更多计算逻辑放到 DSP 上（更复杂的 DSP 代码），后者需要放弃框架的便利（更大的工程量）。

至此，三道墙都已讲完。它们各有成熟的解法：格式转换可以在模型加载时预处理，内存搬运可以用 DMA 流水线隐藏，通信可以打包或走共享内存。这些解法有一个共同前提：**数据是静态的**。模型权重在加载时就确定了，搬运计划可以提前排好。

但 LLM 的 Attention 层打破了这个前提。KV cache 每生成一个 token 就增长一行，格式要重新转、搬运计划要重新排、通信模式要动态调整。三道墙同时被击中。下一章我们来看 KV cache 如何成为端侧 LLM 推理最难的部分，以及 NativeKV 和 SmartMask 两种截然不同的应对策略。

---

*所有数据来自骁龙 8 Gen 3 真机测试。*
