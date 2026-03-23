# 第三章：内存墙——8MB 要装 14GB

上一章我们看到，NPU 的矩阵单元 HMX 对数据格式有严格要求：必须是 32x32 瓦片的 WH Layout，否则根本无法计算。格式转换的开销在长上下文 attention 中可以占到 60%。

但格式墙只是第一道关卡。即使数据格式完全正确，你还面临一个更基本的物理事实：**数据根本不在 NPU 能够触及的地方。**

---

## 1750 倍的鸿沟

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

## 双缓冲：17 倍加速的来源

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

## 一个反直觉的发现：VTCM 并不比 DDR 快

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

## Bump Allocator：最简单的内存管理

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

## QNN Finalize：排一张 8MB 约束下的课表

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

## 小结

内存墙的本质是容量差距（1750 倍）和搬运开销。解决它的核心技术只有一个：**让搬运和计算在时间上重叠**。DMA 引擎提供了硬件基础，双缓冲提供了编程模式，bump allocator 提供了零开销的内存管理，QNN Finalize 提供了自动化的调度。

但内存墙有一个隐含假设：权重是**静态的**。模型加载后权重不变，DMA 的搬运计划可以提前排好。下一章我们会看到，当 CPU 需要和 NPU 交互时——传参数、传激活值、传 KV cache 更新——一道新的墙出现了：通信墙。每次跨越 CPU 和 DSP 的边界，都要付出一笔固定的 RPC 开销。

*本文数据来自骁龙 8 Gen 3（v75 架构，8MB VTCM）真机实验。*
