# llama.cpp 的 Hexagon 后端：19000 行代码如何喂饱 NPU

llama.cpp 的 Hexagon 后端大约 19000 行 C 代码，从零实现了 30 个算子——matmul、RMS norm、SiLU、softmax、RoPE、flash attention，全部手写 HVX intrinsics 和 HMX 内联汇编。它兼容 v68 到 v81 全部六代 Hexagon 架构，每代编译独立的 .so。要理解它的设计，从 ARM 与 DSP 之间的通信链路看起最直观。

ARM 和 DSP 是两颗独立的处理器。常规的跨处理器调用走 FastRPC，每次经过内核态，实测延迟约 364 微秒。llama.cpp 的做法是把 FastRPC 的使用压缩到只有两次：一次 `start()`，一次 `stop()`。整个 IDL 接口就这两个函数。所有推理期间的算子调度走 dspqueue——一块 ARM 和 DSP 共享的环形缓冲区，单次写入延迟约 61 微秒。ARM 侧的 `enqueue()` 往队列写入 `htp_general_req` 消息（约 200 字节，按 64 字节缓存行对齐），DSP 侧的 `htp_packet_callback()` 在 `while(1)` 死循环里用 `dspqueue_read_noblock()` 不断轮询。`enqueue()` 只递增原子计数器 `op_pending`，不等 DSP 回复；攒够一批后调 `flush()` 阻塞等待全部完成。整个流水线绕过内核，用共享内存直接对话。QNN 的通信机制也走 FastRPC/dspqueue，区别在于 llama.cpp 直接操作这一层，没有中间的图编译和运行时调度。

消息到达 DSP 之后，首先要解决的问题是数据格式。HMX 的矩阵乘法单元以 32x32 FP16 tile 为基本操作粒度，每个 tile 2048 字节。QNN 为此设计了 WH Layout，在离线阶段把整个权重矩阵预先排成瓦片交错格式，存成 context binary。llama.cpp 没有离线编译阶段。它设计了自己的 "x4x2 repack" 格式：4 个连续量化块交错排列，quants 在前、scales 在后。Repack 在 ARM 侧加载权重时一次性完成，之后权重布局就固定了。这个格式不是为 HMX 的瓦片结构设计的，而是为 HVX 的 128 字节向量加载优化的——先用 HVX 做反量化和数据重排，再喂给 HMX 的 32x32 tile。支持的量化格式包括 F32、F16、Q4_0、Q8_0、IQ4_NL、MXFP4，反量化在 DSP 侧计算时完成，利用 HVX 的 LUT 指令做查表转换。HMX 矩阵乘的核心代码来自高通开源的 htp-ops-lib 项目，用内联汇编操作 tile：`activation.hf = mxmem(ptr, limit):deep` 加载激活，`weight.hf = mxmem(ptr, limit)` 加载权重，`mxmem(out, 0):after.hf = acc` 写回结果。

数据格式对了，还要解决往哪里放。HMX 唯一能直接读写的内存是 VTCM，一块 8MB 的片上 SRAM，而模型权重动辄数 GB。QNN 在 Finalize 阶段静态规划好每一步计算用 VTCM 的哪一段。llama.cpp 选择运行时动态分配：启动时通过 `HAP_compute_res_acquire()` 请求全部 8MB VTCM，内部用顺序分配器 `vtcm_salloc` 把 VTCM 切成四块——weight tiles、activation tiles、output tiles、scratch。`hmx_compute_chunks()` 函数计算最优的 mc x nc 分块尺寸，目标是在 VTCM 容量内最大化数据复用：weight tile 驻留不动，activation tile 流式换入，output tile 就地累加。工作池最多 10 个 HVX 线程，每个线程有独立的 DMA 队列，实现双缓冲流水线——一块 tile 在计算，下一块已经在搬运。llama.cpp 还注册了 VTCM release callback，推理空闲时如果另一个进程需要 VTCM，它会主动释放，实现协作式共享。

KV cache 的处理体现了另一个设计取舍。llama.cpp 的 KV cache 存在 `rpcmem_alloc2()` + `fastrpc_mmap()` 分配的共享内存缓冲区中，不在 VTCM 里常驻。`SET_ROWS` 算子写入新的 K/V 条目，`GET_ROWS` 算子读取。`FLASH_ATTN_EXT` 实现融合注意力——从共享内存按需 DMA 搬运 K/V 到 VTCM，在 VTCM 中用 F16 做点积，算完释放空间给下一批。QNN 的 NativeKV 方案让 KV cache 始终以 WH Layout 存储来消除格式转换，但需要 v75+ 硬件支持。llama.cpp 的方案代价是每次注意力计算都要做一次 DMA 搬运，换来的是全架构兼容。

回头看这 19000 行代码，做的事情本质上就是三件：把数据排成硬件要的格式（x4x2 repack + HVX 重排）、在 8MB 里编排搬运时刻表（vtcm_salloc + DMA 双缓冲）、用共享内存绕过内核（dspqueue + rpcmem）。没有 Finalize，没有 context binary，没有 QNN 图编译。全部手动，全部透明。

34 TOPS 确实在那里。瓶颈从来不是算不动，而是喂不饱。

---

*所有数据来自骁龙 8 Gen 3 真机测试与 llama.cpp Hexagon 后端源码分析。*
