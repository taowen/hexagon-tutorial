# 第五章：dspqueue vs FastRPC -- 理解 llama.cpp 的 Hexagon 架构

前四章我们走了两条路在 Hexagon DSP 上跑代码：

- **ch01/ch02**：`run_main_on_hexagon` 通过 FastRPC 调用 DSP，每次调用都走内核态
- **ch03/ch04**：QNN 框架整图执行，算子覆盖依赖 QNN 库

但 llama.cpp 选了第三条路：**绕过 QNN，用 dspqueue 通信 + 手写 HVX/HMX 内核**。为什么？

答案藏在 FastRPC 的开销里。LLM 推理是高频小 op 场景：Qwen3-0.6B 每个 token 需要 28 层 x 7 个算子 = 196 次 DSP 调用。如果每次走 FastRPC，光通信开销就不可接受。

本章用自己写的代码（不是 SDK 示例，不是 llama.cpp）复现 dspqueue vs FastRPC 的性能差异，然后对比 llama.cpp 的源码结构，看它如何把这个模式放大到完整的 LLM 推理。

## 实验目标

1. 用 dspqueue 实现 2 种算子（SCALE、ADD），对比 FastRPC 直接调用的开销
2. 通过代码对比，理解 llama.cpp hexagon 后端的架构选择

## 源码结构

```
ch05-llama-cpp-hexagon/src/
├── dspqueue_demo.idl          # FastRPC IDL -- 只定义生命周期（start/stop）
├── dspqueue_demo_shared.h     # ARM ↔ DSP 共享消息协议（op code + req/rsp）
├── demo_cpu.c                 # ARM 端：dspqueue_create, enqueue ops, benchmark
└── demo_dsp.c                 # DSP 端：dspqueue_import, 消息循环, op dispatch
```

源码本身就是教程。下面提取关键片段，每段对比 llama.cpp 的对应代码。

## 代码解读 -- IDL：FastRPC 只管生命周期

```c
// dspqueue_demo.idl
interface dspqueue_demo : remote_handle64 {
    AEEResult start(in uint64 dsp_queue_id);
    AEEResult stop(rout uint64 process_time);
    AEEResult do_op(in sequence<uint8> input, rout sequence<uint8> output,
                    in uint32 op, rout uint64 process_time);
};
```

`start` 传递 dspqueue ID 给 DSP，`stop` 关闭队列。`do_op` 是 FastRPC 对比路径，用于 benchmark。

对比 llama.cpp 的 `htp_iface.idl`：

```c
// llama.cpp: htp/htp_iface.idl
interface htp_iface : remote_handle64 {
    AEEResult start(in uint32 sess_id, in uint64 dsp_queue_id,
                     in uint32 n_hvx, in uint32 use_hmx);
    AEEResult stop();
};
```

模式完全一样：**FastRPC 只用于 start/stop，推理计算全走 dspqueue**。

## 代码解读 -- 消息协议：op code dispatch

```c
// dspqueue_demo_shared.h
#define OP_SCALE    1    /* out[i] = in[i] * factor */
#define OP_ADD      2    /* out[i] = a[i] + b[i]    */

struct demo_req {
    uint32_t op;              /* OP_SCALE, OP_ADD */
    uint32_t param;           /* scale factor 等参数 */
    uint32_t n_elem;          /* 元素数量 */
    uint32_t reserved;
};

struct demo_rsp {
    uint32_t op;
    uint32_t status;          /* 0 = OK */
};
```

对比 llama.cpp 的 `htp-msg.h`：同样的结构，只是规模不同 -- llama.cpp 定义了 29 种 op code（`HTP_OP_MUL_MAT`, `HTP_OP_ADD`, `HTP_OP_RMS_NORM` ...），`htp_general_req` 有 ~200 字节（包含张量元数据），而我们的 `demo_req` 只有 16 字节。

## 代码解读 -- ARM 端：共享内存 + dspqueue

ARM 端的核心流程：分配共享内存 -> 创建 dspqueue -> 发送 op 请求。

```c
// demo_cpu.c -- 分配共享内存（零拷贝）
// 对比 llama.cpp: rpcmem_alloc2 + fastrpc_mmap 在 ggml-hexagon.cpp
buffers[i] = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                           RPCMEM_DEFAULT_FLAGS | RPCMEM_HEAP_NOREG,
                           buf_size);
fds[i] = rpcmem_to_fd(buffers[i]);
fastrpc_mmap(domain_id, fds[i], buffers[i], 0, buf_size, FASTRPC_MAP_FD);
```

```c
// demo_cpu.c -- 创建 dspqueue，通过 FastRPC 传递 queue ID（唯一一次 FastRPC）
// 对比 llama.cpp: dspqueue_create + dspqueue_export + htp_iface_start
dspqueue_create(domain_id, 0, 4096, 4096,
                packet_callback, error_callback, &ctx, &queue);
dspqueue_export(queue, &dsp_queue_id);
dspqueue_demo_start(g_handle, dsp_queue_id);
```

```c
// demo_cpu.c -- 发送 op 请求（不走内核态，写共享内存）
// 对比 llama.cpp: enqueue() 中的 dspqueue_write
struct demo_req req;
req.op = op;
req.param = (op == OP_SCALE) ? 2 : 0;
req.n_elem = buf_size;

struct dspqueue_buffer dbufs[3];
dbufs[0].fd = fds[0];
dbufs[0].flags = DSPQUEUE_BUFFER_FLAG_REF
               | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
               | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;

dspqueue_write(queue, 0, n_bufs, dbufs,
               sizeof(req), (const uint8_t *)&req, 1000000);
```

llama.cpp 的 `enqueue()` 做的事情完全一样，只是 `req` 是 `htp_general_req`（带张量元数据），buffer 可能有 3-6 个（src0/src1/src2/.../dst）。

## 代码解读 -- DSP 端：消息循环 + op dispatch

DSP 端的骨架和 llama.cpp 完全一致：导入队列 -> 回调中读消息 -> switch 分发 -> 写响应。

```c
// demo_dsp.c -- 导入 ARM 端创建的 dspqueue
// 对比 llama.cpp: htp_iface_start 中的 dspqueue_import
AEEResult dspqueue_demo_start(remote_handle64 handle, uint64 dsp_queue_id) {
    struct demo_context *ctx = (struct demo_context *)handle;
    dspqueue_import(dsp_queue_id, packet_callback, error_callback,
                    ctx, &ctx->queue);
    return AEE_SUCCESS;
}
```

```c
// demo_dsp.c -- 消息回调：读取请求 → switch(op) → 执行 → 写响应
// 对比 llama.cpp: htp_packet_callback 中的 switch(req.op)，dispatch 29 种算子
static void packet_callback(dspqueue_t queue, int error, void *context) {
    struct demo_context *ctx = (struct demo_context *)context;
    while (1) {
        struct demo_req req;
        int err = dspqueue_read_noblock(queue, &flags,
                      DEMO_MAX_BUFFERS, &n_bufs, bufs,
                      sizeof(req), &msg_len, (uint8_t *)&req);
        if (err == AEE_EWOULDBLOCK) return;

        switch (req.op) {
            case OP_SCALE:
                op_scale(bufs[0].ptr, bufs[1].ptr, req.n_elem,
                         (uint8_t)req.param);
                break;
            case OP_ADD:
                op_add(bufs[0].ptr, bufs[1].ptr, bufs[2].ptr, req.n_elem);
                break;
        }

        rsp.op = req.op;
        rsp.status = 0;
        dspqueue_write(queue, 0, n_rsp_bufs, rsp_bufs,
                       sizeof(rsp), (const uint8_t *)&rsp,
                       DSPQUEUE_TIMEOUT_NONE);
    }
}
```

llama.cpp 的 `htp_packet_callback` 结构完全相同，只是 switch 里有 29 个 case：`HTP_OP_MUL_MAT` 走 HMX matmul，`HTP_OP_ADD` 走 HVX 向量运算，`HTP_OP_RMS_NORM` 走 HVX 归一化... 每个 case 调用对应的 `proc_*_req` 函数。

## 构建与运行

```bash
bash ch05-llama-cpp-hexagon/build.sh
bash ch05-llama-cpp-hexagon/run_device.sh
```

## 真机实测数据（骁龙 8 Gen 3, v75）

```
Buffer size: 1MB, 1000 iterations

                          Ops    Overhead/op
dspqueue OP_SCALE         1000   61 us
dspqueue OP_ADD           1000   57 us
FastRPC  OP_SCALE         1000   364 us
```

**dspqueue 通信开销 ~60us，FastRPC ~364us -- dspqueue 快约 6 倍。**

DSP 计算时间两条路径相同，差异纯粹在通信开销。

### 对 LLM 推理的影响

以 Qwen3-0.6B 为例，每个 token 需要 196 次 DSP op：

| 通信方式 | 每 op 开销 | 196 ops 总开销 |
|---------|-----------|---------------|
| FastRPC | 364 us | **71 ms** |
| dspqueue | 61 us | **12 ms** |

dspqueue 每 token 省 59ms。如果 DSP 计算本身要 50ms/token，FastRPC 的通信开销（71ms）比计算还大；dspqueue 的通信开销（12ms）则只占 20%。

这就是 llama.cpp 选择 dspqueue 的原因。

## llama.cpp 如何放大这个模式

我们的 demo 和 llama.cpp hexagon 后端是同一个骨架，区别在于规模：

| | 本章 demo | llama.cpp hexagon |
|---|---|---|
| **op 种类** | 2 (SCALE, ADD) | 29 (MUL_MAT, ADD, RMS_NORM, ROPE, SOFTMAX, FLASH_ATTN...) |
| **计算内核** | 标量循环 | HVX intrinsics + HMX 矩阵乘 |
| **消息大小** | 16 字节 | ~200 字节（含张量元数据） |
| **内存管理** | rpcmem 直接用 | rpcmem + VTCM 手动分配 + DMA 流水线 |
| **并行** | 单线程 | worker pool（多 HVX 线程并行） |
| **量化** | uint8 | Q4_0, Q8_0, F16, MXFP4 |
| **DSP 代码量** | ~100 行 | ~15000 行 |

llama.cpp 在这个骨架上加了三层东西：

1. **29 种算子的 HVX/HMX 内核** -- 把 Transformer 的每个计算步骤都搬到 DSP 上，ARM 只负责调度
2. **VTCM + DMA 管理** -- 8MB 片上 SRAM 做 matmul 分块缓冲，DMA 引擎做数据搬运流水线
3. **GGUF 原生支持** -- 直接在 DSP 上反量化 Q4_0/Q8_0，不需要格式转换或 QNN 工具链

## 五章总结

| 章节 | 做了什么 | ARM-DSP 通信 | 核心收获 |
|------|---------|-------------|---------|
| ch01 | hexagon-sim 模拟器跑 HVX+HMX | 无（本地模拟） | HVX/HMX 指令 + VTCM |
| ch02 | 真机 run_main_on_hexagon | FastRPC (~364us/op) | 真机部署 + HAP 电源/VTCM |
| ch03 | QNN + QHPI 自定义算子 | QNN 整图执行 | QNN 框架 + 自定义算子 |
| ch04 | QNN + libnative (x86 模拟) | x86 本地 | 可移植 HVX/HMX 开发 |
| **ch05** | **自己写代码实测 dspqueue** | **dspqueue (~61us/op)** | **理解 llama.cpp 的架构选择** |

从 ch01 到 ch05，一条完整的路径：

1. **ch01** 学会了 HVX/HMX 指令
2. **ch02** 搬到真机，发现 FastRPC 有开销
3. **ch03/ch04** 尝试 QNN 框架，理解它的优势和限制
4. **ch05** 用自己的代码量化了 dspqueue vs FastRPC 的差异，理解 llama.cpp 为什么绕过 QNN，用 dspqueue + 手写内核把整个 LLM 推理搬到 DSP 上
