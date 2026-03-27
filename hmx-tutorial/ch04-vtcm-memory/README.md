# 第四章：VTCM 内存管理 — 从分配到 DMA 到流水线

前五章我们在 Hexagon DSP 上跑了 HVX 向量运算、HMX 矩阵乘、QNN 算子，还用 dspqueue 替代了 FastRPC。但有一个关键资源一直在幕后默默工作：**VTCM（Vector Tightly Coupled Memory）**。

ch02 里我们写了 `HAP_compute_res_acquire` 分配 VTCM，但只是把它当一块内存用。本章深入：VTCM 怎么分配、怎么分区、DMA 怎么填充、不同框架怎么管理它、以及 llama.cpp 如何把 DMA + HVX + HMX 编排成一条完整的流水线。

## VTCM 是什么

| 特性 | DDR | L2 Cache | VTCM |
|---|---|---|---|
| 容量 | 8-16 GB | 1-2 MB | 4-8 MB |
| 管理方式 | OS 管理 | 硬件自动 | **软件手动** |
| HVX 访问延迟 | ~100 周期 (L2 miss) | ~10 周期 | **1 周期** |
| HMX 能否使用 | ✗ | ✗ | **✓（必须）** |
| DMA 目标 | ✓ | ✗ | **✓** |

VTCM 的核心价值：
1. **HMX 矩阵乘必须把数据放在 VTCM 里** — 这就是 llama.cpp 的 HMX matmul 用 VTCM 的根本原因
2. **DMA 可以异步填充 VTCM** — 计算和数据搬运重叠
3. **确定性延迟** — 不受其他进程的 cache 污染影响

## 源码结构

```
ch04-vtcm-memory/
├── src/
│   ├── common.h           # 共享声明：VTCM 全局变量、vtcm_seq_alloc()
│   ├── demo_vtcm_alloc.c  # Part 1: VTCM 分配 + main() 入口
│   ├── demo_bump_alloc.c  # Part 2: llama.cpp bump allocator
│   ├── demo_pool_alloc.c  # Part 3: TVM pool allocator
│   ├── demo_dma.c         # Part 4: UDMA DDR→VTCM + 流水线 benchmark
│   └── demo_hvx_bench.c   # Part 5: HVX vadd VTCM vs DDR
├── build.sh               # 编译所有 src/*.c 为一个 .so
└── run_device.sh           # 推送到真机运行
```

和 ch02 一样的模式：编译为 .so，用 `run_main_on_hexagon` 在 CDSP 上执行。不需要 IDL/stub/skel。

## 构建与运行

```bash
bash ch04-vtcm-memory/build.sh
bash ch04-vtcm-memory/run_device.sh
```

## 实验内容

### Part 1: VTCM 分配 — HAP_compute_res API

SDK 提供的统一资源管理器，管理 VTCM、HMX、HVX 的分配：

```c
// Step 1: 查询可用 VTCM
unsigned int vtcm_size = 8 * 1024 * 1024;
HAP_compute_res_query_VTCM(0, &vtcm_size, NULL, NULL, NULL);

// Step 2: 配置属性
compute_res_attr_t attr;
HAP_compute_res_attr_init(&attr);
HAP_compute_res_attr_set_vtcm_param(&attr, vtcm_size, 1);  // 1 = 单页连续

// Step 3: 获取资源
unsigned int ctx_id = HAP_compute_res_acquire(&attr, 100000);

// Step 4: 拿到 VTCM 地址
void *vtcm_base = HAP_compute_res_attr_get_vtcm_ptr(&attr);
```

实验输出：

```
[2/6] VTCM allocation (HAP_compute_res API)...
  VTCM total: 8192 KB (8 MB)
  VTCM allocated: 8192 KB at FF000000
```

**重要陷阱**：VTCM 的页表映射是**线程本地**的。如果用 FastRPC，`open()` 和后续调用可能在不同线程执行，导致 TLBMISS 崩溃。这也是 llama.cpp 选择 dspqueue（同一线程回调）的另一个原因。

### Part 2: llama.cpp 的 Bump Allocator

llama.cpp 在 `htp/hmx-utils.h` 中定义了最简单的 VTCM 分区方式：

```c
// llama.cpp 原始代码
static inline uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    return p;
}
```

每个 matmul op 开始时，重置指针到 `vtcm_base`，然后依次划出各个区域：

```c
// llama.cpp hmx-matmul-ops.c
uint8_t *vtcm_ptr        = (uint8_t *)ctx->vtcm_base;
__fp16  *vtcm_weight     = (__fp16 *)vtcm_seq_alloc(&vtcm_ptr, weight_area_size);
__fp16  *vtcm_activation = (__fp16 *)vtcm_seq_alloc(&vtcm_ptr, activation_area_size);
__fp16  *vtcm_output     = (__fp16 *)vtcm_seq_alloc(&vtcm_ptr, output_area_size);
void    *vtcm_scratch0   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
void    *vtcm_scratch1   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
```

实验输出：

```
[3/6] Bump allocator (llama.cpp style)...
  VTCM layout (bump allocator):
    weight:   +0x00000  (32 KB)
    act:      +0x08000  (32 KB)
    output:   +0x10000  (32 KB)
    scratch0: +0x18000  (64 KB)
    scratch1: +0x28000  (64 KB)
    scales:   +0x38000  (256 B)
    total:    224 KB / 8192 KB (2.7%)
    read/write: OK
```

Bump allocator 的特点：
- **O(1) 分配**，指针加法
- **不需要 free** — 每个 op 重置指针
- **零碎片** — 连续分配，连续释放
- **适合固定 pipeline** — LLM 推理的 op 序列是固定的

### Part 3: TVM 的 Pool Allocator

TVM 的 `HexagonVtcmPool`（`src/runtime/hexagon/hexagon_vtcm_pool.cc`）解决不同的问题：编译器生成的计算图中，算子的 VTCM 需求动态变化，需要真正的分配/释放：

```cpp
// TVM 的 HexagonVtcmPool（简化）
class HexagonVtcmPool {
    std::list<std::pair<char*, size_t>> free_;  // 空闲块列表

    void* Allocate(size_t nbytes) {
        if (nbytes & 0x7FF) {
            // 非 2K 对齐：从末尾分配（减少碎片）
            auto last = free_.end(); last--;
            char* ptr = last->first + (last->second - nbytes);
            last->second -= nbytes;
            return ptr;
        }
        // 2K 对齐：best-fit 从前面分配
        // ... 遍历 free list 找最佳匹配 ...
    }

    void Free(void* ptr) {
        // 归还到 free list，合并相邻空闲块
    }
};
```

实验输出：

```
[4/6] Pool allocator (TVM style)...
  Pool allocator (TVM style):
    alloc A: 64 KB  at +0x00000
    alloc B: 32 KB  at +0x10000
    alloc C: 128 KB at +0x18000
    blocks: 4
    free B → blocks: 4 (hole at +0x10000)
    alloc D: 16 KB  at +0x10000 (reuse B's hole)
    blocks: 5
    free all → blocks: 1 (coalesced)
    read/write: OK
```

### Part 4: DMA — DDR → VTCM 异步传输

Hexagon 有一个专用的 **UDMA（User DMA）引擎**，可以在后台搬运数据。CPU/HVX/HMX 不需要参与 — 这是实现计算与数据搬运重叠的关键硬件。

#### UDMA 的工作原理

UDMA 通过**描述符（descriptor）**驱动。程序填好一个内存中的描述符结构，然后用 `dmstart` 指令启动 DMA 引擎：

```c
// UDMA type0 描述符（线性传输）
typedef struct {
    void *next;           // 链表下一个描述符（NULL = 最后一个）
    unsigned int length:24;   // 传输字节数（最大 16 MB）
    unsigned int desctype:2;  // 0 = type0（线性）
    unsigned int dstcomp:1;   // 目标压缩
    unsigned int srccomp:1;   // 源压缩
    unsigned int dstbypass:1; // 1 = 绕过 cache
    unsigned int srcbypass:1; // 1 = 绕过 cache
    unsigned int order:1;     // 顺序保证
    unsigned int dstate:1;    // 0=未完成, 1=完成
    void *src;            // 源地址（DDR）
    void *dst;            // 目标地址（VTCM）
} __attribute__((aligned(64))) dma_desc_type0_t;
```

三个关键 intrinsic（来自 `hexagon_protos.h`）：

| Intrinsic | 作用 |
|---|---|
| `Q6_dmstart_A(desc)` | 启动 DMA 引擎处理描述符 |
| `Q6_R_dmwait()` | 阻塞等待 DMA 完成，返回状态 |
| `Q6_R_dmpoll()` | 非阻塞查询 DMA 状态 |

#### 我们的 DMA demo

```c
// 1. 分配 DDR 源 buffer，填充测试数据
uint8_t *ddr_buf = (uint8_t *)memalign(128, 64 * 1024);
for (int i = 0; i < 64 * 1024; i++) ddr_buf[i] = (uint8_t)(i & 0xFF);

// 2. 在 VTCM 中分配目标 buffer
uint8_t *vtcm_buf = vtcm_seq_alloc(&ptr, 64 * 1024);

// 3. 设置描述符
dma_desc_type0_t desc = {
    .next = NULL, .length = 64 * 1024, .desctype = 0,
    .srcbypass = 1,  // DDR 源绕过 cache（DMA 直接从物理内存读）
    .src = ddr_buf, .dst = vtcm_buf,
};

// 4. *** 关键步骤：flush DDR buffer 的 cache ***
//    DMA 引擎绕过 L2 cache 直接读物理内存，
//    如果 CPU 写的数据还在 cache 里没刷回，DMA 读到的是脏数据。
qurt_mem_cache_clean((qurt_addr_t)ddr_buf, 64 * 1024,
                     QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

// 5. 启动 DMA 并等待
Q6_dmstart_A((void *)&desc);
int status = Q6_R_dmwait();  // 返回 0 = 成功
```

实验输出：

```
[5/6] DMA: DDR -> VTCM async transfer...
  DMA transfer: DDR → VTCM, 64 KB
    dmwait status: 0 (0 = success)
    verify: OK
```

#### Cache 一致性陷阱

这是 DMA 编程中最常踩的坑：**DMA 引擎和 CPU 看到的内存视图不同**。

```
CPU 写数据 → 数据在 L2 cache 中 → DMA 读物理内存 → 读到旧数据！
```

解决方案：DMA 之前必须 flush cache：
```c
qurt_mem_cache_clean(addr, size, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
```

llama.cpp 的 `hex-dma.c` 中也做了同样的处理。VTCM 不经过 cache，所以 VTCM 作为目标不需要 flush。

#### DMA 流水线 benchmark

DMA 的真正价值不在于"搬数据更快"（阻塞式 DMA 和 memcpy 差不多），而在于 **DMA 传输期间 CPU/HVX 可以做别的计算**。

我们对比两种模式：
- **串行**: `memcpy(DDR→VTCM)` 然后 `HVX_compute()` — CPU 全程忙
- **重叠**: `dmstart(DDR→VTCM)` 然后 `HVX_compute()` 然后 `dmwait()` — DMA 和计算并行

```
[5/6] DMA: DDR -> VTCM async transfer...
  Pipeline benchmark: 64 KB x 200 iters
    memcpy only:      3464 us
    serial (memcpy+HVX): 3701 us
    overlap (DMA+HVX):   217 us
    speedup: 17.06x (serial/overlap)
```

17x 加速！这就是 llama.cpp 用 DMA 双缓冲的原因 — 数据搬运几乎可以完全隐藏在计算时间里。

### Part 5: HVX Benchmark — 一个反直觉的结果

```
[6/6] HVX vadd benchmark: VTCM vs DDR...
  HVX vadd: 256 KB x 500 iters
    DDR:  2118 us
    VTCM: 2115 us
    Speedup: 1.00x
    Verify: OK
```

**DDR 和 VTCM 速度一样？** 这不是 bug，而是一个重要的教学点：

**HVX vadd 是纯顺序访问**。L2 cache 的硬件预取对顺序读写极其有效，能保持接近峰值带宽。VTCM 的低延迟优势（1 周期 vs ~10 周期）在流式访问时被 L2 预取完全掩盖了。

VTCM 真正发挥作用的场景：

| 场景 | DDR (L2 cache) | VTCM | 差异原因 |
|---|---|---|---|
| HVX 顺序读写 | ★★★ | ★★★ | L2 预取很强 |
| HVX 随机访问 | ★☆☆ | ★★★ | cache miss 代价高 |
| HMX matmul | ✗ 不支持 | ★★★ | HMX **必须**用 VTCM |
| DMA + 计算重叠 | ✗ | ★★★ | VTCM 支持 DMA 目标 |

**llama.cpp 用 VTCM 不是为了让 HVX 更快，而是因为 HMX 矩阵乘别无选择。** 没有 VTCM，HMX 根本无法工作。

### Part 6: 完整流水线 — DMA + HVX + HMX

现在我们已经分别了解了 VTCM 分配、DMA 传输、HVX 计算。llama.cpp 把它们编排成一条 **4 阶段流水线**，这是整个 HMX matmul 的核心：

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐
│ Stage 1  │    │ Stage 2   │    │ Stage 3   │    │ Stage 4  │
│ DMA      │ →  │ HVX       │ →  │ HMX       │ →  │ HVX/DMA  │
│ DDR→VTCM │    │ dequant   │    │ matmul    │    │ VTCM→DDR │
│          │    │ Q4→FP16   │    │ FP16 ACC  │    │          │
└─────────┘    └──────────┘    └──────────┘    └─────────┘
```

#### Stage 1: DMA — 从 DDR 搬运量化权重到 VTCM

```c
// llama.cpp hex-dma.c（简化）
// 把 DDR 中的 Q4_0 量化权重搬到 VTCM scratch buffer
dma_queue_push_chained(ctx->dma[0], vtcm_scratch, ddr_weight_ptr, chunk_size);
```

权重存储在 DDR 中（模型文件 mmap 到内存），大小通常是 GB 级别，远超 VTCM 容量。所以必须分块搬运 — 每次搬一个 chunk（几十 KB），计算完再搬下一个。

#### Stage 2: HVX — 反量化 + 重排

```c
// llama.cpp hmx-matmul-ops.c（简化）
// HVX 把 Q4/Q8 量化数据解包成 FP16，并重排成 HMX 需要的 tile 格式
dequantize_x4x2_weight_chunk_to_fp16_tiles(
    vtcm_scratch,      // 源：DMA 搬来的量化数据
    vtcm_weight,       // 目标：HMX 可以直接读的 FP16 tile
    scales, zeros,     // 量化参数
    chunk_rows, chunk_cols
);
```

HMX 要求输入数据按照特定的 tile 布局排列（类似 GPU 的 shared memory bank 布局）。HVX 的 shuffle/deal 指令非常适合做这种数据重排。

#### Stage 3: HMX — 矩阵乘加

```c
// llama.cpp hmx-matmul-ops.c（简化）
// HMX 在 VTCM 上做 FP16 矩阵乘
core_mma_chunk_fp16(
    vtcm_weight,       // 权重 tile（VTCM）
    vtcm_activation,   // 激活 tile（VTCM）
    vtcm_output,       // 输出累加（VTCM）
    M, N, K_chunk
);
```

HMX 的所有输入和输出**必须在 VTCM 中** — 这就是为什么前面要用 DMA 搬数据、用 bump allocator 分配 VTCM 空间。

#### Stage 4: 写回 DDR

计算完成后，结果从 VTCM 写回 DDR（可以用 DMA 或直接 store）。

#### 双缓冲 — 隐藏 DMA 延迟

llama.cpp 用**两个 scratch buffer**（`vtcm_scratch0` 和 `vtcm_scratch1`）实现 ping-pong：

```c
// llama.cpp 双缓冲模式（概念代码）
for (int chunk = 0; chunk < n_chunks; chunk++) {
    int cur = chunk & 1;       // 当前 buffer: 0 或 1
    int nxt = (chunk + 1) & 1; // 下一个 buffer

    // 启动下一个 chunk 的 DMA（不阻塞）
    if (chunk + 1 < n_chunks)
        dma_start(scratch[nxt], ddr_weight + (chunk+1) * size, size);

    // 处理当前 chunk（和 DMA 并行）
    hvx_dequant(scratch[cur], weight_tile);
    hmx_compute(weight_tile, activation, output);

    // 等上一步 DMA 完成
    if (chunk + 1 < n_chunks)
        dma_wait();
}
```

这就是 Part 4 的 DMA benchmark 展示的效果：DMA 传输隐藏在计算时间里。实测 17x 加速说明 64KB 的 DMA 传输时间远小于 HVX 计算时间，几乎完全被隐藏。

#### VTCM 布局总览

把 bump allocator（Part 2）和流水线结合起来看：

```
VTCM 8 MB
┌──────────────────────────────────────────────────┐
│ weight tiles (FP16)       [vtcm_weight]          │ ← HMX 读
├──────────────────────────────────────────────────┤
│ activation tiles (FP16)   [vtcm_activation]      │ ← HMX 读
├──────────────────────────────────────────────────┤
│ output accumulators       [vtcm_output]          │ ← HMX 写
├──────────────────────────────────────────────────┤
│ scratch buffer 0          [vtcm_scratch0]        │ ← DMA 写，HVX 读
├──────────────────────────────────────────────────┤
│ scratch buffer 1          [vtcm_scratch1]        │ ← DMA 写，HVX 读
├──────────────────────────────────────────────────┤
│ scales/metadata           [vtcm_scales]          │
├──────────────────────────────────────────────────┤
│ unused (~97%)                                    │
└──────────────────────────────────────────────────┘
```

每个 matmul op 开始时 bump allocator 重置指针，重新划分。不同大小的矩阵用不同的布局，但模式固定。

### Part 7: VTCM 抢占 — 为什么 VTCM 数据会"消失"

在 ch09 的 VTCM 训练中，我们发现了一个关键陷阱：**训练数据放在 VTCM 中跨消息复用时，如果 DSP 空闲几秒钟，VTCM 中的权重会被随机覆盖。**

#### 现象

前几个 epoch 训练正常（96% 准确率），但在 epoch 间让 DSP 空闲（ARM 侧做 cpu_evaluate 约 3-5 秒）后，loss 突然飙升，准确率跌到随机水平（9.8%）。

#### 决定性实验

| 配置 | DSP 做什么 | ARM 做什么 | DSP 空闲? | 结果 |
|------|-----------|-----------|-----------|------|
| A: NOOP + eval | 什么都不做 | cpu_evaluate 数秒 | **是** | **Epoch 3 权重损坏** |
| B: memcpy + busy | 400KB memcpy + 50次 dummy matmul | 什么都不做 | **否** | **5 epoch 全部正常** |

配置 A 中 DSP 没有碰 VTCM 的任何数据，权重却被覆盖 — 证明是**外部写入**。配置 B 做了完整 memcpy 但保持 DSP 忙碌，VTCM 完好无损。

#### 根因

`HAP_compute_res_acquire` 获取的 VTCM 并非独占锁。在 Android 设备上，camera、audio、NN 服务等多个 DSP 客户端共享 VTCM。当我们的 DSP 线程空闲（阻塞在 dspqueue_read 等待消息）时，系统调度器可能将 VTCM 分配给其他高优先级客户端。

```
安全:  DSP 持续处理消息（<100ms 间隔）  ✅ VTCM 不被抢
危险:  DSP 空闲数秒（ARM 做大量计算）   ❌ VTCM 被其他客户端覆盖
```

#### 解决方案

1. **在 DSP 上做评估**（推荐）— 将测试数据发送到 DSP，用 VTCM 权重做推理，不让 DSP 空闲
2. **DSP 侧 keepalive** — ARM 做计算期间给 DSP 发心跳消息保持活跃
3. **接受限制** — 不在训练中途导出权重，只在结束后导出

llama.cpp 和 htp-ops-lib 不受影响，因为它们的 DSP 线程始终忙碌（连续消息流）。

### Part 8: QNN 的 VTCM 调度

QNN 如何自动调度 VTCM？见第七章的实验。

### Part 9: VTCM 共享 — 多进程协作

VTCM 共享和多图并行执行？见第七章的实验 3。

## 三种 VTCM 管理策略对比

| | SDK 基础用法 | llama.cpp (bump) | TVM (pool) |
|---|---|---|---|
| **分配器** | 一次 acquire/release | bump allocator (vtcm_seq_alloc) | HexagonVtcmPool |
| **代码量** | ~10 行 | ~20 行 | ~300+ 行 |
| **碎片化** | N/A（整块使用） | 零（每 op 重置） | 低（合并策略） |
| **需要 free?** | 是 | 否（重置指针） | 是 |
| **线程安全** | N/A | 不需要（单线程） | mutex 保护 |
| **DMA 集成** | 无 | 手动 DMA + 双缓冲 | HexagonUserDMA |
| **HMX 支持** | 需手动管理 | 手动 tile 布局 | 编译器自动 tile |
| **适用场景** | 单次计算 | 固定 pipeline (LLM) | 动态 graph / 通用推理 |

QNN 的 VTCM 调度策略（编译器自动分配、spill/fill、VTCM 共享）见第七章。

## 总结

| 章节 | 主题 | 运行环境 | 关键收获 |
|---|---|---|---|
| ch01 | 模拟器 HVX/HMX | x86 模拟器 | Hexagon 基本架构 |
| ch02 | 真机 HVX/HMX | CDSP (run_main) | HAP API, VTCM 分配 |
| ch03 | QNN 自定义算子 | CDSP (QNN) | QNN 框架集成 |
| **ch04** | **VTCM + DMA + 流水线** | **CDSP (run_main)** | **内存管理/DMA/HMX 流水线/VTCM 抢占** |
| ch05 | dspqueue vs FastRPC | CDSP (dspqueue) | 通信开销, llama.cpp 架构 |

下一章（ch05）将通过 dspqueue 实验探索 ARM-DSP 通信优化。
