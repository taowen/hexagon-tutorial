# 第七章：QNN VTCM 管理 — 从配置到 Profiling 到多图共享

第六章我们手动管理 VTCM：`HAP_compute_res_acquire` 分配、bump allocator 分区、UDMA 搬运。这很底层，适合 llama.cpp 这种绕过框架的场景。

但在生产中，大多数 AI 推理用 QNN 框架。QNN 自动管理 VTCM——你只需要告诉它用多少、怎么分。本章通过三个实验，观察 QNN 的 VTCM 管理机制。

## 和前面章节的关系

| 章节 | VTCM 管理方式 |
|------|-------------|
| ch02 | `HAP_compute_res_acquire` — 手动分配整块 VTCM |
| ch03 | `QHPI_MemLoc_TCM_Only` — 声明 tensor 放 VTCM，QNN 自动调度 |
| ch06 | bump/pool allocator + UDMA — 手动分区 + 手动搬运 |
| **ch07** | **QNN 配置 API + Profiling — 控制和观察 QNN 的自动调度** |

## 源码结构

```
ch07-qnn-vtcm/
├── build.sh           # 编译 DSP 算子包 + ARM fallback + host 测试程序
├── run_device.sh      # 推送 + 运行
└── src/
    ├── dsp/
    │   ├── HvxHmxInterface.cpp  # QNN OpPackage 接口 + QHPI 入口
    │   └── HvxHmxOp.cpp         # HVX ReLU + HMX matmul 内核
    └── host/
        └── qnn_vtcm_test.c      # 三个实验的完整代码
```

本章自带 HvxHmxMix 自定义算子的完整源码（与 ch03 相同），无需依赖其他章节。重点在 host 端的 QNN 配置和 profiling，不在 DSP 内核。

## 编译运行

```bash
# 编译 ch07（包含 DSP 算子包 + host 测试程序）
bash ch07-qnn-vtcm/build.sh

# 推送到真机运行
bash ch07-qnn-vtcm/run_device.sh
```

---

## 实验 1+2: VTCM 大小配置 + QNN Profiling

### QNN 怎么配置 VTCM 大小

QNN 通过 `QnnHtpGraph_CustomConfig_t` 在建图时指定 VTCM 大小：

```c
/* 设置图的 VTCM 大小 */
QnnHtpGraph_CustomConfig_t vtcmCfg;
memset(&vtcmCfg, 0, sizeof(vtcmCfg));
vtcmCfg.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE_IN_MB;
vtcmCfg.vtcmSizeInMB = 4;  // 使用 4MB VTCM（设 0 = 使用最大可用）

/* 包装成 QnnGraph_Config_t */
QnnGraph_Config_t graphCfg;
graphCfg.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
graphCfg.customConfig = &vtcmCfg;

const QnnGraph_Config_t* cfgs[] = {&graphCfg, NULL};
g_qnn.graphCreate(ctx, "my_graph", cfgs, &graph);
```

可选的 VTCM 配置：

| 选项 | 值 | 说明 |
|------|-----|------|
| `QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE_IN_MB` | `uint32_t` | 以 MB 为单位 |
| `QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE_IN_BYTES` | `uint32_t` | 以字节为单位（更精确） |
| `QNN_HTP_GRAPH_CONFIG_OPTION_MAX` (值=0) | — | 使用硬件最大 VTCM |

### QNN 怎么做 Profiling

QNN 的 profiling 分两步：

```c
/* 1. 创建 profile handle */
Qnn_ProfileHandle_t profile = NULL;
g_qnn.profileCreate(backend, QNN_PROFILE_LEVEL_DETAILED, &profile);

/* 2. 传入 graphFinalize / graphExecute */
g_qnn.graphFinalize(graph, profile, NULL);   // 收集 finalize 阶段数据
g_qnn.graphExecute(graph, ins, 3, outs, 1, profile, NULL);  // 收集 execute 阶段数据

/* 3. 读取事件 */
const QnnProfile_EventId_t* events;
uint32_t numEvents;
g_qnn.profileGetEvents(profile, &events, &numEvents);

for (uint32_t i = 0; i < numEvents; i++) {
    QnnProfile_EventData_t data;
    g_qnn.profileGetEventData(events[i], &data);
    printf("type=%u value=%lu %s\n", data.type, data.value, data.identifier);

    /* 递归读取 sub-events */
    const QnnProfile_EventId_t* subEvents;
    uint32_t numSub;
    g_qnn.profileGetSubEvents(events[i], &subEvents, &numSub);
    // ...
}
```

Profile level:
- `QNN_PROFILE_LEVEL_BASIC` — 只有总时间
- `QNN_PROFILE_LEVEL_DETAILED` — 包含每个 op 和每个阶段的子事件

### 实验设计

构建一条 **op 链**（1/4/8 个 HvxHmxMix 算子串联），测试 3 种 VTCM 配置（MAX/1MB/4MB），观察：
- finalize 阶段的时间分布（特别是 VTCM Allocation 阶段）
- execute 延迟

```
act → [HvxHmxMix] → mid0 → [HvxHmxMix] → mid1 → ... → output
          ↑                      ↑
        weight                 weight (共享)
```

### 真机数据（Snapdragon 8 Gen 3, v75, 8MB VTCM）

#### Finalize 阶段分布

| 配置 | Graph Prep | Optimization | Sequencing | **VTCM Alloc** | Total |
|------|-----------|-------------|-----------|---------------|-------|
| MAX, chain=1 | 131 us | 959 us | 283 us | **89 us** | 7,455 us |
| MAX, chain=4 | 1,511 us | 6,623 us | 2,574 us | **122 us** | 20,630 us |
| MAX, chain=8 | 558 us | 3,477 us | 1,618 us | **101 us** | 14,242 us |
| **1MB, chain=4** | 1,341 us | 6,276 us | 2,307 us | **563 us** | 20,004 us |
| 4MB, chain=8 | 1,163 us | 4,571 us | 1,486 us | **208 us** | 16,339 us |

关键观察：

**VTCM 分配时间随约束增大而增加**。chain=4 时：
- vtcm=MAX: VTCM Allocation = **122 us**
- vtcm=1MB: VTCM Allocation = **563 us**（4.6 倍）

当 VTCM 充裕时，分配器可以大方地给每个 tensor 独占空间。当 VTCM 受限时，分配器必须做更复杂的计算：哪些 tensor 生命周期不重叠可以复用同一地址？哪些需要 spill 到 DDR？这就是 ch06 Part 7 描述的 QNN 五阶段调度流程的实际开销。

#### Execute 延迟

| 配置 | chain=1 | chain=4 | chain=8 |
|------|---------|---------|---------|
| vtcm=MAX | 616 us | 826 us | 525 us |
| vtcm=1MB | 505 us | 825 us | 598 us |
| vtcm=4MB | 529 us | 817 us | 546 us |

执行延迟几乎没有差异。原因：我们的 32×32 矩阵太小了（每个 tensor 只有 2KB），即使只给 1MB VTCM 也绰绰有余，不会触发 spill/fill。

**在真实模型中**（如 llama.cpp 的 4096×4096 矩阵），VTCM 大小会显著影响性能。8MB VTCM 可以放约 16 个 32×32 FP16 tiles 的 HMX 分区，但如果限制到 1MB，调度器就必须插入 spill/fill 节点，增加 DDR 访问延迟。

#### Profiling 事件类型

| 事件类型 | 值 | 说明 |
|---------|-----|------|
| `FINALIZE` (300) | 总 finalize 时间 | 包含所有子阶段 |
| 子事件: Graph Preparation Initializing | 构建内部表示 | |
| 子事件: Graph Optimizations | 算子融合、布局转换 | |
| 子事件: Graph Sequencing for Target | 按依赖关系排序 | |
| 子事件: **VTCM Allocation** | **VTCM 地址分配** | ch06 Part 7 描述的五阶段 |
| 子事件: Parallelization Optimization | HVX 线程分配 | |
| `EXECUTE` (400) | 总 execute 时间 | |
| `HTP_NUM_HVX_THREADS` (8001) | 使用的 HVX 线程数 | 本实验 = 4 |

---

## 实验 3: 多图 VTCM 共享

### 为什么需要 VTCM 共享

在手机上，CDSP 可能同时运行多个 AI 模型：语音识别 + 相机 ISP + 手势检测。每个模型都想要整块 VTCM。QNN 提供了 **Parallel Graph Execution Config** 让多个图分区共享 VTCM。

### 怎么配置

```c
/* Graph A: 使用 VTCM [0, 4MB) */
QnnHtpGraph_CustomConfig_t cfgA;
cfgA.option = QNN_HTP_GRAPH_CONFIG_OPTION_PARALLEL_GRAPH_EXECUTION_CONFIG;
cfgA.parallelGraphExecutionConfig.concurrency = 0;  /* ALL_SHARED */
cfgA.parallelGraphExecutionConfig.vtcmConfig.sizeInBytes     = 4 * 1024 * 1024;
cfgA.parallelGraphExecutionConfig.vtcmConfig.offsetInBytes   = 0;
cfgA.parallelGraphExecutionConfig.vtcmConfig.sizeTotalInBytes = 8 * 1024 * 1024;

/* Graph B: 使用 VTCM [4MB, 8MB) */
QnnHtpGraph_CustomConfig_t cfgB;
cfgB.option = QNN_HTP_GRAPH_CONFIG_OPTION_PARALLEL_GRAPH_EXECUTION_CONFIG;
cfgB.parallelGraphExecutionConfig.concurrency = 0;
cfgB.parallelGraphExecutionConfig.vtcmConfig.sizeInBytes     = 4 * 1024 * 1024;
cfgB.parallelGraphExecutionConfig.vtcmConfig.offsetInBytes   = 4 * 1024 * 1024;
cfgB.parallelGraphExecutionConfig.vtcmConfig.sizeTotalInBytes = 8 * 1024 * 1024;
```

`VtcmConfig` 的三个字段：

| 字段 | 说明 |
|------|------|
| `sizeInBytes` | 本图实际使用的 VTCM 量 |
| `offsetInBytes` | 本图在 VTCM 中的起始偏移 |
| `sizeTotalInBytes` | 可寻址的 VTCM 总量（通常 = 硬件 VTCM 大小） |

```
VTCM 8MB
┌──────────────────┬──────────────────┐
│   Graph A        │   Graph B        │
│   [0, 4MB)       │   [4MB, 8MB)     │
│                  │                  │
│   weights +      │   weights +      │
│   activations +  │   activations +  │
│   intermediates  │   intermediates  │
└──────────────────┴──────────────────┘
```

### 真机数据

```
graphCreate(graph_a, vtcm=[0, 4MB)):  OK
graphCreate(graph_b, vtcm=[4MB, 8MB)): OK
graphFinalize(graph_a): OK
graphFinalize(graph_b): OK

Sequential (A+B): 1532 us/pair (avg over 100)
graph_a out[0]=32.0  graph_b out[0]=32.0
```

两个图成功创建并执行，各自使用 VTCM 的一半。如果不做分区配置，两个图会争抢整块 VTCM，QNN 运行时需要在切换图时保存/恢复 VTCM 内容（类似操作系统的上下文切换）。分区配置消除了这个开销。

---

## QNN VTCM 管理的完整流程

结合 ch06 的分析和 ch07 的 profiling 数据，QNN 的 VTCM 管理可以分为五个阶段（对应 finalize 的子事件）：

### 阶段 1: Graph Sequencing for Target

按依赖关系对 op 排序。确定哪些 tensor 的生命周期重叠（同时需要在 VTCM 中）。

### 阶段 2: VTCM Allocation

这是核心。QNN 的 VTCM 分配器做两件事：

1. **偏移分配**：给每个 TCM tensor 分配 VTCM 地址偏移。生命周期不重叠的 tensor 可以复用同一偏移（类似寄存器分配中的 liveness analysis）。

2. **Spill/Fill 插入**：如果 VTCM 不够大（所有同时存活的 tensor 总大小 > 可用 VTCM），插入 spill 节点（VTCM→DDR）和 fill 节点（DDR→VTCM）。

这解释了为什么 vtcm=1MB, chain=4 的 VTCM Allocation 耗时是 vtcm=MAX 的 4.6 倍——分配器需要更精细的 liveness 分析。

### 阶段 3: Parallelization Optimization

决定 HVX 线程数和工作分配。本实验总是 4 个 HVX 线程。

### 阶段 4: Finalizing Graph Sequence

生成最终的执行计划，包含所有 spill/fill 节点。

### 阶段 5: 运行时执行

按执行计划运行 op。对于每个 op：
1. fill：DMA 把数据从 DDR 搬到 VTCM（如果需要）
2. compute：HVX/HMX 在 VTCM 上计算
3. spill：DMA 把结果从 VTCM 搬到 DDR（如果需要）

---

## 和 llama.cpp 的对比

| | QNN (ch07) | llama.cpp (ch05/ch06) |
|---|---|---|
| **VTCM 分配** | 框架自动（finalize 时静态分配） | 手动 bump allocator（运行时动态） |
| **VTCM 大小** | 通过 config API 配置 | 硬编码 `HAP_compute_res_acquire` 全量 |
| **Spill/Fill** | 框架自动插入节点 | 手动 DMA queue |
| **多模型共存** | Parallel Graph Config 分区 | 不支持（独占 VTCM） |
| **Profiling** | 内置 API，可观察每个阶段 | 无（需要手动计时） |
| **灵活性** | 受限于框架支持的 op | 完全自由 |
| **性能** | 经过自动优化（融合、并行） | 手工优化（可能更好也可能更差） |

QNN 的优势是**自动化和可观测性**——你不需要手写 DMA 描述符和 VTCM 分区逻辑，框架帮你做。代价是你失去了对执行细节的完全控制。

llama.cpp 选择绕过 QNN 直接操作 VTCM/DMA/HMX，是因为 LLM 推理的 KV cache 管理和动态 batch size 不适合 QNN 的静态图模型。

---

## 本章小结

| 学到了什么 | 关键 API/概念 |
|-----------|-------------|
| 配置 VTCM 大小 | `QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE_IN_MB` |
| QNN Profiling | `profileCreate` → `graphFinalize(profile)` → `profileGetEvents` |
| Finalize 流程 | Prep → Optimization → Sequencing → **VTCM Allocation** → Parallelization |
| VTCM 分配开销 | 1MB VTCM 的分配时间是 MAX 的 4.6 倍（chain=4） |
| 多图 VTCM 共享 | `QNN_HTP_GRAPH_CONFIG_OPTION_PARALLEL_GRAPH_EXECUTION_CONFIG` |
| QNN vs 手动 | QNN 自动但受限，手动灵活但复杂 |
