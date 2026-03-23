# 第五章：KV Cache——三墙汇合之处

前几章我们分别拆解了格式墙（第二章）、内存墙（第三章）和通信墙（第四章）。每道墙都有成熟的解法：权重在模型加载时一次性转成 WH Layout（格式墙），DMA 流水线隐藏搬运延迟（内存墙），算子打包和 dspqueue 减少 RPC 开销（通信墙）。

这些解法有一个共同的前提：**权重是静态的**。模型加载完成后，所有权重的大小、位置、格式都已确定，搬运时刻表可以提前排好。

LLM 的 Attention 层打破了这个前提。KV Cache 是整个推理流程中唯一的"活权重"——它随着每个 token 的生成而增长。本章将展示，为什么这个看似简单的事实，让前面所有优化假设同时失效。

---

## 1. 三墙同时命中

Attention 的核心运算是 `scores = Q x K^T`。Q 是当前 token 的查询向量（固定大小），K 是之前所有 token 的 Key 向量拼成的矩阵（KV Cache 的一部分）。每生成一个新 token，K 就多一行。

这件事为什么致命？因为它同时触发三道墙：

**格式墙**：HMX 要求 weight 必须是 WH Layout。K 每步都变了，WH Layout 要重新转换——而且 K 越来越大，转换成本线性增长。

**内存墙**：K 的大小在推理过程中不断变化，VTCM 的搬运计划（哪些 tile 先搬、放在哪个地址）无法在 Finalize 阶段提前排定。

**通信墙**：K 的更新（写入新 token）和使用（做矩阵乘法）跨越多次 DSP 调用，每次调用都需要同步状态。

前三章的每道墙，我们都找到了绕过它的方法。KV Cache 的问题在于，三条绕行路线在这里汇合成了一个死胡同。

---

## 2. 旧方案：SmartMask——每步都重新转换

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

## 3. 新方案：NativeKV——K 生来就是瓦片格式

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

## 4. 32 对齐约束与 MaskedSoftmax

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

## 5. 为什么需要新硬件（v75+）

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

## 6. Genie 的长上下文策略：多图变体 + 分块预填充

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

## 7. llama.cpp 的另一条路：不转格式，DMA 流式计算

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

**第二，DMA 双缓冲**。和第三章讲的 DMA 流水线一样——搬第 i+1 块的数据和算第 i 块的结果并行进行。KV cache 虽然在 DDR 里，但每个 block 只有 64 个 token × head_dim × 2 bytes，DMA 搬运速度远快于计算速度，搬运延迟被完全隐藏。

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

## 8. 更大的图景：NPU 演进的方向

回顾这五章，NPU 每一代的核心进步都不是"更多 TOPS"，而是减少一道墙：

| 代际 | 关键改进 | 解决的墙 |
|------|---------|---------|
| v68 → v73 | 加入 HMX 矩阵单元 | 算力墙（从向量到矩阵） |
| v73 → v75 | VTCM 扩大到 8MB | 内存墙 |
| v73 → v75 | dspqueue 替代 FastRPC（61us vs 364us） | 通信墙 |
| v73 → v75 | HMX Weight Layout 作为 I/O 格式 | **格式墙（KV Cache 专项）** |

34 TOPS 的算力从第一天起就在那里。五章看下来，真正的工程战场从来不在乘法器上，而在乘法器前面那条越来越短、越来越灵活的数据通路上。理解了这一点，你就能判断下一代芯片的宣传里，哪些参数是真正的进步，哪些只是数字游戏。

最后一章，我们把五章积累的所有术语放回各自的位置，给出选框架、选硬件、设计模型时的实用决策指南。

---

*所有数据来自骁龙 8 Gen 3 真机测试。*
