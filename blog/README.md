# 手机 NPU 的真相：34 TOPS 去哪了？

**一句话**：NPU 的算力从来不是瓶颈，**数据搬运**才是。这篇系列文章用真机实验数据，带你看清 NPU 推理的三道墙。

**目标读者**：知道 PyTorch、听说过 NPU、但被 QNN/HTP/VTCM/HMX 这些术语搞晕的开发者。

---

## 目录

| 章节 | 标题 | 核心问题 |
|------|------|---------|
| [第一章](ch01-opening.md) | 34 TOPS 去哪了？ | NPU 利用率不到 0.1%，问题出在哪？技术栈全景图 |
| [第二章](ch02-format-wall.md) | 格式墙——NPU 只吃一种菜 | HMX 只认 32×32 瓦片，格式转换占 60% 时间 |
| [第三章](ch03-memory-wall.md) | 内存墙——8MB 要装 14GB | DMA 流水线 17 倍加速，QNN 本质是排搬运课表 |
| [第四章](ch04-communication-wall.md) | 通信墙——CPU 和 NPU 隔着一道远程调用 | FastRPC 364μs vs dspqueue 61μs，三条路线的抉择 |
| [第五章](ch05-kv-cache.md) | KV Cache——三墙汇合之处 | 唯一的"活权重"，NativeKV 为什么需要新硬件 |
| [第六章](ch06-conclusion.md) | llama.cpp——一个人如何喂饱 NPU | 19000 行代码不用任何高通框架，从零解决三道墙 |

---

*所有数据来自骁龙 8 Gen 3 真机测试。*
