# 第10章：VTCM 实验室 — VTCM 资源抢占问题定位

## 背景

ch09 尝试将训练数据（权重、梯度、激活值）全部保留在 VTCM 中跨 dspqueue 消息复用。前几个 epoch 训练完全正常，但在 epoch 之间让 DSP 空闲（ARM 侧做 cpu_evaluate）后，VTCM 中的权重出现非确定性 corruption — loss 突然飙升，准确率跌到随机水平（9.8%）。

本章是一个系统性排查实验室，目标是逐步隔离 corruption 的根因。

## 实验设计

### 第一阶段：基础验证（Exp 1-5）

| # | 实验 | 内容 | 结果 |
|---|------|------|------|
| 1 | VTCM Basic | Scalar 写 -> scalar 读，100 轮 x 16K floats | PASS |
| 2 | VTCM Persistence | 消息 N 写 magic，消息 N+1 验证，10,000 条消息 | PASS |
| 3 | HVX Write / Scalar Read | HVX vector store -> scalar load，1000 轮 | PASS |
| 4 | Scalar Write / HVX Read | Scalar store -> HVX vector load，1000 轮 | PASS |
| 5 | HVX SGD | 5000 次 `w -= lr * grad`（HVX qf32），然后 memcpy VTCM->DDR 验证 | PASS |

结论：VTCM 物理可访问、跨消息持久、HVX/scalar 一致、qf32 计算正确。

### 第二阶段：L2 Cache 假说验证（Exp 7-11）

初始怀疑是 memcpy(VTCM->DDR) 导致 L2 cache 持有过期数据。以下实验否定了这一假说：

| # | 实验 | 结果 | 说明 |
|---|------|------|------|
| 7 | L2 stale test (HVX写→memcpy→HVX写→scalar读) | PASS | memcpy 不导致 L2 stale |
| 8 | Cache invalidation test | PASS | invalidation 正常工作 |
| 10a | HVX increment (1000轮, 无memcpy) | PASS | baseline |
| 10b | HVX increment + memcpy + INVALIDATE | PASS | memcpy + cache ops 无影响 |
| 10c | HVX increment + memcpy + FLUSH_INVALIDATE | PASS | 同上 |
| 10d | HVX increment + memcpy, 无cache ops | PASS | memcpy 本身也无影响 |
| 11 | Scalar write → INVALIDATE → scalar read | PASS | invalidation 对 VTCM 有效 |

结论：**L2 cache 假说被否定。** memcpy、cache invalidation 对 VTCM 数据都没有负面影响。

### 第三阶段：训练 SYNC 变体对比（Exp 12）

| 变体 | DSP 侧操作 | ARM 侧操作 | 结果 |
|------|-----------|-----------|------|
| NOOP (无memcpy, 无eval) | 只读 w1[0] | 不做任何事 | 10 epoch PASS |
| SMALL (16KB memcpy + eval) | 拷贝 w2 到 heap | cpu_evaluate (~数秒) | Epoch 3 CORRUPT |
| LARGE (400KB memcpy + eval) | 拷贝 w1+w2 到 heap | cpu_evaluate (~数秒) | Epoch 3 CORRUPT |

关键发现：SMALL 变体只拷贝了 w2 (16KB)，但 **w1 被损坏**（值变为 22,271,834）。w1 从未被 memcpy 读取。如果是 L2 cache 问题，w1 不可能被影响。这是外部写入。

### 第四阶段：决定性实验（Exp 13）

隔离 memcpy 和 DSP 空闲时间的各自影响：

| 变体 | DSP 侧操作 | ARM 侧操作 | DSP 空闲? | 结果 |
|------|-----------|-----------|-----------|------|
| NOOP + cpu_evaluate | 无 memcpy | cpu_evaluate 数秒 | **是** | **Epoch 3 CORRUPT** |
| MCONLY + DSP busy-work | 400KB memcpy + 50次 matmul | 无 | **否** | **5 epoch PASS** |

**决定性结论：**
- NOOP + cpu_evaluate：DSP **没有任何 memcpy**，但因 ARM 做 cpu_evaluate 导致 DSP 空闲数秒 → **VTCM 被损坏**
- MCONLY + DSP busy-work：DSP **做了完整 400KB memcpy**，但随后用 50 次 dummy matmul 保持忙碌 → **VTCM 完好**

**不是 memcpy 的问题，是 DSP 空闲时间的问题。**

## 根因：DSP 空闲时 VTCM 资源被系统抢占

这是本章最重要的发现。

### 机制

在真实 Android 设备上，多个 DSP 客户端（camera、audio、neural network 服务等）共享 VTCM 资源。`HAP_compute_res_acquire` 获取的 VTCM 并非独占锁 — 当我们的 DSP 线程空闲（等待下一条 dspqueue 消息）时，系统调度器可能将 VTCM 分配给其他高优先级客户端。

```
正常训练（DSP 持续忙碌）:
  batch 1 → batch 2 → ... → batch 468 → SYNC → batch 1 → ...
  DSP 线程始终在处理消息，VTCM 被"占用中"         ✅ 不会被抢

有 cpu_evaluate 的训练（DSP 空闲数秒）:
  batch 468 → SYNC → [DSP 空闲] → [ARM 做 cpu_evaluate 3-5秒] → batch 1
                      ↑ 其他 DSP 客户端趁机写入我们的 VTCM 区域  ❌
```

### 证据

1. **w1 从未被 SYNC 读取，却被覆盖** — SMALL SYNC 只拷贝 w2，w1 不可能因 memcpy 或 L2 而损坏
2. **损坏值是随机大数（22M、0）交替** — 不是旧权重值，是完全无关的数据，说明是外部写入
3. **所有短时单元测试 PASS** — 1000 轮 increment（每轮 <1ms）全部正确，因为 DSP 始终忙碌
4. **NOOP 不做 cpu_evaluate 就 PASS** — DSP 空闲时间极短（微秒级），来不及被抢

### 推论

- `HAP_compute_res_acquire` 只是一个"请求"，不是硬件级别的互斥锁
- 当 DSP 线程不活跃时（阻塞在 dspqueue_read），系统可能认为 VTCM 可被复用
- llama.cpp 和 htp-ops-lib 不受影响，因为它们的 DSP 线程始终忙碌（连续消息流或 poller thread）

### 安全模式

```
安全:  DSP 持续处理 dspqueue 消息（<100ms 间隔）  ✅ VTCM 不被抢
危险:  DSP 空闲数秒（ARM 做大量计算）             ❌ VTCM 被其他客户端覆盖
```

### 解决方案

1. **在 DSP 上做评估**（推荐）— 将测试数据发送到 DSP，用 VTCM 权重推理，不让 DSP 空闲
2. **DSP 侧 keepalive** — 在 ARM 做 cpu_evaluate 期间，给 DSP 发心跳消息保持活跃
3. **接受限制** — 不在训练中途导出权重，只在结束后导出

## 文件结构

```
ch10-vtcm-lab/
├── build.sh                    # 构建脚本（复用 ch08 的 IDL stub）
├── run_device.sh               # 推送到设备并运行
├── src/
│   ├── arm/test_vtcm.c         # ARM 侧测试驱动
│   ├── common/lab_protocol.h   # 共享 opcode 和消息结构
│   └── dsp/skel_lab.c          # DSP 侧：VTCM 分配、全部实验实现
└── data/                       # MNIST 数据文件
```

## 构建和运行

```bash
bash build.sh
bash run_device.sh
```
