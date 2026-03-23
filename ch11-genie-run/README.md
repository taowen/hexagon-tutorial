# 第十一章：用 Genie 跑 LLM 推理

上一章我们用 llama.cpp 的 Hexagon 后端在 NPU 上跑了 LLM。那是社区的路径——从 HVX/HMX 指令一路搭上来，每个 matmul 一次 dspqueue 调用。本章走 Qualcomm 的"官方"路径：Genie SDK，看看原厂优化能快多少。

## Genie 是什么

Genie 全称 **Gen AI Inference Extensions**，是 Qualcomm 官方的 LLM 推理运行时，包含在 QAIRT (Qualcomm AI Runtime) SDK 中。它建在 QNN 之上，封装了 tokenizer、KV cache 管理、采样逻辑——也就是说，你不需要自己实现 top-k/top-p，不需要管 KV cache 的内存分配，只要喂一个 prompt 就能拿到文本输出。

关键对比：

| | llama.cpp Hexagon 后端 | Genie |
|---|---|---|
| **模型格式** | GGUF | QNN context binary (.bin) |
| **计算路径** | 逐算子下发 dspqueue | QNN HTP 图级优化 |
| **KV Cache** | 手动管理（ch09 NativeKV） | QNN 内部管理 |
| **优化方** | 开源社区 | Qualcomm 内部团队 |
| **灵活性** | 高，任意 GGUF 模型 | 低，必须经过 QNN 编译 |

Genie 提供两种使用方式：
- **genie-t2t-run**：命令行工具，text-to-text，直接跑推理
- **C/C++ API**：`GenieDialog`、`GenieEngine`，集成到自己的应用中

---

## 准备模型

Genie 不吃 GGUF。它需要 **QNN context binary**——模型经过 QNN 编译器优化后的二进制格式，包含算子融合、内存规划、HTP 指令调度等所有优化结果。

获取 context binary 有三种方式：

### 方式 1: Qualcomm AI Hub 云编译（推荐）

通过 `qai-hub-models` 导出，云端自动完成 QNN 编译。需要 Qualcomm AI Hub 账号。

### 方式 2: HuggingFace 社区模型（本章用的方式）

下载别人预编译好的 context binary，省去编译步骤。我们用了两个模型：

- **Llama 3.2 1B**: [Eddie-L/Llama-3.2-1B-Instruct-Genie-QNN-NPU-8Gen4](https://huggingface.co/Eddie-L/Llama-3.2-1B-Instruct-Genie-QNN-NPU-8Gen4)
  - 单文件 `weight_sharing_model_1_of_1.serialized.bin`（~1.78GB）
- **Llama 3.2 3B**: [imi2/QNN-HTP-LLM-Genie-models](https://huggingface.co/imi2/QNN-HTP-LLM-Genie-models) 中的 `genie_bundle_llama_v3_2_3b_8_elite`
  - 三个文件共 ~2.6GB

### 方式 3: 本地编译

用 QNN SDK 工具链自己编译。最灵活但最复杂，需要处理量化、图优化、context 生成等步骤。

---

## 部署到手机

### 文件结构

部署需要三类文件：Genie 运行时库、模型文件、配置文件。

```
/data/local/tmp/genie_1b/
├── genie-t2t-run                              # 推理工具
├── libGenie.so                                # Genie 运行时
├── libQnnHtp.so                               # QNN HTP 后端
├── libQnnHtpPrepare.so                        # HTP 图准备
├── libQnnHtpV79Stub.so                        # ARM 端 stub（v79 = 骁龙 8 Elite）
├── libQnnHtpV79Skel.so                        # DSP 端 skel
├── libQnnSystem.so                            # QNN 系统库
├── libQnnHtpNetRunExtensions.so               # 网络运行扩展
├── tokenizer.json                             # Llama 3.2 tokenizer
├── htp-model-config-llama32-1b-gqa.json       # Genie 配置
├── htp_backend_ext_config.json                # HTP 后端配置
└── models/
    └── weight_sharing_model_1_of_1.serialized.bin  # context binary
```

### Step 1: 推送运行时库

库文件来自 QAIRT SDK v2.44.0：

```bash
QNN_SDK=/home/taowen/hexagon-tutorial/tools/qnn-sdk
DEST=/data/local/tmp/genie_1b

adb shell mkdir -p $DEST

# Genie 运行时 + QNN HTP 库
adb push $QNN_SDK/bin/aarch64-android/genie-t2t-run $DEST/
adb push $QNN_SDK/lib/aarch64-android/libGenie.so $DEST/
adb push $QNN_SDK/lib/aarch64-android/libQnnHtp.so $DEST/
adb push $QNN_SDK/lib/aarch64-android/libQnnHtpPrepare.so $DEST/
adb push $QNN_SDK/lib/aarch64-android/libQnnHtpV79Stub.so $DEST/
adb push $QNN_SDK/lib/aarch64-android/libQnnSystem.so $DEST/
adb push $QNN_SDK/lib/aarch64-android/libQnnHtpNetRunExtensions.so $DEST/

# DSP 端 skel（unsigned，开发用）
adb push $QNN_SDK/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so $DEST/
```

### Step 2: 推送模型和配置

```bash
# tokenizer
adb push tokenizer.json $DEST/

# 配置文件
adb push htp-model-config-llama32-1b-gqa.json $DEST/
adb push htp_backend_ext_config.json $DEST/

# context binary
adb shell mkdir -p $DEST/models
adb push models/weight_sharing_model_1_of_1.serialized.bin $DEST/models/
```

### Step 3: 运行

```bash
adb shell "cd /data/local/tmp/genie_1b && \
  LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  ./genie-t2t-run \
    -c htp-model-config-llama32-1b-gqa.json \
    -p 'What is the capital of France?'"
```

输出类似：

```
Genie Response: The capital of France is Paris.

Genie Perf:
  Init time: 1268 ms
  Time to first token: 96 ms
  Prefill speed: 125 tokens/sec
  Decode speed: 31 tokens/sec
  Total tokens: 12
```

3B 模型的部署方法相同，只是配置文件和 context binary 不同（3 个分片文件）。

---

## 实测结果

测试设备：骁龙 8 Elite (Hexagon v79)，QAIRT SDK v2.44.0 (Genie 1.16.0)

### Llama 3.2 1B

| 指标 | 短 prompt (12 tok) | 长 prompt (155 tok) |
|------|-------------------|-------------------|
| 预填充速度 | 125 tok/s | **973 tok/s** |
| 解码速度 | 31 tok/s | 31 tok/s |
| TTFT | 96 ms | 159 ms |
| 初始化时间 | 1,268 ms | 1,268 ms |

### Llama 3.2 3B

| 指标 | 短 prompt (8 tok) | 长 prompt (148 tok) |
|------|-------------------|-------------------|
| 预填充速度 | 135 tok/s | **1,227 tok/s** |
| 解码速度 | 24 tok/s | 24 tok/s |
| TTFT | 59 ms | 121 ms |
| 初始化时间 | 1,501 ms | 1,327 ms |

两个值得注意的现象：

1. **预填充速度随 prompt 长度急剧增长**：1B 从 125 tok/s (12 tokens) 跳到 973 tok/s (155 tokens)，涨了近 8 倍。原因是 HMX 矩阵单元需要足够大的批量才能填满计算流水线——prompt 越长，每次 matmul 的矩阵越大，HMX 的利用率越高。
2. **解码速度不受 prompt 长度影响**：始终是 31 tok/s (1B) 和 24 tok/s (3B)。因为解码永远是 batch=1，矩阵大小固定。

---

## Genie vs llama.cpp

这是本章最关键的对比。同一台骁龙 8 Elite，~128 tokens 的 prefill：

| 模型 | 运行时 | 预填充 (tok/s) | 解码 (tok/s) |
|------|--------|---------------:|------------:|
| Llama 3.2 1B | **Genie HTP** | **973** | 31 |
| Llama 3.2 1B | llama.cpp NPU (HTP) | 415 | 49 |
| Llama 3.2 1B | llama.cpp CPU | 303 | 64 |
| Llama 3.2 3B | **Genie HTP** | **1,227** | 24 |
| Qwen3-4B* | llama.cpp NPU (HTP) | 92 | 14 |
| Qwen3-4B* | llama.cpp CPU | 74 | 18 |

*注：llama.cpp 没有 Llama 3.2 3B 的直接对比数据，用最接近的 Qwen3-4B 替代。

### 1. 预填充：Genie 碾压

1B 模型：973 vs 415 tok/s，Genie 快 **2.3 倍**。
3B vs 4B：1,227 vs 92 tok/s，快 **13 倍**（模型大小不同，仅供参考）。

原因在计算路径的根本差异：

- **Genie 走 QNN 图级优化**：整个 transformer 编译成一张 HTP 图，QNN 编译器做算子融合（比如把 LayerNorm + Linear 合并）、全局内存规划（VTCM 分配一次搞定）、NativeKV（KV cache 直接用 HTP 内部格式，不经过格式转换）。
- **llama.cpp 逐算子下发**：每个 matmul 一次 dspqueue 调用，每次调用 ~61us 通信开销。一个 token 的 decode 要 168 次调用（24 层 x 7 算子），光通信就是 10ms。prefill 时虽然可以攒 batch，但算子之间没有融合，VTCM 也是每个算子自己管。

这就是 ch09 讲的"格式墙"和 ch05 讲的"通信墙"在真实场景中的体现：Genie 用 QNN 编译器一次性打通了这两道墙，llama.cpp 则要一个一个翻。

### 2. 解码：llama.cpp CPU 反而更快

1B 模型：llama.cpp CPU 64 tok/s vs Genie 31 tok/s，**CPU 快一倍**。

这个结果看起来反直觉，但原因很简单：

- 解码是 batch=1，矩阵是 hidden_dim x 1（比如 2048 x 1），小到 HMX 的 32x32 tile 根本填不满
- llama.cpp CPU 走 ARM NEON/SVE，矩阵虽小但内存带宽充足，没有通信开销
- Genie 走 HTP，即使图级优化了通信，batch=1 时 RPC 的固定开销仍然存在

这再次印证了本系列的核心观点：**NPU 的优势在大矩阵的计算密集场景，不在小矩阵的访存密集场景。** 解码时 batch=1，计算量太小，通信墙无论怎么优化都是瓶颈。

### 3. 预填充速度与 prompt 长度的关系

| prompt 长度 | 1B 预填充 (tok/s) | 提升倍数 |
|------------|------------------:|--------:|
| 12 tokens | 125 | 1x |
| 155 tokens | 973 | 7.8x |

HMX 的矩阵单元是 32x32 tile。prompt 12 tokens 时，矩阵形状是 hidden_dim x 12，大量 tile 空跑。prompt 155 tokens 时，矩阵变成 hidden_dim x 155，tile 利用率大幅提升。这是 NPU 的本质特征：**吞吐靠批量撑起来。**

---

## 配置文件解读

Genie 的行为由两个配置文件控制。

### genie_config.json（以 3B 为例）

```json
{
    "dialog": {
        "version": 1,
        "type": "basic",
        "max-num-tokens": 200,
        "context": {
            "size": 2048,
            "n-vocab": 128256,
            "bos-token": 128000,
            "eos-token": 128009,
            "pad-token": 128004
        },
        "sampler": {
            "seed": 42,
            "temp": 0.8,
            "top-k": 1,
            "top-p": 0.95
        },
        "tokenizer": {
            "path": "tokenizer.json"
        },
        "engine": {
            "n-threads": 3,
            "backend": {
                "type": "QnnHtp",
                "QnnHtp": {
                    "use-mmap": true,
                    "poll": true,
                    "cpu-mask": "0xe0",
                    "kv-dim": 128
                },
                "extensions": "htp_backend_ext_config.json"
            },
            "model": {
                "type": "binary",
                "binary": {
                    "ctx-bins": [
                        "llama_v3_2_3b_chat_quantized_part_1_of_3.bin",
                        "llama_v3_2_3b_chat_quantized_part_2_of_3.bin",
                        "llama_v3_2_3b_chat_quantized_part_3_of_3.bin"
                    ]
                }
            }
        }
    }
}
```

关键字段：

| 字段 | 说明 |
|------|------|
| `context.size` | 上下文窗口长度。1B 用 1024，3B 用 2048 |
| `context.n-vocab` | 词表大小，Llama 3.2 是 128256 |
| `sampler.top-k` | 设为 1 = greedy decoding，输出确定性 |
| `backend.QnnHtp.poll` | 轮询模式，降低延迟但耗电 |
| `backend.QnnHtp.cpu-mask` | `0xe0` = 使用 CPU 核 5/6/7（大核），避免干扰 NPU 通信线程 |
| `backend.QnnHtp.kv-dim` | KV head dimension。1B 是 64，3B 是 128 |
| `backend.QnnHtp.use-mmap` | mmap 加载 context binary，减少内存拷贝 |
| `model.binary.ctx-bins` | context binary 路径列表。3B 拆成 3 个分片 |

### htp_backend_ext_config.json

```json
{
    "devices": [{
        "soc_id": 69,
        "dsp_arch": "v79",
        "cores": [{
            "core_id": 0,
            "perf_profile": "burst",
            "rpc_control_latency": 100
        }]
    }],
    "memory": {
        "mem_type": "shared_buffer"
    }
}
```

| 字段 | 说明 |
|------|------|
| `soc_id: 69` | 骁龙 8 Elite 的 SoC ID |
| `dsp_arch: v79` | Hexagon v79 架构 |
| `perf_profile: burst` | 最高性能模式，频率拉满 |
| `rpc_control_latency: 100` | RPC 控制延迟（微秒） |
| `mem_type: shared_buffer` | 使用共享内存缓冲区 |

注意 `soc_id` 和 `dsp_arch` 必须与手机硬件匹配。如果在骁龙 8 Gen 3 (v75) 上跑，需要改成 `soc_id: 57`、`dsp_arch: v75`——但 context binary 也需要重新编译，因为 HTP 指令集不兼容。

---

## 小结

本章做了一件事：用 Qualcomm 官方的 Genie SDK 在骁龙 8 Elite 上跑 Llama 3.2 1B/3B，然后和 ch10 的 llama.cpp 结果做对比。

核心发现：

1. **预填充性能 Genie 远超 llama.cpp**，因为走了 QNN 完整的图优化路径——算子融合、全局内存规划、NativeKV。这正是 ch09 里我们手动实现 NativeKV 时追求的效果，只不过 QNN 编译器自动搞定了。

2. **解码速度 Genie 反而不如 CPU**，batch=1 的通信开销在 Genie 上同样存在。NPU 不是万能的——矩阵小的时候，ARM CPU 的 NEON/SVE 加上零通信开销，反而更快。

3. **代价是灵活性**。Genie 是 Qualcomm 的"全家桶"——模型编译、运行时、采样一条龙，但模型必须经过 Qualcomm 的 QNN 工具链编译成 context binary。不能像 llama.cpp 那样下载一个 GGUF 直接跑。而且 context binary 绑定特定 SoC，v79 编译的模型不能在 v75 上跑。

对应博客主题：Genie 用 QNN 编译器打通了格式墙（NativeKV，KV cache 免转换）和内存墙（图级内存规划，VTCM 一次分配），但通信墙在单 token 解码时仍然是瓶颈。**34 TOPS 在预填充时终于发挥出来了，但在解码时依然被喂不饱。**
