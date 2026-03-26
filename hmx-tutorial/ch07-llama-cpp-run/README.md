# 第十章：编译 llama.cpp，在真机 NPU 上跑 LLM

前九章我们从模拟器到真机、从 HVX/HMX 指令到 KV Cache 布局，逐步拆解了骁龙 NPU 的编程模型。本章把这些知识串起来：编译 llama.cpp 的 Hexagon 后端，推送到手机，用 NPU 跑通一个 GGUF 模型。

## 你需要什么

| 组件 | 说明 |
|------|------|
| **llama.cpp 源码** | `git clone https://github.com/ggml-org/llama.cpp` |
| **Hexagon SDK 6.4** | 包含 hexagon-clang 编译器，用于编译 DSP 端 .so |
| **Android NDK r27+** | 包含 aarch64-linux-android-clang，用于编译 ARM 端 |
| **骁龙 8 Gen 3 手机** | 或其他 v75+ 设备，开启 USB 调试 |
| **GGUF 模型** | 推荐 Qwen3-0.6B-Q4_0（约 400MB）或 Llama-3.2-1B-Q4_0（约 730MB） |

## 文件结构

```
ch10-llama-cpp-run/
├── README.md          # 本文
├── build.sh           # 编译 llama.cpp（ARM + DSP）
├── run_device.sh      # 推送到手机，用 NPU 跑推理
└── run_bench.sh       # 跑 llama-bench 测速
```

---

## Part 1: 编译

llama.cpp 的 Hexagon 后端是一次编译产出两套东西：

- **ARM 端**：`llama-cli`、`llama-bench` 等可执行文件 + `libggml-hexagon.so`（负责 ARM↔DSP 通信）
- **DSP 端**：`libggml-htp-v75.so`、`libggml-htp-v79.so` 等（每个 Hexagon 架构版本一个 .so，包含全部 HVX/HMX 算子）

ARM 端用 Android NDK 的 `aarch64-linux-android-clang` 编译。DSP 端用 Hexagon SDK 的 `hexagon-clang` 编译——和我们在 ch01-ch09 用的同一个编译器，同一组 `-mv75 -mhmx -mhvx` 标志。CMake 通过 `ExternalProject_Add` 自动触发 6 个 DSP 版本（v68/v69/v73/v75/v79/v81）的编译，运行时根据手机硬件自动加载匹配的版本。

### 设置环境变量

```bash
export HEXAGON_SDK_ROOT=$HOME/hexagon-tutorial/tools/hexagon-sdk   # 或你的 SDK 路径
export ANDROID_NDK_HOME=$HOME/android-sdk/ndk/27.2.12479018        # NDK 路径
export LLAMA_CPP=$HOME/llama.cpp                                    # llama.cpp 源码路径
```

### 编译

```bash
bash build.sh
```

编译过程大约 5-10 分钟（取决于机器性能）。完成后产物在 `pkg/llama.cpp/`：

```
pkg/llama.cpp/
├── bin/
│   ├── llama-cli              # 交互式推理
│   ├── llama-bench            # 性能测试
│   ├── llama-server           # HTTP API 服务
│   └── ...
└── lib/
    ├── libggml.so
    ├── libggml-cpu.so         # CPU 后端
    ├── libggml-hexagon.so     # ARM 端 Hexagon 后端（dspqueue 通信）
    ├── libggml-htp-v68.so     # DSP 端算子（v68 = 骁龙 888）
    ├── libggml-htp-v73.so     # DSP 端算子（v73 = 骁龙 8 Gen 2）
    ├── libggml-htp-v75.so     # DSP 端算子（v75 = 骁龙 8 Gen 3）
    ├── libggml-htp-v79.so     # DSP 端算子（v79 = 骁龙 8 Elite）
    ├── libggml-htp-v81.so     # DSP 端算子（v81）
    └── ...
```

### 编译过程中发生了什么

CMake configure 时，`GGML_HEXAGON=ON` 触发以下链条：

```
CMakeLists.txt
  └─ ggml/src/ggml-hexagon/CMakeLists.txt
       ├─ 编译 ARM 端：ggml-hexagon.cpp + htp-drv.cpp → libggml-hexagon.so
       ├─ 用 qaic 从 htp_iface.idl 生成 stub/skel
       └─ ExternalProject_Add × 6：
            └─ htp/CMakeLists.txt（hexagon-clang -mv75 ...）
                 ├─ flash-attn-ops.c      Flash Attention（HVX + DMA 双缓冲）
                 ├─ matmul-ops.c          矩阵乘（HMX mxmem 指令）
                 ├─ set-rows-ops.c        KV Cache 写入
                 ├─ hvx-quant.c           HVX 反量化（Q4_0, Q8_0, IQ4_NL...）
                 ├─ worker-pool.c         多 HVX 线程池
                 └─ ...15+ 源文件 → libggml-htp-v75.so
```

`qaic` 是 FastRPC 的 IDL 编译器——和 ch05 里我们用的一样。它从 `htp_iface.idl` 生成 `htp_iface_stub.c`（ARM 端）和 `htp_iface_skel.c`（DSP 端）。FastRPC 只用于建立连接（`start`/`stop`），推理计算全走 dspqueue。

---

## Part 2: 下载模型

llama.cpp 使用 GGUF 格式。推荐从小模型开始：

| 模型 | 大小 | 参数量 | 说明 |
|------|------|--------|------|
| Qwen3-0.6B-Q4_0 | ~400 MB | 0.6B | 最小，适合验证 NPU 是否工作 |
| Llama-3.2-1B-Instruct-Q4_0 | ~730 MB | 1.2B | 质量更好，单 HTP session 能跑 |
| Qwen3-4B-Q4_0 | ~2.5 GB | 4B | 需要更多内存，但仍可单 session |

下载并推送到手机：

```bash
# 下载 Qwen3-0.6B（约 364MB）
wget https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_0.gguf

# 推送到手机
adb shell mkdir -p /data/local/tmp/gguf
adb push Qwen3-0.6B-Q4_0.gguf /data/local/tmp/gguf/
```

如果你已有其他 GGUF 模型，直接推送到 `/data/local/tmp/gguf/` 即可。

---

## Part 3: 在 NPU 上运行

### 推理

```bash
bash run_device.sh
```

或指定模型和 prompt：

```bash
MODEL=Llama-3.2-1B-Instruct-Q4_0.gguf PROMPT="What is 1+1?" bash run_device.sh
```

你会看到类似这样的输出：

```
ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev 1
ggml-hex: Hexagon Arch version v75
ggml-hex: allocating new session: HTP0
ggml-hex: new session: HTP0 : session-id 0 domain-id 3 uri file:///libggml-htp-v75.so?htp_iface_skel_handle_invoke&_modver=1.0
...
load_tensors: offloaded 17/17 layers to GPU
load_tensors:  HTP0-REPACK model buffer size =   358.78 MiB
...
> What is 1+1? Answer briefly.
[Start thinking]
Okay, the user is asking, "What is 1+1? Answer briefly."...
[End thinking]
The sum of 1 + 1 is 2.
[ Prompt: 28.8 t/s | Generation: 2.5 t/s ]
```

关键信息拆解：

- **`Hexagon Arch version v75`**：运行时自动检测到骁龙 8 Gen 3 的 NPU 版本
- **`file:///libggml-htp-v75.so`**：自动加载匹配的 DSP 算子库
- **`offloaded 17/17 layers to GPU`**：`-ngl 99` 把所有 transformer 层都放到 NPU（llama.cpp 把 NPU 当作 GPU 设备处理）
- **`HTP0-REPACK`**：权重经过 x4x2 repack 格式转换后存在共享内存中
- **Generation: 2.5 t/s**：Qwen3 默认开启 thinking 模式，大量 token 花在思考过程上，实际生成速度被拉低

### 性能测试

```bash
bash run_bench.sh
```

### 真机实测数据

**骁龙 8 Gen 3 (v75)：**

| 模型 | 后端 | pp128 (tokens/s) | tg64 (tokens/s) |
|------|------|------------------:|----------------:|
| Qwen3-0.6B | NPU | 367 | 40 |
| Qwen3-0.6B | CPU | **686** | **106** |
| Llama-3.2-1B | NPU | 263 | 35 |
| Llama-3.2-1B | CPU | **371** | **55** |
| Qwen3-1.7B | NPU | 169 | 25 |
| Qwen3-1.7B | CPU | **226** | **36** |
| Qwen3-4B | NPU | **84** | 12 |
| Qwen3-4B | CPU | 82 | **16** |

**骁龙 8 Elite (v79)：**

| 模型 | 后端 | pp128 (tokens/s) | tg64 (tokens/s) |
|------|------|------------------:|----------------:|
| Qwen3-0.6B | NPU | 494 | 51 |
| Qwen3-0.6B | CPU | **516** | **105** |
| Llama-3.2-1B | NPU | **415** | 49 |
| Llama-3.2-1B | CPU | 303 | **64** |
| Qwen3-1.7B | NPU | **236** | 33 |
| Qwen3-1.7B | CPU | 214 | **41** |
| Qwen3-4B | NPU | **92** | 14 |
| Qwen3-4B | CPU | 74 | **18** |

**趋势很清楚：**

- **Prefill（pp128）**：模型越大，NPU 优势越明显。两台设备上 4B 都是 NPU 反超 CPU 的拐点
- **Decode（tg64）**：NPU 始终输给 CPU。因为 decode 是 batch=1，矩阵太小（hidden_dim x 1），通信开销占比大
- **8 Elite 比 8 Gen 3 快**：同模型同后端，8 Elite NPU prefill 普遍快 10-35%（如 1B NPU: 415 vs 263），CPU 侧差距更小

在小模型（0.6B）上 NPU 反而比 CPU 慢，这看起来反直觉——34 TOPS 的 NPU 居然跑不过 CPU？原因正是本系列博客一直在讲的三道墙：

1. **通信墙**：Qwen3-0.6B 有 24 层，每个 token 的 decode 需要 ~168 次 dspqueue 调用（24 层 x 7 算子）。每次 61us，通信开销就是 ~10ms
2. **格式墙**：每次 matmul 前要做 HVX 反量化 + 数据重排，才能喂给 HMX 的 32x32 tile
3. **内存墙**：0.6B 模型的矩阵很小（896x896），HMX 的 32x32 tile 还没热起来就算完了

**NPU 的优势在大矩阵**。回顾 Part 1 开头的实验：256x1024x4096 的矩阵乘 NPU 快 166 倍。但小模型的矩阵太小，通信和格式转换的固定开销淹没了 HMX 的吞吐优势。模型越大，NPU 在 prefill 上的优势越明显——4B 以上 NPU 才真正值回票价。

- **pp128**：prefill 128 tokens 的速度（tokens/s），衡量首字延迟
- **tg64**：生成 64 tokens 的速度（tokens/s），衡量吐字速度

---

## Part 4: 它是怎么跑起来的

从你敲下 `run_device.sh` 到屏幕上出现第一个字，发生了什么？

### 启动阶段

```
ARM: llama-cli 启动
  │
  ├─ dlopen("libggml-hexagon.so")          加载 Hexagon 后端
  │    └─ dlopen("libcdsprpc.so")          加载 FastRPC 运行时（系统库）
  │
  ├─ 检测硬件：查询 DSP 的 ISA 版本 → v75
  │
  ├─ dspqueue_create() + dspqueue_export()  创建共享内存队列
  │
  ├─ FastRPC: htp_iface_start()            唯一一次 FastRPC 调用
  │    └─ DSP 端: open("file:///libggml-htp-v75.so")
  │         └─ dspqueue_import()           DSP 侧连接队列
  │
  └─ 从此 ARM↔DSP 通信全走 dspqueue（~61us/op）
```

### 加载模型

```
ARM: 读取 GGUF 文件
  │
  ├─ 解析模型结构（层数、head_dim、vocab_size...）
  │
  ├─ rpcmem_alloc2() + fastrpc_mmap()     分配共享内存
  │    └─ ARM 和 DSP 都能直接访问，零拷贝
  │
  └─ x4x2 repack                          权重格式转换
       └─ 4 个量化块交错排列，quants 在前 scales 在后
          为 HVX 128 字节向量加载优化
```

### 推理循环

每生成一个 token，ARM 向 dspqueue 提交一批算子请求：

```
ARM                              DSP (Hexagon v75)
 │                                │
 │  enqueue(RMS_NORM)  ───────>   │  HVX: RMS normalization
 │  enqueue(MUL_MAT)   ───────>   │  HVX 反量化 → HMX 32x32 matmul
 │  enqueue(MUL_MAT)   ───────>   │  （VTCM 双缓冲：DMA 搬一块，HMX 算一块）
 │  enqueue(ADD)        ───────>   │  HVX: residual add
 │  enqueue(ROPE)       ───────>   │  HVX: 旋转位置编码
 │  enqueue(FLASH_ATTN) ───────>   │  HVX: flash attention (DDR→VTCM 流式)
 │  enqueue(MUL_MAT)   ───────>   │  FFN down projection
 │  ...×28 层
 │  flush() 阻塞等待  ──────────>  │  全部完成
 │                                │
 │  <── 读回 logits               │
 │  采样下一个 token               │
```

这就是 ch05 里我们手写的 dspqueue demo 的放大版。同样的 `enqueue/flush` 模式，同样的 `dspqueue_write/dspqueue_read_noblock` 通信，只是算子从 2 个变成了 30 个，计算从标量循环变成了 HVX/HMX 内核。

---

## Part 5: 常用环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GGML_HEXAGON_NDEV` | 1 | HTP session 数量。4B 以下模型用 1，8B 用 2 |
| `GGML_HEXAGON_NHVX` | 全部 | HVX 线程数。减少可降低功耗 |
| `GGML_HEXAGON_USE_HMX` | 1 | 是否启用 HMX 矩阵加速。设 0 退化为纯 HVX |
| `GGML_HEXAGON_VERBOSE` | 0 | 打印每个算子的详细日志 |
| `GGML_HEXAGON_EXPERIMENTAL` | 0 | 启用实验特性（如 FLASH_ATTN_EXT） |
| `GGML_HEXAGON_PROFILE` | 0 | 生成算子耗时 profile |

示例——开启详细日志看每个算子的执行：

```bash
adb shell " \
    cd /data/local/tmp/llama.cpp && \
    LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib \
    GGML_HEXAGON_VERBOSE=1 \
    ./bin/llama-cli --no-mmap -m /data/local/tmp/gguf/Qwen3-0.6B-Q4_0.gguf \
        -ngl 99 --device HTP0 -t 6 --ctx-size 2048 -fa on -no-cnv \
        -p '1+1=' 2>&1 | head -50
"
```

输出中你会看到每个算子的类型、输入输出张量形状和后端分配：

```
ggml-hex: HTP0 matmul : blk.0.attn_q.weight x attn_norm-0 -> attn_q-0 : 896:896 x 896:1 -> 896:1
ggml-hex: HTP0 matmul : blk.0.attn_k.weight x attn_norm-0 -> attn_k-0 : 896:128 x 896:1 -> 128:1
ggml-hex: HTP0 matmul : blk.0.attn_v.weight x attn_norm-0 -> attn_v-0 : 896:128 x 896:1 -> 128:1
```

这正是 ch05 里 `htp_packet_callback` 中 `switch(req.op)` 分发的那些算子。

---

## Part 6: 常见问题

### Q: 编译报错找不到 hexagon-clang

确认 `HEXAGON_SDK_ROOT` 指向正确的 Hexagon SDK 目录。SDK 内部结构应该是：

```
hexagon-sdk/
├── tools/HEXAGON_Tools/19.0.04/Tools/bin/hexagon-clang
├── rtos/qurt/computev75/
├── incs/
└── ipc/fastrpc/
```

如果你的 SDK 没有 `HEXAGON_Tools`，CMake 会尝试从 `hexagon_sdk.json` 自动定位。

### Q: 运行时报 "failed to open session"

1. 确认 `ADSP_LIBRARY_PATH` 包含 `libggml-htp-v75.so` 所在的目录
2. 确认手机的 CDSP 支持 Unsigned PD（骁龙 8 Gen 3 默认支持）
3. 用 `adb shell ls -la /dev/cdsp` 确认 CDSP 设备节点存在

### Q: 模型太大，报内存不足

每个 HTP session 有约 2GB 共享内存限制。如果模型权重（repack 后）超过 2GB，需要多个 session：

```bash
GGML_HEXAGON_NDEV=2 adb shell " \
    cd /data/local/tmp/llama.cpp && \
    LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib \
    GGML_HEXAGON_NDEV=2 \
    ./bin/llama-cli --no-mmap -m /data/local/tmp/gguf/Large-Model-Q4_0.gguf \
        -ngl 99 --device HTP0,HTP1 -t 6 --ctx-size 2048 -fa on -no-cnv \
        -p 'Hello' 2>&1
"
```

### Q: 想对比 CPU vs NPU 性能

用 `GGML_HEXAGON_NDEV=0` 禁用 NPU session，退化为纯 CPU 推理：

```bash
# CPU（设 NDEV=0 禁用 NPU）
adb shell " \
    cd /data/local/tmp/llama.cpp && \
    LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib \
    GGML_HEXAGON_NDEV=0 \
    ./bin/llama-bench -m /data/local/tmp/gguf/Qwen3-0.6B-Q4_0.gguf \
        -t 6 -fa 1 -p 128 -n 64 2>&1
"

# NPU
bash run_bench.sh
```

---

## 总结

本章做了两件事：

1. 用 llama.cpp 在骁龙 NPU 上跑通了 Qwen3-0.6B 模型
2. 用实测数据揭示了一个反直觉的事实——**0.6B 小模型在 NPU 上比 CPU 慢**

编译只需要两个工具链（Android NDK + Hexagon SDK），CMake 自动处理 ARM/DSP 双编译。运行时 `libggml-hexagon.so` 通过 FastRPC 建立一次连接，然后所有推理都走 dspqueue 共享内存。30 个算子——matmul、softmax、RoPE、flash attention——全部在 DSP 上用 HVX/HMX 执行。

NPU 的 34 TOPS 没有消失，但在 0.6B 模型上被三道墙吞掉了。这正是整个系列的核心论点：**瓶颈从来不是算不动，而是喂不饱。**

从 ch01 的模拟器 Hello World 到 ch10 的真机 LLM 推理，这条路径是完整的：

| 章节 | 里程碑 |
|------|--------|
| ch01 | 模拟器上跑通 HVX + HMX |
| ch02 | 真机上跑通 FastRPC |
| ch03-04 | QNN 框架 + 自定义算子 |
| ch05 | dspqueue 通信，理解 llama.cpp 架构 |
| ch06-07 | VTCM 内存管理 |
| ch08 | HexKL HMX 矩阵乘 |
| ch09 | NativeKV Cache |
| **ch10** | **llama.cpp 编译 + 真机 NPU 推理** |
