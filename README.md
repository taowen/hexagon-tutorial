# Hexagon DSP 开发教程

从零开始学习 Qualcomm Hexagon DSP/NPU 编程。从模拟器上的 Hello World 到真机上跑通 LLM 推理，覆盖 HVX（向量处理）、HMX（矩阵加速）、VTCM（片上高速内存）、dspqueue（低延迟通信）、QNN 框架和 Genie SDK。

## 目录

### 第一部分：从零搭建开发环境

- **[第一章：安装模拟器，跑通 HVX + HMX](ch01-simulator-setup/)** — 在 x86 Linux 上配置 Hexagon SDK 6.4，用 H2 Hypervisor 模拟器运行第一个 HVX 向量加法 + HMX 矩阵乘法程序。不需要真机，验证工具链能用即可。

```c
// HMX 矩阵乘法：activation 和 weight 必须在同一个 VLIW packet 里加载
asm volatile(
    "{ activation.hf = mxmem(%0, %1)\n"
    "  weight.hf = mxmem(%2, %3) }\n"
    :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
```
```bash
# 编译（-mhmx 开启矩阵扩展）+ H2 模拟器运行
hexagon-clang -O2 -mv75 -mhvx -mhvx-length=128B -mhmx -moslib=h2 ...
hexagon-sim --mv75 --mhmx 1 -- h2-install/bin/booter test_hvx_hmx
```

- **[第二章：在真机上跑 HVX + HMX](ch02-real-device/)** — 把代码从模拟器搬到骁龙 8 Gen 3 真机。通过 FastRPC 加载 DSP 共享库，用 HAP API 申请 VTCM/HVX/HMX 硬件资源，完成从"能编译"到"能在手机上跑"的跨越。

```c
// 真机上需要手动申请硬件资源（模拟器上不需要）
HAP_compute_res_attr_set_vtcm_param(&attr, vtcm_size, 1);  // 申请 VTCM
HAP_compute_res_attr_set_hmx_param(&attr, 1);               // 申请 HMX
unsigned int ctx_id = HAP_compute_res_acquire(&attr, 100000);
void *vtcm_base = HAP_compute_res_attr_get_vtcm_ptr(&attr);
HAP_compute_res_hmx_lock(ctx_id);
// WARNING: 千万不要设 cache_mode 或 serialize —— 会导致 DSP 永久挂死！
```
```bash
# 编译为共享库（不是 ELF），通过 FastRPC 部署
hexagon-clang -mv75 -shared -fPIC -o libtest_hvx_hmx_device.so ...
adb shell run_main_on_hexagon 3 libtest_hvx_hmx_device.so  # 3 = CDSP
```

### 第二部分：QNN 框架与算子开发

- **[第三章：QNN 自定义算子 (HVX+HMX)](ch03-qnn-custom-op/)** — 用 QHPI 接口在 QNN 图中插入自定义算子，在一个 QNN 节点里混合使用 HVX（做 ReLU 等前处理）和 HMX（做矩阵乘法）。理解 QNN 框架如何管理 VTCM 和 HMX 资源。

```cpp
// QHPI 声明：数据在 VTCM，内核需要 HMX —— QNN 自动管理资源分配
static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_Float16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
};
static QHPI_Kernel_v1 kernels[] = {{
    .function = hvx_hmx_mix_kernel,
    .resources = QHPI_RESOURCE_HMX,   // QNN 自动锁 HMX，不需要手动 HAP_compute_res
}};
```
```bash
# 三个编译目标：DSP 内核 + ARM CPU 回退 + ARM 测试程序
hexagon-clang++ -mv75 -mhvx -mhmx -shared -fPIC -o libHvxHmxMix_htp.so src/dsp/*.cpp
clang++ --target=aarch64-android -shared -fPIC -o libHvxHmxMix_cpu.so src/dsp/*.cpp
```

- **[第四章：QNN 自定义算子 x86 模拟](ch04-qnn-simulator/)** — 用 libnative 在 x86 上模拟 HVX/HMX 指令，实现一套代码同时跑 Hexagon 和 x86。开发调试不再依赖真机，大幅提升迭代速度。

```cpp
// 同一份代码，两套头文件 —— #ifdef 只在 include 处，业务逻辑完全一致
#ifdef __hexagon__
#include <hvx_hexagon_protos.h>    // Hexagon 编译器原生
#else
#include "hvx_hexagon_protos.h"    // libnative x86 模拟
#endif
HVX_Vector v_zero = Q6_V_vzero();
vp[i] = Q6_Vh_vmax_VhVh(vp[i], v_zero);   // HVX ReLU，两个平台行为一致
```

### 第三部分：深入 NPU 架构

- **[第五章：dspqueue vs FastRPC](ch05-llama-cpp-hexagon/)** — 以 llama.cpp 的 Hexagon 后端为切入点，分析 FastRPC 在高频小算子场景的延迟瓶颈（364μs/次），对比 dspqueue 的共享内存队列方案（61μs/次）。理解 ARM↔DSP 通信的本质开销。

```c
// dspqueue：共享内存 + 零拷贝，唯一的 FastRPC 调用只是传递 queue ID
rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, flags, buf_size);  // 分配共享内存
dspqueue_create(domain_id, 0, 4096, 4096, callback, &queue);
dspqueue_export(queue, &dsp_queue_id);
dspqueue_demo_start(g_handle, dsp_queue_id);  // 这是仅有的一次 FastRPC 调用
```
```c
// DSP 侧：无内核切换的消息循环
dspqueue_read_noblock(queue, &flags, MAX_BUFS, &n_bufs, bufs, ...);
switch (req.op) {
    case OP_SCALE: op_scale(bufs[0].ptr, bufs[1].ptr, ...); break;
    case OP_ADD:   op_add(bufs[0].ptr, bufs[1].ptr, ...);   break;
}
dspqueue_write(queue, 0, n_rsp_bufs, rsp_bufs, ...);  // 写回响应
```

- **[第六章：VTCM 内存管理](ch06-vtcm-memory/)** — 深入 VTCM（1 周期延迟的片上高速内存，v75 上 8MB）。实现 DMA/UDMA 异步搬运、双缓冲流水线、bump/pool 自定义分配器。理解"8MB 要装 14GB 模型"的内存墙挑战。

```c
// UDMA 异步搬运：硬件在后台搬数据，HVX 同时计算 —— 17 倍加速
Q6_dmstart_A((void *)&desc);                           // 启动 DMA（立即返回）
hvx_vadd(compute_a, compute_b, compute_c, DMA_SIZE);   // HVX 同时计算
Q6_R_dmwait();                                          // DMA 早就搬完了
// 串行：3701μs → 重叠：217μs → 17x 加速
```

- **[第七章：QNN VTCM 实验](ch07-qnn-vtcm/)** — 从 QNN 框架层面观察 VTCM：配置 spill-fill buffer 大小、profiling VTCM 使用量、多图共享 VTCM。看 QNN 编译器如何自动调度这 8MB 片上内存。

```c
// 两个 AI 模型并行共享 8MB VTCM —— 无需上下文切换
// Graph A: 使用 VTCM [0, 4MB)
cfgA.parallelGraphExecutionConfig.vtcmConfig.sizeInBytes   = 4 * 1024 * 1024;
cfgA.parallelGraphExecutionConfig.vtcmConfig.offsetInBytes = 0;
// Graph B: 使用 VTCM [4MB, 8MB)
cfgB.parallelGraphExecutionConfig.vtcmConfig.sizeInBytes   = 4 * 1024 * 1024;
cfgB.parallelGraphExecutionConfig.vtcmConfig.offsetInBytes = 4 * 1024 * 1024;
```

- **[第八章：HexKL 矩阵乘法](ch08-hexkl-matmul/)** — 用 HexKL（Hexagon Kernel Library）从底层搭建 HMX 矩阵乘法。对比三个 API 层级：SDKL（ARM 侧一行调用）、Micro（DSP 侧 tile 控制）、HVX+HMX 组合流水线。理解 weight layout 转换和 tile 调度的细节。

```c
// HVX 反量化流水线：int8 → int16 → f16 → 乘以 scale（LLM 推理的核心模式）
HVX_VectorPair wp = Q6_Wh_vunpack_Vb(v_i8);                  // int8 → int16
HVX_Vector v_f16 = Q6_Vhf_equals_Vh(Q6_V_lo_W(wp));          // int16 → f16
HVX_Vector v_prod = Q6_Vhf_equals_Vqf16(
    Q6_Vqf16_vmpy_VhfVhf(v_f16, v_scale));                   // f16 × scale
```
```c
// SDKL 一行调用 vs 手动 layout 转换 —— 166 倍加速
sdkl_npu_mm_f32f16_f32(CDSP, N, M, K, A, X, W);              // 自动模式
sdkl_cpu_rm_to_wh_f16_inplace(M, K, W2);                     // 手动：预转换 weight layout
sdkl_npu_mm_f16(CDSP, N, M, K, A2, X2, W2);                  // 手动：直接用转好的 layout
```

- **[第九章：KV Cache 优化](ch09-kv-cache/)** — LLM 推理中 KV cache 的格式墙问题：每次 attention 都要把行优先数据转成 HMX weight layout，转换开销吞掉计算收益。NativeKV 方案直接以 weight layout 存储 KV cache，跳过转换，但需要 v75+ 硬件支持。

```c
// SmartMask：每步都要 rm_to_wh 转换（格式墙）
hexkl_micro_hmx_rm_to_wh_f16(vtcm, offset, K_rm, row, col, ctx_size);

// NativeKV：K 已经是 WH 格式，直接拷贝 tile（跳过转换）
hexkl_micro_hmx_copy_psubmatrix_to_f16_weight(vtcm, offset, K_wh, row, col, ...);
```

### 第四部分：端到端 LLM 推理

- **[第十章：llama.cpp 真机推理](ch10-llama-cpp-run/)** — 编译 llama.cpp 的 Hexagon 后端，在骁龙 8 Gen 3 和 8 Elite 上实测 4 个模型（0.6B-4B）的 NPU vs CPU 性能。发现 4B 是 NPU prefill 反超 CPU 的拐点，decode 始终 CPU 更快。三道墙的真实体现。

```bash
# 一个 CMake 命令同时触发 ARM + DSP 双编译
cmake -S llama.cpp -B build \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DGGML_HEXAGON=ON \
    -DHEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT
```
```bash
# NPU 推理：-ngl 99 把所有层 offload 到 HTP
./llama-cli --no-mmap -m Qwen3-4B-Q4_0.gguf \
    -ngl 99 --device HTP0 -t 6 -fa on -p 'What is 1+1?'
```

- **[第十一章：Genie SDK 推理](ch11-genie-run/)** — 走 Qualcomm 官方路径：用 Genie SDK + QNN context binary 跑 Llama 3.2 1B/3B。Genie prefill 比 llama.cpp 快 2-13 倍（QNN 图级优化打通格式墙+内存墙），但 decode 反而更慢（batch=1 通信墙仍是瓶颈）。

```bash
# Genie：一行命令跑 NPU 推理（对比 ch10 的从源码编译）
./genie-t2t-run -c htp-model-config-llama32-1b-gqa.json \
    -p 'What is the capital of France?'
```
```json
// 配置文件控制 QNN HTP 后端、KV cache 维度、CPU 亲和性
"backend": {
    "type": "QnnHtp",
    "QnnHtp": { "use-mmap": true, "poll": true, "cpu-mask": "0xe0", "kv-dim": 128 }
}
```