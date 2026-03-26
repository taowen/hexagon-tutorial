# QNN 教程：框架级 NPU 开发

通过 Qualcomm QNN 框架开发自定义算子，用 x86 模拟加速迭代，最终用 Genie SDK 跑通 LLM 推理。

## 前置：共同基础

- **[第一章：安装模拟器，跑通 HVX + HMX](../ch01-simulator-setup/)** — 在 x86 Linux 上配置 Hexagon SDK 6.4，用 H2 Hypervisor 模拟器运行第一个 HVX 向量加法 + HMX 矩阵乘法程序。不需要真机，验证工具链能用即可。

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

- **[第二章：在真机上跑 HVX + HMX](../ch02-real-device/)** — 把代码从模拟器搬到骁龙 8 Gen 3 真机。通过 FastRPC 加载 DSP 共享库，用 HAP API 申请 VTCM/HVX/HMX 硬件资源，完成从"能编译"到"能在手机上跑"的跨越。

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

---

## 第三章：QNN 自定义算子 (HVX+HMX)

**目录**: [`ch03-qnn-custom-op/`](ch03-qnn-custom-op/)

用 QHPI 接口在 QNN 图中插入自定义算子，在一个 QNN 节点里混合使用 HVX（做 ReLU 等前处理）和 HMX（做矩阵乘法）。理解 QNN 框架如何管理 VTCM 和 HMX 资源。

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

---

## 第四章：QNN x86 模拟开发

**目录**: [`ch04-qnn-simulator/`](ch04-qnn-simulator/)

用 libnative 在 x86 上模拟 HVX/HMX 指令，实现一套代码同时跑 Hexagon 和 x86。开发调试不再依赖真机，大幅提升迭代速度。

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

---

## 第五章：QNN VTCM 管理

**目录**: [`ch05-qnn-vtcm/`](ch05-qnn-vtcm/)

从 QNN 框架层面观察 VTCM：配置 spill-fill buffer 大小、profiling VTCM 使用量、多图共享 VTCM。看 QNN 编译器如何自动调度这 8MB 片上内存。

```c
// 两个 AI 模型并行共享 8MB VTCM —— 无需上下文切换
// Graph A: 使用 VTCM [0, 4MB)
cfgA.parallelGraphExecutionConfig.vtcmConfig.sizeInBytes   = 4 * 1024 * 1024;
cfgA.parallelGraphExecutionConfig.vtcmConfig.offsetInBytes = 0;
// Graph B: 使用 VTCM [4MB, 8MB)
cfgB.parallelGraphExecutionConfig.vtcmConfig.sizeInBytes   = 4 * 1024 * 1024;
cfgB.parallelGraphExecutionConfig.vtcmConfig.offsetInBytes = 4 * 1024 * 1024;
```

---

## 第六章：Genie SDK 推理

**目录**: [`ch06-genie-run/`](ch06-genie-run/)

走 Qualcomm 官方路径：用 Genie SDK + QNN context binary 跑 Llama 3.2 1B/3B。Genie prefill 比 llama.cpp 快 2-13 倍（QNN 图级优化自动处理算子融合、内存规划和 NativeKV），但 decode 时 batch=1 矩阵太小，通信延迟仍是瓶颈。

```bash
# Genie：一行命令跑 NPU 推理
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
