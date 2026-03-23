# 第三章：QNN 自定义算子 — 在 QNN 图内混合 HVX + HMX

本章目标：用 QNN 的 QHPI（Qualcomm Hexagon Plugin Interface）编写一个自定义算子，在 QNN 图内同时使用 HVX 和 HMX。

## 为什么需要 QNN 自定义算子？

前两章我们直接在 DSP 上写代码。这很灵活，但在生产环境中，大多数 AI 推理框架使用 QNN（Qualcomm AI Engine Direct）来调度 HTP 后端。QNN 内置了 180+ 个优化内核，但如果你需要一个 QNN 不支持的算子（比如 RoPE、SwiGLU、自定义 attention），就需要自定义算子。

**QHPI**（QNN SDK 2.44+）是高通推出的新 API，替代了旧的 `DEF_PACKAGE_OP` 宏系统。关键突破：

| 能力 | 旧 API | QHPI |
|------|--------|------|
| VTCM 访问 | 被禁用 | `QHPI_MemLoc_TCM_Only` |
| HMX 资源 | 不可用 | `QHPI_RESOURCE_HMX` |
| 注册方式 | C++ 宏 | 纯 C 结构体 |

## 架构

```
ARM 端 (Android)                           DSP 端 (CDSP)
┌─────────────────────────┐               ┌──────────────────────────────┐
│ qnn_hvx_hmx_test.c      │   QNN HTP    │ libHvxHmxMix_htp.so         │
│                          │  ────────→   │                              │
│ 1. dlopen(libQnnHtp.so)  │              │ qhpi_init() → 注册算子       │
│ 2. register(HvxHmxMix)  │              │                              │
│ 3. graphCreate + addNode │              │ hvx_hmx_mix_kernel():        │
│ 4. graphFinalize         │              │   Step 1: HVX ReLU (预处理)  │
│ 5. graphExecute          │              │   Step 2: HMX matmul        │
│ 6. 验证结果              │              │   Step 3: HVX ReLU (后处理)  │
└─────────────────────────┘               └──────────────────────────────┘
                                                      ↑
                                               所有数据在 VTCM
                                           (QHPI_MemLoc_TCM_Only)
```

和第二章的关键区别：
- 第二章：自己管理 VTCM（`HAP_compute_res`）和 HMX 锁
- 本章：**QNN 框架自动管理** VTCM 和 HMX——你只需要在 tensor signature 中声明 `TCM_Only`

## 源代码结构

```
ch03-qnn-custom-op/
├── src/
│   ├── dsp/
│   │   ├── HvxHmxOp.cpp          # 算子内核 (HVX ReLU + HMX matmul)
│   │   └── HvxHmxInterface.cpp   # QNN 包注册接口
│   └── host/
│       └── qnn_hvx_hmx_test.c    # ARM 端测试 (构建图、执行、验证)
├── build.sh
├── run_device.sh
└── expected_output/
```

## DSP 端：算子内核

### QHPI 注册

每个自定义算子需要两个入口点：

```cpp
/* 1. QNN 标准接口 — 包信息、验证 */
extern "C" Qnn_ErrorHandle_t HvxHmxMixInterfaceProvider(QnnOpPackage_Interface_t *interface);

/* 2. QHPI 入口 — 注册内核 */
extern "C" const char *qhpi_init() {
    register_hvx_hmx_mix_ops();
    return THIS_PKG_NAME_STR;    // "HvxHmxMixPackage"
}
```

### Tensor 签名：声明 VTCM + HMX

```cpp
/* 所有输入输出都在 VTCM */
static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_Float16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
    {QHPI_Float16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
    {QHPI_Float16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
};

/* 内核声明需要 HMX 资源 */
static QHPI_Kernel_v1 kernels[] = {
    {
        .function = hvx_hmx_mix_kernel,
        .resources = QHPI_RESOURCE_HMX,      // 告诉 QNN 我们需要 HMX
        .source_destructive = true,            // 我们修改了输入 (in-place ReLU)
        .min_inputs = 3,
        .input_signature = sig_inputs,
        .min_outputs = 1,
        .output_signature = sig_outputs,
        // ...
    },
};
```

`QHPI_MemLoc_TCM_Only` 让 QNN 框架自动把数据放到 VTCM。`QHPI_RESOURCE_HMX` 让 QNN 自动锁定 HMX 上下文。不需要手动调用 `HAP_compute_res`。

### 内核实现：HVX → HMX → HVX 流水线

```cpp
static uint32_t hvx_hmx_mix_kernel(QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
    /* 获取 VTCM 指针 — QNN 已经把数据放在 VTCM 了 */
    uint16_t *act_ptr = (uint16_t *)qhpi_tensor_raw_data(inputs[0]);
    uint16_t *wt_ptr  = (uint16_t *)qhpi_tensor_raw_data(inputs[1]);
    uint8_t  *bias    = (uint8_t  *)qhpi_tensor_raw_data(inputs[2]);
    uint16_t *out_ptr = (uint16_t *)qhpi_tensor_raw_data(outputs[0]);

    /* Step 1: HVX ReLU 预处理 — 负值清零 */
    HVX_Vector v_zero = Q6_V_vzero();
    HVX_Vector *vp = (HVX_Vector *)act_ptr;
    for (int i = 0; i < 16; i++)              // 16 vectors × 64 F16 = 1024 元素
        vp[i] = Q6_Vh_vmax_VhVh(vp[i], v_zero);

    /* Step 2: HMX matmul — 和第一、二章完全相同的指令 */
    Q6_bias_mxmem2_A(bias);
    Q6_mxclracc_hf();
    hmx_matmul_f16((uint32_t)act_ptr, (uint32_t)wt_ptr);  // activation + weight 配对
    Q6_mxmem_AR_after_hf(out_ptr, 0);                       // 累加器写回

    /* Step 3: HVX ReLU 后处理 — 确保输出无负值 */
    vp = (HVX_Vector *)out_ptr;
    for (int i = 0; i < 16; i++)
        vp[i] = Q6_Vh_vmax_VhVh(vp[i], v_zero);

    return QHPI_Success;
}
```

### ARM fallback

同一个 .cpp 文件通过 `#ifdef __hexagon__` 区分 DSP 和 ARM。ARM 版用标量 C++ 实现相同逻辑，用于 QNN graph 的 CPU 验证。

## ARM 端：构建和执行 QNN 图

```c
/* 1. 加载 QNN 后端 */
dlopen("libQnnHtp.so", ...);

/* 2. 注册自定义 op 包 — CPU 和 HTP 各一个 .so */
g_qnn.backendRegisterOpPackage(backend, "libHvxHmxMix_cpu.so", "HvxHmxMixInterfaceProvider", "CPU");
g_qnn.backendRegisterOpPackage(backend, "libHvxHmxMix_htp.so", "HvxHmxMixInterfaceProvider", "HTP");

/* 3. 构建图 */
g_qnn.graphCreate(context, "hvx_hmx_mix_graph", ...);

/* 4. 创建张量和节点 */
Qnn_OpConfig_t op = {
    .packageName = "HvxHmxMixPackage",
    .typeName    = "HvxHmxMix",
    .numOfInputs = 3,   // activation, weight, bias_cfg
    .numOfOutputs = 1,  // result
};
g_qnn.graphAddNode(graph, op);
g_qnn.graphFinalize(graph, ...);

/* 5. 执行 */
g_qnn.graphExecute(graph, inputs, 3, outputs, 1, ...);
```

## 编译

需要三个编译目标：

```bash
# 1. DSP 端 HTP 内核 (hexagon-clang++)
hexagon-clang++ -mv75 -mhvx -mhmx -shared -fPIC \
    -I $QNN_SDK/include/QNN \
    -o libHvxHmxMix_htp.so  src/dsp/*.cpp

# 2. ARM 端 CPU fallback (Android NDK clang++)
clang++ --target=aarch64-android -shared -fPIC \
    -I $QNN_SDK/include/QNN \
    -o libHvxHmxMix_cpu.so  src/dsp/*.cpp \
    -lQnnHtp -lQnnHtpPrepare

# 3. ARM 端测试程序
aarch64-clang -I $QNN_SDK/include/QNN \
    -o qnn_hvx_hmx_test  src/host/*.c -ldl -lm
```

或直接运行：

```bash
bash ch03-qnn-custom-op/build.sh
```

## 运行

```bash
bash ch03-qnn-custom-op/run_device.sh
```

脚本会推送以下文件到设备：

| 文件 | 用途 |
|------|------|
| `qnn_hvx_hmx_test` | ARM 端测试程序 |
| `libHvxHmxMix_cpu.so` | ARM 端 CPU fallback 包 |
| `libHvxHmxMix_htp.so` | DSP 端 HTP 内核包 |
| `libQnnHtp.so` | QNN HTP 后端运行时 |
| `libQnnHtpPrepare.so` | QNN HTP 图编译器 |
| `libQnnHtpV75Stub.so` | QNN HTP V75 ARM 端桩 |
| `libQnnHtpV75Skel.so` | QNN HTP V75 DSP 端骨架 |
| `libQnnSystem.so` | QNN 系统库 |

## 实验结果

```
========================================
  Chapter 3: QNN Custom Op (HVX+HMX)
========================================

[Load] QNN backend: ./libQnnHtp.so
[Load] OK

[Register] Custom op packages...
[Register] OK

[Graph] Building...
[Graph] OK

-- Test 1: HVX ReLU(1.0) -> HMX matmul -> HVX ReLU --
  act=1.0, wt=1.0
  ReLU(1.0)=1.0, matmul=1*1*32~=32, ReLU(32)=32
  out[0..3]: 0x5000(32.0) 0x5000(32.0) 0x5000(32.0) 0x5000(32.0)
  expected=32.0  maxErr=0.00  nonzero=1024/1024
  [PASS] ReLU(+) -> matmul -> ReLU

-- Test 2: HVX ReLU(-1.0) -> HMX matmul -> HVX ReLU --
  act=-1.0, wt=1.0
  ReLU(-1.0)=0.0, matmul=0*1*32=0, ReLU(0)=0
  out[0..3]: 0x0000(0.0) 0x0000(0.0) 0x0000(0.0) 0x0000(0.0)
  expected=0.0  maxErr=0.00  nonzero=0/1024
  [PASS] ReLU(-) -> matmul -> ReLU

========================================
  Results: 2 PASS / 0 FAIL
========================================
```

### 结果分析

**Test 1** — activation 全为 1.0（正数），ReLU 不改变。HMX matmul 得到 32.0，后处理 ReLU 也不改变。输出精确 0x5000 = 32.0。注意这里得到的是 32.0 而非 32.125——和第一二章略有不同，因为 QNN 内部的 HMX 调度路径不完全一样。

**Test 2** — activation 全为 -1.0（负数），前处理 ReLU 清零。HMX matmul 的 activation 全为 0，结果自然是 0。验证了 HVX 预处理 → HMX 计算的正确串联。

## 三章对比

| | 第一章 (模拟器) | 第二章 (真机直接) | 第三章 (QNN QHPI) |
|---|---|---|---|
| **环境** | hexagon-sim + H2 | CDSP + HAP API | CDSP + QNN HTP |
| **VTCM 管理** | H2 自动 | `HAP_compute_res` 手动 | QNN 自动 (`TCM_Only`) |
| **HMX 锁** | H2 自动 | `hmx_lock` 手动 | QNN 自动 (`RESOURCE_HMX`) |
| **HMX 指令** | inline asm | inline asm | Q6 intrinsics |
| **编译产物** | ELF | .so (FastRPC) | .so (QNN HTP 包) |
| **通信** | hexagon-sim | `run_main_on_hexagon` | QNN `graphExecute` |
| **适用场景** | 开发调试 | 自定义推理引擎 | QNN 生态集成 |

## 要点总结

1. **QHPI 是 QNN 自定义算子的推荐 API** — 纯 C 结构体注册，支持 VTCM 和 HMX
2. **`QHPI_MemLoc_TCM_Only`** — 告诉 QNN 把数据放在 VTCM，无需手动分配
3. **`QHPI_RESOURCE_HMX`** — 声明内核需要 HMX，QNN 自动管理上下文
4. **HMX 指令跨三章不变** — `mxmem`、`mxclracc`、VLIW packet 配对，无论什么环境都是同一套
5. **两个 .so** — DSP 端用 hexagon-clang++ 编译实际内核，ARM 端用 NDK 编译 CPU fallback
6. **`qhpi_init()` 是关键入口** — QNN 通过这个函数发现并注册你的自定义内核
