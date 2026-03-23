# 第四章：QNN 自定义算子 — 用 libnative 在 x86 上模拟 HVX/HMX

本章目标：使用 QNN 推荐的可移植写法，让同一份 HVX + HMX 自定义算子代码在 x86 主机上通过 `libnative` 直接运行，无需真机。

## 和第三章的根本区别

第三章用 `#ifdef __hexagon__` 把代码分成两条路径：

```cpp
// ch03 的写法 — 不可移植
#ifdef __hexagon__
    HVX_Vector v = Q6_V_vzero();              // 只能在 hexagon 上编译
    Q6_activation_hf_mxmem_RR(addr, 32767);   // 只能在 hexagon 上编译
#else
    for (int i = 0; ...) { sum += a[i]*b[i]; } // CPU fallback，完全不同的代码
#endif
```

本章改为**单一代码路径**，HVX/HMX intrinsics 在两个平台上都能编译运行：

```cpp
// ch04 的写法 — 可移植
#ifdef __hexagon__
#include <hvx_hexagon_protos.h>    // hexagon 编译器原生支持
#include <hmx_hexagon_protos.h>
#else
#include "hvx_hexagon_protos.h"    // libnative 提供 x86 模拟实现
#include "hmx_hexagon_protos.h"
#endif

// 以下代码在两个平台上完全相同
HVX_Vector v = Q6_V_vzero();
Q6_activation_hf_mxmem_RR(addr, 32767);
```

## libnative 是什么？

`libnative` 是 Hexagon SDK 自带的一个静态库（`libnative.a`），它在 x86 上用纯 C/C++ 实现了所有 HVX 和 HMX intrinsics 的功能等价模拟：

| intrinsic | hexagon 上 | x86 + libnative |
|---|---|---|
| `Q6_V_vzero()` | 硬件清零 HVX 寄存器 | C++ 实现：`memset(vec, 0, 128)` |
| `Q6_Vh_vmax_VhVh(a, b)` | 硬件 SIMD max | C++ 循环：逐元素 max |
| `Q6_activation_hf_mxmem_RR(addr, rt)` | 硬件 HMX 矩阵加载 | C++ 模拟 HMX 累加器状态 |
| `Q6_mxmem_AR_after_hf(ptr, rt)` | 硬件 HMX 输出写回 | C++ 模拟写回结果 |

关键：**不是 QEMU**，不是指令级仿真。`libnative` 是函数级模拟——每个 intrinsic 调用被替换为等价的 C++ 函数，直接在 x86 上执行。速度比 QEMU 快，但行为在数学上等价。

## 架构

```
x86 主机 (Linux)
┌───────────────────────────────────────────────────────────────┐
│                                                               │
│  qnn_hvx_hmx_test              libQnnHtp.so (x86_64)        │
│  ┌──────────────────┐          ┌────────────────────────┐    │
│  │ 1. dlopen backend│─────────→│ QNN HTP 后端 (x86)     │    │
│  │ 2. register op   │          │                        │    │
│  │ 3. build graph   │          │ 图编译 + 调度          │    │
│  │ 4. execute       │          │                        │    │
│  │ 5. verify result │          │                        │    │
│  └──────────────────┘          └───────────┬────────────┘    │
│                                            │ 调用             │
│                                            ▼                  │
│                          libHvxHmxMix_htp.so (x86_64)        │
│                          ┌────────────────────────────┐       │
│                          │ HVX ReLU → HMX matmul     │       │
│                          │ → HVX ReLU                 │       │
│                          │                            │       │
│                          │ Q6_V_vzero()          ───→ libnative.a 模拟  │
│                          │ Q6_Vh_vmax_VhVh()     ───→ libnative.a 模拟  │
│                          │ Q6_activation_hf_mxmem ──→ libnative.a 模拟  │
│                          │ Q6_mxmem_AR_after_hf  ───→ libnative.a 模拟  │
│                          └────────────────────────────┘       │
└───────────────────────────────────────────────────────────────┘
```

## 和真机的对比

| | 第三章 (真机) | 第四章 (libnative 模拟) |
|---|---|---|
| **运行环境** | Android 设备 (aarch64) | x86_64 Linux 主机 |
| **HVX/HMX 代码** | hexagon 硬件执行 | libnative C++ 模拟 |
| **代码路径** | `#ifdef __hexagon__` 分支 | **统一代码路径，无 #ifdef** |
| **编译 op 包** | hexagon-clang++ | clang++ + libnative.a |
| **QNN 后端** | libQnnHtp.so (aarch64) → CDSP | libQnnHtp.so (x86_64) |
| **部署方式** | adb push + adb shell | 本地直接运行 |

## 编译

```bash
# clang++ 编译 op 包，链接 libnative.a 提供 HVX/HMX 模拟
clang++ -std=c++17 -O2 -fPIC -shared \
    -D__HVXDBL__ -DUSE_OS_LINUX \
    -I $QNN_SDK/include/QNN \
    -I $HEXAGON_TOOLS/libnative/include \
    -o libHvxHmxMix_htp.so  src/dsp/*.cpp \
    -Wl,--whole-archive -lnative -Wl,--no-whole-archive -lpthread

# 测试程序
gcc -I $QNN_SDK/include/QNN -o qnn_hvx_hmx_test src/host/*.c -ldl -lm
```

关键编译选项：
- **`-D__HVXDBL__`** — libnative 要求，启用 128 字节 HVX 模式
- **`-Wl,--whole-archive -lnative`** — 链接整个 libnative.a，确保所有 intrinsic 符号可用
- **`-lpthread`** — libnative 内部依赖
- **不链接 `-lQnnHtp`** — op 包是插件，由后端 dlopen 加载，运行时才解析 QHPI 符号

或直接运行：

```bash
bash ch04-qnn-simulator/build.sh
```

## 运行

```bash
bash ch04-qnn-simulator/run_sim.sh
```

## 实验结果

```
========================================
  Chapter 4: QNN Custom Op on Simulator
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

结果和真机完全一致：**HVX ReLU + HMX matmul 在 x86 上通过 libnative 模拟得到了正确结果**。

## 源代码结构

```
ch04-qnn-simulator/
├── src/
│   ├── dsp/
│   │   ├── HvxHmxOp.cpp          # 统一代码路径 (无 #ifdef __hexagon__)
│   │   └── HvxHmxInterface.cpp   # QNN 包注册接口
│   └── host/
│       └── qnn_hvx_hmx_test.c    # x86 测试程序
├── build.sh                       # 编译 (clang++ + libnative)
├── run_sim.sh                     # 运行
├── expected_output/
│   └── sim_output.txt
└── README.md
```

## 要点总结

1. **libnative 是关键** — Hexagon SDK 自带，在 x86 上函数级模拟所有 HVX/HMX intrinsics
2. **统一代码路径** — 消除 `#ifdef __hexagon__`，同一份 intrinsic 代码在 hexagon 和 x86 上都能编译运行
3. **不是 QEMU** — libnative 是 C++ 函数库，比指令仿真快，结果数学等价
4. **QNN 框架正常工作** — QHPI 注册、图编译、VTCM 分配、graphExecute 全部正常
5. **开发工作流** — 先用 libnative 在 x86 上快速验证算子正确性，再部署到真机验证性能
6. **host 端需要 RTLD_GLOBAL** — dlopen 后端时用 `RTLD_GLOBAL`，让 op 包能解析 QHPI 符号
