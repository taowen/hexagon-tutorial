# 第四章：QNN 自定义算子 — 在模拟器上运行 (无需真机)

本章目标：把第三章的 QNN QHPI 自定义算子（HVX ReLU + HMX matmul）在 x86 主机上运行，通过 QNN 内置的 QEMU Hexagon 模拟器，无需连接 Qualcomm 设备。

## 为什么需要模拟器？

- **无设备开发**：不需要 Qualcomm 手机或开发板，任何 x86_64 Linux 机器都能跑
- **快速迭代**：省去 adb push / 远程执行的等待时间，编译即运行
- **CI 友好**：可以在 CI/CD 流水线中自动测试自定义算子，无需物理设备
- **调试方便**：x86 环境下可以用标准 Linux 调试工具（gdb、valgrind、asan）

## 架构

```
x86 主机 (Linux)
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  qnn_hvx_hmx_test          libQnnHtp.so (x86_64)              │
│  ┌──────────────────┐      ┌──────────────────────────┐        │
│  │ 1. dlopen backend│─────→│ QNN HTP 后端 (x86 版本)  │        │
│  │ 2. register op   │      │                          │        │
│  │ 3. build graph   │      │  libQnnHtpQemu.so        │        │
│  │ 4. execute       │      │  ┌──────────────────┐    │        │
│  │ 5. verify result │      │  │ QEMU Hexagon 仿真│    │        │
│  └──────────────────┘      │  │                  │    │        │
│                            │  │ libHvxHmxMix_htp │    │        │
│  libHvxHmxMix_cpu.so      │  │ .so (hex-v75)    │    │        │
│  (CPU fallback, x86)      │  │                  │    │        │
│                            │  │ HVX ReLU → HMX  │    │        │
│                            │  │ matmul → HVX    │    │        │
│                            │  └──────────────────┘    │        │
│                            └──────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

关键点：
- **x86 版 libQnnHtp.so** 内部使用 **libQnnHtpQemu.so** 来仿真 Hexagon DSP
- QEMU 加载并执行 hexagon-v75 编译的 `libHvxHmxMix_htp.so`
- 从应用程序视角，API 调用方式和真机完全一样

## 和第三章的区别

| | 第三章 (真机) | 第四章 (模拟器) |
|---|---|---|
| **运行环境** | Android 设备 (aarch64) | x86_64 Linux 主机 |
| **QNN 后端** | libQnnHtp.so (aarch64) | libQnnHtp.so (x86_64) + QEMU |
| **DSP 代码** | 完全相同 | 完全相同 |
| **CPU fallback** | aarch64 (Android NDK) | x86_64 (系统 g++) |
| **测试程序** | aarch64 (Android NDK) | x86_64 (系统 gcc) |
| **Op 包注册** | CPU + HTP 两个 .so | 仅 x86 .so, target=NULL |
| **部署方式** | adb push + adb shell | 本地直接运行 |
| **DSP 执行** | 真实 CDSP 硬件 | QEMU Hexagon 仿真 |

**DSP 端代码是字节级别完全相同的** — `HvxHmxOp.cpp` 和 `HvxHmxInterface.cpp` 从第三章原封不动复制，hexagon-clang++ 编译参数也一样。区别仅在于：
1. CPU fallback .so 用系统编译器而非 Android NDK
2. 测试程序用系统编译器而非 Android NDK
3. 运行时链接 x86_64 版 QNN 库而非 aarch64 版

### Op 包注册的关键区别

在真机上（ch03），需要分别注册 CPU 和 HTP 两个 .so：

```c
backendRegisterOpPackage(backend, "libHvxHmxMix_cpu.so", "...", "CPU");
backendRegisterOpPackage(backend, "libHvxHmxMix_htp.so", "...", "HTP");
```

在 x86 模拟器上，只需注册一个 x86 编译的 .so，target 设为 `NULL`：

```c
backendRegisterOpPackage(backend, "libHvxHmxMix_cpu.so", "...", NULL);
```

原因：x86 HTP 后端不支持 "HTP" target（会报 `RouterX86 not support Op package target HTP`），也不能直接 dlopen hexagon ELF。x86 后端通过 `qhpi_init()` 入口点从 x86 .so 中获取算子定义，然后通过 QEMU 执行实际的 hexagon 代码。

## 源代码结构

```
ch04-qnn-simulator/
├── src/
│   ├── dsp/
│   │   ├── HvxHmxOp.cpp          # 算子内核 (和 ch03 完全相同)
│   │   └── HvxHmxInterface.cpp   # QNN 包注册接口 (和 ch03 完全相同)
│   └── host/
│       └── qnn_hvx_hmx_test.c    # x86 端测试 (标题改为 Chapter 4)
├── build.sh                       # 编译脚本 (x86_64 目标)
├── run_sim.sh                     # 本地运行脚本 (无需 adb)
├── expected_output/
│   └── sim_output.txt
└── README.md
```

## 编译

三个编译目标：

```bash
# 1. DSP 端 HTP 内核 — 和 ch03 完全一样, hexagon-clang++ 编译
hexagon-clang++ -mv75 -mhvx -mhmx -shared -fPIC \
    -I $QNN_SDK/include/QNN \
    -o libHvxHmxMix_htp.so  src/dsp/*.cpp

# 2. x86_64 CPU fallback — 用系统 g++ (不是 Android NDK!)
g++ -shared -fPIC \
    -I $QNN_SDK/include/QNN \
    -o libHvxHmxMix_cpu.so  src/dsp/*.cpp \
    -L $QNN_SDK/lib/x86_64-linux-clang -lQnnHtp -lHtpPrepare

# 3. x86_64 测试程序 — 用系统 gcc
gcc -I $QNN_SDK/include/QNN \
    -o qnn_hvx_hmx_test  src/host/*.c -ldl -lm
```

或直接运行：

```bash
bash ch04-qnn-simulator/build.sh
```

## 运行

```bash
bash ch04-qnn-simulator/run_sim.sh
```

脚本会设置 `LD_LIBRARY_PATH` 并在本地直接执行测试程序。需要的运行时库：

| 库 | 来源 | 用途 |
|---|---|---|
| `libHvxHmxMix_cpu.so` | 本地编译 (x86_64) | CPU fallback 包 |
| `libHvxHmxMix_htp.so` | 本地编译 (hexagon-v75) | DSP 端 HTP 内核 |
| `libQnnHtp.so` | QNN SDK (x86_64) | QNN HTP 后端 |
| `libQnnHtpQemu.so` | QNN SDK (x86_64) | QEMU Hexagon 仿真器 |
| `libHtpPrepare.so` | QNN SDK (x86_64) | HTP 图编译器 |
| `libQnnHtpV75Skel.so` | QNN SDK (hexagon-v75) | HTP V75 DSP 骨架 |

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

模拟器输出和真机完全一致。

## 要点总结

1. **QNN 提供 x86_64 版后端库** — `libQnnHtp.so` (x86) 内部使用 QEMU 仿真 Hexagon
2. **DSP 代码零修改** — `HvxHmxOp.cpp` 和 `HvxHmxInterface.cpp` 与 ch03 完全相同
3. **Op 包注册方式不同** — x86 上用 `NULL` target 注册单个 x86 .so，而非分别注册 CPU/HTP
4. **编译改动极小** — 只需把 Android NDK 换成系统编译器，链接路径换成 x86_64
5. **无需 adb** — 所有操作在本地完成，适合开发调试和 CI
6. **QEMU 仿真有局限** — 性能不代表真实硬件，某些时序相关的行为可能不同
7. **开发工作流推荐** — 先用模拟器快速验证正确性，再到真机验证性能
