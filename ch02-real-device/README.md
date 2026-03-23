# 第二章：在真机上跑 HVX + HMX

本章目标：把第一章的 HVX + HMX 混合代码从模拟器搬到真机（骁龙 8 Gen 3）的 CDSP 上运行。

## 模拟器 vs 真机：什么变了？

第一章用 H2 Hypervisor + hexagon-sim 跑在 x86 上。真机上没有 H2，取而代之的是 QuRT 操作系统和 HAP（Hexagon Application Programming）API。

| | 模拟器 (第一章) | 真机 (本章) |
|---|---|---|
| **OS** | H2 Hypervisor | QuRT (手机自带) |
| **编译产物** | ELF 可执行文件 | 共享库 (.so) |
| **加载方式** | hexagon-sim 直接加载 | `run_main_on_hexagon` 通过 FastRPC 加载 |
| **架构标志** | `-mv73` | `-mv75` |
| **VTCM 获取** | `h2_info(INFO_VTCM_BASE)` | `HAP_compute_res_acquire()` |
| **HVX 上下文** | `h2_vecaccess_acquire()` | 自动可用 |
| **HMX 上下文** | `h2_mxaccess_acquire()` | `HAP_compute_res_hmx_lock()` |
| **上电** | H2 booter `--ext_power 1` | `HAP_power_set()` |
| **HMX 指令** | inline asm（相同） | inline asm（相同） |

关键发现：**HMX 的 inline asm 代码完全相同**——`mxmem` 指令、`mxclracc`、VLIW packet pairing 等，在模拟器和真机上是同一套。变化的只是"外围"：怎么获取 VTCM、怎么上电、怎么加载程序。

## 真机运行架构

```
ARM 端 (Android)                      DSP 端 (CDSP)
┌────────────────────┐  FastRPC   ┌─────────────────────────┐
│ run_main_on_hexagon │ ────────→ │ libtest_hvx_hmx_device.so │
│ (SDK 自带工具)      │           │                           │
│                    │           │ main() {                  │
│                    │           │   HAP_power_set(HMX)      │
│                    │           │   HAP_compute_res(VTCM)   │
│                    │           │   HVX fill → HMX matmul   │
│                    │           │   HVX ReLU                │
│                    │           │ }                         │
└────────────────────┘           └─────────────────────────┘
```

`run_main_on_hexagon` 是 Hexagon SDK 自带的工具，它做一件事：通过 FastRPC 把一个 `.so` 加载到 CDSP 上，调用其 `main()` 函数。参数 `3` 表示 CDSP（0=ADSP, 3=CDSP）。

## 源代码解读

完整代码见 [`test_hvx_hmx_device.c`](test_hvx_hmx_device.c)。和第一章相比，HMX/HVX 的计算代码完全一样，变化的是初始化部分。

### 上电 HVX/HMX（HAP_power_set）

```c
static int power_ctx;  /* 非 NULL 上下文指针 */

/* 设置 client class */
HAP_power_request_t req = {0};
req.type = HAP_power_set_apptype;
req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
HAP_power_set((void *)&power_ctx, &req);

/* DCVS 性能模式 — 锁定最高频率 */
req.type = HAP_power_set_DCVS_v3;
req.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
req.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
/* ... */
HAP_power_set((void *)&power_ctx, &req);

/* 上电 HVX */
req.type = HAP_power_set_HVX;
req.hvx.power_up = 1;
HAP_power_set((void *)&power_ctx, &req);

/* 上电 HMX */
req.type = HAP_power_set_HMX;
req.hmx.power_up = 1;
HAP_power_set((void *)&power_ctx, &req);
```

这段代码照搬了 llama.cpp 的模式。注意 `power_ctx` 必须是非 NULL 指针，传 NULL 会报 "bad parameter"。

### 分配 VTCM + 锁定 HMX（HAP_compute_res）

```c
/* 查询 VTCM 大小 */
unsigned int vtcm_size = 8 * 1024 * 1024;
HAP_compute_res_query_VTCM(0, &vtcm_size, NULL, NULL, NULL);

/* 分配 VTCM + HMX */
compute_res_attr_t attr;
HAP_compute_res_attr_init(&attr);
HAP_compute_res_attr_set_vtcm_param(&attr, vtcm_size, 1);
HAP_compute_res_attr_set_hmx_param(&attr, 1);
/* 注意: 不要设置 cache_mode 和 serialize! */

unsigned int ctx_id = HAP_compute_res_acquire(&attr, 100000);
void *vtcm_base = HAP_compute_res_attr_get_vtcm_ptr(&attr);
HAP_compute_res_hmx_lock(ctx_id);
```

**踩坑警告**：绝对不要调用 `HAP_compute_res_attr_set_cache_mode(&attr, 1)` 或 `HAP_compute_res_attr_set_serialize(&attr, 0)`。这两个选项会导致 `hmx_lock` 返回错误 1，之后执行 `mxmem` 指令会让 CDSP 永久挂死。这是我们（和很多人）之前误以为"HMX 在 Unsigned PD 下不可用"的根本原因。

### HMX/HVX 计算代码

和第一章完全相同：

```c
/* HVX 填充 */
hvx_fill_f16(act, F16_ONE, TILE_ELEMS);
hvx_fill_f16(wt, F16_TWO, TILE_ELEMS);
hvx_fill_scales(scl, F16_ONE);

/* HMX matmul */
hmx_set_scales(scl);
hmx_clear_acc();
hmx_load_tile_pair(act, wt);
hmx_store_acc(out);

/* HVX ReLU */
hvx_relu_f16(out, TILE_ELEMS);
```

## 编译

```bash
hexagon-clang -mv75 -O2 \
    -mhvx -mhvx-length=128B \
    -mhmx \
    -shared -fPIC \                           # 共享库, 不是 ELF
    -I $HEXAGON_SDK/incs \                    # HAP API 头文件
    -I $HEXAGON_SDK/incs/stddef \
    -I $HEXAGON_SDK/rtos/qurt/computev75/include/qurt \
    test_hvx_hmx_device.c \
    -o build/libtest_hvx_hmx_device.so
```

和第一章编译的关键区别：

| | 第一章 (模拟器) | 第二章 (真机) |
|---|---|---|
| 输出格式 | ELF 可执行文件 | `-shared -fPIC` 共享库 |
| 架构 | `-mv73` | `-mv75` |
| OS 运行时 | `-moslib=h2` | 不需要（QuRT 自带） |
| 链接 | `-Wl,-L,h2-install/lib` | 不需要 |
| 头文件 | H2 install/include | SDK incs/ |

## 运行

```bash
# 编译
bash ch02-real-device/build.sh

# 推送到设备并运行
bash ch02-real-device/run_device.sh
```

`run_device.sh` 做三件事：
1. `adb push` 把 .so 和 `run_main_on_hexagon` 推到设备
2. 在设备上执行 `run_main_on_hexagon 3 libtest_hvx_hmx_device.so`（`3` = CDSP）
3. 从 `adb logcat` 提取 DSP 端的 FARF 日志

## 预期结果

```
========================================
  Chapter 2: HVX + HMX on Real Device
========================================
[Power] Setting up HVX + HMX...
[Power] OK
[Init] Allocating VTCM + HMX lock...
[Init] VTCM total = 8388608 bytes (8192 KB)
[Init] VTCM=FF000000  HMX locked

-- Test 1: HVX fill -> HMX matmul ----------
  act: 0x3C00 0x3C00 0x3C00 0x3C00
  wt : 0x4000 0x4000 0x4000 0x4000
  out: 0x5410 0x5410 0x5410 0x5410
  result=65.0  expected~=64.0
[PASS] Test 1

-- Test 2: HMX matmul -> HVX ReLU ----------
  matmul: 0xCFC0 0xCFC0 (-31.0)
  relu:   0x0000 0x0000 (0.0)
[PASS] Test 2

-- Test 3: Multi-tile accumulation ---------
  out: 0x5410 0x5410 0x5410 0x5410
  result=65.0  expected~=64.0
[PASS] Test 3

========================================
  Results: 3 PASS / 0 FAIL
========================================
```

模拟器和真机的输出是一致的——HMX 的 F16 精度特性（32.0→32.125, 64.0→65.0）在两个环境下完全相同。

## 不需要签名

在 SM8650 (V75) 上，`run_main_on_hexagon` 使用 **Unsigned PD（无签名进程域）**加载 .so。不需要 `elfsigner.py` 生成 test signature，.so 可以直接在任何同型号设备上运行。

## 踩坑记录

### 1. HAP_compute_res 的 cache_mode / serialize 选项

**这是最关键的坑。** 很多人（包括我们）一度认为 HMX 在 Unsigned PD 下无法使用，因为读取 HMX 累加器输出时 CDSP 会永久挂死。

根本原因：`HAP_compute_res_attr_set_cache_mode(&attr, 1)` 和 `HAP_compute_res_attr_set_serialize(&attr, 0)` 会导致 `hmx_lock` 返回错误 1（但不会 crash），之后执行 `mxmem` 指令时才会挂死。

修复：**不要设置这两个选项**。只需要 `set_vtcm_param` 和 `set_hmx_param`。

### 2. HAP_power_set 需要非 NULL 上下文

`HAP_power_set(NULL, &req)` 会返回 "bad parameter" 错误。必须传一个非 NULL 指针：

```c
static int power_ctx;
HAP_power_set((void *)&power_ctx, &req);  // OK
```

### 3. Rt 参数必须精确

HMX 的 `mxmem` 指令的 Rt 参数是 streaming region size = `n_tiles * 2048 - 1`。

- 单个 tile: `Rt = 2047`
- 用 `32767` 这样的大值会导致跨 4MB VTCM bank 边界的总线错误 (0x2601)

### 4. 输出在 logcat 里

DSP 端的 `FARF(ALWAYS, ...)` 输出通过 `adb logcat` 查看，不是 stdout。需要用 grep 过滤 `CDSP0.*DU` 行。

### 5. printf vs FARF

DSP 端也可以用 `printf`，但 `FARF` 是推荐的方式——它支持日志级别（ALWAYS/HIGH/MEDIUM/LOW），且在生产环境可以被关闭。本章用 `FARF(ALWAYS, ...)` 确保日志始终输出。

## llama.cpp 做了什么更多的事

本章只是最基本的真机验证。llama.cpp 的 Hexagon 后端在此基础上做了更多：

| 能力 | 本章 | llama.cpp |
|------|------|-----------|
| 通信方式 | `run_main_on_hexagon` (FastRPC) | `dspqueue` 共享内存队列 |
| VTCM 管理 | 固定偏移分配 | Bump allocator + 双缓冲 |
| 数据搬运 | 直接在 VTCM 上操作 | DMA (DDR↔VTCM) 异步传输 |
| 量化支持 | F16 only | Q4_0/Q8_0/MXFP4 + HVX 反量化 |
| 算子覆盖 | matmul + ReLU | 20+ 算子全 DSP 执行 |
| 流水线 | 无 | DMA→HVX 反量化→HMX matmul→写回 |

用 `dspqueue` 替代 FastRPC 是性能关键——FastRPC 每次调用 ~0.38ms 固定开销，而 `dspqueue` 是微秒级的共享内存写入。对 LLM 推理来说，每个 token 需要 ~196 次 matmul 调用，FastRPC 的开销会让 decode 速度骤降。
