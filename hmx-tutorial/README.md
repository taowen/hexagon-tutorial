# HMX 教程：从指令到 LLM 推理

直接用 HVX/HMX 指令编程，手动管理 VTCM 和 DMA，逐步搭建出完整的 LLM 推理和训练能力。

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

## 第三章：dspqueue vs FastRPC

**目录**: [`ch03-dspqueue/`](ch03-dspqueue/)

以 llama.cpp 的 Hexagon 后端为切入点，分析 FastRPC 在高频小算子场景的延迟瓶颈（364μs/次），对比 dspqueue 的共享内存队列方案（61μs/次）。理解 ARM↔DSP 通信的本质开销。

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

---

## 第四章：VTCM 内存管理与 DMA

**目录**: [`ch04-vtcm-memory/`](ch04-vtcm-memory/)

深入 VTCM（1 周期延迟的片上高速内存，v75 上 8MB）。实现 DMA/UDMA 异步搬运、双缓冲流水线、bump/pool 自定义分配器。理解如何用 8MB 片上内存高效服务远超其容量的模型数据。

```c
// UDMA 异步搬运：硬件在后台搬数据，HVX 同时计算 —— 17 倍加速
Q6_dmstart_A((void *)&desc);                           // 启动 DMA（立即返回）
hvx_vadd(compute_a, compute_b, compute_c, DMA_SIZE);   // HVX 同时计算
Q6_R_dmwait();                                          // DMA 早就搬完了
// 串行：3701μs → 重叠：217μs → 17x 加速
```

---

## 第五章：HMX 矩阵乘法

**目录**: [`ch05-hmx/`](ch05-hmx/)

从 tile 基础到生产级优化。4 个实验逐步深入：tile 手动操作 → 权重布局优化（L0→L1→L2 逐级加速）→ VTCM 流式处理 LLM 级大矩阵 → hexkl vs 直接 ASM 的完整管线对比。理解 weight layout 转换、chunk 分块策略和瓶颈图谱。

```c
// hexkl_micro 手动 tile 操作：DDR→VTCM→AH/WH 格式→HMX 计算→回写
hexkl_micro_hmx_rm_to_ah_f16(vtcm, act_off, A, row, k, M, K);  // 激活→AH 格式
hexkl_micro_hmx_rm_to_wh_f16(vtcm, wt_off, W, k, col, K, N);   // 权重→WH 格式
hexkl_micro_hmx_acc_clear_f16(vtcm, &cfg);                       // 清零累加器
hexkl_micro_hmx_mm_f16(vtcm, act_off, wt_off, &cfg);            // tile 乘法
```
```
// 瓶颈图谱：DDR I/O → VTCM → hexkl API → 纯 HMX，跨越三个数量级
Exp 2 L0 (full DDR path):           ~1200 us    DDR I/O 占 99.5%
Exp 4 Method A (VTCM, cache hot):     ~6 us    HMX compute 占 84%
Exp 4 Method C (batch ASM):            ~3 us    ASM 批量加载优化
Exp 4 compute only:                    ~1 us    纯 HMX 吞吐 28,000 GFLOPS
```

---

## 第六章：KV Cache 与数据布局优化

**目录**: [`ch06-kv-cache/`](ch06-kv-cache/)

LLM 推理中 KV cache 的数据布局问题：每次 attention 都要把行优先数据转成 HMX weight layout，转换开销吞掉计算收益。NativeKV 方案直接以 weight layout 存储 KV cache，跳过转换，但需要 v75+ 硬件支持。

```c
// SmartMask：每步都要 rm_to_wh 转换（布局转换开销大）
hexkl_micro_hmx_rm_to_wh_f16(vtcm, offset, K_rm, row, col, ctx_size);

// NativeKV：K 已经是 WH 格式，直接拷贝 tile（跳过转换）
hexkl_micro_hmx_copy_psubmatrix_to_f16_weight(vtcm, offset, K_wh, row, col, ...);
```

---

## 第七章：llama.cpp 真机推理

**目录**: [`ch07-llama-cpp-run/`](ch07-llama-cpp-run/)

编译 llama.cpp 的 Hexagon 后端，在骁龙 8 Gen 3 和 8 Elite 上实测 4 个模型（0.6B-4B）的 NPU vs CPU 性能。发现 4B 是 NPU prefill 反超 CPU 的拐点，decode 始终 CPU 更快——通信延迟、数据布局转换、小矩阵低利用率三个瓶颈的真实体现。

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

---

## 第八章：MNIST DSP 训练

**目录**: [`ch08-mnist-train/`](ch08-mnist-train/)

从推理到训练：在 DSP 上实现完整的 MNIST 神经网络训练（前向 + 反向 + SGD）。用 dspqueue 把整个 batch 融合为一条消息，消除 ARM↔DSP 同步开销，HVX 向量化实现 4.29 倍加速。

```c
// 整个 batch 的 forward + backward + SGD 在一条 dspqueue 消息里完成
dspqueue_write(queue, 0, n_bufs, bufs, ...);  // 一次调用，DSP 内部跑完整个 batch
dspqueue_read(queue, &flags, ...);             // 等结果
// 对比 FastRPC 方案每个 op 都要一次跨域调用 —— 4.29x 加速
```
