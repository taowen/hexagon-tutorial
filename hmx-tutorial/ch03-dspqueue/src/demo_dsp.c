/*
 * ch05: DSP 端 — dspqueue 消息循环
 *
 * 对应 llama.cpp 的 ggml/src/ggml-hexagon/htp/main.c
 *
 * 核心模式完全一样：
 *   1. open:  电源配置（锁频 + 关闭 DCVS）
 *   2. start: dspqueue_import（导入 ARM 创建的队列）
 *   3. 回调:  dspqueue_read_noblock → switch(op) → 执行 → dspqueue_write
 *   4. stop:  dspqueue_close
 *
 * llama.cpp 的 DSP 端有 ~15000 行代码（29 种算子 + HMX/HVX 内核 + DMA + VTCM 管理）。
 * 我们用 ~100 行代码展示相同的骨架。
 */

#define FARF_ERROR 1
#define FARF_HIGH 1
#include <HAP_farf.h>
#include <string.h>
#include <HAP_power.h>
#include <HAP_perf.h>
#include <AEEStdErr.h>
#include "dspqueue_demo.h"
#include "dspqueue_demo_shared.h"
#include "dspqueue.h"

/* ---------- DSP 上下文 ----------
 *
 * 对应 llama.cpp 的 struct htp_context (htp-ctx.h)
 * llama.cpp 的 context 还包含 VTCM、DMA 引擎、worker pool 等。
 */
struct demo_context {
    dspqueue_t queue;
    uint64_t   process_time;   /* 累计 DSP 处理时间 */
};


/* ========== 算子实现 ==========
 *
 * 对应 llama.cpp 的 htp/binary-ops.c, htp/unary-ops.c 等
 * llama.cpp 用 HVX intrinsics 做向量化；我们用标量循环演示模式。
 */

/* OP_SCALE: out[i] = in[i] * factor
 * 对应 llama.cpp 的 proc_unary_req → HTP_OP_SCALE */
static void op_scale(const uint8_t *in, uint8_t *out, uint32_t n, uint8_t factor) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] = (uint8_t)(in[i] * factor);
    }
}

/* OP_ADD: out[i] = a[i] + b[i]
 * 对应 llama.cpp 的 proc_binary_req → HTP_OP_ADD */
static void op_add(const uint8_t *a, const uint8_t *b, uint8_t *out, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] = (uint8_t)(a[i] + b[i]);
    }
}


/* ========== FastRPC 生命周期 ==========
 *
 * 对应 llama.cpp 的 htp_iface_open / htp_iface_close
 */

AEEResult dspqueue_demo_open(const char *uri, remote_handle64 *handle) {

    struct demo_context *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return AEE_ENOMEMORY;
    *handle = (remote_handle64)ctx;

    /* 电源配置 — 对应 llama.cpp htp_iface_open 中的 HAP_power_set
     *
     * llama.cpp 用 HAP_power_set_DCVS_v3 锁到 VCORNER_MAX 并禁用睡眠。
     * SDK 示例用 DCVS_v2 设到 TURBO。效果一样：全速运行，禁用降频。
     */
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_apptype;
    request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    HAP_power_set((void *)ctx, &request);

    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_DCVS_v2;
    request.dcvs_v2.dcvs_enable = FALSE;
    request.dcvs_v2.set_dcvs_params = TRUE;
    request.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_DISABLE;
    request.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_DISABLE;
    request.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_TURBO;
    request.dcvs_v2.set_latency = TRUE;
    request.dcvs_v2.latency = 40;
    return HAP_power_set((void *)ctx, &request);
}

AEEResult dspqueue_demo_close(remote_handle64 handle) {
    struct demo_context *ctx = (struct demo_context *)handle;
    if (!ctx) return AEE_EBADPARM;
    if (ctx->queue) return AEE_EITEMBUSY;
    free(ctx);
    return AEE_SUCCESS;
}


/* ========== dspqueue 消息回调 ==========
 *
 * 这是整个 DSP 端的核心 — 对应 llama.cpp 的 htp_packet_callback (htp/main.c)
 *
 * llama.cpp 的 callback 有 ~200 行，dispatch 29 种 op。
 * 我们用同样的结构，dispatch 2 种 op。
 */

static void error_callback(dspqueue_t queue, int error, void *context) {
    FARF(ERROR, "dspqueue error: 0x%08x", (unsigned)error);
}

static void packet_callback(dspqueue_t queue, int error, void *context) {

    struct demo_context *ctx = (struct demo_context *)context;

    /* 循环读取所有可用消息 — 和 llama.cpp 完全一样 */
    while (1) {
        struct demo_req req;
        struct demo_rsp rsp;
        uint32_t flags, msg_len, n_bufs;
        struct dspqueue_buffer bufs[DEMO_MAX_BUFFERS];
        struct dspqueue_buffer rsp_bufs[DEMO_MAX_BUFFERS];

        int err = dspqueue_read_noblock(queue, &flags,
                                        DEMO_MAX_BUFFERS, &n_bufs, bufs,
                                        sizeof(req), &msg_len, (uint8_t *)&req);
        if (err == AEE_EWOULDBLOCK) return;   /* 队列空了 */
        if (err != 0) {
            FARF(ERROR, "dspqueue_read failed: 0x%08x", (unsigned)err);
            return;
        }

        uint64_t t1 = HAP_perf_get_time_us();

        /*
         * Op dispatch — 对应 llama.cpp htp_packet_callback 中的 switch(req.op)
         *
         * llama.cpp:
         *   case HTP_OP_MUL_MAT: proc_hmx_matmul_req(...); break;
         *   case HTP_OP_ADD:     proc_binary_req(...);      break;
         *   case HTP_OP_SCALE:   proc_unary_req(...);       break;
         *   ... 29 种 ...
         */
        switch (req.op) {

            case OP_SCALE:
                /* 1 input buffer + 1 output buffer */
                if (n_bufs >= 2) {
                    op_scale(bufs[0].ptr, bufs[1].ptr, req.n_elem,
                             (uint8_t)req.param);
                }
                break;

            case OP_ADD:
                /* 2 input buffers + 1 output buffer */
                if (n_bufs >= 3) {
                    op_add(bufs[0].ptr, bufs[1].ptr, bufs[2].ptr, req.n_elem);
                }
                break;

            case OP_ECHO:
                /* Keepalive — 不做计算 */
                break;

            default:
                FARF(ERROR, "Unknown op %u", req.op);
                break;
        }

        uint64_t t2 = HAP_perf_get_time_us();
        ctx->process_time += (t2 - t1);

        /* 写回响应 — 对应 llama.cpp 的 send_htp_rsp()
         *
         * 注意 cache 维护标志：DSP 写完的 buffer 需要 FLUSH_SENDER
         * 让 ARM 能看到最新数据。这就是零拷贝的关键。
         */
        rsp.op = req.op;
        rsp.status = 0;

        uint32_t n_rsp_bufs = 0;
        memset(rsp_bufs, 0, sizeof(rsp_bufs));

        if (req.op == OP_SCALE && n_bufs >= 2) {
            /* 释放 input ref，flush output */
            rsp_bufs[0].fd = bufs[0].fd;
            rsp_bufs[0].flags = DSPQUEUE_BUFFER_FLAG_DEREF;
            rsp_bufs[1].fd = bufs[1].fd;
            rsp_bufs[1].flags = DSPQUEUE_BUFFER_FLAG_DEREF
                              | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
                              | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
            n_rsp_bufs = 2;
        } else if (req.op == OP_ADD && n_bufs >= 3) {
            rsp_bufs[0].fd = bufs[0].fd;
            rsp_bufs[0].flags = DSPQUEUE_BUFFER_FLAG_DEREF;
            rsp_bufs[1].fd = bufs[1].fd;
            rsp_bufs[1].flags = DSPQUEUE_BUFFER_FLAG_DEREF;
            rsp_bufs[2].fd = bufs[2].fd;
            rsp_bufs[2].flags = DSPQUEUE_BUFFER_FLAG_DEREF
                              | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
                              | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
            n_rsp_bufs = 3;
        }

        dspqueue_write(queue, 0, n_rsp_bufs, rsp_bufs,
                        sizeof(rsp), (const uint8_t *)&rsp,
                        DSPQUEUE_TIMEOUT_NONE);
    }
}


/* ========== dspqueue_demo_start / stop ==========
 *
 * 对应 llama.cpp 的 htp_iface_start / htp_iface_stop
 */

AEEResult dspqueue_demo_start(remote_handle64 handle, uint64 dsp_queue_id) {
    struct demo_context *ctx = (struct demo_context *)handle;
    if (!ctx) return AEE_EBADPARM;
    if (ctx->queue) return AEE_EITEMBUSY;
    ctx->process_time = 0;

    /* 导入 ARM 端创建的 dspqueue — 和 llama.cpp 完全一样 */
    int err = dspqueue_import(dsp_queue_id, packet_callback, error_callback,
                              ctx, &ctx->queue);
    if (err) {
        FARF(ERROR, "dspqueue_import failed: 0x%08x", (unsigned)err);
        return err;
    }
    return AEE_SUCCESS;
}

AEEResult dspqueue_demo_stop(remote_handle64 handle, uint64 *process_time) {
    struct demo_context *ctx = (struct demo_context *)handle;
    if (!ctx) return AEE_EBADPARM;
    if (!ctx->queue) return AEE_EBADSTATE;

    int err = dspqueue_close(ctx->queue);
    ctx->queue = NULL;
    if (err) return err;

    *process_time = ctx->process_time;
    return AEE_SUCCESS;
}


/* ========== FastRPC 对比路径 ==========
 *
 * do_op 走标准 FastRPC（每次内核态切换），用于和 dspqueue 对比。
 * 对应 SDK 示例的 dspqueue_sample_do_process。
 */

AEEResult dspqueue_demo_do_op(remote_handle64 handle,
                               const uint8 *input, int input_len,
                               uint8 *output, int output_len,
                               uint32 op, uint64 *process_time) {
    uint64_t t1 = HAP_perf_get_time_us();

    switch (op) {
        case OP_SCALE:
            op_scale(input, output, input_len, 2);
            break;
        case OP_ADD:
            /* FastRPC 路径 input 和 output 一样大小，做 self-add */
            op_add(input, input, output, input_len);
            break;
        default:
            return AEE_EBADPARM;
    }

    uint64_t t2 = HAP_perf_get_time_us();
    *process_time = t2 - t1;
    return AEE_SUCCESS;
}
