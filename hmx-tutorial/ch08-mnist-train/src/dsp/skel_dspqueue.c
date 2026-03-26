/*
 * Step 3: DSP receives matmul requests via dspqueue (userspace, no kernel transition).
 *
 * Handles dspqueue messages but only OP_MATMUL.
 * The FastRPC matmul function is stubbed out.
 */

#include "dsp/skel_common.h"
#include "dsp/hvx_matmul.h"
#include "common/protocol.h"

/* ---------- DSP context ---------- */
struct mnist_context {
    dspqueue_t queue;
    uint64_t   process_time;   /* cumulative DSP processing time (us) */
};


/* ========== FastRPC lifecycle ========== */

AEEResult mnist_train_open(const char *uri, remote_handle64 *handle) {
    return mnist_train_open_impl(uri, handle, sizeof(struct mnist_context));
}

AEEResult mnist_train_close(remote_handle64 handle) {
    struct mnist_context *ctx = (struct mnist_context *)handle;
    if (!ctx) return AEE_EBADPARM;
    if (ctx->queue) return AEE_EITEMBUSY;
    free(ctx);
    return AEE_SUCCESS;
}


/* ========== dspqueue message callback ========== */

static void packet_callback(dspqueue_t queue, int error, void *context) {
    struct mnist_context *ctx = (struct mnist_context *)context;

    while (1) {
        struct matmul_req msg;
        uint32_t flags, msg_len, n_bufs;
        struct dspqueue_buffer bufs[MATMUL_MAX_BUFFERS];

        int err = dspqueue_read_noblock(queue, &flags,
                                        MATMUL_MAX_BUFFERS, &n_bufs, bufs,
                                        sizeof(msg), &msg_len, (uint8_t *)&msg);
        if (err == AEE_EWOULDBLOCK) return;
        if (err != 0) {
            FARF(ERROR, "dspqueue_read failed: 0x%08x", (unsigned)err);
            return;
        }

        if (msg.op == OP_MATMUL && n_bufs >= 3) {
            uint64_t t1 = HAP_perf_get_time_us();
            do_matmul((float *)bufs[2].ptr,
                      (const float *)bufs[0].ptr,
                      (const float *)bufs[1].ptr,
                      msg.m, msg.n, msg.k,
                      msg.transpose, msg.accumulate);
            ctx->process_time += (HAP_perf_get_time_us() - t1);

            struct matmul_rsp rsp = { OP_MATMUL, 0 };
            struct dspqueue_buffer rsp_bufs[3];
            memset(rsp_bufs, 0, sizeof(rsp_bufs));
            rsp_bufs[0].fd = bufs[0].fd;
            rsp_bufs[0].flags = DSPQUEUE_BUFFER_FLAG_DEREF;
            rsp_bufs[1].fd = bufs[1].fd;
            rsp_bufs[1].flags = DSPQUEUE_BUFFER_FLAG_DEREF;
            rsp_bufs[2].fd = bufs[2].fd;
            rsp_bufs[2].flags = DSPQUEUE_BUFFER_FLAG_DEREF
                              | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
                              | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
            dspqueue_write(queue, 0, 3, rsp_bufs,
                           sizeof(rsp), (const uint8_t *)&rsp,
                           DSPQUEUE_TIMEOUT_NONE);
        } else {
            FARF(ERROR, "Unknown op %u or bad n_bufs=%u", msg.op, n_bufs);
            struct matmul_rsp rsp = { msg.op, (uint32_t)AEE_EBADPARM };
            dspqueue_write(queue, 0, 0, NULL,
                           sizeof(rsp), (const uint8_t *)&rsp,
                           DSPQUEUE_TIMEOUT_NONE);
        }
    }
}


/* ========== dspqueue start/stop ========== */

AEEResult mnist_train_start(remote_handle64 handle, uint64 dsp_queue_id) {
    struct mnist_context *ctx = (struct mnist_context *)handle;
    if (!ctx) return AEE_EBADPARM;
    if (ctx->queue) return AEE_EITEMBUSY;
    ctx->process_time = 0;

    int err = dspqueue_import(dsp_queue_id, packet_callback, error_callback,
                              ctx, &ctx->queue);
    if (err) {
        FARF(ERROR, "dspqueue_import failed: 0x%08x", (unsigned)err);
        return err;
    }
    return AEE_SUCCESS;
}

AEEResult mnist_train_stop(remote_handle64 handle, uint64 *process_time) {
    struct mnist_context *ctx = (struct mnist_context *)handle;
    if (!ctx) return AEE_EBADPARM;
    if (!ctx->queue) return AEE_EBADSTATE;

    int err = dspqueue_close(ctx->queue);
    ctx->queue = NULL;
    if (err) return err;

    FARF(HIGH, "DSP process_time=%llu us", ctx->process_time);
    *process_time = ctx->process_time;
    return AEE_SUCCESS;
}


/* ========== Stub (not used by step 3) ========== */

AEEResult mnist_train_do_matmul(remote_handle64 handle,
                                 const uint8 *a_buf, int a_buf_len,
                                 const uint8 *b_buf, int b_buf_len,
                                 uint8 *c_buf, int c_buf_len,
                                 uint32 m, uint32 n, uint32 k, uint32 transpose,
                                 uint64 *process_time) {
    return AEE_EUNSUPPORTED;
}
