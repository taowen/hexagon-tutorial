/*
 * f16-native fused training skel.
 *
 * All weights, activations, and gradients stay in _Float16.
 * HMX matmuls operate on f16 directly -- no f32 conversion anywhere.
 * Only supports: OP_REGISTER_NET, OP_TRAIN_BATCH, OP_SYNC.
 */

#include "dsp/skel_common.h"
#include "dsp/hvx_ops_f16.h"
#include "dsp/hmx_matmul_f16.h"
#include "common/protocol.h"
#include <math.h>
#include <stdlib.h>  /* for memalign, free */

/* ---------- Static scratch buffers for training ---------- */
static _Float16 s_pre_relu[256 * 128] __attribute__((aligned(128)));
static _Float16 s_dhidden[256 * 128] __attribute__((aligned(128)));

/* ---------- DSP context ---------- */
struct mnist_context {
    dspqueue_t queue;
    uint64_t   process_time;   /* cumulative DSP processing time (us) */
    uint64_t time_fwd_mm;     /* forward matmuls */
    uint64_t time_fwd_other;  /* bias + relu + softmax */
    uint64_t time_bwd_mm;     /* backward matmuls */
    uint64_t time_bwd_other;  /* dlogits prep + relu_backward + db sums */
    uint64_t time_sgd;        /* SGD update */
    /* Registered network buffers (all _Float16) */
    _Float16 *net_bufs[NET_BUF_COUNT];
    int       net_fds[NET_BUF_COUNT];
    int       net_registered;
    int       hmx_initialized;
    /* Pre-transposed weight copies (DSP-local, not shared memory) */
    _Float16 *w1_t;       /* [INPUT_DIM_PAD x HIDDEN_DIM] = W1^T */
    _Float16 *w2_t;       /* [HIDDEN_DIM x OUTPUT_DIM_PAD] = W2^T */
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
        /* Read with largest possible message buffer */
        union {
            struct register_net_req reg;
            struct train_batch_req train;
            struct sync_req sync;
        } msg;
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

        uint32_t op = msg.reg.op;  /* op is always first field */

        /* Lazy HMX init in callback thread (HMX lock is thread-local) */
        if (!ctx->hmx_initialized) {
            int hmx_err = hexkl_micro_hw_init(&g_vtcm_base_f16, &g_vtcm_size_f16);
            if (hmx_err == AEE_SUCCESS) {
                hmx_err = hexkl_micro_hmx_lock();
            }
            if (hmx_err == AEE_SUCCESS) {
                ctx->hmx_initialized = 1;
                FARF(HIGH, "HMX f16 initialized: VTCM %u KB", g_vtcm_size_f16 / 1024);
            } else {
                FARF(ERROR, "HMX f16 init failed: 0x%08x", (unsigned)hmx_err);
            }
        }

        if (op == OP_REGISTER_NET && n_bufs >= NET_BUF_COUNT) {
            /* Store buffer pointers as _Float16* */
            for (int i = 0; i < NET_BUF_COUNT; i++) {
                ctx->net_bufs[i] = (_Float16 *)bufs[i].ptr;
                ctx->net_fds[i] = bufs[i].fd;
            }
            ctx->net_registered = 1;
            FARF(HIGH, "f16: Network buffers registered (%d bufs)", NET_BUF_COUNT);

            /* Allocate and compute transposed weight copies */
            if (!ctx->w1_t) {
                ctx->w1_t = (_Float16 *)memalign(128,
                    NET_INPUT_DIM_PAD * NET_HIDDEN_DIM * sizeof(_Float16));
                ctx->w2_t = (_Float16 *)memalign(128,
                    NET_HIDDEN_DIM * NET_OUTPUT_DIM_PAD * sizeof(_Float16));
            }
            if (ctx->w1_t && ctx->w2_t) {
                blocked_transpose_f16(ctx->w1_t, ctx->net_bufs[NET_BUF_W1],
                    NET_HIDDEN_DIM, NET_INPUT_DIM_PAD);
                blocked_transpose_f16(ctx->w2_t, ctx->net_bufs[NET_BUF_W2],
                    NET_OUTPUT_DIM_PAD, NET_HIDDEN_DIM);
                FARF(HIGH, "f16: Pre-transposed weights allocated and initialized");
            }

            /* Response: deref all buffers */
            struct dspqueue_buffer rsp_bufs[NET_BUF_COUNT];
            memset(rsp_bufs, 0, sizeof(rsp_bufs));
            for (int i = 0; i < NET_BUF_COUNT; i++) {
                rsp_bufs[i].fd = bufs[i].fd;
                rsp_bufs[i].flags = DSPQUEUE_BUFFER_FLAG_DEREF;
            }
            struct matmul_rsp rsp = { OP_REGISTER_NET, 0 };
            dspqueue_write(queue, 0, NET_BUF_COUNT, rsp_bufs,
                           sizeof(rsp), (const uint8_t *)&rsp,
                           DSPQUEUE_TIMEOUT_NONE);

        } else if (op == OP_TRAIN_BATCH && ctx->net_registered) {
            /* Fused forward + backward + SGD -- all in f16 */
            struct train_batch_req *treq = &msg.train;
            uint32_t bs = treq->batch_size;
            float lr = treq->learning_rate;

            _Float16 *input   = (_Float16 *)bufs[0].ptr;
            _Float16 *w1      = ctx->net_bufs[NET_BUF_W1];
            _Float16 *b1      = ctx->net_bufs[NET_BUF_B1];
            _Float16 *w2      = ctx->net_bufs[NET_BUF_W2];
            _Float16 *b2      = ctx->net_bufs[NET_BUF_B2];
            _Float16 *dw1     = ctx->net_bufs[NET_BUF_DW1];
            _Float16 *dw2     = ctx->net_bufs[NET_BUF_DW2];
            _Float16 *hidden  = ctx->net_bufs[NET_BUF_HIDDEN];
            _Float16 *logits  = ctx->net_bufs[NET_BUF_LOGITS];
            _Float16 *dlogits = ctx->net_bufs[NET_BUF_DLOGITS];
            _Float16 *probs   = ctx->net_bufs[NET_BUF_PROBS];

            uint64_t t1 = HAP_perf_get_time_us();
            uint64_t t_phase;

            /* === FORWARD === */
            /* Layer 1: hidden = input @ W1_t + b1, then ReLU (using pre-transposed W1) */
            t_phase = HAP_perf_get_time_us();
            hmx_matmul_f16_dispatch(g_vtcm_base_f16, g_vtcm_size_f16,
                hidden, input, ctx->w1_t, bs, NET_HIDDEN_DIM, NET_INPUT_DIM_PAD, 0, 0);
            ctx->time_fwd_mm += (HAP_perf_get_time_us() - t_phase);

            t_phase = HAP_perf_get_time_us();
            hvx_add_bias_f16(hidden, b1, bs, NET_HIDDEN_DIM);
            memcpy(s_pre_relu, hidden, bs * NET_HIDDEN_DIM * sizeof(_Float16));
            hvx_relu_forward_f16(hidden, hidden, bs * NET_HIDDEN_DIM);
            ctx->time_fwd_other += (HAP_perf_get_time_us() - t_phase);

            /* Layer 2: logits = hidden @ W2_t + b2 (using pre-transposed W2) */
            t_phase = HAP_perf_get_time_us();
            hmx_matmul_f16_dispatch(g_vtcm_base_f16, g_vtcm_size_f16,
                logits, hidden, ctx->w2_t, bs, NET_OUTPUT_DIM_PAD, NET_HIDDEN_DIM, 0, 0);
            ctx->time_fwd_mm += (HAP_perf_get_time_us() - t_phase);

            t_phase = HAP_perf_get_time_us();
            hvx_add_bias_f16(logits, b2, bs, NET_OUTPUT_DIM_PAD);

            /* Softmax + cross-entropy loss */
            float loss = hvx_softmax_cross_entropy_f16(probs, logits,
                treq->labels, bs, NET_OUTPUT_DIM_PAD);

            /* Training accuracy */
            uint32_t correct = 0;
            for (uint32_t i = 0; i < bs; i++) {
                int pred = 0;
                _Float16 max_p = probs[i * NET_OUTPUT_DIM_PAD];
                for (uint32_t j = 1; j < 10; j++) {  /* only 10 real classes */
                    if (probs[i * NET_OUTPUT_DIM_PAD + j] > max_p) {
                        max_p = probs[i * NET_OUTPUT_DIM_PAD + j];
                        pred = j;
                    }
                }
                if (pred == treq->labels[i]) correct++;
            }
            ctx->time_fwd_other += (HAP_perf_get_time_us() - t_phase);

            /* === BACKWARD === */
            /* dlogits = (probs - one_hot) / batch */
            t_phase = HAP_perf_get_time_us();
            hvx_compute_dlogits_f16(dlogits, probs, treq->labels,
                bs, NET_OUTPUT_DIM_PAD);
            ctx->time_bwd_other += (HAP_perf_get_time_us() - t_phase);

            /* dW2 = dlogits^T @ hidden */
            t_phase = HAP_perf_get_time_us();
            memset(dw2, 0, NET_OUTPUT_DIM_PAD * NET_HIDDEN_DIM * sizeof(_Float16));
            hmx_matmul_f16_dispatch(g_vtcm_base_f16, g_vtcm_size_f16,
                dw2, dlogits, hidden, NET_OUTPUT_DIM_PAD, NET_HIDDEN_DIM, bs, 2, 0);
            ctx->time_bwd_mm += (HAP_perf_get_time_us() - t_phase);

            /* db2 = sum_rows(dlogits) */
            t_phase = HAP_perf_get_time_us();
            _Float16 db2_local[NET_OUTPUT_DIM_PAD] __attribute__((aligned(128)));
            hvx_bias_backward_f16(db2_local, dlogits, bs, NET_OUTPUT_DIM_PAD);
            ctx->time_bwd_other += (HAP_perf_get_time_us() - t_phase);

            /* dhidden = dlogits @ W2 */
            t_phase = HAP_perf_get_time_us();
            hmx_matmul_f16_dispatch(g_vtcm_base_f16, g_vtcm_size_f16,
                s_dhidden, dlogits, w2, bs, NET_HIDDEN_DIM, NET_OUTPUT_DIM_PAD, 0, 0);
            ctx->time_bwd_mm += (HAP_perf_get_time_us() - t_phase);

            /* ReLU backward */
            t_phase = HAP_perf_get_time_us();
            hvx_relu_backward_f16(s_dhidden, s_dhidden, s_pre_relu,
                bs * NET_HIDDEN_DIM);
            ctx->time_bwd_other += (HAP_perf_get_time_us() - t_phase);

            /* dW1 = dhidden^T @ input */
            t_phase = HAP_perf_get_time_us();
            memset(dw1, 0, NET_HIDDEN_DIM * NET_INPUT_DIM_PAD * sizeof(_Float16));
            hmx_matmul_f16_dispatch(g_vtcm_base_f16, g_vtcm_size_f16,
                dw1, s_dhidden, input, NET_HIDDEN_DIM, NET_INPUT_DIM_PAD, bs, 2, 0);
            ctx->time_bwd_mm += (HAP_perf_get_time_us() - t_phase);

            /* db1 = sum_rows(dhidden) */
            t_phase = HAP_perf_get_time_us();
            _Float16 db1_local[NET_HIDDEN_DIM] __attribute__((aligned(128)));
            hvx_bias_backward_f16(db1_local, s_dhidden, bs, NET_HIDDEN_DIM);
            ctx->time_bwd_other += (HAP_perf_get_time_us() - t_phase);

            /* === SGD UPDATE === */
            uint64_t t_sgd = HAP_perf_get_time_us();
            hvx_sgd_update_f16(w1, dw1, lr, NET_HIDDEN_DIM * NET_INPUT_DIM_PAD);
            for (int i = 0; i < NET_HIDDEN_DIM; i++)
                b1[i] = (_Float16)((float)b1[i] - lr * (float)db1_local[i]);
            hvx_sgd_update_f16(w2, dw2, lr, NET_OUTPUT_DIM_PAD * NET_HIDDEN_DIM);
            for (int i = 0; i < NET_OUTPUT_DIM_PAD; i++)
                b2[i] = (_Float16)((float)b2[i] - lr * (float)db2_local[i]);

            /* Re-transpose weights for next batch's forward pass */
            if (ctx->w1_t)
                blocked_transpose_f16(ctx->w1_t, w1,
                    NET_HIDDEN_DIM, NET_INPUT_DIM_PAD);
            if (ctx->w2_t)
                blocked_transpose_f16(ctx->w2_t, w2,
                    NET_OUTPUT_DIM_PAD, NET_HIDDEN_DIM);
            ctx->time_sgd += (HAP_perf_get_time_us() - t_sgd);

            uint64_t t2 = HAP_perf_get_time_us();
            ctx->process_time += (t2 - t1);

            /* Response: deref input, return loss+accuracy */
            struct train_batch_rsp rsp;
            rsp.op = OP_TRAIN_BATCH;
            rsp.status = 0;
            rsp.loss = loss;
            rsp.correct = correct;

            struct dspqueue_buffer rsp_bufs[1];
            memset(rsp_bufs, 0, sizeof(rsp_bufs));
            rsp_bufs[0].fd = bufs[0].fd;
            rsp_bufs[0].flags = DSPQUEUE_BUFFER_FLAG_DEREF;

            dspqueue_write(queue, 0, 1, rsp_bufs,
                           sizeof(rsp), (const uint8_t *)&rsp,
                           DSPQUEUE_TIMEOUT_NONE);

        } else if (op == OP_SYNC && n_bufs >= 4) {
            /* Flush weight buffers from DSP cache back to DDR */
            struct dspqueue_buffer rsp_bufs[4];
            memset(rsp_bufs, 0, sizeof(rsp_bufs));
            for (int i = 0; i < 4; i++) {
                rsp_bufs[i].fd = bufs[i].fd;
                rsp_bufs[i].flags = DSPQUEUE_BUFFER_FLAG_DEREF
                                  | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
                                  | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
            }
            struct matmul_rsp rsp = { OP_SYNC, 0 };
            dspqueue_write(queue, 0, 4, rsp_bufs,
                           sizeof(rsp), (const uint8_t *)&rsp,
                           DSPQUEUE_TIMEOUT_NONE);

        } else {
            FARF(ERROR, "f16: Unknown op %u or bad state (n_bufs=%u, registered=%d)",
                 op, n_bufs, ctx->net_registered);
            /* Still need to respond to avoid deadlock */
            struct matmul_rsp rsp = { op, (uint32_t)AEE_EBADPARM };
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
    ctx->time_fwd_mm = 0;
    ctx->time_fwd_other = 0;
    ctx->time_bwd_mm = 0;
    ctx->time_bwd_other = 0;
    ctx->time_sgd = 0;

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

    /* Free pre-transposed weight buffers */
    if (ctx->w1_t) { free(ctx->w1_t); ctx->w1_t = NULL; }
    if (ctx->w2_t) { free(ctx->w2_t); ctx->w2_t = NULL; }

    /* Release HMX before closing queue */
    if (ctx->hmx_initialized) {
        hexkl_micro_hmx_unlock();
        ctx->hmx_initialized = 0;
    }

    int err = dspqueue_close(ctx->queue);
    ctx->queue = NULL;
    if (err) return err;

    FARF(HIGH, "f16 DSP timing: fwd_mm=%llu bwd_mm=%llu fwd_other=%llu bwd_other=%llu sgd=%llu total=%llu us",
         ctx->time_fwd_mm, ctx->time_bwd_mm, ctx->time_fwd_other, ctx->time_bwd_other, ctx->time_sgd, ctx->process_time);
    *process_time = ctx->process_time;
    return AEE_SUCCESS;
}


/* ========== Stub (not used by f16 fused path) ========== */

AEEResult mnist_train_do_matmul(remote_handle64 handle,
                                 const uint8 *a_buf, int a_buf_len,
                                 const uint8 *b_buf, int b_buf_len,
                                 uint8 *c_buf, int c_buf_len,
                                 uint32 m, uint32 n, uint32 k, uint32 transpose,
                                 uint64 *process_time) {
    return AEE_EUNSUPPORTED;
}
