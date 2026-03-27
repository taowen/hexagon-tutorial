/*
 * ch09: dspqueue + VTCM-resident training skel
 *
 * Variant of ch08's skel_fused.c that mirrors all network buffers into VTCM.
 * The dspqueue callback thread keeps VTCM alive between messages, so weights
 * stay resident across batches. Only input is copied from DDR per batch;
 * weights are synced back to DDR on OP_SYNC.
 */

#include <stdlib.h>
#include <string.h>

#include "dsp/skel_common.h"
#include "dsp/hvx_ops.h"
#include "common/protocol.h"

#include "HAP_compute_res.h"

#define USE_VTCM_SCRATCH
#include "dsp/hvx_matmul_vtcm.h"

/* ---------- HVX copy: VTCM→DDR without L2 pollution ----------
 *
 * memcpy() uses scalar loads which pull VTCM addresses into L2 cache.
 * Subsequent HVX writes to VTCM bypass L2, creating stale entries.
 * HVX vector loads from VTCM go through the direct VTCM port (not L2),
 * so using HVX to copy avoids the pollution entirely.
 * See ch10 README for the full VTCM resource preemption analysis.
 */
static void hvx_copy(void *dst, const void *src, uint32_t bytes) {
    uint32_t n_vecs = bytes / 128;
    const HVX_Vector *s = (const HVX_Vector *)src;
    HVX_Vector *d = (HVX_Vector *)dst;
    for (uint32_t i = 0; i < n_vecs; i++) {
        d[i] = s[i];
    }
    /* Handle remainder (< 128 bytes) with scalar copy */
    uint32_t done = n_vecs * 128;
    if (done < bytes) {
        memcpy((uint8_t *)dst + done, (const uint8_t *)src + done, bytes - done);
    }
}

/* ---------- Buffer sizes (bytes) ---------- */
#define W1_BYTES   (NET_HIDDEN_DIM * NET_INPUT_DIM_PAD * 4)   /* 400KB */
#define B1_BYTES   (NET_HIDDEN_DIM * 4)                        /* 512B  */
#define W2_BYTES   (NET_OUTPUT_DIM_PAD * NET_HIDDEN_DIM * 4)   /* 16KB  */
#define B2_BYTES   (NET_OUTPUT_DIM_PAD * 4)                    /* 128B  */
#define DW1_BYTES  W1_BYTES
#define DW2_BYTES  W2_BYTES
#define MAX_BATCH  256

#define HIDDEN_BYTES(bs)  ((bs) * NET_HIDDEN_DIM * 4)
#define LOGITS_BYTES(bs)  ((bs) * NET_OUTPUT_DIM_PAD * 4)
#define INPUT_BYTES(bs)   ((bs) * NET_INPUT_DIM_PAD * 4)

#define SCRATCH_BYTES     (128 * 800 * 4)


/* ---------- DSP context ---------- */
struct mnist_context {
    dspqueue_t queue;
    uint64_t process_time, time_fwd_mm, time_fwd_other, time_bwd_mm, time_bwd_other, time_sgd;

    /* DDR pointers from OP_REGISTER_NET */
    float *net_bufs[NET_BUF_COUNT];
    int    net_fds[NET_BUF_COUNT];
    int    net_registered;

    /* VTCM state */
    unsigned int vtcm_res_id;
    void        *vtcm_base;
    uint32_t     vtcm_size;

    /* VTCM mirror buffers (bump-allocated from vtcm_base) */
    float *v_w1, *v_b1, *v_w2, *v_b2;        /* weights        */
    float *v_dw1, *v_dw2;                      /* gradients      */
    float *v_hidden, *v_logits;                /* fwd activations */
    float *v_dhidden, *v_dlogits;              /* bwd gradients  */
    float *v_hidden_pre, *v_probs;             /* intermediates  */
    float *v_input;                            /* per-batch input */
    float *v_scratch;                          /* matmul scratch  */
    int    vtcm_ready;
};


/* ---------- Bump allocator (128-byte aligned) ---------- */
static float *bump_alloc(uint8_t **bump, uint32_t bytes) {
    uintptr_t addr = (uintptr_t)*bump;
    addr = (addr + 127) & ~(uintptr_t)127;
    float *ptr = (float *)addr;
    *bump = (uint8_t *)(addr + bytes);
    return ptr;
}


/* ========== FastRPC lifecycle ========== */

AEEResult mnist_train_open(const char *uri, remote_handle64 *handle) {
    return mnist_train_open_impl(uri, handle, sizeof(struct mnist_context));
}

AEEResult mnist_train_close(remote_handle64 handle) {
    struct mnist_context *ctx = (struct mnist_context *)handle;
    if (!ctx) return AEE_EBADPARM;
    if (ctx->queue) return AEE_EITEMBUSY;

    if (ctx->vtcm_res_id) {
        HAP_compute_res_release(ctx->vtcm_res_id);
        ctx->vtcm_res_id = 0;
        FARF(HIGH, "VTCM released in close");
    }

    free(ctx);
    return AEE_SUCCESS;
}


/* ========== VTCM acquisition and buffer layout ========== */

static int acquire_vtcm_and_populate(struct mnist_context *ctx) {
#ifdef USE_HEAP_MIRROR
    /* Heap fallback for debugging */
    ctx->vtcm_base = memalign(4096, 4 * 1024 * 1024);
    if (!ctx->vtcm_base) {
        FARF(ERROR, "Heap alloc failed (4MB)");
        return AEE_ENOMEMORY;
    }
    ctx->vtcm_size = 4 * 1024 * 1024;
    ctx->vtcm_res_id = 0;
    FARF(HIGH, "HEAP mirror: %u KB at %p", ctx->vtcm_size / 1024, ctx->vtcm_base);
#else
    /* Request 4MB VTCM */
    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_vtcm_param(&attr, 4 * 1024 * 1024, 1);

    ctx->vtcm_res_id = HAP_compute_res_acquire(&attr, 100000);
    if (!ctx->vtcm_res_id) {
        FARF(ERROR, "VTCM acquire failed (4MB)");
        return AEE_ENOMEMORY;
    }
    ctx->vtcm_base = HAP_compute_res_attr_get_vtcm_ptr(&attr);
    ctx->vtcm_size = 4 * 1024 * 1024;
    FARF(HIGH, "VTCM acquired: %u KB at %p", ctx->vtcm_size / 1024, ctx->vtcm_base);
#endif

    /* Bump-allocate all buffers */
    uint8_t *bump = (uint8_t *)ctx->vtcm_base;
    ctx->v_w1         = bump_alloc(&bump, W1_BYTES);
    ctx->v_b1         = bump_alloc(&bump, B1_BYTES);
    ctx->v_w2         = bump_alloc(&bump, W2_BYTES);
    ctx->v_b2         = bump_alloc(&bump, B2_BYTES);
    ctx->v_dw1        = bump_alloc(&bump, DW1_BYTES);
    ctx->v_dw2        = bump_alloc(&bump, DW2_BYTES);
    ctx->v_hidden     = bump_alloc(&bump, HIDDEN_BYTES(MAX_BATCH));
    ctx->v_logits     = bump_alloc(&bump, LOGITS_BYTES(MAX_BATCH));
    ctx->v_dhidden    = bump_alloc(&bump, HIDDEN_BYTES(MAX_BATCH));
    ctx->v_dlogits    = bump_alloc(&bump, LOGITS_BYTES(MAX_BATCH));
    ctx->v_hidden_pre = bump_alloc(&bump, HIDDEN_BYTES(MAX_BATCH));
    ctx->v_probs      = bump_alloc(&bump, LOGITS_BYTES(MAX_BATCH));
    ctx->v_input      = bump_alloc(&bump, INPUT_BYTES(MAX_BATCH));
    ctx->v_scratch    = bump_alloc(&bump, SCRATCH_BYTES);

    /* Scratch is in VTCM too. The L2 coherency issue (scalar reads
     * seeing stale L2 entries after HVX writes) is solved by using
     * hvx_copy() in OP_SYNC instead of memcpy. */
    g_scratch = ctx->v_scratch;

    uint32_t used = (uint32_t)((uintptr_t)bump - (uintptr_t)ctx->vtcm_base);
    FARF(HIGH, "VTCM bump: %u KB used / %u KB allocated",
         used / 1024, ctx->vtcm_size / 1024);

    /* Copy weights from DDR to VTCM */
    memcpy(ctx->v_w1, ctx->net_bufs[NET_BUF_W1], W1_BYTES);
    memcpy(ctx->v_b1, ctx->net_bufs[NET_BUF_B1], B1_BYTES);
    memcpy(ctx->v_w2, ctx->net_bufs[NET_BUF_W2], W2_BYTES);
    memcpy(ctx->v_b2, ctx->net_bufs[NET_BUF_B2], B2_BYTES);
    FARF(HIGH, "Weights copied DDR -> VTCM (W1=%uKB B1=%uB W2=%uKB B2=%uB)",
         W1_BYTES / 1024, B1_BYTES, W2_BYTES / 1024, B2_BYTES);

    ctx->vtcm_ready = 1;
    return AEE_SUCCESS;
}


/* ========== dspqueue message callback ========== */

static void packet_callback(dspqueue_t queue, int error, void *context) {
    struct mnist_context *ctx = (struct mnist_context *)context;

    while (1) {
        union {
            struct register_net_req reg;
            struct train_batch_req  train;
            struct sync_req         sync;
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

        uint32_t op = msg.reg.op;

        if (op == OP_REGISTER_NET && n_bufs >= NET_BUF_COUNT) {
            /* Store DDR buffer pointers and fds */
            for (int i = 0; i < NET_BUF_COUNT; i++) {
                ctx->net_bufs[i] = (float *)bufs[i].ptr;
                ctx->net_fds[i] = bufs[i].fd;
            }
            ctx->net_registered = 1;

            /* Acquire VTCM and copy weights */
            int vtcm_err = acquire_vtcm_and_populate(ctx);
            if (vtcm_err != AEE_SUCCESS) {
                FARF(ERROR, "VTCM setup failed: 0x%08x", (unsigned)vtcm_err);
            }

            FARF(HIGH, "Network registered (%d bufs), VTCM %s",
                 NET_BUF_COUNT, ctx->vtcm_ready ? "ready" : "FAILED");

            /* Response: deref all buffers */
            struct dspqueue_buffer rsp_bufs[NET_BUF_COUNT];
            memset(rsp_bufs, 0, sizeof(rsp_bufs));
            for (int i = 0; i < NET_BUF_COUNT; i++) {
                rsp_bufs[i].fd = bufs[i].fd;
                rsp_bufs[i].flags = DSPQUEUE_BUFFER_FLAG_DEREF;
            }
            struct matmul_rsp rsp = { OP_REGISTER_NET, (uint32_t)vtcm_err };
            dspqueue_write(queue, 0, NET_BUF_COUNT, rsp_bufs,
                           sizeof(rsp), (const uint8_t *)&rsp,
                           DSPQUEUE_TIMEOUT_NONE);

        } else if (op == OP_TRAIN_BATCH && ctx->net_registered && ctx->vtcm_ready) {
            struct train_batch_req *treq = &msg.train;
            uint32_t bs = treq->batch_size;
            float lr = treq->learning_rate;

            /* Copy input from DDR to VTCM */
            memcpy(ctx->v_input, (float *)bufs[0].ptr, bs * NET_INPUT_DIM_PAD * sizeof(float));

            /* All pointers are VTCM */
            float *inp        = ctx->v_input;
            float *w1         = ctx->v_w1;
            float *b1         = ctx->v_b1;
            float *w2         = ctx->v_w2;
            float *b2         = ctx->v_b2;
            float *dw1        = ctx->v_dw1;
            float *dw2        = ctx->v_dw2;
            float *hidden     = ctx->v_hidden;
            float *logits     = ctx->v_logits;
            float *dhidden    = ctx->v_dhidden;
            float *dlogits    = ctx->v_dlogits;
            float *hidden_pre = ctx->v_hidden_pre;
            float *probs      = ctx->v_probs;

            uint64_t t1 = HAP_perf_get_time_us();
            uint64_t t_phase;

            /* === FORWARD === */
            /* Layer 1: hidden = input @ W1^T + b1 */
            t_phase = HAP_perf_get_time_us();
            matmul_nt(hidden, inp, w1, bs, NET_HIDDEN_DIM, NET_INPUT_DIM_PAD);
            ctx->time_fwd_mm += (HAP_perf_get_time_us() - t_phase);

            t_phase = HAP_perf_get_time_us();
            hvx_add_bias(hidden, b1, bs, NET_HIDDEN_DIM);
            memcpy(hidden_pre, hidden, bs * NET_HIDDEN_DIM * sizeof(float));
            hvx_relu_forward(hidden, bs * NET_HIDDEN_DIM);
            ctx->time_fwd_other += (HAP_perf_get_time_us() - t_phase);

            /* Layer 2: logits = hidden @ W2^T + b2 */
            t_phase = HAP_perf_get_time_us();
            matmul_nt(logits, hidden, w2, bs, NET_OUTPUT_DIM_PAD, NET_HIDDEN_DIM);
            ctx->time_fwd_mm += (HAP_perf_get_time_us() - t_phase);

            t_phase = HAP_perf_get_time_us();
            hvx_add_bias(logits, b2, bs, NET_OUTPUT_DIM_PAD);

            /* Softmax + cross-entropy loss */
            float loss = hvx_softmax_cross_entropy(logits, treq->labels, probs, bs);

            /* Training accuracy */
            uint32_t correct = 0;
            for (uint32_t b_idx = 0; b_idx < bs; b_idx++) {
                const float *row = logits + b_idx * NET_OUTPUT_DIM_PAD;
                float max_val = row[0];
                int max_j = 0;
                for (int j = 1; j < NET_OUTPUT_DIM; j++) {
                    if (row[j] > max_val) { max_val = row[j]; max_j = j; }
                }
                if (max_j == treq->labels[b_idx]) correct++;
            }
            ctx->time_fwd_other += (HAP_perf_get_time_us() - t_phase);

            /* === BACKWARD === */
            /* dlogits = (probs - one_hot) / batch */
            t_phase = HAP_perf_get_time_us();
            hvx_compute_dlogits(dlogits, probs, treq->labels, bs);
            ctx->time_bwd_other += (HAP_perf_get_time_us() - t_phase);

            /* dW2 = dlogits^T @ hidden */
            t_phase = HAP_perf_get_time_us();
            matmul_tn(dw2, dlogits, hidden, NET_OUTPUT_DIM_PAD, NET_HIDDEN_DIM, bs);
            ctx->time_bwd_mm += (HAP_perf_get_time_us() - t_phase);

            /* db2 = sum_rows(dlogits) */
            t_phase = HAP_perf_get_time_us();
            float db2_local[NET_OUTPUT_DIM_PAD] __attribute__((aligned(128)));
            hvx_bias_backward(db2_local, dlogits, bs, NET_OUTPUT_DIM_PAD);
            ctx->time_bwd_other += (HAP_perf_get_time_us() - t_phase);

            /* dhidden = dlogits @ W2 */
            t_phase = HAP_perf_get_time_us();
            matmul_nn(dhidden, dlogits, w2, bs, NET_HIDDEN_DIM, NET_OUTPUT_DIM_PAD);
            ctx->time_bwd_mm += (HAP_perf_get_time_us() - t_phase);

            /* ReLU backward */
            t_phase = HAP_perf_get_time_us();
            hvx_relu_backward(dhidden, hidden_pre, bs * NET_HIDDEN_DIM);
            ctx->time_bwd_other += (HAP_perf_get_time_us() - t_phase);

            /* dW1 = dhidden^T @ input */
            t_phase = HAP_perf_get_time_us();
            matmul_tn(dw1, dhidden, inp, NET_HIDDEN_DIM, NET_INPUT_DIM_PAD, bs);
            ctx->time_bwd_mm += (HAP_perf_get_time_us() - t_phase);

            /* db1 = sum_rows(dhidden) */
            t_phase = HAP_perf_get_time_us();
            float db1_local[NET_HIDDEN_DIM] __attribute__((aligned(128)));
            hvx_bias_backward(db1_local, dhidden, bs, NET_HIDDEN_DIM);
            ctx->time_bwd_other += (HAP_perf_get_time_us() - t_phase);

            /* === SGD UPDATE (in VTCM) === */
            uint64_t t_sgd = HAP_perf_get_time_us();
            hvx_sgd_update(w1, dw1, lr, NET_HIDDEN_DIM * NET_INPUT_DIM_PAD);
            for (int i = 0; i < NET_HIDDEN_DIM; i++)
                b1[i] -= lr * db1_local[i];
            hvx_sgd_update(w2, dw2, lr, NET_OUTPUT_DIM_PAD * NET_HIDDEN_DIM);
            for (int i = 0; i < NET_OUTPUT_DIM_PAD; i++)
                b2[i] -= lr * db2_local[i];
            ctx->time_sgd += (HAP_perf_get_time_us() - t_sgd);

            uint64_t t2 = HAP_perf_get_time_us();
            ctx->process_time += (t2 - t1);

            /* Response: deref input buffer, return loss + accuracy */
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

        } else if (op == 5 /* OP_EVAL */ && ctx->vtcm_ready) {
            struct train_batch_req *treq = &msg.train;
            uint32_t bs = treq->batch_size;

            /* Copy test input from DDR to VTCM */
            memcpy(ctx->v_input, (float *)bufs[0].ptr, bs * NET_INPUT_DIM_PAD * sizeof(float));

            /* Forward pass only (using VTCM weights) */
            float *inp    = ctx->v_input;
            float *hidden = ctx->v_hidden;
            float *logits = ctx->v_logits;

            matmul_nt(hidden, inp, ctx->v_w1, bs, NET_HIDDEN_DIM, NET_INPUT_DIM_PAD);
            hvx_add_bias(hidden, ctx->v_b1, bs, NET_HIDDEN_DIM);
            hvx_relu_forward(hidden, bs * NET_HIDDEN_DIM);
            matmul_nt(logits, hidden, ctx->v_w2, bs, NET_OUTPUT_DIM_PAD, NET_HIDDEN_DIM);
            hvx_add_bias(logits, ctx->v_b2, bs, NET_OUTPUT_DIM_PAD);

            /* Count correct predictions */
            uint32_t correct = 0;
            for (uint32_t b_idx = 0; b_idx < bs; b_idx++) {
                const float *row = logits + b_idx * NET_OUTPUT_DIM_PAD;
                float max_val = row[0];
                int max_j = 0;
                for (int j = 1; j < NET_OUTPUT_DIM; j++) {
                    if (row[j] > max_val) { max_val = row[j]; max_j = j; }
                }
                if (max_j == treq->labels[b_idx]) correct++;
            }

            /* Response */
            struct dspqueue_buffer rsp_bufs[1];
            memset(rsp_bufs, 0, sizeof(rsp_bufs));
            rsp_bufs[0].fd = bufs[0].fd;
            rsp_bufs[0].flags = DSPQUEUE_BUFFER_FLAG_DEREF;

            struct train_batch_rsp rsp;
            rsp.op = 5;
            rsp.status = 0;
            rsp.loss = 0.0f;
            rsp.correct = correct;

            dspqueue_write(queue, 0, 1, rsp_bufs,
                           sizeof(rsp), (const uint8_t *)&rsp,
                           DSPQUEUE_TIMEOUT_NONE);

        } else if (op == OP_SYNC && ctx->vtcm_ready) {
            /* Copy weights from VTCM to DDR (final export only, not used during training) */
            hvx_copy(ctx->net_bufs[NET_BUF_W1], ctx->v_w1, W1_BYTES);
            hvx_copy(ctx->net_bufs[NET_BUF_B1], ctx->v_b1, B1_BYTES);
            hvx_copy(ctx->net_bufs[NET_BUF_W2], ctx->v_w2, W2_BYTES);
            hvx_copy(ctx->net_bufs[NET_BUF_B2], ctx->v_b2, B2_BYTES);

            struct dspqueue_buffer rsp_bufs[4];
            memset(rsp_bufs, 0, sizeof(rsp_bufs));
            for (int i = 0; i < 4; i++) {
                rsp_bufs[i].fd = bufs[i].fd;
                rsp_bufs[i].flags = DSPQUEUE_BUFFER_FLAG_DEREF
                                  | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER;
            }
            struct matmul_rsp rsp = { OP_SYNC, 0 };
            dspqueue_write(queue, 0, 4, rsp_bufs,
                           sizeof(rsp), (const uint8_t *)&rsp,
                           DSPQUEUE_TIMEOUT_NONE);

        } else {
            FARF(ERROR, "Unknown op %u or bad state (n_bufs=%u, registered=%d, vtcm=%d)",
                 op, n_bufs, ctx->net_registered, ctx->vtcm_ready);
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

    int err = dspqueue_close(ctx->queue);
    ctx->queue = NULL;
    if (err) return err;

    /* Release VTCM */
    if (ctx->vtcm_res_id) {
        HAP_compute_res_release(ctx->vtcm_res_id);
        ctx->vtcm_res_id = 0;
        ctx->vtcm_ready = 0;
        FARF(HIGH, "VTCM released in stop");
    }

    FARF(HIGH, "DSP timing: fwd_mm=%llu bwd_mm=%llu fwd_other=%llu bwd_other=%llu sgd=%llu total=%llu us",
         ctx->time_fwd_mm, ctx->time_bwd_mm, ctx->time_fwd_other, ctx->time_bwd_other,
         ctx->time_sgd, ctx->process_time);
    *process_time = ctx->process_time;
    return AEE_SUCCESS;
}


/* ========== Stub (not used by VTCM path) ========== */

AEEResult mnist_train_do_matmul(remote_handle64 handle,
                                 const uint8 *a_buf, int a_buf_len,
                                 const uint8 *b_buf, int b_buf_len,
                                 uint8 *c_buf, int c_buf_len,
                                 uint32 m, uint32 n, uint32 k, uint32 transpose,
                                 uint64 *process_time) {
    return AEE_EUNSUPPORTED;
}
