/*
 * ch09: dspqueue + VTCM-resident f16 training skel (HMX matrix accelerator)
 *
 * All network data (weights, activations, gradients) stored as f16 in VTCM.
 * Matrix multiplications use HMX (Hexagon Matrix eXtension) via direct ASM
 * for compute and hexkl_micro for tile format conversions.
 *
 * All non-matmul ops use HVX vectorized functions directly on VTCM data
 * (128B-aligned rows, NET_OUTPUT_DIM_PAD=64 = 1 HVX vector per row).
 *
 * ARM sends f16 input directly (no f32->f16 conversion on DSP).
 *
 * VTCM layout:
 *   [f16 data buffers (~800KB)] [HMX workspace: AH/WH/out tiles, scales, staging]
 */

#include <stdlib.h>
#include <string.h>

#include "dsp/skel_common.h"
#include "dsp/hvx_ops.h"
#include "dsp/hvx_ops_f16.h"
#include "dsp/hvx_matmul_f16_vtcm.h"
#include "dsp/hmx_matmul_f16_vtcm.h"
#include "hexkl_micro.h"
#include "common/protocol.h"


/* ---------- Buffer sizes (f16 bytes) ---------- */
#define W1_F16   (NET_HIDDEN_DIM * NET_INPUT_DIM_PAD * 2)     /* 200KB */
#define B1_F16   (NET_HIDDEN_DIM * 2)                          /* 256B  */
#define W2_F16   (NET_OUTPUT_DIM_PAD * NET_HIDDEN_DIM * 2)     /* 16KB  */
#define B2_F16   (NET_OUTPUT_DIM_PAD * 2)                      /* 128B  */
#define DW1_F16  W1_F16
#define DW2_F16  W2_F16
#define MAX_BATCH 256

#define HIDDEN_F16(bs)  ((bs) * NET_HIDDEN_DIM * 2)
#define LOGITS_F16(bs)  ((bs) * NET_OUTPUT_DIM_PAD * 2)
#define INPUT_F16(bs)   ((bs) * NET_INPUT_DIM_PAD * 2)


/* ---------- HVX helpers for VTCM access ---------- */

/* HVX vector copy: dst and src both in VTCM (or any HVX-accessible memory) */
static void hvx_memcpy(void *dst, const void *src, uint32_t bytes) {
    uint32_t vecs = bytes / 128;
    const HVX_Vector *s = (const HVX_Vector *)src;
    HVX_Vector *d = (HVX_Vector *)dst;
    for (uint32_t i = 0; i < vecs; i++)
        d[i] = s[i];
    /* Tail bytes (should not happen with 128-byte aligned VTCM allocations) */
    uint32_t rem = bytes & 127;
    if (rem) {
        uint8_t *db = (uint8_t *)dst + vecs * 128;
        const uint8_t *sb = (const uint8_t *)src + vecs * 128;
        for (uint32_t i = 0; i < rem; i++)
            db[i] = sb[i];
    }
}

/* HVX zero-fill: VTCM-safe */
static void hvx_memzero(void *dst, uint32_t bytes) {
    uint32_t vecs = bytes / 128;
    HVX_Vector *d = (HVX_Vector *)dst;
    HVX_Vector zero = Q6_V_vzero();
    for (uint32_t i = 0; i < vecs; i++)
        d[i] = zero;
    uint32_t rem = bytes & 127;
    if (rem) {
        uint8_t *db = (uint8_t *)dst + vecs * 128;
        for (uint32_t i = 0; i < rem; i++)
            db[i] = 0;
    }
}



/* ---------- DSP context ---------- */
struct mnist_context {
    dspqueue_t queue;
    uint64_t process_time, time_fwd_mm, time_fwd_other, time_bwd_mm, time_bwd_other, time_sgd;

    /* DDR pointers from OP_REGISTER_NET (f32) */
    float *net_bufs[NET_BUF_COUNT];
    int    net_fds[NET_BUF_COUNT];
    int    net_registered;

    /* VTCM state */
    uint8_t *vtcm_base;
    uint32_t vtcm_size;

    /* VTCM f16 data buffers (bump-allocated from vtcm_base) */
    _Float16 *v_b1, *v_b2;                      /* biases           */
    _Float16 *v_w1_t, *v_w2_t;                  /* transposed weights (primary) */
    _Float16 *v_dw1_t, *v_dw2_t;                /* gradients (transposed form) */
    _Float16 *v_hidden, *v_logits;               /* fwd activations  */
    _Float16 *v_dlogits, *v_probs;               /* fwd intermediates */
    _Float16 *v_hidden_pre, *v_dhidden;          /* bwd intermediates */
    _Float16 *v_input;                           /* per-batch input   */
    _Float16 *v_scratch;                         /* transpose scratch buffer */

    /* HMX state */
    struct hmx_workspace hmx_ws;
    uint32_t wh_w1t_off;    /* cached WH tiles for W1^T */
    uint32_t wh_w2t_off;    /* cached WH tiles for W2^T */
    int hmx_locked;

    int vtcm_ready;
};


/* ---------- Bump allocator (128-byte aligned) ---------- */
static _Float16 *bump_alloc_f16(uint8_t **bump, uint32_t bytes) {
    uintptr_t addr = (uintptr_t)*bump;
    addr = (addr + 127) & ~(uintptr_t)127;
    _Float16 *ptr = (_Float16 *)addr;
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
    if (ctx->hmx_locked) {
        hexkl_micro_hmx_unlock();
        ctx->hmx_locked = 0;
    }
    free(ctx);
    return AEE_SUCCESS;
}


/* ========== VTCM setup: allocate f16 data buffers + HMX workspace ========== */

/* Largest scratch needed: X^T at bs=256 -> 832*256*2 = 416KB */
#define SCRATCH_F16  (NET_INPUT_DIM_PAD * MAX_BATCH * 2)

static int setup_vtcm(struct mnist_context *ctx) {
    /* hexkl_micro_hw_init acquires VTCM */
    int err = hexkl_micro_hw_init(&ctx->vtcm_base, &ctx->vtcm_size);
    if (err != AEE_SUCCESS) {
        FARF(ERROR, "hexkl_micro_hw_init failed: 0x%08x", (unsigned)err);
        return err;
    }
    FARF(HIGH, "VTCM acquired: %u KB at %p", ctx->vtcm_size / 1024, ctx->vtcm_base);

    /* Lock HMX */
    err = hexkl_micro_hmx_lock();
    if (err != AEE_SUCCESS) {
        FARF(ERROR, "hexkl_micro_hmx_lock failed: 0x%08x", (unsigned)err);
        return err;
    }
    ctx->hmx_locked = 1;

    /* Bump-allocate f16 data buffers from front of VTCM.
     * Only transposed weights are stored (no v_w1/v_w2). */
    uint8_t *bump = ctx->vtcm_base;
    ctx->v_b1         = bump_alloc_f16(&bump, B1_F16);
    ctx->v_b2         = bump_alloc_f16(&bump, B2_F16);
    ctx->v_w1_t       = bump_alloc_f16(&bump, W1_F16);    /* W1^T[832,128] */
    ctx->v_w2_t       = bump_alloc_f16(&bump, W2_F16);    /* W2^T[128,64]  */
    ctx->v_dw1_t      = bump_alloc_f16(&bump, W1_F16);    /* dW1^T[832,128] */
    ctx->v_dw2_t      = bump_alloc_f16(&bump, DW2_F16);   /* dW2^T[128,64]  */
    ctx->v_hidden     = bump_alloc_f16(&bump, HIDDEN_F16(MAX_BATCH));
    ctx->v_logits     = bump_alloc_f16(&bump, LOGITS_F16(MAX_BATCH));
    ctx->v_dlogits    = bump_alloc_f16(&bump, LOGITS_F16(MAX_BATCH));
    ctx->v_probs      = bump_alloc_f16(&bump, LOGITS_F16(MAX_BATCH));
    ctx->v_hidden_pre = bump_alloc_f16(&bump, HIDDEN_F16(MAX_BATCH));
    ctx->v_dhidden    = bump_alloc_f16(&bump, HIDDEN_F16(MAX_BATCH));
    ctx->v_input      = bump_alloc_f16(&bump, INPUT_F16(MAX_BATCH));
    ctx->v_scratch    = bump_alloc_f16(&bump, SCRATCH_F16);

    uint32_t data_used = (uint32_t)((uintptr_t)bump - (uintptr_t)ctx->vtcm_base);
    FARF(HIGH, "VTCM layout: data=%uKB (incl scratch), total=%uKB",
         data_used / 1024, ctx->vtcm_size / 1024);

    /* Setup HMX workspace after data buffers */
    err = setup_hmx_workspace(&ctx->hmx_ws, ctx->vtcm_base, ctx->vtcm_size, data_used);
    if (err != AEE_SUCCESS) {
        FARF(ERROR, "setup_hmx_workspace failed: 0x%08x", (unsigned)err);
        return err;
    }

    /* Convert f32 weights from DDR to f16, then transpose to VTCM.
     * Use v_dw1_t as temp buffer for the non-transposed f16 weights. */
    hvx_f32_to_f16(ctx->v_dw1_t, ctx->net_bufs[NET_BUF_W1],
                    NET_HIDDEN_DIM * NET_INPUT_DIM_PAD);
    blocked_transpose_f16_vtcm(ctx->v_w1_t, ctx->v_dw1_t,
                           NET_HIDDEN_DIM, NET_INPUT_DIM_PAD);

    hvx_f32_to_f16(ctx->v_dw2_t, ctx->net_bufs[NET_BUF_W2],
                    NET_OUTPUT_DIM_PAD * NET_HIDDEN_DIM);
    blocked_transpose_f16_vtcm(ctx->v_w2_t, ctx->v_dw2_t,
                           NET_OUTPUT_DIM_PAD, NET_HIDDEN_DIM);

    hvx_f32_to_f16(ctx->v_b1, ctx->net_bufs[NET_BUF_B1], NET_HIDDEN_DIM);
    hvx_f32_to_f16(ctx->v_b2, ctx->net_bufs[NET_BUF_B2], NET_OUTPUT_DIM_PAD);

    FARF(HIGH, "Weights: f32 DDR -> f16 transposed VTCM (%uKB)",
         (W1_F16 + W2_F16) / 1024);

    ctx->vtcm_ready = 1;
    return AEE_SUCCESS;
}


/* ========== Training/eval helper functions ========== */

/*
 * do_train_batch -- Execute one forward+backward+SGD step on a batch
 *                   already loaded into ctx->v_input.
 *
 * Returns batch loss in *out_loss, batch correct count in *out_correct.
 * Labels are read from the labels array (batch_size elements).
 */
static void do_train_batch(struct mnist_context *ctx,
                           const uint8_t *labels, uint32_t bs, float lr,
                           float *out_loss, uint32_t *out_correct)
{
    _Float16 *inp     = ctx->v_input;
    _Float16 *b1      = ctx->v_b1;
    _Float16 *b2      = ctx->v_b2;
    _Float16 *w1_t    = ctx->v_w1_t;
    _Float16 *w2_t    = ctx->v_w2_t;
    _Float16 *dw1_t   = ctx->v_dw1_t;
    _Float16 *dw2_t   = ctx->v_dw2_t;
    _Float16 *hidden  = ctx->v_hidden;
    _Float16 *logits  = ctx->v_logits;
    _Float16 *dlogits = ctx->v_dlogits;
    _Float16 *probs   = ctx->v_probs;
    _Float16 *hidden_pre = ctx->v_hidden_pre;
    _Float16 *dhidden = ctx->v_dhidden;

    struct hmx_workspace *ws = &ctx->hmx_ws;
    _Float16 *scratch = ctx->v_scratch;

    uint64_t t1 = HAP_perf_get_time_us();
    uint64_t t_phase;

    /* === FORWARD === */

    /* Cache W1^T and W2^T as WH tiles (weights don't change during forward) */
    t_phase = HAP_perf_get_time_us();
    convert_rm_to_wh(ws, ws->wh_base, w1_t, NET_INPUT_DIM_PAD, NET_HIDDEN_DIM);
    ctx->wh_w1t_off = ws->wh_base;
    /* W2^T WH tiles placed after W1^T tiles */
    {
        uint32_t w1t_K  = NET_INPUT_DIM_PAD / HMX_TILE;  /* 26 */
        uint32_t w1t_Nc = NET_HIDDEN_DIM / HMX_TILE;     /* 4  */
        uint32_t w1t_wh_count = w1t_Nc * w1t_K;          /* 104 */
        ctx->wh_w2t_off = ws->wh_base + w1t_wh_count * HMX_ALIGN;
    }
    convert_rm_to_wh(ws, ctx->wh_w2t_off, w2_t, NET_HIDDEN_DIM, NET_OUTPUT_DIM_PAD);

    /* Fwd L1: hidden[bs,128] = inp[bs,832] @ w1_t[832,128] */
    hmx_matmul_nn_f16_cached_wh(ws, hidden, inp, ctx->wh_w1t_off,
                                 bs, NET_HIDDEN_DIM, NET_INPUT_DIM_PAD);
    ctx->time_fwd_mm += (HAP_perf_get_time_us() - t_phase);

    t_phase = HAP_perf_get_time_us();
    hvx_add_bias_f16(hidden, b1, bs, NET_HIDDEN_DIM);
    hvx_memcpy(hidden_pre, hidden, bs * NET_HIDDEN_DIM * sizeof(_Float16));
    hvx_relu_forward_f16(hidden, hidden, bs * NET_HIDDEN_DIM);
    ctx->time_fwd_other += (HAP_perf_get_time_us() - t_phase);

    /* Fwd L2: logits[bs,64] = hidden[bs,128] @ w2_t[128,64] */
    t_phase = HAP_perf_get_time_us();
    hmx_matmul_nn_f16_cached_wh(ws, logits, hidden, ctx->wh_w2t_off,
                                 bs, NET_OUTPUT_DIM_PAD, NET_HIDDEN_DIM);
    ctx->time_fwd_mm += (HAP_perf_get_time_us() - t_phase);

    t_phase = HAP_perf_get_time_us();
    hvx_add_bias_f16(logits, b2, bs, NET_OUTPUT_DIM_PAD);

    float loss = 0.0f;
    int correct_i = 0;
    hvx_softmax_cross_entropy_f16_vec((const uint16_t *)logits,
        (uint16_t *)probs, labels, bs, &loss, &correct_i);
    ctx->time_fwd_other += (HAP_perf_get_time_us() - t_phase);

    /* === BACKWARD === */
    t_phase = HAP_perf_get_time_us();
    hvx_memcpy(dlogits, probs, bs * NET_OUTPUT_DIM_PAD * sizeof(_Float16));
    {
        _Float16 inv_batch_h = (_Float16)(1.0f / (float)bs);
        uint16_t inv_bits;
        memcpy(&inv_bits, &inv_batch_h, sizeof(inv_bits));
        HVX_Vector inv_vec = Q6_Vh_vsplat_R((int32_t)inv_bits);
        for (uint32_t i = 0; i < bs; i++) {
            _Float16 *row = dlogits + i * NET_OUTPUT_DIM_PAD;
            row[labels[i]] = (_Float16)((float)row[labels[i]] - 1.0f);
            HVX_Vector v = *(HVX_Vector *)row;
            HVX_Vector vq = Q6_Vqf16_vmpy_VhfVhf(v, inv_vec);
            *(HVX_Vector *)row = Q6_Vhf_equals_Vqf16(vq);
        }
    }
    ctx->time_bwd_other += (HAP_perf_get_time_us() - t_phase);

    /* Bwd dW2: dw2_t[128,64] = hidden^T[128,bs] @ dlogits[bs,64]
     * Transpose hidden[bs,128] -> scratch[128,bs], then NN matmul */
    t_phase = HAP_perf_get_time_us();
    blocked_transpose_f16_vtcm(scratch, hidden, bs, NET_HIDDEN_DIM);
    hmx_matmul_nn_f16(ws, dw2_t, scratch, dlogits,
                       NET_HIDDEN_DIM, NET_OUTPUT_DIM_PAD, bs);
    ctx->time_bwd_mm += (HAP_perf_get_time_us() - t_phase);

    t_phase = HAP_perf_get_time_us();
    _Float16 db2_local[NET_OUTPUT_DIM_PAD] __attribute__((aligned(128)));
    hvx_bias_backward_f16(db2_local, dlogits, bs, NET_OUTPUT_DIM_PAD);
    ctx->time_bwd_other += (HAP_perf_get_time_us() - t_phase);

    /* Bwd dH: dhidden[bs,128] = dlogits[bs,64] @ w2_t^T[64,128]
     * w2_t is [128,64], transpose to scratch[64,128], then NN matmul */
    t_phase = HAP_perf_get_time_us();
    blocked_transpose_f16_vtcm(scratch, w2_t, NET_HIDDEN_DIM, NET_OUTPUT_DIM_PAD);
    hmx_matmul_nn_f16(ws, dhidden, dlogits, scratch,
                       bs, NET_HIDDEN_DIM, NET_OUTPUT_DIM_PAD);
    ctx->time_bwd_mm += (HAP_perf_get_time_us() - t_phase);

    t_phase = HAP_perf_get_time_us();
    hvx_relu_backward_f16(dhidden, dhidden, hidden_pre,
        bs * NET_HIDDEN_DIM);
    ctx->time_bwd_other += (HAP_perf_get_time_us() - t_phase);

    /* Bwd dW1: dw1_t[832,128] = inp^T[832,bs] @ dhidden[bs,128]
     * Transpose inp[bs,832] -> scratch[832,bs], then NN matmul */
    t_phase = HAP_perf_get_time_us();
    blocked_transpose_f16_vtcm(scratch, inp, bs, NET_INPUT_DIM_PAD);
    hmx_matmul_nn_f16(ws, dw1_t, scratch, dhidden,
                       NET_INPUT_DIM_PAD, NET_HIDDEN_DIM, bs);
    ctx->time_bwd_mm += (HAP_perf_get_time_us() - t_phase);

    t_phase = HAP_perf_get_time_us();
    _Float16 db1_local[NET_HIDDEN_DIM] __attribute__((aligned(128)));
    hvx_bias_backward_f16(db1_local, dhidden, bs, NET_HIDDEN_DIM);
    ctx->time_bwd_other += (HAP_perf_get_time_us() - t_phase);

    /* === SGD UPDATE === */
    uint64_t t_sgd = HAP_perf_get_time_us();
    hvx_sgd_update_f16(w1_t, dw1_t, lr, NET_INPUT_DIM_PAD * NET_HIDDEN_DIM);
    hvx_sgd_update_f16(b1, db1_local, lr, NET_HIDDEN_DIM);
    hvx_sgd_update_f16(w2_t, dw2_t, lr, NET_HIDDEN_DIM * NET_OUTPUT_DIM_PAD);
    hvx_sgd_update_f16(b2, db2_local, lr, NET_OUTPUT_DIM_PAD);
    ctx->time_sgd += (HAP_perf_get_time_us() - t_sgd);

    ctx->process_time += (HAP_perf_get_time_us() - t1);

    *out_loss = loss;
    *out_correct = (uint32_t)correct_i;
}

/*
 * do_eval_batch -- Forward pass only on a batch already loaded into ctx->v_input.
 *
 * Returns number of correct predictions.
 */
static uint32_t do_eval_batch(struct mnist_context *ctx,
                              const uint8_t *labels, uint32_t bs)
{
    struct hmx_workspace *ws = &ctx->hmx_ws;

    /* Cache weight WH tiles for eval (weights unchanged during eval) */
    convert_rm_to_wh(ws, ws->wh_base, ctx->v_w1_t, NET_INPUT_DIM_PAD, NET_HIDDEN_DIM);
    uint32_t w1t_wh_count = (NET_HIDDEN_DIM / HMX_TILE) * (NET_INPUT_DIM_PAD / HMX_TILE);
    uint32_t wh_w2t_off = ws->wh_base + w1t_wh_count * HMX_ALIGN;
    convert_rm_to_wh(ws, wh_w2t_off, ctx->v_w2_t, NET_HIDDEN_DIM, NET_OUTPUT_DIM_PAD);

    /* Forward pass using HMX matmul */
    hmx_matmul_nn_f16_cached_wh(ws, ctx->v_hidden, ctx->v_input, ws->wh_base,
                                 bs, NET_HIDDEN_DIM, NET_INPUT_DIM_PAD);
    hvx_add_bias_f16(ctx->v_hidden, ctx->v_b1, bs, NET_HIDDEN_DIM);
    hvx_relu_forward_f16(ctx->v_hidden, ctx->v_hidden, bs * NET_HIDDEN_DIM);

    hmx_matmul_nn_f16_cached_wh(ws, ctx->v_logits, ctx->v_hidden, wh_w2t_off,
                                 bs, NET_OUTPUT_DIM_PAD, NET_HIDDEN_DIM);
    hvx_add_bias_f16(ctx->v_logits, ctx->v_b2, bs, NET_OUTPUT_DIM_PAD);

    /* Count correct predictions */
    uint32_t correct = 0;
    for (uint32_t i = 0; i < bs; i++) {
        const _Float16 *row = ctx->v_logits + i * NET_OUTPUT_DIM_PAD;
        _Float16 max_val = row[0];
        int max_j = 0;
        for (int j = 1; j < NET_OUTPUT_DIM; j++) {
            if (row[j] > max_val) { max_val = row[j]; max_j = j; }
        }
        if (max_j == labels[i]) correct++;
    }
    return correct;
}


/* ========== dspqueue message callback ========== */

static void packet_callback(dspqueue_t queue, int error, void *context) {
    struct mnist_context *ctx = (struct mnist_context *)context;

    while (1) {
        union {
            struct register_net_req  reg;
            struct train_batch_req   train;
            struct sync_req          sync;
            struct train_all_req     train_all;
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
            /* Store DDR buffer pointers (f32) */
            for (int i = 0; i < NET_BUF_COUNT; i++) {
                ctx->net_bufs[i] = (float *)bufs[i].ptr;
                ctx->net_fds[i] = bufs[i].fd;
            }
            ctx->net_registered = 1;

            /* Setup VTCM + convert weights */
            int setup_err = setup_vtcm(ctx);

            FARF(HIGH, "Network registered (%d bufs), VTCM %s",
                 NET_BUF_COUNT, ctx->vtcm_ready ? "ready" : "FAILED");

            struct dspqueue_buffer rsp_bufs[NET_BUF_COUNT];
            memset(rsp_bufs, 0, sizeof(rsp_bufs));
            for (int i = 0; i < NET_BUF_COUNT; i++) {
                rsp_bufs[i].fd = bufs[i].fd;
                rsp_bufs[i].flags = DSPQUEUE_BUFFER_FLAG_DEREF;
            }
            struct matmul_rsp rsp = { OP_REGISTER_NET, (uint32_t)setup_err };
            dspqueue_write(queue, 0, NET_BUF_COUNT, rsp_bufs,
                           sizeof(rsp), (const uint8_t *)&rsp,
                           DSPQUEUE_TIMEOUT_NONE);

        } else if (op == OP_TRAIN_BATCH && ctx->net_registered && ctx->vtcm_ready) {
            struct train_batch_req *treq = &msg.train;
            uint32_t bs = treq->batch_size;
            float lr = treq->learning_rate;

            /* Copy f16 input from DDR to VTCM via HVX bulk load */
            hvx_memcpy(ctx->v_input, bufs[0].ptr,
                       bs * NET_INPUT_DIM_PAD * sizeof(_Float16));

            float loss = 0.0f;
            uint32_t correct = 0;
            do_train_batch(ctx, treq->labels, bs, lr, &loss, &correct);

            /* Response */
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

        } else if (op == OP_EVAL && ctx->vtcm_ready) {
            struct train_batch_req *treq = &msg.train;
            uint32_t bs = treq->batch_size;

            /* Copy f16 test input from DDR to VTCM via HVX bulk load */
            hvx_memcpy(ctx->v_input, bufs[0].ptr,
                       bs * NET_INPUT_DIM_PAD * sizeof(_Float16));

            uint32_t correct = do_eval_batch(ctx, treq->labels, bs);

            struct dspqueue_buffer rsp_bufs[1];
            memset(rsp_bufs, 0, sizeof(rsp_bufs));
            rsp_bufs[0].fd = bufs[0].fd;
            rsp_bufs[0].flags = DSPQUEUE_BUFFER_FLAG_DEREF;

            struct train_batch_rsp rsp;
            rsp.op = OP_EVAL;
            rsp.status = 0;
            rsp.loss = 0.0f;
            rsp.correct = correct;

            dspqueue_write(queue, 0, 1, rsp_bufs,
                           sizeof(rsp), (const uint8_t *)&rsp,
                           DSPQUEUE_TIMEOUT_NONE);

        } else if (op == OP_TRAIN_ALL && ctx->net_registered && ctx->vtcm_ready) {
            /*
             * OP_TRAIN_ALL: Run all epochs on DSP, single round-trip.
             *
             * Input buffers:
             *   [0] = train_all_req struct (in rpcmem)
             *   [1] = all training images [num_train x INPUT_DIM_PAD] f16
             *   [2] = all training labels [num_train] uint8
             *   [3] = all test images [num_test x INPUT_DIM_PAD] f16
             *   [4] = all test labels [num_test] uint8
             * Output buffer:
             *   [5] = train_all_rsp struct (in rpcmem)
             *
             * No shuffling -- trains in sequential order.
             * TODO: shuffling could be added by ARM pre-generating a shuffle
             * index array per epoch and passing it, or DSP implementing a
             * simple PRNG for index shuffling.
             */
            struct train_all_req *areq = (struct train_all_req *)bufs[0].ptr;
            const uint16_t *all_train_images = (const uint16_t *)bufs[1].ptr;
            const uint8_t  *all_train_labels = (const uint8_t  *)bufs[2].ptr;
            const uint16_t *all_test_images  = (const uint16_t *)bufs[3].ptr;
            const uint8_t  *all_test_labels  = (const uint8_t  *)bufs[4].ptr;
            struct train_all_rsp *arsp = (struct train_all_rsp *)bufs[5].ptr;

            uint32_t num_epochs  = areq->num_epochs;
            uint32_t num_train   = areq->num_train_samples;
            uint32_t num_test    = areq->num_test_samples;
            uint32_t bs          = areq->batch_size;
            float    lr          = areq->learning_rate;

            if (num_epochs > MAX_EPOCHS) num_epochs = MAX_EPOCHS;

            uint32_t train_batches = num_train / bs;
            uint32_t test_batches  = num_test / bs;

            /* Local label buffer for one batch (labels are uint8, small) */
            uint8_t labels_local[MAX_BATCH];

            FARF(HIGH, "OP_TRAIN_ALL: %u epochs, %u train, %u test, bs=%u, lr=%.4f",
                 num_epochs, num_train, num_test, bs, (double)lr);

            for (uint32_t epoch = 0; epoch < num_epochs; epoch++) {
                float epoch_loss = 0.0f;
                uint32_t epoch_correct = 0;

                /* --- Training batches --- */
                for (uint32_t batch = 0; batch < train_batches; batch++) {
                    uint32_t offset = batch * bs;

                    /* Copy this batch's f16 input from DDR to VTCM */
                    hvx_memcpy(ctx->v_input,
                               all_train_images + (size_t)offset * NET_INPUT_DIM_PAD,
                               bs * NET_INPUT_DIM_PAD * sizeof(uint16_t));

                    /* Copy labels to local buffer */
                    memcpy(labels_local, all_train_labels + offset, bs);

                    float batch_loss = 0.0f;
                    uint32_t batch_correct = 0;
                    do_train_batch(ctx, labels_local, bs, lr,
                                   &batch_loss, &batch_correct);

                    epoch_loss += batch_loss;
                    epoch_correct += batch_correct;
                }

                arsp->epoch_losses[epoch] = epoch_loss / (float)train_batches;
                arsp->epoch_train_acc[epoch] =
                    (float)epoch_correct / (float)(train_batches * bs);

                /* --- Evaluation batches --- */
                uint32_t test_correct = 0;
                for (uint32_t batch = 0; batch < test_batches; batch++) {
                    uint32_t offset = batch * bs;

                    hvx_memcpy(ctx->v_input,
                               all_test_images + (size_t)offset * NET_INPUT_DIM_PAD,
                               bs * NET_INPUT_DIM_PAD * sizeof(uint16_t));

                    memcpy(labels_local, all_test_labels + offset, bs);

                    test_correct += do_eval_batch(ctx, labels_local, bs);
                }

                arsp->epoch_test_acc[epoch] =
                    (float)test_correct / (float)(test_batches * bs);

                FARF(HIGH, "Epoch %u: loss=%.4f train_acc=%.4f test_acc=%.4f",
                     epoch + 1,
                     (double)arsp->epoch_losses[epoch],
                     (double)arsp->epoch_train_acc[epoch],
                     (double)arsp->epoch_test_acc[epoch]);
            }
            arsp->num_epochs_done = num_epochs;

            /* Deref all 6 buffers */
            struct dspqueue_buffer rsp_bufs[6];
            memset(rsp_bufs, 0, sizeof(rsp_bufs));
            for (int i = 0; i < 6; i++) {
                rsp_bufs[i].fd = bufs[i].fd;
                rsp_bufs[i].flags = DSPQUEUE_BUFFER_FLAG_DEREF;
            }
            /* Flush the response buffer so ARM can read it */
            rsp_bufs[5].flags |= DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER;

            struct matmul_rsp rsp = { OP_TRAIN_ALL, 0 };
            dspqueue_write(queue, 0, 6, rsp_bufs,
                           sizeof(rsp), (const uint8_t *)&rsp,
                           DSPQUEUE_TIMEOUT_NONE);

        } else if (op == OP_SYNC && ctx->vtcm_ready) {
            /* Transpose W_T back to standard layout, then convert f16->f32 to DDR.
             * Use g_tn_a_buf (DDR static buffer) as scratch for the transposed f16. */

            /* W1_T[832,128] -> W1[128,832] in scratch, then f16->f32 to DDR */
            blocked_transpose_f16_vtcm((_Float16 *)g_tn_a_buf, ctx->v_w1_t,
                                       NET_INPUT_DIM_PAD, NET_HIDDEN_DIM);
            hvx_f16_to_f32(ctx->net_bufs[NET_BUF_W1], (_Float16 *)g_tn_a_buf,
                            NET_HIDDEN_DIM * NET_INPUT_DIM_PAD);

            hvx_f16_to_f32(ctx->net_bufs[NET_BUF_B1], ctx->v_b1, NET_HIDDEN_DIM);

            /* W2_T[128,64] -> W2[64,128] in scratch, then f16->f32 to DDR */
            blocked_transpose_f16_vtcm((_Float16 *)g_tn_a_buf, ctx->v_w2_t,
                                       NET_HIDDEN_DIM, NET_OUTPUT_DIM_PAD);
            hvx_f16_to_f32(ctx->net_bufs[NET_BUF_W2], (_Float16 *)g_tn_a_buf,
                            NET_OUTPUT_DIM_PAD * NET_HIDDEN_DIM);

            hvx_f16_to_f32(ctx->net_bufs[NET_BUF_B2], ctx->v_b2, NET_OUTPUT_DIM_PAD);

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
    ctx->time_fwd_mm = ctx->time_fwd_other = 0;
    ctx->time_bwd_mm = ctx->time_bwd_other = 0;
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

    FARF(HIGH, "DSP timing: fwd_mm=%llu bwd_mm=%llu fwd_other=%llu bwd_other=%llu sgd=%llu total=%llu us",
         ctx->time_fwd_mm, ctx->time_bwd_mm, ctx->time_fwd_other, ctx->time_bwd_other,
         ctx->time_sgd, ctx->process_time);
    *process_time = ctx->process_time;
    return AEE_SUCCESS;
}


/* ========== Stub ========== */

AEEResult mnist_train_do_matmul(remote_handle64 handle,
                                 const uint8 *a_buf, int a_buf_len,
                                 const uint8 *b_buf, int b_buf_len,
                                 uint8 *c_buf, int c_buf_len,
                                 uint32 m, uint32 n, uint32 k, uint32 transpose,
                                 uint64 *process_time) {
    return AEE_EUNSUPPORTED;
}
