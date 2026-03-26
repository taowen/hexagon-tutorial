/*
 * Step 4: Entire training loop runs on DSP. ARM sends 1 message per batch.
 *
 * Handles OP_REGISTER_NET, OP_TRAIN_BATCH, OP_SYNC.
 * Does NOT handle OP_MATMUL since the fused path doesn't use it.
 */

#include "dsp_common.h"
#include "hvx_matmul.h"
#include "hvx_ops.h"
#include "hmx_matmul.h"
#include "mnist_train_shared.h"

/* ---------- DSP context ---------- */
struct mnist_context {
    dspqueue_t queue;
    uint64_t   process_time;   /* cumulative DSP processing time (us) */
    uint64_t time_fwd_mm;     /* forward matmuls */
    uint64_t time_fwd_other;  /* bias + relu + softmax */
    uint64_t time_bwd_mm;     /* backward matmuls */
    uint64_t time_bwd_other;  /* dlogits prep + relu_backward + db sums */
    uint64_t time_sgd;        /* SGD update */
    /* Registered network buffers (pointers and fds) */
    float   *net_bufs[NET_BUF_COUNT];
    int      net_fds[NET_BUF_COUNT];
    int      net_registered;
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

        if (op == OP_REGISTER_NET && n_bufs >= NET_BUF_COUNT) {
            /* Store buffer pointers and fds */
            for (int i = 0; i < NET_BUF_COUNT; i++) {
                ctx->net_bufs[i] = (float *)bufs[i].ptr;
                ctx->net_fds[i] = bufs[i].fd;
            }
            ctx->net_registered = 1;
            FARF(HIGH, "Network buffers registered (%d bufs)", NET_BUF_COUNT);

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
            /* Fused forward + backward + SGD */
            struct train_batch_req *treq = &msg.train;
            uint32_t bs = treq->batch_size;
            float lr = treq->learning_rate;

            float *input      = (float *)bufs[0].ptr;
            float *w1         = ctx->net_bufs[NET_BUF_W1];
            float *b1         = ctx->net_bufs[NET_BUF_B1];
            float *w2         = ctx->net_bufs[NET_BUF_W2];
            float *b2         = ctx->net_bufs[NET_BUF_B2];
            float *dw1        = ctx->net_bufs[NET_BUF_DW1];
            float *dw2        = ctx->net_bufs[NET_BUF_DW2];
            float *hidden     = ctx->net_bufs[NET_BUF_HIDDEN];
            float *logits     = ctx->net_bufs[NET_BUF_LOGITS];
            float *dhidden    = ctx->net_bufs[NET_BUF_DHIDDEN];
            float *dlogits    = ctx->net_bufs[NET_BUF_DLOGITS];
            float *hidden_pre = ctx->net_bufs[NET_BUF_HIDDEN_PRE];
            float *probs      = ctx->net_bufs[NET_BUF_PROBS];

            uint64_t t1 = HAP_perf_get_time_us();
            uint64_t t_phase;

            /* === FORWARD === */
            /* Layer 1: hidden = input @ W1^T + b1 */
            t_phase = HAP_perf_get_time_us();
            matmul_nt(hidden, input, w1, bs, NET_HIDDEN_DIM, NET_INPUT_DIM_PAD);
            ctx->time_fwd_mm += (HAP_perf_get_time_us() - t_phase);

            t_phase = HAP_perf_get_time_us();
            dsp_add_bias(hidden, b1, bs, NET_HIDDEN_DIM);
            memcpy(hidden_pre, hidden, bs * NET_HIDDEN_DIM * sizeof(float));
            dsp_relu_forward(hidden, bs * NET_HIDDEN_DIM);
            ctx->time_fwd_other += (HAP_perf_get_time_us() - t_phase);

            /* Layer 2: logits = hidden @ W2^T + b2 */
            t_phase = HAP_perf_get_time_us();
            matmul_nt(logits, hidden, w2, bs, NET_OUTPUT_DIM_PAD, NET_HIDDEN_DIM);
            ctx->time_fwd_mm += (HAP_perf_get_time_us() - t_phase);

            t_phase = HAP_perf_get_time_us();
            dsp_add_bias(logits, b2, bs, NET_OUTPUT_DIM_PAD);

            /* Softmax + cross-entropy loss */
            float loss = dsp_softmax_cross_entropy(logits, treq->labels, probs, bs);

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
            memcpy(dlogits, probs, bs * NET_OUTPUT_DIM_PAD * sizeof(float));
            {
                float inv_bs = 1.0f / (float)bs;
                HVX_Vector inv_bs_vec = Q6_V_vsplat_R(*(int32_t *)&inv_bs);
                for (uint32_t b_idx = 0; b_idx < bs; b_idx++) {
                    float *row = dlogits + b_idx * NET_OUTPUT_DIM_PAD;
                    /* Subtract 1.0 from the label position */
                    row[treq->labels[b_idx]] -= 1.0f;
                    /* Divide by batch_size using multiply-by-reciprocal */
                    HVX_Vector v = *(HVX_Vector *)row;
                    *(HVX_Vector *)row = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v, inv_bs_vec));
                }
            }
            ctx->time_bwd_other += (HAP_perf_get_time_us() - t_phase);

            /* dW2 = dlogits^T @ hidden (NOT accumulating) */
            t_phase = HAP_perf_get_time_us();
            matmul_tn(dw2, dlogits, hidden,
                      NET_OUTPUT_DIM_PAD, NET_HIDDEN_DIM, bs);
            ctx->time_bwd_mm += (HAP_perf_get_time_us() - t_phase);

            /* db2 = sum_rows(dlogits) -- compute on stack */
            t_phase = HAP_perf_get_time_us();
            float db2_local[NET_OUTPUT_DIM_PAD] __attribute__((aligned(128)));
            {
                HVX_Vector one_v = Q6_V_vsplat_R(0x3f800000);
                HVX_Vector acc = Q6_V_vzero();
                for (uint32_t b_idx = 0; b_idx < bs; b_idx++) {
                    HVX_Vector row = *(HVX_Vector *)(dlogits + b_idx * NET_OUTPUT_DIM_PAD);
                    acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(row, one_v));
                }
                *(HVX_Vector *)db2_local = Q6_Vsf_equals_Vqf32(acc);
            }
            ctx->time_bwd_other += (HAP_perf_get_time_us() - t_phase);

            /* dhidden = dlogits @ W2 */
            t_phase = HAP_perf_get_time_us();
            matmul_nn(dhidden, dlogits, w2,
                      bs, NET_HIDDEN_DIM, NET_OUTPUT_DIM_PAD);
            ctx->time_bwd_mm += (HAP_perf_get_time_us() - t_phase);

            /* ReLU backward */
            t_phase = HAP_perf_get_time_us();
            dsp_relu_backward(dhidden, hidden_pre, bs * NET_HIDDEN_DIM);
            ctx->time_bwd_other += (HAP_perf_get_time_us() - t_phase);

            /* dW1 = dhidden^T @ input (NOT accumulating) */
            t_phase = HAP_perf_get_time_us();
            matmul_tn(dw1, dhidden, input,
                      NET_HIDDEN_DIM, NET_INPUT_DIM_PAD, bs);
            ctx->time_bwd_mm += (HAP_perf_get_time_us() - t_phase);

            /* db1 = sum_rows(dhidden) -- compute on stack */
            t_phase = HAP_perf_get_time_us();
            float db1_local[NET_HIDDEN_DIM] __attribute__((aligned(128)));
            {
                HVX_Vector one_v = Q6_V_vsplat_R(0x3f800000);
                HVX_Vector acc0 = Q6_V_vzero();
                HVX_Vector acc1 = Q6_V_vzero();
                HVX_Vector acc2 = Q6_V_vzero();
                HVX_Vector acc3 = Q6_V_vzero();
                for (uint32_t b_idx = 0; b_idx < bs; b_idx++) {
                    const float *row = dhidden + b_idx * NET_HIDDEN_DIM;
                    acc0 = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, Q6_Vqf32_vmpy_VsfVsf(*(HVX_Vector *)(row), one_v));
                    acc1 = Q6_Vqf32_vadd_Vqf32Vqf32(acc1, Q6_Vqf32_vmpy_VsfVsf(*(HVX_Vector *)(row + HVX_FLOATS), one_v));
                    acc2 = Q6_Vqf32_vadd_Vqf32Vqf32(acc2, Q6_Vqf32_vmpy_VsfVsf(*(HVX_Vector *)(row + HVX_FLOATS * 2), one_v));
                    acc3 = Q6_Vqf32_vadd_Vqf32Vqf32(acc3, Q6_Vqf32_vmpy_VsfVsf(*(HVX_Vector *)(row + HVX_FLOATS * 3), one_v));
                }
                *(HVX_Vector *)(db1_local) = Q6_Vsf_equals_Vqf32(acc0);
                *(HVX_Vector *)(db1_local + HVX_FLOATS) = Q6_Vsf_equals_Vqf32(acc1);
                *(HVX_Vector *)(db1_local + HVX_FLOATS * 2) = Q6_Vsf_equals_Vqf32(acc2);
                *(HVX_Vector *)(db1_local + HVX_FLOATS * 3) = Q6_Vsf_equals_Vqf32(acc3);
            }
            ctx->time_bwd_other += (HAP_perf_get_time_us() - t_phase);

            /* === SGD UPDATE (HVX vectorized) === */
            uint64_t t_sgd = HAP_perf_get_time_us();
            HVX_Vector one = Q6_V_vsplat_R(0x3f800000); /* 1.0f */
            {
                HVX_Vector lr_vec = Q6_V_vsplat_R(*(int32_t *)&lr);
                /* W1 update: 128 * 800 = 102400 floats */
                uint32_t sz1 = NET_HIDDEN_DIM * NET_INPUT_DIM_PAD;
                uint32_t sz1_vec = sz1 & ~(HVX_FLOATS - 1);
                for (uint32_t i = 0; i < sz1_vec; i += HVX_FLOATS) {
                    HVX_Vector wv = *(HVX_Vector *)(w1 + i);
                    HVX_Vector gv = *(HVX_Vector *)(dw1 + i);
                    HVX_Vector prod = Q6_Vqf32_vmpy_VsfVsf(lr_vec, gv);
                    HVX_Vector wq = Q6_Vqf32_vmpy_VsfVsf(wv, one);
                    *(HVX_Vector *)(w1 + i) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(wq, prod));
                }
                for (uint32_t i = sz1_vec; i < sz1; i++)
                    w1[i] -= lr * dw1[i];
            }
            /* b1 update */
            for (int i = 0; i < NET_HIDDEN_DIM; i++)
                b1[i] -= lr * db1_local[i];
            {
                HVX_Vector lr_vec = Q6_V_vsplat_R(*(int32_t *)&lr);
                /* W2 update: 32 * 128 = 4096 floats */
                uint32_t sz2 = NET_OUTPUT_DIM_PAD * NET_HIDDEN_DIM;
                uint32_t sz2_vec = sz2 & ~(HVX_FLOATS - 1);
                for (uint32_t i = 0; i < sz2_vec; i += HVX_FLOATS) {
                    HVX_Vector wv = *(HVX_Vector *)(w2 + i);
                    HVX_Vector gv = *(HVX_Vector *)(dw2 + i);
                    HVX_Vector prod = Q6_Vqf32_vmpy_VsfVsf(lr_vec, gv);
                    HVX_Vector wq = Q6_Vqf32_vmpy_VsfVsf(wv, one);
                    *(HVX_Vector *)(w2 + i) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(wq, prod));
                }
                for (uint32_t i = sz2_vec; i < sz2; i++)
                    w2[i] -= lr * dw2[i];
            }
            /* b2 update */
            for (int i = 0; i < NET_OUTPUT_DIM_PAD; i++)
                b2[i] -= lr * db2_local[i];
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
            FARF(ERROR, "Unknown op %u or bad state (n_bufs=%u, registered=%d)",
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

    int err = dspqueue_close(ctx->queue);
    ctx->queue = NULL;
    if (err) return err;

    FARF(HIGH, "DSP timing: fwd_mm=%llu bwd_mm=%llu fwd_other=%llu bwd_other=%llu sgd=%llu total=%llu us",
         ctx->time_fwd_mm, ctx->time_bwd_mm, ctx->time_fwd_other, ctx->time_bwd_other, ctx->time_sgd, ctx->process_time);
    *process_time = ctx->process_time;
    return AEE_SUCCESS;
}


/* ========== Stub (not used by step 4) ========== */

AEEResult mnist_train_do_matmul(remote_handle64 handle,
                                 const uint8 *a_buf, int a_buf_len,
                                 const uint8 *b_buf, int b_buf_len,
                                 uint8 *c_buf, int c_buf_len,
                                 uint32 m, uint32 n, uint32 k, uint32 transpose,
                                 uint64 *process_time) {
    return AEE_EUNSUPPORTED;
}
