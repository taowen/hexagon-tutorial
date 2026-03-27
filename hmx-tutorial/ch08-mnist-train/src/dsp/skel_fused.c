/*
 * Step 4: Entire training loop runs on DSP. ARM sends 1 message per batch.
 *
 * Handles OP_REGISTER_NET, OP_TRAIN_BATCH, OP_SYNC.
 * Does NOT handle OP_MATMUL since the fused path doesn't use it.
 */

#include "dsp/skel_common.h"
#include "dsp/hvx_matmul.h"
#include "dsp/hvx_ops.h"
#include "dsp/hmx_matmul.h"
#include "common/protocol.h"

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
#ifdef USE_HMX
    int      hmx_initialized;
#endif
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
            struct matmul_req matmul;
            struct register_net_req reg;
            struct train_batch_req train;
            struct sync_req sync;
            struct test_op_req test;
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

#ifdef USE_HMX
        /* Lazy HMX init in callback thread (HMX lock is thread-local) */
        if (!ctx->hmx_initialized) {
            int hmx_err = hexkl_micro_hw_init(&g_vtcm_base, &g_vtcm_size);
            if (hmx_err == AEE_SUCCESS) {
                hmx_err = hexkl_micro_hmx_lock();
            }
            if (hmx_err == AEE_SUCCESS) {
                ctx->hmx_initialized = 1;
                FARF(HIGH, "HMX initialized in callback: VTCM %u KB", g_vtcm_size / 1024);
            } else {
                FARF(ERROR, "HMX init failed: 0x%08x", (unsigned)hmx_err);
            }
        }
#endif

        if (op == OP_MATMUL && n_bufs >= 3) {
            struct matmul_req *mreq = (struct matmul_req *)&msg;
#ifdef USE_HMX
            hmx_matmul_dispatch(g_vtcm_base, g_vtcm_size,
                (float *)bufs[2].ptr,
                (const float *)bufs[0].ptr,
                (const float *)bufs[1].ptr,
                mreq->m, mreq->n, mreq->k,
                mreq->transpose, mreq->accumulate);
#else
            do_matmul((float *)bufs[2].ptr,
                      (const float *)bufs[0].ptr,
                      (const float *)bufs[1].ptr,
                      mreq->m, mreq->n, mreq->k,
                      mreq->transpose, mreq->accumulate);
#endif
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
            struct matmul_rsp rsp = { OP_MATMUL, 0 };
            dspqueue_write(queue, 0, 3, rsp_bufs,
                           sizeof(rsp), (const uint8_t *)&rsp,
                           DSPQUEUE_TIMEOUT_NONE);

        } else if (op == OP_REGISTER_NET && n_bufs >= NET_BUF_COUNT) {
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
#ifdef USE_HMX
            hmx_matmul_dispatch(g_vtcm_base, g_vtcm_size,
                hidden, input, w1, bs, NET_HIDDEN_DIM, NET_INPUT_DIM_PAD, 1, 0);
#else
            matmul_nt(hidden, input, w1, bs, NET_HIDDEN_DIM, NET_INPUT_DIM_PAD);
#endif
            ctx->time_fwd_mm += (HAP_perf_get_time_us() - t_phase);

            t_phase = HAP_perf_get_time_us();
            hvx_add_bias(hidden, b1, bs, NET_HIDDEN_DIM);
            memcpy(hidden_pre, hidden, bs * NET_HIDDEN_DIM * sizeof(float));
            hvx_relu_forward(hidden, bs * NET_HIDDEN_DIM);
            ctx->time_fwd_other += (HAP_perf_get_time_us() - t_phase);

            /* Layer 2: logits = hidden @ W2^T + b2 */
            t_phase = HAP_perf_get_time_us();
#ifdef USE_HMX
            hmx_matmul_dispatch(g_vtcm_base, g_vtcm_size,
                logits, hidden, w2, bs, NET_OUTPUT_DIM_PAD, NET_HIDDEN_DIM, 1, 0);
#else
            matmul_nt(logits, hidden, w2, bs, NET_OUTPUT_DIM_PAD, NET_HIDDEN_DIM);
#endif
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

            /* dW2 = dlogits^T @ hidden (NOT accumulating) */
            t_phase = HAP_perf_get_time_us();
#ifdef USE_HMX
            memset(dw2, 0, NET_OUTPUT_DIM_PAD * NET_HIDDEN_DIM * sizeof(float));
            hmx_matmul_dispatch(g_vtcm_base, g_vtcm_size,
                dw2, dlogits, hidden, NET_OUTPUT_DIM_PAD, NET_HIDDEN_DIM, bs, 2, 0);
#else
            matmul_tn(dw2, dlogits, hidden,
                      NET_OUTPUT_DIM_PAD, NET_HIDDEN_DIM, bs);
#endif
            ctx->time_bwd_mm += (HAP_perf_get_time_us() - t_phase);

            /* db2 = sum_rows(dlogits) */
            t_phase = HAP_perf_get_time_us();
            float db2_local[NET_OUTPUT_DIM_PAD] __attribute__((aligned(128)));
            hvx_bias_backward(db2_local, dlogits, bs, NET_OUTPUT_DIM_PAD);
            ctx->time_bwd_other += (HAP_perf_get_time_us() - t_phase);

            /* dhidden = dlogits @ W2 */
            t_phase = HAP_perf_get_time_us();
#ifdef USE_HMX
            hmx_matmul_dispatch(g_vtcm_base, g_vtcm_size,
                dhidden, dlogits, w2, bs, NET_HIDDEN_DIM, NET_OUTPUT_DIM_PAD, 0, 0);
#else
            matmul_nn(dhidden, dlogits, w2,
                      bs, NET_HIDDEN_DIM, NET_OUTPUT_DIM_PAD);
#endif
            ctx->time_bwd_mm += (HAP_perf_get_time_us() - t_phase);

            /* ReLU backward */
            t_phase = HAP_perf_get_time_us();
            hvx_relu_backward(dhidden, hidden_pre, bs * NET_HIDDEN_DIM);
            ctx->time_bwd_other += (HAP_perf_get_time_us() - t_phase);

            /* dW1 = dhidden^T @ input (NOT accumulating) */
            t_phase = HAP_perf_get_time_us();
#ifdef USE_HMX
            memset(dw1, 0, NET_HIDDEN_DIM * NET_INPUT_DIM_PAD * sizeof(float));
            hmx_matmul_dispatch(g_vtcm_base, g_vtcm_size,
                dw1, dhidden, input, NET_HIDDEN_DIM, NET_INPUT_DIM_PAD, bs, 2, 0);
#else
            matmul_tn(dw1, dhidden, input,
                      NET_HIDDEN_DIM, NET_INPUT_DIM_PAD, bs);
#endif
            ctx->time_bwd_mm += (HAP_perf_get_time_us() - t_phase);

            /* db1 = sum_rows(dhidden) */
            t_phase = HAP_perf_get_time_us();
            float db1_local[NET_HIDDEN_DIM] __attribute__((aligned(128)));
            hvx_bias_backward(db1_local, dhidden, bs, NET_HIDDEN_DIM);
            ctx->time_bwd_other += (HAP_perf_get_time_us() - t_phase);

            /* === SGD UPDATE === */
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

        } else if (op == OP_TEST_RELU_FWD && n_bufs >= 1) {
            float *x = (float *)bufs[0].ptr;
            uint32_t n = msg.test.param1;
            hvx_relu_forward(x, n);
            struct dspqueue_buffer rsp_bufs[1];
            memset(rsp_bufs, 0, sizeof(rsp_bufs));
            rsp_bufs[0].fd = bufs[0].fd;
            rsp_bufs[0].flags = DSPQUEUE_BUFFER_FLAG_DEREF
                              | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
                              | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
            struct matmul_rsp rsp = { op, 0 };
            dspqueue_write(queue, 0, 1, rsp_bufs, sizeof(rsp), (const uint8_t *)&rsp, DSPQUEUE_TIMEOUT_NONE);

        } else if (op == OP_TEST_RELU_BWD && n_bufs >= 2) {
            float *dx = (float *)bufs[0].ptr;
            const float *pre_relu = (const float *)bufs[1].ptr;
            uint32_t n = msg.test.param1;
            hvx_relu_backward(dx, pre_relu, n);
            struct dspqueue_buffer rsp_bufs[2];
            memset(rsp_bufs, 0, sizeof(rsp_bufs));
            rsp_bufs[0].fd = bufs[0].fd;
            rsp_bufs[0].flags = DSPQUEUE_BUFFER_FLAG_DEREF
                              | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
                              | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
            rsp_bufs[1].fd = bufs[1].fd;
            rsp_bufs[1].flags = DSPQUEUE_BUFFER_FLAG_DEREF;
            struct matmul_rsp rsp = { op, 0 };
            dspqueue_write(queue, 0, 2, rsp_bufs, sizeof(rsp), (const uint8_t *)&rsp, DSPQUEUE_TIMEOUT_NONE);

        } else if (op == OP_TEST_BIAS_BWD && n_bufs >= 2) {
            const float *dout = (const float *)bufs[0].ptr;
            float *db = (float *)bufs[1].ptr;
            uint32_t batch = msg.test.param1;
            uint32_t dim = msg.test.param2;
            hvx_bias_backward(db, dout, batch, dim);
            struct dspqueue_buffer rsp_bufs[2];
            memset(rsp_bufs, 0, sizeof(rsp_bufs));
            rsp_bufs[0].fd = bufs[0].fd;
            rsp_bufs[0].flags = DSPQUEUE_BUFFER_FLAG_DEREF;
            rsp_bufs[1].fd = bufs[1].fd;
            rsp_bufs[1].flags = DSPQUEUE_BUFFER_FLAG_DEREF
                              | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
                              | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
            struct matmul_rsp rsp = { op, 0 };
            dspqueue_write(queue, 0, 2, rsp_bufs, sizeof(rsp), (const uint8_t *)&rsp, DSPQUEUE_TIMEOUT_NONE);

        } else if (op == OP_TEST_ADD_BIAS && n_bufs >= 2) {
            float *out = (float *)bufs[0].ptr;
            const float *bias = (const float *)bufs[1].ptr;
            uint32_t batch = msg.test.param1;
            uint32_t dim = msg.test.param2;
            hvx_add_bias(out, bias, batch, dim);
            struct dspqueue_buffer rsp_bufs[2];
            memset(rsp_bufs, 0, sizeof(rsp_bufs));
            rsp_bufs[0].fd = bufs[0].fd;
            rsp_bufs[0].flags = DSPQUEUE_BUFFER_FLAG_DEREF
                              | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
                              | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
            rsp_bufs[1].fd = bufs[1].fd;
            rsp_bufs[1].flags = DSPQUEUE_BUFFER_FLAG_DEREF;
            struct matmul_rsp rsp = { op, 0 };
            dspqueue_write(queue, 0, 2, rsp_bufs, sizeof(rsp), (const uint8_t *)&rsp, DSPQUEUE_TIMEOUT_NONE);

        } else if (op == OP_TEST_SOFTMAX_CE && n_bufs >= 2) {
            const float *logits = (const float *)bufs[0].ptr;
            float *probs = (float *)bufs[1].ptr;
            uint32_t batch = msg.test.param1;
            float loss = hvx_softmax_cross_entropy(logits, msg.test.labels, probs, batch);
            struct dspqueue_buffer rsp_bufs[2];
            memset(rsp_bufs, 0, sizeof(rsp_bufs));
            rsp_bufs[0].fd = bufs[0].fd;
            rsp_bufs[0].flags = DSPQUEUE_BUFFER_FLAG_DEREF;
            rsp_bufs[1].fd = bufs[1].fd;
            rsp_bufs[1].flags = DSPQUEUE_BUFFER_FLAG_DEREF
                              | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
                              | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
            struct test_op_rsp rsp = { op, 0, loss, 0 };
            dspqueue_write(queue, 0, 2, rsp_bufs, sizeof(rsp), (const uint8_t *)&rsp, DSPQUEUE_TIMEOUT_NONE);

        } else if (op == OP_TEST_DLOGITS && n_bufs >= 2) {
            float *dlogits = (float *)bufs[0].ptr;
            const float *probs = (const float *)bufs[1].ptr;
            uint32_t batch = msg.test.param1;
            hvx_compute_dlogits(dlogits, probs, msg.test.labels, batch);
            struct dspqueue_buffer rsp_bufs[2];
            memset(rsp_bufs, 0, sizeof(rsp_bufs));
            rsp_bufs[0].fd = bufs[0].fd;
            rsp_bufs[0].flags = DSPQUEUE_BUFFER_FLAG_DEREF
                              | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
                              | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
            rsp_bufs[1].fd = bufs[1].fd;
            rsp_bufs[1].flags = DSPQUEUE_BUFFER_FLAG_DEREF;
            struct matmul_rsp rsp = { op, 0 };
            dspqueue_write(queue, 0, 2, rsp_bufs, sizeof(rsp), (const uint8_t *)&rsp, DSPQUEUE_TIMEOUT_NONE);

        } else if (op == OP_TEST_SGD && n_bufs >= 2) {
            float *w = (float *)bufs[0].ptr;
            const float *grad = (const float *)bufs[1].ptr;
            uint32_t n = msg.test.param1;
            uint32_t lr_bits = msg.test.param2;
            float lr = *(float *)&lr_bits;
            hvx_sgd_update(w, grad, lr, n);
            struct dspqueue_buffer rsp_bufs[2];
            memset(rsp_bufs, 0, sizeof(rsp_bufs));
            rsp_bufs[0].fd = bufs[0].fd;
            rsp_bufs[0].flags = DSPQUEUE_BUFFER_FLAG_DEREF
                              | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
                              | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
            rsp_bufs[1].fd = bufs[1].fd;
            rsp_bufs[1].flags = DSPQUEUE_BUFFER_FLAG_DEREF;
            struct matmul_rsp rsp = { op, 0 };
            dspqueue_write(queue, 0, 2, rsp_bufs, sizeof(rsp), (const uint8_t *)&rsp, DSPQUEUE_TIMEOUT_NONE);

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
