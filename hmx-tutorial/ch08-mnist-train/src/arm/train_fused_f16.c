/*
 * train_fused_f16.c -- MNIST f16 training: dspqueue fused training (1 call/batch)
 *
 * Same architecture as train_fused.c but all network buffers use _Float16.
 * The DSP skel (skel_fused_f16.c) uses HMX for f16 matmul operations.
 *
 * Setup:
 *   1. OP_REGISTER_NET -- one-time registration of all network buffer fds
 *   2. OP_TRAIN_BATCH  -- one message per batch (forward+backward+SGD on DSP)
 *   3. OP_SYNC         -- flush DSP caches so ARM can read weights for eval
 *
 * Network: 784->128(ReLU)->10 (Softmax+CrossEntropy, SGD)
 * Padded:  800->128->32
 *
 * Usage: ./train_fused_f16 [epochs] [batch_size]
 *   Default: 5 epochs, 32 batch size
 */

#include "common/common.h"
#include "arm/data.h"
#include "arm/dspqueue_mgr.h"

#ifndef CDSP_DOMAIN_ID
#define CDSP_DOMAIN_ID 3
#endif

/* ====================================================================
 * f16 network structure (overrides the f32 network_t from common.h)
 * ==================================================================== */

typedef struct {
    /* Weights and biases */
    _Float16 *w1;          /* [HIDDEN_DIM x INPUT_DIM_PAD] */
    _Float16 *b1;          /* [HIDDEN_DIM] */
    _Float16 *w2;          /* [OUTPUT_DIM_PAD x HIDDEN_DIM] */
    _Float16 *b2;          /* [OUTPUT_DIM_PAD] */

    /* Gradients */
    _Float16 *dw1, *db1, *dw2, *db2;

    /* Intermediate buffers */
    _Float16 *hidden;          /* [batch x HIDDEN_DIM] */
    _Float16 *logits;          /* [batch x OUTPUT_DIM_PAD] */
    _Float16 *probs;           /* [batch x OUTPUT_DIM_PAD] */

    /* Input batch buffer */
    _Float16 *batch_buf;       /* [batch x INPUT_DIM_PAD] */
} network_f16_t;

/* ====================================================================
 * f16 network init / free
 * ==================================================================== */

static int network_f16_init(network_f16_t *net) {
    memset(net, 0, sizeof(*net));

    /* All buffers in shared memory via net_alloc (rpcmem in dspqueue mode) */
    net->w1  = (_Float16 *)net_alloc(HIDDEN_DIM * INPUT_DIM_PAD * sizeof(_Float16));
    net->b1  = (_Float16 *)net_alloc(HIDDEN_DIM * sizeof(_Float16));
    net->w2  = (_Float16 *)net_alloc(OUTPUT_DIM_PAD * HIDDEN_DIM * sizeof(_Float16));
    net->b2  = (_Float16 *)net_alloc(OUTPUT_DIM_PAD * sizeof(_Float16));

    net->dw1 = (_Float16 *)net_alloc(HIDDEN_DIM * INPUT_DIM_PAD * sizeof(_Float16));
    net->db1 = (_Float16 *)net_alloc(HIDDEN_DIM * sizeof(_Float16));
    net->dw2 = (_Float16 *)net_alloc(OUTPUT_DIM_PAD * HIDDEN_DIM * sizeof(_Float16));
    net->db2 = (_Float16 *)net_alloc(OUTPUT_DIM_PAD * sizeof(_Float16));

    net->hidden = (_Float16 *)net_alloc(g_batch_size * HIDDEN_DIM * sizeof(_Float16));
    net->logits = (_Float16 *)net_alloc(g_batch_size * OUTPUT_DIM_PAD * sizeof(_Float16));
    net->probs  = (_Float16 *)net_alloc(g_batch_size * OUTPUT_DIM_PAD * sizeof(_Float16));

    net->batch_buf = (_Float16 *)net_alloc(g_batch_size * INPUT_DIM_PAD * sizeof(_Float16));

    if (!net->w1 || !net->b1 || !net->w2 || !net->b2 ||
        !net->dw1 || !net->db1 || !net->dw2 || !net->db2 ||
        !net->hidden || !net->logits || !net->probs || !net->batch_buf) {
        printf("[ERROR] f16 network malloc failed\n");
        return -1;
    }

    /* He init: scale = sqrt(2.0 / fan_in) */
    float scale1 = sqrtf(2.0f / (float)INPUT_DIM_PAD);
    for (int i = 0; i < HIDDEN_DIM * INPUT_DIM_PAD; i++)
        net->w1[i] = (_Float16)(rand_normal() * scale1);

    float scale2 = sqrtf(2.0f / (float)HIDDEN_DIM);
    for (int i = 0; i < OUTPUT_DIM_PAD * HIDDEN_DIM; i++)
        net->w2[i] = (_Float16)(rand_normal() * scale2);

    /* Zero padding rows of W2 */
    for (int i = OUTPUT_DIM; i < OUTPUT_DIM_PAD; i++)
        for (int j = 0; j < HIDDEN_DIM; j++)
            net->w2[i * HIDDEN_DIM + j] = (_Float16)0;

    /* Zero biases */
    for (int i = 0; i < HIDDEN_DIM; i++) net->b1[i] = (_Float16)0;
    for (int i = 0; i < OUTPUT_DIM_PAD; i++) net->b2[i] = (_Float16)0;

    return 0;
}

static void network_f16_free(network_f16_t *net) {
    net_free(net->batch_buf);
    net_free(net->probs);
    net_free(net->logits);
    net_free(net->hidden);
    net_free(net->db2); net_free(net->db1);
    net_free(net->dw2); net_free(net->dw1);
    net_free(net->b2);  net_free(net->b1);
    net_free(net->w2);  net_free(net->w1);
}

/* ====================================================================
 * Register f16 network buffers with DSP via dspqueue
 *
 * Same protocol as f32: send NET_BUF_COUNT buffer fds with OP_REGISTER_NET.
 * The DSP skel maps them identically -- it just interprets them as f16.
 *
 * Note: We do NOT register hidden_pre_relu or dhidden/dlogits separately
 * because the fused skel manages those internally. We reuse the same
 * NET_BUF_* indices but some slots may be unused by the f16 skel.
 * ==================================================================== */

static void dspqueue_register_net_f16(network_f16_t *net) {
    struct register_net_req req;
    memset(&req, 0, sizeof(req));
    req.op = OP_REGISTER_NET;

    struct dspqueue_buffer bufs[NET_BUF_COUNT];
    memset(bufs, 0, sizeof(bufs));

    /* Map buffer index -> network pointer
     * For f16, we register the same slots. The DSP skel knows the types. */
    void *ptrs[NET_BUF_COUNT];
    ptrs[NET_BUF_W1]         = net->w1;
    ptrs[NET_BUF_B1]         = net->b1;
    ptrs[NET_BUF_W2]         = net->w2;
    ptrs[NET_BUF_B2]         = net->b2;
    ptrs[NET_BUF_DW1]        = net->dw1;
    ptrs[NET_BUF_DW2]        = net->dw2;
    ptrs[NET_BUF_HIDDEN]     = net->hidden;
    ptrs[NET_BUF_LOGITS]     = net->logits;
    ptrs[NET_BUF_DHIDDEN]    = net->hidden;   /* reuse hidden slot for dhidden */
    ptrs[NET_BUF_DLOGITS]    = net->logits;   /* reuse logits slot for dlogits */
    ptrs[NET_BUF_HIDDEN_PRE] = net->hidden;   /* reuse hidden slot */
    ptrs[NET_BUF_PROBS]      = net->probs;

    for (int i = 0; i < NET_BUF_COUNT; i++) {
        int fd = dspq_find_fd(ptrs[i]);
        if (fd < 0) {
            fprintf(stderr, "[ERROR] f16 buffer %d not in shared memory\n", i);
            return;
        }
        bufs[i].fd = fd;
        bufs[i].flags = DSPQUEUE_BUFFER_FLAG_REF
                       | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER;
    }

    g_dspq.ops_done = 0;
    g_dspq.ops_total = 1;
    int err = dspqueue_write(g_dspq.queue, 0, NET_BUF_COUNT, bufs,
                              sizeof(req), (const uint8_t *)&req, 1000000);
    if (err != 0) {
        fprintf(stderr, "[ERROR] f16 register_net write failed: 0x%08x\n", (unsigned)err);
        return;
    }
    sem_wait(&g_dspq.done_sem);
    printf("[DSPQ] f16 network buffers registered with DSP\n");
}

/* ====================================================================
 * Sync: flush DSP caches so ARM can read updated f16 weights
 * ==================================================================== */

static void dspqueue_sync_f16(network_f16_t *net) {
    struct sync_req req;
    memset(&req, 0, sizeof(req));
    req.op = OP_SYNC;

    struct dspqueue_buffer bufs[4];
    memset(bufs, 0, sizeof(bufs));

    void *weight_ptrs[4] = { net->w1, net->b1, net->w2, net->b2 };
    for (int i = 0; i < 4; i++) {
        bufs[i].fd = dspq_find_fd(weight_ptrs[i]);
        bufs[i].flags = DSPQUEUE_BUFFER_FLAG_REF;
    }

    g_dspq.ops_done = 0;
    g_dspq.ops_total = 1;
    dspqueue_write(g_dspq.queue, 0, 4, bufs,
                   sizeof(req), (const uint8_t *)&req, 1000000);
    sem_wait(&g_dspq.done_sem);
}

/* ====================================================================
 * Test evaluation: f16 weights, f32 argmax
 * ==================================================================== */

static float evaluate_f16(network_f16_t *net, const float *images,
                           const uint8_t *labels, int count) {
    /*
     * For evaluation we do a simple CPU forward pass using f16 weights
     * converted to f32 on the fly. This avoids needing a full f16 CPU
     * matmul path -- evaluation is not perf critical.
     */
    int correct = 0, total = 0;
    int eval_batch = g_batch_size;

    /* Temporary f32 buffers for CPU eval */
    float *w1_f32  = (float *)malloc(HIDDEN_DIM * INPUT_DIM_PAD * sizeof(float));
    float *b1_f32  = (float *)malloc(HIDDEN_DIM * sizeof(float));
    float *w2_f32  = (float *)malloc(OUTPUT_DIM_PAD * HIDDEN_DIM * sizeof(float));
    float *b2_f32  = (float *)malloc(OUTPUT_DIM_PAD * sizeof(float));
    float *hidden  = (float *)malloc(eval_batch * HIDDEN_DIM * sizeof(float));
    float *logits  = (float *)malloc(eval_batch * OUTPUT_DIM_PAD * sizeof(float));

    if (!w1_f32 || !b1_f32 || !w2_f32 || !b2_f32 || !hidden || !logits) {
        printf("[ERROR] eval malloc failed\n");
        free(w1_f32); free(b1_f32); free(w2_f32); free(b2_f32);
        free(hidden); free(logits);
        return 0.0f;
    }

    /* Convert f16 weights to f32 */
    for (int i = 0; i < HIDDEN_DIM * INPUT_DIM_PAD; i++)
        w1_f32[i] = (float)net->w1[i];
    for (int i = 0; i < HIDDEN_DIM; i++)
        b1_f32[i] = (float)net->b1[i];
    for (int i = 0; i < OUTPUT_DIM_PAD * HIDDEN_DIM; i++)
        w2_f32[i] = (float)net->w2[i];
    for (int i = 0; i < OUTPUT_DIM_PAD; i++)
        b2_f32[i] = (float)net->b2[i];

    for (int start = 0; start + eval_batch <= count; start += eval_batch) {
        const float *batch_input = images + (size_t)start * INPUT_DIM_PAD;

        /* Layer 1: hidden = input @ W1^T + b1, ReLU */
        cpu_matmul_nt(eval_batch, HIDDEN_DIM, INPUT_DIM_PAD,
                      hidden, batch_input, w1_f32);
        add_bias(hidden, b1_f32, eval_batch, HIDDEN_DIM);
        relu_forward(hidden, eval_batch * HIDDEN_DIM);

        /* Layer 2: logits = hidden @ W2^T + b2 */
        cpu_matmul_nt(eval_batch, OUTPUT_DIM_PAD, HIDDEN_DIM,
                      logits, hidden, w2_f32);
        add_bias(logits, b2_f32, eval_batch, OUTPUT_DIM_PAD);

        /* Argmax */
        for (int b = 0; b < eval_batch; b++) {
            float max_val = logits[b * OUTPUT_DIM_PAD];
            int max_idx = 0;
            for (int j = 1; j < OUTPUT_DIM; j++) {
                if (logits[b * OUTPUT_DIM_PAD + j] > max_val) {
                    max_val = logits[b * OUTPUT_DIM_PAD + j];
                    max_idx = j;
                }
            }
            if (max_idx == labels[start + b]) correct++;
            total++;
        }
    }

    free(logits); free(hidden);
    free(b2_f32); free(w2_f32); free(b1_f32); free(w1_f32);

    return (total > 0) ? (float)correct / (float)total : 0.0f;
}

/* ====================================================================
 * Fused f16 training loop: 1 dspqueue message per batch
 * ==================================================================== */

static int train_fused_f16(network_f16_t *net, int epochs,
                            float *train_images, uint8_t *train_labels, int train_count,
                            float *test_images, uint8_t *test_labels, int test_count) {
    int n_batches = train_count / g_batch_size;
    int *indices = (int *)malloc(train_count * sizeof(int));
    uint8_t *label_buf = (uint8_t *)malloc(g_batch_size * sizeof(uint8_t));

    if (!indices || !label_buf) {
        printf("[ERROR] malloc failed\n");
        free(indices); free(label_buf);
        return -1;
    }
    for (int i = 0; i < train_count; i++) indices[i] = i;

    /* Register network buffers with DSP (one-time) */
    dspqueue_register_net_f16(net);

    int fd_input = dspq_find_fd(net->batch_buf);
    if (fd_input < 0) {
        printf("[ERROR] f16 batch_buf not in shared memory\n");
        free(indices); free(label_buf);
        return -1;
    }

    printf("\n");
    printf("==========================================================\n");
    printf(" === MNIST f16 Training (HMX) ===\n");
    printf(" epochs=%d, batch=%d, lr=%.3f\n",
           epochs, g_batch_size, LEARNING_RATE);
    printf(" Network: %d -> %d (ReLU) -> %d (Softmax)\n",
           INPUT_DIM_PAD, HIDDEN_DIM, OUTPUT_DIM);
    printf(" Mode: dspqueue fused f16 (1 round-trip per batch)\n");
    printf("==========================================================\n\n");

    for (int epoch = 0; epoch < epochs; epoch++) {
        struct timespec t_start, t_end;
        float epoch_loss = 0.0f;
        int epoch_correct = 0, epoch_total = 0;

        shuffle_indices(indices, train_count);
        clock_gettime(CLOCK_MONOTONIC, &t_start);

        for (int bi = 0; bi < n_batches; bi++) {
            /* Assemble batch: convert f32 images to f16 */
            for (int s = 0; s < g_batch_size; s++) {
                int idx = indices[bi * g_batch_size + s];
                const float *img = &train_images[(size_t)idx * INPUT_DIM_PAD];
                _Float16 *dst = &net->batch_buf[s * INPUT_DIM_PAD];
                for (int j = 0; j < 784; j++)
                    dst[j] = (_Float16)img[j];
                for (int j = 784; j < INPUT_DIM_PAD; j++)
                    dst[j] = (_Float16)0;
                label_buf[s] = train_labels[idx];
            }

            /* Build request */
            struct train_batch_req req;
            memset(&req, 0, sizeof(req));
            req.op = OP_TRAIN_BATCH;
            req.batch_size = g_batch_size;
            req.learning_rate = LEARNING_RATE;
            memcpy(req.labels, label_buf, g_batch_size);

            /* Send with input buffer (FLUSH_SENDER to push batch data to DSP) */
            struct dspqueue_buffer bufs[1];
            memset(bufs, 0, sizeof(bufs));
            bufs[0].fd = fd_input;
            bufs[0].flags = DSPQUEUE_BUFFER_FLAG_REF
                          | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
                          | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;

            g_dspq.ops_done = 0;
            g_dspq.ops_total = 1;
            int err = dspqueue_write(g_dspq.queue, 0, 1, bufs,
                                     sizeof(req), (const uint8_t *)&req, 1000000);
            if (err != 0) {
                fprintf(stderr, "[ERROR] f16 train_batch write failed: 0x%08x\n", (unsigned)err);
                break;
            }
            sem_wait(&g_dspq.done_sem);

            epoch_loss += g_dspq.last_loss;
            epoch_correct += g_dspq.last_correct;
            epoch_total += g_batch_size;
        }

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double epoch_ms = time_ms(&t_start, &t_end);

        /* Sync weights back from DSP for evaluation */
        dspqueue_sync_f16(net);

        float test_acc = evaluate_f16(net, test_images, test_labels, test_count);

        printf("Epoch %d/%d: loss=%.4f  train_acc=%.4f  test_acc=%.4f  "
               "time=%.1fms\n",
               epoch + 1, epochs,
               epoch_loss / (float)n_batches,
               (float)epoch_correct / (float)epoch_total,
               test_acc, epoch_ms);
    }

    free(label_buf);
    free(indices);
    return 0;
}

/* ====================================================================
 * main
 * ==================================================================== */

int main(int argc, char *argv[]) {
    int ret = 0;
    int epochs = DEFAULT_EPOCHS;

    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    /* Parse command line: [epochs] [batch_size] */
    if (argc >= 2) {
        epochs = atoi(argv[1]);
        if (epochs <= 0) { printf("[ERROR] Invalid epochs: %s\n", argv[1]); return 1; }
    }
    if (argc >= 3) {
        g_batch_size = atoi(argv[2]);
        if (g_batch_size <= 0 || g_batch_size > 1024) {
            printf("[ERROR] Invalid batch size: %s\n", argv[2]); return 1;
        }
    }

    printf("=== MNIST f16 Training (HMX) ===\n");
    printf("[MNIST] Epochs: %d, Batch: %d\n", epochs, g_batch_size);

    /* Initialize dspqueue (sets g_alloc_fn/g_free_fn to shared memory) */
    ret = dspqueue_init(CDSP_DOMAIN_ID);
    if (ret != 0) {
        printf("[ERROR] dspqueue init failed\n");
        return 1;
    }

    /* Load MNIST data (f32 from disk, converted to f16 per batch) */
    int train_count = 0, test_count = 0;
    int train_label_count = 0, test_label_count = 0;

    float *train_images = load_mnist_images(MNIST_TRAIN_IMAGES, &train_count);
    uint8_t *train_labels = load_mnist_labels(MNIST_TRAIN_LABELS, &train_label_count);
    float *test_images = load_mnist_images(MNIST_TEST_IMAGES, &test_count);
    uint8_t *test_labels = load_mnist_labels(MNIST_TEST_LABELS, &test_label_count);

    if (!train_images || !train_labels || !test_images || !test_labels) {
        printf("[ERROR] Failed to load MNIST data\n");
        ret = -1;
        goto cleanup;
    }
    if (train_count != train_label_count || test_count != test_label_count) {
        printf("[ERROR] Image/label count mismatch\n");
        ret = -1;
        goto cleanup;
    }
    printf("[DATA] Training: %d images, Test: %d images\n", train_count, test_count);

    /* Initialize f16 network (buffers allocated in shared memory via g_alloc_fn) */
    network_f16_t net;
    ret = network_f16_init(&net);
    if (ret != 0) goto cleanup;

    /* Train with fused DSP dispatch */
    ret = train_fused_f16(&net, epochs,
                           train_images, train_labels, train_count,
                           test_images, test_labels, test_count);

    network_f16_free(&net);

cleanup:
    free(test_labels);
    free(test_images);
    free(train_labels);
    free(train_images);

    dspqueue_cleanup();

    printf("[MNIST] Done (ret=%d)\n", ret);
    return ret;
}
