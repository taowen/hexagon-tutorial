/*
 * train_vtcm_dspq.c -- MNIST training with VTCM-resident weights (dspqueue)
 *
 * Based on ch08's train_fused.c, modified to evaluate on DSP side to prevent
 * VTCM corruption during DSP idle time. See ch10 README for analysis.
 *
 * Key difference from ch08: evaluation uses OP_EVAL (forward pass on DSP using
 * VTCM weights) instead of OP_SYNC + ARM cpu_evaluate. This keeps the DSP busy
 * and prevents other DSP clients from overwriting VTCM.
 */

#include "common/common.h"
#include "arm/data.h"
#include "arm/network.h"
#include "arm/dspqueue_mgr.h"

#ifndef CDSP_DOMAIN_ID
#define CDSP_DOMAIN_ID 3
#endif

/* OP_EVAL: forward-only evaluation on DSP using VTCM weights */
#define OP_EVAL 5

/* Shared memory buffer for evaluation input batches */
static float *g_eval_buf = NULL;
static int    g_eval_fd  = -1;

/*
 * dspqueue_evaluate -- evaluate test set on DSP via OP_EVAL
 *
 * Sends test images in batches to DSP. The DSP performs forward pass using
 * VTCM-resident weights and returns the number of correct predictions.
 * This keeps the DSP busy, preventing VTCM corruption from idle eviction.
 */
static float dspqueue_evaluate(const float *images, const uint8_t *labels, int count) {
    int bs = g_batch_size;
    int n_batches = count / bs;
    int total_correct = 0;

    for (int bi = 0; bi < n_batches; bi++) {
        const float *batch_in = images + (size_t)bi * bs * INPUT_DIM_PAD;
        const uint8_t *batch_lab = labels + bi * bs;

        /* Copy input to shared memory buffer */
        memcpy(g_eval_buf, batch_in, bs * INPUT_DIM_PAD * sizeof(float));

        /* Send OP_EVAL */
        struct train_batch_req req;
        memset(&req, 0, sizeof(req));
        req.op = OP_EVAL;
        req.batch_size = bs;
        req.learning_rate = 0;
        memcpy(req.labels, batch_lab, bs);

        struct dspqueue_buffer bufs[1];
        memset(bufs, 0, sizeof(bufs));
        bufs[0].fd = g_eval_fd;
        bufs[0].flags = DSPQUEUE_BUFFER_FLAG_REF
                      | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
                      | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;

        g_dspq.ops_done = 0;
        g_dspq.ops_total = 1;
        int err = dspqueue_write(g_dspq.queue, 0, 1, bufs,
                                  sizeof(req), (const uint8_t *)&req, 1000000);
        if (err != 0) {
            fprintf(stderr, "[ERROR] eval write failed: 0x%08x\n", (unsigned)err);
            return -1.0f;
        }
        sem_wait(&g_dspq.done_sem);
        total_correct += g_dspq.last_correct;
    }

    return (float)total_correct / (float)(n_batches * bs);
}

/*
 * Fused training loop with DSP-side evaluation
 *
 * Training: OP_TRAIN_BATCH (same as ch08)
 * Evaluation: OP_EVAL (DSP forward pass, no weight sync needed)
 */
static int train_fused(network_t *net, int epochs,
                       float *train_images, uint8_t *train_labels, int train_count,
                       float *test_images, uint8_t *test_labels, int test_count) {
    int n_batches = train_count / g_batch_size;
    int *indices = (int *)malloc(train_count * sizeof(int));
    float *batch_buf = (float *)net_alloc(g_batch_size * INPUT_DIM_PAD * sizeof(float));
    uint8_t *label_buf = (uint8_t *)malloc(g_batch_size * sizeof(uint8_t));

    if (!indices || !batch_buf || !label_buf) {
        printf("[ERROR] malloc failed\n");
        free(indices); net_free(batch_buf); free(label_buf);
        return -1;
    }
    for (int i = 0; i < train_count; i++) indices[i] = i;

    /* Register network buffers with DSP (one-time) */
    dspqueue_register_net(net);

    int fd_input = dspq_find_fd(batch_buf);
    if (fd_input < 0) {
        printf("[ERROR] batch_buf not in shared memory\n");
        free(indices); net_free(batch_buf); free(label_buf);
        return -1;
    }

    /* Allocate eval buffer in shared memory */
    g_eval_buf = (float *)net_alloc(g_batch_size * INPUT_DIM_PAD * sizeof(float));
    g_eval_fd = dspq_find_fd(g_eval_buf);
    if (!g_eval_buf || g_eval_fd < 0) {
        printf("[ERROR] eval buffer allocation failed\n");
        free(indices); net_free(batch_buf); free(label_buf);
        return -1;
    }

    printf("\n");
    printf("==========================================================\n");
    printf(" MNIST Training (VTCM dspqueue): epochs=%d, batch=%d, lr=%.3f\n",
           epochs, g_batch_size, LEARNING_RATE);
    printf(" Network: %d -> %d (ReLU) -> %d (Softmax)\n",
           INPUT_DIM_PAD, HIDDEN_DIM, OUTPUT_DIM);
    printf(" Mode: dspqueue fused + DSP-side evaluation (no VTCM idle)\n");
    printf("==========================================================\n\n");

    for (int epoch = 0; epoch < epochs; epoch++) {
        struct timespec t_start, t_end;
        float epoch_loss = 0.0f;
        int epoch_correct = 0, epoch_total = 0;

        shuffle_indices(indices, train_count);
        clock_gettime(CLOCK_MONOTONIC, &t_start);

        for (int bi = 0; bi < n_batches; bi++) {
            /* Assemble batch */
            for (int b = 0; b < g_batch_size; b++) {
                int idx = indices[bi * g_batch_size + b];
                memcpy(batch_buf + b * INPUT_DIM_PAD,
                       train_images + (size_t)idx * INPUT_DIM_PAD,
                       INPUT_DIM_PAD * sizeof(float));
                label_buf[b] = train_labels[idx];
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
                fprintf(stderr, "[ERROR] train_batch write failed: 0x%08x\n", (unsigned)err);
                break;
            }
            sem_wait(&g_dspq.done_sem);

            epoch_loss += g_dspq.last_loss;
            epoch_correct += g_dspq.last_correct;
            epoch_total += g_batch_size;
        }

        /* Evaluate on DSP using VTCM weights (keeps DSP busy, prevents VTCM corruption) */
        float test_acc = dspqueue_evaluate(test_images, test_labels, test_count);

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double epoch_ms = time_ms(&t_start, &t_end);

        printf("Epoch %d/%d: loss=%.4f  train_acc=%.4f  test_acc=%.4f  "
               "time=%.1fms\n",
               epoch + 1, epochs,
               epoch_loss / (float)n_batches,
               (float)epoch_correct / (float)epoch_total,
               test_acc, epoch_ms);
    }

    net_free(g_eval_buf);
    g_eval_buf = NULL;
    g_eval_fd = -1;

    free(label_buf);
    net_free(batch_buf);
    free(indices);
    return 0;
}

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
        if (g_batch_size <= 0 || g_batch_size > 256) {
            printf("[ERROR] Invalid batch size (max 256): %s\n", argv[2]); return 1;
        }
    }

    printf("[MNIST] VTCM Training with DSP-side evaluation (dspqueue)\n");
    printf("[MNIST] Epochs: %d, Batch: %d\n", epochs, g_batch_size);

    /* Initialize dspqueue (sets g_alloc_fn/g_free_fn to shared memory) */
    ret = dspqueue_init(CDSP_DOMAIN_ID);
    if (ret != 0) {
        printf("[ERROR] dspqueue init failed\n");
        return 1;
    }

    /* Load MNIST data */
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

    /* Initialize network (buffers allocated in shared memory via g_alloc_fn) */
    network_t net;
    ret = network_init(&net);
    if (ret != 0) goto cleanup;

    /* Train with fused DSP dispatch + DSP-side evaluation */
    ret = train_fused(&net, epochs,
                      train_images, train_labels, train_count,
                      test_images, test_labels, test_count);

    network_free(&net);

cleanup:
    free(test_labels);
    free(test_images);
    free(train_labels);
    free(train_images);

    dspqueue_cleanup();

    printf("[MNIST] Done (ret=%d)\n", ret);
    return ret;
}
