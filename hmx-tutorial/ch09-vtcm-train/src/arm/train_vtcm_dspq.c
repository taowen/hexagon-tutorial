/*
 * train_vtcm_dspq.c -- MNIST training with VTCM-resident weights (dspqueue)
 *
 * Based on ch08's train_fused.c, modified to evaluate on DSP side to prevent
 * VTCM corruption during DSP idle time. See ch10 README for analysis.
 *
 * Key difference from ch08: evaluation uses OP_EVAL (forward pass on DSP using
 * VTCM weights) instead of OP_SYNC + ARM cpu_evaluate. This keeps the DSP busy
 * and prevents other DSP clients from overwriting VTCM.
 *
 * ARM sends f16 input directly to DSP (converted from f32 on ARM side).
 */

#include "common/common.h"
#include "arm/data.h"
#include "arm/network.h"
#include "arm/dspqueue_mgr.h"

#ifndef CDSP_DOMAIN_ID
#define CDSP_DOMAIN_ID 3
#endif

/* Shared memory f16 buffer for evaluation input batches */
static _Float16 *g_eval_buf_f16 = NULL;
static int        g_eval_fd      = -1;

/*
 * Convert f32 array to f16 (ARM-side, scalar)
 */
static void arm_f32_to_f16(_Float16 *dst, const float *src, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = (_Float16)src[i];
}

/*
 * dspqueue_evaluate -- evaluate test set on DSP via OP_EVAL
 *
 * Sends test images in batches to DSP as f16. The DSP performs forward pass
 * using VTCM-resident weights and returns the number of correct predictions.
 * This keeps the DSP busy, preventing VTCM corruption from idle eviction.
 */
static float dspqueue_evaluate(const float *images, const uint8_t *labels, int count) {
    int bs = g_batch_size;
    int n_batches = count / bs;
    int total_correct = 0;

    for (int bi = 0; bi < n_batches; bi++) {
        const float *batch_in = images + (size_t)bi * bs * INPUT_DIM_PAD;
        const uint8_t *batch_lab = labels + bi * bs;

        /* Convert f32 input to f16 in shared memory buffer */
        arm_f32_to_f16(g_eval_buf_f16, batch_in, bs * INPUT_DIM_PAD);

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
 * Training: OP_TRAIN_BATCH with f16 input
 * Evaluation: OP_EVAL (DSP forward pass, no weight sync needed)
 */
static int train_fused(network_t *net, int epochs,
                       float *train_images, uint8_t *train_labels, int train_count,
                       float *test_images, uint8_t *test_labels, int test_count) {
    int n_batches = train_count / g_batch_size;
    int *indices = (int *)malloc(train_count * sizeof(int));

    /* f32 staging buffer in regular memory (fast ARM writes) */
    float *batch_buf_f32 = (float *)malloc(g_batch_size * INPUT_DIM_PAD * sizeof(float));

    /* f16 shared memory buffer sent to DSP */
    _Float16 *batch_buf_f16 = (_Float16 *)net_alloc(
        g_batch_size * INPUT_DIM_PAD * sizeof(_Float16));

    uint8_t *label_buf = (uint8_t *)malloc(g_batch_size * sizeof(uint8_t));

    if (!indices || !batch_buf_f32 || !batch_buf_f16 || !label_buf) {
        printf("[ERROR] malloc failed\n");
        free(indices); free(batch_buf_f32); net_free(batch_buf_f16); free(label_buf);
        return -1;
    }
    for (int i = 0; i < train_count; i++) indices[i] = i;

    /* Register network buffers with DSP (one-time) */
    dspqueue_register_net(net);

    int fd_input = dspq_find_fd(batch_buf_f16);
    if (fd_input < 0) {
        printf("[ERROR] batch_buf_f16 not in shared memory\n");
        free(indices); free(batch_buf_f32); net_free(batch_buf_f16); free(label_buf);
        return -1;
    }

    /* Allocate f16 eval buffer in shared memory */
    g_eval_buf_f16 = (_Float16 *)net_alloc(
        g_batch_size * INPUT_DIM_PAD * sizeof(_Float16));
    g_eval_fd = dspq_find_fd(g_eval_buf_f16);
    if (!g_eval_buf_f16 || g_eval_fd < 0) {
        printf("[ERROR] eval buffer allocation failed\n");
        free(indices); free(batch_buf_f32); net_free(batch_buf_f16); free(label_buf);
        return -1;
    }

    printf("\n");
    printf("==========================================================\n");
    printf(" MNIST Training (VTCM dspqueue, HVX f16): epochs=%d, batch=%d, lr=%.3f\n",
           epochs, g_batch_size, LEARNING_RATE);
    printf(" Network: %d -> %d (ReLU) -> %d (Softmax)\n",
           INPUT_DIM_PAD, HIDDEN_DIM, OUTPUT_DIM);
    printf(" Mode: dspqueue fused + DSP-side evaluation (f16 input)\n");
    printf("==========================================================\n\n");

    for (int epoch = 0; epoch < epochs; epoch++) {
        struct timespec t_start, t_end;
        float epoch_loss = 0.0f;
        int epoch_correct = 0, epoch_total = 0;

        shuffle_indices(indices, train_count);
        clock_gettime(CLOCK_MONOTONIC, &t_start);

        for (int bi = 0; bi < n_batches; bi++) {
            /* Assemble batch into fast local f32 buffer */
            for (int b = 0; b < g_batch_size; b++) {
                int idx = indices[bi * g_batch_size + b];
                memcpy(batch_buf_f32 + b * INPUT_DIM_PAD,
                       train_images + (size_t)idx * INPUT_DIM_PAD,
                       INPUT_DIM_PAD * sizeof(float));
                label_buf[b] = train_labels[idx];
            }

            /* Convert f32 to f16 in shared memory */
            arm_f32_to_f16(batch_buf_f16, batch_buf_f32,
                           g_batch_size * INPUT_DIM_PAD);

            /* Build request */
            struct train_batch_req req;
            memset(&req, 0, sizeof(req));
            req.op = OP_TRAIN_BATCH;
            req.batch_size = g_batch_size;
            req.learning_rate = LEARNING_RATE;
            memcpy(req.labels, label_buf, g_batch_size);

            /* Send with f16 input buffer */
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

    net_free(g_eval_buf_f16);
    g_eval_buf_f16 = NULL;
    g_eval_fd = -1;

    free(label_buf);
    net_free(batch_buf_f16);
    free(batch_buf_f32);
    free(indices);
    return 0;
}

/*
 * train_all -- Train all epochs with per-batch data transfer, eval only at end.
 *
 * Reuses the same small rpcmem buffers as per-batch mode (LLC-friendly ~208KB),
 * but skips evaluation and printing between intermediate epochs. After all
 * epochs complete, does one final eval pass and prints a summary.
 */
static int train_all(network_t *net, int epochs,
                     float *train_images, uint8_t *train_labels, int train_count,
                     float *test_images, uint8_t *test_labels, int test_count)
{
    int n_batches = train_count / g_batch_size;
    int *indices = (int *)malloc(train_count * sizeof(int));

    /* f32 staging buffer in regular memory (fast ARM writes) */
    float *batch_buf_f32 = (float *)malloc(g_batch_size * INPUT_DIM_PAD * sizeof(float));

    /* f16 shared memory buffer sent to DSP */
    _Float16 *batch_buf_f16 = (_Float16 *)net_alloc(
        g_batch_size * INPUT_DIM_PAD * sizeof(_Float16));

    uint8_t *label_buf = (uint8_t *)malloc(g_batch_size * sizeof(uint8_t));

    if (!indices || !batch_buf_f32 || !batch_buf_f16 || !label_buf) {
        printf("[ERROR] malloc failed\n");
        free(indices); free(batch_buf_f32); net_free(batch_buf_f16); free(label_buf);
        return -1;
    }
    for (int i = 0; i < train_count; i++) indices[i] = i;

    /* Register network buffers with DSP (one-time) */
    dspqueue_register_net(net);

    int fd_input = dspq_find_fd(batch_buf_f16);
    if (fd_input < 0) {
        printf("[ERROR] batch_buf_f16 not in shared memory\n");
        free(indices); free(batch_buf_f32); net_free(batch_buf_f16); free(label_buf);
        return -1;
    }

    /* Allocate f16 eval buffer in shared memory */
    g_eval_buf_f16 = (_Float16 *)net_alloc(
        g_batch_size * INPUT_DIM_PAD * sizeof(_Float16));
    g_eval_fd = dspq_find_fd(g_eval_buf_f16);
    if (!g_eval_buf_f16 || g_eval_fd < 0) {
        printf("[ERROR] eval buffer allocation failed\n");
        free(indices); free(batch_buf_f32); net_free(batch_buf_f16); free(label_buf);
        return -1;
    }

    /* Per-epoch accumulators */
    float epoch_losses[MAX_EPOCHS];
    float epoch_train_acc[MAX_EPOCHS];
    if (epochs > MAX_EPOCHS) epochs = MAX_EPOCHS;

    printf("\n");
    printf("==========================================================\n");
    printf(" MNIST Training (train-all, per-batch transfer, eval at end)\n");
    printf(" epochs=%d, batch=%d, lr=%.3f\n", epochs, g_batch_size, LEARNING_RATE);
    printf(" Network: %d -> %d (ReLU) -> %d (Softmax)\n",
           INPUT_DIM_PAD, HIDDEN_DIM, OUTPUT_DIM);
    printf(" Train: %d samples, Test: %d samples\n", train_count, test_count);
    printf("==========================================================\n\n");

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        int total_correct = 0;

        shuffle_indices(indices, train_count);

        for (int bi = 0; bi < n_batches; bi++) {
            /* Assemble batch into fast local f32 buffer */
            for (int b = 0; b < g_batch_size; b++) {
                int idx = indices[bi * g_batch_size + b];
                memcpy(batch_buf_f32 + b * INPUT_DIM_PAD,
                       train_images + (size_t)idx * INPUT_DIM_PAD,
                       INPUT_DIM_PAD * sizeof(float));
                label_buf[b] = train_labels[idx];
            }

            /* Convert f32 to f16 in shared memory */
            arm_f32_to_f16(batch_buf_f16, batch_buf_f32,
                           g_batch_size * INPUT_DIM_PAD);

            /* Build request */
            struct train_batch_req req;
            memset(&req, 0, sizeof(req));
            req.op = OP_TRAIN_BATCH;
            req.batch_size = g_batch_size;
            req.learning_rate = LEARNING_RATE;
            memcpy(req.labels, label_buf, g_batch_size);

            /* Send with f16 input buffer */
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

            total_loss += g_dspq.last_loss;
            total_correct += g_dspq.last_correct;
        }

        epoch_losses[epoch] = total_loss / (float)n_batches;
        epoch_train_acc[epoch] = (float)total_correct / (float)(n_batches * g_batch_size);
        /* No eval, no print between epochs */
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double total_ms = time_ms(&t_start, &t_end);

    /* One final evaluation pass */
    float test_acc = dspqueue_evaluate(test_images, test_labels, test_count);

    /* Print summary of all epochs */
    printf("\n");
    for (int e = 0; e < epochs; e++) {
        printf("Epoch %d/%d: loss=%.4f  train_acc=%.4f\n",
               e + 1, epochs, epoch_losses[e], epoch_train_acc[e]);
    }
    printf("\nFinal test_acc=%.4f\n", test_acc);
    printf("\n[TRAIN_ALL] Total training time: %.1f ms (%.1f ms/epoch)\n",
           total_ms, total_ms / epochs);

    net_free(g_eval_buf_f16);
    g_eval_buf_f16 = NULL;
    g_eval_fd = -1;

    free(label_buf);
    net_free(batch_buf_f16);
    free(batch_buf_f32);
    free(indices);
    return 0;
}


int main(int argc, char *argv[]) {
    int ret = 0;
    int epochs = DEFAULT_EPOCHS;
    int use_train_all = 0;

    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    /* Parse command line: [--train-all] [epochs] [batch_size] */
    int argi = 1;
    if (argi < argc && strcmp(argv[argi], "--train-all") == 0) {
        use_train_all = 1;
        argi++;
    }
    if (argi < argc) {
        epochs = atoi(argv[argi]);
        if (epochs <= 0) { printf("[ERROR] Invalid epochs: %s\n", argv[argi]); return 1; }
        argi++;
    }
    if (argi < argc) {
        g_batch_size = atoi(argv[argi]);
        if (g_batch_size <= 0 || g_batch_size > 256) {
            printf("[ERROR] Invalid batch size (max 256): %s\n", argv[argi]); return 1;
        }
        argi++;
    }

    printf("[MNIST] VTCM Training with HVX f16 matmul (dspqueue)\n");
    printf("[MNIST] Epochs: %d, Batch: %d, Mode: %s\n",
           epochs, g_batch_size, use_train_all ? "train-all" : "per-batch");

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

    /* Train: either single-message OP_TRAIN_ALL or per-batch OP_TRAIN_BATCH */
    if (use_train_all) {
        ret = train_all(&net, epochs,
                        train_images, train_labels, train_count,
                        test_images, test_labels, test_count);
    } else {
        ret = train_fused(&net, epochs,
                          train_images, train_labels, train_count,
                          test_images, test_labels, test_count);
    }

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
