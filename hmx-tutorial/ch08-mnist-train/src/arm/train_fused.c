/*
 * step4_fused.c -- MNIST training: dspqueue fused training (1 call/batch)
 *
 * The entire forward/backward/update loop runs on the DSP in a single
 * dspqueue message per batch. The ARM side only assembles the batch input
 * and sends one OP_TRAIN_BATCH request. This eliminates per-op communication
 * overhead entirely.
 *
 * Setup:
 *   1. OP_REGISTER_NET -- one-time registration of all network buffer fds
 *   2. OP_TRAIN_BATCH  -- one message per batch (forward+backward+SGD on DSP)
 *   3. OP_SYNC         -- flush DSP caches so ARM can read weights for eval
 *
 * Compare with:
 *   step1_cpu.c      -- pure CPU baseline
 *   step2_fastrpc.c  -- 5 FastRPC calls/batch (kernel transitions)
 *   step3_dspqueue.c -- 5 dspqueue calls/batch (userspace, zero copy)
 *   step4_fused.c    -- 1 dspqueue call/batch (this file, full DSP fusion)
 *
 * Network: 784->128(ReLU)->10 (Softmax+CrossEntropy, SGD)
 * Padded:  800->128->32
 *
 * Usage: ./step4_fused [epochs] [batch_size]
 *   Default: 5 epochs, 32 batch size
 */

#include "common/common.h"
#include "arm/data.h"
#include "arm/cpu_matmul.h"
#include "arm/network.h"
#include "arm/dspqueue_mgr.h"

#ifndef CDSP_DOMAIN_ID
#define CDSP_DOMAIN_ID 3
#endif

/*
 * Fused training loop: 1 dspqueue message per batch
 *
 * The DSP receives OP_TRAIN_BATCH with the input batch buffer and labels,
 * then performs forward pass, loss computation, backward pass, and SGD update
 * all in one shot. Only the batch loss and accuracy are returned.
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

    printf("\n");
    printf("==========================================================\n");
    printf(" MNIST Training (fused DSP): epochs=%d, batch=%d, lr=%.3f\n",
           epochs, g_batch_size, LEARNING_RATE);
    printf(" Network: %d -> %d (ReLU) -> %d (Softmax)\n",
           INPUT_DIM_PAD, HIDDEN_DIM, OUTPUT_DIM);
    printf(" Mode: dspqueue fused (1 round-trip per batch)\n");
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

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double epoch_ms = time_ms(&t_start, &t_end);

        /* Sync weights back from DSP for evaluation */
        dspqueue_sync(net);

        float test_acc = evaluate(net, test_images, test_labels, test_count,
                                  cpu_matmul_dispatch);

        printf("Epoch %d/%d: loss=%.4f  train_acc=%.4f  test_acc=%.4f  "
               "time=%.1fms\n",
               epoch + 1, epochs,
               epoch_loss / (float)n_batches,
               (float)epoch_correct / (float)epoch_total,
               test_acc, epoch_ms);
    }

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
        if (g_batch_size <= 0 || g_batch_size > 1024) {
            printf("[ERROR] Invalid batch size: %s\n", argv[2]); return 1;
        }
    }

    printf("[MNIST] Step 4: Fused DSP Training (1 call/batch)\n");
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

    /* Train with fused DSP dispatch */
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
