/*
 * train_vtcm.c -- MNIST training with weights resident in VTCM
 *
 * Pure FastRPC approach: weights live on the DSP in VTCM. The ARM side
 * sends input batches via FastRPC, and the DSP performs the entire
 * forward/backward/SGD loop in VTCM. No dspqueue, no rpcmem, no shared
 * memory -- just plain FastRPC calls.
 *
 * FastRPC calls per batch: 1 (mnist_vtcm_train_batch)
 * FastRPC calls per epoch: n_batches + 1 (sync_weights for eval)
 *
 * Network: 784->128(ReLU)->10 (Softmax+CrossEntropy, SGD)
 * Padded:  800->128->32
 *
 * Usage: ./train_vtcm [epochs] [batch_size]
 *   Default: 5 epochs, 128 batch size
 */

#include "common/common.h"
#include "arm/data.h"
#include "arm/cpu_matmul.h"
#include "arm/network.h"

#include "mnist_vtcm.h"
#include "remote.h"

#ifndef CDSP_DOMAIN_ID
#define CDSP_DOMAIN_ID 3
#endif

int main(int argc, char *argv[]) {
    int ret = 0;
    int epochs = DEFAULT_EPOCHS;
    g_batch_size = 128;  /* default for VTCM mode */

    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    /* Parse command line: [epochs] [batch_size] */
    if (argc >= 2) {
        epochs = atoi(argv[1]);
        if (epochs <= 0) {
            printf("[ERROR] Invalid epochs: %s\n", argv[1]);
            return 1;
        }
    }
    if (argc >= 3) {
        g_batch_size = atoi(argv[2]);
        if (g_batch_size <= 0 || g_batch_size > 1024) {
            printf("[ERROR] Invalid batch size: %s\n", argv[2]);
            return 1;
        }
    }

    printf("[MNIST] VTCM Training (pure FastRPC)\n");
    printf("[MNIST] Epochs: %d, Batch: %d\n", epochs, g_batch_size);

    /* ----------------------------------------------------------------
     * Open FastRPC session
     * ---------------------------------------------------------------- */
    remote_handle64 h = 0;
    char uri[256];
    snprintf(uri, sizeof(uri), "%s&_dom=cdsp", mnist_vtcm_URI);

    /* Allow unsigned PD */
    struct remote_rpc_control_unsigned_module udata;
    udata.domain = CDSP_DOMAIN_ID;
    udata.enable = 1;
    remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&udata, sizeof(udata));

    int err = mnist_vtcm_open(uri, &h);
    if (err != 0) {
        printf("[ERROR] mnist_vtcm_open failed: 0x%08x\n", (unsigned)err);
        return 1;
    }
    printf("[RPC] Session opened\n");

    /* Enable low-latency QoS */
    struct remote_rpc_control_latency ldata;
    ldata.enable = 1;
    remote_handle64_control(h, DSPRPC_CONTROL_LATENCY, (void *)&ldata, sizeof(ldata));

    /* ----------------------------------------------------------------
     * Load MNIST data
     * ---------------------------------------------------------------- */
    int train_count = 0, test_count = 0;
    int train_label_count = 0, test_label_count = 0;

    float *train_images = load_mnist_images(MNIST_TRAIN_IMAGES, &train_count);
    uint8_t *train_labels = load_mnist_labels(MNIST_TRAIN_LABELS, &train_label_count);
    float *test_images = load_mnist_images(MNIST_TEST_IMAGES, &test_count);
    uint8_t *test_labels = load_mnist_labels(MNIST_TEST_LABELS, &test_label_count);

    if (!train_images || !train_labels || !test_images || !test_labels) {
        printf("[ERROR] Failed to load MNIST data\n");
        ret = 1;
        goto cleanup_rpc;
    }
    if (train_count != train_label_count || test_count != test_label_count) {
        printf("[ERROR] Image/label count mismatch\n");
        ret = 1;
        goto cleanup_data;
    }
    printf("[DATA] Training: %d images, Test: %d images\n", train_count, test_count);

    /* ----------------------------------------------------------------
     * Initialize network on ARM side (regular malloc, no shared memory)
     * ---------------------------------------------------------------- */
    network_t net;
    memset(&net, 0, sizeof(net));

    /* Weights and biases -- these get copied to VTCM and synced back */
    net.w1 = (float *)malloc(HIDDEN_DIM * INPUT_DIM_PAD * sizeof(float));
    net.b1 = (float *)calloc(HIDDEN_DIM, sizeof(float));
    net.w2 = (float *)malloc(OUTPUT_DIM_PAD * HIDDEN_DIM * sizeof(float));
    net.b2 = (float *)calloc(OUTPUT_DIM_PAD, sizeof(float));

    /* Gradients -- only used by ARM-side evaluate path (not needed, but
     * network_t has them). Allocate as dummies. */
    net.dw1 = (float *)calloc(HIDDEN_DIM * INPUT_DIM_PAD, sizeof(float));
    net.db1 = (float *)calloc(HIDDEN_DIM, sizeof(float));
    net.dw2 = (float *)calloc(OUTPUT_DIM_PAD * HIDDEN_DIM, sizeof(float));
    net.db2 = (float *)calloc(OUTPUT_DIM_PAD, sizeof(float));

    /* Intermediate buffers -- used by ARM-side forward() in evaluate() */
    net.hidden          = (float *)malloc(g_batch_size * HIDDEN_DIM * sizeof(float));
    net.hidden_pre_relu = (float *)malloc(g_batch_size * HIDDEN_DIM * sizeof(float));
    net.logits          = (float *)malloc(g_batch_size * OUTPUT_DIM_PAD * sizeof(float));
    net.probs           = (float *)malloc(g_batch_size * OUTPUT_DIM_PAD * sizeof(float));
    net.dlogits         = (float *)malloc(g_batch_size * OUTPUT_DIM_PAD * sizeof(float));
    net.dhidden         = (float *)malloc(g_batch_size * HIDDEN_DIM * sizeof(float));

    if (!net.w1 || !net.b1 || !net.w2 || !net.b2 ||
        !net.dw1 || !net.db1 || !net.dw2 || !net.db2 ||
        !net.hidden || !net.hidden_pre_relu || !net.logits ||
        !net.probs || !net.dlogits || !net.dhidden) {
        printf("[ERROR] malloc failed for network\n");
        ret = 1;
        goto cleanup_net;
    }

    /* He initialization */
    he_init(net.w1, HIDDEN_DIM, INPUT_DIM_PAD, INPUT_DIM_PAD);
    he_init(net.w2, OUTPUT_DIM_PAD, HIDDEN_DIM, HIDDEN_DIM);
    /* Zero padding rows of W2 (rows OUTPUT_DIM..OUTPUT_DIM_PAD-1) */
    for (int i = OUTPUT_DIM; i < OUTPUT_DIM_PAD; i++)
        for (int j = 0; j < HIDDEN_DIM; j++)
            net.w2[i * HIDDEN_DIM + j] = 0.0f;

    printf("[NET] Weights initialized (He init)\n");

    /* ----------------------------------------------------------------
     * Copy weights to VTCM on DSP
     * ---------------------------------------------------------------- */
    /* Check VTCM self-test from open() */
    {
        uint64 test_total = 0, d1, d2, d3, d4;
        mnist_vtcm_get_timing(h, &test_total, &d1, &d2, &d3, &d4);
        if (test_total == 0xAAAA) printf("[DEBUG] VTCM self-test: PASS (scalar + memcpy)\n");
        else if (test_total == 0xBBBB) printf("[DEBUG] VTCM self-test: scalar OK, memcpy BROKEN\n");
        else if (test_total == 0xCCCC) printf("[DEBUG] VTCM self-test: scalar access BROKEN\n");
        else printf("[DEBUG] VTCM self-test: result=0x%llx\n", (unsigned long long)test_total);
    }

    err = mnist_vtcm_init_net(h,
        (const uint8 *)net.w1, HIDDEN_DIM * INPUT_DIM_PAD * (int)sizeof(float),
        (const uint8 *)net.b1, HIDDEN_DIM * (int)sizeof(float),
        (const uint8 *)net.w2, OUTPUT_DIM_PAD * HIDDEN_DIM * (int)sizeof(float),
        (const uint8 *)net.b2, OUTPUT_DIM_PAD * (int)sizeof(float));
    if (err != 0) {
        printf("[ERROR] mnist_vtcm_init_net failed: 0x%08x\n", (unsigned)err);
        ret = 1;
        goto cleanup_net;
    }
    /* Check if VTCM is accessible from init_net's thread */
    {
        uint64 d0, fwd, bwd, d3, sgd_flag;
        mnist_vtcm_get_timing(h, &d0, &fwd, &bwd, &d3, &sgd_flag);
        if (sgd_flag == 0xBEEF) printf("[DEBUG] init_net: VTCM copy PASS\n");
        else if (sgd_flag == 0xFADE) {
            uint32_t w1_bits = (uint32_t)fwd, src_bits = (uint32_t)bwd;
            float vtcm_val, src_val;
            memcpy(&vtcm_val, &w1_bits, sizeof(float));
            memcpy(&src_val, &src_bits, sizeof(float));
            printf("[DEBUG] init_net: VTCM copy FAIL (got=%f expected=%f, "
                   "hex: got=0x%08x exp=0x%08x)\n",
                   vtcm_val, src_val, w1_bits, src_bits);
        }
        else printf("[DEBUG] init_net: result=0x%llx\n", (unsigned long long)sgd_flag);
    }
    printf("[RPC] Weights copied to VTCM\n");

    /* Debug: round-trip weight verification */
    {
        float *w1_check = (float *)malloc(HIDDEN_DIM * INPUT_DIM_PAD * sizeof(float));
        float *b1_check = (float *)malloc(HIDDEN_DIM * sizeof(float));
        float *w2_check = (float *)malloc(OUTPUT_DIM_PAD * HIDDEN_DIM * sizeof(float));
        float *b2_check = (float *)malloc(OUTPUT_DIM_PAD * sizeof(float));
        if (w1_check && b1_check && w2_check && b2_check) {
            err = mnist_vtcm_sync_weights(h,
                (uint8 *)w1_check, HIDDEN_DIM * INPUT_DIM_PAD * (int)sizeof(float),
                (uint8 *)b1_check, HIDDEN_DIM * (int)sizeof(float),
                (uint8 *)w2_check, OUTPUT_DIM_PAD * HIDDEN_DIM * (int)sizeof(float),
                (uint8 *)b2_check, OUTPUT_DIM_PAD * (int)sizeof(float));
            if (err == 0) {
                int w1_match = (memcmp(net.w1, w1_check, HIDDEN_DIM * INPUT_DIM_PAD * sizeof(float)) == 0);
                int b1_match = (memcmp(net.b1, b1_check, HIDDEN_DIM * sizeof(float)) == 0);
                int w2_match = (memcmp(net.w2, w2_check, OUTPUT_DIM_PAD * HIDDEN_DIM * sizeof(float)) == 0);
                int b2_match = (memcmp(net.b2, b2_check, OUTPUT_DIM_PAD * sizeof(float)) == 0);
                printf("[DEBUG] Weight round-trip: W1=%s B1=%s W2=%s B2=%s\n",
                       w1_match ? "MATCH" : "MISMATCH",
                       b1_match ? "MATCH" : "MISMATCH",
                       w2_match ? "MATCH" : "MISMATCH",
                       b2_match ? "MATCH" : "MISMATCH");
                if (!w1_match) {
                    printf("[DEBUG] W1[0..7]: sent=");
                    for (int i = 0; i < 8; i++) printf("%.6f ", net.w1[i]);
                    printf("\n[DEBUG] W1[0..7]: recv=");
                    for (int i = 0; i < 8; i++) printf("%.6f ", w1_check[i]);
                    printf("\n");
                }
            } else {
                printf("[DEBUG] sync_weights for verification failed: 0x%08x\n", (unsigned)err);
            }
        }
        free(w1_check); free(b1_check); free(w2_check); free(b2_check);
    }

    /* ----------------------------------------------------------------
     * Training loop
     * ---------------------------------------------------------------- */
    {
        int n_batches = train_count / g_batch_size;
        int *indices = (int *)malloc(train_count * sizeof(int));
        float *batch_buf = (float *)malloc(g_batch_size * INPUT_DIM_PAD * sizeof(float));
        uint8_t *label_buf = (uint8_t *)malloc(g_batch_size * sizeof(uint8_t));

        if (!indices || !batch_buf || !label_buf) {
            printf("[ERROR] malloc failed for training buffers\n");
            free(indices); free(batch_buf); free(label_buf);
            ret = 1;
            goto cleanup_net;
        }
        for (int i = 0; i < train_count; i++) indices[i] = i;

        float lr = LEARNING_RATE;
        uint32_t lr_bits;
        memcpy(&lr_bits, &lr, sizeof(uint32_t));

        printf("\n");
        printf("==========================================================\n");
        printf(" MNIST Training (VTCM): epochs=%d, batch=%d, lr=%.3f\n",
               epochs, g_batch_size, lr);
        printf(" Network: %d -> %d (ReLU) -> %d (Softmax)\n",
               INPUT_DIM_PAD, HIDDEN_DIM, OUTPUT_DIM);
        printf(" Mode: pure FastRPC, weights in VTCM\n");
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

                /* FastRPC: train one batch on DSP */
                uint32_t loss_bits = 0;
                uint32_t correct = 0;
                err = mnist_vtcm_train_batch(h,
                    (const uint8 *)batch_buf,
                    g_batch_size * INPUT_DIM_PAD * (int)sizeof(float),
                    (const uint8 *)label_buf,
                    g_batch_size,
                    (uint32)g_batch_size,
                    lr_bits,
                    &loss_bits,
                    &correct);
                if (err != 0) {
                    printf("[ERROR] train_batch failed at epoch %d batch %d: 0x%08x\n",
                           epoch + 1, bi, (unsigned)err);
                    free(label_buf); free(batch_buf); free(indices);
                    ret = 1;
                    goto cleanup_net;
                }

                float loss;
                memcpy(&loss, &loss_bits, sizeof(float));
                epoch_loss += loss;
                epoch_correct += (int)correct;
                epoch_total += g_batch_size;
            }

            clock_gettime(CLOCK_MONOTONIC, &t_end);
            double epoch_ms = time_ms(&t_start, &t_end);

            /* Sync weights back from VTCM for evaluation */
            err = mnist_vtcm_sync_weights(h,
                (uint8 *)net.w1, HIDDEN_DIM * INPUT_DIM_PAD * (int)sizeof(float),
                (uint8 *)net.b1, HIDDEN_DIM * (int)sizeof(float),
                (uint8 *)net.w2, OUTPUT_DIM_PAD * HIDDEN_DIM * (int)sizeof(float),
                (uint8 *)net.b2, OUTPUT_DIM_PAD * (int)sizeof(float));
            if (err != 0) {
                printf("[ERROR] sync_weights failed: 0x%08x\n", (unsigned)err);
                free(label_buf); free(batch_buf); free(indices);
                ret = 1;
                goto cleanup_net;
            }

            /* Evaluate on test set using CPU matmul */
            float test_acc = evaluate(&net, test_images, test_labels, test_count,
                                      cpu_matmul_dispatch);

            printf("Epoch %d/%d: loss=%.4f  train_acc=%.4f  test_acc=%.4f  "
                   "time=%.1fms\n",
                   epoch + 1, epochs,
                   epoch_loss / (float)n_batches,
                   (float)epoch_correct / (float)epoch_total,
                   test_acc, epoch_ms);
        }

        free(label_buf);
        free(batch_buf);
        free(indices);
    }

    /* ----------------------------------------------------------------
     * Print DSP timing breakdown
     * ---------------------------------------------------------------- */
    {
        uint64 total_us = 0, fwd_mm_us = 0, bwd_mm_us = 0, other_us = 0, sgd_us = 0;
        err = mnist_vtcm_get_timing(h, &total_us, &fwd_mm_us, &bwd_mm_us,
                                    &other_us, &sgd_us);
        if (err == 0) {
            printf("\n");
            printf("DSP timing breakdown:\n");
            printf("  total    = %" PRIu64 " us\n", (uint64_t)total_us);
            printf("  fwd_mm   = %" PRIu64 " us\n", (uint64_t)fwd_mm_us);
            printf("  bwd_mm   = %" PRIu64 " us\n", (uint64_t)bwd_mm_us);
            printf("  other    = %" PRIu64 " us\n", (uint64_t)other_us);
            printf("  sgd      = %" PRIu64 " us\n", (uint64_t)sgd_us);
        } else {
            printf("[WARN] get_timing failed: 0x%08x\n", (unsigned)err);
        }
    }

    /* ----------------------------------------------------------------
     * Cleanup
     * ---------------------------------------------------------------- */
cleanup_net:
    free(net.dhidden);
    free(net.dlogits);
    free(net.probs);
    free(net.logits);
    free(net.hidden_pre_relu);
    free(net.hidden);
    free(net.db2); free(net.db1);
    free(net.dw2); free(net.dw1);
    free(net.b2);  free(net.b1);
    free(net.w2);  free(net.w1);

cleanup_data:
    free(test_labels);
    free(test_images);
    free(train_labels);
    free(train_images);

cleanup_rpc:
    mnist_vtcm_close(h);
    printf("[MNIST] Done (ret=%d)\n", ret);
    return ret;
}
