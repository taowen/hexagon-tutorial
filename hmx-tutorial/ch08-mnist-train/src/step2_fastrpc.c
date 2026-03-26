/*
 * step2_fastrpc.c -- MNIST training: FastRPC matmul offload (5 calls/batch)
 *
 * Each matrix multiplication is offloaded to the Hexagon DSP via FastRPC.
 * The training loop (forward + backward) makes 5 matmul calls per batch,
 * each requiring a kernel-mode transition (ARM -> kernel -> DSP -> kernel -> ARM).
 *
 * This shows the overhead of FastRPC per-op dispatch. Compare with:
 *   step1_cpu.c      -- pure CPU baseline (no DSP overhead)
 *   step3_dspqueue.c -- same 5 calls/batch but via dspqueue (userspace only)
 *   step4_fused.c    -- 1 call/batch with entire training fused on DSP
 *
 * Network: 784->128(ReLU)->10 (Softmax+CrossEntropy, SGD)
 * Padded:  800->128->32
 *
 * Usage: ./step2_fastrpc [epochs] [batch_size]
 *   Default: 5 epochs, 32 batch size
 */

#include "mnist_common.h"
#include "mnist_data.h"
#include "mnist_cpu_matmul.h"
#include "mnist_network.h"
#include "fastrpc_mgr.h"

#ifndef CDSP_DOMAIN_ID
#define CDSP_DOMAIN_ID 3
#endif

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

    printf("[MNIST] Step 2: FastRPC Matmul Offload (5 calls/batch)\n");
    printf("[MNIST] Epochs: %d, Batch: %d\n", epochs, g_batch_size);

    /* Initialize FastRPC */
    ret = fastrpc_init(CDSP_DOMAIN_ID);
    if (ret != 0) {
        printf("[ERROR] FastRPC init failed\n");
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

    /* Initialize network */
    network_t net;
    ret = network_init(&net);
    if (ret != 0) goto cleanup;

    /* Train with FastRPC matmul dispatch */
    ret = train(&net, epochs, fastrpc_matmul_dispatch, "fastrpc",
                train_images, train_labels, train_count,
                test_images, test_labels, test_count);

    network_free(&net);

cleanup:
    free(test_labels);
    free(test_images);
    free(train_labels);
    free(train_images);

    fastrpc_cleanup();

    printf("[MNIST] Done (ret=%d)\n", ret);
    return ret;
}
