/*
 * step1_cpu.c -- MNIST training: Pure CPU baseline
 *
 * This is the simplest version. All matrix multiplications run on the ARM CPU.
 * No DSP, no shared memory, no communication overhead -- just straightforward
 * neural network training in plain C.
 *
 * Network: 784->128(ReLU)->10 (Softmax+CrossEntropy, SGD)
 * Padded:  800->128->32
 *
 * Usage: ./step1_cpu [epochs] [batch_size]
 *   Default: 5 epochs, 32 batch size
 *
 * Compare with:
 *   step2_fastrpc.c  -- same math, matmul offloaded to DSP via FastRPC
 *   step3_dspqueue.c -- same math, matmul offloaded to DSP via dspqueue
 *   step4_fused.c    -- entire forward/backward/update fused into 1 DSP call
 */

#include "common/common.h"
#include "arm/data.h"
#include "arm/cpu_matmul.h"
#include "arm/network.h"
#include "arm/synthetic_test.h"

int main(int argc, char *argv[]) {
    int epochs = DEFAULT_EPOCHS;
    int use_test_synthetic = 0;

    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    /* Parse command line: [--test-synthetic] [epochs] [batch_size] */
    int argi = 1;
    while (argi < argc && argv[argi][0] == '-') {
        if (strcmp(argv[argi], "--test-synthetic") == 0) {
            use_test_synthetic = 1;
        } else {
            printf("[ERROR] Unknown flag: %s\n", argv[argi]);
            return 1;
        }
        argi++;
    }
    if (argi < argc) {
        epochs = atoi(argv[argi]);
        if (epochs <= 0) { printf("[ERROR] Invalid epochs: %s\n", argv[argi]); return 1; }
        argi++;
    }
    if (argi < argc) {
        g_batch_size = atoi(argv[argi]);
        if (g_batch_size <= 0 || g_batch_size > 1024) {
            printf("[ERROR] Invalid batch size: %s\n", argv[argi]); return 1;
        }
        argi++;
    }

    printf("[MNIST] Step 1: Pure CPU Training\n");
    printf("[MNIST] Epochs: %d, Batch: %d%s\n", epochs, g_batch_size,
           use_test_synthetic ? ", + synthetic test" : "");

    /* Load MNIST data */
    int train_count = 0, test_count = 0;
    int train_label_count = 0, test_label_count = 0;

    float *train_images = load_mnist_images(MNIST_TRAIN_IMAGES, &train_count);
    uint8_t *train_labels = load_mnist_labels(MNIST_TRAIN_LABELS, &train_label_count);
    float *test_images = load_mnist_images(MNIST_TEST_IMAGES, &test_count);
    uint8_t *test_labels = load_mnist_labels(MNIST_TEST_LABELS, &test_label_count);

    if (!train_images || !train_labels || !test_images || !test_labels) {
        printf("[ERROR] Failed to load MNIST data\n");
        goto cleanup;
    }
    if (train_count != train_label_count || test_count != test_label_count) {
        printf("[ERROR] Image/label count mismatch\n");
        goto cleanup;
    }
    printf("[DATA] Training: %d images, Test: %d images\n", train_count, test_count);

    /* Initialize network */
    network_t net;
    if (network_init(&net) != 0) goto cleanup;

    /* Train */
    train(&net, epochs, cpu_matmul_dispatch, "cpu",
          train_images, train_labels, train_count,
          test_images, test_labels, test_count);

    /* Run synthetic digit test after training */
    if (use_test_synthetic)
        test_synthetic(&net);

    network_free(&net);

cleanup:
    free(test_labels);
    free(test_images);
    free(train_labels);
    free(train_images);

    printf("[MNIST] Done\n");
    return 0;
}
