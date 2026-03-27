/*
 * train_vtcm_dspq.c -- MNIST training with VTCM-resident weights (dspqueue)
 *
 * Based on ch08's train_fused.c, modified to evaluate on DSP side to prevent
 * VTCM corruption during DSP idle time. See ch04 README (Part 7: VTCM 抢占) for analysis.
 *
 * Key difference from ch08: evaluation uses OP_EVAL (forward pass on DSP using
 * VTCM weights) instead of OP_SYNC + ARM cpu_evaluate. This keeps the DSP busy
 * and prevents other DSP clients from overwriting VTCM.
 *
 * ARM sends f16 input directly to DSP (converted from f32 on ARM side).
 *
 * --test-synthetic: After training, generate synthetic digit images 0-9,
 * run inference on DSP, and display ASCII art + predictions.
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


/* ====================================================================
 * Synthetic digit generation for --test-synthetic
 *
 * Each digit is a 28x28 float image in [0,1], white-on-black like MNIST.
 * Uses simple line/circle drawing primitives.
 * ==================================================================== */

/* Set a pixel (with anti-aliased thickness) */
static void set_pixel(float *img, int x, int y, float val) {
    if (x >= 0 && x < 28 && y >= 0 && y < 28)
        img[y * 28 + x] = fmaxf(img[y * 28 + x], val);
}

/* Draw a thick line from (x0,y0) to (x1,y1) using Bresenham with thickness */
static void draw_line(float *img, int x0, int y0, int x1, int y1, int thickness) {
    int dx = abs(x1 - x0), dy = abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;
    int half = thickness / 2;
    while (1) {
        for (int tx = -half; tx <= half; tx++)
            for (int ty = -half; ty <= half; ty++)
                set_pixel(img, x0 + tx, y0 + ty, 1.0f);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 <  dx) { err += dx; y0 += sy; }
    }
}

/* Draw a circle outline */
static void draw_circle(float *img, int cx, int cy, int r, int thickness) {
    int half = thickness / 2;
    for (int angle = 0; angle < 360; angle++) {
        float rad = (float)angle * (float)M_PI / 180.0f;
        int x = cx + (int)(r * cosf(rad) + 0.5f);
        int y = cy + (int)(r * sinf(rad) + 0.5f);
        for (int tx = -half; tx <= half; tx++)
            for (int ty = -half; ty <= half; ty++)
                set_pixel(img, x + tx, y + ty, 1.0f);
    }
}

/* Draw an arc from angle_start to angle_end (degrees) */
static void draw_arc(float *img, int cx, int cy, int r,
                     int angle_start, int angle_end, int thickness) {
    int half = thickness / 2;
    for (int angle = angle_start; angle <= angle_end; angle++) {
        float rad = (float)angle * (float)M_PI / 180.0f;
        int x = cx + (int)(r * cosf(rad) + 0.5f);
        int y = cy + (int)(r * sinf(rad) + 0.5f);
        for (int tx = -half; tx <= half; tx++)
            for (int ty = -half; ty <= half; ty++)
                set_pixel(img, x + tx, y + ty, 1.0f);
    }
}

/*
 * Apply a 3x3 box blur to a 28x28 image to simulate MNIST anti-aliasing.
 * MNIST images have soft, gray edges -- not hard binary edges.
 */
static void blur_image(float *img) {
    float tmp[28 * 28];
    memcpy(tmp, img, 28 * 28 * sizeof(float));
    for (int y = 1; y < 27; y++) {
        for (int x = 1; x < 27; x++) {
            float sum = 0.0f;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                    sum += tmp[(y + dy) * 28 + (x + dx)];
            img[y * 28 + x] = sum / 9.0f;
        }
    }
}

/* Generate synthetic digit images.
 * imgs: array of 10 images, each 28*28 floats, pre-zeroed.
 *
 * Digits are drawn in MNIST style: thick strokes (~2-3px), centered in
 * a ~20x20 area within the 28x28 frame, then blurred for soft edges.
 */
static void generate_synthetic_digits(float imgs[10][28 * 28]) {
    memset(imgs, 0, 10 * 28 * 28 * sizeof(float));
    int t = 3;  /* stroke thickness -- MNIST strokes are thick */

    /* Digit 0: oval */
    draw_circle(imgs[0], 14, 14, 9, t);

    /* Digit 1: simple thick vertical line, slightly right of center.
     * MNIST 1s are just vertical strokes -- no serifs, no base. */
    draw_line(imgs[1], 15, 5, 15, 23, t);

    /* Digit 2: flatter top bump + steep diagonal + wide base.
     * Use a smaller radius, wider arc for a less circular top. */
    draw_arc(imgs[2], 14, 10, 6, 200, 360, t);
    draw_line(imgs[2], 20, 10, 8, 23, t);    /* steeper diagonal */
    draw_line(imgs[2], 7, 23, 22, 23, t);    /* wide base */

    /* Digit 3: two arcs open to the left */
    draw_arc(imgs[3], 13, 10, 7, 270, 450, t);
    draw_arc(imgs[3], 13, 18, 7, 270, 450, t);

    /* Digit 4: inverted-y shape. Diagonal from top-left to middle-right,
     * horizontal bar at ~60% height, vertical line through intersection. */
    draw_line(imgs[4], 6, 5, 18, 17, t);     /* diagonal stroke down-right */
    draw_line(imgs[4], 6, 17, 22, 17, t);    /* horizontal bar at 60% */
    draw_line(imgs[4], 18, 5, 18, 24, t);    /* vertical stroke */

    /* Digit 5: top horizontal + left vertical + middle horizontal + bottom arc */
    draw_line(imgs[5], 8, 5, 20, 5, t);
    draw_line(imgs[5], 8, 5, 8, 13, t);
    draw_line(imgs[5], 8, 13, 16, 13, t);
    draw_arc(imgs[5], 14, 18, 6, 270, 450, t);

    /* Digit 6: curved stroke from upper-right down to lower-left,
     * then a clear bottom circle. The top is a sweeping curve, not straight. */
    draw_arc(imgs[6], 20, 6, 10, 120, 200, t);  /* curved top stroke */
    draw_circle(imgs[6], 14, 18, 6, t);          /* bottom circle */
    draw_line(imgs[6], 10, 10, 9, 13, t);        /* connect curve to circle */

    /* Digit 7: short top stroke + mostly-vertical diagonal.
     * MNIST 7s are fairly upright, not a wide diagonal. */
    draw_line(imgs[7], 9, 5, 20, 5, t);      /* shorter top bar */
    draw_line(imgs[7], 20, 5, 14, 24, t);    /* more vertical diagonal */

    /* Digit 8: two circles stacked */
    draw_circle(imgs[8], 14, 10, 5, t);
    draw_circle(imgs[8], 14, 19, 6, t);

    /* Digit 9: top circle + right vertical down */
    draw_circle(imgs[9], 14, 10, 6, t);
    draw_line(imgs[9], 20, 10, 16, 24, t);

    /* Apply Gaussian-like blur to all digits for MNIST-style soft edges */
    for (int d = 0; d < 10; d++) {
        blur_image(imgs[d]);
    }
}

/* Print a 28x28 image as ASCII art */
static void print_ascii_art(const float *img, int digit) {
    printf("  Synthetic digit %d:\n", digit);
    printf("  +----------------------------+\n");
    for (int y = 0; y < 28; y++) {
        printf("  |");
        for (int x = 0; x < 28; x++) {
            float v = img[y * 28 + x];
            if (v > 0.7f)      printf("#");
            else if (v > 0.3f) printf(".");
            else               printf(" ");
        }
        printf("|\n");
    }
    printf("  +----------------------------+\n");
}

/*
 * Run inference for a single image on DSP via OP_EVAL.
 * Returns the predicted digit by trying all 10 possible labels.
 */
static int infer_single_image(const float *img_f32_padded) {
    /* Convert to f16 in the eval shared buffer */
    arm_f32_to_f16(g_eval_buf_f16, img_f32_padded, INPUT_DIM_PAD);

    /* Try each label 0-9; the one that returns correct=1 is the prediction */
    for (int label = 0; label < 10; label++) {
        struct train_batch_req req;
        memset(&req, 0, sizeof(req));
        req.op = OP_EVAL;
        req.batch_size = 1;
        req.learning_rate = 0;
        req.labels[0] = (uint8_t)label;

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
            fprintf(stderr, "[ERROR] infer write failed: 0x%08x\n", (unsigned)err);
            return -1;
        }
        sem_wait(&g_dspq.done_sem);

        if (g_dspq.last_correct == 1) {
            return label;
        }
    }
    return -1;  /* should not happen */
}

/*
 * test_synthetic -- Generate synthetic digits, run inference, display results
 *
 * Must be called after training (weights loaded on DSP) and while
 * g_eval_buf_f16 / g_eval_fd are still valid.
 */
static void test_synthetic(void) {
    printf("\n");
    printf("==========================================================\n");
    printf(" Synthetic Digit Recognition Test\n");
    printf("==========================================================\n\n");

    float synthetic[10][28 * 28];
    generate_synthetic_digits(synthetic);

    /* Padded input buffer (reuse for each image) */
    float padded[INPUT_DIM_PAD];

    int num_correct = 0;

    printf("%-8s %-12s %-10s\n", "Digit", "Predicted", "Result");
    printf("------   ---------    ------\n");

    for (int d = 0; d < 10; d++) {
        /* Print ASCII art */
        print_ascii_art(synthetic[d], d);

        /* Pad 784 -> INPUT_DIM_PAD (832) with zeros */
        memset(padded, 0, sizeof(padded));
        memcpy(padded, synthetic[d], 784 * sizeof(float));

        /* Run inference */
        int predicted = infer_single_image(padded);

        int correct = (predicted == d);
        if (correct) num_correct++;

        printf("  %-8d %-12d %s\n\n", d, predicted,
               correct ? "CORRECT" : "WRONG");
    }

    printf("==========================================================\n");
    printf(" Summary: %d / 10 correct (%.0f%%)\n", num_correct, num_correct * 10.0f);
    printf("==========================================================\n\n");
}

int main(int argc, char *argv[]) {
    int ret = 0;
    int epochs = DEFAULT_EPOCHS;
    int use_train_all = 0;
    int use_test_synthetic = 0;

    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    /* Parse command line: [--train-all] [--test-synthetic] [epochs] [batch_size] */
    int argi = 1;
    while (argi < argc && argv[argi][0] == '-') {
        if (strcmp(argv[argi], "--train-all") == 0) {
            use_train_all = 1;
        } else if (strcmp(argv[argi], "--test-synthetic") == 0) {
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
        if (g_batch_size <= 0 || g_batch_size > 256) {
            printf("[ERROR] Invalid batch size (max 256): %s\n", argv[argi]); return 1;
        }
        argi++;
    }

    printf("[MNIST] VTCM Training with HVX f16 matmul (dspqueue)\n");
    printf("[MNIST] Epochs: %d, Batch: %d, Mode: %s%s\n",
           epochs, g_batch_size, use_train_all ? "train-all" : "per-batch",
           use_test_synthetic ? " + synthetic test" : "");

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

    /* Run synthetic digit test after training (weights still in VTCM) */
    if (use_test_synthetic && ret == 0) {
        /* Allocate fresh eval buffer for synthetic test
         * (training freed its eval buffer) */
        g_eval_buf_f16 = (_Float16 *)net_alloc(
            INPUT_DIM_PAD * sizeof(_Float16));  /* batch_size=1 */
        g_eval_fd = dspq_find_fd(g_eval_buf_f16);
        if (!g_eval_buf_f16 || g_eval_fd < 0) {
            printf("[ERROR] synthetic test eval buffer allocation failed\n");
        } else {
            test_synthetic();
            net_free(g_eval_buf_f16);
            g_eval_buf_f16 = NULL;
            g_eval_fd = -1;
        }
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
