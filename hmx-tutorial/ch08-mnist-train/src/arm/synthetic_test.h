/*
 * synthetic_test.h -- Synthetic digit generation and CPU inference test
 *
 * Generates handwritten-style digit images 0-9, runs forward pass using
 * trained weights, and displays ASCII art with predictions.
 *
 * Used by train_cpu.c with --test-synthetic flag to verify the trained
 * network can recognize simple digit patterns.
 */

#ifndef SYNTHETIC_TEST_H
#define SYNTHETIC_TEST_H

#include "common/common.h"
#include "arm/cpu_matmul.h"

/* ====================================================================
 * Drawing primitives (28x28 float images, white-on-black like MNIST)
 * ==================================================================== */

/* Set a pixel with max-blending */
static void set_pixel(float *img, int x, int y, float val) {
    if (x >= 0 && x < 28 && y >= 0 && y < 28)
        img[y * 28 + x] = fmaxf(img[y * 28 + x], val);
}

/* Bresenham line with thickness */
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

/* Circle outline */
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

/* Arc from angle_start to angle_end (degrees) */
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

/* 3x3 box blur for MNIST-style soft edges */
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

/* ====================================================================
 * Generate synthetic digit images 0-9
 *
 * Same patterns as ch09's generate_synthetic_digits: thick strokes
 * centered in ~20x20 area, then blurred for soft edges.
 * ==================================================================== */

static void generate_synthetic_digits(float imgs[10][28 * 28]) {
    memset(imgs, 0, 10 * 28 * 28 * sizeof(float));
    int t = 3;  /* stroke thickness */

    /* Digit 0: oval */
    draw_circle(imgs[0], 14, 14, 9, t);

    /* Digit 1: vertical line, slightly right of center */
    draw_line(imgs[1], 15, 5, 15, 23, t);

    /* Digit 2: top arc + steep diagonal + wide base */
    draw_arc(imgs[2], 14, 10, 6, 200, 360, t);
    draw_line(imgs[2], 20, 10, 8, 23, t);
    draw_line(imgs[2], 7, 23, 22, 23, t);

    /* Digit 3: two arcs open to the left */
    draw_arc(imgs[3], 13, 10, 7, 270, 450, t);
    draw_arc(imgs[3], 13, 18, 7, 270, 450, t);

    /* Digit 4: diagonal + horizontal bar + vertical stroke */
    draw_line(imgs[4], 6, 5, 18, 17, t);
    draw_line(imgs[4], 6, 17, 22, 17, t);
    draw_line(imgs[4], 18, 5, 18, 24, t);

    /* Digit 5: top horizontal + left vertical + middle horizontal + bottom arc */
    draw_line(imgs[5], 8, 5, 20, 5, t);
    draw_line(imgs[5], 8, 5, 8, 13, t);
    draw_line(imgs[5], 8, 13, 16, 13, t);
    draw_arc(imgs[5], 14, 18, 6, 270, 450, t);

    /* Digit 6: curved top stroke + bottom circle */
    draw_arc(imgs[6], 20, 6, 10, 120, 200, t);
    draw_circle(imgs[6], 14, 18, 6, t);
    draw_line(imgs[6], 10, 10, 9, 13, t);

    /* Digit 7: top stroke + vertical diagonal */
    draw_line(imgs[7], 9, 5, 20, 5, t);
    draw_line(imgs[7], 20, 5, 14, 24, t);

    /* Digit 8: two circles stacked */
    draw_circle(imgs[8], 14, 10, 5, t);
    draw_circle(imgs[8], 14, 19, 6, t);

    /* Digit 9: top circle + right vertical down */
    draw_circle(imgs[9], 14, 10, 6, t);
    draw_line(imgs[9], 20, 10, 16, 24, t);

    /* Blur all digits for MNIST-style soft edges */
    for (int d = 0; d < 10; d++)
        blur_image(imgs[d]);
}

/* ====================================================================
 * ASCII art display
 * ==================================================================== */

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

/* ====================================================================
 * CPU inference: single-image forward pass, returns predicted digit
 *
 * Network: input[INPUT_DIM_PAD] -> hidden[HIDDEN_DIM] (ReLU) -> output[OUTPUT_DIM_PAD]
 * Weights stored as W1[HIDDEN_DIM x INPUT_DIM_PAD], W2[OUTPUT_DIM_PAD x HIDDEN_DIM]
 * Forward uses transpose=1 (matmul_nt): C = A @ B^T
 * ==================================================================== */

static int cpu_infer(const float *image_padded, const network_t *net) {
    float hidden[HIDDEN_DIM];
    float logits[OUTPUT_DIM_PAD];

    /* Layer 1: hidden = image @ W1^T + b1, then ReLU */
    cpu_matmul_nt(1, HIDDEN_DIM, INPUT_DIM_PAD, hidden, image_padded, net->w1);
    for (int j = 0; j < HIDDEN_DIM; j++) {
        hidden[j] += net->b1[j];
        if (hidden[j] < 0.0f) hidden[j] = 0.0f;  /* ReLU */
    }

    /* Layer 2: logits = hidden @ W2^T + b2 */
    cpu_matmul_nt(1, OUTPUT_DIM_PAD, HIDDEN_DIM, logits, hidden, net->w2);
    for (int j = 0; j < OUTPUT_DIM_PAD; j++)
        logits[j] += net->b2[j];

    /* Argmax over first OUTPUT_DIM logits */
    int best = 0;
    float best_val = logits[0];
    for (int j = 1; j < OUTPUT_DIM; j++) {
        if (logits[j] > best_val) {
            best_val = logits[j];
            best = j;
        }
    }
    return best;
}

/* ====================================================================
 * test_synthetic -- Generate digits, run CPU inference, display results
 * ==================================================================== */

static void test_synthetic(const network_t *net) {
    printf("\n");
    printf("==========================================================\n");
    printf(" Synthetic Digit Recognition Test (CPU)\n");
    printf("==========================================================\n\n");

    float synthetic[10][28 * 28];
    generate_synthetic_digits(synthetic);

    float padded[INPUT_DIM_PAD];
    int num_correct = 0;

    printf("%-8s %-12s %-10s\n", "Digit", "Predicted", "Result");
    printf("------   ---------    ------\n");

    for (int d = 0; d < 10; d++) {
        print_ascii_art(synthetic[d], d);

        /* Pad 784 -> INPUT_DIM_PAD with zeros */
        memset(padded, 0, sizeof(padded));
        memcpy(padded, synthetic[d], 784 * sizeof(float));

        int predicted = cpu_infer(padded, net);
        int correct = (predicted == d);
        if (correct) num_correct++;

        printf("  %-8d %-12d %s\n\n", d, predicted,
               correct ? "CORRECT" : "WRONG");
    }

    printf("==========================================================\n");
    printf(" Summary: %d / 10 correct (%.0f%%)\n", num_correct, num_correct * 10.0f);
    printf("==========================================================\n\n");
}

#endif /* SYNTHETIC_TEST_H */
