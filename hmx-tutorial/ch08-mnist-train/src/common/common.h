/*
 * mnist_common.h -- Shared constants, types, and utility functions
 */

#ifndef MNIST_COMMON_H
#define MNIST_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>

/* ====================================================================
 * Constants
 * ==================================================================== */

#define INPUT_DIM       784
#define INPUT_DIM_PAD   832     /* padded to 13x64 for HVX alignment (832*2=13*128B) */
#define HIDDEN_DIM      128
#define OUTPUT_DIM      10
#define OUTPUT_DIM_PAD  64      /* padded to 64 for HVX alignment (64*2=128B=1 vector) */

static int g_batch_size = 32;
#define LEARNING_RATE   0.1f
#define DEFAULT_EPOCHS  5

#define MNIST_TRAIN_IMAGES  "train-images-idx3-ubyte"
#define MNIST_TRAIN_LABELS  "train-labels-idx1-ubyte"
#define MNIST_TEST_IMAGES   "t10k-images-idx3-ubyte"
#define MNIST_TEST_LABELS   "t10k-labels-idx1-ubyte"

/* ====================================================================
 * Training mode
 * ==================================================================== */

typedef enum {
    MODE_CPU = 0,
    MODE_FASTRPC = 1,
    MODE_DSPQUEUE = 2
} train_mode_t;

static const char *mode_name(train_mode_t m) {
    switch (m) {
        case MODE_CPU:      return "cpu";
        case MODE_FASTRPC:  return "fastrpc";
        case MODE_DSPQUEUE: return "dspqueue";
    }
    return "unknown";
}

/* ====================================================================
 * Matmul function pointer type
 *
 * All training matmuls dispatch through this:
 *   transpose=0: C[m x n]  = A[m x k] @ B[k x n]
 *   transpose=1: C[m x n]  = A[m x k] @ B^T[n x k]
 *   transpose=2: C[m x n] += A^T[k x m]^T @ B[k x n]
 * ==================================================================== */

typedef void (*matmul_fn_t)(float *C, const float *A, const float *B,
                            int m, int n, int k, int transpose);

/* ====================================================================
 * Simple LCG random number generator
 * ==================================================================== */

static uint64_t g_rng_state = 42;

static uint32_t lcg_rand(void) {
    g_rng_state = g_rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(g_rng_state >> 33);
}

static float rand_uniform(void) {
    return (float)(lcg_rand() & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

static float rand_normal(void) {
    float u1 = rand_uniform();
    float u2 = rand_uniform();
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

/* ====================================================================
 * Timing utility
 * ==================================================================== */

static double time_ms(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 +
           (end->tv_nsec - start->tv_nsec) / 1000000.0;
}

/* ====================================================================
 * Weight initialization
 * ==================================================================== */

static void he_init(float *W, int rows, int cols, int fan_in) {
    float std = sqrtf(2.0f / (float)fan_in);
    for (int i = 0; i < rows * cols; i++)
        W[i] = rand_normal() * std;
}

/* ====================================================================
 * Shuffle
 * ==================================================================== */

static void shuffle_indices(int *indices, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(lcg_rand() % (uint32_t)(i + 1));
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

/* ====================================================================
 * Pluggable allocator (for shared memory in dspqueue mode)
 *
 * In dspqueue mode, network buffers are allocated in shared memory
 * (rpcmem) so the DSP can access them directly without memcpy.
 * ==================================================================== */

typedef void* (*alloc_fn_t)(size_t size);
typedef void  (*free_fn_t)(void *ptr);

static alloc_fn_t g_alloc_fn = NULL;   /* NULL = use malloc */
static free_fn_t  g_free_fn  = NULL;   /* NULL = use free */

static void *net_alloc(size_t size) {
    return g_alloc_fn ? g_alloc_fn(size) : malloc(size);
}

static void *net_calloc(size_t count, size_t size) {
    void *p = net_alloc(count * size);
    if (p) memset(p, 0, count * size);
    return p;
}

static void net_free(void *p) {
    if (g_free_fn) g_free_fn(p);
    else free(p);
}

/* ====================================================================
 * Network structure
 * ==================================================================== */

typedef struct {
    /* Weights and biases */
    float *w1;          /* [HIDDEN_DIM x INPUT_DIM_PAD] */
    float *b1;          /* [HIDDEN_DIM] */
    float *w2;          /* [OUTPUT_DIM_PAD x HIDDEN_DIM] */
    float *b2;          /* [OUTPUT_DIM_PAD] */

    /* Gradients */
    float *dw1, *db1, *dw2, *db2;

    /* Intermediate buffers */
    float *hidden;          /* [batch x HIDDEN_DIM] */
    float *hidden_pre_relu; /* [batch x HIDDEN_DIM] */
    float *logits;          /* [batch x OUTPUT_DIM_PAD] */
    float *probs;           /* [batch x OUTPUT_DIM_PAD] */
    float *dlogits;         /* [batch x OUTPUT_DIM_PAD] */
    float *dhidden;         /* [batch x HIDDEN_DIM] */
} network_t;

#endif /* MNIST_COMMON_H */
