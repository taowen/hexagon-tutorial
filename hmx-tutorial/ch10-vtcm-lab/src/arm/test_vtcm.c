/*
 * ch10: VTCM lab -- ARM-side test driver
 *
 * Sends systematic experiments to DSP via dspqueue, verifies results.
 * Reuses ch08's IDL (mnist_train) and dspqueue infrastructure.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <semaphore.h>

#include <rpcmem.h>
#include <AEEStdErr.h>
#include "dspqueue.h"
#include "mnist_train.h"
#include "common/lab_protocol.h"

/* ================================================================
 * Minimal dspqueue ARM infrastructure (inlined from ch08)
 * ================================================================ */

#define CDSP_DOMAIN_ID 3
#define MAX_SHARED_BUFS 20

struct shared_buf_entry { void *ptr; int fd; size_t size; };
static struct shared_buf_entry g_bufs[MAX_SHARED_BUFS];
static int g_n_bufs = 0;

static void *shm_alloc(size_t size) {
    size_t aligned = (size + 4095) & ~4095UL;
    void *p = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                           RPCMEM_DEFAULT_FLAGS | RPCMEM_HEAP_NOREG, aligned);
    if (!p) return NULL;
    int fd = rpcmem_to_fd(p);
    if (fd <= 0) { rpcmem_free(p); return NULL; }
    int err = fastrpc_mmap(CDSP_DOMAIN_ID, fd, p, 0, aligned, FASTRPC_MAP_FD);
    if (err) { rpcmem_free(p); return NULL; }
    g_bufs[g_n_bufs++] = (struct shared_buf_entry){p, fd, aligned};
    return p;
}

static void shm_free(void *p) {
    for (int i = 0; i < g_n_bufs; i++) {
        if (g_bufs[i].ptr == p) {
            fastrpc_munmap(CDSP_DOMAIN_ID, g_bufs[i].fd, NULL, 0);
            rpcmem_free(p);
            g_bufs[i] = g_bufs[--g_n_bufs];
            return;
        }
    }
    free(p);
}

static int shm_fd(const void *p) {
    for (int i = 0; i < g_n_bufs; i++) {
        if (p >= (const void *)g_bufs[i].ptr &&
            (const char *)p < (const char *)g_bufs[i].ptr + g_bufs[i].size)
            return g_bufs[i].fd;
    }
    return -1;
}

/* Response tracking */
static sem_t g_done;
static struct lab_rsp g_last_rsp;

static void arm_error_cb(dspqueue_t q, AEEResult err, void *ctx) {
    fprintf(stderr, "[ERROR] dspqueue: 0x%x\n", err);
}

static void arm_packet_cb(dspqueue_t q, AEEResult err, void *ctx) {
    while (1) {
        struct lab_rsp rsp;
        uint32_t flags, msg_len, n_bufs;
        struct dspqueue_buffer bufs[LAB_MAX_BUFFERS];
        int e = dspqueue_read_noblock(q, &flags, LAB_MAX_BUFFERS, &n_bufs, bufs,
                                       sizeof(rsp), &msg_len, (uint8_t *)&rsp);
        if (e == AEE_EWOULDBLOCK) return;
        if (e) return;
        g_last_rsp = rsp;
        sem_post(&g_done);
    }
}

/* Send a message and wait for response */
static struct lab_rsp send_msg(dspqueue_t queue, struct lab_req *req,
                                int n_bufs, struct dspqueue_buffer *bufs) {
    dspqueue_write(queue, 0, n_bufs, bufs,
                   sizeof(*req), (const uint8_t *)req, 1000000);
    sem_wait(&g_done);
    return g_last_rsp;
}


/* ================================================================
 * Experiments
 * ================================================================ */

static int test_basic(dspqueue_t queue) {
    printf("\n=== Exp 1: VTCM basic write/read ===\n");
    int fails = 0;

    for (int i = 0; i < 100; i++) {
        struct lab_req req = {0};
        req.op = OP_LAB_VTCM_BASIC;
        req.iter = i;
        req.size = 16384;  /* 16K floats */

        struct lab_rsp rsp = send_msg(queue, &req, 0, NULL);
        if (rsp.status != 0) {
            printf("  FAIL iter %d: %u errors\n", i, rsp.errors);
            fails++;
        }
    }
    printf("  Result: %d/100 passed\n", 100 - fails);
    return fails;
}

static int test_persist(dspqueue_t queue) {
    printf("\n=== Exp 2: VTCM persistence across 10000 messages ===\n");
    int fails = 0;

    for (int i = 0; i < 10000; i++) {
        struct lab_req req = {0};
        req.op = OP_LAB_VTCM_PERSIST;
        req.iter = i;

        struct lab_rsp rsp = send_msg(queue, &req, 0, NULL);
        if (rsp.status != 0) {
            printf("  FAIL at message %d: %u errors\n", i, rsp.errors);
            fails++;
            if (fails >= 5) {
                printf("  (stopping after 5 failures)\n");
                break;
            }
        }
    }
    printf("  Result: %s (%d failures in 10000 messages)\n",
           fails == 0 ? "PASS" : "FAIL", fails);
    return fails;
}

static int test_hvx_write(dspqueue_t queue) {
    printf("\n=== Exp 3: HVX write to VTCM, scalar read ===\n");
    int fails = 0;

    for (int i = 0; i < 1000; i++) {
        struct lab_req req = {0};
        req.op = OP_LAB_HVX_WRITE;
        req.iter = i;
        req.size = 4096;

        struct lab_rsp rsp = send_msg(queue, &req, 0, NULL);
        if (rsp.status != 0) {
            printf("  FAIL iter %d: %u errors\n", i, rsp.errors);
            fails++;
            if (fails >= 5) break;
        }
    }
    printf("  Result: %s (%d failures in 1000 iters)\n",
           fails == 0 ? "PASS" : "FAIL", fails);
    return fails;
}

static int test_scalar_write(dspqueue_t queue) {
    printf("\n=== Exp 4: Scalar write to VTCM, HVX read ===\n");
    int fails = 0;

    for (int i = 0; i < 1000; i++) {
        struct lab_req req = {0};
        req.op = OP_LAB_SCALAR_WRITE;
        req.iter = i;
        req.size = 4096;

        struct lab_rsp rsp = send_msg(queue, &req, 0, NULL);
        if (rsp.status != 0) {
            printf("  FAIL iter %d: %u errors\n", i, rsp.errors);
            fails++;
            if (fails >= 5) break;
        }
    }
    printf("  Result: %s (%d failures in 1000 iters)\n",
           fails == 0 ? "PASS" : "FAIL", fails);
    return fails;
}

static int test_sgd(dspqueue_t queue) {
    printf("\n=== Exp 5: HVX SGD on VTCM (5000 iters) ===\n");
    float lr = 0.1f;

    /* Run 5000 SGD iterations on DSP */
    for (int i = 0; i < 5000; i++) {
        struct lab_req req = {0};
        req.op = OP_LAB_SGD_ITER;
        req.iter = i;
        req.learning_rate = lr;
        send_msg(queue, &req, 0, NULL);
    }

    /* Read back weights */
    float *w_dsp = (float *)shm_alloc(102400 * 4);  /* W1_FLOATS */
    if (!w_dsp) {
        printf("  FAIL: shm_alloc failed\n");
        return 1;
    }

    struct dspqueue_buffer bufs[1];
    memset(bufs, 0, sizeof(bufs));
    bufs[0].fd = shm_fd(w_dsp);
    bufs[0].flags = DSPQUEUE_BUFFER_FLAG_REF
                  | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;

    struct lab_req req = {0};
    req.op = OP_LAB_SGD_VERIFY;
    req.iter = 5000;
    send_msg(queue, &req, 1, bufs);

    /* CPU reference: w[i] = initial - 5000 * lr * 0.001
     * initial = (i % 100) * 0.01
     * grad = 0.001 (constant)
     * After 5000 iters: w[i] = (i%100)*0.01 - 5000 * 0.1 * 0.001 = (i%100)*0.01 - 0.5
     *
     * But qf32 has precision loss, so use a tolerance.
     */
    int errors = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < 102400; i++) {
        /* CPU reference: each SGD step does w -= lr * grad in IEEE f32 */
        float w_cpu = (float)(i % 100) * 0.01f;
        for (int s = 0; s < 5000; s++)
            w_cpu -= lr * 0.001f;

        float diff = fabsf(w_dsp[i] - w_cpu);
        if (diff > max_diff) max_diff = diff;

        /* qf32 accumulates error over 5000 iters, use generous tolerance */
        if (diff > 0.1f) {
            if (errors < 3) {
                printf("  idx=%d: dsp=%.6f cpu=%.6f diff=%.6f\n",
                       i, w_dsp[i], w_cpu, diff);
            }
            errors++;
        }
    }

    printf("  Max diff: %.6f, errors (>0.1): %d/102400\n", max_diff, errors);
    printf("  Result: %s\n", errors == 0 ? "PASS" : "FAIL");

    shm_free(w_dsp);
    return errors > 0 ? 1 : 0;
}


static int test_l2_stale(dspqueue_t queue) {
    printf("\n=== Exp 7: L2 stale read after memcpy ===\n");
    printf("  Tests: HVX writes A -> memcpy(heap,VTCM) -> HVX writes B -> scalar reads\n");
    printf("  If L2 caches VTCM during memcpy, scalar reads may return A instead of B\n");

    /* Run 10 rounds to be thorough */
    int fails = 0;
    for (int i = 0; i < 10; i++) {
        struct lab_req req = {0};
        req.op = OP_LAB_L2_STALE_TEST;
        req.iter = i;

        struct lab_rsp rsp = send_msg(queue, &req, 0, NULL);
        if (rsp.status == 2) {
            printf("  INFRASTRUCTURE FAIL: basic HVX->scalar broken\n");
            return 1;
        }
        if (rsp.status != 0) {
            printf("  Round %d: STALE DETECTED — %u/%d reads returned %.1f instead of 2.0\n",
                   i, rsp.errors, 16384, rsp.detail);
            fails++;
        } else {
            if (i == 0) printf("  Round %d: no stale reads\n", i);
        }
    }

    printf("  Result: %s (%d/10 rounds had stale reads)\n",
           fails > 0 ? "L2 STALE CONFIRMED" : "NO STALE (L2 coherent)", fails);
    return fails > 0 ? 1 : 0;
}

static int test_l2_inval(dspqueue_t queue) {
    printf("\n=== Exp 8: Cache invalidation after memcpy ===\n");
    printf("  Same as Exp 7 but with qurt_mem_cache_clean(INVALIDATE) before scalar reads\n");

    int fails = 0;
    for (int i = 0; i < 10; i++) {
        struct lab_req req = {0};
        req.op = OP_LAB_L2_INVAL_TEST;
        req.iter = i;

        struct lab_rsp rsp = send_msg(queue, &req, 0, NULL);
        if (rsp.status != 0) {
            printf("  Round %d: STILL STALE — %u reads returned %.1f instead of 4.0\n",
                   i, rsp.errors, rsp.detail);
            fails++;
        } else {
            if (i == 0) printf("  Round %d: invalidation worked\n", i);
        }
    }

    printf("  Result: %s\n",
           fails > 0 ? "INVALIDATION FAILED (stale data persists)" : "INVALIDATION WORKS");
    return fails > 0 ? 1 : 0;
}


/* ================================================================
 * Exp 10a-d: Incremental accumulation tests
 * ================================================================ */

static int test_increment(dspqueue_t queue) {
    printf("\n=== Exp 10a: HVX increment on VTCM (1000 rounds, no sync) ===\n");
    int fails = 0;

    for (int i = 0; i < 1000; i++) {
        struct lab_req req = {0};
        req.op = OP_LAB_INCREMENT;
        req.iter = i;

        struct lab_rsp rsp = send_msg(queue, &req, 0, NULL);
        if (rsp.status != 0) {
            printf("  FAIL at iter %d: %u errors, expected %.1f got %.4f\n",
                   i, rsp.errors, (float)(i + 1), rsp.detail);
            fails++;
            if (fails >= 3) { printf("  (stopping after 3 failures)\n"); break; }
        }
    }

    printf("  Result: %s (%d failures in 1000)\n",
           fails == 0 ? "PASS" : "FAIL", fails);
    return fails > 0 ? 1 : 0;
}

static int test_increment_sync(dspqueue_t queue) {
    printf("\n=== Exp 10b: HVX increment + memcpy + INVALIDATE (1000 rounds) ===\n");
    int fails = 0;
    int first_fail = -1;

    for (int i = 0; i < 1000; i++) {
        struct lab_req req = {0};
        req.op = OP_LAB_INCREMENT_SYNC;
        req.iter = i;

        struct lab_rsp rsp = send_msg(queue, &req, 0, NULL);
        if (rsp.status != 0) {
            if (first_fail < 0) first_fail = i;
            printf("  FAIL at iter %d: %u errors, expected %.1f got %.4f\n",
                   i, rsp.errors, (float)(i + 1), rsp.detail);
            fails++;
            if (fails >= 5) { printf("  (stopping after 5 failures)\n"); break; }
        }
    }

    printf("  Result: %s (%d failures, first at iter %d)\n",
           fails == 0 ? "PASS" : "FAIL", fails, first_fail);
    return fails > 0 ? 1 : 0;
}

static int test_increment_flush(dspqueue_t queue) {
    printf("\n=== Exp 10c: HVX increment + memcpy + FLUSH_INVALIDATE (1000 rounds) ===\n");
    int fails = 0;
    int first_fail = -1;

    for (int i = 0; i < 1000; i++) {
        struct lab_req req = {0};
        req.op = OP_LAB_INCREMENT_FLUSH;
        req.iter = i;

        struct lab_rsp rsp = send_msg(queue, &req, 0, NULL);
        if (rsp.status != 0) {
            if (first_fail < 0) first_fail = i;
            printf("  FAIL at iter %d: %u errors, expected %.1f got %.4f\n",
                   i, rsp.errors, (float)(i + 1), rsp.detail);
            fails++;
            if (fails >= 5) { printf("  (stopping after 5 failures)\n"); break; }
        }
    }

    printf("  Result: %s (%d failures, first at iter %d)\n",
           fails == 0 ? "PASS" : "FAIL", fails, first_fail);
    return fails > 0 ? 1 : 0;
}

static int test_increment_nocache(dspqueue_t queue) {
    printf("\n=== Exp 10d: HVX increment + memcpy, no cache ops (1000 rounds) ===\n");
    int fails = 0;
    int first_fail = -1;

    for (int i = 0; i < 1000; i++) {
        struct lab_req req = {0};
        req.op = OP_LAB_INCREMENT_NOCACHE;
        req.iter = i;

        struct lab_rsp rsp = send_msg(queue, &req, 0, NULL);
        if (rsp.status != 0) {
            if (first_fail < 0) first_fail = i;
            printf("  FAIL at iter %d: %u errors, expected %.1f got %.4f\n",
                   i, rsp.errors, (float)(i + 1), rsp.detail);
            fails++;
            if (fails >= 5) { printf("  (stopping after 5 failures)\n"); break; }
        }
    }

    printf("  Result: %s (%d failures, first at iter %d)\n",
           fails == 0 ? "PASS" : "FAIL", fails, first_fail);
    return fails > 0 ? 1 : 0;
}


/* ================================================================
 * Exp 11: Scalar write VTCM → INVALIDATE → scalar read
 *
 * Tests if scalar writes to VTCM go through L2 (write-back).
 * If so, INVALIDATE drops the dirty lines and data is LOST.
 * ================================================================ */

static int test_scalar_write_inval(dspqueue_t queue) {
    printf("\n=== Exp 11: Scalar write VTCM -> INVALIDATE -> scalar read ===\n");
    printf("  Tests: if scalar writes to VTCM create dirty L2 lines that INVALIDATE drops\n");
    int fails = 0;

    for (int i = 0; i < 100; i++) {
        struct lab_req req = {0};
        req.op = OP_LAB_SCALAR_WRITE_INVAL;
        req.iter = i;

        struct lab_rsp rsp = send_msg(queue, &req, 0, NULL);
        if (rsp.status != 0) {
            if (fails == 0) {
                printf("  FAIL at iter %d: %u/4096 values lost (got %.1f)\n",
                       i, rsp.errors, rsp.detail);
            }
            fails++;
        }
    }

    if (fails > 0) {
        printf("  Result: SCALAR WRITES LOST ON INVALIDATE (%d/100 rounds)\n", fails);
        printf("  ** This confirms: scalar writes to VTCM use L2 write-back **\n");
        printf("  ** INVALIDATE drops dirty lines, losing scalar-written data **\n");
    } else {
        printf("  Result: PASS (scalar writes survive INVALIDATE)\n");
    }
    return fails > 0 ? 1 : 0;
}


/* ================================================================
 * Exp 6: Full MNIST training in VTCM
 * ================================================================ */

/* Minimal MNIST loading (from ch08) */
static int read_int32_be(FILE *f) {
    unsigned char buf[4];
    fread(buf, 1, 4, f);
    return (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];
}

static float *load_images(const char *path, int *count) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    read_int32_be(f);  /* magic */
    *count = read_int32_be(f);
    int rows = read_int32_be(f);
    int cols = read_int32_be(f);
    int dim = rows * cols;  /* 784 */
    float *data = (float *)malloc((size_t)*count * 800 * sizeof(float));
    if (!data) { fclose(f); return NULL; }
    memset(data, 0, (size_t)*count * 800 * sizeof(float));
    unsigned char *raw = (unsigned char *)malloc(dim);
    for (int i = 0; i < *count; i++) {
        fread(raw, 1, dim, f);
        for (int j = 0; j < dim; j++)
            data[i * 800 + j] = raw[j] / 255.0f;
    }
    free(raw);
    fclose(f);
    return data;
}

static uint8_t *load_labels(const char *path, int *count) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    read_int32_be(f);
    *count = read_int32_be(f);
    uint8_t *data = (uint8_t *)malloc(*count);
    fread(data, 1, *count, f);
    fclose(f);
    return data;
}

/* Simple LCG */
static uint64_t rng_state = 42;
static uint32_t lcg_rand(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(rng_state >> 33);
}
static float rand_normal(void) {
    float u1 = (float)(lcg_rand() & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    float u2 = (float)(lcg_rand() & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static void he_init(float *W, int rows, int cols, int fan_in) {
    float std = sqrtf(2.0f / (float)fan_in);
    for (int i = 0; i < rows * cols; i++)
        W[i] = rand_normal() * std;
}

/* CPU forward for evaluation */
static void cpu_matmul_nt(float *C, const float *A, const float *B,
                           int m, int n, int k) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++)
                s += A[i * k + p] * B[j * k + p];
            C[i * n + j] = s;
        }
}

static float cpu_evaluate(const float *w1, const float *b1,
                           const float *w2, const float *b2,
                           const float *images, const uint8_t *labels,
                           int count) {
    int correct = 0;
    int batch = 128;
    float *hidden = (float *)malloc(batch * 128 * sizeof(float));
    float *logits = (float *)malloc(batch * 32 * sizeof(float));

    for (int bi = 0; bi < count / batch; bi++) {
        const float *inp = images + (size_t)bi * batch * 800;
        const uint8_t *lab = labels + bi * batch;

        cpu_matmul_nt(hidden, inp, w1, batch, 128, 800);
        for (int i = 0; i < batch * 128; i++) {
            hidden[i] += b1[i % 128];
            if (hidden[i] < 0) hidden[i] = 0;
        }
        cpu_matmul_nt(logits, hidden, w2, batch, 32, 128);
        for (int i = 0; i < batch * 32; i++)
            logits[i] += b2[i % 32];

        for (int b = 0; b < batch; b++) {
            float *row = logits + b * 32;
            float mv = row[0]; int mj = 0;
            for (int j = 1; j < 10; j++)
                if (row[j] > mv) { mv = row[j]; mj = j; }
            if (mj == lab[b]) correct++;
        }
    }

    free(logits);
    free(hidden);
    return (float)correct / (float)((count / batch) * batch);
}

static int test_training(dspqueue_t queue, int epochs) {
    printf("\n=== Exp 6: Full MNIST training in VTCM (%d epochs) ===\n", epochs);

    /* Load data */
    int train_count = 0, test_count = 0;
    int train_lc = 0, test_lc = 0;
    float *train_img = load_images("train-images-idx3-ubyte", &train_count);
    uint8_t *train_lab = load_labels("train-labels-idx1-ubyte", &train_lc);
    float *test_img = load_images("t10k-images-idx3-ubyte", &test_count);
    uint8_t *test_lab = load_labels("t10k-labels-idx1-ubyte", &test_lc);

    if (!train_img || !train_lab || !test_img || !test_lab) {
        printf("  FAIL: Cannot load MNIST data\n");
        free(train_img); free(train_lab); free(test_img); free(test_lab);
        return 1;
    }
    printf("  Data: %d train, %d test\n", train_count, test_count);

    /* Allocate weights in shared memory */
    int batch_size = 128;
    float *w1 = (float *)shm_alloc(128 * 800 * sizeof(float));
    float *b1 = (float *)shm_alloc(128 * sizeof(float));
    float *w2 = (float *)shm_alloc(32 * 128 * sizeof(float));
    float *b2 = (float *)shm_alloc(32 * sizeof(float));
    float *batch_buf = (float *)shm_alloc(batch_size * 800 * sizeof(float));

    if (!w1 || !b1 || !w2 || !b2 || !batch_buf) {
        printf("  FAIL: shm_alloc\n");
        return 1;
    }

    /* Initialize weights */
    he_init(w1, 128, 800, 800);
    memset(b1, 0, 128 * sizeof(float));
    he_init(w2, 32, 128, 128);
    memset(b2, 0, 32 * sizeof(float));
    /* Zero padding rows */
    for (int j = 0; j < 800; j++) {
        for (int i = 10; i < 32; i++)
            w2[i * 128] = 0;  /* actually just zero init the padding rows */
    }
    for (int i = 10; i < 32; i++) {
        for (int j = 0; j < 128; j++)
            w2[i * 128 + j] = 0;
        b2[i] = 0;
    }

    /* Send weights to VTCM */
    struct dspqueue_buffer init_bufs[4];
    memset(init_bufs, 0, sizeof(init_bufs));
    init_bufs[0].fd = shm_fd(w1);
    init_bufs[0].flags = DSPQUEUE_BUFFER_FLAG_REF | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER;
    init_bufs[1].fd = shm_fd(b1);
    init_bufs[1].flags = DSPQUEUE_BUFFER_FLAG_REF | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER;
    init_bufs[2].fd = shm_fd(w2);
    init_bufs[2].flags = DSPQUEUE_BUFFER_FLAG_REF | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER;
    init_bufs[3].fd = shm_fd(b2);
    init_bufs[3].flags = DSPQUEUE_BUFFER_FLAG_REF | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER;

    struct lab_req init_req = {0};
    init_req.op = OP_LAB_TRAIN_INIT;
    struct lab_rsp init_rsp = send_msg(queue, &init_req, 4, init_bufs);
    if (init_rsp.status != 0) {
        printf("  FAIL: TRAIN_INIT status=%u\n", init_rsp.status);
        return 1;
    }
    printf("  Weights sent to VTCM\n");

    /* Shuffle indices */
    int n_batches = train_count / batch_size;
    int *indices = (int *)malloc(train_count * sizeof(int));
    for (int i = 0; i < train_count; i++) indices[i] = i;
    uint8_t *label_buf = (uint8_t *)malloc(batch_size);

    int fail_epoch = -1;

    for (int epoch = 0; epoch < epochs; epoch++) {
        /* Shuffle */
        for (int i = train_count - 1; i > 0; i--) {
            int j = lcg_rand() % (uint32_t)(i + 1);
            int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
        }

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        float epoch_loss = 0;
        int epoch_correct = 0;

        for (int bi = 0; bi < n_batches; bi++) {
            /* Assemble batch */
            for (int b = 0; b < batch_size; b++) {
                int idx = indices[bi * batch_size + b];
                memcpy(batch_buf + b * 800,
                       train_img + (size_t)idx * 800, 800 * sizeof(float));
                label_buf[b] = train_lab[idx];
            }

            /* Send training batch */
            struct lab_req req = {0};
            req.op = OP_LAB_TRAIN_BATCH;
            req.batch_size = batch_size;
            req.learning_rate = 0.1f;
            memcpy(req.labels, label_buf, batch_size);

            struct dspqueue_buffer bufs[1];
            memset(bufs, 0, sizeof(bufs));
            bufs[0].fd = shm_fd(batch_buf);
            bufs[0].flags = DSPQUEUE_BUFFER_FLAG_REF
                          | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
                          | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;

            struct lab_rsp rsp = send_msg(queue, &req, 1, bufs);
            epoch_loss += rsp.detail;
            epoch_correct += rsp.errors;  /* errors field reused for correct count */
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                    (t1.tv_nsec - t0.tv_nsec) / 1e6;

        float train_acc = (float)epoch_correct / (float)(n_batches * batch_size);

        /* No eval, just report train accuracy */
        float test_acc = -1.0f;

        printf("  Epoch %d/%d: loss=%.4f train_acc=%.4f test_acc=%.4f time=%.0fms",
               epoch + 1, epochs,
               epoch_loss / n_batches, train_acc, test_acc, ms);

        if (test_acc >= 0 && test_acc < 0.2f && epoch >= 2) {
            printf(" ** CORRUPTION DETECTED **\n");
            if (fail_epoch < 0) fail_epoch = epoch + 1;
        } else {
            printf("\n");
        }
    }

    free(label_buf);
    free(indices);
    free(test_lab);
    free(test_img);
    free(train_lab);
    free(train_img);
    shm_free(batch_buf);
    shm_free(b2);
    shm_free(w2);
    shm_free(b1);
    shm_free(w1);

    if (fail_epoch > 0) {
        printf("  Result: FAIL (corruption at epoch %d)\n", fail_epoch);
        return 1;
    }
    printf("  Result: PASS\n");
    return 0;
}


/* Generic training-with-sync test — sync_op selects which SYNC variant to use */
static int test_training_sync_variant(dspqueue_t queue, int epochs,
                                       uint32_t sync_op, const char *sync_name) {
    printf("\n=== Training + %s (%d epochs) ===\n", sync_name, epochs);

    int train_count = 0, test_count = 0;
    int train_lc = 0, test_lc = 0;
    float *train_img = load_images("train-images-idx3-ubyte", &train_count);
    uint8_t *train_lab = load_labels("train-labels-idx1-ubyte", &train_lc);
    float *test_img = load_images("t10k-images-idx3-ubyte", &test_count);
    uint8_t *test_lab = load_labels("t10k-labels-idx1-ubyte", &test_lc);

    if (!train_img || !train_lab || !test_img || !test_lab) {
        printf("  FAIL: Cannot load MNIST data\n");
        free(train_img); free(train_lab); free(test_img); free(test_lab);
        return 1;
    }

    int batch_size = 128;
    float *w1 = (float *)shm_alloc(128 * 800 * sizeof(float));
    float *b1 = (float *)shm_alloc(128 * sizeof(float));
    float *w2 = (float *)shm_alloc(32 * 128 * sizeof(float));
    float *b2 = (float *)shm_alloc(32 * sizeof(float));
    float *batch_buf = (float *)shm_alloc(batch_size * 800 * sizeof(float));

    if (!w1 || !b1 || !w2 || !b2 || !batch_buf) {
        printf("  FAIL: shm_alloc\n");
        return 1;
    }

    rng_state = 42;
    he_init(w1, 128, 800, 800);
    memset(b1, 0, 128 * sizeof(float));
    he_init(w2, 32, 128, 128);
    memset(b2, 0, 32 * sizeof(float));
    for (int i = 10; i < 32; i++) {
        for (int j = 0; j < 128; j++) w2[i * 128 + j] = 0;
        b2[i] = 0;
    }

    struct dspqueue_buffer init_bufs[4];
    memset(init_bufs, 0, sizeof(init_bufs));
    init_bufs[0].fd = shm_fd(w1);
    init_bufs[0].flags = DSPQUEUE_BUFFER_FLAG_REF | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER;
    init_bufs[1].fd = shm_fd(b1);
    init_bufs[1].flags = DSPQUEUE_BUFFER_FLAG_REF | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER;
    init_bufs[2].fd = shm_fd(w2);
    init_bufs[2].flags = DSPQUEUE_BUFFER_FLAG_REF | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER;
    init_bufs[3].fd = shm_fd(b2);
    init_bufs[3].flags = DSPQUEUE_BUFFER_FLAG_REF | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER;

    struct lab_req init_req = {0};
    init_req.op = OP_LAB_TRAIN_INIT;
    struct lab_rsp init_rsp = send_msg(queue, &init_req, 4, init_bufs);
    if (init_rsp.status != 0) {
        printf("  FAIL: TRAIN_INIT status=%u\n", init_rsp.status);
        return 1;
    }

    int n_batches = train_count / batch_size;
    int *indices = (int *)malloc(train_count * sizeof(int));
    for (int i = 0; i < train_count; i++) indices[i] = i;
    uint8_t *label_buf = (uint8_t *)malloc(batch_size);

    int fail_epoch = -1;

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = train_count - 1; i > 0; i--) {
            int j = lcg_rand() % (uint32_t)(i + 1);
            int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
        }

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        float epoch_loss = 0;
        int epoch_correct = 0;

        for (int bi = 0; bi < n_batches; bi++) {
            for (int b = 0; b < batch_size; b++) {
                int idx = indices[bi * batch_size + b];
                memcpy(batch_buf + b * 800, train_img + (size_t)idx * 800, 800 * sizeof(float));
                label_buf[b] = train_lab[idx];
            }

            struct lab_req req = {0};
            req.op = OP_LAB_TRAIN_BATCH;
            req.batch_size = batch_size;
            req.learning_rate = 0.1f;
            memcpy(req.labels, label_buf, batch_size);

            struct dspqueue_buffer bufs[1];
            memset(bufs, 0, sizeof(bufs));
            bufs[0].fd = shm_fd(batch_buf);
            bufs[0].flags = DSPQUEUE_BUFFER_FLAG_REF
                          | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
                          | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;

            struct lab_rsp rsp = send_msg(queue, &req, 1, bufs);
            epoch_loss += rsp.detail;
            epoch_correct += rsp.errors;
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

        float train_acc = (float)epoch_correct / (float)(n_batches * batch_size);

        /* SYNC with selected variant */
        struct lab_req sync_req = {0};
        sync_req.op = sync_op;
        struct lab_rsp sync_rsp = send_msg(queue, &sync_req, 0, NULL);

        /* CPU evaluation — OPTIONAL. ARM reading large DDR may interfere with DSP VTCM */
        float test_acc = -1.0f;
        /* CPU-side eval: skip for NOOP (no data to eval) and DSPEVAL (DSP stays busy) */
        if (sync_op != OP_LAB_TRAIN_SYNC_NOOP && sync_op != OP_LAB_TRAIN_SYNC_DSPEVAL)
            test_acc = cpu_evaluate(w1, b1, w2, b2, test_img, test_lab, test_count);

        printf("  Epoch %d/%d: loss=%.4f train=%.4f test=%.4f w1[0]=%.4f %.0fms",
               epoch + 1, epochs, epoch_loss / n_batches, train_acc, test_acc,
               sync_rsp.detail, ms);

        if (train_acc < 0.2f && epoch >= 1) {
            printf(" ** TRAIN CORRUPTION **\n");
            if (fail_epoch < 0) fail_epoch = epoch + 1;
        } else if (test_acc >= 0 && test_acc < 0.2f && epoch >= 2) {
            printf(" ** TEST CORRUPTION **\n");
            if (fail_epoch < 0) fail_epoch = epoch + 1;
        } else {
            printf("\n");
        }
    }

    free(label_buf);
    free(indices);
    free(test_lab);
    free(test_img);
    free(train_lab);
    free(train_img);
    shm_free(batch_buf);
    shm_free(b2);
    shm_free(w2);
    shm_free(b1);
    shm_free(w1);

    if (fail_epoch > 0) {
        printf("  Result: FAIL (corruption at epoch %d)\n", fail_epoch);
        return 1;
    }
    printf("  Result: PASS\n");
    return 0;
}


/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char *argv[]) {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    int epochs = 10;
    if (argc >= 2) epochs = atoi(argv[1]);
    if (epochs <= 0) epochs = 10;

    printf("========================================\n");
    printf("  ch10: VTCM Lab\n");
    printf("========================================\n");

    /* Open FastRPC session */
    struct remote_rpc_control_unsigned_module udata = { .domain = CDSP_DOMAIN_ID, .enable = 1 };
    remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&udata, sizeof(udata));

    remote_handle64 handle;
    char uri[256];
    snprintf(uri, sizeof(uri), "%s&_dom=cdsp", mnist_train_URI);
    int err = mnist_train_open(uri, &handle);
    if (err) {
        printf("FATAL: mnist_train_open failed: 0x%08x\n", (unsigned)err);
        return 1;
    }

    struct remote_rpc_control_latency ldata = { .enable = 1 };
    remote_handle64_control(handle, DSPRPC_CONTROL_LATENCY, &ldata, sizeof(ldata));

    rpcmem_init();
    sem_init(&g_done, 0, 0);

    /* Create dspqueue */
    dspqueue_t queue;
    err = dspqueue_create(CDSP_DOMAIN_ID, 0, 4096, 4096,
                          arm_packet_cb, arm_error_cb, NULL, &queue);
    if (err) {
        printf("FATAL: dspqueue_create: 0x%08x\n", (unsigned)err);
        return 1;
    }

    uint64_t dsp_queue_id;
    dspqueue_export(queue, &dsp_queue_id);
    err = mnist_train_start(handle, dsp_queue_id);
    if (err) {
        printf("FATAL: mnist_train_start: 0x%08x\n", (unsigned)err);
        return 1;
    }
    printf("DSP session ready (VTCM allocated)\n");

    /* Run experiments */
    int total_fails = 0;

    total_fails += test_basic(queue);
    total_fails += test_persist(queue);
    total_fails += test_hvx_write(queue);
    total_fails += test_scalar_write(queue);
    total_fails += test_sgd(queue);
    total_fails += test_l2_stale(queue);
    total_fails += test_l2_inval(queue);
    total_fails += test_increment(queue);
    total_fails += test_increment_sync(queue);
    total_fails += test_increment_flush(queue);
    total_fails += test_increment_nocache(queue);
    total_fails += test_scalar_write_inval(queue);

    /* Training + SYNC variants — select via CLI arg 2 (default: run all) */
    int sync_mode = -1;  /* -1 = run all */
    if (argc >= 3) sync_mode = atoi(argv[2]);

    if (sync_mode == -1 || sync_mode == 0)
        total_fails += test_training_sync_variant(queue, epochs,
            OP_LAB_TRAIN_SYNC_NOOP, "SYNC no-op (read w1[0] only)");
    if (sync_mode == -1 || sync_mode == 1)
        total_fails += test_training_sync_variant(queue, epochs,
            OP_LAB_TRAIN_SYNC_SMALL, "SYNC small (w2 only, 16KB→heap)");
    if (sync_mode == -1 || sync_mode == 2)
        total_fails += test_training_sync_variant(queue, epochs,
            OP_LAB_TRAIN_SYNC_MCONLY, "SYNC large (all weights, 400KB→heap)");
    if (sync_mode == -1 || sync_mode == 3)
        total_fails += test_training_sync_variant(queue, epochs,
            OP_LAB_TRAIN_SYNC_NOOP_EVAL, "SYNC no-op + ARM cpu_evaluate (proves idle=corrupt)");
    if (sync_mode == -1 || sync_mode == 4)
        total_fails += test_training_sync_variant(queue, epochs,
            OP_LAB_TRAIN_SYNC_DSPEVAL, "SYNC memcpy + DSP busy-work (proves busy=safe)");

    /* Cleanup */
    printf("\n========================================\n");
    printf("  Total: %d experiment%s failed\n",
           total_fails, total_fails == 1 ? "" : "s");
    printf("========================================\n");

    uint64 process_time;
    mnist_train_stop(handle, &process_time);
    dspqueue_close(queue);
    rpcmem_deinit();
    mnist_train_close(handle);
    sem_destroy(&g_done);

    return total_fails > 0 ? 1 : 0;
}
