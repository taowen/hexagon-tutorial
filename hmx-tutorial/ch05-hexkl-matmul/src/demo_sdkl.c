/*
 * demo_sdkl.c -- ARM-side HexKL SDKL matmul demo
 *
 * Experiment 1: Auto matmul
 *   sdkl_npu_mm_f32f16_f32() -- one call does everything.
 *   X is f32 row-major, W is f16 WH layout, A is f32 row-major.
 *
 * Experiment 2: Manual-transform matmul (the "LLM optimization path")
 *   All buffers allocated with sdkl_npu_alloc (shared memory).
 *   X transformed to AH layout, W transformed to WH layout.
 *   sdkl_npu_mm_f16() -- all inputs/outputs in HMX-native layout.
 *   Output converted back to row-major for verification.
 *
 *   Key insight: in LLM inference, weights are transformed ONCE at model
 *   load time, then reused across many forward passes. The transform cost
 *   is amortized over thousands of calls.
 */

#include "remote.h"
#include "sdkl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define N_ROW   256
#define N_COL   1024
#define N_INNER 4096

#define SDKL_CHECK(call)                                                  \
    do {                                                                  \
        int _rc = (call);                                                 \
        if (_rc != 0) {                                                   \
            printf("[SDKL] ERROR: %s returned %d (0x%x)\n",              \
                   #call, _rc, _rc);                                      \
            ret = _rc;                                                    \
            goto cleanup;                                                 \
        }                                                                 \
    } while (0)

/*
 * Reference matmul on CPU: A = X * W^T
 *   X:  f32  [n_row][n_inner]
 *   W:  f16  [n_col][n_inner]
 *   A:  f32  [n_row][n_col]
 */
static void cpu_matmul_f32f16(int n_row, int n_col, int n_inner,
                              float *A, const float *X, const _Float16 *W)
{
    for (int r = 0; r < n_row; r++) {
        for (int c = 0; c < n_col; c++) {
            float acc = 0.0f;
            for (int k = 0; k < n_inner; k++) {
                acc += X[r * n_inner + k] * (float)W[c * n_inner + k];
            }
            A[r * n_col + c] = acc;
        }
    }
}

static double elapsed_ms(struct timeval *start, struct timeval *end)
{
    return (end->tv_sec - start->tv_sec) * 1000.0 +
           (end->tv_usec - start->tv_usec) / 1000.0;
}

/* ---------- main ---------- */

int main(void)
{
    int ret = 0;
    char version[SDKL_VERSION_STR_LEN] = {0};

    /* Experiment 1 buffers */
    float    *X_f32     = NULL;
    _Float16 *W_f16_cpu = NULL;
    _Float16 *W_f16_npu = NULL;
    float    *A_ref     = NULL;
    float    *A_npu     = NULL;

    /* Experiment 2 buffers */
    _Float16 *X2_f16    = NULL;
    _Float16 *W2_f16    = NULL;
    _Float16 *A2_f16    = NULL;

    struct timeval t0, t1;

    printf("[SDKL] === HexKL SDKL matmul demo ===\n");
    printf("[SDKL] Dimensions: N_ROW=%d  N_COL=%d  N_INNER=%d\n",
           N_ROW, N_COL, N_INNER);

    /* ---- initialise NPU session ---- */
    SDKL_CHECK(sdkl_npu_initialize(CDSP_DOMAIN_ID, NULL, NULL));
    printf("[SDKL] NPU initialised on CDSP\n");

    SDKL_CHECK(sdkl_npu_get_version(CDSP_DOMAIN_ID, version));
    printf("[SDKL] SDKL version: %s\n", version);

    /* ==================================================================
     * Experiment 1: Auto matmul -- sdkl_npu_mm_f32f16_f32
     *   X is f32, W is f16 (WH layout), output A is f32.
     *   SDKL handles activation layout transform internally.
     * ================================================================== */
    printf("\n");
    printf("[SDKL] ========================================\n");
    printf("[SDKL] Experiment 1: Auto matmul (f32f16_f32)\n");
    printf("[SDKL] ========================================\n");

    /* ---- allocate ---- */
    size_t X_bytes = (size_t)N_ROW * N_INNER * sizeof(float);
    size_t W_bytes = (size_t)N_COL * N_INNER * sizeof(_Float16);
    size_t A_bytes = (size_t)N_ROW * N_COL   * sizeof(float);

    X_f32     = (float *)   malloc(X_bytes);
    W_f16_cpu = (_Float16 *)malloc(W_bytes);
    A_ref     = (float *)   malloc(A_bytes);
    A_npu     = (float *)   malloc(A_bytes);

    if (!X_f32 || !W_f16_cpu || !A_ref || !A_npu) {
        printf("[SDKL] ERROR: malloc failed\n");
        ret = -1;
        goto cleanup;
    }

    /* W for NPU must live in shared memory */
    SDKL_CHECK(sdkl_npu_alloc(W_bytes, (void **)&W_f16_npu));
    printf("[SDKL] Allocated X (%zu B), W (%zu B), A (%zu B)\n",
           X_bytes, W_bytes, A_bytes);

    /* ---- initialise data ---- */
    srand(42);
    for (size_t i = 0; i < (size_t)N_ROW * N_INNER; i++)
        X_f32[i] = (float)rand() / (float)RAND_MAX;

    for (size_t i = 0; i < (size_t)N_COL * N_INNER; i++)
        W_f16_cpu[i] = (_Float16)((float)rand() / (float)RAND_MAX);

    /* copy W to NPU-visible buffer (still row-major at this point) */
    memcpy(W_f16_npu, W_f16_cpu, W_bytes);

    memset(A_ref, 0, A_bytes);
    memset(A_npu, 0, A_bytes);

    /* ---- CPU reference matmul ---- */
    printf("[SDKL] Running CPU reference matmul ...\n");
    gettimeofday(&t0, NULL);
    cpu_matmul_f32f16(N_ROW, N_COL, N_INNER, A_ref, X_f32, W_f16_cpu);
    gettimeofday(&t1, NULL);
    double cpu_ms = elapsed_ms(&t0, &t1);
    printf("[SDKL] CPU matmul done: %.2f ms\n", cpu_ms);

    /* ---- convert weight layout for HMX ---- */
    printf("[SDKL] Converting weights to WH layout ...\n");
    gettimeofday(&t0, NULL);
    SDKL_CHECK(sdkl_cpu_rm_to_wh_f16_inplace(N_COL, N_INNER, W_f16_npu));
    gettimeofday(&t1, NULL);
    double layout_ms = elapsed_ms(&t0, &t1);
    printf("[SDKL] Weight layout conversion done: %.2f ms\n", layout_ms);

    /* ---- NPU matmul ---- */
    printf("[SDKL] Running NPU matmul (f32f16_f32) ...\n");
    gettimeofday(&t0, NULL);
    SDKL_CHECK(sdkl_npu_mm_f32f16_f32(CDSP_DOMAIN_ID,
                                       N_ROW, N_COL, N_INNER,
                                       A_npu, X_f32, W_f16_npu));
    gettimeofday(&t1, NULL);
    double exp1_npu_ms = elapsed_ms(&t0, &t1);
    printf("[SDKL] NPU matmul done: %.2f ms\n", exp1_npu_ms);

    /* ---- verify experiment 1 ---- */
    {
        int mismatches = 0;
        for (int i = 0; i < N_ROW * N_COL; i++) {
            float ref = A_ref[i];
            float npu = A_npu[i];
            float diff = fabsf(ref - npu);
            float tol  = fabsf(ref / 1000.0f);  /* 0.1% of ref */
            if (diff > tol) {
                if (mismatches < 5) {
                    printf("[SDKL]   mismatch [%d]: ref=%.6f  npu=%.6f  diff=%.6f  tol=%.6f\n",
                           i, ref, npu, diff, tol);
                }
                mismatches++;
            }
        }
        if (mismatches == 0)
            printf("[SDKL] Experiment 1 verification PASSED\n");
        else
            printf("[SDKL] Experiment 1 verification FAILED: %d / %d mismatches\n",
                   mismatches, N_ROW * N_COL);
    }

    /* ==================================================================
     * Experiment 2: Manual-transform matmul -- sdkl_npu_mm_f16
     *   All f16, all in HMX-native layouts (AH/WH).
     *   This is the path hexagon-mlir uses for LLM inference:
     *   weights transformed once at load, reused across many calls.
     * ================================================================== */
    printf("\n");
    printf("[SDKL] ========================================\n");
    printf("[SDKL] Experiment 2: Manual-transform matmul (f16)\n");
    printf("[SDKL] ========================================\n");

    {
        size_t X2_bytes = (size_t)N_ROW * N_INNER * sizeof(_Float16);
        size_t W2_bytes = (size_t)N_COL * N_INNER * sizeof(_Float16);
        size_t A2_bytes = (size_t)N_ROW * N_COL   * sizeof(_Float16);

        /* All buffers must be in shared memory for sdkl_npu_mm_f16 */
        SDKL_CHECK(sdkl_npu_alloc(X2_bytes, (void **)&X2_f16));
        SDKL_CHECK(sdkl_npu_alloc(W2_bytes, (void **)&W2_f16));
        SDKL_CHECK(sdkl_npu_alloc(A2_bytes, (void **)&A2_f16));
        printf("[SDKL] Allocated shared memory: X2 (%zu B), W2 (%zu B), A2 (%zu B)\n",
               X2_bytes, W2_bytes, A2_bytes);

        /* Fill X2 with the same data as experiment 1, converted to f16 */
        for (size_t i = 0; i < (size_t)N_ROW * N_INNER; i++)
            X2_f16[i] = (_Float16)X_f32[i];

        /* Copy W from the original row-major CPU buffer */
        memcpy(W2_f16, W_f16_cpu, W2_bytes);

        memset(A2_f16, 0, A2_bytes);

        /* Transform W to WH layout */
        printf("[SDKL] Converting W2 to WH layout ...\n");
        gettimeofday(&t0, NULL);
        SDKL_CHECK(sdkl_cpu_rm_to_wh_f16_inplace(N_COL, N_INNER, W2_f16));
        gettimeofday(&t1, NULL);
        double w_layout_ms = elapsed_ms(&t0, &t1);
        printf("[SDKL] W2 layout conversion done: %.2f ms\n", w_layout_ms);

        /* Transform X to AH layout */
        printf("[SDKL] Converting X2 to AH layout ...\n");
        gettimeofday(&t0, NULL);
        SDKL_CHECK(sdkl_cpu_rm_to_ah_f16_inplace(N_ROW, N_INNER, X2_f16));
        gettimeofday(&t1, NULL);
        double x_layout_ms = elapsed_ms(&t0, &t1);
        printf("[SDKL] X2 layout conversion done: %.2f ms\n", x_layout_ms);

        /* NPU matmul: all AH/WH layout, output in AH layout */
        printf("[SDKL] Running NPU matmul (mm_f16) ...\n");
        gettimeofday(&t0, NULL);
        SDKL_CHECK(sdkl_npu_mm_f16(CDSP_DOMAIN_ID,
                                    N_ROW, N_COL, N_INNER,
                                    A2_f16, X2_f16, W2_f16));
        gettimeofday(&t1, NULL);
        double exp2_npu_ms = elapsed_ms(&t0, &t1);
        printf("[SDKL] NPU matmul done: %.2f ms\n", exp2_npu_ms);

        /* Convert output from AH back to row-major */
        printf("[SDKL] Converting A2 from AH to row-major ...\n");
        gettimeofday(&t0, NULL);
        SDKL_CHECK(sdkl_cpu_ah_to_rm_f16_inplace(N_ROW, N_COL, A2_f16));
        gettimeofday(&t1, NULL);
        double a_layout_ms = elapsed_ms(&t0, &t1);
        printf("[SDKL] A2 layout conversion done: %.2f ms\n", a_layout_ms);

        /* Verify against experiment 1 CPU reference (f32 vs f16 -- use wider tolerance) */
        int mismatches = 0;
        float max_rel_err = 0.0f;
        for (int i = 0; i < N_ROW * N_COL; i++) {
            float ref = A_ref[i];
            float val = (float)A2_f16[i];
            float diff = fabsf(ref - val);
            float denom = fabsf(ref) > 1e-6f ? fabsf(ref) : 1e-6f;
            float rel = diff / denom;
            if (rel > max_rel_err) max_rel_err = rel;
            /* f16 accumulation loses more precision -- use 2% tolerance */
            if (rel > 0.02f) {
                if (mismatches < 5) {
                    printf("[SDKL]   mismatch [%d]: ref=%.6f  f16=%.6f  rel_err=%.4f%%\n",
                           i, ref, val, rel * 100.0f);
                }
                mismatches++;
            }
        }
        printf("[SDKL] Max relative error vs f32 reference: %.4f%%\n", max_rel_err * 100.0f);
        if (mismatches == 0)
            printf("[SDKL] Experiment 2 verification PASSED\n");
        else
            printf("[SDKL] Experiment 2 verification FAILED: %d / %d mismatches (>2%% tol)\n",
                   mismatches, N_ROW * N_COL);

        /* ---- summary ---- */
        printf("\n");
        printf("[SDKL] ------ summary ------\n");
        printf("[SDKL] CPU reference        : %8.2f ms\n", cpu_ms);
        printf("[SDKL] Exp 1 (f32f16_f32)   : %8.2f ms  (includes internal AH transform)\n", exp1_npu_ms);
        printf("[SDKL] Exp 2 W->WH transform: %8.2f ms  (one-time cost at model load)\n", w_layout_ms);
        printf("[SDKL] Exp 2 X->AH transform: %8.2f ms  (per-inference cost)\n", x_layout_ms);
        printf("[SDKL] Exp 2 mm_f16 matmul  : %8.2f ms  (pure HMX compute)\n", exp2_npu_ms);
        printf("[SDKL] Exp 2 A AH->RM       : %8.2f ms  (output conversion)\n", a_layout_ms);
        printf("[SDKL] Exp 2 total (no W)   : %8.2f ms  (inference hot path)\n",
               x_layout_ms + exp2_npu_ms + a_layout_ms);
        if (exp1_npu_ms > 0.0)
            printf("[SDKL] Speedup (CPU / Exp1 NPU): %.1fx\n", cpu_ms / exp1_npu_ms);
        if (exp2_npu_ms > 0.0)
            printf("[SDKL] Speedup (CPU / Exp2 NPU): %.1fx\n", cpu_ms / exp2_npu_ms);
        printf("[SDKL] ========================\n");
    }

cleanup:
    if (A2_f16)    sdkl_npu_free(A2_f16);
    if (W2_f16)    sdkl_npu_free(W2_f16);
    if (X2_f16)    sdkl_npu_free(X2_f16);
    if (W_f16_npu) sdkl_npu_free(W_f16_npu);
    free(A_npu);
    free(A_ref);
    free(W_f16_cpu);
    free(X_f32);

    sdkl_npu_finalize(CDSP_DOMAIN_ID);
    printf("[SDKL] Done (ret=%d)\n", ret);
    return ret;
}
