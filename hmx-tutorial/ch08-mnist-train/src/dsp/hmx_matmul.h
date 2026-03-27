/*
 * HMX f16 Matmul -- weight-cached (L1 optimization)
 *
 * Extracted from mnist_train_dsp.c for readability.
 * All functions are static -- this header is intended to be included
 * by exactly one translation unit.
 *
 * Optimization: pre-load ALL weight tiles into VTCM in WH format
 * before the compute loop. This eliminates redundant DDR reads and
 * is the only optimization with measurable benefit at MNIST scale
 * (~39% on Fwd L1, ~5% on total training time).
 *
 * See ch05 demo_llama_hmx.c for the benchmark that proved inline ASM
 * and bias=mxmem2 add ~0% at this matrix scale.
 *
 * Requires hvx_matmul.h to be included first (for g_scratch, MAX_SCRATCH).
 */
#ifndef HMX_MATMUL_H
#define HMX_MATMUL_H

#ifdef USE_HMX

#include <stddef.h>        /* size_t -- needed before hexkl headers */
#include "hexkl_micro.h"

/* Global VTCM pointers for HMX path */
static uint8_t *g_vtcm_base;
static uint32_t g_vtcm_size;

/* Static f16 buffers for f32->f16 conversion.
 * Largest matrix in training: batch=256, hidden=128, input=784
 *   A max: 256*800 = 204800, B max: 128*800 = 102400 */
#define HMX_MAX_ELEMS (256 * 800)
static _Float16 g_a_f16[HMX_MAX_ELEMS] __attribute__((aligned(128)));
static _Float16 g_b_f16[HMX_MAX_ELEMS] __attribute__((aligned(128)));
static float    g_hmx_out[HMX_MAX_ELEMS] __attribute__((aligned(128)));  /* temp output for accumulate mode */

/*
 * Core HMX tiled matmul: C[m x n] = A[m x k] @ B[k x n]
 *
 * A and B are f32 inputs; internally converted to f16 for HMX.
 * Output C is f32.
 *
 * Weight-cached: pre-load ALL weight tiles into VTCM in WH format
 * before the row loop, eliminating DDR reads in the inner loop.
 *
 * VTCM layout:
 *   [act 0..K-1][staging][readback][wt_0_0..wt_Nc*K][hmx_config]
 */
static void hmx_matmul_f16(
    uint8_t *vtcm_base, uint32_t vtcm_size,
    float *C, const float *A, const float *B,
    uint32_t m, uint32_t n, uint32_t k)
{
    /* Convert f32 inputs to f16 (HVX vectorized) */
    hvx_f32_to_f16(g_a_f16, A, m * k);
    hvx_f32_to_f16(g_b_f16, B, k * n);

    const uint32_t K  = (k + HEXKL_HMX_F16_BLOCK_N_INNER - 1) / HEXKL_HMX_F16_BLOCK_N_INNER;
    const uint32_t Nc = (n + HEXKL_HMX_F16_BLOCK_N_COL - 1) / HEXKL_HMX_F16_BLOCK_N_COL;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    /* VTCM offsets */
    const uint32_t staging_off  = ALIGN * K;
    const uint32_t readback_off = ALIGN * (K + 1);
    const uint32_t wt_base      = ALIGN * (K + 2);

    /* HMX config at end of VTCM */
    uint32_t cfg_off = vtcm_size - hexkl_micro_hmx_config_size();
    hexkl_micro_hmx_setup_acc_read_f16(vtcm_base, cfg_off);

    /* Pre-cache ALL weight tiles into VTCM in WH format */
    for (uint32_t ct = 0; ct < Nc; ct++)
        for (uint32_t kt = 0; kt < K; kt++)
            hexkl_micro_hmx_rm_to_wh_f16(
                vtcm_base, wt_base + (ct * K + kt) * ALIGN,
                g_b_f16, kt, ct, n);

    /* Row loop */
    for (uint32_t row = 0; row < m; row += HEXKL_HMX_F16_BLOCK_N_ROW) {

        /* Load activation strip into VTCM and convert to AH layout */
        for (uint32_t i = 0; i < K; i++) {
            hexkl_micro_hmx_copy_submatrix_to_f16(
                vtcm_base, staging_off,
                g_a_f16, row / HEXKL_HMX_F16_BLOCK_N_ROW, i,
                m, k);
            hexkl_micro_hmx_rm_to_ah_f16(
                vtcm_base, ALIGN * i, staging_off);
        }

        /* Column loop */
        for (uint32_t col = 0; col < n; col += HEXKL_HMX_F16_BLOCK_N_COL) {
            uint32_t ct = col / HEXKL_HMX_F16_BLOCK_N_COL;

            hexkl_micro_hmx_acc_clear_f16();

            /* Accumulate across K dimension using cached weight tiles */
            for (uint32_t i = 0; i < K; i++) {
                hexkl_micro_hmx_mm_f16(
                    vtcm_base, ALIGN * i,
                    wt_base + (ct * K + i) * ALIGN);
            }

            /* Read accumulator and convert back to row-major f32 */
            hexkl_micro_hmx_acc_read_f16(
                vtcm_base, cfg_off, readback_off);
            hexkl_micro_hmx_ah_to_rm_f16(
                vtcm_base, staging_off, readback_off);
            hexkl_micro_hmx_copy_f16_to_f32_submatrix(
                vtcm_base, staging_off, C,
                row / HEXKL_HMX_F16_BLOCK_N_ROW,
                ct,
                m, n);
        }
    }
}

/*
 * HMX matmul dispatch with transpose/accumulate support.
 * Handles the same 3 transpose modes as the HVX do_matmul.
 */
static void hmx_matmul_dispatch(
    uint8_t *vtcm_base, uint32_t vtcm_size,
    float *C, const float *A, const float *B,
    uint32_t m, uint32_t n, uint32_t k,
    uint32_t transpose, uint32_t accumulate)
{
    switch (transpose) {
    case 0:  /* C = A @ B  (A is [m x k], B is [k x n]) */
        if (accumulate) {
            hmx_matmul_f16(vtcm_base, vtcm_size, g_hmx_out, A, B, m, n, k);
            for (uint32_t i = 0; i < m * n; i++) C[i] += g_hmx_out[i];
        } else {
            hmx_matmul_f16(vtcm_base, vtcm_size, C, A, B, m, n, k);
        }
        break;

    case 1:  /* C = A @ B^T  (A is [m x k], B is [n x k] -> transpose to [k x n]) */
    {
        for (uint32_t p = 0; p < k; p++)
            for (uint32_t j = 0; j < n; j++)
                g_scratch[p * n + j] = B[j * k + p];
        hmx_matmul_f16(vtcm_base, vtcm_size, C, A, g_scratch, m, n, k);
        break;
    }

    case 2:  /* C += A^T @ B  (A is [k x m] -> transpose to [m x k], B is [k x n]) */
    {
        for (uint32_t i = 0; i < m; i++)
            for (uint32_t p = 0; p < k; p++)
                g_scratch[i * k + p] = A[p * m + i];
        hmx_matmul_f16(vtcm_base, vtcm_size, g_hmx_out, g_scratch, B, m, n, k);
        for (uint32_t idx = 0; idx < m * n; idx++) C[idx] += g_hmx_out[idx];
        break;
    }

    default:
        FARF(ERROR, "HMX: unknown transpose mode %u", transpose);
        break;
    }
}

#endif /* USE_HMX */

#endif /* HMX_MATMUL_H */
