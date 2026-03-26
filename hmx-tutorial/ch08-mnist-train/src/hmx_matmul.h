/*
 * HMX f16 Matmul (via HexKL micro API)
 *
 * Extracted from mnist_train_dsp.c for readability.
 * All functions are static -- this header is intended to be included
 * by exactly one translation unit.
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
static _Float16 g_a_f16[HMX_MAX_ELEMS];
static _Float16 g_b_f16[HMX_MAX_ELEMS];
static float    g_hmx_out[HMX_MAX_ELEMS];  /* temp output for accumulate mode */

/*
 * Core HMX tiled matmul: C[m x n] = A[m x k] @ B[k x n]
 *
 * A and B are f32 inputs; internally converted to f16 for HMX.
 * Output C is f32.  Uses the same tiling pattern as ch08 demo_micro.c.
 *
 * VTCM layout:
 *   [act tile 0][act tile 1]...[act tile K-1][weight tile][scratch][...][hmx_config]
 */
static void hmx_matmul_f16(
    uint8_t *vtcm_base, uint32_t vtcm_size,
    float *C, const float *A, const float *B,
    uint32_t m, uint32_t n, uint32_t k)
{
    /* Convert f32 inputs to f16 */
    for (uint32_t i = 0; i < m * k; i++) g_a_f16[i] = (_Float16)A[i];
    for (uint32_t i = 0; i < k * n; i++) g_b_f16[i] = (_Float16)B[i];

    /* Number of 32-wide tiles across the inner dimension */
    const uint32_t row_tiles_in_A =
        (k + HEXKL_HMX_F16_BLOCK_N_INNER - 1) / HEXKL_HMX_F16_BLOCK_N_INNER;

    /* Weight tile sits right after activation tiles in VTCM */
    const uint32_t weight_offset =
        HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A;

    /* HMX config at end of VTCM */
    uint32_t hmx_config_offset = vtcm_size - hexkl_micro_hmx_config_size();
    hexkl_micro_hmx_setup_acc_read_f16(vtcm_base, hmx_config_offset);

    /* Triple-nested tiling loop */
    for (uint32_t row = 0; row < m; row += HEXKL_HMX_F16_BLOCK_N_ROW) {

        /* Load activation strip into VTCM and convert to AH layout */
        for (uint32_t i = 0; i < row_tiles_in_A; i++) {
            hexkl_micro_hmx_copy_submatrix_to_f16(
                vtcm_base,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + i),
                g_a_f16,
                row / HEXKL_HMX_F16_BLOCK_N_ROW,
                i,
                m, k);
            hexkl_micro_hmx_rm_to_ah_f16(
                vtcm_base,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * i,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + i));
        }

        for (uint32_t col = 0; col < n; col += HEXKL_HMX_F16_BLOCK_N_COL) {

            hexkl_micro_hmx_acc_clear_f16();

            /* Accumulate across K dimension */
            for (uint32_t i = 0; i < row_tiles_in_A; i++) {
                hexkl_micro_hmx_rm_to_wh_f16(
                    vtcm_base, weight_offset,
                    g_b_f16, i,
                    col / HEXKL_HMX_F16_BLOCK_N_COL,
                    n);
                hexkl_micro_hmx_mm_f16(
                    vtcm_base,
                    HEXKL_HMX_ACTIVATION_ALIGNMENT * i,
                    weight_offset);
            }

            /* Read accumulator and convert back to row-major f32 */
            hexkl_micro_hmx_acc_read_f16(
                vtcm_base, hmx_config_offset,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + 1));
            hexkl_micro_hmx_ah_to_rm_f16(
                vtcm_base,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + 1));
            hexkl_micro_hmx_copy_f16_to_f32_submatrix(
                vtcm_base,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A,
                C,
                row / HEXKL_HMX_F16_BLOCK_N_ROW,
                col / HEXKL_HMX_F16_BLOCK_N_COL,
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
        for (uint32_t i = 0; i < m * n; i++) C[i] += g_hmx_out[i];
        break;
    }

    default:
        FARF(ERROR, "HMX: unknown transpose mode %u", transpose);
        break;
    }
}

#endif /* USE_HMX */

#endif /* HMX_MATMUL_H */
