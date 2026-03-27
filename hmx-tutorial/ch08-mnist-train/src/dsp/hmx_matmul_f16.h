#ifndef HMX_MATMUL_F16_H
#define HMX_MATMUL_F16_H

#include <stddef.h>
#include "hexkl_micro.h"
#include "HAP_farf.h"
#include <string.h>

/* Global VTCM pointers (set by hexkl_micro_hw_init) */
static uint8_t *g_vtcm_base_f16;
static uint32_t g_vtcm_size_f16;

/* Scratch buffer for transpose operations.
 * Largest transpose: max(128*832, 128*128) = 106496 elements */
#define F16_MAX_SCRATCH (256 * 832)
static _Float16 g_scratch_f16[F16_MAX_SCRATCH] __attribute__((aligned(128)));

/* Temporary output buffer for accumulate mode */
#define F16_MAX_OUT (256 * 832)
static _Float16 g_out_f16[F16_MAX_OUT] __attribute__((aligned(128)));

/* Cache-friendly blocked transpose for f16 matrices.
 * dst[cols x rows] = transpose(src[rows x cols])
 * Uses BLK x BLK tiling for cache locality. */
#define TRANSPOSE_BLK 32
static void blocked_transpose_f16(_Float16 *dst, const _Float16 *src,
                                   uint32_t rows, uint32_t cols) {
    for (uint32_t rb = 0; rb < rows; rb += TRANSPOSE_BLK) {
        uint32_t re = (rb + TRANSPOSE_BLK <= rows) ? rb + TRANSPOSE_BLK : rows;
        for (uint32_t cb = 0; cb < cols; cb += TRANSPOSE_BLK) {
            uint32_t ce = (cb + TRANSPOSE_BLK <= cols) ? cb + TRANSPOSE_BLK : cols;
            for (uint32_t i = rb; i < re; i++)
                for (uint32_t j = cb; j < ce; j++)
                    dst[j * rows + i] = src[i * cols + j];
        }
    }
}

#define F16_TILE 32
#define F16_ALIGN HEXKL_HMX_ACTIVATION_ALIGNMENT  /* 2048 */

/*
 * Core f16-native HMX matmul: C_f16[m x n] = A_f16[m x k] @ B_f16[k x n]
 *
 * Both input and output are _Float16. No f32 conversion at any point.
 * HMX accumulator is f32 internally, but we read it back as f16.
 *
 * Weight-cached: pre-loads ALL weight tiles into VTCM in WH format
 * before the compute loop. Eliminates per-tile DDR reads for B.
 *
 * VTCM layout:
 *   [act_0..act_K-1][staging][readback][wt_cache_0..wt_cache_Nwt-1][hmx_config]
 */
static void hmx_matmul_f16_core(
    uint8_t *vtcm, uint32_t vtcm_size,
    _Float16 *C, const _Float16 *A, const _Float16 *B,
    uint32_t m, uint32_t n, uint32_t k)
{
    const uint32_t K  = (k + F16_TILE - 1) / F16_TILE;
    const uint32_t Nc = (n + F16_TILE - 1) / F16_TILE;
    const uint32_t Nwt = K * Nc;

    const uint32_t staging_off  = F16_ALIGN * K;
    const uint32_t readback_off = F16_ALIGN * (K + 1);
    const uint32_t wt_base      = F16_ALIGN * (K + 2);

    uint32_t needed = wt_base + Nwt * F16_ALIGN + hexkl_micro_hmx_config_size();
    if (needed > vtcm_size) {
        FARF(ERROR, "f16: VTCM too small: need %u, have %u", needed, vtcm_size);
        return;
    }

    uint32_t cfg_off = vtcm_size - hexkl_micro_hmx_config_size();
    hexkl_micro_hmx_setup_acc_read_f16(vtcm, cfg_off);

    /* Pre-cache ALL weight tiles into VTCM in WH format */
    for (uint32_t ct = 0; ct < Nc; ct++)
        for (uint32_t kt = 0; kt < K; kt++)
            hexkl_micro_hmx_rm_to_wh_f16(
                vtcm, wt_base + (ct * K + kt) * F16_ALIGN, B, kt, ct, n);

    /* Compute with cached weights */
    for (uint32_t row = 0; row < m; row += F16_TILE) {
        /* Load activation strip into VTCM */
        for (uint32_t i = 0; i < K; i++) {
            hexkl_micro_hmx_copy_submatrix_to_f16(
                vtcm, staging_off, A, row / F16_TILE, i, m, k);
            hexkl_micro_hmx_rm_to_ah_f16(vtcm, F16_ALIGN * i, staging_off);
        }

        for (uint32_t col = 0; col < n; col += F16_TILE) {
            hexkl_micro_hmx_acc_clear_f16();

            for (uint32_t i = 0; i < K; i++) {
                uint32_t wt_off = wt_base + ((col / F16_TILE) * K + i) * F16_ALIGN;
                hexkl_micro_hmx_mm_f16(vtcm, F16_ALIGN * i, wt_off);
            }

            /* Read accumulator as f16 and copy to DDR as f16 */
            hexkl_micro_hmx_acc_read_f16(vtcm, cfg_off, readback_off);
            hexkl_micro_hmx_ah_to_rm_f16(vtcm, staging_off, readback_off);
            hexkl_micro_hmx_copy_f16_to_submatrix(
                vtcm, staging_off, C, row / F16_TILE, col / F16_TILE, m, n);
        }
    }
}

/*
 * HMX matmul with pre-cached WH weights already in VTCM.
 * C_f16[m x n] = A_f16[m x k] @ B_wh (already in WH tiles at wh_off)
 *
 * All offsets are relative to vtcm_base (the overall VTCM base pointer).
 * The WH tiles are stored at: wh_off + (ct * K + kt) * F16_ALIGN
 * where ct = col_tile, kt = k_tile, K = ceil(k/32)
 *
 * Workspace region: [act AH tiles][staging][readback] + config at end
 */
static void hmx_matmul_f16_wh_cached(
    uint8_t *vtcm_base,           /* overall VTCM base (ctx->vtcm_base) */
    uint32_t ws_off,              /* workspace offset from vtcm_base */
    uint32_t ws_size,             /* workspace size */
    _Float16 *C, const _Float16 *A,
    uint32_t wh_off,              /* WH cache offset from vtcm_base */
    uint32_t m, uint32_t n, uint32_t k)
{
    const uint32_t K  = (k + F16_TILE - 1) / F16_TILE;
    const uint32_t Nc = (n + F16_TILE - 1) / F16_TILE;

    /* Workspace layout (all offsets from vtcm_base):
     * [act_0..act_K-1][staging][readback] ... [config] */
    const uint32_t act_base     = ws_off;
    const uint32_t staging_off  = ws_off + F16_ALIGN * K;
    const uint32_t readback_off = ws_off + F16_ALIGN * (K + 1);

    uint32_t needed = F16_ALIGN * (K + 2) + hexkl_micro_hmx_config_size();
    if (needed > ws_size) {
        FARF(ERROR, "f16 wh_cached: workspace too small: need %u, have %u",
             needed, ws_size);
        return;
    }

    uint32_t cfg_off = ws_off + ws_size - hexkl_micro_hmx_config_size();
    hexkl_micro_hmx_setup_acc_read_f16(vtcm_base, cfg_off);

    /* NO weight conversion -- weights already in WH at wh_off */

    for (uint32_t row = 0; row < m; row += F16_TILE) {
        /* Load activation strip into workspace */
        for (uint32_t i = 0; i < K; i++) {
            hexkl_micro_hmx_copy_submatrix_to_f16(
                vtcm_base, staging_off, A, row / F16_TILE, i, m, k);
            hexkl_micro_hmx_rm_to_ah_f16(
                vtcm_base, act_base + F16_ALIGN * i, staging_off);
        }

        for (uint32_t col = 0; col < n; col += F16_TILE) {
            hexkl_micro_hmx_acc_clear_f16();

            for (uint32_t i = 0; i < K; i++) {
                /* Reference pre-cached WH tile directly */
                uint32_t wt_off = wh_off + ((col / F16_TILE) * K + i) * F16_ALIGN;
                hexkl_micro_hmx_mm_f16(
                    vtcm_base, act_base + F16_ALIGN * i, wt_off);
            }

            /* Read accumulator as f16 and copy out */
            hexkl_micro_hmx_acc_read_f16(vtcm_base, cfg_off, readback_off);
            hexkl_micro_hmx_ah_to_rm_f16(vtcm_base, staging_off, readback_off);
            hexkl_micro_hmx_copy_f16_to_submatrix(
                vtcm_base, staging_off, C,
                row / F16_TILE, col / F16_TILE, m, n);
        }
    }
}

/*
 * f16 matmul dispatch with transpose/accumulate support.
 *
 * transpose=0: C = A @ B        (A[m x k], B[k x n])
 * transpose=1: C = A @ B^T      (A[m x k], B[n x k])
 * transpose=2: C += A^T @ B     (A[k x m], B[k x n])
 */
static void hmx_matmul_f16_dispatch(
    uint8_t *vtcm, uint32_t vtcm_size,
    _Float16 *C, const _Float16 *A, const _Float16 *B,
    uint32_t m, uint32_t n, uint32_t k,
    uint32_t transpose, uint32_t accumulate)
{
    switch (transpose) {
    case 0:  /* C = A @ B */
        if (accumulate) {
            hmx_matmul_f16_core(vtcm, vtcm_size, g_out_f16, A, B, m, n, k);
            for (uint32_t i = 0; i < m * n; i++)
                C[i] = (_Float16)((float)C[i] + (float)g_out_f16[i]);
        } else {
            hmx_matmul_f16_core(vtcm, vtcm_size, C, A, B, m, n, k);
        }
        break;

    case 1:  /* C = A @ B^T  (B is [n x k], transpose to [k x n]) */
    {
        blocked_transpose_f16(g_scratch_f16, B, n, k);
        if (accumulate) {
            hmx_matmul_f16_core(vtcm, vtcm_size, g_out_f16, A, g_scratch_f16, m, n, k);
            for (uint32_t i = 0; i < m * n; i++)
                C[i] = (_Float16)((float)C[i] + (float)g_out_f16[i]);
        } else {
            hmx_matmul_f16_core(vtcm, vtcm_size, C, A, g_scratch_f16, m, n, k);
        }
        break;
    }

    case 2:  /* C += A^T @ B  (A is [k x m], transpose to [m x k]) */
    {
        blocked_transpose_f16(g_scratch_f16, A, k, m);
        hmx_matmul_f16_core(vtcm, vtcm_size, g_out_f16, g_scratch_f16, B, m, n, k);
        for (uint32_t i = 0; i < m * n; i++)
            C[i] = (_Float16)((float)C[i] + (float)g_out_f16[i]);
        break;
    }

    default:
        FARF(ERROR, "f16: unknown transpose mode %u", transpose);
        break;
    }
}

/*
 * Fused: dW = A^T @ B, then W -= lr * dW (tile by tile in VTCM).
 *
 * Computes the gradient dW and immediately applies the SGD update
 * to the weight matrix W, eliminating the need for a separate dW buffer.
 *
 * A is [k x m], B is [k x n], W is [m x n].
 * Equivalent to: dW = A^T @ B; W -= lr * dW;
 *
 * W must be writable (shared memory), and is updated in-place.
 */
static void hmx_fused_grad_sgd_f16(
    uint8_t *vtcm, uint32_t vtcm_size,
    _Float16 *W, const _Float16 *A, const _Float16 *B,
    uint32_t m, uint32_t n, uint32_t k, float lr)
{
    /* Transpose A[k x m] to A_t[m x k] */
    blocked_transpose_f16(g_scratch_f16, A, k, m);

    /* Now compute dW = A_t[m x k] @ B[k x n], and fuse W -= lr * dW */
    const uint32_t K  = (k + F16_TILE - 1) / F16_TILE;
    const uint32_t Nc = (n + F16_TILE - 1) / F16_TILE;
    const uint32_t Nwt = K * Nc;

    const uint32_t staging_off  = F16_ALIGN * K;
    const uint32_t readback_off = F16_ALIGN * (K + 1);
    const uint32_t w_tile_off   = F16_ALIGN * (K + 2);  /* for loading W tile */
    const uint32_t wt_base      = F16_ALIGN * (K + 3);  /* cached B WH tiles */

    uint32_t needed = wt_base + Nwt * F16_ALIGN + hexkl_micro_hmx_config_size();
    if (needed > vtcm_size) {
        FARF(ERROR, "f16 fused: VTCM too small: need %u, have %u", needed, vtcm_size);
        return;
    }

    uint32_t cfg_off = vtcm_size - hexkl_micro_hmx_config_size();
    hexkl_micro_hmx_setup_acc_read_f16(vtcm, cfg_off);

    /* Pre-cache ALL B tiles into VTCM in WH format */
    for (uint32_t ct = 0; ct < Nc; ct++)
        for (uint32_t kt = 0; kt < K; kt++)
            hexkl_micro_hmx_rm_to_wh_f16(
                vtcm, wt_base + (ct * K + kt) * F16_ALIGN, B, kt, ct, n);

    /* Compute dW tiles and fuse SGD */
    for (uint32_t row = 0; row < m; row += F16_TILE) {
        /* Load activation strip (A_t) */
        for (uint32_t i = 0; i < K; i++) {
            hexkl_micro_hmx_copy_submatrix_to_f16(
                vtcm, staging_off, g_scratch_f16, row / F16_TILE, i, m, k);
            hexkl_micro_hmx_rm_to_ah_f16(vtcm, F16_ALIGN * i, staging_off);
        }

        for (uint32_t col = 0; col < n; col += F16_TILE) {
            hexkl_micro_hmx_acc_clear_f16();

            for (uint32_t i = 0; i < K; i++) {
                uint32_t woff = wt_base + ((col / F16_TILE) * K + i) * F16_ALIGN;
                hexkl_micro_hmx_mm_f16(vtcm, F16_ALIGN * i, woff);
            }

            /* Read dW tile from accumulator to VTCM */
            hexkl_micro_hmx_acc_read_f16(vtcm, cfg_off, readback_off);
            hexkl_micro_hmx_ah_to_rm_f16(vtcm, staging_off, readback_off);

            /* Load corresponding W tile from DDR to VTCM */
            hexkl_micro_hmx_copy_submatrix_to_f16(
                vtcm, w_tile_off, W, row / F16_TILE, col / F16_TILE, m, n);

            /* Fused SGD update in VTCM: W_tile -= lr * dW_tile */
            _Float16 *dw_tile = (_Float16 *)(vtcm + staging_off);
            _Float16 *w_tile  = (_Float16 *)(vtcm + w_tile_off);
            uint32_t tile_rows = (row + F16_TILE <= m) ? F16_TILE : (m - row);
            uint32_t tile_cols = (col + F16_TILE <= n) ? F16_TILE : (n - col);
            for (uint32_t r = 0; r < tile_rows; r++)
                for (uint32_t c = 0; c < tile_cols; c++) {
                    uint32_t idx = r * F16_TILE + c;
                    w_tile[idx] = (_Float16)((float)w_tile[idx] - lr * (float)dw_tile[idx]);
                }

            /* Write updated W tile back to DDR */
            hexkl_micro_hmx_copy_f16_to_submatrix(
                vtcm, w_tile_off, W, row / F16_TILE, col / F16_TILE, m, n);
        }
    }
}

#endif /* HMX_MATMUL_F16_H */
