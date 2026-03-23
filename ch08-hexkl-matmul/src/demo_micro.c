/*
 * demo_micro.c -- HexKL NPU Micro API: tiled HMX f16 matrix multiply
 *
 * This file demonstrates the lowest level of HMX programming available
 * through HexKL. The programmer manually controls:
 *   - VTCM allocation and layout
 *   - Tile copies from DDR to VTCM
 *   - Layout conversions (row-major <-> AH/WH formats)
 *   - HMX matmul instructions
 *   - Accumulator management
 *
 * The computation is:  Out[N_ROW x N_COL] = X[N_ROW x N_INNER] * W[N_INNER x N_COL]
 * where X and W are f16, and Out is f32.
 *
 * W is stored in row-major order (NOT transposed), so W[k][col].
 */

#include "AEEStdErr.h"
#include <stddef.h>        /* size_t — needed before hexkl headers */
#include "hexkl_micro.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "HAP_farf.h"

/* ------------------------------------------------------------------ */
/*  Test dimensions                                                    */
/* ------------------------------------------------------------------ */
#define N_ROW   32    /* rows of activation / output */
#define N_COL   128   /* cols of weight / output     */
#define N_INNER 64    /* shared inner dimension      */

/*
 * HMX tile sizes (from hexkl_micro.h, repeated here for clarity):
 *   HEXKL_HMX_F16_BLOCK_N_ROW   = 32
 *   HEXKL_HMX_F16_BLOCK_N_COL   = 32
 *   HEXKL_HMX_F16_BLOCK_N_INNER = 32
 *   HEXKL_HMX_ACTIVATION_ALIGNMENT = 2048  (bytes per tile slot in VTCM)
 *
 * Each 32x32 f16 tile occupies 32*32*2 = 2048 bytes = one ACTIVATION_ALIGNMENT slot.
 */

/* ------------------------------------------------------------------ */
/*  Reference matmul (plain C, f16 inputs, f32 output)                */
/* ------------------------------------------------------------------ */
static void reference_matmul(
    size_t n_row,
    size_t n_col,
    size_t n_inner,
    float *restrict out,
    const _Float16 *restrict act,   /* [n_row][n_inner]  */
    const _Float16 *restrict wt     /* [n_inner][n_col]  */
) {
    for (uint32_t r = 0; r < n_row; r++) {
        for (uint32_t c = 0; c < n_col; c++) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < n_inner; k++) {
                acc += (float)act[r * n_inner + k] * (float)wt[k * n_col + c];
            }
            out[r * n_col + c] = acc;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Result comparison (0.1 % relative tolerance, 0.01 absolute floor) */
/* ------------------------------------------------------------------ */
static int check_results_f32(size_t count, const float *ref, const float *got) {
    for (size_t i = 0; i < count; i++) {
        if (isnan(ref[i]) || isinf(got[i])) {
            FARF(ALWAYS, "[MICRO][ERROR] NaN/Inf at index %u: ref=%f got=%f",
                   (unsigned)i, (double)ref[i], (double)got[i]);
            return AEE_EFAILED;
        }
        float diff    = fabsf(ref[i] - got[i]);
        float epsilon = fabsf(ref[i] / 1000.0f);   /* 0.1 % */
        if (diff > epsilon && diff > 0.01f) {
            FARF(ALWAYS, "[MICRO][ERROR] Mismatch at index %u: ref=%f got=%f "
                   "diff=%f eps=%f",
                   (unsigned)i, (double)ref[i], (double)got[i],
                   (double)diff, (double)epsilon);
            return AEE_EFAILED;
        }
    }
    return AEE_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  HMX tiled matmul                                                   */
/*                                                                     */
/*  VTCM layout:                                                       */
/*  [act tile 0][act tile 1]...[act tile K-1][weight tile][scratch][...][hmx_config] */
/*                                                                     */
/*  "act tiles" hold one row-strip of the activation matrix converted  */
/*  to AH (Activation HMX) layout.  There are row_tiles_in_A of them. */
/*                                                                     */
/*  "weight tile" is a single scratch slot where we convert each       */
/*  weight tile to WH (Weight HMX) layout right before the multiply.  */
/*                                                                     */
/*  "scratch" is a second scratch slot used for accumulator readback   */
/*  and AH-to-RM conversion.                                           */
/*                                                                     */
/*  "hmx_config" lives at the very end of VTCM -- the HMX hardware    */
/*  configuration block written once by setup_acc_read.                */
/*                                                                     */
/*  ASCII diagram (offsets in units of ACTIVATION_ALIGNMENT = 2048 B): */
/*                                                                     */
/*    offset 0          K-1     K      K+1                  end        */
/*    +-------+-----+-------+-------+-------+-----+----------+        */
/*    | act 0 | ... | act   | wt    | scratch     |hmx_config|        */
/*    |  (AH) |     | (K-1) | (WH)  | (readback)  |          |        */
/*    +-------+-----+-------+-------+-------+-----+----------+        */
/*                                                                     */
/* ------------------------------------------------------------------ */
int hexkl_micro_matmul_f16f16_f32(
    uint8_t  *vtcm_base,
    uint32_t  vtcm_size,
    size_t    n_row,
    size_t    n_col,
    size_t    n_inner,
    float          *restrict out,
    const _Float16 *restrict act,   /* [n_row][n_inner]  */
    const _Float16 *restrict wt     /* [n_inner][n_col]  */
) {
    if (vtcm_size == 0 || (vtcm_size % HEXKL_HMX_ACTIVATION_ALIGNMENT) != 0) {
        FARF(ALWAYS, "[MICRO][ERROR] Bad VTCM size: 0x%x", (unsigned)vtcm_size);
        return AEE_ENOMEMORY;
    }

    /*
     * How many 32-wide tiles span the inner dimension?
     * e.g. N_INNER=64 => row_tiles_in_A = 2
     */
    const uint32_t row_tiles_in_A =
        (n_inner + HEXKL_HMX_F16_BLOCK_N_INNER - 1) / HEXKL_HMX_F16_BLOCK_N_INNER;

    /*
     * The weight tile sits right after the activation tiles in VTCM.
     * Each tile slot is HEXKL_HMX_ACTIVATION_ALIGNMENT (2048) bytes.
     */
    const uint32_t weight_offset =
        HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A;

    /* ---- Step 1: write HMX configuration to end of VTCM ---- */
    uint32_t hmx_config_offset = vtcm_size - hexkl_micro_hmx_config_size();
    hexkl_micro_hmx_setup_acc_read_f16(vtcm_base, hmx_config_offset);

    /* ---- Step 2: triple-nested tiling loop ---- */

    /* Outer loop: iterate over rows of the output, one tile-row (32) at a time */
    for (uint32_t row = 0; row < n_row; row += HEXKL_HMX_F16_BLOCK_N_ROW) {

        /*
         * Load one horizontal strip of the activation matrix into VTCM.
         * For each k-tile:
         *   1. Copy a 32x32 f16 sub-matrix from DDR into a temporary VTCM slot
         *   2. Convert from row-major to AH (Activation HMX) layout in-place
         *
         * We reuse the slot at (row_tiles_in_A + i) as a staging area for
         * the DDR copy, then write the AH result to slot i.
         */
        for (uint32_t i = 0; i < row_tiles_in_A; i++) {
            /* Copy 32x32 f16 tile from DDR -> VTCM (row-major) */
            hexkl_micro_hmx_copy_submatrix_to_f16(
                vtcm_base,
                /*out_offset=*/  HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + i),
                /*input_matrix=*/act,
                /*tile_row=*/    row / HEXKL_HMX_F16_BLOCK_N_ROW,
                /*tile_col=*/    i,
                /*input_rows=*/  n_row,
                /*input_cols=*/  n_inner
            );
            /* Convert row-major -> AH layout */
            hexkl_micro_hmx_rm_to_ah_f16(
                vtcm_base,
                /*activation_out_offset=*/HEXKL_HMX_ACTIVATION_ALIGNMENT * i,
                /*flat_in_offset=*/       HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + i)
            );
        }
        /* Activation tiles 0..K-1 now hold AH-formatted data in VTCM */

        /* Middle loop: iterate over columns of the output, 32 at a time */
        for (uint32_t col = 0; col < n_col; col += HEXKL_HMX_F16_BLOCK_N_COL) {

            /* Clear the HMX accumulator before accumulating dot products */
            hexkl_micro_hmx_acc_clear_f16();

            /* Inner loop: accumulate across the K dimension */
            for (uint32_t i = 0; i < row_tiles_in_A; i++) {
                /*
                 * Convert one 32x32 weight tile from DDR row-major
                 * directly into WH (Weight HMX) layout in VTCM.
                 * This is done on-the-fly for each (col, k) pair.
                 */
                hexkl_micro_hmx_rm_to_wh_f16(
                    vtcm_base,
                    /*weight_offset=*/weight_offset,
                    /*input_matrix=*/ wt,
                    /*row_tile=*/     i,
                    /*col_tile=*/     col / HEXKL_HMX_F16_BLOCK_N_COL,
                    /*wt_cols=*/      n_col
                );

                /*
                 * Execute one HMX matrix multiply:
                 *   accumulator += act_tile[i] * wt_tile
                 * Both tiles must already be in VTCM in AH/WH layout.
                 */
                hexkl_micro_hmx_mm_f16(
                    vtcm_base,
                    /*activation_offset=*/HEXKL_HMX_ACTIVATION_ALIGNMENT * i,
                    /*weight_offset=*/    weight_offset
                );
            }

            /*
             * Read the 32x32 f16 accumulator out of HMX into VTCM.
             * We write it to the slot after the weight tile (row_tiles_in_A + 1).
             */
            hexkl_micro_hmx_acc_read_f16(
                vtcm_base,
                hmx_config_offset,
                /*out_offset=*/HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + 1)
            );

            /*
             * Convert from AH layout back to row-major f16.
             * Input:  slot (row_tiles_in_A + 1) in AH format
             * Output: slot (row_tiles_in_A) in row-major format
             */
            hexkl_micro_hmx_ah_to_rm_f16(
                vtcm_base,
                /*flat_out_offset=*/      HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A,
                /*activation_in_offset=*/ HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + 1)
            );

            /*
             * Copy the 32x32 f16 result tile from VTCM to DDR,
             * widening from f16 to f32 on the fly.
             */
            hexkl_micro_hmx_copy_f16_to_f32_submatrix(
                vtcm_base,
                /*in_offset=*/     HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A,
                /*output_matrix=*/ out,
                /*tile_row=*/      row / HEXKL_HMX_F16_BLOCK_N_ROW,
                /*tile_col=*/      col / HEXKL_HMX_F16_BLOCK_N_COL,
                /*output_rows=*/   n_row,
                /*output_cols=*/   n_col
            );
        }
    }

    return AEE_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  main -- runs on DSP via run_main_on_hexagon                        */
/* ------------------------------------------------------------------ */
int main(void) {
    int      res  = AEE_SUCCESS;
    int      res2 = AEE_SUCCESS;
    uint8_t *vtcm_base = NULL;
    uint32_t vtcm_size = 0;

    FARF(ALWAYS, "[MICRO] === HexKL Micro API: HMX f16 MatMul Demo ===");

    /* ---- Hardware init: map VTCM ---- */
    res = hexkl_micro_hw_init(&vtcm_base, &vtcm_size);
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[MICRO][ERROR] hexkl_micro_hw_init failed (%d)", res);
        return res;
    }
    FARF(ALWAYS, "[MICRO] VTCM base = %p, size = %u bytes (%u KB)",
           (void *)vtcm_base, (unsigned)vtcm_size, (unsigned)vtcm_size / 1024);

    /* ---- Print library version ---- */
    {
        int  major = 0, minor = 0, patch = 0, hex_ver = 0;
        char prerel[HEXKL_PREREL_STR_LEN];
        hexkl_micro_get_version(&major, &minor, &patch, prerel, &hex_ver);
        FARF(ALWAYS, "[MICRO] HexKL version: %d.%d.%d-%s  (Hexagon V%d)",
               major, minor, patch, prerel, hex_ver);
    }

    /* ---- Lock HMX (exclusive access to the matrix accelerator) ---- */
    res = hexkl_micro_hmx_lock();
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[MICRO][ERROR] hexkl_micro_hmx_lock failed (%d)", res);
        return res;
    }
    FARF(ALWAYS, "[MICRO] HMX locked -- we have exclusive NPU access");

    /* ---- Allocate test matrices ---- */
    size_t act_bytes = N_ROW   * N_INNER * sizeof(_Float16);
    size_t wt_bytes  = N_INNER * N_COL   * sizeof(_Float16);
    size_t out_bytes = N_ROW   * N_COL   * sizeof(float);

    _Float16 *act      = (_Float16 *)malloc(act_bytes);
    _Float16 *wt       = (_Float16 *)malloc(wt_bytes);
    float    *out_ref  = (float *)malloc(out_bytes);
    float    *out_hmx  = (float *)malloc(out_bytes);

    if (!act || !wt || !out_ref || !out_hmx) {
        FARF(ALWAYS, "[MICRO][ERROR] malloc failed");
        res = AEE_ENOMEMORY;
        goto cleanup;
    }

    /* ---- Initialize with simple deterministic values ---- */
    for (size_t r = 0; r < N_ROW; r++)
        for (size_t k = 0; k < N_INNER; k++)
            act[r * N_INNER + k] = (_Float16)((float)((r % 5) + 0.067f));

    for (size_t k = 0; k < N_INNER; k++)
        for (size_t c = 0; c < N_COL; c++)
            wt[k * N_COL + c] = (_Float16)((float)((c % 3) + 0.049f));

    memset(out_ref, 0, out_bytes);
    memset(out_hmx, 0, out_bytes);
    FARF(ALWAYS, "[MICRO] Test data initialized: X[%d x %d] * W[%d x %d] -> Out[%d x %d]",
           N_ROW, N_INNER, N_INNER, N_COL, N_ROW, N_COL);

    /* ---- Reference matmul (plain C on DSP scalar core) ---- */
    reference_matmul(N_ROW, N_COL, N_INNER, out_ref, act, wt);
    FARF(ALWAYS, "[MICRO] Reference C matmul complete");

    /* ---- HMX tiled matmul ---- */
    res = hexkl_micro_matmul_f16f16_f32(
        vtcm_base, vtcm_size,
        N_ROW, N_COL, N_INNER,
        out_hmx, act, wt
    );
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[MICRO][ERROR] HMX matmul failed (%d)", res);
        goto cleanup;
    }
    FARF(ALWAYS, "[MICRO] HMX tiled matmul complete");

    /* ---- Compare results ---- */
    res = check_results_f32(N_ROW * N_COL, out_ref, out_hmx);
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[MICRO][ERROR] Verification FAILED -- HMX result out of tolerance");
    } else {
        FARF(ALWAYS, "[MICRO] Verification PASSED -- all %d elements match within tolerance",
               N_ROW * N_COL);
    }

    /* ---- Spot-check: print a few output values ---- */
    FARF(ALWAYS, "[MICRO] Sample outputs: out[0]=%f  out[1]=%f  out[%d]=%f",
           (double)out_hmx[0], (double)out_hmx[1],
           N_ROW * N_COL - 1, (double)out_hmx[N_ROW * N_COL - 1]);

cleanup:
    /* ---- Unlock HMX ---- */
    res2 = hexkl_micro_hmx_unlock();
    if (res2 != AEE_SUCCESS) {
        FARF(ALWAYS, "[MICRO][ERROR] hexkl_micro_hmx_unlock failed (%d)", res2);
        res |= res2;
    } else {
        FARF(ALWAYS, "[MICRO] HMX unlocked");
    }

    free(act);
    free(wt);
    free(out_ref);
    free(out_hmx);

    FARF(ALWAYS, "[MICRO] === %s ===", (res == AEE_SUCCESS) ? "PASS" : "FAIL");
    return res;
}
