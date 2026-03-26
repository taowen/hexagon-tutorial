/*
 * demo_hvx_hmx.c -- HVX + HMX combined in one FastRPC call
 *
 * This demonstrates the LLM inference pattern on Hexagon:
 *   Step 1 (HVX): Dequantize int8 weights -> f16 using vector intrinsics
 *   Step 2 (HMX): Matrix multiply via hexkl_micro API
 *   Step 3 (HVX): Add bias to output using vector f32 add
 *
 * All three steps run on the DSP in a single run_main_on_hexagon call.
 * No extra FastRPC round-trips between steps!
 *
 * In production (llama.cpp), a typical transformer layer does:
 *   HVX dequant Q4_0/Q8_0 -> f16  |  HVX layout RM->AH/WH  |  HMX matmul  |  HVX RMSNorm/RoPE
 * All in one DSP function, all in one FastRPC call.
 *
 * Computation:
 *   wt_f16 = dequant(wt_q8)                          [HVX]
 *   out = act[N_ROW x N_INNER] * wt_f16[N_INNER x N_COL]  [HMX]
 *   out += bias                                        [HVX]
 */

#include "AEEStdErr.h"
#include <stddef.h>
#include "hexkl_micro.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "HAP_farf.h"

/* ------------------------------------------------------------------ */
/*  Dimensions                                                         */
/* ------------------------------------------------------------------ */
#define N_ROW      32     /* activation rows (= HMX tile row) */
#define N_COL      128    /* weight cols / output cols */
#define N_INNER    64     /* shared inner dimension */
#define GROUP_SIZE 32     /* Q8_0: one scale per 32 elements */

/* ------------------------------------------------------------------ */
/*  Q8_0-like quantization format (same idea as llama.cpp block_q8_0)  */
/* ------------------------------------------------------------------ */
typedef struct {
    _Float16 scale;                 /* scale for this group */
    int8_t   quants[GROUP_SIZE];    /* quantized values: real = quants[i] * scale */
} block_q8;

/* ------------------------------------------------------------------ */
/*  Quantize f16 weights to Q8_0 format                                */
/* ------------------------------------------------------------------ */
static void quantize_f16_to_q8(
    block_q8 *restrict out_q8,
    const _Float16 *restrict wt_f16,
    int n_elements
) {
    int n_blocks = n_elements / GROUP_SIZE;
    for (int b = 0; b < n_blocks; b++) {
        /* Find max absolute value in this group */
        float amax = 0.0f;
        for (int j = 0; j < GROUP_SIZE; j++) {
            float v = fabsf((float)wt_f16[b * GROUP_SIZE + j]);
            if (v > amax) amax = v;
        }
        /* Compute scale */
        float scale = amax / 127.0f;
        out_q8[b].scale = (_Float16)scale;

        /* Quantize */
        float inv_scale = (scale > 0.0f) ? 127.0f / amax : 0.0f;
        for (int j = 0; j < GROUP_SIZE; j++) {
            float v = (float)wt_f16[b * GROUP_SIZE + j] * inv_scale;
            int iv = (int)roundf(v);
            if (iv > 127) iv = 127;
            if (iv < -128) iv = -128;
            out_q8[b].quants[j] = (int8_t)iv;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Step 1 (HVX): Dequantize int8 weights -> f16                      */
/*                                                                     */
/*  Process 64 values per HVX iteration (2 Q8_0 blocks):              */
/*    1. Load 128 int8 into HVX vector (use lower 64)                 */
/*    2. Sign-extend int8 -> int16 via Q6_Wh_vunpack_Vb               */
/*    3. Convert int16 -> f16 via Q6_Vhf_equals_Vh                    */
/*    4. Broadcast scale to all lanes via Q6_Vh_vsplat_R               */
/*    5. f16 multiply via Q6_Vqf16_vmpy_VhfVhf                        */
/*    6. Convert qf16 -> f16 via Q6_Vhf_equals_Vqf16                  */
/* ------------------------------------------------------------------ */
static void hvx_dequant_q8_to_f16(
    _Float16 *restrict out_f16,     /* [n_inner][n_col] row-major output */
    const block_q8 *restrict wt_q8, /* [n_inner * n_col / GROUP_SIZE] blocks */
    int n_inner,
    int n_col
) {
    const int n_elements = n_inner * n_col;
    const int n_blocks = n_elements / GROUP_SIZE;

    /* Process 2 blocks (64 f16 values) per HVX vector iteration.
     * One HVX vector = 128 bytes = 64 f16 values = 2 Q8_0 blocks. */
    for (int b = 0; b < n_blocks; b += 2) {
        /* Prepare 64 int8 values from 2 consecutive blocks into an
         * aligned buffer so we can load them as a single HVX vector. */
        int8_t tmp[128] __attribute__((aligned(128)));
        memset(tmp, 0, 128);
        memcpy(tmp,      wt_q8[b].quants,     GROUP_SIZE);
        if (b + 1 < n_blocks)
            memcpy(tmp + 32, wt_q8[b + 1].quants, GROUP_SIZE);

        /* Load 128 int8 values into HVX vector */
        HVX_Vector v_i8 = *(HVX_Vector *)tmp;

        /* Sign-extend lower 64 int8 -> 64 int16 */
        HVX_VectorPair wp = Q6_Wh_vunpack_Vb(v_i8);
        HVX_Vector v_i16 = Q6_V_lo_W(wp);

        /* Convert int16 -> f16 */
        HVX_Vector v_f16 = Q6_Vhf_equals_Vh(v_i16);

        /* Build scale vector: first 32 lanes = scale[b], last 32 = scale[b+1].
         * Q6_Vh_vsplat_R broadcasts a 16-bit value to all 64 halfword lanes. */
        _Float16 s0 = wt_q8[b].scale;
        _Float16 s1 = (b + 1 < n_blocks) ? wt_q8[b + 1].scale : (_Float16)0.0f;
        uint16_t s0_bits, s1_bits;
        memcpy(&s0_bits, &s0, 2);
        memcpy(&s1_bits, &s1, 2);

        HVX_Vector v_s0 = Q6_Vh_vsplat_R(s0_bits);
        HVX_Vector v_s1 = Q6_Vh_vsplat_R(s1_bits);

        /* Predicate selects first 64 bytes (32 halfwords) from v_s0,
         * remaining 32 halfwords from v_s1. */
        HVX_VectorPred pred = Q6_Q_vsetq2_R(64);
        HVX_Vector v_scale = Q6_V_vmux_QVV(pred, v_s0, v_s1);

        /* f16 multiply: dequantized = int_as_f16 * scale.
         * HVX quasi-float16 multiply, then convert qf16 back to f16. */
        HVX_Vector v_prod_q = Q6_Vqf16_vmpy_VhfVhf(v_f16, v_scale);
        HVX_Vector v_prod = Q6_Vhf_equals_Vqf16(v_prod_q);

        /* Store 64 f16 values to output */
        int out_idx = b * GROUP_SIZE;
        memcpy(&out_f16[out_idx], &v_prod, 64 * sizeof(_Float16));
    }
}

/* ------------------------------------------------------------------ */
/*  Step 3 (HVX): Add bias to each output row                         */
/*                                                                     */
/*  out[r][c] += bias[c] for all rows r.                               */
/*  HMX output is f32, so we use HVX f32 vector add.                  */
/*  One HVX vector = 128 bytes = 32 float32 values.                   */
/* ------------------------------------------------------------------ */
static void hvx_bias_add_f32(
    float *restrict out,          /* [n_row][n_col] */
    const float *restrict bias,   /* [n_col] */
    int n_row,
    int n_col
) {
    for (int r = 0; r < n_row; r++) {
        for (int c = 0; c < n_col; c += 32) {
            HVX_Vector v_out  = *(HVX_Vector *)&out[r * n_col + c];
            HVX_Vector v_bias = *(HVX_Vector *)&bias[c];
            /* v75 has no direct Vsf_vadd — use quasi-float32 path */
            HVX_Vector v_qf32 = Q6_Vqf32_vadd_VsfVsf(v_out, v_bias);
            HVX_Vector v_sum  = Q6_Vsf_equals_Vqf32(v_qf32);
            *(HVX_Vector *)&out[r * n_col + c] = v_sum;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Reference: dequant + matmul + bias in plain C (for verification)   */
/* ------------------------------------------------------------------ */
static void reference_dequant_matmul_bias(
    float *restrict out,            /* [n_row][n_col] output */
    const _Float16 *restrict act,   /* [n_row][n_inner] */
    const block_q8 *restrict wt_q8, /* quantized weights */
    const float *restrict bias,     /* [n_col] */
    int n_row, int n_col, int n_inner
) {
    /* First dequantize all weights to f16 */
    int n_wt = n_inner * n_col;
    _Float16 *wt_f16 = (_Float16 *)malloc(n_wt * sizeof(_Float16));
    if (!wt_f16) return;

    int n_blocks = n_wt / GROUP_SIZE;
    for (int b = 0; b < n_blocks; b++) {
        for (int j = 0; j < GROUP_SIZE; j++) {
            wt_f16[b * GROUP_SIZE + j] =
                (_Float16)((float)wt_q8[b].quants[j] * (float)wt_q8[b].scale);
        }
    }

    /* Matmul: out[r][c] = sum_k act[r][k] * wt[k][c] */
    for (int r = 0; r < n_row; r++) {
        for (int c = 0; c < n_col; c++) {
            float acc = 0.0f;
            for (int k = 0; k < n_inner; k++) {
                acc += (float)act[r * n_inner + k] * (float)wt_f16[k * n_col + c];
            }
            out[r * n_col + c] = acc + bias[c];
        }
    }

    free(wt_f16);
}

/* ------------------------------------------------------------------ */
/*  HMX tiled matmul (same pattern as demo_micro.c)                    */
/*                                                                     */
/*  Copied here so this file is self-contained.                        */
/*  See demo_micro.c for detailed comments on VTCM layout.            */
/* ------------------------------------------------------------------ */
static int hmx_tiled_matmul(
    uint8_t  *vtcm_base,
    uint32_t  vtcm_size,
    size_t    n_row,
    size_t    n_col,
    size_t    n_inner,
    float          *restrict out,
    const _Float16 *restrict act,
    const _Float16 *restrict wt
) {
    const uint32_t row_tiles_in_A =
        (n_inner + HEXKL_HMX_F16_BLOCK_N_INNER - 1) / HEXKL_HMX_F16_BLOCK_N_INNER;
    const uint32_t weight_offset =
        HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A;

    uint32_t hmx_config_offset = vtcm_size - hexkl_micro_hmx_config_size();
    hexkl_micro_hmx_setup_acc_read_f16(vtcm_base, hmx_config_offset);

    for (uint32_t row = 0; row < n_row; row += HEXKL_HMX_F16_BLOCK_N_ROW) {
        for (uint32_t i = 0; i < row_tiles_in_A; i++) {
            hexkl_micro_hmx_copy_submatrix_to_f16(
                vtcm_base,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + i),
                act, row / HEXKL_HMX_F16_BLOCK_N_ROW, i, n_row, n_inner);
            hexkl_micro_hmx_rm_to_ah_f16(
                vtcm_base,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * i,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + i));
        }

        for (uint32_t col = 0; col < n_col; col += HEXKL_HMX_F16_BLOCK_N_COL) {
            hexkl_micro_hmx_acc_clear_f16();

            for (uint32_t i = 0; i < row_tiles_in_A; i++) {
                hexkl_micro_hmx_rm_to_wh_f16(
                    vtcm_base, weight_offset,
                    wt, i, col / HEXKL_HMX_F16_BLOCK_N_COL, n_col);
                hexkl_micro_hmx_mm_f16(
                    vtcm_base,
                    HEXKL_HMX_ACTIVATION_ALIGNMENT * i,
                    weight_offset);
            }

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
                out,
                row / HEXKL_HMX_F16_BLOCK_N_ROW,
                col / HEXKL_HMX_F16_BLOCK_N_COL,
                n_row, n_col);
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

    /* Buffers */
    _Float16 *act       = NULL;
    _Float16 *wt_f16    = NULL;   /* original f16 weights (for quantization) */
    _Float16 *wt_deq    = NULL;   /* HVX-dequantized weights */
    block_q8 *wt_q8     = NULL;   /* quantized weights */
    float    *bias      = NULL;
    float    *out_ref   = NULL;
    float    *out_hmx   = NULL;

    FARF(ALWAYS, "[HVX+HMX] === Combined HVX + HMX Pipeline Demo ===");
    FARF(ALWAYS, "[HVX+HMX] Simulating LLM inference: HVX dequant -> HMX matmul -> HVX bias add");
    FARF(ALWAYS, "[HVX+HMX] All steps in ONE FastRPC call (no round-trip overhead)");
    FARF(ALWAYS, "[HVX+HMX] Dimensions: act[%d x %d] * wt[%d x %d] -> out[%d x %d]",
         N_ROW, N_INNER, N_INNER, N_COL, N_ROW, N_COL);

    /* ---- Hardware init ---- */
    res = hexkl_micro_hw_init(&vtcm_base, &vtcm_size);
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[HVX+HMX][ERROR] hw_init failed (%d)", res);
        return res;
    }
    FARF(ALWAYS, "[HVX+HMX] VTCM: %u KB at %p", (unsigned)vtcm_size / 1024, (void *)vtcm_base);

    res = hexkl_micro_hmx_lock();
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[HVX+HMX][ERROR] hmx_lock failed (%d)", res);
        return res;
    }
    FARF(ALWAYS, "[HVX+HMX] HMX locked");

    /* ---- Allocate ---- */
    size_t n_wt = (size_t)N_INNER * N_COL;
    int n_blocks = n_wt / GROUP_SIZE;

    act     = (_Float16 *)malloc(N_ROW * N_INNER * sizeof(_Float16));
    wt_f16  = (_Float16 *)malloc(n_wt * sizeof(_Float16));
    wt_deq  = (_Float16 *)memalign(128, n_wt * sizeof(_Float16)); /* HVX 128B alignment */
    wt_q8   = (block_q8 *)malloc(n_blocks * sizeof(block_q8));
    bias    = (float *)memalign(128, N_COL * sizeof(float));       /* HVX 128B alignment */
    out_ref = (float *)malloc(N_ROW * N_COL * sizeof(float));
    out_hmx = (float *)memalign(128, N_ROW * N_COL * sizeof(float));

    if (!act || !wt_f16 || !wt_deq || !wt_q8 || !bias || !out_ref || !out_hmx) {
        FARF(ALWAYS, "[HVX+HMX][ERROR] malloc failed");
        res = AEE_ENOMEMORY;
        goto cleanup;
    }

    /* ---- Initialize test data ---- */
    for (int r = 0; r < N_ROW; r++)
        for (int k = 0; k < N_INNER; k++)
            act[r * N_INNER + k] = (_Float16)((float)((r % 5) + 0.067f));

    for (int k = 0; k < N_INNER; k++)
        for (int c = 0; c < N_COL; c++)
            wt_f16[k * N_COL + c] = (_Float16)((float)((c % 7) - 3.0f) * 0.1f);

    for (int c = 0; c < N_COL; c++)
        bias[c] = (float)(c % 4) * 0.5f;

    /* ---- Quantize weights to Q8_0 ---- */
    quantize_f16_to_q8(wt_q8, wt_f16, n_wt);
    FARF(ALWAYS, "[HVX+HMX] Quantized %d weights into %d Q8_0 blocks", (int)n_wt, n_blocks);

    /* ---- Reference: scalar dequant + matmul + bias ---- */
    memset(out_ref, 0, N_ROW * N_COL * sizeof(float));
    reference_dequant_matmul_bias(out_ref, act, wt_q8, bias, N_ROW, N_COL, N_INNER);
    FARF(ALWAYS, "[HVX+HMX] Reference (scalar C) done");

    /* ================================================================
     * THE COMBINED PIPELINE -- this is the key demo
     *
     *   Step 1 (HVX): dequantize Q8_0 weights -> f16
     *   Step 2 (HMX): tiled matrix multiply
     *   Step 3 (HVX): add bias to output
     *
     * All three steps execute right here, on the DSP, in the same
     * run_main_on_hexagon call. No FastRPC overhead between steps!
     * ================================================================ */

    FARF(ALWAYS, "[HVX+HMX] --- Pipeline start ---");

    /* Step 1: HVX dequantization */
    FARF(ALWAYS, "[HVX+HMX] Step 1: HVX dequant int8 -> f16 (%d blocks)", n_blocks);
    memset(wt_deq, 0, n_wt * sizeof(_Float16));
    hvx_dequant_q8_to_f16(wt_deq, wt_q8, N_INNER, N_COL);
    FARF(ALWAYS, "[HVX+HMX] Step 1 done: dequantized %d values", (int)n_wt);

    /* Spot check dequant accuracy */
    {
        int dq_err = 0;
        float max_dq_diff = 0.0f;
        for (int i = 0; i < (int)n_wt; i++) {
            /* Reference scalar dequant for comparison */
            int blk = i / GROUP_SIZE;
            int j = i % GROUP_SIZE;
            float ref_val = (float)wt_q8[blk].quants[j] * (float)wt_q8[blk].scale;
            float hvx_val = (float)wt_deq[i];
            float d = fabsf(ref_val - hvx_val);
            if (d > max_dq_diff) max_dq_diff = d;
            if (d > 0.01f) dq_err++;
        }
        FARF(ALWAYS, "[HVX+HMX] Dequant check: max_diff=%.6f, errors=%d/%d",
             (double)max_dq_diff, dq_err, (int)n_wt);
    }

    /* Step 2: HMX matmul */
    FARF(ALWAYS, "[HVX+HMX] Step 2: HMX matmul [%d x %d] * [%d x %d]",
         N_ROW, N_INNER, N_INNER, N_COL);
    memset(out_hmx, 0, N_ROW * N_COL * sizeof(float));
    res = hmx_tiled_matmul(vtcm_base, vtcm_size,
                           N_ROW, N_COL, N_INNER,
                           out_hmx, act, wt_deq);
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[HVX+HMX][ERROR] HMX matmul failed (%d)", res);
        goto cleanup;
    }
    FARF(ALWAYS, "[HVX+HMX] Step 2 done: HMX matmul complete");

    /* Step 3: HVX bias add */
    FARF(ALWAYS, "[HVX+HMX] Step 3: HVX bias add (%d rows x %d cols)", N_ROW, N_COL);
    hvx_bias_add_f32(out_hmx, bias, N_ROW, N_COL);
    FARF(ALWAYS, "[HVX+HMX] Step 3 done: bias added");

    FARF(ALWAYS, "[HVX+HMX] --- Pipeline complete ---");

    /* ---- Verify against reference ---- */
    {
        int mismatches = 0;
        float max_diff = 0.0f;
        for (int i = 0; i < N_ROW * N_COL; i++) {
            float diff = fabsf(out_ref[i] - out_hmx[i]);
            if (diff > max_diff) max_diff = diff;
            /* 1% tolerance for quantized pipeline */
            float eps = fabsf(out_ref[i] / 100.0f);
            if (diff > eps && diff > 0.1f) {
                if (mismatches < 5) {
                    FARF(ALWAYS, "[HVX+HMX][MISMATCH] [%d] ref=%f got=%f diff=%f",
                         i, (double)out_ref[i], (double)out_hmx[i], (double)diff);
                }
                mismatches++;
            }
        }
        FARF(ALWAYS, "[HVX+HMX] Max diff: %f", (double)max_diff);
        if (mismatches == 0) {
            FARF(ALWAYS, "[HVX+HMX] Verification PASSED (%d elements)", N_ROW * N_COL);
        } else {
            FARF(ALWAYS, "[HVX+HMX] Verification FAILED: %d / %d mismatches",
                 mismatches, N_ROW * N_COL);
        }
    }

    /* Spot check outputs */
    FARF(ALWAYS, "[HVX+HMX] out[0]=%f (ref=%f)  out[1]=%f (ref=%f)",
         (double)out_hmx[0], (double)out_ref[0],
         (double)out_hmx[1], (double)out_ref[1]);

cleanup:
    res2 = hexkl_micro_hmx_unlock();
    if (res2 != AEE_SUCCESS) {
        FARF(ALWAYS, "[HVX+HMX][ERROR] hmx_unlock failed (%d)", res2);
        res |= res2;
    }

    free(act);
    free(wt_f16);
    free(wt_deq);
    free(wt_q8);
    free(bias);
    free(out_ref);
    free(out_hmx);

    FARF(ALWAYS, "[HVX+HMX] === %s ===", (res == AEE_SUCCESS) ? "PASS" : "FAIL");
    return res;
}
