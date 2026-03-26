/*
 * demo_native_kv.c -- NativeKV vs SmartMask KV cache benchmark
 *
 * Runs on DSP via run_main_on_hexagon.  Compares two strategies for
 * the Q x K^T attention matmul on HMX:
 *
 *   SmartMask:  K stored row-major, rm_to_wh conversion every step
 *   NativeKV:   K pre-formatted in WH layout, just copy tiles
 */

#include "AEEStdErr.h"
#include <stddef.h>
#include "hexkl_micro.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "HAP_farf.h"
#include "qurt_sclk.h"

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */
#define HEAD_DIM    64      /* per-head embedding dim */
#define CTX_SIZE    256     /* context window (multiple of 256 for NativeKV) */
#define N_QUERY     32      /* query rows = 1 HMX tile */
#define N_ITER      100     /* benchmark iterations */
#define TILE        32      /* HMX tile size */
#define TILE_ALIGN  2048    /* HEXKL_HMX_ACTIVATION_ALIGNMENT */

/* ------------------------------------------------------------------ */
/*  reference_qkt -- plain C scalar Q x K^T                            */
/* ------------------------------------------------------------------ */
static void reference_qkt(
    float *out,               /* [N_QUERY x CTX_SIZE] */
    const _Float16 *Q,        /* [N_QUERY x HEAD_DIM] */
    const _Float16 *K_rm,     /* [HEAD_DIM x CTX_SIZE] */
    int n_query, int head_dim, int ctx_size
) {
    for (int r = 0; r < n_query; r++)
        for (int c = 0; c < ctx_size; c++) {
            float acc = 0.0f;
            for (int k = 0; k < head_dim; k++)
                acc += (float)Q[r * head_dim + k] * (float)K_rm[k * ctx_size + c];
            out[r * ctx_size + c] = acc;
        }
}

/* ------------------------------------------------------------------ */
/*  attention_smartmask -- K in row-major, convert every step           */
/* ------------------------------------------------------------------ */
static int attention_smartmask(
    uint8_t *vtcm_base, uint32_t vtcm_size,
    float *out,
    const _Float16 *Q,
    const _Float16 *K_rm,
    int n_query, int head_dim, int ctx_size
) {
    const uint32_t row_tiles_in_A = (head_dim + HEXKL_HMX_F16_BLOCK_N_INNER - 1) / HEXKL_HMX_F16_BLOCK_N_INNER;
    const uint32_t weight_offset = HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A;
    uint32_t hmx_config_offset = vtcm_size - hexkl_micro_hmx_config_size();
    hexkl_micro_hmx_setup_acc_read_f16(vtcm_base, hmx_config_offset);

    for (uint32_t row = 0; row < (uint32_t)n_query; row += HEXKL_HMX_F16_BLOCK_N_ROW) {
        /* Load activation strip (Q rows) into VTCM */
        for (uint32_t i = 0; i < row_tiles_in_A; i++) {
            hexkl_micro_hmx_copy_submatrix_to_f16(vtcm_base,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + i),
                Q, row / HEXKL_HMX_F16_BLOCK_N_ROW, i, n_query, head_dim);
            hexkl_micro_hmx_rm_to_ah_f16(vtcm_base,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * i,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + i));
        }
        for (uint32_t col = 0; col < (uint32_t)ctx_size; col += HEXKL_HMX_F16_BLOCK_N_COL) {
            hexkl_micro_hmx_acc_clear_f16();
            for (uint32_t i = 0; i < row_tiles_in_A; i++) {
                /* SmartMask: convert K from row-major to WH EVERY TIME */
                hexkl_micro_hmx_rm_to_wh_f16(vtcm_base, weight_offset,
                    K_rm, i, col / HEXKL_HMX_F16_BLOCK_N_COL, ctx_size);
                hexkl_micro_hmx_mm_f16(vtcm_base,
                    HEXKL_HMX_ACTIVATION_ALIGNMENT * i, weight_offset);
            }
            hexkl_micro_hmx_acc_read_f16(vtcm_base, hmx_config_offset,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + 1));
            hexkl_micro_hmx_ah_to_rm_f16(vtcm_base,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + 1));
            hexkl_micro_hmx_copy_f16_to_f32_submatrix(vtcm_base,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A,
                out, row / HEXKL_HMX_F16_BLOCK_N_ROW,
                col / HEXKL_HMX_F16_BLOCK_N_COL, n_query, ctx_size);
        }
    }
    return AEE_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  attention_native -- K pre-formatted in WH, copy only               */
/* ------------------------------------------------------------------ */
static int attention_native(
    uint8_t *vtcm_base, uint32_t vtcm_size,
    float *out,
    const _Float16 *Q,
    const _Float16 *K_wh,    /* already in WH layout */
    int n_query, int head_dim, int ctx_size
) {
    const uint32_t row_tiles_in_A = (head_dim + HEXKL_HMX_F16_BLOCK_N_INNER - 1) / HEXKL_HMX_F16_BLOCK_N_INNER;
    const uint32_t weight_offset = HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A;
    uint32_t hmx_config_offset = vtcm_size - hexkl_micro_hmx_config_size();
    hexkl_micro_hmx_setup_acc_read_f16(vtcm_base, hmx_config_offset);

    for (uint32_t row = 0; row < (uint32_t)n_query; row += HEXKL_HMX_F16_BLOCK_N_ROW) {
        /* Load activation strip (Q rows) into VTCM */
        for (uint32_t i = 0; i < row_tiles_in_A; i++) {
            hexkl_micro_hmx_copy_submatrix_to_f16(vtcm_base,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + i),
                Q, row / HEXKL_HMX_F16_BLOCK_N_ROW, i, n_query, head_dim);
            hexkl_micro_hmx_rm_to_ah_f16(vtcm_base,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * i,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + i));
        }
        for (uint32_t col = 0; col < (uint32_t)ctx_size; col += HEXKL_HMX_F16_BLOCK_N_COL) {
            hexkl_micro_hmx_acc_clear_f16();
            for (uint32_t i = 0; i < row_tiles_in_A; i++) {
                /* NativeKV: just copy pre-formatted WH tiles, no conversion */
                hexkl_micro_hmx_copy_psubmatrix_to_f16_weight(vtcm_base,
                    weight_offset, K_wh, i,
                    col / HEXKL_HMX_F16_BLOCK_N_COL, head_dim, ctx_size);
                hexkl_micro_hmx_mm_f16(vtcm_base,
                    HEXKL_HMX_ACTIVATION_ALIGNMENT * i, weight_offset);
            }
            hexkl_micro_hmx_acc_read_f16(vtcm_base, hmx_config_offset,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + 1));
            hexkl_micro_hmx_ah_to_rm_f16(vtcm_base,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + 1));
            hexkl_micro_hmx_copy_f16_to_f32_submatrix(vtcm_base,
                HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A,
                out, row / HEXKL_HMX_F16_BLOCK_N_ROW,
                col / HEXKL_HMX_F16_BLOCK_N_COL, n_query, ctx_size);
        }
    }
    return AEE_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  preformat_k_to_wh -- one-time conversion at model load             */
/* ------------------------------------------------------------------ */
static int preformat_k_to_wh(
    uint8_t *vtcm_base,
    _Float16 *K_wh,
    const _Float16 *K_rm,
    int head_dim, int ctx_size
) {
    uint32_t row_tiles = head_dim / TILE;
    uint32_t col_tiles = ctx_size / TILE;
    for (uint32_t rt = 0; rt < row_tiles; rt++) {
        for (uint32_t ct = 0; ct < col_tiles; ct++) {
            hexkl_micro_hmx_rm_to_wh_f16(vtcm_base, 0, K_rm, rt, ct, ctx_size);
            uint32_t tile_idx = rt * col_tiles + ct;
            memcpy((uint8_t *)K_wh + tile_idx * TILE_ALIGN,
                   vtcm_base, TILE_ALIGN);
        }
    }
    return AEE_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  Result comparison (0.1% relative tolerance, 0.01 absolute floor)   */
/* ------------------------------------------------------------------ */
static int check_results_f32(size_t count, const float *ref, const float *got) {
    for (size_t i = 0; i < count; i++) {
        if (isnan(ref[i]) || isinf(got[i])) {
            FARF(ALWAYS, "[KV][ERROR] NaN/Inf at index %u: ref=%f got=%f",
                   (unsigned)i, (double)ref[i], (double)got[i]);
            return AEE_EFAILED;
        }
        float diff    = fabsf(ref[i] - got[i]);
        float epsilon = fabsf(ref[i] / 1000.0f);   /* 0.1 % */
        if (diff > epsilon && diff > 0.01f) {
            FARF(ALWAYS, "[KV][ERROR] Mismatch at index %u: ref=%f got=%f "
                   "diff=%f eps=%f",
                   (unsigned)i, (double)ref[i], (double)got[i],
                   (double)diff, (double)epsilon);
            return AEE_EFAILED;
        }
    }
    return AEE_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  fromFlatOffset -- Genie WH tile layout address computation          */
/* ------------------------------------------------------------------ */
static int fromFlatOffset(int DIN, int DOUT, int N_TILE, int din, int dout) {
    int tile_size = DOUT < N_TILE ? DOUT : N_TILE;
    int tile_stride = DIN * tile_size;
    int tile_idx = dout / tile_size;
    int dout_0 = (dout % tile_size) >> 5;
    int dout_1 = dout & 0x1f;
    int din_0 = din >> 5;
    int din_1 = (din & 0x1f) >> 2;
    int din_2 = din & 0x3;
    int din_0_stride = tile_size << 5;
    (void)DIN;  /* used only in tile_stride */
    return tile_idx * tile_stride + din_0 * din_0_stride +
           (dout_0 << 10 | din_1 << 7 | dout_1 << 2 | din_2);
}

/* ------------------------------------------------------------------ */
/*  main -- runs on DSP via run_main_on_hexagon                        */
/* ------------------------------------------------------------------ */
int main(void) {
    int      res  = AEE_SUCCESS;
    int      res2 = AEE_SUCCESS;
    uint8_t *vtcm_base = NULL;
    uint32_t vtcm_size = 0;

    _Float16 *Q       = NULL;
    _Float16 *K_rm    = NULL;
    _Float16 *K_wh    = NULL;
    float    *out_ref  = NULL;
    float    *out_sm   = NULL;
    float    *out_nk   = NULL;

    FARF(ALWAYS, "[KV] === NativeKV vs SmartMask KV Cache Benchmark ===");
    FARF(ALWAYS, "[KV] Q[%d x %d] x K^T[%d x %d] -> scores[%d x %d]",
           N_QUERY, HEAD_DIM, HEAD_DIM, CTX_SIZE, N_QUERY, CTX_SIZE);

    /* ---- Hardware init: map VTCM ---- */
    res = hexkl_micro_hw_init(&vtcm_base, &vtcm_size);
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[KV][ERROR] hexkl_micro_hw_init failed (%d)", res);
        return res;
    }
    FARF(ALWAYS, "[KV] VTCM: %u KB", (unsigned)(vtcm_size / 1024));

    /* ---- Lock HMX ---- */
    res = hexkl_micro_hmx_lock();
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[KV][ERROR] hexkl_micro_hmx_lock failed (%d)", res);
        return res;
    }
    FARF(ALWAYS, "[KV] HMX locked");

    /* ---- Allocate buffers ---- */
    size_t q_bytes    = N_QUERY  * HEAD_DIM * sizeof(_Float16);
    size_t k_rm_bytes = HEAD_DIM * CTX_SIZE * sizeof(_Float16);
    size_t k_wh_bytes = (HEAD_DIM / TILE) * (CTX_SIZE / TILE) * TILE_ALIGN;
    size_t out_bytes  = N_QUERY  * CTX_SIZE * sizeof(float);

    Q       = (_Float16 *)memalign(128, q_bytes);
    K_rm    = (_Float16 *)memalign(128, k_rm_bytes);
    K_wh    = (_Float16 *)memalign(128, k_wh_bytes);
    out_ref = (float *)memalign(128, out_bytes);
    out_sm  = (float *)memalign(128, out_bytes);
    out_nk  = (float *)memalign(128, out_bytes);

    if (!Q || !K_rm || !K_wh || !out_ref || !out_sm || !out_nk) {
        FARF(ALWAYS, "[KV][ERROR] memalign failed");
        res = AEE_ENOMEMORY;
        goto cleanup;
    }

    /* ---- Initialize test data ---- */
    for (int r = 0; r < N_QUERY; r++)
        for (int k = 0; k < HEAD_DIM; k++)
            Q[r * HEAD_DIM + k] = (_Float16)(0.1f * ((r * HEAD_DIM + k) % 17));

    for (int k = 0; k < HEAD_DIM; k++)
        for (int c = 0; c < CTX_SIZE; c++)
            K_rm[k * CTX_SIZE + c] = (_Float16)(0.05f * ((k * CTX_SIZE + c) % 23));

    memset(out_ref, 0, out_bytes);
    memset(out_sm, 0, out_bytes);
    memset(out_nk, 0, out_bytes);

    /* ---- Reference matmul ---- */
    reference_qkt(out_ref, Q, K_rm, N_QUERY, HEAD_DIM, CTX_SIZE);

    /* ================================================================ */
    /*  Part 1: Pre-format K to WH layout (one-time cost)               */
    /* ================================================================ */
    FARF(ALWAYS, "[KV] --- Part 1: Pre-format K -> WH layout (one-time) ---");
    {
        unsigned long long t0 = qurt_sysclock_get_hw_ticks();
        res = preformat_k_to_wh(vtcm_base, K_wh, K_rm, HEAD_DIM, CTX_SIZE);
        unsigned long long t1 = qurt_sysclock_get_hw_ticks();
        if (res != AEE_SUCCESS) {
            FARF(ALWAYS, "[KV][ERROR] preformat_k_to_wh failed (%d)", res);
            goto cleanup;
        }
        unsigned preformat_ticks = (unsigned)(t1 - t0);
        FARF(ALWAYS, "[KV] Preformat: %u ticks (row_tiles=%d, col_tiles=%d)",
               preformat_ticks, HEAD_DIM / TILE, CTX_SIZE / TILE);

        /* ============================================================ */
        /*  Part 2: SmartMask attention (converts K every time)         */
        /* ============================================================ */
        FARF(ALWAYS, "[KV] --- Part 2: SmartMask (converts K every time) ---");

        /* Verify once */
        res = attention_smartmask(vtcm_base, vtcm_size, out_sm, Q, K_rm,
                                  N_QUERY, HEAD_DIM, CTX_SIZE);
        if (res != AEE_SUCCESS) {
            FARF(ALWAYS, "[KV][ERROR] attention_smartmask failed (%d)", res);
            goto cleanup;
        }
        res = check_results_f32(N_QUERY * CTX_SIZE, out_ref, out_sm);
        if (res != AEE_SUCCESS) {
            FARF(ALWAYS, "[KV] SmartMask verify: FAIL");
            goto cleanup;
        }
        FARF(ALWAYS, "[KV] SmartMask verify: PASS");

        /* Benchmark */
        unsigned long long sm_t0 = qurt_sysclock_get_hw_ticks();
        for (int iter = 0; iter < N_ITER; iter++) {
            attention_smartmask(vtcm_base, vtcm_size, out_sm, Q, K_rm,
                                N_QUERY, HEAD_DIM, CTX_SIZE);
        }
        unsigned long long sm_t1 = qurt_sysclock_get_hw_ticks();
        unsigned sm_per_iter = (unsigned)((sm_t1 - sm_t0) / N_ITER);
        FARF(ALWAYS, "[KV] SmartMask: %u ticks/iter (%d iters)",
               sm_per_iter, N_ITER);

        /* ============================================================ */
        /*  Part 3: NativeKV attention (K already in WH)                */
        /* ============================================================ */
        FARF(ALWAYS, "[KV] --- Part 3: NativeKV (K already in WH) ---");

        /* Verify once */
        res = attention_native(vtcm_base, vtcm_size, out_nk, Q, K_wh,
                               N_QUERY, HEAD_DIM, CTX_SIZE);
        if (res != AEE_SUCCESS) {
            FARF(ALWAYS, "[KV][ERROR] attention_native failed (%d)", res);
            goto cleanup;
        }
        res = check_results_f32(N_QUERY * CTX_SIZE, out_ref, out_nk);
        if (res != AEE_SUCCESS) {
            FARF(ALWAYS, "[KV] NativeKV verify: FAIL");
            goto cleanup;
        }
        FARF(ALWAYS, "[KV] NativeKV verify: PASS");

        /* Benchmark */
        unsigned long long nk_t0 = qurt_sysclock_get_hw_ticks();
        for (int iter = 0; iter < N_ITER; iter++) {
            attention_native(vtcm_base, vtcm_size, out_nk, Q, K_wh,
                             N_QUERY, HEAD_DIM, CTX_SIZE);
        }
        unsigned long long nk_t1 = qurt_sysclock_get_hw_ticks();
        unsigned nk_per_iter = (unsigned)((nk_t1 - nk_t0) / N_ITER);
        FARF(ALWAYS, "[KV] NativeKV: %u ticks/iter (%d iters)",
               nk_per_iter, N_ITER);

        /* ============================================================ */
        /*  Part 4: Summary                                             */
        /* ============================================================ */
        FARF(ALWAYS, "[KV] --- Part 4: Summary ---");
        FARF(ALWAYS, "[KV] SmartMask: %u ticks/iter (converts HEAD_DIM x CTX_SIZE every step)",
               sm_per_iter);
        FARF(ALWAYS, "[KV] NativeKV:  %u ticks/iter (just copies tiles)",
               nk_per_iter);

        if (nk_per_iter > 0) {
            float speedup = (float)sm_per_iter / (float)nk_per_iter;
            FARF(ALWAYS, "[KV] Speedup: %.2fx", (double)speedup);
        }

        if (sm_per_iter > nk_per_iter) {
            unsigned delta = sm_per_iter - nk_per_iter;
            unsigned breakeven = (preformat_ticks + delta - 1) / delta;
            FARF(ALWAYS, "[KV] Break-even: preformat cost recovered after %u steps",
                   breakeven);
        }
    }

    /* ================================================================ */
    /*  Part 5: 32-alignment constraint                                 */
    /* ================================================================ */
    FARF(ALWAYS, "[KV] --- Part 5: 32-alignment constraint ---");
    {
        int test_vals[] = { 0, 1, 33 };
        for (int i = 0; i < 3; i++) {
            int n_valid = test_vals[i];
            int new_idx = ((n_valid + 31) / 32) * 32;
            int skipped = new_idx - n_valid;
            if (n_valid == 0) {
                FARF(ALWAYS, "[KV] n_valid=%d -> new_idx=%d",
                       n_valid, new_idx);
            } else {
                FARF(ALWAYS, "[KV] n_valid=%d -> new_idx=%d (%d skipped, masked in attention)",
                       n_valid, new_idx, skipped);
            }
        }
    }

    /* ================================================================ */
    /*  Part 6: fromFlatOffset layout                                   */
    /* ================================================================ */
    FARF(ALWAYS, "[KV] --- Part 6: fromFlatOffset layout ---");
    {
        int embed_dim = 128;
        int ctx_sz    = 512;
        int din       = 65;
        int dout      = 300;
        int K_TILE    = 256;

        int offset = fromFlatOffset(embed_dim, ctx_sz, K_TILE, din, dout);
        FARF(ALWAYS, "[KV] Key [%d,%d] din=%d dout=%d K_TILE=%d -> offset=%d",
               embed_dim, ctx_sz, din, dout, K_TILE, offset);
        FARF(ALWAYS, "[KV] Inner 1024B block: [8:din_1][32:dout_1][4:din_2]");
    }

    FARF(ALWAYS, "[KV] === PASS ===");

cleanup:
    /* ---- Unlock HMX ---- */
    res2 = hexkl_micro_hmx_unlock();
    if (res2 != AEE_SUCCESS) {
        FARF(ALWAYS, "[KV][ERROR] hexkl_micro_hmx_unlock failed (%d)", res2);
        res |= res2;
    }

    free(Q);
    free(K_rm);
    free(K_wh);
    free(out_ref);
    free(out_sm);
    free(out_nk);

    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[KV] === FAIL ===");
    }

    return res;
}
