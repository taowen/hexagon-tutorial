/*
 * demo_streaming.c -- VTCM streaming for large matrix HMX matmul
 *
 * Experiment 5: when weight matrices exceed VTCM capacity (e.g. LLM layers
 * at 4096x4096), we chunk the N (output column) dimension and stream weight
 * chunks through VTCM. Activations are loaded once per row strip and reused
 * across all column chunks.
 *
 * This solves the AEE_ENOMEMORY problem in L1/L2 for large matrices.
 *
 * Approaches benchmarked:
 *   L2 preformatted  -- existing non-streaming (may SKIP for large N)
 *   Stream baseline  -- streaming with rm_to_wh conversion at load time
 *   Stream prefmt    -- streaming with pre-formatted WH weights (just memcpy)
 *
 * Key insight: activation tiles (K tiles) are loaded once per row strip.
 * Weight tiles are loaded in column chunks that fit in remaining VTCM.
 * The outer loop is rows, the middle loop is column chunks, the inner loop
 * is the K-dimension accumulation.
 *
 * Compile: hexagon-clang -mv75 -mhvx -mhvx-length=128B -O3 ...
 */

#include "AEEStdErr.h"
#include <stddef.h>
#include "hexkl_micro.h"
#include "hexkl_macro.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "HAP_farf.h"
#include "HAP_perf.h"

#define TILE         32
#define TILE_ELMS    (TILE * TILE)      /* 1024 f16 elements per tile */
#define TILE_BYTES   (TILE_ELMS * 2)    /* 2048 bytes per 32x32 f16 tile */
#define VLEN         128                /* HVX vector length in bytes */
#define N_ITERS      20

/* ================================================================== */
/*  VTCM budget calculator                                             */
/*                                                                     */
/*  VTCM layout for streaming:                                         */
/*  [act tiles 0..K-1][staging][readback][wt chunk tiles][hmx_config]  */
/*                                                                     */
/*  Fixed cost: K activation tiles + 1 staging + 1 readback + config   */
/*  Remaining: divided into weight column tile sets (each column needs  */
/*  K tiles in the K dimension).                                       */
/*                                                                     */
/*  Returns the number of output columns (in tiles) that fit per chunk */
/*  and the VTCM offset where weight tiles start.                      */
/* ================================================================== */

struct chunk_params {
    uint32_t n_chunk_tiles;   /* how many output column tiles fit per chunk */
    uint32_t wt_base;         /* VTCM offset where weight chunk starts */
    uint32_t staging_off;     /* VTCM offset of staging tile */
    uint32_t readback_off;    /* VTCM offset of readback tile */
    uint32_t cfg_off;         /* VTCM offset of HMX config (at end) */
};

static int compute_chunk_params(
    uint32_t vtcm_size,
    uint32_t K,               /* number of inner-dimension tiles */
    struct chunk_params *out)
{
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;
    uint32_t cfg_size = hexkl_micro_hmx_config_size();

    /* Fixed VTCM cost: K act tiles + staging + readback */
    uint32_t fixed_tiles = K + 2;
    uint32_t fixed_cost  = fixed_tiles * ALIGN + cfg_size;

    if (fixed_cost >= vtcm_size) return AEE_ENOMEMORY;

    uint32_t remaining = vtcm_size - fixed_cost;

    /* Each output column needs K weight tiles in VTCM */
    uint32_t per_col = K * ALIGN;
    uint32_t n_chunk_tiles = remaining / per_col;

    if (n_chunk_tiles == 0) return AEE_ENOMEMORY;

    out->n_chunk_tiles = n_chunk_tiles;
    out->staging_off   = ALIGN * K;
    out->readback_off  = ALIGN * (K + 1);
    out->wt_base       = ALIGN * (K + 2);
    out->cfg_off       = vtcm_size - cfg_size;

    return AEE_SUCCESS;
}

/* ================================================================== */
/*  Streaming matmul with pre-formatted WH weights                     */
/*                                                                     */
/*  Outer loop: row strips (TILE rows at a time)                       */
/*    Load activation strip to VTCM once per row strip.                */
/*    Middle loop: column chunks (n_chunk_tiles columns per iteration)  */
/*      Load weight chunk to VTCM (pre-formatted, just memcpy).        */
/*      Inner loop: for each column in chunk, accumulate K tiles.       */
/*                                                                     */
/*  This handles arbitrarily large N by chunking columns.              */
/* ================================================================== */

static int hmx_streaming_preformatted(
    uint8_t *vtcm, uint32_t vtcm_size,
    float *out_f32, const _Float16 *act, const _Float16 *wt_preformatted,
    uint32_t m, uint32_t n, uint32_t k)
{
    const uint32_t K  = (k + TILE - 1) / TILE;
    const uint32_t Nc = (n + TILE - 1) / TILE;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    struct chunk_params cp;
    int err = compute_chunk_params(vtcm_size, K, &cp);
    if (err != AEE_SUCCESS) return err;

    uint32_t k_pad = K * TILE;
    uint32_t n_pad = Nc * TILE;

    hexkl_micro_hmx_setup_acc_read_f16(vtcm, cp.cfg_off);

    for (uint32_t row = 0; row < m; row += TILE) {
        uint32_t rt = row / TILE;

        /* Load activation strip to VTCM (once per row strip) */
        for (uint32_t i = 0; i < K; i++) {
            hexkl_micro_hmx_copy_submatrix_to_f16(
                vtcm, cp.staging_off, act, rt, i, m, k);
            hexkl_micro_hmx_rm_to_ah_f16(vtcm, ALIGN * i, cp.staging_off);
        }

        /* Stream column chunks through VTCM */
        for (uint32_t col_base = 0; col_base < Nc; col_base += cp.n_chunk_tiles) {
            uint32_t chunk_cols = Nc - col_base;
            if (chunk_cols > cp.n_chunk_tiles)
                chunk_cols = cp.n_chunk_tiles;

            /* Load weight chunk into VTCM (pre-formatted, just memcpy) */
            for (uint32_t c = 0; c < chunk_cols; c++)
                for (uint32_t i = 0; i < K; i++)
                    hexkl_micro_hmx_copy_psubmatrix_to_f16_weight(vtcm,
                        cp.wt_base + (c * K + i) * ALIGN,
                        wt_preformatted, i, col_base + c, k_pad, n_pad);

            /* Compute: for each column in this chunk */
            for (uint32_t c = 0; c < chunk_cols; c++) {
                hexkl_micro_hmx_acc_clear_f16();
                for (uint32_t i = 0; i < K; i++)
                    hexkl_micro_hmx_mm_f16(vtcm, ALIGN * i,
                                            cp.wt_base + (c * K + i) * ALIGN);
                hexkl_micro_hmx_acc_read_f16(vtcm, cp.cfg_off, cp.readback_off);
                hexkl_micro_hmx_ah_to_rm_f16(vtcm, cp.staging_off, cp.readback_off);
                hexkl_micro_hmx_copy_f16_to_f32_submatrix(
                    vtcm, cp.staging_off, out_f32, rt, col_base + c, m, n);
            }
        }
    }
    return AEE_SUCCESS;
}

/* ================================================================== */
/*  Streaming matmul with rm_to_wh conversion at load time (baseline)  */
/*                                                                     */
/*  Same streaming structure as above, but weights are in row-major    */
/*  DDR layout. Each chunk load does rm_to_wh conversion. This shows   */
/*  that the L2 benefit (pre-formatted weights) persists in streaming  */
/*  mode.                                                              */
/* ================================================================== */

static int hmx_streaming_baseline(
    uint8_t *vtcm, uint32_t vtcm_size,
    float *out_f32, const _Float16 *act, const _Float16 *wt,
    uint32_t m, uint32_t n, uint32_t k)
{
    const uint32_t K  = (k + TILE - 1) / TILE;
    const uint32_t Nc = (n + TILE - 1) / TILE;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    struct chunk_params cp;
    int err = compute_chunk_params(vtcm_size, K, &cp);
    if (err != AEE_SUCCESS) return err;

    hexkl_micro_hmx_setup_acc_read_f16(vtcm, cp.cfg_off);

    for (uint32_t row = 0; row < m; row += TILE) {
        uint32_t rt = row / TILE;

        /* Load activation strip to VTCM (once per row strip) */
        for (uint32_t i = 0; i < K; i++) {
            hexkl_micro_hmx_copy_submatrix_to_f16(
                vtcm, cp.staging_off, act, rt, i, m, k);
            hexkl_micro_hmx_rm_to_ah_f16(vtcm, ALIGN * i, cp.staging_off);
        }

        /* Stream column chunks through VTCM */
        for (uint32_t col_base = 0; col_base < Nc; col_base += cp.n_chunk_tiles) {
            uint32_t chunk_cols = Nc - col_base;
            if (chunk_cols > cp.n_chunk_tiles)
                chunk_cols = cp.n_chunk_tiles;

            /* Load weight chunk with format conversion (rm -> wh) */
            for (uint32_t c = 0; c < chunk_cols; c++)
                for (uint32_t i = 0; i < K; i++)
                    hexkl_micro_hmx_rm_to_wh_f16(vtcm,
                        cp.wt_base + (c * K + i) * ALIGN,
                        wt, i, col_base + c, n);

            /* Compute: for each column in this chunk */
            for (uint32_t c = 0; c < chunk_cols; c++) {
                hexkl_micro_hmx_acc_clear_f16();
                for (uint32_t i = 0; i < K; i++)
                    hexkl_micro_hmx_mm_f16(vtcm, ALIGN * i,
                                            cp.wt_base + (c * K + i) * ALIGN);
                hexkl_micro_hmx_acc_read_f16(vtcm, cp.cfg_off, cp.readback_off);
                hexkl_micro_hmx_ah_to_rm_f16(vtcm, cp.staging_off, cp.readback_off);
                hexkl_micro_hmx_copy_f16_to_f32_submatrix(
                    vtcm, cp.staging_off, out_f32, rt, col_base + c, m, n);
            }
        }
    }
    return AEE_SUCCESS;
}

/* ================================================================== */
/*  Non-streaming L2 preformatted (same as demo_llama_hmx.c)           */
/*  Included here for direct comparison -- will SKIP on large matrices */
/* ================================================================== */

static int hmx_l2_preformatted(
    uint8_t *vtcm, uint32_t vtcm_size,
    float *out_f32, const _Float16 *act, const _Float16 *wt_preformatted,
    uint32_t m, uint32_t n, uint32_t k)
{
    const uint32_t K  = (k + TILE - 1) / TILE;
    const uint32_t Nc = (n + TILE - 1) / TILE;
    const uint32_t Nwt = K * Nc;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    const uint32_t staging_off  = ALIGN * K;
    const uint32_t readback_off = ALIGN * (K + 1);
    const uint32_t wt_base      = ALIGN * (K + 2);
    uint32_t needed = wt_base + Nwt * ALIGN + hexkl_micro_hmx_config_size();

    if (needed > vtcm_size) return AEE_ENOMEMORY;

    uint32_t cfg_off = vtcm_size - hexkl_micro_hmx_config_size();
    hexkl_micro_hmx_setup_acc_read_f16(vtcm, cfg_off);

    uint32_t k_pad = K * TILE;
    uint32_t n_pad = Nc * TILE;

    /* Pre-load ALL weight tiles (requires all to fit in VTCM) */
    for (uint32_t ct = 0; ct < Nc; ct++)
        for (uint32_t kt = 0; kt < K; kt++)
            hexkl_micro_hmx_copy_psubmatrix_to_f16_weight(vtcm,
                wt_base + (ct * K + kt) * ALIGN,
                wt_preformatted, kt, ct, k_pad, n_pad);

    for (uint32_t row = 0; row < m; row += TILE) {
        for (uint32_t i = 0; i < K; i++) {
            hexkl_micro_hmx_copy_submatrix_to_f16(
                vtcm, staging_off, act, row / TILE, i, m, k);
            hexkl_micro_hmx_rm_to_ah_f16(vtcm, ALIGN * i, staging_off);
        }
        for (uint32_t col = 0; col < n; col += TILE) {
            uint32_t ct = col / TILE;
            hexkl_micro_hmx_acc_clear_f16();
            for (uint32_t i = 0; i < K; i++)
                hexkl_micro_hmx_mm_f16(vtcm, ALIGN * i,
                                        wt_base + (ct * K + i) * ALIGN);
            hexkl_micro_hmx_acc_read_f16(vtcm, cfg_off, readback_off);
            hexkl_micro_hmx_ah_to_rm_f16(vtcm, staging_off, readback_off);
            hexkl_micro_hmx_copy_f16_to_f32_submatrix(
                vtcm, staging_off, out_f32, row / TILE, col / TILE, m, n);
        }
    }
    return AEE_SUCCESS;
}

/* ================================================================== */
/*  Reference matmul + verification                                    */
/* ================================================================== */

static void ref_matmul(
    _Float16 *C, const _Float16 *A, const _Float16 *B,
    uint32_t m, uint32_t n, uint32_t k)
{
    for (uint32_t r = 0; r < m; r++)
        for (uint32_t c = 0; c < n; c++) {
            float acc = 0.0f;
            for (uint32_t p = 0; p < k; p++)
                acc += (float)A[r * k + p] * (float)B[p * n + c];
            C[r * n + c] = (_Float16)acc;
        }
}

static int verify_f32_vs_f16ref(
    const float *got, const _Float16 *ref,
    uint32_t count, float *max_diff)
{
    float md = 0.0f;
    int errors = 0;
    for (uint32_t i = 0; i < count; i++) {
        float r = (float)ref[i];
        float g = got[i];
        float diff = fabsf(r - g);
        if (diff > md) md = diff;
        float denom = fabsf(r) > 1e-6f ? fabsf(r) : 1e-6f;
        if (diff > 0.5f && (diff / denom) > 0.05f) errors++;
    }
    *max_diff = md;
    return errors == 0;
}

/* ================================================================== */
/*  Benchmark configurations                                           */
/* ================================================================== */

struct bench_config {
    const char *name;
    uint32_t m, n, k;
};

static struct bench_config configs[] = {
    /* Small configs (same as demo_llama_hmx.c for comparison) */
    {"Fwd L1  [128x128]  = [128x800]  @ [800x128]",    128,  128,  800},
    {"Medium  [128x256]  = [128x512]  @ [512x256]",     128,  256,  512},
    /* LLM-sized configs (these SKIP in L1/L2 due to VTCM limits) */
    {"LLM decode  [1x4096]   = [1x4096]   @ [4096x4096]",      1, 4096, 4096},
    {"LLM prefill [32x4096]  = [32x4096]  @ [4096x4096]",     32, 4096, 4096},
    {"LLM prefill [128x4096] = [128x4096] @ [4096x4096]",    128, 4096, 4096},
    {"LLM wide    [1x11008]  = [1x4096]   @ [4096x11008]",     1, 11008, 4096},
};
#define N_CONFIGS (sizeof(configs) / sizeof(configs[0]))

/* ================================================================== */
/*  Benchmark helpers                                                   */
/* ================================================================== */

typedef int (*matmul_fn)(uint8_t *, uint32_t, float *, const _Float16 *,
                          const _Float16 *, uint32_t, uint32_t, uint32_t);

static void bench_approach(
    const char *label,
    matmul_fn fn,
    uint8_t *vtcm, uint32_t vtcm_size,
    float *C_f32, const _Float16 *A, const _Float16 *B,
    const _Float16 *C_ref,
    uint32_t m, uint32_t n, uint32_t k, double flops)
{
    memset(C_f32, 0, m * n * sizeof(float));
    uint64_t total_us = 0;

    for (int iter = 0; iter < N_ITERS; iter++) {
        uint64_t t0 = HAP_perf_get_time_us();
        int err = fn(vtcm, vtcm_size, C_f32, A, B, m, n, k);
        uint64_t t1 = HAP_perf_get_time_us();
        if (err != AEE_SUCCESS) {
            FARF(ALWAYS, "[STREAM]   %-22s SKIP (err=%d, need too much VTCM)", label, err);
            return;
        }
        total_us += (t1 - t0);
    }

    double avg = (double)total_us / N_ITERS;
    double gflops = flops / (avg * 1e3);

    if (C_ref) {
        float md = 0;
        int pass = verify_f32_vs_f16ref(C_f32, C_ref, m * n, &md);
        FARF(ALWAYS, "[STREAM]   %-22s %7.0f us  %6.2f GFLOPS  %s  max_diff=%.3f",
             label, avg, gflops, pass ? "PASS" : "FAIL", (double)md);
    } else {
        FARF(ALWAYS, "[STREAM]   %-22s %7.0f us  %6.2f GFLOPS  (no ref)",
             label, avg, gflops);
    }
}

/* ================================================================== */
/*  Main benchmark                                                     */
/* ================================================================== */

int main(void) {
    int res = AEE_SUCCESS;
    uint8_t *vtcm = NULL;
    uint32_t vtcm_size = 0;

    FARF(ALWAYS, "[STREAM] === Experiment 5: VTCM Streaming for Large Matrices ===");

    res = hexkl_micro_hw_init(&vtcm, &vtcm_size);
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[STREAM] hw_init failed: %d", res);
        return res;
    }
    FARF(ALWAYS, "[STREAM] VTCM: %u KB at %p", vtcm_size / 1024, (void *)vtcm);

    res = hexkl_micro_hmx_lock();
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[STREAM] hmx_lock failed: %d", res);
        return res;
    }

    /* Report streaming parameters for the largest config */
    {
        uint32_t K_big = (4096 + TILE - 1) / TILE;   /* 128 tiles */
        struct chunk_params cp;
        int err = compute_chunk_params(vtcm_size, K_big, &cp);
        if (err == AEE_SUCCESS) {
            FARF(ALWAYS, "[STREAM] K=4096 -> %u K-tiles, %u col-tiles/chunk, "
                 "wt_base=%u KB",
                 K_big, cp.n_chunk_tiles, cp.wt_base / 1024);
        } else {
            FARF(ALWAYS, "[STREAM] K=4096 -> cannot fit even 1 column chunk");
        }
    }

    for (uint32_t ci = 0; ci < N_CONFIGS; ci++) {
        struct bench_config *cfg = &configs[ci];
        uint32_t m = cfg->m, n = cfg->n, k = cfg->k;
        double flops = 2.0 * m * n * k;

        FARF(ALWAYS, "[STREAM] ------------------------------------------------");
        FARF(ALWAYS, "[STREAM] %s", cfg->name);
        FARF(ALWAYS, "[STREAM] M=%u N=%u K=%u  FLOPS=%.1fM", m, n, k, flops / 1e6);

        /* Report chunk parameters for this config */
        {
            uint32_t K_t = (k + TILE - 1) / TILE;
            uint32_t Nc_t = (n + TILE - 1) / TILE;
            struct chunk_params cp;
            int err = compute_chunk_params(vtcm_size, K_t, &cp);
            if (err == AEE_SUCCESS) {
                uint32_t n_chunks = (Nc_t + cp.n_chunk_tiles - 1) / cp.n_chunk_tiles;
                FARF(ALWAYS, "[STREAM] K-tiles=%u  N-tiles=%u  chunk=%u cols/iter  "
                     "%u chunks total",
                     K_t, Nc_t, cp.n_chunk_tiles, n_chunks);
            }
        }

        _Float16 *A     = (_Float16 *)malloc(m * k * sizeof(_Float16));
        _Float16 *B     = (_Float16 *)malloc(k * n * sizeof(_Float16));
        _Float16 *C_ref = (_Float16 *)malloc(m * n * sizeof(_Float16));
        float    *C_f32 = (float *)malloc(m * n * sizeof(float));

        if (!A || !B || !C_ref || !C_f32) {
            FARF(ALWAYS, "[STREAM] malloc failed for M=%u N=%u K=%u", m, n, k);
            goto next;
        }

        /* Deterministic init */
        for (uint32_t i = 0; i < m * k; i++)
            A[i] = (_Float16)(0.01f * (float)((i % 97) - 48));
        for (uint32_t i = 0; i < k * n; i++)
            B[i] = (_Float16)(0.01f * (float)((i % 83) - 41));

        /* Reference matmul -- skip for large configs (scalar O(n^3) too slow) */
        double ref_ops = 2.0 * m * n * k;
        int skip_ref = (ref_ops > 500e6);  /* skip if > 500M FLOPS */
        if (skip_ref) {
            FARF(ALWAYS, "[STREAM] Skipping reference (%.0fM FLOPS too slow on scalar)",
                 ref_ops / 1e6);
            free(C_ref);
            C_ref = NULL;
        } else {
            FARF(ALWAYS, "[STREAM] Computing reference...");
            ref_matmul(C_ref, A, B, m, n, k);
        }

        /* (a) L2 preformatted -- non-streaming, will SKIP for large matrices */
        {
            uint32_t K_t  = (k + TILE - 1) / TILE;
            uint32_t Nc_t = (n + TILE - 1) / TILE;
            uint32_t k_pad = K_t * TILE;
            uint32_t n_pad = Nc_t * TILE;
            _Float16 *B_wh = (_Float16 *)calloc(k_pad * n_pad, sizeof(_Float16));
            if (B_wh) {
                for (uint32_t r = 0; r < k; r++)
                    memcpy(&B_wh[r * n_pad], &B[r * n], n * sizeof(_Float16));
                hexkl_macro_rm_to_wh_f16_inplace(k_pad, n_pad, B_wh);
                bench_approach("L2 preformatted", hmx_l2_preformatted,
                               vtcm, vtcm_size, C_f32, A, B_wh, C_ref,
                               m, n, k, flops);
                free(B_wh);
            } else {
                FARF(ALWAYS, "[STREAM]   L2 preformatted        SKIP (malloc failed)");
            }
        }

        /* (b) Streaming baseline -- rm_to_wh at load time */
        bench_approach("Stream baseline", hmx_streaming_baseline,
                       vtcm, vtcm_size, C_f32, A, B, C_ref,
                       m, n, k, flops);

        /* (c) Streaming preformatted -- pre-formatted WH, just memcpy load */
        {
            uint32_t K_t  = (k + TILE - 1) / TILE;
            uint32_t Nc_t = (n + TILE - 1) / TILE;
            uint32_t k_pad = K_t * TILE;
            uint32_t n_pad = Nc_t * TILE;
            _Float16 *B_wh = (_Float16 *)calloc(k_pad * n_pad, sizeof(_Float16));
            if (B_wh) {
                for (uint32_t r = 0; r < k; r++)
                    memcpy(&B_wh[r * n_pad], &B[r * n], n * sizeof(_Float16));
                hexkl_macro_rm_to_wh_f16_inplace(k_pad, n_pad, B_wh);
                bench_approach("Stream prefmt", hmx_streaming_preformatted,
                               vtcm, vtcm_size, C_f32, A, B_wh, C_ref,
                               m, n, k, flops);
                free(B_wh);
            } else {
                FARF(ALWAYS, "[STREAM]   Stream prefmt           SKIP (malloc failed)");
            }
        }

next:
        free(A); free(B); free(C_ref); free(C_f32);
    }

    hexkl_micro_hmx_unlock();
    FARF(ALWAYS, "[STREAM] === Done ===");
    return res;
}
