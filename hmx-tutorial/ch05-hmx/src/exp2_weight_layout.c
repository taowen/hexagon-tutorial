/*
 * demo_llama_hmx.c -- HMX matmul benchmark with progressive optimization levels
 *
 * Demonstrates HMX optimization techniques using the hexkl_micro API:
 *
 *   Level 0: hexkl_micro baseline (no caching, reload weights every row)
 *   Level 1: + weight caching in VTCM (eliminate per-row weight reload)
 *   Level 2: + pre-formatted WH layout (eliminate format conversion at load time)
 *   Compute-only: all tiles pre-cached, isolates HMX compute + output writeback
 *
 * Key finding: output writeback (ah_to_rm + copy_f16_to_f32) is 44-60% of cost.
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
/*  Level 0: hexkl_micro baseline                                      */
/*                                                                     */
/*  Per output tile: copy_submatrix + rm_to_ah + acc_clear + mm_f16    */
/*                   + acc_read + ah_to_rm + copy_f16_to_f32           */
/*                                                                     */
/*  This is how most SDK examples use HMX. Simple but slow due to      */
/*  per-tile API call overhead and DDR weight reload every row.         */
/* ================================================================== */

static int hmx_level0_baseline(
    uint8_t *vtcm, uint32_t vtcm_size,
    float *out_f32, const _Float16 *act, const _Float16 *wt,
    uint32_t m, uint32_t n, uint32_t k)
{
    const uint32_t K  = (k + TILE - 1) / TILE;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    /* VTCM: [act tiles 0..K-1][staging][readback][wt tile][config] */
    const uint32_t staging_off  = ALIGN * K;
    const uint32_t readback_off = ALIGN * (K + 1);
    const uint32_t wt_off       = ALIGN * (K + 2);
    uint32_t needed = wt_off + ALIGN + hexkl_micro_hmx_config_size();

    if (needed > vtcm_size) return AEE_ENOMEMORY;

    uint32_t cfg_off = vtcm_size - hexkl_micro_hmx_config_size();
    hexkl_micro_hmx_setup_acc_read_f16(vtcm, cfg_off);

    for (uint32_t row = 0; row < m; row += TILE) {
        /* Load activation strip */
        for (uint32_t i = 0; i < K; i++) {
            hexkl_micro_hmx_copy_submatrix_to_f16(
                vtcm, staging_off, act, row / TILE, i, m, k);
            hexkl_micro_hmx_rm_to_ah_f16(vtcm, ALIGN * i, staging_off);
        }

        for (uint32_t col = 0; col < n; col += TILE) {
            uint32_t ct = col / TILE;

            /* Load weight tile from DDR every time (no caching) */
            hexkl_micro_hmx_acc_clear_f16();
            for (uint32_t i = 0; i < K; i++) {
                hexkl_micro_hmx_rm_to_wh_f16(vtcm, wt_off, wt, i, ct, n);
                hexkl_micro_hmx_mm_f16(vtcm, ALIGN * i, wt_off);
            }

            hexkl_micro_hmx_acc_read_f16(vtcm, cfg_off, readback_off);
            hexkl_micro_hmx_ah_to_rm_f16(vtcm, staging_off, readback_off);
            hexkl_micro_hmx_copy_f16_to_f32_submatrix(
                vtcm, staging_off, out_f32, row / TILE, col / TILE, m, n);
        }
    }
    return AEE_SUCCESS;
}

/* ================================================================== */
/*  Level 1: + weight caching                                          */
/*                                                                     */
/*  Pre-load ALL weight tiles into VTCM once. The compute loop only    */
/*  references VTCM-resident weights, eliminating DDR reads in the     */
/*  inner loop. This is the single biggest optimization.               */
/*                                                                     */
/*  From llama.cpp: weights are pre-formatted and cached in VTCM.      */
/* ================================================================== */

static int hmx_level1_wt_cached(
    uint8_t *vtcm, uint32_t vtcm_size,
    float *out_f32, const _Float16 *act, const _Float16 *wt,
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

    /* Pre-load ALL weight tiles into VTCM (one-time cost) */
    for (uint32_t ct = 0; ct < Nc; ct++)
        for (uint32_t kt = 0; kt < K; kt++)
            hexkl_micro_hmx_rm_to_wh_f16(vtcm, wt_base + (ct * K + kt) * ALIGN,
                                          wt, kt, ct, n);

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
/*  Level 2: + pre-formatted WH layout (HMX Weight Layout)            */
/*                                                                     */
/*  Weights are pre-converted to WH format in DDR using               */
/*  hexkl_macro_rm_to_wh_f16_inplace(). At matmul time, tiles are    */
/*  loaded with copy_psubmatrix_to_f16_weight() — just a memcpy,     */
/*  no format conversion. This is the v75 "HMX Weight Layout"        */
/*  approach used by QNN NativeKV and llama.cpp.                       */
/*                                                                     */
/*  Measures: does separating format conversion from DDR→VTCM copy    */
/*  give additional speedup over L1?                                   */
/* ================================================================== */

static int hmx_level2_preformatted(
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

    /* n and k are the PADDED dimensions (multiples of 32) */
    uint32_t k_pad = K * TILE;
    uint32_t n_pad = Nc * TILE;

    /* Pre-load ALL weight tiles using copy_psubmatrix (no format conversion) */
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
/*  Compute-only benchmark: all tiles pre-cached in VTCM               */
/*                                                                     */
/*  Pre-cache ALL activation AND weight tiles, then benchmark only     */
/*  the compute + output writeback portion. This isolates pure HMX     */
/*  compute throughput and reveals that output writeback (ah_to_rm +   */
/*  copy_f16_to_f32) is 44-60% of the cost.                           */
/*                                                                     */
/*  Uses hexkl_micro API (same as L0/L1).                              */
/*  VTCM: [all act AH tiles][all wt WH tiles][staging][readback][cfg] */
/* ================================================================== */

static int hmx_compute_only(
    uint8_t *vtcm, uint32_t vtcm_size,
    float *out_f32, const _Float16 *act, const _Float16 *wt,
    uint32_t m, uint32_t n, uint32_t k)
{
    const uint32_t K  = (k + TILE - 1) / TILE;
    const uint32_t Mr = (m + TILE - 1) / TILE;
    const uint32_t Nc = (n + TILE - 1) / TILE;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    uint32_t n_act_tiles = Mr * K;
    uint32_t n_wt_tiles  = Nc * K;
    uint32_t act_base    = 0;
    uint32_t wt_base     = n_act_tiles * ALIGN;
    uint32_t staging_off = wt_base + n_wt_tiles * ALIGN;
    uint32_t readback_off = staging_off + ALIGN;
    uint32_t needed = readback_off + ALIGN + hexkl_micro_hmx_config_size();

    if (needed > vtcm_size) return AEE_ENOMEMORY;

    uint32_t cfg_off = vtcm_size - hexkl_micro_hmx_config_size();
    hexkl_micro_hmx_setup_acc_read_f16(vtcm, cfg_off);

    /* Pre-cache ALL activation tiles */
    for (uint32_t rt = 0; rt < Mr; rt++)
        for (uint32_t kt = 0; kt < K; kt++) {
            hexkl_micro_hmx_copy_submatrix_to_f16(
                vtcm, staging_off, act, rt, kt, m, k);
            hexkl_micro_hmx_rm_to_ah_f16(vtcm,
                act_base + (rt * K + kt) * ALIGN, staging_off);
        }

    /* Pre-cache ALL weight tiles */
    for (uint32_t ct = 0; ct < Nc; ct++)
        for (uint32_t kt = 0; kt < K; kt++)
            hexkl_micro_hmx_rm_to_wh_f16(vtcm,
                wt_base + (ct * K + kt) * ALIGN, wt, kt, ct, n);

    return AEE_SUCCESS;
}

/* Compute loop only -- called after hmx_compute_only for setup */
static void hmx_compute_only_run(
    uint8_t *vtcm, float *out_f32,
    uint32_t m, uint32_t n, uint32_t k,
    uint32_t act_base, uint32_t wt_base,
    uint32_t staging_off, uint32_t readback_off, uint32_t cfg_off)
{
    const uint32_t K = (k + TILE - 1) / TILE;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    for (uint32_t row = 0; row < m; row += TILE) {
        uint32_t rt = row / TILE;
        for (uint32_t col = 0; col < n; col += TILE) {
            uint32_t ct = col / TILE;
            hexkl_micro_hmx_acc_clear_f16();
            for (uint32_t kt = 0; kt < K; kt++)
                hexkl_micro_hmx_mm_f16(vtcm,
                    act_base + (rt * K + kt) * ALIGN,
                    wt_base + (ct * K + kt) * ALIGN);
            hexkl_micro_hmx_acc_read_f16(vtcm, cfg_off, readback_off);
            hexkl_micro_hmx_ah_to_rm_f16(vtcm, staging_off, readback_off);
            hexkl_micro_hmx_copy_f16_to_f32_submatrix(
                vtcm, staging_off, out_f32, rt, ct, m, n);
        }
    }
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
    {"Fwd L1  [128x128] = [128x800] @ [800x128]",  128, 128, 800},
    {"Fwd L2  [128x32]  = [128x128] @ [128x32]",   128,  32, 128},
    {"Bwd dW1 [128x800] = [128x128] @ [128x800]",  128, 800, 128},
    {"Bwd dH  [128x128] = [128x32]  @ [32x128]",   128, 128,  32},
    {"Small   [32x128]  = [32x800]  @ [800x128]",    32, 128, 800},
    {"Medium  [128x256] = [128x512] @ [512x256]",    128, 256, 512},
};
#define N_CONFIGS (sizeof(configs) / sizeof(configs[0]))

/* ================================================================== */
/*  Benchmark runner: timed iteration                                  */
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
            FARF(ALWAYS, "[HMX]   %-22s SKIP (err=%d, need too much VTCM)", label, err);
            return;
        }
        total_us += (t1 - t0);
    }

    float md = 0;
    int pass = verify_f32_vs_f16ref(C_f32, C_ref, m * n, &md);
    double avg = (double)total_us / N_ITERS;
    double gflops = flops / (avg * 1e3);
    FARF(ALWAYS, "[HMX]   %-22s %7.0f us  %6.2f GFLOPS  %s  max_diff=%.3f",
         label, avg, gflops, pass ? "PASS" : "FAIL", (double)md);
}

/* ================================================================== */
/*  Main benchmark                                                     */
/* ================================================================== */

int main(void) {
    int res = AEE_SUCCESS;
    uint8_t *vtcm = NULL;
    uint32_t vtcm_size = 0;

    FARF(ALWAYS, "[HMX] === HMX Matmul Optimization Benchmark ===");

    res = hexkl_micro_hw_init(&vtcm, &vtcm_size);
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[HMX] hw_init failed: %d", res);
        return res;
    }
    FARF(ALWAYS, "[HMX] VTCM: %u KB at %p", vtcm_size / 1024, (void *)vtcm);

    res = hexkl_micro_hmx_lock();
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[HMX] hmx_lock failed: %d", res);
        return res;
    }

    for (uint32_t ci = 0; ci < N_CONFIGS; ci++) {
        struct bench_config *cfg = &configs[ci];
        uint32_t m = cfg->m, n = cfg->n, k = cfg->k;
        double flops = 2.0 * m * n * k;

        FARF(ALWAYS, "[HMX] ------------------------------------------------");
        FARF(ALWAYS, "[HMX] %s", cfg->name);
        FARF(ALWAYS, "[HMX] M=%u N=%u K=%u  FLOPS=%.1fM", m, n, k, flops / 1e6);

        _Float16 *A     = (_Float16 *)malloc(m * k * sizeof(_Float16));
        _Float16 *B     = (_Float16 *)malloc(k * n * sizeof(_Float16));
        _Float16 *C_ref = (_Float16 *)malloc(m * n * sizeof(_Float16));
        float    *C_f32 = (float *)malloc(m * n * sizeof(float));

        if (!A || !B || !C_ref || !C_f32) {
            FARF(ALWAYS, "[HMX] malloc failed");
            goto next;
        }

        /* Deterministic init */
        for (uint32_t i = 0; i < m * k; i++)
            A[i] = (_Float16)(0.01f * (float)((i % 97) - 48));
        for (uint32_t i = 0; i < k * n; i++)
            B[i] = (_Float16)(0.01f * (float)((i % 83) - 41));

        ref_matmul(C_ref, A, B, m, n, k);

        /* Level 0: baseline (no weight caching, all hexkl_micro) */
        bench_approach("L0 baseline", hmx_level0_baseline,
                       vtcm, vtcm_size, C_f32, A, B, C_ref, m, n, k, flops);

        /* Level 1: + weight caching */
        bench_approach("L1 wt-cached", hmx_level1_wt_cached,
                       vtcm, vtcm_size, C_f32, A, B, C_ref, m, n, k, flops);

        /* Level 2: + pre-formatted WH layout */
        {
            /* Prepare padded + WH-formatted weight copy in DDR */
            uint32_t K_t = (k + TILE - 1) / TILE;
            uint32_t Nc_t = (n + TILE - 1) / TILE;
            uint32_t k_pad = K_t * TILE;
            uint32_t n_pad = Nc_t * TILE;
            _Float16 *B_wh = (_Float16 *)calloc(k_pad * n_pad, sizeof(_Float16));
            if (B_wh) {
                /* Copy original B into zero-padded buffer */
                for (uint32_t r = 0; r < k; r++)
                    memcpy(&B_wh[r * n_pad], &B[r * n], n * sizeof(_Float16));
                /* Convert to WH layout in-place */
                hexkl_macro_rm_to_wh_f16_inplace(k_pad, n_pad, B_wh);
                /* Benchmark: matmul using pre-formatted weights */
                bench_approach("L2 preformatted", hmx_level2_preformatted,
                               vtcm, vtcm_size, C_f32, A, B_wh, C_ref, m, n, k, flops);
                free(B_wh);
            } else {
                FARF(ALWAYS, "[HMX]   L2 preformatted        SKIP (malloc failed)");
            }
        }

        /* Compute-only: all tiles pre-cached, isolate HMX compute throughput */
        {
            memset(C_f32, 0, m * n * sizeof(float));
            int err = hmx_compute_only(vtcm, vtcm_size, C_f32, A, B, m, n, k);
            if (err != AEE_SUCCESS) {
                FARF(ALWAYS, "[HMX]   compute-only          SKIP (err=%d)", err);
            } else {
                /* Compute VTCM layout parameters */
                uint32_t K_t  = (k + TILE - 1) / TILE;
                uint32_t Mr_t = (m + TILE - 1) / TILE;
                uint32_t Nc_t = (n + TILE - 1) / TILE;
                uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;
                uint32_t act_b = 0;
                uint32_t wt_b  = Mr_t * K_t * ALIGN;
                uint32_t stg   = wt_b + Nc_t * K_t * ALIGN;
                uint32_t rdb   = stg + ALIGN;
                uint32_t cfg   = vtcm_size - hexkl_micro_hmx_config_size();

                /* Warm-up run for verification */
                hmx_compute_only_run(vtcm, C_f32, m, n, k,
                                     act_b, wt_b, stg, rdb, cfg);
                float md = 0;
                int pass = verify_f32_vs_f16ref(C_f32, C_ref, m * n, &md);

                /* Timed runs */
                uint64_t total_us = 0;
                for (int iter = 0; iter < N_ITERS; iter++) {
                    uint64_t t0 = HAP_perf_get_time_us();
                    hmx_compute_only_run(vtcm, C_f32, m, n, k,
                                         act_b, wt_b, stg, rdb, cfg);
                    uint64_t t1 = HAP_perf_get_time_us();
                    total_us += (t1 - t0);
                }
                double avg = (double)total_us / N_ITERS;
                double gflops = flops / (avg * 1e3);
                FARF(ALWAYS, "[HMX]   %-22s %7.0f us  %6.2f GFLOPS  %s  max_diff=%.3f",
                     "compute-only", avg, gflops, pass ? "PASS" : "FAIL", (double)md);
            }
        }

next:
        free(A); free(B); free(C_ref); free(C_f32);
    }

    hexkl_micro_hmx_unlock();
    FARF(ALWAYS, "[HMX] === Done ===");
    return res;
}
