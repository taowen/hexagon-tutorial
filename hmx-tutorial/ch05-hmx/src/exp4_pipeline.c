/*
 * demo_direct_hmx.c -- HMX API overhead breakdown + htp-ops-lib output path
 *
 * Four approaches compared (all use hexkl for tile preparation):
 *
 * Method A: hexkl full — acc_clear + mm_f16(×K) + acc_read + ah_to_rm + copy_f16_to_f32
 * Method B: hexkl breakdown — same as A, timing each phase separately
 * Method C: direct ASM compute + hexkl readback (proves compute overhead ≈ 0)
 * Method D: direct ASM compute + HVX output (htp-ops-lib approach, eliminates f16→f32 bottleneck)
 *
 * Key insight from Experiment 6 data:
 *   - hexkl_micro API call overhead ≈ 0 (mm_f16 = 0.01 µs/call)
 *   - The REAL bottleneck: copy_f16_to_f32_submatrix = 30 µs/tile (99% of compute time)
 *   - Method D replaces hexkl's per-tile readback with:
 *     1. hmx_consume_accumulator_fp16 → stores f16 tile directly in VTCM
 *     2. HVX batch conversion → reads interleaved tile, outputs f32 row-major
 *     This is exactly how htp-ops-lib does it (transfer_output_chunk_fp16_to_fp32).
 *
 * Compile: hexagon-clang -mv75 -mhvx -mhvx-length=128B -mhmx -O3 ...
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

/* ================================================================== */
/*  Constants                                                          */
/* ================================================================== */

#define TILE              32
#define TILE_ELMS         (TILE * TILE)      /* 1024 f16 elements per tile */
#define TILE_BYTES        (TILE_ELMS * 2)    /* 2048 bytes per 32x32 f16 tile */
#define N_ITERS           1000

/* ================================================================== */
/*  HMX inline assembly wrappers (from htp-ops-lib hmx_utils.h)       */
/* ================================================================== */

static inline __attribute__((always_inline)) void hmx_clear_acc(void) {
    asm volatile("mxclracc.hf" ::: "memory");
}

static inline __attribute__((always_inline)) void hmx_set_scales(const void *scales) {
    asm volatile("bias = mxmem2(%0)" :: "r"(scales) : "memory");
}

/*
 * Batch-load activation and weight tiles into HMX.
 * The limit parameter (n_tiles * 2048 - 1) tells HMX to load n_tiles
 * consecutive tiles from contiguous memory, eliminating per-tile API overhead.
 */
static inline __attribute__((always_inline)) void hmx_load_tiles(
    const void *act_tiles, const void *wt_tiles, uint32_t n_tiles)
{
    uint32_t limit = n_tiles * TILE_BYTES - 1;
    asm volatile(
        "{ activation.hf = mxmem(%0, %1):deep\n"
        "  weight.hf = mxmem(%2, %3) }\n"
        :: "r"(act_tiles), "r"(limit), "r"(wt_tiles), "r"(limit)
        : "memory");
}

/*
 * Store HMX accumulator as f16 tile to VTCM.
 * cvt.hf = acc(2) converts accumulator to f16 with rounding mode 2.
 * mxmem(out, 0) = cvt stores the 32x32 f16 tile.
 */
static inline __attribute__((always_inline)) void hmx_store_acc(_Float16 *out) {
    asm volatile(
        "cvt.hf = acc(%0)\n"
        "mxmem(%1, %2) = cvt\n"
        :: "r"(2), "r"(out), "r"(0)
        : "memory");
}

/* ================================================================== */
/*  HVX f16↔f32 conversion (from htp-ops-lib hvx_convert.h)           */
/*                                                                     */
/*  These handle the interleaved tile format:                          */
/*    tile[r/2] = {row_r[0], row_{r+1}[0], row_r[1], ...}            */
/*                                                                     */
/*  hvx_vhf_to_wsf: interleaved f16 vector → 2 × f32 vectors         */
/*    lo = 32 f32 from even rows, hi = 32 f32 from odd rows           */
/* ================================================================== */

static inline __attribute__((always_inline))
HVX_Vector hvx_vhf_to_vqf16(HVX_Vector vx) {
    return Q6_Vqf16_vadd_VhfVhf(vx, Q6_V_vzero());
}

static inline __attribute__((always_inline))
HVX_VectorPair hvx_vqf16_to_wqf32(HVX_Vector v_src) {
    const HVX_Vector v_lo_mask = Q6_V_vsplat_R(0x0000ffff);
    const HVX_Vector v_hi_mask = Q6_V_vsplat_R(0xffff0000);
    const HVX_Vector v_shift16 = Q6_V_vsplat_R(16);

    HVX_Vector exp_comp = Q6_V_vand_VV(v_src, Q6_Vh_vsplat_R(0x1f));
    HVX_Vector mantissa = Q6_V_vand_VV(v_src, Q6_Vh_vsplat_R(0xffe0));
    exp_comp = Q6_Vh_vadd_VhVh(exp_comp, Q6_Vh_vsplat_R(112));

    HVX_Vector exp_comp0 = Q6_V_vand_VV(exp_comp, v_lo_mask);
    HVX_Vector exp_comp1 = Q6_Vw_vlsr_VwVw(exp_comp, v_shift16);
    HVX_Vector mantissa0 = Q6_Vw_vasl_VwVw(mantissa, v_shift16);
    HVX_Vector mantissa1 = Q6_V_vand_VV(mantissa, v_hi_mask);

    HVX_Vector v0_qf32 = Q6_Vw_vadd_VwVw(mantissa0, exp_comp0);
    HVX_Vector v1_qf32 = Q6_Vw_vadd_VwVw(mantissa1, exp_comp1);
    return Q6_W_vcombine_VV(v1_qf32, v0_qf32);
}

/* Convert interleaved f16 tile vector → two f32 vectors (row pair) */
static inline __attribute__((always_inline))
HVX_VectorPair hvx_vhf_to_wsf(HVX_Vector vx) {
    HVX_VectorPair vp = hvx_vqf16_to_wqf32(hvx_vhf_to_vqf16(vx));
    HVX_Vector v0_sf = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
    HVX_Vector v1_sf = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
    return Q6_W_vcombine_VV(v1_sf, v0_sf);
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
    {"Fwd L1  [128x128]",  128, 128, 800},
    {"Fwd L2  [128x32]",   128,  32, 128},
    {"Bwd dW1 [128x800]",  128, 800, 128},
    {"Bwd dH  [128x128]",  128, 128,  32},
    {"Small   [32x128]",    32, 128, 800},
    {"Medium  [128x256]",  128, 256, 512},
};
#define N_CONFIGS (sizeof(configs) / sizeof(configs[0]))

/* ================================================================== */
/*  Setup: pre-cache ALL tiles in VTCM using hexkl_micro               */
/*                                                                     */
/*  VTCM layout:                                                       */
/*    [act AH tiles: Mr*K * ALIGN]                                     */
/*    [wt  WH tiles: Nc*K * ALIGN]                                     */
/*    [output tiles: Mr*Nc * ALIGN]  (for Method D)                    */
/*    [scales:       256 bytes]       (for Method D)                   */
/*    [staging:      ALIGN]                                            */
/*    [readback:     ALIGN]                                            */
/*    ...                                                              */
/*    [config:       config_size at end of VTCM]                       */
/* ================================================================== */

static int setup_all_tiles(
    uint8_t *vtcm, uint32_t vtcm_size,
    const _Float16 *act, const _Float16 *wt,
    uint32_t m, uint32_t n, uint32_t k,
    uint32_t *out_act_base, uint32_t *out_wt_base,
    uint32_t *out_tile_base, uint32_t *out_scales,
    uint32_t *out_staging, uint32_t *out_readback, uint32_t *out_cfg)
{
    const uint32_t K  = (k + TILE - 1) / TILE;
    const uint32_t Mr = (m + TILE - 1) / TILE;
    const uint32_t Nc = (n + TILE - 1) / TILE;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    uint32_t n_act = Mr * K;
    uint32_t n_wt  = Nc * K;
    uint32_t n_out = Mr * Nc;
    uint32_t act_base   = 0;
    uint32_t wt_base    = n_act * ALIGN;
    uint32_t tile_base  = wt_base + n_wt * ALIGN;
    /* Scales: 256 bytes, 256-aligned */
    uint32_t scales_off = tile_base + n_out * ALIGN;
    scales_off = (scales_off + 255) & ~255u;
    uint32_t staging    = scales_off + 256;
    uint32_t readback   = staging + ALIGN;
    uint32_t cfg_size   = hexkl_micro_hmx_config_size();
    uint32_t needed     = readback + ALIGN + cfg_size;

    if (needed > vtcm_size) return AEE_ENOMEMORY;

    uint32_t cfg = vtcm_size - cfg_size;
    hexkl_micro_hmx_setup_acc_read_f16(vtcm, cfg);

    /* Load ALL activation tiles into AH format using hexkl_micro */
    for (uint32_t rt = 0; rt < Mr; rt++)
        for (uint32_t kt = 0; kt < K; kt++) {
            hexkl_micro_hmx_copy_submatrix_to_f16(
                vtcm, staging, act, rt, kt, m, k);
            hexkl_micro_hmx_rm_to_ah_f16(
                vtcm, act_base + (rt * K + kt) * ALIGN, staging);
        }

    /* Load ALL weight tiles into WH format using hexkl_micro */
    for (uint32_t ct = 0; ct < Nc; ct++)
        for (uint32_t kt = 0; kt < K; kt++)
            hexkl_micro_hmx_rm_to_wh_f16(
                vtcm, wt_base + (ct * K + kt) * ALIGN,
                wt, kt, ct, n);

    /* Initialize scales for Method D: f16 1.0 */
    HVX_Vector v_scale = Q6_Vh_vsplat_R(0x3C00);
    HVX_Vector *pv = (HVX_Vector *)(vtcm + scales_off);
    pv[0] = v_scale;
    pv[1] = Q6_V_vzero();

    *out_act_base  = act_base;
    *out_wt_base   = wt_base;
    *out_tile_base = tile_base;
    *out_scales    = scales_off;
    *out_staging   = staging;
    *out_readback  = readback;
    *out_cfg       = cfg;
    return AEE_SUCCESS;
}

/* ================================================================== */
/*  Method A: hexkl_micro compute loop (baseline)                      */
/*                                                                     */
/*  Per output tile: acc_clear + mm_f16(×K) + acc_read + ah_to_rm     */
/*                   + copy_f16_to_f32 = (3+K) API calls               */
/* ================================================================== */

static void compute_hexkl(
    uint8_t *vtcm, float *out_f32,
    uint32_t m, uint32_t n, uint32_t k,
    uint32_t act_base, uint32_t wt_base,
    uint32_t staging, uint32_t readback, uint32_t cfg)
{
    const uint32_t K     = (k + TILE - 1) / TILE;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    for (uint32_t row = 0; row < m; row += TILE) {
        uint32_t rt = row / TILE;
        for (uint32_t col = 0; col < n; col += TILE) {
            uint32_t ct = col / TILE;

            hexkl_micro_hmx_acc_clear_f16();
            for (uint32_t kt = 0; kt < K; kt++)
                hexkl_micro_hmx_mm_f16(vtcm,
                    act_base + (rt * K + kt) * ALIGN,
                    wt_base  + (ct * K + kt) * ALIGN);

            hexkl_micro_hmx_acc_read_f16(vtcm, cfg, readback);
            hexkl_micro_hmx_ah_to_rm_f16(vtcm, staging, readback);
            hexkl_micro_hmx_copy_f16_to_f32_submatrix(
                vtcm, staging, out_f32, rt, ct, m, n);
        }
    }
}

/* ================================================================== */
/*  Method B: hexkl_micro compute with per-phase breakdown timing      */
/* ================================================================== */

static void compute_hexkl_breakdown(
    uint8_t *vtcm, float *out_f32,
    uint32_t m, uint32_t n, uint32_t k,
    uint32_t act_base, uint32_t wt_base,
    uint32_t staging, uint32_t readback, uint32_t cfg,
    uint64_t *t_clear, uint64_t *t_mm,
    uint64_t *t_readback_out, uint64_t *t_writeback)
{
    const uint32_t K     = (k + TILE - 1) / TILE;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;
    *t_clear = *t_mm = *t_readback_out = *t_writeback = 0;

    for (uint32_t row = 0; row < m; row += TILE) {
        uint32_t rt = row / TILE;
        for (uint32_t col = 0; col < n; col += TILE) {
            uint32_t ct = col / TILE;

            uint64_t t0 = HAP_perf_get_time_us();
            hexkl_micro_hmx_acc_clear_f16();
            uint64_t t1 = HAP_perf_get_time_us();
            *t_clear += (t1 - t0);

            t0 = HAP_perf_get_time_us();
            for (uint32_t kt = 0; kt < K; kt++)
                hexkl_micro_hmx_mm_f16(vtcm,
                    act_base + (rt * K + kt) * ALIGN,
                    wt_base  + (ct * K + kt) * ALIGN);
            t1 = HAP_perf_get_time_us();
            *t_mm += (t1 - t0);

            t0 = HAP_perf_get_time_us();
            hexkl_micro_hmx_acc_read_f16(vtcm, cfg, readback);
            hexkl_micro_hmx_ah_to_rm_f16(vtcm, staging, readback);
            t1 = HAP_perf_get_time_us();
            *t_readback_out += (t1 - t0);

            t0 = HAP_perf_get_time_us();
            hexkl_micro_hmx_copy_f16_to_f32_submatrix(
                vtcm, staging, out_f32, rt, ct, m, n);
            t1 = HAP_perf_get_time_us();
            *t_writeback += (t1 - t0);
        }
    }
}

/* ================================================================== */
/*  Method C: direct ASM compute + hexkl readback                      */
/*                                                                     */
/*  Replace acc_clear + mm_f16(×K) with direct HMX assembly.          */
/*  Keep hexkl for readback. Proves compute API overhead ≈ 0.          */
/* ================================================================== */

static void compute_direct_asm(
    uint8_t *vtcm, float *out_f32,
    uint32_t m, uint32_t n, uint32_t k,
    uint32_t act_base, uint32_t wt_base,
    uint32_t staging, uint32_t readback, uint32_t cfg)
{
    const uint32_t K     = (k + TILE - 1) / TILE;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    for (uint32_t row = 0; row < m; row += TILE) {
        uint32_t rt = row / TILE;
        for (uint32_t col = 0; col < n; col += TILE) {
            uint32_t ct = col / TILE;

            hmx_clear_acc();

            const void *act_ptr = (const void *)(vtcm + act_base + rt * K * ALIGN);
            const void *wt_ptr  = (const void *)(vtcm + wt_base  + ct * K * ALIGN);

            for (uint32_t kt = 0; kt < K; kt += 32) {
                uint32_t batch = (K - kt < 32) ? K - kt : 32;
                hmx_load_tiles(
                    (const void *)((uint8_t *)act_ptr + kt * ALIGN),
                    (const void *)((uint8_t *)wt_ptr  + kt * ALIGN),
                    batch);
            }

            hexkl_micro_hmx_acc_read_f16(vtcm, cfg, readback);
            hexkl_micro_hmx_ah_to_rm_f16(vtcm, staging, readback);
            hexkl_micro_hmx_copy_f16_to_f32_submatrix(
                vtcm, staging, out_f32, rt, ct, m, n);
        }
    }
}

/* ================================================================== */
/*  Method D: direct ASM compute + HVX output (htp-ops-lib approach)   */
/*                                                                     */
/*  Full pipeline without hexkl for compute+output:                    */
/*  1. mxclracc.hf → clear accumulator                                */
/*  2. hmx_load_tiles → batch load K tiles                             */
/*  3. hmx_store_acc → write f16 tile to VTCM (NOT back to DDR!)      */
/*  4. HVX hvx_vhf_to_wsf → batch convert tiles to f32 row-major      */
/*                                                                     */
/*  This eliminates hexkl's copy_f16_to_f32_submatrix (30 µs/tile)    */
/*  by keeping data in VTCM and using HVX for the format conversion.   */
/* ================================================================== */

static void compute_direct_full(
    uint8_t *vtcm, float *out_f32,
    uint32_t m, uint32_t n, uint32_t k,
    uint32_t act_base, uint32_t wt_base,
    uint32_t tile_base, uint32_t scales_off)
{
    const uint32_t K     = (k + TILE - 1) / TILE;
    const uint32_t Mr    = (m + TILE - 1) / TILE;
    const uint32_t Nc    = (n + TILE - 1) / TILE;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    hmx_set_scales(vtcm + scales_off);

    /* Phase 1: HMX compute — all output tiles */
    for (uint32_t rt = 0; rt < Mr; rt++) {
        for (uint32_t ct = 0; ct < Nc; ct++) {
            hmx_clear_acc();

            const void *act_ptr = (const void *)(vtcm + act_base + rt * K * ALIGN);
            const void *wt_ptr  = (const void *)(vtcm + wt_base  + ct * K * ALIGN);

            for (uint32_t kt = 0; kt < K; kt += 32) {
                uint32_t batch = (K - kt < 32) ? K - kt : 32;
                hmx_load_tiles(
                    (const void *)((uint8_t *)act_ptr + kt * ALIGN),
                    (const void *)((uint8_t *)wt_ptr  + kt * ALIGN),
                    batch);
            }

            _Float16 *out_tile = (_Float16 *)(vtcm + tile_base + (rt * Nc + ct) * ALIGN);
            hmx_store_acc(out_tile);
        }
    }

    /* Phase 2: HVX output conversion — tile f16 → row-major f32
     * Following htp-ops-lib transfer_output_chunk_fp16_to_fp32:
     * Each tile vector[r1/2] contains interleaved row pair,
     * hvx_vhf_to_wsf unpacks to lo=row_r(f32), hi=row_{r+1}(f32) */
    for (uint32_t r = 0; r < m; r += 2) {
        uint32_t r0 = r / TILE;
        uint32_t r1 = r % TILE;

        for (uint32_t c = 0; c < n; c += TILE) {
            uint32_t c0 = c / TILE;
            const _Float16 *tile = (_Float16 *)(vtcm + tile_base + (r0 * Nc + c0) * ALIGN);

            HVX_Vector v_src = ((const HVX_Vector *)tile)[r1 / 2];
            HVX_VectorPair vp = hvx_vhf_to_wsf(v_src);

            /* lo = 32 f32 for row r, hi = 32 f32 for row r+1 */
            if (c + TILE <= n) {
                *(HVX_Vector *)(out_f32 + r * n + c) = Q6_V_lo_W(vp);
                if (r + 1 < m)
                    *(HVX_Vector *)(out_f32 + (r + 1) * n + c) = Q6_V_hi_W(vp);
            } else {
                /* Partial tile: copy element by element */
                float tmp0[32], tmp1[32];
                HVX_Vector v_lo = Q6_V_lo_W(vp);
                HVX_Vector v_hi = Q6_V_hi_W(vp);
                memcpy(tmp0, &v_lo, 32 * sizeof(float));
                memcpy(tmp1, &v_hi, 32 * sizeof(float));
                for (uint32_t j = 0; c + j < n; j++) {
                    out_f32[r * n + c + j] = tmp0[j];
                    if (r + 1 < m)
                        out_f32[(r + 1) * n + c + j] = tmp1[j];
                }
            }
        }
    }
}

/* Method D with per-phase timing */
static void compute_direct_full_timed(
    uint8_t *vtcm, float *out_f32,
    uint32_t m, uint32_t n, uint32_t k,
    uint32_t act_base, uint32_t wt_base,
    uint32_t tile_base, uint32_t scales_off,
    uint64_t *t_compute, uint64_t *t_output)
{
    const uint32_t K     = (k + TILE - 1) / TILE;
    const uint32_t Mr    = (m + TILE - 1) / TILE;
    const uint32_t Nc    = (n + TILE - 1) / TILE;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    hmx_set_scales(vtcm + scales_off);

    uint64_t t0 = HAP_perf_get_time_us();

    for (uint32_t rt = 0; rt < Mr; rt++) {
        for (uint32_t ct = 0; ct < Nc; ct++) {
            hmx_clear_acc();

            const void *act_ptr = (const void *)(vtcm + act_base + rt * K * ALIGN);
            const void *wt_ptr  = (const void *)(vtcm + wt_base  + ct * K * ALIGN);

            for (uint32_t kt = 0; kt < K; kt += 32) {
                uint32_t batch = (K - kt < 32) ? K - kt : 32;
                hmx_load_tiles(
                    (const void *)((uint8_t *)act_ptr + kt * ALIGN),
                    (const void *)((uint8_t *)wt_ptr  + kt * ALIGN),
                    batch);
            }

            _Float16 *out_tile = (_Float16 *)(vtcm + tile_base + (rt * Nc + ct) * ALIGN);
            hmx_store_acc(out_tile);
        }
    }

    uint64_t t1 = HAP_perf_get_time_us();
    *t_compute = t1 - t0;

    t0 = HAP_perf_get_time_us();

    for (uint32_t r = 0; r < m; r += 2) {
        uint32_t r0 = r / TILE;
        uint32_t r1 = r % TILE;

        for (uint32_t c = 0; c < n; c += TILE) {
            uint32_t c0 = c / TILE;
            const _Float16 *tile = (_Float16 *)(vtcm + tile_base + (r0 * Nc + c0) * ALIGN);

            HVX_Vector v_src = ((const HVX_Vector *)tile)[r1 / 2];
            HVX_VectorPair vp = hvx_vhf_to_wsf(v_src);

            if (c + TILE <= n) {
                *(HVX_Vector *)(out_f32 + r * n + c) = Q6_V_lo_W(vp);
                if (r + 1 < m)
                    *(HVX_Vector *)(out_f32 + (r + 1) * n + c) = Q6_V_hi_W(vp);
            } else {
                float tmp0[32], tmp1[32];
                HVX_Vector v_lo2 = Q6_V_lo_W(vp);
                HVX_Vector v_hi2 = Q6_V_hi_W(vp);
                memcpy(tmp0, &v_lo2, 32 * sizeof(float));
                memcpy(tmp1, &v_hi2, 32 * sizeof(float));
                for (uint32_t j = 0; c + j < n; j++) {
                    out_f32[r * n + c + j] = tmp0[j];
                    if (r + 1 < m)
                        out_f32[(r + 1) * n + c + j] = tmp1[j];
                }
            }
        }
    }

    t1 = HAP_perf_get_time_us();
    *t_output = t1 - t0;
}

/* ================================================================== */
/*  Main benchmark                                                     */
/* ================================================================== */

int main(void) {
    int res = AEE_SUCCESS;
    uint8_t *vtcm = NULL;
    uint32_t vtcm_size = 0;

    FARF(ALWAYS, "[DIRECT] === HMX Compute Pipeline Comparison ===");
    FARF(ALWAYS, "[DIRECT] Method A: hexkl full (baseline)");
    FARF(ALWAYS, "[DIRECT] Method B: hexkl per-phase breakdown");
    FARF(ALWAYS, "[DIRECT] Method C: direct ASM compute + hexkl readback");
    FARF(ALWAYS, "[DIRECT] Method D: direct ASM compute + HVX output (htp-ops-lib)");

    if (HEXKL_HMX_ACTIVATION_ALIGNMENT != TILE_BYTES) {
        FARF(ALWAYS, "[DIRECT] WARNING: ALIGN=%u != TILE_BYTES=%u",
             HEXKL_HMX_ACTIVATION_ALIGNMENT, TILE_BYTES);
    }

    res = hexkl_micro_hw_init(&vtcm, &vtcm_size);
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[DIRECT] hw_init failed: %d", res);
        return res;
    }
    FARF(ALWAYS, "[DIRECT] VTCM: %u KB at %p", vtcm_size / 1024, (void *)vtcm);

    res = hexkl_micro_hmx_lock();
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[DIRECT] hmx_lock failed: %d", res);
        return res;
    }

    for (uint32_t ci = 0; ci < N_CONFIGS; ci++) {
        struct bench_config *bcfg = &configs[ci];
        uint32_t m = bcfg->m, n = bcfg->n, k = bcfg->k;
        uint32_t K  = (k + TILE - 1) / TILE;
        uint32_t Mr = (m + TILE - 1) / TILE;
        uint32_t Nc = (n + TILE - 1) / TILE;
        uint32_t n_output_tiles = Mr * Nc;
        double flops = 2.0 * m * n * k;

        FARF(ALWAYS, "[DIRECT] ------------------------------------------------");
        FARF(ALWAYS, "[DIRECT] %s  M=%u N=%u K=%u  FLOPS=%.1fM",
             bcfg->name, m, n, k, flops / 1e6);
        FARF(ALWAYS, "[DIRECT]   tiles: %u row x %u col x %u K = %u output tiles",
             Mr, Nc, K, n_output_tiles);

        _Float16 *A     = (_Float16 *)memalign(128, m * k * sizeof(_Float16));
        _Float16 *B     = (_Float16 *)memalign(128, k * n * sizeof(_Float16));
        _Float16 *C_ref = (_Float16 *)malloc(m * n * sizeof(_Float16));
        float    *C_f32 = (float *)memalign(128, m * n * sizeof(float));

        if (!A || !B || !C_ref || !C_f32) {
            FARF(ALWAYS, "[DIRECT] malloc failed");
            goto next;
        }

        /* Deterministic init */
        for (uint32_t i = 0; i < m * k; i++)
            A[i] = (_Float16)(0.01f * (float)((i % 97) - 48));
        for (uint32_t i = 0; i < k * n; i++)
            B[i] = (_Float16)(0.01f * (float)((i % 83) - 41));

        ref_matmul(C_ref, A, B, m, n, k);

        /* Setup all tiles in VTCM */
        uint32_t act_base, wt_base, tile_base, scales_off, staging, readback, cfg_off;
        {
            int err = setup_all_tiles(vtcm, vtcm_size, A, B, m, n, k,
                                      &act_base, &wt_base, &tile_base,
                                      &scales_off, &staging, &readback, &cfg_off);
            if (err != AEE_SUCCESS) {
                FARF(ALWAYS, "[DIRECT]   SKIP (VTCM too small, need more than %u KB)",
                     vtcm_size / 1024);
                goto next;
            }
        }

        /* ---- Method A: hexkl compute (wrap all iters in one timer) ---- */
        {
            /* Warmup */
            compute_hexkl(vtcm, C_f32, m, n, k,
                          act_base, wt_base, staging, readback, cfg_off);
            uint64_t t0 = HAP_perf_get_time_us();
            for (int iter = 0; iter < N_ITERS; iter++)
                compute_hexkl(vtcm, C_f32, m, n, k,
                              act_base, wt_base, staging, readback, cfg_off);
            uint64_t t1 = HAP_perf_get_time_us();
            float md = 0;
            int pass = verify_f32_vs_f16ref(C_f32, C_ref, m * n, &md);
            double avg = (double)(t1 - t0) / N_ITERS;
            double gflops = avg > 0 ? flops / (avg * 1e3) : 0;
            FARF(ALWAYS, "[DIRECT]   A hexkl:     %7.1f us  %6.2f GFLOPS  %s  max_diff=%.3f",
                 avg, gflops, pass ? "PASS" : "FAIL", (double)md);
        }

        /* ---- Method B: hexkl breakdown (wrap all iters per-phase) ---- */
        {
            /* Warmup */
            compute_hexkl(vtcm, C_f32, m, n, k,
                          act_base, wt_base, staging, readback, cfg_off);
            uint64_t sum_clear = 0, sum_mm = 0, sum_rb = 0, sum_wb = 0;
            for (int iter = 0; iter < N_ITERS; iter++) {
                uint64_t t_clear, t_mm, t_rb, t_wb;
                compute_hexkl_breakdown(vtcm, C_f32, m, n, k,
                                        act_base, wt_base, staging,
                                        readback, cfg_off,
                                        &t_clear, &t_mm, &t_rb, &t_wb);
                sum_clear += t_clear;
                sum_mm    += t_mm;
                sum_rb    += t_rb;
                sum_wb    += t_wb;
            }
            double avg_clear = (double)sum_clear / N_ITERS;
            double avg_mm    = (double)sum_mm / N_ITERS;
            double avg_rb    = (double)sum_rb / N_ITERS;
            double avg_wb    = (double)sum_wb / N_ITERS;
            double total     = avg_clear + avg_mm + avg_rb + avg_wb;

            FARF(ALWAYS, "[DIRECT]   B breakdown (avg of %d runs):", N_ITERS);
            FARF(ALWAYS, "[DIRECT]     acc_clear:   %7.1f us  (%4.1f%%)",
                 avg_clear, total > 0 ? 100.0 * avg_clear / total : 0);
            FARF(ALWAYS, "[DIRECT]     mm_f16:      %7.1f us  (%4.1f%%)  %u calls x %.3f us/call",
                 avg_mm, total > 0 ? 100.0 * avg_mm / total : 0,
                 n_output_tiles * K,
                 n_output_tiles * K > 0 ? avg_mm / (n_output_tiles * K) : 0.0);
            FARF(ALWAYS, "[DIRECT]     acc_rd+rm:   %7.1f us  (%4.1f%%)",
                 avg_rb, total > 0 ? 100.0 * avg_rb / total : 0);
            FARF(ALWAYS, "[DIRECT]     f16_to_f32:  %7.1f us  (%4.1f%%)  %u tiles x %.2f us/tile",
                 avg_wb, total > 0 ? 100.0 * avg_wb / total : 0,
                 n_output_tiles,
                 n_output_tiles > 0 ? avg_wb / n_output_tiles : 0.0);
        }

        /* ---- Method C: direct ASM compute + hexkl readback ---- */
        {
            compute_direct_asm(vtcm, C_f32, m, n, k,
                               act_base, wt_base, staging, readback, cfg_off);
            uint64_t t0 = HAP_perf_get_time_us();
            for (int iter = 0; iter < N_ITERS; iter++)
                compute_direct_asm(vtcm, C_f32, m, n, k,
                                   act_base, wt_base, staging, readback, cfg_off);
            uint64_t t1 = HAP_perf_get_time_us();
            float md = 0;
            int pass = verify_f32_vs_f16ref(C_f32, C_ref, m * n, &md);
            double avg = (double)(t1 - t0) / N_ITERS;
            double gflops = avg > 0 ? flops / (avg * 1e3) : 0;
            FARF(ALWAYS, "[DIRECT]   C ASM+hexkl: %7.1f us  %6.2f GFLOPS  %s  max_diff=%.3f",
                 avg, gflops, pass ? "PASS" : "FAIL", (double)md);
        }

        /* ---- Method D: direct ASM compute + HVX output ---- */
        {
            compute_direct_full(vtcm, C_f32, m, n, k,
                                act_base, wt_base, tile_base, scales_off);
            uint64_t t0 = HAP_perf_get_time_us();
            for (int iter = 0; iter < N_ITERS; iter++)
                compute_direct_full(vtcm, C_f32, m, n, k,
                                    act_base, wt_base, tile_base, scales_off);
            uint64_t t1 = HAP_perf_get_time_us();
            float md = 0;
            int pass = verify_f32_vs_f16ref(C_f32, C_ref, m * n, &md);
            double avg = (double)(t1 - t0) / N_ITERS;
            double gflops = avg > 0 ? flops / (avg * 1e3) : 0;
            FARF(ALWAYS, "[DIRECT]   D ASM+HVX:   %7.1f us  %6.2f GFLOPS  %s  max_diff=%.3f",
                 avg, gflops, pass ? "PASS" : "FAIL", (double)md);

            /* Per-phase timing for Method D (wrap all iters per phase) */
            uint64_t sum_compute = 0, sum_output = 0;
            for (int iter = 0; iter < N_ITERS; iter++) {
                uint64_t t_comp, t_out;
                compute_direct_full_timed(vtcm, C_f32, m, n, k,
                                          act_base, wt_base, tile_base, scales_off,
                                          &t_comp, &t_out);
                sum_compute += t_comp;
                sum_output  += t_out;
            }
            double avg_comp = (double)sum_compute / N_ITERS;
            double avg_out  = (double)sum_output / N_ITERS;
            double comp_gflops = avg_comp > 0 ? flops / (avg_comp * 1e3) : 0;
            FARF(ALWAYS, "[DIRECT]     compute:   %7.1f us  %6.2f GFLOPS (pure HMX)",
                 avg_comp, comp_gflops);
            FARF(ALWAYS, "[DIRECT]     hvx_out:   %7.1f us  (%u tiles x %.2f us/tile)",
                 avg_out, n_output_tiles,
                 n_output_tiles > 0 ? avg_out / n_output_tiles : 0.0);
        }

        /* ---- Speedup summary ---- */
        {
            uint64_t t0, t1;

            compute_hexkl(vtcm, C_f32, m, n, k,
                          act_base, wt_base, staging, readback, cfg_off);
            t0 = HAP_perf_get_time_us();
            for (int iter = 0; iter < N_ITERS; iter++)
                compute_hexkl(vtcm, C_f32, m, n, k,
                              act_base, wt_base, staging, readback, cfg_off);
            t1 = HAP_perf_get_time_us();
            double avg_a = (double)(t1 - t0) / N_ITERS;

            compute_direct_asm(vtcm, C_f32, m, n, k,
                               act_base, wt_base, staging, readback, cfg_off);
            t0 = HAP_perf_get_time_us();
            for (int iter = 0; iter < N_ITERS; iter++)
                compute_direct_asm(vtcm, C_f32, m, n, k,
                                   act_base, wt_base, staging, readback, cfg_off);
            t1 = HAP_perf_get_time_us();
            double avg_c = (double)(t1 - t0) / N_ITERS;

            compute_direct_full(vtcm, C_f32, m, n, k,
                                act_base, wt_base, tile_base, scales_off);
            t0 = HAP_perf_get_time_us();
            for (int iter = 0; iter < N_ITERS; iter++)
                compute_direct_full(vtcm, C_f32, m, n, k,
                                    act_base, wt_base, tile_base, scales_off);
            t1 = HAP_perf_get_time_us();
            double avg_d = (double)(t1 - t0) / N_ITERS;

            FARF(ALWAYS, "[DIRECT]   --- speedup vs hexkl (A) ---");
            FARF(ALWAYS, "[DIRECT]   C ASM+hexkl: %.2fx  (%.1f -> %.1f us)",
                 avg_c > 0 ? avg_a / avg_c : 0, avg_a, avg_c);
            FARF(ALWAYS, "[DIRECT]   D ASM+HVX:   %.2fx  (%.1f -> %.1f us)",
                 avg_d > 0 ? avg_a / avg_d : 0, avg_a, avg_d);
        }

next:
        free(A); free(B); free(C_ref); free(C_f32);
    }

    hexkl_micro_hmx_unlock();
    FARF(ALWAYS, "[DIRECT] === Done ===");
    return res;
}
