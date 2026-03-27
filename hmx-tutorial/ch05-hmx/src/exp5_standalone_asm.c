/*
 * exp5_standalone_asm.c -- HMX matmul without hexkl in the runtime hot path
 *
 * htp-ops-lib (Qualcomm's production HMX library) does NOT use hexkl_micro
 * for compute or readback.  It uses:
 *   - HVX vshuff to prepare activation tiles (RM → AH interleaved format)
 *   - Direct ASM (mxmem) for compute
 *   - hmx_store_acc + HVX vdeal for readback (AH → RM deinterleave)
 *
 * hexkl is only used for weight tile prep (RM → WH), which is done offline
 * at graph-prepare time, not in the runtime hot path.
 *
 * This experiment proves the full pipeline works and benchmarks each approach:
 *
 *   V1: hexkl full       — hexkl compute + hexkl readback → f32 DDR (baseline)
 *   V2: ASM + hexkl rb   — ASM compute + hexkl readback → f32 DDR
 *   V3: ASM + HVX f16    — ASM compute + ASM store + HVX deinterleave → f16 VTCM
 *   V4: ASM + HVX f32    — same as V3 but with f16→f32 conversion → f32 DDR
 *
 * Key findings from exp5 v1:
 *   - hmx_set_scales() is REQUIRED before direct ASM compute
 *   - Without it, HMX produces garbage (scale/bias registers uninitialized)
 *   - hexkl_micro_hmx_mm_f16 sets scales internally as a side-effect
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
#define TILE_ELMS         (TILE * TILE)
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

static inline __attribute__((always_inline)) void hmx_store_acc(_Float16 *out) {
    asm volatile(
        "cvt.hf = acc(%0)\n"
        "mxmem(%1, %2) = cvt\n"
        :: "r"(2), "r"(out), "r"(0)
        : "memory");
}

/* ================================================================== */
/*  HVX AH tile format conversion (replaces hexkl_micro_hmx_rm_to_ah) */
/*                                                                     */
/*  AH (Activation HMX) format: interleaved row pairs.                 */
/*  Vector[i] = {row[2i][0], row[2i+1][0], row[2i][1], ...}          */
/*                                                                     */
/*  For a 32x32 tile: 16 vectors, each 128 bytes (64 f16 elements).   */
/*  Two consecutive rows (32 f16 each) are interleaved with vshuff.   */
/* ================================================================== */

/*
 * Convert one 32x32 submatrix from row-major f16 to AH tile format.
 * src points to the full matrix, (rt, kt) selects the 32x32 sub-tile.
 * ah_out must be 2048-byte aligned in VTCM.
 */
static void hvx_rm_to_ah_tile(
    _Float16 *ah_out,
    const _Float16 *src, uint32_t m, uint32_t k,
    uint32_t rt, uint32_t kt)
{
    uint32_t row0 = rt * TILE;
    uint32_t col0 = kt * TILE;
    uint32_t cols = (col0 + TILE <= k) ? TILE : k - col0;
    HVX_Vector *out = (HVX_Vector *)ah_out;

    for (uint32_t rr = 0; rr < TILE / 2; rr++) {
        uint32_t r_even = row0 + rr * 2;
        uint32_t r_odd  = r_even + 1;

        /* Load 32 f16 elements from each row into the lower half of a vector.
         * Upper half stays zero for partial tiles (last row/col block). */
        HVX_Vector v_r0 = Q6_V_vzero();
        HVX_Vector v_r1 = Q6_V_vzero();

        if (r_even < m)
            memcpy(&v_r0, src + (size_t)r_even * k + col0, cols * sizeof(_Float16));
        if (r_odd < m)
            memcpy(&v_r1, src + (size_t)r_odd * k + col0, cols * sizeof(_Float16));

        /* Interleave at halfword (2-byte) granularity:
         * vshuff(-2) with (v_r1, v_r0) produces:
         *   lo = {r0[0], r1[0], r0[1], r1[1], ..., r0[31], r1[31]} = AH vector */
        HVX_VectorPair vp = Q6_W_vshuff_VVR(v_r1, v_r0, -2);
        out[rr] = Q6_V_lo_W(vp);
    }
}

/* ================================================================== */
/*  HVX AH tile readback (replaces hexkl acc_read + ah_to_rm)         */
/*                                                                     */
/*  After hmx_store_acc, the output tile is in AH (interleaved) format */
/*  in VTCM.  Deinterleave with vdeal to get row-major f16.           */
/* ================================================================== */

/*
 * Convert one 32x32 AH tile (from hmx_store_acc) to row-major f16.
 * Writes directly to the destination submatrix region.
 */
static void hvx_ah_to_rm_f16(
    _Float16 *dst, uint32_t m, uint32_t n,
    uint32_t rt, uint32_t ct,
    const _Float16 *ah_tile)
{
    uint32_t row0 = rt * TILE;
    uint32_t col0 = ct * TILE;
    uint32_t cols = (col0 + TILE <= n) ? TILE : n - col0;
    const HVX_Vector *tile = (const HVX_Vector *)ah_tile;

    for (uint32_t rr = 0; rr < TILE / 2; rr++) {
        uint32_t r_even = row0 + rr * 2;
        uint32_t r_odd  = r_even + 1;

        /* Deinterleave at halfword granularity:
         * vdeal(-2) with (zero, v_tile) produces:
         *   lo = {r_even[0], r_even[1], ..., r_even[31], 0, ...}
         *   hi = {r_odd[0],  r_odd[1],  ..., r_odd[31],  0, ...} */
        HVX_VectorPair dealt = Q6_W_vdeal_VVR(Q6_V_vzero(), tile[rr], -2);
        HVX_Vector v_even = Q6_V_lo_W(dealt);
        HVX_Vector v_odd  = Q6_V_hi_W(dealt);

        if (r_even < m)
            memcpy(dst + (size_t)r_even * n + col0,
                   &v_even, cols * sizeof(_Float16));
        if (r_odd < m)
            memcpy(dst + (size_t)r_odd * n + col0,
                   &v_odd, cols * sizeof(_Float16));
    }
}

/*
 * Same as hvx_ah_to_rm_f16 but converts to f32 for verification.
 */
static inline __attribute__((always_inline))
HVX_VectorPair hvx_vhf_to_wsf(HVX_Vector vx) {
    HVX_Vector vqf16 = Q6_Vqf16_vadd_VhfVhf(vx, Q6_V_vzero());

    const HVX_Vector v_lo_mask = Q6_V_vsplat_R(0x0000ffff);
    const HVX_Vector v_hi_mask = Q6_V_vsplat_R(0xffff0000);
    const HVX_Vector v_shift16 = Q6_V_vsplat_R(16);

    HVX_Vector exp_comp = Q6_V_vand_VV(vqf16, Q6_Vh_vsplat_R(0x1f));
    HVX_Vector mantissa = Q6_V_vand_VV(vqf16, Q6_Vh_vsplat_R(0xffe0));
    exp_comp = Q6_Vh_vadd_VhVh(exp_comp, Q6_Vh_vsplat_R(112));

    HVX_Vector exp_comp0 = Q6_V_vand_VV(exp_comp, v_lo_mask);
    HVX_Vector exp_comp1 = Q6_Vw_vlsr_VwVw(exp_comp, v_shift16);
    HVX_Vector mantissa0 = Q6_Vw_vasl_VwVw(mantissa, v_shift16);
    HVX_Vector mantissa1 = Q6_V_vand_VV(mantissa, v_hi_mask);

    HVX_Vector v0_qf32 = Q6_Vw_vadd_VwVw(mantissa0, exp_comp0);
    HVX_Vector v1_qf32 = Q6_Vw_vadd_VwVw(mantissa1, exp_comp1);

    HVX_Vector v0_sf = Q6_Vsf_equals_Vqf32(v0_qf32);
    HVX_Vector v1_sf = Q6_Vsf_equals_Vqf32(v1_qf32);
    return Q6_W_vcombine_VV(v1_sf, v0_sf);
}

static void hvx_ah_to_rm_f32(
    float *dst, uint32_t m, uint32_t n,
    uint32_t rt, uint32_t ct,
    const _Float16 *ah_tile)
{
    uint32_t row0 = rt * TILE;
    uint32_t col0 = ct * TILE;
    const HVX_Vector *tile = (const HVX_Vector *)ah_tile;

    for (uint32_t rr = 0; rr < TILE / 2; rr++) {
        uint32_t r_even = row0 + rr * 2;
        uint32_t r_odd  = r_even + 1;

        /* hvx_vhf_to_wsf converts interleaved f16 → deinterleaved f32 pair.
         * lo = 32 f32 for r_even, hi = 32 f32 for r_odd */
        HVX_VectorPair vp = hvx_vhf_to_wsf(tile[rr]);

        if (col0 + TILE <= n) {
            if (r_even < m)
                *(HVX_Vector *)(dst + r_even * n + col0) = Q6_V_lo_W(vp);
            if (r_odd < m)
                *(HVX_Vector *)(dst + r_odd * n + col0) = Q6_V_hi_W(vp);
        } else {
            float tmp[32];
            if (r_even < m) {
                HVX_Vector v = Q6_V_lo_W(vp);
                memcpy(tmp, &v, 32 * sizeof(float));
                for (uint32_t j = 0; col0 + j < n; j++)
                    dst[r_even * n + col0 + j] = tmp[j];
            }
            if (r_odd < m) {
                HVX_Vector v = Q6_V_hi_W(vp);
                memcpy(tmp, &v, 32 * sizeof(float));
                for (uint32_t j = 0; col0 + j < n; j++)
                    dst[r_odd * n + col0 + j] = tmp[j];
            }
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

static int verify_f32(const float *got, const _Float16 *ref,
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

static int verify_f16(const _Float16 *got, const _Float16 *ref,
                      uint32_t count, float *max_diff)
{
    float md = 0.0f;
    int errors = 0;
    for (uint32_t i = 0; i < count; i++) {
        float r = (float)ref[i];
        float g = (float)got[i];
        float diff = fabsf(r - g);
        if (diff > md) md = diff;
        float denom = fabsf(r) > 1e-6f ? fabsf(r) : 1e-6f;
        if (diff > 0.5f && (diff / denom) > 0.05f) errors++;
    }
    *max_diff = md;
    return errors == 0;
}

/* ================================================================== */
/*  VTCM layout and tile setup                                         */
/*                                                                     */
/*  [AH tiles: Mr*K * ALIGN]                                          */
/*  [WH tiles: Nc*K * ALIGN]                                          */
/*  [output tiles: Mr*Nc * ALIGN]                                     */
/*  [f16 output buffer: m*n*2 bytes]  (for V3 f16 readback)           */
/*  [scales: 256 bytes]                                                */
/*  [staging: ALIGN]                                                   */
/*  [readback: ALIGN]                                                  */
/*  ...                                                                */
/*  [config: config_size at end of VTCM]                               */
/* ================================================================== */

struct vtcm_layout {
    uint32_t act_base;
    uint32_t wt_base;
    uint32_t tile_base;
    uint32_t f16_out;       /* f16 row-major output buffer in VTCM */
    uint32_t scales_off;
    uint32_t staging;
    uint32_t readback;
    uint32_t cfg;
};

static int setup_vtcm_layout(
    uint8_t *vtcm, uint32_t vtcm_size,
    uint32_t m, uint32_t n, uint32_t k,
    struct vtcm_layout *lay)
{
    const uint32_t K  = (k + TILE - 1) / TILE;
    const uint32_t Mr = (m + TILE - 1) / TILE;
    const uint32_t Nc = (n + TILE - 1) / TILE;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    uint32_t off = 0;
    lay->act_base = off;  off += Mr * K * ALIGN;
    lay->wt_base  = off;  off += Nc * K * ALIGN;
    lay->tile_base= off;  off += Mr * Nc * ALIGN;

    /* f16 output: aligned to 128 for HVX stores */
    off = (off + 127) & ~127u;
    lay->f16_out = off;    off += m * n * sizeof(_Float16);

    off = (off + 255) & ~255u;
    lay->scales_off = off; off += 256;

    off = (off + ALIGN - 1) & ~(ALIGN - 1);
    lay->staging = off;    off += ALIGN;

    off = (off + ALIGN - 1) & ~(ALIGN - 1);
    lay->readback = off;   off += ALIGN;

    uint32_t cfg_size = hexkl_micro_hmx_config_size();
    if (off + cfg_size > vtcm_size) return AEE_ENOMEMORY;

    lay->cfg = vtcm_size - cfg_size;
    hexkl_micro_hmx_setup_acc_read_f16(vtcm, lay->cfg);

    /* Scales: f16 1.0 + zero bias */
    HVX_Vector v_scale = Q6_Vh_vsplat_R(0x3C00);
    HVX_Vector *pv = (HVX_Vector *)(vtcm + lay->scales_off);
    pv[0] = v_scale;
    pv[1] = Q6_V_vzero();

    return AEE_SUCCESS;
}

/* Prepare AH tiles using hexkl (baseline) */
static void prep_ah_hexkl(
    uint8_t *vtcm, struct vtcm_layout *lay,
    const _Float16 *src, uint32_t m, uint32_t k)
{
    const uint32_t K  = (k + TILE - 1) / TILE;
    const uint32_t Mr = (m + TILE - 1) / TILE;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    for (uint32_t rt = 0; rt < Mr; rt++)
        for (uint32_t kt = 0; kt < K; kt++) {
            hexkl_micro_hmx_copy_submatrix_to_f16(
                vtcm, lay->staging, src, rt, kt, m, k);
            hexkl_micro_hmx_rm_to_ah_f16(
                vtcm, lay->act_base + (rt * K + kt) * ALIGN, lay->staging);
        }
}

/* Prepare AH tiles using HVX vshuff (no hexkl) */
static void prep_ah_hvx(
    uint8_t *vtcm, struct vtcm_layout *lay,
    const _Float16 *src, uint32_t m, uint32_t k)
{
    const uint32_t K  = (k + TILE - 1) / TILE;
    const uint32_t Mr = (m + TILE - 1) / TILE;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    for (uint32_t rt = 0; rt < Mr; rt++)
        for (uint32_t kt = 0; kt < K; kt++)
            hvx_rm_to_ah_tile(
                (_Float16 *)(vtcm + lay->act_base + (rt * K + kt) * ALIGN),
                src, m, k, rt, kt);
}

/* Prepare WH tiles (always hexkl — offline weight prep) */
static void prep_wh(
    uint8_t *vtcm, struct vtcm_layout *lay,
    const _Float16 *wt, uint32_t k, uint32_t n)
{
    const uint32_t K  = (k + TILE - 1) / TILE;
    const uint32_t Nc = (n + TILE - 1) / TILE;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    for (uint32_t ct = 0; ct < Nc; ct++)
        for (uint32_t kt = 0; kt < K; kt++)
            hexkl_micro_hmx_rm_to_wh_f16(
                vtcm, lay->wt_base + (ct * K + kt) * ALIGN,
                wt, kt, ct, n);
}

/* ================================================================== */
/*  Shared ASM compute kernel (used by V2, V3, V4)                     */
/* ================================================================== */

static inline void hmx_compute_all_tiles(
    uint8_t *vtcm, struct vtcm_layout *lay,
    uint32_t m, uint32_t n, uint32_t k)
{
    const uint32_t K     = (k + TILE - 1) / TILE;
    const uint32_t Mr    = (m + TILE - 1) / TILE;
    const uint32_t Nc    = (n + TILE - 1) / TILE;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    hmx_set_scales(vtcm + lay->scales_off);

    for (uint32_t rt = 0; rt < Mr; rt++) {
        for (uint32_t ct = 0; ct < Nc; ct++) {
            hmx_clear_acc();

            const void *act_ptr = (const void *)(vtcm + lay->act_base + rt * K * ALIGN);
            const void *wt_ptr  = (const void *)(vtcm + lay->wt_base  + ct * K * ALIGN);

            for (uint32_t kt = 0; kt < K; kt += 32) {
                uint32_t batch = (K - kt < 32) ? K - kt : 32;
                hmx_load_tiles(
                    (const void *)((uint8_t *)act_ptr + kt * ALIGN),
                    (const void *)((uint8_t *)wt_ptr  + kt * ALIGN),
                    batch);
            }

            _Float16 *out_tile = (_Float16 *)(vtcm + lay->tile_base + (rt * Nc + ct) * ALIGN);
            hmx_store_acc(out_tile);
        }
    }
}

/* ================================================================== */
/*  V1: hexkl full (baseline)                                          */
/* ================================================================== */

static void compute_v1_hexkl(
    uint8_t *vtcm, float *out_f32, struct vtcm_layout *lay,
    uint32_t m, uint32_t n, uint32_t k)
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
                    lay->act_base + (rt * K + kt) * ALIGN,
                    lay->wt_base  + (ct * K + kt) * ALIGN);

            hexkl_micro_hmx_acc_read_f16(vtcm, lay->cfg, lay->readback);
            hexkl_micro_hmx_ah_to_rm_f16(vtcm, lay->staging, lay->readback);
            hexkl_micro_hmx_copy_f16_to_f32_submatrix(
                vtcm, lay->staging, out_f32, rt, ct, m, n);
        }
    }
}

/* ================================================================== */
/*  V2: ASM compute + hexkl readback                                   */
/* ================================================================== */

static void compute_v2_asm_hexkl(
    uint8_t *vtcm, float *out_f32, struct vtcm_layout *lay,
    uint32_t m, uint32_t n, uint32_t k)
{
    const uint32_t K     = (k + TILE - 1) / TILE;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    hmx_set_scales(vtcm + lay->scales_off);

    for (uint32_t row = 0; row < m; row += TILE) {
        uint32_t rt = row / TILE;
        for (uint32_t col = 0; col < n; col += TILE) {
            uint32_t ct = col / TILE;

            hmx_clear_acc();

            const void *act_ptr = (const void *)(vtcm + lay->act_base + rt * K * ALIGN);
            const void *wt_ptr  = (const void *)(vtcm + lay->wt_base  + ct * K * ALIGN);

            for (uint32_t kt = 0; kt < K; kt += 32) {
                uint32_t batch = (K - kt < 32) ? K - kt : 32;
                hmx_load_tiles(
                    (const void *)((uint8_t *)act_ptr + kt * ALIGN),
                    (const void *)((uint8_t *)wt_ptr  + kt * ALIGN),
                    batch);
            }

            hexkl_micro_hmx_acc_read_f16(vtcm, lay->cfg, lay->readback);
            hexkl_micro_hmx_ah_to_rm_f16(vtcm, lay->staging, lay->readback);
            hexkl_micro_hmx_copy_f16_to_f32_submatrix(
                vtcm, lay->staging, out_f32, rt, ct, m, n);
        }
    }
}

/* ================================================================== */
/*  V3: ASM compute + HVX f16 readback (htp-ops-lib approach)          */
/*                                                                     */
/*  No hexkl in compute or readback.  Output is f16 in VTCM.          */
/* ================================================================== */

static void compute_v3_asm_hvx_f16(
    uint8_t *vtcm, struct vtcm_layout *lay,
    uint32_t m, uint32_t n, uint32_t k)
{
    const uint32_t Mr    = (m + TILE - 1) / TILE;
    const uint32_t Nc    = (n + TILE - 1) / TILE;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    /* Phase 1: HMX compute — all output tiles */
    hmx_compute_all_tiles(vtcm, lay, m, n, k);

    /* Phase 2: HVX deinterleave — tile AH → row-major f16 in VTCM */
    _Float16 *out_f16 = (_Float16 *)(vtcm + lay->f16_out);
    for (uint32_t rt = 0; rt < Mr; rt++)
        for (uint32_t ct = 0; ct < Nc; ct++) {
            const _Float16 *tile = (_Float16 *)(vtcm + lay->tile_base + (rt * Nc + ct) * ALIGN);
            hvx_ah_to_rm_f16(out_f16, m, n, rt, ct, tile);
        }
}

/* ================================================================== */
/*  V4: ASM compute + HVX f32 readback (for fair comparison)           */
/*                                                                     */
/*  Same as V3 but outputs f32 to DDR (like V1/V2) so timing is       */
/*  comparable.  Uses hvx_vhf_to_wsf on interleaved tiles directly.    */
/* ================================================================== */

static void compute_v4_asm_hvx_f32(
    uint8_t *vtcm, float *out_f32, struct vtcm_layout *lay,
    uint32_t m, uint32_t n, uint32_t k)
{
    const uint32_t Mr    = (m + TILE - 1) / TILE;
    const uint32_t Nc    = (n + TILE - 1) / TILE;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    hmx_compute_all_tiles(vtcm, lay, m, n, k);

    for (uint32_t rt = 0; rt < Mr; rt++)
        for (uint32_t ct = 0; ct < Nc; ct++) {
            const _Float16 *tile = (_Float16 *)(vtcm + lay->tile_base + (rt * Nc + ct) * ALIGN);
            hvx_ah_to_rm_f32(out_f32, m, n, rt, ct, tile);
        }
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
};
#define N_CONFIGS (sizeof(configs) / sizeof(configs[0]))

/* ================================================================== */
/*  Main benchmark                                                     */
/* ================================================================== */

int main(void) {
    int res = AEE_SUCCESS;
    uint8_t *vtcm = NULL;
    uint32_t vtcm_size = 0;

    FARF(ALWAYS, "[EXP5] === HMX Matmul: hexkl vs hexkl-free runtime ===");
    FARF(ALWAYS, "[EXP5]");
    FARF(ALWAYS, "[EXP5] V1: hexkl full (compute + readback)");
    FARF(ALWAYS, "[EXP5] V2: ASM compute + hexkl readback -> f32 DDR");
    FARF(ALWAYS, "[EXP5] V3: ASM compute + HVX readback -> f16 VTCM (htp-ops-lib)");
    FARF(ALWAYS, "[EXP5] V4: ASM compute + HVX readback -> f32 DDR");
    FARF(ALWAYS, "[EXP5]");
    FARF(ALWAYS, "[EXP5] Tile prep: hexkl (AH+WH) for V1/V2, HVX AH + hexkl WH for V3/V4");

    res = hexkl_micro_hw_init(&vtcm, &vtcm_size);
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[EXP5] hw_init failed: %d", res);
        return res;
    }
    FARF(ALWAYS, "[EXP5] VTCM: %u KB at %p", vtcm_size / 1024, (void *)vtcm);

    res = hexkl_micro_hmx_lock();
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[EXP5] hmx_lock failed: %d", res);
        return res;
    }

    for (uint32_t ci = 0; ci < N_CONFIGS; ci++) {
        struct bench_config *bcfg = &configs[ci];
        uint32_t m = bcfg->m, n = bcfg->n, k = bcfg->k;
        double flops = 2.0 * m * n * k;

        FARF(ALWAYS, "[EXP5] ------------------------------------------------");
        FARF(ALWAYS, "[EXP5] %s  M=%u N=%u K=%u", bcfg->name, m, n, k);

        _Float16 *A     = (_Float16 *)memalign(128, m * k * sizeof(_Float16));
        _Float16 *B     = (_Float16 *)memalign(128, k * n * sizeof(_Float16));
        _Float16 *C_ref = (_Float16 *)malloc(m * n * sizeof(_Float16));
        float    *C_f32 = (float *)memalign(128, m * n * sizeof(float));

        if (!A || !B || !C_ref || !C_f32) {
            FARF(ALWAYS, "[EXP5] malloc failed");
            goto next;
        }

        for (uint32_t i = 0; i < m * k; i++)
            A[i] = (_Float16)(0.01f * (float)((i % 97) - 48));
        for (uint32_t i = 0; i < k * n; i++)
            B[i] = (_Float16)(0.01f * (float)((i % 83) - 41));

        ref_matmul(C_ref, A, B, m, n, k);

        struct vtcm_layout lay;
        if (setup_vtcm_layout(vtcm, vtcm_size, m, n, k, &lay) != AEE_SUCCESS) {
            FARF(ALWAYS, "[EXP5]   SKIP (VTCM too small)");
            goto next;
        }

        /* ---- Tile prep comparison ---- */
        {
            uint64_t t0, t1;

            t0 = HAP_perf_get_time_us();
            for (int i = 0; i < 100; i++)
                prep_ah_hexkl(vtcm, &lay, A, m, k);
            t1 = HAP_perf_get_time_us();
            double avg_hexkl = (double)(t1 - t0) / 100.0;

            t0 = HAP_perf_get_time_us();
            for (int i = 0; i < 100; i++)
                prep_ah_hvx(vtcm, &lay, A, m, k);
            t1 = HAP_perf_get_time_us();
            double avg_hvx = (double)(t1 - t0) / 100.0;

            FARF(ALWAYS, "[EXP5]   AH prep: hexkl %.1f us  HVX %.1f us  (%.1fx)",
                 avg_hexkl, avg_hvx,
                 avg_hvx > 0 ? avg_hexkl / avg_hvx : 0);
        }

        /* Prepare tiles for V1/V2 (hexkl AH) */
        prep_ah_hexkl(vtcm, &lay, A, m, k);
        prep_wh(vtcm, &lay, B, k, n);

        /* ---- V1: hexkl full ---- */
        {
            compute_v1_hexkl(vtcm, C_f32, &lay, m, n, k);
            uint64_t t0 = HAP_perf_get_time_us();
            for (int i = 0; i < N_ITERS; i++)
                compute_v1_hexkl(vtcm, C_f32, &lay, m, n, k);
            uint64_t t1 = HAP_perf_get_time_us();
            float md;
            int pass = verify_f32(C_f32, C_ref, m * n, &md);
            double avg = (double)(t1 - t0) / N_ITERS;
            FARF(ALWAYS, "[EXP5]   V1 hexkl:       %7.1f us  %6.2f GFLOPS  %s  diff=%.3f",
                 avg, avg > 0 ? flops / (avg * 1e3) : 0,
                 pass ? "PASS" : "FAIL", (double)md);
        }

        /* ---- V2: ASM compute + hexkl readback ---- */
        {
            compute_v2_asm_hexkl(vtcm, C_f32, &lay, m, n, k);
            uint64_t t0 = HAP_perf_get_time_us();
            for (int i = 0; i < N_ITERS; i++)
                compute_v2_asm_hexkl(vtcm, C_f32, &lay, m, n, k);
            uint64_t t1 = HAP_perf_get_time_us();
            float md;
            int pass = verify_f32(C_f32, C_ref, m * n, &md);
            double avg = (double)(t1 - t0) / N_ITERS;
            FARF(ALWAYS, "[EXP5]   V2 ASM+hexkl:   %7.1f us  %6.2f GFLOPS  %s  diff=%.3f",
                 avg, avg > 0 ? flops / (avg * 1e3) : 0,
                 pass ? "PASS" : "FAIL", (double)md);
        }

        /* Re-prepare AH tiles with HVX for V3/V4 */
        prep_ah_hvx(vtcm, &lay, A, m, k);
        /* WH tiles unchanged (hexkl, offline) */

        /* ---- V3: ASM compute + HVX f16 readback (htp-ops-lib path) ---- */
        {
            compute_v3_asm_hvx_f16(vtcm, &lay, m, n, k);
            uint64_t t0 = HAP_perf_get_time_us();
            for (int i = 0; i < N_ITERS; i++)
                compute_v3_asm_hvx_f16(vtcm, &lay, m, n, k);
            uint64_t t1 = HAP_perf_get_time_us();
            _Float16 *out_f16 = (_Float16 *)(vtcm + lay.f16_out);
            float md;
            int pass = verify_f16(out_f16, C_ref, m * n, &md);
            double avg = (double)(t1 - t0) / N_ITERS;
            FARF(ALWAYS, "[EXP5]   V3 ASM+HVX f16: %7.1f us  %6.2f GFLOPS  %s  diff=%.3f",
                 avg, avg > 0 ? flops / (avg * 1e3) : 0,
                 pass ? "PASS" : "FAIL", (double)md);
        }

        /* ---- V4: ASM compute + HVX f32 readback ---- */
        {
            compute_v4_asm_hvx_f32(vtcm, C_f32, &lay, m, n, k);
            uint64_t t0 = HAP_perf_get_time_us();
            for (int i = 0; i < N_ITERS; i++)
                compute_v4_asm_hvx_f32(vtcm, C_f32, &lay, m, n, k);
            uint64_t t1 = HAP_perf_get_time_us();
            float md;
            int pass = verify_f32(C_f32, C_ref, m * n, &md);
            double avg = (double)(t1 - t0) / N_ITERS;
            FARF(ALWAYS, "[EXP5]   V4 ASM+HVX f32: %7.1f us  %6.2f GFLOPS  %s  diff=%.3f",
                 avg, avg > 0 ? flops / (avg * 1e3) : 0,
                 pass ? "PASS" : "FAIL", (double)md);
        }

next:
        free(A); free(B); free(C_ref); free(C_f32);
    }

    hexkl_micro_hmx_unlock();
    FARF(ALWAYS, "[EXP5] === Done ===");
    return res;
}
