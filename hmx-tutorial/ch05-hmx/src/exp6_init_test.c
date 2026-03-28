/*
 * exp6_init_test.c -- What HMX initialization is ACTUALLY needed?
 *
 * LLVM IR disassembly of hexkl_micro.a revealed:
 *   - hexkl_micro_hmx_mm_f16: just 2 instructions (mxmem.blk.sm.act.hf + mxmem.wei.hf)
 *     NO state configuration at all.
 *   - hexkl_micro_hmx_setup_acc_read_f16: writes [scale=1.0, bias=0.0] x 32
 *     then calls M8.mxmem2.bias -- THIS configures HMX scale/bias registers.
 *   - hmx_set_scales (exp5 inline ASM): also calls mxmem2.bias -- same instruction.
 *
 * Hypothesis: The dummy hexkl_micro_hmx_mm_f16() call in ch09 is unnecessary.
 * The HMX configuration comes entirely from setup_acc_read_f16 (mxmem2.bias).
 *
 * This experiment tests 4 initialization variants with the SAME compute+readback:
 *   V1: setup_acc_read + dummy mm_f16 (current ch09 approach)
 *   V2: setup_acc_read ONLY (no dummy mm_f16)
 *   V3: hmx_set_scales ONLY (exp5 approach, no setup_acc_read)
 *   V4: No initialization at all (just hmx_lock)
 *
 * Between each variant, HMX state is fully reset via unlock + lock.
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

/* Fixed test size: 128x128x128 = 4x4x4 tiles */
#define M  128
#define N  128
#define K  128

/* ================================================================== */
/*  HMX inline assembly wrappers                                       */
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
/*  HVX AH tile format conversion (from exp5)                          */
/* ================================================================== */

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

        HVX_Vector v_r0 = Q6_V_vzero();
        HVX_Vector v_r1 = Q6_V_vzero();

        if (r_even < m)
            memcpy(&v_r0, src + (size_t)r_even * k + col0, cols * sizeof(_Float16));
        if (r_odd < m)
            memcpy(&v_r1, src + (size_t)r_odd * k + col0, cols * sizeof(_Float16));

        HVX_VectorPair vp = Q6_W_vshuff_VVR(v_r1, v_r0, -2);
        out[rr] = Q6_V_lo_W(vp);
    }
}

/* ================================================================== */
/*  HVX AH tile readback to f16 (from exp5)                            */
/* ================================================================== */

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
/*  VTCM layout                                                        */
/*                                                                     */
/*  [AH tiles: Mr*Kt * ALIGN]                                         */
/*  [WH tiles: Nc*Kt * ALIGN]                                         */
/*  [output tiles: Mr*Nc * ALIGN]                                      */
/*  [f16 output: m*n*2]                                                */
/*  [scales: 256]                                                      */
/*  [staging: ALIGN]  (for dummy mm_f16 in V1)                         */
/*  ...                                                                */
/*  [config: config_size at end]                                       */
/* ================================================================== */

struct vtcm_layout {
    uint32_t act_base;
    uint32_t wt_base;
    uint32_t tile_base;
    uint32_t f16_out;
    uint32_t scales_off;
    uint32_t staging;
    uint32_t cfg;
};

/* ================================================================== */
/*  Common compute + readback (same for all variants)                  */
/* ================================================================== */

static void compute_and_readback(
    uint8_t *vtcm, struct vtcm_layout *lay,
    _Float16 *out_f16)
{
    const uint32_t Kt    = K / TILE;
    const uint32_t Mr    = M / TILE;
    const uint32_t Nc    = N / TILE;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    /* HMX compute: all output tiles */
    for (uint32_t rt = 0; rt < Mr; rt++) {
        for (uint32_t ct = 0; ct < Nc; ct++) {
            hmx_clear_acc();

            const void *act_ptr = (const void *)(vtcm + lay->act_base + rt * Kt * ALIGN);
            const void *wt_ptr  = (const void *)(vtcm + lay->wt_base  + ct * Kt * ALIGN);

            hmx_load_tiles(act_ptr, wt_ptr, Kt);

            _Float16 *out_tile = (_Float16 *)(vtcm + lay->tile_base + (rt * Nc + ct) * ALIGN);
            hmx_store_acc(out_tile);
        }
    }

    /* HVX deinterleave: AH tile -> row-major f16 */
    for (uint32_t rt = 0; rt < Mr; rt++)
        for (uint32_t ct = 0; ct < Nc; ct++) {
            const _Float16 *tile = (_Float16 *)(vtcm + lay->tile_base + (rt * Nc + ct) * ALIGN);
            hvx_ah_to_rm_f16(out_f16, M, N, rt, ct, tile);
        }
}

/* ================================================================== */
/*  Main: test 4 initialization variants                               */
/* ================================================================== */

int main(void) {
    int res = AEE_SUCCESS;
    uint8_t *vtcm = NULL;
    uint32_t vtcm_size = 0;

    FARF(ALWAYS, "[EXP6] === HMX Init Test: What initialization is ACTUALLY needed? ===");
    FARF(ALWAYS, "[EXP6]");
    FARF(ALWAYS, "[EXP6] V1: setup_acc_read + dummy mm_f16 (current ch09 approach)");
    FARF(ALWAYS, "[EXP6] V2: setup_acc_read ONLY (no dummy mm_f16)");
    FARF(ALWAYS, "[EXP6] V3: hmx_set_scales ONLY (exp5 inline ASM)");
    FARF(ALWAYS, "[EXP6] V4: No initialization at all (just hmx_lock)");
    FARF(ALWAYS, "[EXP6]");
    FARF(ALWAYS, "[EXP6] All variants use SAME compute: direct ASM + store_acc + vdeal");
    FARF(ALWAYS, "[EXP6] Test: M=%u N=%u K=%u", M, N, K);

    /* ---- hw_init ---- */
    res = hexkl_micro_hw_init(&vtcm, &vtcm_size);
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[EXP6] hw_init failed: %d", res);
        return res;
    }
    FARF(ALWAYS, "[EXP6] VTCM: %u KB at %p", vtcm_size / 1024, (void *)vtcm);

    /* ---- Allocate test matrices ---- */
    _Float16 *A     = (_Float16 *)memalign(128, M * K * sizeof(_Float16));
    _Float16 *B     = (_Float16 *)memalign(128, K * N * sizeof(_Float16));
    _Float16 *C_ref = (_Float16 *)malloc(M * N * sizeof(_Float16));
    _Float16 *C_got = (_Float16 *)memalign(128, M * N * sizeof(_Float16));

    if (!A || !B || !C_ref || !C_got) {
        FARF(ALWAYS, "[EXP6] malloc failed");
        return AEE_ENOMEMORY;
    }

    /* Fill with deterministic values */
    for (uint32_t i = 0; i < M * K; i++)
        A[i] = (_Float16)(0.01f * (float)((i % 97) - 48));
    for (uint32_t i = 0; i < K * N; i++)
        B[i] = (_Float16)(0.01f * (float)((i % 83) - 41));

    FARF(ALWAYS, "[EXP6] Computing reference...");
    ref_matmul(C_ref, A, B, M, N, K);

    /* ---- Compute VTCM layout ---- */
    const uint32_t Kt = K / TILE;
    const uint32_t Mr = M / TILE;
    const uint32_t Nc = N / TILE;
    const uint32_t ALIGN = HEXKL_HMX_ACTIVATION_ALIGNMENT;

    struct vtcm_layout lay;
    uint32_t off = 0;
    lay.act_base = off;   off += Mr * Kt * ALIGN;
    lay.wt_base  = off;   off += Nc * Kt * ALIGN;
    lay.tile_base = off;  off += Mr * Nc * ALIGN;

    off = (off + 127) & ~127u;
    lay.f16_out = off;    off += M * N * sizeof(_Float16);

    off = (off + 255) & ~255u;
    lay.scales_off = off; off += 256;

    off = (off + ALIGN - 1) & ~(ALIGN - 1);
    lay.staging = off;    off += ALIGN;

    uint32_t cfg_size = hexkl_micro_hmx_config_size();
    if (off + cfg_size > vtcm_size) {
        FARF(ALWAYS, "[EXP6] VTCM too small: need %u, have %u", off + cfg_size, vtcm_size);
        return AEE_ENOMEMORY;
    }
    lay.cfg = vtcm_size - cfg_size;

    FARF(ALWAYS, "[EXP6] Layout: AH@%u WH@%u tile@%u f16@%u scales@%u staging@%u cfg@%u",
         lay.act_base, lay.wt_base, lay.tile_base, lay.f16_out,
         lay.scales_off, lay.staging, lay.cfg);

    /* ---- Prepare scales buffer (used by V3) ---- */
    HVX_Vector v_scale = Q6_Vh_vsplat_R(0x3C00);  /* f16 1.0 */
    HVX_Vector *pv = (HVX_Vector *)(vtcm + lay.scales_off);
    pv[0] = v_scale;
    pv[1] = Q6_V_vzero();

    /* ================================================================ */
    /*  V1: setup_acc_read + dummy mm_f16 (current ch09 approach)       */
    /* ================================================================ */
    FARF(ALWAYS, "[EXP6]");
    FARF(ALWAYS, "[EXP6] --- V1: setup_acc_read + dummy mm_f16 ---");

    res = hexkl_micro_hmx_lock();
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[EXP6] V1: hmx_lock failed: %d", res);
        return res;
    }

    /* setup_acc_read configures HMX via mxmem2.bias */
    hexkl_micro_hmx_setup_acc_read_f16(vtcm, lay.cfg);

    /* Prepare tiles */
    for (uint32_t rt = 0; rt < Mr; rt++)
        for (uint32_t kt = 0; kt < Kt; kt++)
            hvx_rm_to_ah_tile(
                (_Float16 *)(vtcm + lay.act_base + (rt * Kt + kt) * ALIGN),
                A, M, K, rt, kt);
    for (uint32_t ct = 0; ct < Nc; ct++)
        for (uint32_t kt = 0; kt < Kt; kt++)
            hexkl_micro_hmx_rm_to_wh_f16(
                vtcm, lay.wt_base + (ct * Kt + kt) * ALIGN,
                B, kt, ct, N);

    /* Dummy mm_f16 call (the thing we are testing whether it's needed) */
    hexkl_micro_hmx_acc_clear_f16();
    hexkl_micro_hmx_mm_f16(vtcm, lay.act_base, lay.wt_base);

    /* Now do actual compute + readback */
    memset(C_got, 0, M * N * sizeof(_Float16));
    compute_and_readback(vtcm, &lay, C_got);

    {
        float md;
        int pass = verify_f16(C_got, C_ref, M * N, &md);
        FARF(ALWAYS, "[EXP6] V1: %s  max_diff=%.4f", pass ? "PASS" : "FAIL", (double)md);
    }

    /* Reset HMX state */
    hexkl_micro_hmx_unlock();

    /* ================================================================ */
    /*  V2: setup_acc_read ONLY (no dummy mm_f16)                       */
    /* ================================================================ */
    FARF(ALWAYS, "[EXP6]");
    FARF(ALWAYS, "[EXP6] --- V2: setup_acc_read ONLY ---");

    res = hexkl_micro_hmx_lock();
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[EXP6] V2: hmx_lock failed: %d", res);
        return res;
    }

    /* setup_acc_read only -- NO dummy mm_f16 */
    hexkl_micro_hmx_setup_acc_read_f16(vtcm, lay.cfg);

    /* Tiles already prepared from V1, re-prepare to be safe */
    for (uint32_t rt = 0; rt < Mr; rt++)
        for (uint32_t kt = 0; kt < Kt; kt++)
            hvx_rm_to_ah_tile(
                (_Float16 *)(vtcm + lay.act_base + (rt * Kt + kt) * ALIGN),
                A, M, K, rt, kt);
    for (uint32_t ct = 0; ct < Nc; ct++)
        for (uint32_t kt = 0; kt < Kt; kt++)
            hexkl_micro_hmx_rm_to_wh_f16(
                vtcm, lay.wt_base + (ct * Kt + kt) * ALIGN,
                B, kt, ct, N);

    /* Compute + readback (no dummy mm_f16 before this) */
    memset(C_got, 0, M * N * sizeof(_Float16));
    compute_and_readback(vtcm, &lay, C_got);

    {
        float md;
        int pass = verify_f16(C_got, C_ref, M * N, &md);
        FARF(ALWAYS, "[EXP6] V2: %s  max_diff=%.4f", pass ? "PASS" : "FAIL", (double)md);
    }

    hexkl_micro_hmx_unlock();

    /* ================================================================ */
    /*  V3: hmx_set_scales ONLY (exp5 inline ASM approach)              */
    /* ================================================================ */
    FARF(ALWAYS, "[EXP6]");
    FARF(ALWAYS, "[EXP6] --- V3: hmx_set_scales ONLY ---");

    res = hexkl_micro_hmx_lock();
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[EXP6] V3: hmx_lock failed: %d", res);
        return res;
    }

    /* NO setup_acc_read -- just hmx_set_scales (inline ASM mxmem2.bias) */
    /* Re-write scales in case unlock cleared VTCM */
    pv = (HVX_Vector *)(vtcm + lay.scales_off);
    pv[0] = v_scale;
    pv[1] = Q6_V_vzero();
    hmx_set_scales(vtcm + lay.scales_off);

    /* Re-prepare tiles */
    for (uint32_t rt = 0; rt < Mr; rt++)
        for (uint32_t kt = 0; kt < Kt; kt++)
            hvx_rm_to_ah_tile(
                (_Float16 *)(vtcm + lay.act_base + (rt * Kt + kt) * ALIGN),
                A, M, K, rt, kt);
    for (uint32_t ct = 0; ct < Nc; ct++)
        for (uint32_t kt = 0; kt < Kt; kt++)
            hexkl_micro_hmx_rm_to_wh_f16(
                vtcm, lay.wt_base + (ct * Kt + kt) * ALIGN,
                B, kt, ct, N);

    /* Compute + readback */
    memset(C_got, 0, M * N * sizeof(_Float16));
    compute_and_readback(vtcm, &lay, C_got);

    {
        float md;
        int pass = verify_f16(C_got, C_ref, M * N, &md);
        FARF(ALWAYS, "[EXP6] V3: %s  max_diff=%.4f", pass ? "PASS" : "FAIL", (double)md);
    }

    hexkl_micro_hmx_unlock();

    /* ================================================================ */
    /*  V4: No initialization at all (just hmx_lock)                    */
    /* ================================================================ */
    FARF(ALWAYS, "[EXP6]");
    FARF(ALWAYS, "[EXP6] --- V4: No initialization at all ---");

    res = hexkl_micro_hmx_lock();
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[EXP6] V4: hmx_lock failed: %d", res);
        return res;
    }

    /* NO setup_acc_read, NO hmx_set_scales, NO dummy mm_f16 */

    /* Re-prepare tiles */
    for (uint32_t rt = 0; rt < Mr; rt++)
        for (uint32_t kt = 0; kt < Kt; kt++)
            hvx_rm_to_ah_tile(
                (_Float16 *)(vtcm + lay.act_base + (rt * Kt + kt) * ALIGN),
                A, M, K, rt, kt);
    for (uint32_t ct = 0; ct < Nc; ct++)
        for (uint32_t kt = 0; kt < Kt; kt++)
            hexkl_micro_hmx_rm_to_wh_f16(
                vtcm, lay.wt_base + (ct * Kt + kt) * ALIGN,
                B, kt, ct, N);

    /* Compute + readback */
    memset(C_got, 0, M * N * sizeof(_Float16));
    compute_and_readback(vtcm, &lay, C_got);

    {
        float md;
        int pass = verify_f16(C_got, C_ref, M * N, &md);
        FARF(ALWAYS, "[EXP6] V4: %s  max_diff=%.4f", pass ? "PASS" : "FAIL", (double)md);
    }

    hexkl_micro_hmx_unlock();

    /* ---- Summary ---- */
    FARF(ALWAYS, "[EXP6]");
    FARF(ALWAYS, "[EXP6] === Done ===");

    free(A); free(B); free(C_ref); free(C_got);
    return res;
}
