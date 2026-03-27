#ifndef HMX_MATMUL_F16_VTCM_H
#define HMX_MATMUL_F16_VTCM_H

/*
 * HMX matrix multiply for f16 data in VTCM.
 *
 * Hot path uses direct HMX ASM + HVX, minimal hexkl:
 *   AH (activation): HVX vshuff(-2) — 2-row interleave
 *   WH (weight):     hexkl rm_to_wh_f16 (DDR copy workaround)
 *   Compute:         ASM mxclracc.hf + mxmem:deep (batch tile loading)
 *   Readback:        ASM cvt.hf + mxmem store + HVX vdeal(-2)
 *
 * Key finding: hexkl_micro_hmx_mm_f16 must be called ONCE during setup
 * to configure hidden HMX registers (beyond what hmx_set_scales does).
 * After that, pure ASM compute works. Do NOT call hmx_set_scales() —
 * it clobbers the mm_f16 configuration and produces garbage.
 *
 * HMX tile = 32x32 f16, ALIGN = 2048 bytes.
 */

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>
#include <string.h>
#include "hexkl_micro.h"
#include "HAP_farf.h"

#define HMX_TILE         32
#define HMX_TILE_ELMS    (HMX_TILE * HMX_TILE)
#define HMX_TILE_BYTES   (HMX_TILE_ELMS * 2)
#define HMX_ALIGN        HEXKL_HMX_ACTIVATION_ALIGNMENT   /* 2048 */

/* DDR scratch buffer from hvx_matmul_f16_vtcm.h (for hexkl WH conversion) */
extern _Float16 g_tn_a_buf[];

/* ================================================================== */
/*  Direct HMX ASM wrappers                                             */
/* ================================================================== */

static inline __attribute__((always_inline)) void hmx_clear_acc(void) {
    asm volatile("mxclracc.hf" ::: "memory");
}

static inline __attribute__((always_inline)) void hmx_load_tiles(
    const void *act_tiles, const void *wt_tiles, uint32_t n_tiles)
{
    uint32_t limit = n_tiles * HMX_TILE_BYTES - 1;
    asm volatile(
        "{ activation.hf = mxmem(%0, %1):deep\n"
        "  weight.hf = mxmem(%2, %3) }\n"
        :: "r"(act_tiles), "r"(limit), "r"(wt_tiles), "r"(limit)
        : "memory");
}

static inline __attribute__((always_inline)) void hmx_store_acc_tile(_Float16 *out) {
    asm volatile(
        "cvt.hf = acc(%0)\n"
        "mxmem(%1, %2) = cvt\n"
        :: "r"(2), "r"(out), "r"(0)
        : "memory");
}

/* ================================================================== */
/*  HMX workspace                                                      */
/* ================================================================== */

struct hmx_workspace {
    uint8_t *vtcm_base;
    uint32_t vtcm_size;

    uint32_t ah_base;       /* AH activation tiles */
    uint32_t wh_base;       /* WH weight tiles */
    uint32_t staging_off;   /* staging tile (used for HMX init + readback) */
    uint32_t staging2_off;  /* second staging tile (readback output) */
    uint32_t cfg_off;       /* hexkl config at end of VTCM */

    uint32_t max_ah_tiles;
    uint32_t max_wh_tiles;
};

/*
 * Max tiles:
 * - AH: 208 (inp[256,832]: Mr=8, K=26 → 208 tiles)
 * - WH: 112 (fwd caches W1^T(104) + W2^T(8))
 */
#define HMX_MAX_AH_TILES  208
#define HMX_MAX_WH_TILES  112

static int setup_hmx_workspace(struct hmx_workspace *ws,
                                uint8_t *vtcm_base, uint32_t vtcm_size,
                                uint32_t data_end_offset)
{
    ws->vtcm_base = vtcm_base;
    ws->vtcm_size = vtcm_size;

    uint32_t off = (data_end_offset + HMX_ALIGN - 1) & ~(HMX_ALIGN - 1);

    ws->ah_base = off;
    ws->max_ah_tiles = HMX_MAX_AH_TILES;
    off += HMX_MAX_AH_TILES * HMX_ALIGN;

    ws->wh_base = off;
    ws->max_wh_tiles = HMX_MAX_WH_TILES;
    off += HMX_MAX_WH_TILES * HMX_ALIGN;

    ws->staging_off = (off + HMX_ALIGN - 1) & ~(HMX_ALIGN - 1);
    off = ws->staging_off + HMX_ALIGN;

    ws->staging2_off = (off + HMX_ALIGN - 1) & ~(HMX_ALIGN - 1);
    off = ws->staging2_off + HMX_ALIGN;

    uint32_t cfg_size = hexkl_micro_hmx_config_size();
    if (off + cfg_size > vtcm_size) {
        FARF(ERROR, "HMX workspace too large: need %u, have %u",
             off + cfg_size, vtcm_size);
        return AEE_ENOMEMORY;
    }

    ws->cfg_off = vtcm_size - cfg_size;
    hexkl_micro_hmx_setup_acc_read_f16(vtcm_base, ws->cfg_off);

    /*
     * One-time HMX initialization via dummy hexkl mm_f16.
     *
     * hexkl_micro_hmx_mm_f16 configures hidden HMX internal registers
     * (data type, accumulator format, scale/bias) that are NOT set by
     * the explicit ASM instructions alone (mxclracc.hf, mxmem).
     *
     * CRITICAL: Do NOT call hmx_set_scales() (bias=mxmem2) after this —
     * it overwrites the scale configuration with an incompatible format,
     * causing all subsequent matmuls to produce zeros.
     *
     * This state persists across hmx_clear_acc and hmx_store_acc_tile calls,
     * so a single initialization is sufficient for the session lifetime.
     */
    {
        HVX_Vector zero = Q6_V_vzero();
        HVX_Vector *sv;

        /* Prepare dummy AH tile */
        sv = (HVX_Vector *)(vtcm_base + ws->staging_off);
        for (uint32_t i = 0; i < HMX_TILE_BYTES / 128; i++)
            sv[i] = zero;
        hexkl_micro_hmx_rm_to_ah_f16(vtcm_base, ws->ah_base, ws->staging_off);

        /* Prepare dummy WH tile (needs DDR source for hexkl restrict) */
        memset(g_tn_a_buf, 0, HMX_TILE * HMX_TILE * sizeof(_Float16));
        hexkl_micro_hmx_rm_to_wh_f16(vtcm_base, ws->wh_base,
                                       g_tn_a_buf, 0, 0, HMX_TILE);

        /* Execute dummy matmul to configure HMX state */
        hexkl_micro_hmx_acc_clear_f16();
        hexkl_micro_hmx_mm_f16(vtcm_base, ws->ah_base, ws->wh_base);

        FARF(HIGH, "HMX state initialized via dummy mm_f16");
    }

    FARF(HIGH, "HMX workspace: %uKB (ah=%u wh=%u tiles) data_end=%uKB vtcm=%uKB",
         off / 1024, HMX_MAX_AH_TILES, HMX_MAX_WH_TILES,
         data_end_offset / 1024, vtcm_size / 1024);

    return AEE_SUCCESS;
}

/* ================================================================== */
/*  AH tile conversion — HVX vshuff (2-row interleave)                 */
/*                                                                     */
/*  AH format: vector[i] = {row[2i][0], row[2i+1][0], row[2i][1], ...}*/
/*  16 vectors per 32x32 tile.                                         */
/* ================================================================== */

static void convert_rm_to_ah(struct hmx_workspace *ws,
                              uint32_t ah_offset,
                              const _Float16 *src, uint32_t m, uint32_t k)
{
    uint32_t Mr = (m + HMX_TILE - 1) / HMX_TILE;
    uint32_t K  = (k + HMX_TILE - 1) / HMX_TILE;

    for (uint32_t rt = 0; rt < Mr; rt++) {
        for (uint32_t kt = 0; kt < K; kt++) {
            uint32_t tile_idx = rt * K + kt;
            HVX_Vector *out = (HVX_Vector *)(ws->vtcm_base + ah_offset
                                              + tile_idx * HMX_ALIGN);

            uint32_t row0 = rt * HMX_TILE;
            uint32_t col0 = kt * HMX_TILE;
            uint32_t cols = (col0 + HMX_TILE <= k) ? HMX_TILE : k - col0;

            for (uint32_t rr = 0; rr < HMX_TILE / 2; rr++) {
                uint32_t r_even = row0 + rr * 2;
                uint32_t r_odd  = r_even + 1;

                HVX_Vector v_r0 = Q6_V_vzero();
                HVX_Vector v_r1 = Q6_V_vzero();

                if (r_even < m)
                    memcpy(&v_r0, src + (size_t)r_even * k + col0,
                           cols * sizeof(_Float16));
                if (r_odd < m)
                    memcpy(&v_r1, src + (size_t)r_odd * k + col0,
                           cols * sizeof(_Float16));

                HVX_VectorPair vp = Q6_W_vshuff_VVR(v_r1, v_r0, -2);
                out[rr] = Q6_V_lo_W(vp);
            }
        }
    }
}

/* ================================================================== */
/*  WH tile conversion — hexkl (DDR copy workaround for restrict)      */
/* ================================================================== */

static void convert_rm_to_wh(struct hmx_workspace *ws,
                              uint32_t wh_offset,
                              const _Float16 *src, uint32_t k, uint32_t n)
{
    /* Copy from VTCM to DDR (hexkl has restrict aliasing on src) */
    uint32_t total_bytes = k * n * sizeof(_Float16);
    uint32_t total_vecs = (total_bytes + 127) / 128;
    for (uint32_t v = 0; v < total_vecs; v++)
        ((HVX_Vector *)g_tn_a_buf)[v] = ((const HVX_Vector *)src)[v];

    uint32_t Nc = (n + HMX_TILE - 1) / HMX_TILE;
    uint32_t K  = (k + HMX_TILE - 1) / HMX_TILE;

    for (uint32_t ct = 0; ct < Nc; ct++) {
        for (uint32_t kt = 0; kt < K; kt++) {
            uint32_t tile_idx = ct * K + kt;
            hexkl_micro_hmx_rm_to_wh_f16(
                ws->vtcm_base,
                wh_offset + tile_idx * HMX_ALIGN,
                g_tn_a_buf, kt, ct, n);
        }
    }
}

/* ================================================================== */
/*  Output readback — hmx_store_acc + HVX vdeal (AH → RM f16)         */
/* ================================================================== */

static void readback_acc_to_rm(struct hmx_workspace *ws,
                                _Float16 *dst, uint32_t m, uint32_t n,
                                uint32_t rt, uint32_t ct)
{
    _Float16 *tile = (_Float16 *)(ws->vtcm_base + ws->staging2_off);
    hmx_store_acc_tile(tile);

    uint32_t row0 = rt * HMX_TILE;
    uint32_t col0 = ct * HMX_TILE;
    uint32_t cols = (col0 + HMX_TILE <= n) ? HMX_TILE : n - col0;
    const HVX_Vector *tv = (const HVX_Vector *)tile;

    for (uint32_t rr = 0; rr < HMX_TILE / 2; rr++) {
        uint32_t r_even = row0 + rr * 2;
        uint32_t r_odd  = r_even + 1;

        HVX_VectorPair dealt = Q6_W_vdeal_VVR(Q6_V_vzero(), tv[rr], -2);
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
/*  HMX matmul: C[m,n] = A[m,k] @ B[k,n]                             */
/*                                                                     */
/*  Uses batch :deep tile loading (up to 32 tiles per mxmem).         */
/*  No hmx_set_scales — mm_f16 init in setup provides the config.     */
/* ================================================================== */

static void hmx_matmul_nn_f16(
    struct hmx_workspace *ws,
    _Float16 *C, const _Float16 *A, const _Float16 *B,
    uint32_t m, uint32_t n, uint32_t k)
{
    uint32_t Mr = (m + HMX_TILE - 1) / HMX_TILE;
    uint32_t Nc = (n + HMX_TILE - 1) / HMX_TILE;
    uint32_t K  = (k + HMX_TILE - 1) / HMX_TILE;

    convert_rm_to_ah(ws, ws->ah_base, A, m, k);
    convert_rm_to_wh(ws, ws->wh_base, B, k, n);

    for (uint32_t rt = 0; rt < Mr; rt++) {
        for (uint32_t ct = 0; ct < Nc; ct++) {
            hmx_clear_acc();

            const uint8_t *act_ptr = ws->vtcm_base + ws->ah_base
                                     + rt * K * HMX_ALIGN;
            const uint8_t *wt_ptr  = ws->vtcm_base + ws->wh_base
                                     + ct * K * HMX_ALIGN;

            for (uint32_t kt = 0; kt < K; kt += 32) {
                uint32_t batch = (K - kt < 32) ? K - kt : 32;
                hmx_load_tiles(act_ptr + kt * HMX_ALIGN,
                               wt_ptr  + kt * HMX_ALIGN,
                               batch);
            }

            readback_acc_to_rm(ws, C, m, n, rt, ct);
        }
    }
}

/*
 * Same as hmx_matmul_nn_f16 but WH tiles pre-converted at wh_offset.
 */
static void hmx_matmul_nn_f16_cached_wh(
    struct hmx_workspace *ws,
    _Float16 *C, const _Float16 *A,
    uint32_t wh_offset,
    uint32_t m, uint32_t n, uint32_t k)
{
    uint32_t Mr = (m + HMX_TILE - 1) / HMX_TILE;
    uint32_t Nc = (n + HMX_TILE - 1) / HMX_TILE;
    uint32_t K  = (k + HMX_TILE - 1) / HMX_TILE;

    convert_rm_to_ah(ws, ws->ah_base, A, m, k);

    for (uint32_t rt = 0; rt < Mr; rt++) {
        for (uint32_t ct = 0; ct < Nc; ct++) {
            hmx_clear_acc();

            const uint8_t *act_ptr = ws->vtcm_base + ws->ah_base
                                     + rt * K * HMX_ALIGN;
            const uint8_t *wt_ptr  = ws->vtcm_base + wh_offset
                                     + ct * K * HMX_ALIGN;

            for (uint32_t kt = 0; kt < K; kt += 32) {
                uint32_t batch = (K - kt < 32) ? K - kt : 32;
                hmx_load_tiles(act_ptr + kt * HMX_ALIGN,
                               wt_ptr  + kt * HMX_ALIGN,
                               batch);
            }

            readback_acc_to_rm(ws, C, m, n, rt, ct);
        }
    }
}

#endif /* HMX_MATMUL_F16_VTCM_H */
