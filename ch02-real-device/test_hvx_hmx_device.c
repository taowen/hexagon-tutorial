/*
 * test_hvx_hmx_device.c — 第二章: 真机上跑 HVX + HMX
 *
 * 编译为 .so, 用 run_main_on_hexagon 加载到 CDSP 执行.
 * 和第一章的模拟器版本对比:
 *   - 模拟器用 H2 Hypervisor 管理 VTCM/HVX/HMX
 *   - 真机用 HAP API (HAP_compute_res / HAP_power_set)
 *
 * 参考 llama.cpp 的做法:
 *   - 用 HAP_power_set 显式上电 HVX/HMX
 *   - 用 HAP_compute_res 分配 VTCM + 锁定 HMX
 *   - 用 inline asm 驱动 HMX (不用 Q6_ intrinsics)
 *   - Rt = 2047 (精确一个 tile), 不用大值如 32767
 */

#include <stdio.h>
#include <string.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include "HAP_farf.h"
#include "HAP_compute_res.h"
#include "HAP_power.h"

/* ============================================================
 * 常量
 * ============================================================ */
#define TILE_DIM    32
#define TILE_ELEMS  (TILE_DIM * TILE_DIM)   /* 1024 */
#define TILE_BYTES  (TILE_ELEMS * 2)        /* 2048 */
#define HVX_VLEN    128
#define F16_PER_VEC (HVX_VLEN / 2)         /* 64 */

#define F16_ONE     0x3C00
#define F16_TWO     0x4000
#define F16_NEG_ONE 0xBC00

/* ============================================================
 * HMX inline asm wrappers (同第一章, llama.cpp 风格)
 * ============================================================ */
static inline __attribute__((always_inline))
void hmx_set_scales(const void *scales)
{
    asm volatile("bias = mxmem2(%0)" :: "r"(scales) : "memory");
}

static inline __attribute__((always_inline))
void hmx_clear_acc(void)
{
    asm volatile("mxclracc.hf" ::: "memory");
}

static inline __attribute__((always_inline))
void hmx_load_tile_pair(const void *act, const void *wt)
{
    asm volatile(
        "{ activation.hf = mxmem(%0, %1)\n"
        "  weight.hf = mxmem(%2, %3) }\n"
        :: "r"(act), "r"(2047),
           "r"(wt),  "r"(2047)
        : "memory");
}

static inline __attribute__((always_inline))
void hmx_store_acc(void *out)
{
    asm volatile(
        "mxmem(%0, %1):after.hf = acc\n"
        :: "r"(out), "r"(0)
        : "memory");
}

/* ============================================================
 * HVX 工具函数 (同第一章)
 * ============================================================ */
static void hvx_fill_f16(unsigned short *buf, unsigned short val, int count)
{
    int splat_word = (val << 16) | val;
    HVX_Vector v_val = Q6_V_vsplat_R(splat_word);
    HVX_Vector *vp = (HVX_Vector *)buf;
    int i;
    for (i = 0; i < count / F16_PER_VEC; i++)
        vp[i] = v_val;
}

static void hvx_fill_scales(unsigned char *scale_buf, unsigned short val)
{
    int splat_word = (val << 16) | val;
    ((HVX_Vector *)scale_buf)[0] = Q6_V_vsplat_R(splat_word);
    ((HVX_Vector *)scale_buf)[1] = Q6_V_vzero();
}

static void hvx_relu_f16(unsigned short *buf, int count)
{
    HVX_Vector v_zero = Q6_V_vzero();
    HVX_Vector *vp = (HVX_Vector *)buf;
    int i;
    for (i = 0; i < count / F16_PER_VEC; i++)
        vp[i] = Q6_Vh_vmax_VhVh(vp[i], v_zero);
}

/* ============================================================
 * F16 辅助
 * ============================================================ */
static float f16_to_f32(unsigned short h)
{
    int sign = (h >> 15) & 1;
    int exp  = (h >> 10) & 0x1F;
    int mant = h & 0x3FF;
    float val;
    if (exp == 0)       val = (float)mant / 1024.0f * (1.0f / 16384.0f);
    else if (exp == 31) val = (mant == 0) ? 1e30f : -1.0f;
    else                val = (1.0f + (float)mant / 1024.0f) * (float)(1 << (exp - 15));
    if (sign) val = -val;
    return val;
}

/* ============================================================
 * HAP 初始化: 上电 + 分配资源
 *
 * 真机和模拟器的关键区别:
 *   模拟器 (H2):  h2_info() + h2_vecaccess_acquire() + h2_mxaccess_acquire()
 *   真机 (HAP):   HAP_power_set() + HAP_compute_res_acquire() + hmx_lock()
 * ============================================================ */
static int power_ctx;  /* HAP_power_set 需要非 NULL 上下文 */

static int power_on_hvx_hmx(void)
{
    HAP_power_request_t req;

    /* 设置 client class (llama.cpp 的做法) */
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_apptype;
    req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    if (HAP_power_set((void *)&power_ctx, &req) != 0) return -1;

    /* DCVS 性能模式 */
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_DCVS_v3;
    req.dcvs_v3.set_dcvs_enable = 1;
    req.dcvs_v3.dcvs_enable = 1;
    req.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
    req.dcvs_v3.set_bus_params = 1;
    req.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_core_params = 1;
    req.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_sleep_disable = 1;
    req.dcvs_v3.sleep_disable = 1;
    if (HAP_power_set((void *)&power_ctx, &req) != 0) return -2;

    /* 上电 HVX */
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HVX;
    req.hvx.power_up = 1;
    if (HAP_power_set((void *)&power_ctx, &req) != 0) return -3;

    /* 上电 HMX */
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HMX;
    req.hmx.power_up = 1;
    if (HAP_power_set((void *)&power_ctx, &req) != 0) return -4;

    return 0;
}

/* ============================================================
 * main
 * ============================================================ */
int main(int argc, char **argv)
{
    int pass = 0, fail = 0;
    int ret;

    FARF(ALWAYS, "========================================");
    FARF(ALWAYS, "  Chapter 2: HVX + HMX on Real Device");
    FARF(ALWAYS, "========================================");

    /* ---- Step 1: 上电 HVX/HMX ---- */
    FARF(ALWAYS, "[Power] Setting up HVX + HMX...");
    ret = power_on_hvx_hmx();
    if (ret != 0) {
        FARF(ALWAYS, "[Power] FAILED (ret=%d)", ret);
        return 1;
    }
    FARF(ALWAYS, "[Power] OK");

    /* ---- Step 2: 分配 VTCM + 锁定 HMX ---- */
    FARF(ALWAYS, "[Init] Allocating VTCM + HMX lock...");
    unsigned int vtcm_size = 8 * 1024 * 1024;
    HAP_compute_res_query_VTCM(0, &vtcm_size, NULL, NULL, NULL);
    FARF(ALWAYS, "[Init] VTCM total = %u bytes (%u KB)", vtcm_size, vtcm_size / 1024);

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_vtcm_param(&attr, vtcm_size, 1);
    HAP_compute_res_attr_set_hmx_param(&attr, 1);
    /* 注意: 不要设置 cache_mode 和 serialize, 否则 hmx_lock 会失败! */

    unsigned int ctx_id = HAP_compute_res_acquire(&attr, 100000);
    if (ctx_id == 0) {
        FARF(ALWAYS, "[Init] FAILED: compute_res_acquire returned 0");
        return 1;
    }

    void *vtcm_base = HAP_compute_res_attr_get_vtcm_ptr(&attr);
    if (!vtcm_base) {
        FARF(ALWAYS, "[Init] FAILED: VTCM pointer is NULL");
        HAP_compute_res_release(ctx_id);
        return 1;
    }

    ret = HAP_compute_res_hmx_lock(ctx_id);
    if (ret != 0) {
        FARF(ALWAYS, "[Init] FAILED: hmx_lock returned %d", ret);
        HAP_compute_res_release(ctx_id);
        return 1;
    }
    FARF(ALWAYS, "[Init] VTCM=%p  HMX locked", vtcm_base);

    /* 分配 VTCM 缓冲区 */
    unsigned char *base = (unsigned char *)vtcm_base;
    unsigned short *act = (unsigned short *)(base + 0x0000);
    unsigned short *wt  = (unsigned short *)(base + 0x1000);
    unsigned short *out = (unsigned short *)(base + 0x2000);
    unsigned char  *scl = (unsigned char  *)(base + 0x3000);

    /* ================================================================
     * Test 1: HVX fill → HMX matmul
     *
     * 和第一章完全相同的测试, 但跑在真机 CDSP 上
     * ================================================================ */
    FARF(ALWAYS, "");
    FARF(ALWAYS, "-- Test 1: HVX fill -> HMX matmul ----------");

    hvx_fill_f16(act, F16_ONE, TILE_ELEMS);
    hvx_fill_f16(wt, F16_TWO, TILE_ELEMS);
    hvx_fill_scales(scl, F16_ONE);

    FARF(ALWAYS, "  act: 0x%04X 0x%04X 0x%04X 0x%04X",
         act[0], act[1], act[2], act[3]);
    FARF(ALWAYS, "  wt : 0x%04X 0x%04X 0x%04X 0x%04X",
         wt[0], wt[1], wt[2], wt[3]);

    hmx_set_scales(scl);
    hmx_clear_acc();
    hmx_load_tile_pair(act, wt);
    hmx_store_acc(out);

    FARF(ALWAYS, "  out: 0x%04X 0x%04X 0x%04X 0x%04X",
         out[0], out[1], out[2], out[3]);
    float got = f16_to_f32(out[0]);
    FARF(ALWAYS, "  result=%.1f  expected~=64.0", got);
    if (got >= 63.0f && got <= 66.0f) { FARF(ALWAYS, "[PASS] Test 1"); pass++; }
    else                              { FARF(ALWAYS, "[FAIL] Test 1"); fail++; }

    /* ================================================================
     * Test 2: HMX matmul → HVX ReLU
     * ================================================================ */
    FARF(ALWAYS, "");
    FARF(ALWAYS, "-- Test 2: HMX matmul -> HVX ReLU ----------");

    hvx_fill_f16(act, F16_NEG_ONE, TILE_ELEMS);
    hvx_fill_f16(wt, F16_ONE, TILE_ELEMS);
    hvx_fill_scales(scl, F16_ONE);

    hmx_set_scales(scl);
    hmx_clear_acc();
    hmx_load_tile_pair(act, wt);
    hmx_store_acc(out);

    FARF(ALWAYS, "  matmul: 0x%04X 0x%04X (%.1f)",
         out[0], out[1], f16_to_f32(out[0]));

    hvx_relu_f16(out, TILE_ELEMS);

    FARF(ALWAYS, "  relu:   0x%04X 0x%04X (%.1f)",
         out[0], out[1], f16_to_f32(out[0]));

    int all_zero = 1;
    { int i; for (i = 0; i < TILE_ELEMS; i++) if (out[i] != 0) all_zero = 0; }
    if (all_zero) { FARF(ALWAYS, "[PASS] Test 2"); pass++; }
    else          { FARF(ALWAYS, "[FAIL] Test 2"); fail++; }

    /* ================================================================
     * Test 3: 多 tile 累加
     * ================================================================ */
    FARF(ALWAYS, "");
    FARF(ALWAYS, "-- Test 3: Multi-tile accumulation ---------");

    hvx_fill_f16(act, F16_ONE, TILE_ELEMS);
    hvx_fill_f16(wt, F16_ONE, TILE_ELEMS);
    hvx_fill_scales(scl, F16_ONE);

    hmx_set_scales(scl);
    hmx_clear_acc();
    hmx_load_tile_pair(act, wt);   /* +32 */
    hmx_load_tile_pair(act, wt);   /* +32 */
    hmx_store_acc(out);

    FARF(ALWAYS, "  out: 0x%04X 0x%04X 0x%04X 0x%04X",
         out[0], out[1], out[2], out[3]);
    got = f16_to_f32(out[0]);
    FARF(ALWAYS, "  result=%.1f  expected~=64.0", got);
    if (got >= 63.0f && got <= 66.0f) { FARF(ALWAYS, "[PASS] Test 3"); pass++; }
    else                              { FARF(ALWAYS, "[FAIL] Test 3"); fail++; }

    /* ================================================================ */
    FARF(ALWAYS, "");
    FARF(ALWAYS, "========================================");
    FARF(ALWAYS, "  Results: %d PASS / %d FAIL", pass, fail);
    FARF(ALWAYS, "========================================");

    /* 释放资源 */
    HAP_compute_res_hmx_unlock(ctx_id);
    HAP_compute_res_release(ctx_id);
    return fail;
}
