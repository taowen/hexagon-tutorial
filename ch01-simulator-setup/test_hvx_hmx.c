/*
 * test_hvx_hmx.c — 第一章示例: HVX + HMX 混合运行在模拟器上
 *
 * 演示:
 *   1. H2 Hypervisor 初始化 (获取 VTCM, HVX/HMX 上下文)
 *   2. HVX 向量化填充 F16 矩阵
 *   3. HMX 执行 32x32 matmul
 *   4. HVX ReLU 后处理
 */

#include <stdio.h>
#include <string.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_vecaccess.h>
#include <h2_mxaccess.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

/* ============================================================
 * 常量
 * ============================================================ */
#define TILE_DIM    32
#define TILE_ELEMS  (TILE_DIM * TILE_DIM)   /* 1024 */
#define TILE_BYTES  (TILE_ELEMS * 2)        /* 2048 */
#define HVX_VLEN    128                     /* HVX 128B 模式 */
#define F16_PER_VEC (HVX_VLEN / 2)         /* 64 个 F16/vector */

/* F16 bit patterns */
#define F16_ONE     0x3C00  /* 1.0 */
#define F16_TWO     0x4000  /* 2.0 */
#define F16_NEG_ONE 0xBC00  /* -1.0 */

/* ============================================================
 * HMX inline asm wrappers
 *
 * HMX 指令必须用 inline asm 调用, 因为:
 * - activation + weight load 必须在同一个 VLIW packet (大括号)
 * - 编译器目前不支持单独的 HMX intrinsics
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
    /* activation 和 weight 必须在同一 VLIW packet 中加载 */
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
 * HVX 工具函数
 * ============================================================ */

/* 用 HVX 向量化填充 F16 缓冲区, 一次写 64 个 F16 (128B) */
static void hvx_fill_f16(unsigned short *buf, unsigned short val, int count)
{
    int splat_word = (val << 16) | val;
    HVX_Vector v_val = Q6_V_vsplat_R(splat_word);
    HVX_Vector *vp = (HVX_Vector *)buf;
    int i;
    for (i = 0; i < count / F16_PER_VEC; i++)
        vp[i] = v_val;
}

/*
 * 填充 256B 的 HMX scale 缓冲区
 * 格式: 前 128B = 64 个 F16 scale 值, 后 128B 必须为 0
 */
static void hvx_fill_scales(unsigned char *scale_buf, unsigned short val)
{
    int splat_word = (val << 16) | val;
    ((HVX_Vector *)scale_buf)[0] = Q6_V_vsplat_R(splat_word);
    ((HVX_Vector *)scale_buf)[1] = Q6_V_vzero();
}

/* HVX ReLU: max(x, 0), 利用 F16 与有符号整数排序一致的特性 */
static void hvx_relu_f16(unsigned short *buf, int count)
{
    HVX_Vector v_zero = Q6_V_vzero();
    HVX_Vector *vp = (HVX_Vector *)buf;
    int i;
    for (i = 0; i < count / F16_PER_VEC; i++)
        vp[i] = Q6_Vh_vmax_VhVh(vp[i], v_zero);
}

/* ============================================================
 * F16 打印辅助
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

static void print_f16(const char *label, unsigned short *buf, int n)
{
    int i;
    printf("  %s:", label);
    for (i = 0; i < n && i < 4; i++)
        printf(" 0x%04X(%.1f)", buf[i], f16_to_f32(buf[i]));
    printf("\n");
}

/* ============================================================
 * main
 * ============================================================ */
int main(int argc, char **argv)
{
    int pass = 0, fail = 0;

    printf("========================================\n");
    printf("  Chapter 1: HVX + HMX on Simulator\n");
    printf("========================================\n\n");

    /* ---- 初始化 ---- */
    unsigned int vtcm_base = h2_info(INFO_VTCM_BASE);
    unsigned int vtcm_size = h2_info(INFO_VTCM_SIZE);
    printf("[Init] VTCM base=0x%08x  size=%u KB\n", vtcm_base, vtcm_size);

    if (vtcm_base == 0) {
        printf("ERROR: No VTCM!\n");
        h2_thread_stop(1);
        return 1;
    }

    /* 在 VTCM 上分配缓冲区 (2048B 对齐) */
    unsigned short *act = (unsigned short *)(unsigned long)(vtcm_base + 0x0000);
    unsigned short *wt  = (unsigned short *)(unsigned long)(vtcm_base + 0x1000);
    unsigned short *out = (unsigned short *)(unsigned long)(vtcm_base + 0x2000);
    unsigned char  *scl = (unsigned char  *)(unsigned long)(vtcm_base + 0x3000);

    /* 获取 HVX 上下文 */
    h2_vecaccess_state_t vacc;
    h2_vecaccess_unit_init(&vacc, H2_VECACCESS_HVX_128,
                           CFG_TYPE_VXU0, CFG_SUBTYPE_VXU0,
                           CFG_HVX_CONTEXTS, 0x1);
    h2_vecaccess_ret_t vret = h2_vecaccess_acquire(&vacc);
    printf("[Init] HVX acquired (idx=%d)\n", vret.idx);

    /* 获取 HMX 上下文 */
    h2_mxaccess_state_t mxacc;
    h2_mxaccess_unit_init(&mxacc, CFG_TYPE_VXU0, CFG_SUBTYPE_VXU0,
                          CFG_HMX_CONTEXTS, 0x1);
    int mret = h2_mxaccess_acquire(&mxacc);
    printf("[Init] HMX acquired (%d)\n\n", mret);

    /* ================================================================
     * Test 1: HVX fill → HMX matmul
     *
     * act = ones(32,32), wt = 2*ones(32,32)
     * 结果 = 1.0 * 2.0 * 32 = 64.0 (F16 约 65.0)
     * ================================================================ */
    printf("-- Test 1: HVX fill -> HMX matmul ----------\n");

    hvx_fill_f16(act, F16_ONE, TILE_ELEMS);
    hvx_fill_f16(wt, F16_TWO, TILE_ELEMS);
    hvx_fill_scales(scl, F16_ONE);

    print_f16("act", act, 4);
    print_f16("wt ", wt,  4);

    hmx_set_scales(scl);
    hmx_clear_acc();
    hmx_load_tile_pair(act, wt);
    hmx_store_acc(out);

    print_f16("out", out, 4);
    float got = f16_to_f32(out[0]);
    printf("  result=%.1f  expected~=64.0\n", got);
    if (got >= 63.0f && got <= 66.0f) { printf("[PASS] Test 1\n\n"); pass++; }
    else                              { printf("[FAIL] Test 1\n\n"); fail++; }

    /* ================================================================
     * Test 2: HMX matmul → HVX ReLU
     *
     * act = -1.0, wt = 1.0 → matmul = -32.0 → ReLU → 0.0
     * ================================================================ */
    printf("-- Test 2: HMX matmul -> HVX ReLU ----------\n");

    hvx_fill_f16(act, F16_NEG_ONE, TILE_ELEMS);
    hvx_fill_f16(wt, F16_ONE, TILE_ELEMS);
    hvx_fill_scales(scl, F16_ONE);

    hmx_set_scales(scl);
    hmx_clear_acc();
    hmx_load_tile_pair(act, wt);
    hmx_store_acc(out);

    print_f16("matmul", out, 4);
    printf("  matmul result=%.1f  expected~=-32.0\n", f16_to_f32(out[0]));

    hvx_relu_f16(out, TILE_ELEMS);
    print_f16("relu  ", out, 4);

    /* 验证全部清零 */
    int all_zero = 1;
    { int i; for (i = 0; i < TILE_ELEMS; i++) if (out[i] != 0) all_zero = 0; }
    if (all_zero) { printf("[PASS] Test 2\n\n"); pass++; }
    else          { printf("[FAIL] Test 2\n\n"); fail++; }

    /* ================================================================
     * Test 3: 多 tile 累加
     *
     * 不清零累加器, 连续 2 次 MAC: 32 + 32 = 64.0
     * ================================================================ */
    printf("-- Test 3: Multi-tile accumulation ---------\n");

    hvx_fill_f16(act, F16_ONE, TILE_ELEMS);
    hvx_fill_f16(wt, F16_ONE, TILE_ELEMS);
    hvx_fill_scales(scl, F16_ONE);

    hmx_set_scales(scl);
    hmx_clear_acc();
    hmx_load_tile_pair(act, wt);   /* +32 */
    hmx_load_tile_pair(act, wt);   /* +32 */
    hmx_store_acc(out);

    print_f16("out", out, 4);
    got = f16_to_f32(out[0]);
    printf("  result=%.1f  expected~=64.0\n", got);
    if (got >= 63.0f && got <= 66.0f) { printf("[PASS] Test 3\n\n"); pass++; }
    else                              { printf("[FAIL] Test 3\n\n"); fail++; }

    /* ================================================================ */
    printf("========================================\n");
    printf("  Results: %d PASS / %d FAIL\n", pass, fail);
    printf("========================================\n");

    h2_thread_stop(fail);
    return fail;
}
