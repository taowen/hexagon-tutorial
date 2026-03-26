/*
 * Part 5: HVX benchmark — VTCM vs DDR
 *
 * Compare VTCM and DDR access performance using HVX vadd.
 */

#include "common.h"

/* HVX vector add — used by both this file and demo_dma.c */
void hvx_vadd(const uint8_t *a, const uint8_t *b, uint8_t *c,
              uint32_t n_bytes)
{
    for (uint32_t i = 0; i < n_bytes; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)&a[i];
        HVX_Vector vb = *(const HVX_Vector *)&b[i];
        *(HVX_Vector *)&c[i] = Q6_Vb_vadd_VbVb(va, vb);
    }
}

void demo_hvx_bench(void)
{
    /* Use 256 KB buffers — larger than L2 cache (~1 MB shared, less available) */
    uint32_t buf_kb = 256;
    uint32_t buf_bytes = buf_kb * 1024;
    uint32_t n_iters = 500;

    /* DDR path */
    uint8_t *ddr_a = (uint8_t *)malloc(buf_bytes + 128);
    uint8_t *ddr_b = (uint8_t *)malloc(buf_bytes + 128);
    uint8_t *ddr_c = (uint8_t *)malloc(buf_bytes + 128);
    /* Manual 128-byte alignment */
    uint8_t *a_ddr = (uint8_t *)(((uintptr_t)ddr_a + 127) & ~(uintptr_t)127);
    uint8_t *b_ddr = (uint8_t *)(((uintptr_t)ddr_b + 127) & ~(uintptr_t)127);
    uint8_t *c_ddr = (uint8_t *)(((uintptr_t)ddr_c + 127) & ~(uintptr_t)127);

    if (!ddr_a || !ddr_b || !ddr_c) {
        FARF(ALWAYS, "  DDR malloc failed");
        free(ddr_a); free(ddr_b); free(ddr_c);
        return;
    }

    memset(a_ddr, 1, buf_bytes);
    memset(b_ddr, 2, buf_bytes);

    uint64_t t1 = HAP_perf_get_time_us();
    for (uint32_t i = 0; i < n_iters; i++)
        hvx_vadd(a_ddr, b_ddr, c_ddr, buf_bytes);
    uint64_t t_ddr = HAP_perf_get_time_us() - t1;

    /* VTCM path */
    uint8_t *ptr = g_vtcm_base;
    uint8_t *a_vtcm = vtcm_seq_alloc(&ptr, buf_bytes);
    uint8_t *b_vtcm = vtcm_seq_alloc(&ptr, buf_bytes);
    uint8_t *c_vtcm = vtcm_seq_alloc(&ptr, buf_bytes);

    memset(a_vtcm, 1, buf_bytes);
    memset(b_vtcm, 2, buf_bytes);

    uint64_t t2 = HAP_perf_get_time_us();
    for (uint32_t i = 0; i < n_iters; i++)
        hvx_vadd(a_vtcm, b_vtcm, c_vtcm, buf_bytes);
    uint64_t t_vtcm = HAP_perf_get_time_us() - t2;

    /* Verify */
    int ok = (c_ddr[0] == 3 && c_vtcm[0] == 3);

    FARF(ALWAYS, "  HVX vadd: %u KB x %u iters", buf_kb, n_iters);
    FARF(ALWAYS, "    DDR:  %llu us", (unsigned long long)t_ddr);
    FARF(ALWAYS, "    VTCM: %llu us", (unsigned long long)t_vtcm);
    if (t_vtcm > 0)
        FARF(ALWAYS, "    Speedup: %.2fx", (float)t_ddr / (float)t_vtcm);
    FARF(ALWAYS, "    Verify: %s", ok ? "OK" : "FAIL");

    free(ddr_a); free(ddr_b); free(ddr_c);
}
