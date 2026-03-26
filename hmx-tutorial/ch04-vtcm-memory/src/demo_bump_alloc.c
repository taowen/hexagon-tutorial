/*
 * Part 2: llama.cpp bump allocator — vtcm_seq_alloc
 *
 * Simplest VTCM partitioning: pointer increment, 128-byte aligned, no free.
 * Reset to vtcm_base at the start of each op.
 */

#include "common.h"

void demo_bump_alloc(void)
{
    /*
     * Simulate llama.cpp hmx-matmul-ops.c VTCM layout:
     *
     * llama.cpp code:
     *   uint8_t *vtcm_ptr = (uint8_t *)ctx->vtcm_base;
     *   __fp16 *vtcm_weight     = (__fp16 *)vtcm_seq_alloc(&vtcm_ptr, weight_size);
     *   __fp16 *vtcm_activation = (__fp16 *)vtcm_seq_alloc(&vtcm_ptr, act_size);
     *   __fp16 *vtcm_output     = (__fp16 *)vtcm_seq_alloc(&vtcm_ptr, output_size);
     *   void   *vtcm_scratch0   = vtcm_seq_alloc(&vtcm_ptr, scratch_size);
     *   void   *vtcm_scratch1   = vtcm_seq_alloc(&vtcm_ptr, scratch_size);
     *   __fp16 *vtcm_scales     = (__fp16 *)vtcm_seq_alloc(&vtcm_ptr, 256);
     *
     * We allocate 5 regions using the same pattern:
     */
    uint8_t *ptr = g_vtcm_base;

    /* Assume matmul: M=128, N=128, K=128, FP16 */
    size_t weight_size  = 128 * 128 * 2;   /* 32 KB */
    size_t act_size     = 128 * 128 * 2;   /* 32 KB */
    size_t output_size  = 128 * 128 * 2;   /* 32 KB */
    size_t scratch_size = 64 * 1024;       /* 64 KB (DMA double buffer) */
    size_t scales_size  = 256;             /* scales vector */

    uint8_t *weight  = vtcm_seq_alloc(&ptr, weight_size);
    uint8_t *act     = vtcm_seq_alloc(&ptr, act_size);
    uint8_t *output  = vtcm_seq_alloc(&ptr, output_size);
    uint8_t *scratch0= vtcm_seq_alloc(&ptr, scratch_size);
    uint8_t *scratch1= vtcm_seq_alloc(&ptr, scratch_size);
    uint8_t *scales  = vtcm_seq_alloc(&ptr, scales_size);

    size_t used = (size_t)(ptr - g_vtcm_base);

    FARF(ALWAYS, "  VTCM layout (bump allocator):");
    FARF(ALWAYS, "    weight:   +0x%05X  (%u KB)",
         (unsigned)(weight - g_vtcm_base), (unsigned)(weight_size / 1024));
    FARF(ALWAYS, "    act:      +0x%05X  (%u KB)",
         (unsigned)(act - g_vtcm_base), (unsigned)(act_size / 1024));
    FARF(ALWAYS, "    output:   +0x%05X  (%u KB)",
         (unsigned)(output - g_vtcm_base), (unsigned)(output_size / 1024));
    FARF(ALWAYS, "    scratch0: +0x%05X  (%u KB)",
         (unsigned)(scratch0 - g_vtcm_base), (unsigned)(scratch_size / 1024));
    FARF(ALWAYS, "    scratch1: +0x%05X  (%u KB)",
         (unsigned)(scratch1 - g_vtcm_base), (unsigned)(scratch_size / 1024));
    FARF(ALWAYS, "    scales:   +0x%05X  (%u B)",
         (unsigned)(scales - g_vtcm_base), (unsigned)scales_size);
    FARF(ALWAYS, "    total:    %u KB / %u KB (%.1f%%)",
         (unsigned)(used / 1024), g_vtcm_size / 1024,
         100.0f * used / g_vtcm_size);

    /* Verify: write and read back */
    memset(weight, 0xAA, weight_size);
    memset(act, 0xBB, act_size);
    int ok = (weight[0] == 0xAA && act[0] == 0xBB &&
              weight[weight_size - 1] == 0xAA);
    FARF(ALWAYS, "    read/write: %s", ok ? "OK" : "FAIL");
}
