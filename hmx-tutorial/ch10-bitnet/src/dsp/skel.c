/*
 * skel.c -- DSP side for ch10 BitNet VLUT16 exploration
 *
 * Implements FastRPC lifecycle (open/close/start/stop) and
 * dspqueue packet callback that dispatches VLUT16 tests.
 */

#define FARF_ERROR 1
#define FARF_HIGH 1
#include <HAP_farf.h>
#include <HAP_power.h>
#include <HAP_perf.h>
#include <string.h>
#include <stdlib.h>
#include <AEEStdErr.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

#include "bitnet_test.h"
#include "dspqueue.h"
#include "common/protocol.h"
#include "dsp/bitnet_gemv.h"
#include "dsp/bitnet_ops.h"
#include "dsp/bitnet_decoder.h"

/* ========== DSP context ========== */

struct dsp_ctx {
    dspqueue_t queue;
    uint64_t   total_us;
};

/* ========== VLUT16 Tests ========== */

/*
 * Test 1: Basic VLUT16 behavior
 *
 * VLUT16 does a 16-entry lookup: each byte in the index vector provides
 * a 4-bit index (lower nibble) selecting one of 16 int16 entries.
 *
 * The result is a VectorPair:
 *   lo = results for even-indexed bytes (byte 0, 2, 4, ..., 126)
 *   hi = results for odd-indexed bytes (byte 1, 3, 5, ..., 127)
 *
 * The LUT vector must be in a shuffled/interleaved layout — a simple
 * linear array does NOT work. We use vlut16_build_lut() to prepare it.
 */
static uint32_t run_vlut16_test1(void) {
    FARF(HIGH, "=== Test 1: Basic VLUT16 behavior ===");

    /* Create LUT: 16 entries {0, 100, 200, ..., 1500} as int16 */
    int16_t lut_data[16];
    for (int i = 0; i < 16; i++) {
        lut_data[i] = (int16_t)(i * 100);
    }

    /* Build properly shuffled LUT vector */
    HVX_Vector lut_vec = vlut16_build_lut(lut_data);

    /* Create index vector: each byte has a 4-bit index.
     * Even bytes get indices 0,1,2,...,15 repeating.
     * Odd bytes get the same pattern. */
    uint8_t __attribute__((aligned(128))) idx_arr[128];
    for (int i = 0; i < 128; i++) {
        idx_arr[i] = (uint8_t)(i % 16);
    }
    HVX_Vector idx_vec = *(HVX_Vector *)idx_arr;

    /* Lookup with segment 0 */
    HVX_VectorPair result = Q6_Wh_vlut16_VbVhR_nomatch(idx_vec, lut_vec, 0);

    /* lo vector: results for even bytes (0, 2, 4, ..., 126) */
    int16_t __attribute__((aligned(128))) out_lo[64];
    *(HVX_Vector *)out_lo = Q6_V_lo_W(result);

    /* hi vector: results for odd bytes (1, 3, 5, ..., 127) */
    int16_t __attribute__((aligned(128))) out_hi[64];
    *(HVX_Vector *)out_hi = Q6_V_hi_W(result);

    /* Verify: out_lo[j] should be lut_data[idx_arr[2*j] % 16]
     *         out_hi[j] should be lut_data[idx_arr[2*j+1] % 16] */
    uint32_t pass = 1;
    int lo_ok = 0, hi_ok = 0;
    for (int j = 0; j < 64; j++) {
        int16_t expected_lo = lut_data[idx_arr[2 * j] % 16];
        int16_t expected_hi = lut_data[idx_arr[2 * j + 1] % 16];
        if (out_lo[j] == expected_lo) lo_ok++;
        if (out_hi[j] == expected_hi) hi_ok++;
    }

    FARF(HIGH, "Segment 0: lo %d/64 correct, hi %d/64 correct", lo_ok, hi_ok);

    /* Dump first 16 of lo for inspection */
    FARF(HIGH, "lo[0..15]: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
         (int)out_lo[0], (int)out_lo[1], (int)out_lo[2], (int)out_lo[3],
         (int)out_lo[4], (int)out_lo[5], (int)out_lo[6], (int)out_lo[7],
         (int)out_lo[8], (int)out_lo[9], (int)out_lo[10], (int)out_lo[11],
         (int)out_lo[12], (int)out_lo[13], (int)out_lo[14], (int)out_lo[15]);
    FARF(HIGH, "expected:  %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
         (int)lut_data[idx_arr[0] % 16], (int)lut_data[idx_arr[2] % 16],
         (int)lut_data[idx_arr[4] % 16], (int)lut_data[idx_arr[6] % 16],
         (int)lut_data[idx_arr[8] % 16], (int)lut_data[idx_arr[10] % 16],
         (int)lut_data[idx_arr[12] % 16], (int)lut_data[idx_arr[14] % 16],
         (int)lut_data[idx_arr[16] % 16], (int)lut_data[idx_arr[18] % 16],
         (int)lut_data[idx_arr[20] % 16], (int)lut_data[idx_arr[22] % 16],
         (int)lut_data[idx_arr[24] % 16], (int)lut_data[idx_arr[26] % 16],
         (int)lut_data[idx_arr[28] % 16], (int)lut_data[idx_arr[30] % 16]);

    /* Also try all 4 segments to see if they all work the same */
    for (int seg = 0; seg < 4; seg++) {
        HVX_VectorPair r = Q6_Wh_vlut16_VbVhR_nomatch(idx_vec, lut_vec, seg);
        int16_t __attribute__((aligned(128))) tmp[64];
        *(HVX_Vector *)tmp = Q6_V_lo_W(r);
        int ok = 0;
        for (int j = 0; j < 64; j++) {
            if (tmp[j] == lut_data[idx_arr[2 * j] % 16]) ok++;
        }
        FARF(HIGH, "  Segment %d: lo %d/64 correct", seg, ok);
    }

    if (lo_ok == 64 && hi_ok == 64) {
        FARF(HIGH, "Test 1: PASS");
    } else {
        FARF(HIGH, "Test 1: FAIL (lo=%d/64, hi=%d/64)", lo_ok, hi_ok);
        pass = 0;
    }

    return pass;
}

/*
 * Test 2: Even/odd byte mapping
 *
 * Demonstrates that VLUT16 splits results by even/odd byte index:
 * - Put distinct indices in even vs odd byte positions
 * - Verify lo vector has even-byte results, hi has odd-byte results
 */
static uint32_t run_vlut16_test2(void) {
    FARF(HIGH, "=== Test 2: Even/odd byte mapping ===");

    int16_t lut_data[16];
    for (int i = 0; i < 16; i++) {
        lut_data[i] = (int16_t)(i * 100);
    }
    HVX_Vector lut_vec = vlut16_build_lut(lut_data);

    /* Even bytes: index 5 (expect 500), Odd bytes: index 10 (expect 1000) */
    uint8_t __attribute__((aligned(128))) idx_arr[128];
    for (int i = 0; i < 128; i++) {
        idx_arr[i] = (i % 2 == 0) ? 5 : 10;
    }
    HVX_Vector idx_vec = *(HVX_Vector *)idx_arr;

    HVX_VectorPair result = Q6_Wh_vlut16_VbVhR_nomatch(idx_vec, lut_vec, 0);

    int16_t __attribute__((aligned(128))) out_lo[64];
    int16_t __attribute__((aligned(128))) out_hi[64];
    *(HVX_Vector *)out_lo = Q6_V_lo_W(result);
    *(HVX_Vector *)out_hi = Q6_V_hi_W(result);

    /* lo should be all 500, hi should be all 1000 */
    int lo_ok = 0, hi_ok = 0;
    for (int j = 0; j < 64; j++) {
        if (out_lo[j] == 500) lo_ok++;
        if (out_hi[j] == 1000) hi_ok++;
    }

    FARF(HIGH, "Even bytes (idx=5, expect 500): lo = %d (first), %d/64 correct",
         (int)out_lo[0], lo_ok);
    FARF(HIGH, "Odd bytes (idx=10, expect 1000): hi = %d (first), %d/64 correct",
         (int)out_hi[0], hi_ok);

    uint32_t pass = (lo_ok == 64 && hi_ok == 64) ? 1 : 0;
    FARF(HIGH, "Test 2: %s", pass ? "PASS" : "FAIL");
    return pass;
}

/*
 * Test 3: "Matmul via lookup" demo (T-MAN style)
 *
 * Binary weights {-1, +1}: for 4 activations, there are 2^4=16 possible
 * dot products. We precompute all 16 into a LUT and use VLUT16 to look
 * up results for specific weight patterns.
 *
 * Weight encoding: bit i: 0 = -1, 1 = +1 (matching T-MAN convention)
 *   index 0b1111 = +x0 +x1 +x2 +x3
 *   index 0b0000 = -x0 -x1 -x2 -x3
 *   index 0b0101 = -x0 +x1 -x2 +x3
 */
static uint32_t run_vlut16_test3(void) {
    FARF(HIGH, "=== Test 3: Matmul via VLUT16 lookup (T-MAN style) ===");

    float act[4] = { 1.0f, 2.0f, 3.0f, 4.0f };

    /* Precompute 16 LUT entries: bit i=1 means +act[i], bit i=0 means -act[i] */
    int16_t lut_q[16];
    for (int idx = 0; idx < 16; idx++) {
        float sum = 0.0f;
        for (int b = 0; b < 4; b++) {
            float sign = (idx & (1 << b)) ? 1.0f : -1.0f;
            sum += sign * act[b];
        }
        lut_q[idx] = (int16_t)(sum * 256.0f);  /* scale by 256 */
        FARF(HIGH, "  LUT[%d] = %d (dot=%.1f)", idx, (int)lut_q[idx], (double)sum);
    }

    HVX_Vector lut_vec = vlut16_build_lut(lut_q);

    /* Test cases — put indices in even byte positions */
    struct { uint8_t index; float expected; } cases[] = {
        { 0x0F, 10.0f },   /* +1+2+3+4 = 10 */
        { 0x00, -10.0f },  /* -1-2-3-4 = -10 */
        { 0x05,  -2.0f },  /* -1+2-3+4 = -2 (bits: 0101) -- wait: bit0=1(+1), bit1=0(-2), bit2=1(+3), bit3=0(-4) = 1-2+3-4=-2 */
        { 0x0A,   2.0f },  /* +1-2+3-4? No: bit0=0(-1), bit1=1(+2), bit2=0(-3), bit3=1(+4) = -1+2-3+4=2 */
    };
    int n_cases = sizeof(cases) / sizeof(cases[0]);

    /* Place test indices in even byte positions (0, 2, 4, 6) so results
     * appear in the lo vector at positions 0, 1, 2, 3 */
    uint8_t __attribute__((aligned(128))) idx_arr[128];
    memset(idx_arr, 0, 128);
    for (int i = 0; i < n_cases; i++) {
        idx_arr[i * 2] = cases[i].index;  /* even bytes */
    }
    HVX_Vector idx_vec = *(HVX_Vector *)idx_arr;

    HVX_VectorPair result = Q6_Wh_vlut16_VbVhR_nomatch(idx_vec, lut_vec, 0);
    int16_t __attribute__((aligned(128))) out_lo[64];
    *(HVX_Vector *)out_lo = Q6_V_lo_W(result);

    uint32_t pass = 1;
    for (int i = 0; i < n_cases; i++) {
        int16_t expected_q = (int16_t)(cases[i].expected * 256.0f);
        int16_t got = out_lo[i];
        int ok = (got == expected_q);
        FARF(HIGH, "  Case %d: idx=0x%02x, got=%d, expected=%d (%s)",
             i, cases[i].index, (int)got, (int)expected_q, ok ? "PASS" : "FAIL");
        if (!ok) pass = 0;
    }

    FARF(HIGH, "Test 3: %s", pass ? "PASS" : "FAIL");
    return pass;
}

/* ========== Segment Mapping Test ========== */

/*
 * Test 4: Verify VLUT16 segment-to-quarter mapping
 *
 * Builds a 4-LUT vector with known distinct values per quarter,
 * then calls VLUT16 with segments 0-3 to see which segment reads which quarter.
 * This is essential for getting the 4Q optimization correct.
 */
static uint32_t run_segment_mapping_test(void) {
    FARF(HIGH, "=== Test 4: VLUT16 Segment Mapping ===");

    /* 4 LUTs with distinct values so we can identify which segment reads which */
    int16_t lut0[16], lut1[16], lut2[16], lut3[16];
    for (int i = 0; i < 16; i++) {
        lut0[i] = (int16_t)(100 + i);    /* 100-115 */
        lut1[i] = (int16_t)(200 + i);    /* 200-215 */
        lut2[i] = (int16_t)(300 + i);    /* 300-315 */
        lut3[i] = (int16_t)(400 + i);    /* 400-415 */
    }

    HVX_Vector lut_vec = vlut16_build_lut_4q(lut0, lut1, lut2, lut3);

    /* Index vector: all bytes = 5 (lookup entry 5 from each LUT) */
    HVX_Vector idx_vec = Q6_Vb_vsplat_R(5);

    /* Expected values for entry 5: lut0=105, lut1=205, lut2=305, lut3=405 */
    int16_t expected[4] = { 105, 205, 305, 405 };

    uint32_t pass = 1;
    int seg_map[4] = {-1, -1, -1, -1};  /* which LUT each segment reads */

    for (int seg = 0; seg < 4; seg++) {
        HVX_VectorPair r = Q6_Wh_vlut16_VbVhR_nomatch(idx_vec, lut_vec, seg);
        int16_t __attribute__((aligned(128))) out[64];
        *(HVX_Vector *)out = Q6_V_lo_W(r);

        int16_t got = out[0];
        int found = -1;
        for (int i = 0; i < 4; i++) {
            if (got == expected[i]) { found = i; break; }
        }
        seg_map[seg] = found;

        FARF(HIGH, "  Segment %d: got %d -> maps to lut%d (%s)",
             seg, (int)got, found,
             (found == seg) ? "expected" : "REMAPPED");
    }

    /* Verify all 4 LUTs are accessible (each segment maps to a unique LUT) */
    int used[4] = {0, 0, 0, 0};
    for (int s = 0; s < 4; s++) {
        if (seg_map[s] >= 0 && seg_map[s] < 4) {
            used[seg_map[s]] = 1;
        } else {
            FARF(HIGH, "  Segment %d: unmapped value!", s);
            pass = 0;
        }
    }
    for (int i = 0; i < 4; i++) {
        if (!used[i]) {
            FARF(HIGH, "  LUT %d not accessible by any segment!", i);
            pass = 0;
        }
    }

    FARF(HIGH, "Segment mapping: seg0->lut%d, seg1->lut%d, seg2->lut%d, seg3->lut%d",
         seg_map[0], seg_map[1], seg_map[2], seg_map[3]);
    FARF(HIGH, "Test 4 (Segment Mapping): %s", pass ? "PASS" : "FAIL");
    return pass;
}

/* ========== GEMV Test ========== */

/*
 * Simple LCG random number generator for reproducible test data.
 * Returns value in [0, 1).
 */
static uint32_t rng_state = 12345;
static float rng_float(void) {
    rng_state = rng_state * 1103515245 + 12345;
    return (float)(rng_state >> 16) / 65536.0f;
}

static uint32_t run_gemv_test(void) {
    FARF(HIGH, "=== GEMV Test: BitNet GEMV via VLUT16 ===");

    /* Test dimensions: M=256 (two VLUT16 passes), K=256 (64 Q groups) */
    #define GEMV_M 256
    #define GEMV_K 256
    #define GEMV_Q (GEMV_K / 4)

    /* Allocate test data (too large for stack at bigger sizes) */
    int8_t *W = (int8_t *)malloc(GEMV_M * GEMV_K);
    float *x = (float *)malloc(GEMV_K * sizeof(float));
    float *y_ref = (float *)malloc(GEMV_M * sizeof(float));
    float *y_vlut = (float *)malloc(GEMV_M * sizeof(float));
    /* packed_w must be 128-byte aligned for HVX */
    uint8_t *packed_w = (uint8_t *)memalign(128, 2 * GEMV_Q * GEMV_M);

    if (!W || !x || !y_ref || !y_vlut || !packed_w) {
        FARF(ERROR, "GEMV test: malloc failed");
        free(W); free(x); free(y_ref); free(y_vlut); free(packed_w);
        return 0;
    }

    /* Generate random ternary weights W[M][K] ∈ {-1, 0, +1} */
    rng_state = 42;
    for (int i = 0; i < GEMV_M * GEMV_K; i++) {
        float r = rng_float();
        if (r < 0.33f) W[i] = -1;
        else if (r < 0.66f) W[i] = 0;
        else W[i] = 1;
    }

    /* Generate random activations x[K] ∈ [-1, 1] */
    for (int k = 0; k < GEMV_K; k++) {
        x[k] = rng_float() * 2.0f - 1.0f;
    }

    /* Reference output (scalar) */
    bitnet_gemv_reference(x, W, y_ref, GEMV_M, GEMV_K);

    /* Pack weights for VLUT16 */
    bitnet_pack_weights(W, packed_w, GEMV_M, GEMV_K);

    /* VLUT16 GEMV */
    uint64_t t_start = HAP_perf_get_time_us();
    bitnet_gemv(x, packed_w, y_vlut, GEMV_M, GEMV_K);
    uint64_t t_end = HAP_perf_get_time_us();
    FARF(HIGH, "GEMV time: %llu us (M=%d, K=%d)", t_end - t_start, GEMV_M, GEMV_K);

    /* Compare results */
    float max_err = 0;
    float max_ref = 0;
    int mismatches = 0;
    for (int m = 0; m < GEMV_M; m++) {
        float err = y_vlut[m] - y_ref[m];
        if (err < 0) err = -err;
        if (err > max_err) max_err = err;
        float ref_abs = y_ref[m] > 0 ? y_ref[m] : -y_ref[m];
        if (ref_abs > max_ref) max_ref = ref_abs;
        if (err > 0.1f) mismatches++;
    }

    float rel_err = max_ref > 0 ? max_err / max_ref : 0;

    /* Print first few outputs for inspection */
    FARF(HIGH, "First 8 outputs (ref vs vlut16):");
    for (int m = 0; m < 8; m++) {
        FARF(HIGH, "  y[%d]: ref=%d/1000, vlut=%d/1000, err=%d/1000",
             m, (int)(y_ref[m] * 1000), (int)(y_vlut[m] * 1000),
             (int)((y_vlut[m] - y_ref[m]) * 1000));
    }
    FARF(HIGH, "Max absolute error: %d/10000", (int)(max_err * 10000));
    FARF(HIGH, "Max relative error: %d/10000", (int)(rel_err * 10000));
    FARF(HIGH, "Mismatches (>0.1): %d / %d", mismatches, GEMV_M);

    uint32_t pass = (rel_err < 0.05f) ? 1 : 0;  /* 5% relative error threshold */
    FARF(HIGH, "GEMV Test (original): %s (rel_err=%.4f)", pass ? "PASS" : "FAIL", (double)rel_err);

    /* ====== Optimized GEMV (global scale + int32 accum) ====== */
    FARF(HIGH, "--- Optimized GEMV (global scale + int32 accum) ---");

    float *y_opt = (float *)malloc(GEMV_M * sizeof(float));
    if (!y_opt) {
        FARF(ERROR, "Optimized GEMV: malloc failed");
        free(W); free(x); free(y_ref); free(y_vlut); free(packed_w);
        return 0;
    }

    /* Warm up */
    bitnet_gemv_opt(x, packed_w, y_opt, GEMV_M, GEMV_K);

    /* Timed run */
    uint64_t t_opt_start = HAP_perf_get_time_us();
    bitnet_gemv_opt(x, packed_w, y_opt, GEMV_M, GEMV_K);
    uint64_t t_opt_end = HAP_perf_get_time_us();
    FARF(HIGH, "GEMV opt time: %llu us (M=%d, K=%d)", t_opt_end - t_opt_start, GEMV_M, GEMV_K);

    /* Compare optimized vs reference */
    float max_err_opt = 0;
    float max_ref_opt = 0;
    int mismatches_opt = 0;
    for (int m = 0; m < GEMV_M; m++) {
        float err = y_opt[m] - y_ref[m];
        if (err < 0) err = -err;
        if (err > max_err_opt) max_err_opt = err;
        float ref_abs = y_ref[m] > 0 ? y_ref[m] : -y_ref[m];
        if (ref_abs > max_ref_opt) max_ref_opt = ref_abs;
        if (err > 0.1f) mismatches_opt++;
    }

    float rel_err_opt = max_ref_opt > 0 ? max_err_opt / max_ref_opt : 0;

    FARF(HIGH, "First 8 outputs (ref vs opt):");
    for (int m = 0; m < 8; m++) {
        FARF(HIGH, "  y[%d]: ref=%d/1000, opt=%d/1000, err=%d/1000",
             m, (int)(y_ref[m] * 1000), (int)(y_opt[m] * 1000),
             (int)((y_opt[m] - y_ref[m]) * 1000));
    }
    FARF(HIGH, "Opt max absolute error: %d/10000", (int)(max_err_opt * 10000));
    FARF(HIGH, "Opt max relative error: %d/10000", (int)(rel_err_opt * 10000));
    FARF(HIGH, "Opt mismatches (>0.1): %d / %d", mismatches_opt, GEMV_M);

    uint32_t pass_opt = (rel_err_opt < 0.05f) ? 1 : 0;
    FARF(HIGH, "GEMV Test (optimized): %s (rel_err=%.4f)", pass_opt ? "PASS" : "FAIL", (double)rel_err_opt);

    FARF(HIGH, "Speedup: original=%llu us, optimized=%llu us",
         t_end - t_start, t_opt_end - t_opt_start);

    /* ====== 4Q GEMV (4 Q positions per VLUT16 cycle) ====== */
    FARF(HIGH, "--- 4Q GEMV (4 Q positions per VLUT16 cycle) ---");

    /* Run segment mapping test first */
    uint32_t seg_pass = run_segment_mapping_test();

    float *y_4q = (float *)malloc(GEMV_M * sizeof(float));
    /* packed_4q: 2 * (Q/4) * 2 * M bytes = Q * M bytes */
    uint8_t *packed_4q = (uint8_t *)memalign(128, GEMV_Q * GEMV_M);

    if (!y_4q || !packed_4q) {
        FARF(ERROR, "4Q GEMV: malloc failed");
        free(y_4q); free(packed_4q);
        free(y_opt);
        free(W); free(x); free(y_ref); free(y_vlut); free(packed_w);
        return 0;
    }

    /* Pack weights in 4Q format */
    bitnet_pack_weights_4q(W, packed_4q, GEMV_M, GEMV_K);

    /* Warm up */
    bitnet_gemv_4q(x, packed_4q, y_4q, GEMV_M, GEMV_K);

    /* Timed run */
    uint64_t t_4q_start = HAP_perf_get_time_us();
    bitnet_gemv_4q(x, packed_4q, y_4q, GEMV_M, GEMV_K);
    uint64_t t_4q_end = HAP_perf_get_time_us();
    FARF(HIGH, "GEMV 4Q time: %llu us (M=%d, K=%d)", t_4q_end - t_4q_start, GEMV_M, GEMV_K);

    /* Compare 4Q vs reference */
    float max_err_4q = 0;
    float max_ref_4q = 0;
    int mismatches_4q = 0;
    for (int m = 0; m < GEMV_M; m++) {
        float err = y_4q[m] - y_ref[m];
        if (err < 0) err = -err;
        if (err > max_err_4q) max_err_4q = err;
        float ref_abs = y_ref[m] > 0 ? y_ref[m] : -y_ref[m];
        if (ref_abs > max_ref_4q) max_ref_4q = ref_abs;
        if (err > 0.1f) mismatches_4q++;
    }

    float rel_err_4q = max_ref_4q > 0 ? max_err_4q / max_ref_4q : 0;

    FARF(HIGH, "First 8 outputs (ref vs 4Q):");
    for (int m = 0; m < 8; m++) {
        FARF(HIGH, "  y[%d]: ref=%d/1000, 4q=%d/1000, err=%d/1000",
             m, (int)(y_ref[m] * 1000), (int)(y_4q[m] * 1000),
             (int)((y_4q[m] - y_ref[m]) * 1000));
    }
    FARF(HIGH, "4Q max absolute error: %d/10000", (int)(max_err_4q * 10000));
    FARF(HIGH, "4Q max relative error: %d/10000", (int)(rel_err_4q * 10000));
    FARF(HIGH, "4Q mismatches (>0.1): %d / %d", mismatches_4q, GEMV_M);

    uint32_t pass_4q = (rel_err_4q < 0.05f) ? 1 : 0;
    FARF(HIGH, "GEMV Test (4Q): %s (rel_err=%.4f)", pass_4q ? "PASS" : "FAIL", (double)rel_err_4q);

    FARF(HIGH, "=== Timing Summary (M=%d, K=%d) ===", GEMV_M, GEMV_K);
    FARF(HIGH, "  Original:  %llu us", t_end - t_start);
    FARF(HIGH, "  Optimized: %llu us", t_opt_end - t_opt_start);
    FARF(HIGH, "  4Q:        %llu us", t_4q_end - t_4q_start);
    FARF(HIGH, "  Speedup vs original: opt=%.1fx, 4Q=%.1fx",
         (double)(t_end - t_start) / (double)(t_opt_end - t_opt_start),
         (double)(t_end - t_start) / (double)(t_4q_end - t_4q_start));

    free(y_4q);
    free(packed_4q);
    free(y_opt);
    free(W); free(x); free(y_ref); free(y_vlut); free(packed_w);
    return pass & pass_opt & pass_4q & seg_pass;
}

/* ========== Softmax + Attention Tests ========== */

static uint32_t run_softmax_test(void) {
    FARF(HIGH, "=== Softmax Test ===");

    /* Input: [1.0, 2.0, 3.0, 4.0] padded to 32 for HVX alignment */
    float __attribute__((aligned(128))) input[32];
    float __attribute__((aligned(128))) output[32];
    memset(input, 0, sizeof(input));
    memset(output, 0, sizeof(output));
    input[0] = 1.0f;
    input[1] = 2.0f;
    input[2] = 3.0f;
    input[3] = 4.0f;

    hvx_softmax_f32(input, output, 4);

    /* Expected: [0.0321, 0.0871, 0.2369, 0.6439] */
    float expected[4] = { 0.0321f, 0.0871f, 0.2369f, 0.6439f };
    float tolerance = 0.002f;

    uint32_t pass = 1;
    for (int i = 0; i < 4; i++) {
        float err = output[i] - expected[i];
        if (err < 0) err = -err;
        int ok = (err < tolerance);
        FARF(HIGH, "  softmax[%d]: got=%d/10000, expected=%d/10000 %s",
             i, (int)(output[i] * 10000), (int)(expected[i] * 10000),
             ok ? "PASS" : "FAIL");
        if (!ok) pass = 0;
    }

    /* Verify sum ~= 1.0 */
    float sum = 0;
    for (int i = 0; i < 4; i++) sum += output[i];
    FARF(HIGH, "  softmax sum=%d/10000 (expect 10000)", (int)(sum * 10000));
    if (sum < 0.999f || sum > 1.001f) pass = 0;

    FARF(HIGH, "Softmax Test: %s", pass ? "PASS" : "FAIL");
    return pass;
}

static uint32_t run_attention_test(void) {
    FARF(HIGH, "=== Attention Test (minimal) ===");

    /* seq_len=2, head_dim=128 (padded from logical 4 to meet HVX requirements).
     * We put meaningful data in first 4 dims, rest zero. */
    #define ATTN_TEST_DIM 128
    #define ATTN_TEST_SEQ 2

    float *q = (float *)memalign(128, ATTN_TEST_DIM * sizeof(float));
    float *k_cache = (float *)memalign(128, ATTN_TEST_SEQ * ATTN_TEST_DIM * sizeof(float));
    float *v_cache = (float *)memalign(128, ATTN_TEST_SEQ * ATTN_TEST_DIM * sizeof(float));
    float *out = (float *)memalign(128, ATTN_TEST_DIM * sizeof(float));

    if (!q || !k_cache || !v_cache || !out) {
        FARF(ERROR, "Attention test: malloc failed");
        free(q); free(k_cache); free(v_cache); free(out);
        return 0;
    }

    memset(q, 0, ATTN_TEST_DIM * sizeof(float));
    memset(k_cache, 0, ATTN_TEST_SEQ * ATTN_TEST_DIM * sizeof(float));
    memset(v_cache, 0, ATTN_TEST_SEQ * ATTN_TEST_DIM * sizeof(float));
    memset(out, 0, ATTN_TEST_DIM * sizeof(float));

    /* q = [1,0,0,0, 0,...,0] */
    q[0] = 1.0f;

    /* k_cache[0] = [1,0,0,0, 0,...,0] */
    k_cache[0] = 1.0f;
    /* k_cache[1] = [0,1,0,0, 0,...,0] */
    k_cache[ATTN_TEST_DIM + 1] = 1.0f;

    /* v_cache[0] = [10,20,30,40, 0,...,0] */
    v_cache[0] = 10.0f; v_cache[1] = 20.0f;
    v_cache[2] = 30.0f; v_cache[3] = 40.0f;
    /* v_cache[1] = [50,60,70,80, 0,...,0] */
    v_cache[ATTN_TEST_DIM + 0] = 50.0f; v_cache[ATTN_TEST_DIM + 1] = 60.0f;
    v_cache[ATTN_TEST_DIM + 2] = 70.0f; v_cache[ATTN_TEST_DIM + 3] = 80.0f;

    /* scale = 1.0 (not 1/sqrt(128) -- keep simple for verification) */
    float scale = 1.0f;

    hvx_attention_decode_f32(q, k_cache, v_cache, out,
                             ATTN_TEST_DIM, ATTN_TEST_SEQ, scale);

    /* Expected:
     * score[0] = dot(q, k0) * 1.0 = 1.0
     * score[1] = dot(q, k1) * 1.0 = 0.0
     * attn = softmax([1.0, 0.0]) = [0.7311, 0.2689]
     * out = 0.7311*[10,20,30,40] + 0.2689*[50,60,70,80]
     *     = [20.76, 30.76, 40.76, 50.76] */
    float attn0 = expf(1.0f) / (expf(1.0f) + expf(0.0f));
    float attn1 = expf(0.0f) / (expf(1.0f) + expf(0.0f));
    float expected[4];
    expected[0] = attn0 * 10.0f + attn1 * 50.0f;
    expected[1] = attn0 * 20.0f + attn1 * 60.0f;
    expected[2] = attn0 * 30.0f + attn1 * 70.0f;
    expected[3] = attn0 * 40.0f + attn1 * 80.0f;

    uint32_t pass = 1;
    float tolerance = 0.1f;
    for (int i = 0; i < 4; i++) {
        float err = out[i] - expected[i];
        if (err < 0) err = -err;
        int ok = (err < tolerance);
        FARF(HIGH, "  attn_out[%d]: got=%d/100, expected=%d/100 %s",
             i, (int)(out[i] * 100), (int)(expected[i] * 100),
             ok ? "PASS" : "FAIL");
        if (!ok) pass = 0;
    }

    /* Rest of output should be ~0 */
    float rest_max = 0;
    for (int i = 4; i < ATTN_TEST_DIM; i++) {
        float a = out[i] > 0 ? out[i] : -out[i];
        if (a > rest_max) rest_max = a;
    }
    FARF(HIGH, "  rest max abs=%d/10000 (expect ~0)", (int)(rest_max * 10000));
    if (rest_max > 0.01f) {
        FARF(HIGH, "  WARNING: non-zero values in unused dimensions");
        pass = 0;
    }

    FARF(HIGH, "Attention Test: %s", pass ? "PASS" : "FAIL");

    free(q); free(k_cache); free(v_cache); free(out);
    return pass;
}

static uint32_t run_mha_gqa_test(void) {
    FARF(HIGH, "=== MHA GQA Test (2 heads, 1 kv head) ===");

    /* Minimal GQA test: num_heads=2, num_kv_heads=1, head_dim=128, seq_len=1 */
    #define MHA_HEADS     2
    #define MHA_KV_HEADS  1
    #define MHA_DIM       128
    #define MHA_SEQ       1

    float *q_all  = (float *)memalign(128, MHA_HEADS * MHA_DIM * sizeof(float));
    float *k_cache = (float *)memalign(128, MHA_KV_HEADS * MHA_SEQ * MHA_DIM * sizeof(float));
    float *v_cache = (float *)memalign(128, MHA_KV_HEADS * MHA_SEQ * MHA_DIM * sizeof(float));
    float *out     = (float *)memalign(128, MHA_HEADS * MHA_DIM * sizeof(float));

    if (!q_all || !k_cache || !v_cache || !out) {
        FARF(ERROR, "MHA GQA test: malloc failed");
        free(q_all); free(k_cache); free(v_cache); free(out);
        return 0;
    }

    memset(q_all, 0, MHA_HEADS * MHA_DIM * sizeof(float));
    memset(k_cache, 0, MHA_KV_HEADS * MHA_SEQ * MHA_DIM * sizeof(float));
    memset(v_cache, 0, MHA_KV_HEADS * MHA_SEQ * MHA_DIM * sizeof(float));
    memset(out, 0, MHA_HEADS * MHA_DIM * sizeof(float));

    /* q head 0 = [1,0,...], q head 1 = [0,1,0,...] */
    q_all[0] = 1.0f;
    q_all[MHA_DIM + 1] = 1.0f;

    /* single kv: k = [1,1,0,...], v = [5,10,15,20, 0,...] */
    k_cache[0] = 1.0f; k_cache[1] = 1.0f;
    v_cache[0] = 5.0f; v_cache[1] = 10.0f;
    v_cache[2] = 15.0f; v_cache[3] = 20.0f;

    float scale = 1.0f;
    hvx_mha_decode_f32(q_all, k_cache, v_cache, out,
                       MHA_HEADS, MHA_KV_HEADS, MHA_DIM, MHA_SEQ, scale);

    /* With seq_len=1, softmax([score]) = [1.0], so out = v_cache.
     * Both heads should output the same v_cache values. */
    uint32_t pass = 1;
    float tolerance = 0.1f;
    for (int h = 0; h < MHA_HEADS; h++) {
        for (int d = 0; d < 4; d++) {
            float got = out[h * MHA_DIM + d];
            float exp_val = v_cache[d];
            float err = got - exp_val;
            if (err < 0) err = -err;
            int ok = (err < tolerance);
            FARF(HIGH, "  head%d out[%d]: got=%d/100, expected=%d/100 %s",
                 h, d, (int)(got * 100), (int)(exp_val * 100),
                 ok ? "PASS" : "FAIL");
            if (!ok) pass = 0;
        }
    }

    FARF(HIGH, "MHA GQA Test: %s", pass ? "PASS" : "FAIL");

    free(q_all); free(k_cache); free(v_cache); free(out);
    return pass;
}

static void run_attn_test_dispatch(dspqueue_t queue) {
    uint32_t p1 = run_softmax_test();
    uint32_t p2 = run_attention_test();
    uint32_t p3 = run_mha_gqa_test();

    uint32_t pass = p1 & p2 & p3;
    FARF(HIGH, "Attn tests overall: softmax=%u attn=%u mha=%u => %s",
         p1, p2, p3, pass ? "ALL PASS" : "SOME FAILED");

    struct vlut16_test_rsp rsp;
    rsp.op = OP_ATTN_TEST;
    rsp.status = 0;
    rsp.pass = pass;

    dspqueue_write(queue, 0, 0, NULL,
                   sizeof(rsp), (const uint8_t *)&rsp,
                   DSPQUEUE_TIMEOUT_NONE);
}

/* ========== F32 Operator Tests ========== */

static uint32_t run_rmsnorm_test(void) {
    FARF(HIGH, "=== RMSNorm Test ===");
    #define NORM_N 64  /* 2 HVX vectors */
    float __attribute__((aligned(128))) x[NORM_N];
    float __attribute__((aligned(128))) w[NORM_N];
    float __attribute__((aligned(128))) out[NORM_N];
    float __attribute__((aligned(128))) ref[NORM_N];

    /* x = [1, 2, 3, ..., 64], weight = all 1.0 */
    for (int i = 0; i < NORM_N; i++) {
        x[i] = (float)(i + 1);
        w[i] = 1.0f;
    }

    /* Scalar reference: rms = sqrt(mean(x^2) + eps) */
    float sum_sq = 0.0f;
    for (int i = 0; i < NORM_N; i++) sum_sq += x[i] * x[i];
    float rms = sqrtf(sum_sq / (float)NORM_N + 1e-5f);
    for (int i = 0; i < NORM_N; i++) ref[i] = x[i] / rms;

    hvx_rmsnorm_f32(x, w, out, NORM_N, 1e-5f);

    uint32_t pass = 1;
    float max_err = 0;
    for (int i = 0; i < NORM_N; i++) {
        float err = out[i] - ref[i];
        if (err < 0) err = -err;
        if (err > max_err) max_err = err;
    }
    FARF(HIGH, "  RMSNorm max_err=%d/100000 (64 elements, w=1)", (int)(max_err * 100000));
    /* qf32 has ~1e-3 relative error, so allow 0.01 absolute for values up to ~2.5 */
    if (max_err > 0.02f) {
        FARF(HIGH, "  RMSNorm FAIL: error too large");
        pass = 0;
    }

    /* Also test with non-trivial weights */
    for (int i = 0; i < NORM_N; i++) w[i] = 2.0f;
    for (int i = 0; i < NORM_N; i++) ref[i] = x[i] / rms * 2.0f;
    hvx_rmsnorm_f32(x, w, out, NORM_N, 1e-5f);
    float max_err2 = 0;
    for (int i = 0; i < NORM_N; i++) {
        float err = out[i] - ref[i];
        if (err < 0) err = -err;
        if (err > max_err2) max_err2 = err;
    }
    FARF(HIGH, "  RMSNorm(w=2) max_err=%d/100000", (int)(max_err2 * 100000));
    if (max_err2 > 0.05f) {
        FARF(HIGH, "  RMSNorm(w=2) FAIL");
        pass = 0;
    }

    FARF(HIGH, "RMSNorm Test: %s", pass ? "PASS" : "FAIL");
    return pass;
}

static uint32_t run_relu2_test(void) {
    FARF(HIGH, "=== ReLU2 Test ===");
    #define RELU_N 32
    float __attribute__((aligned(128))) x[RELU_N];
    float __attribute__((aligned(128))) out[RELU_N];

    /* Mix of positive, negative, and zero */
    for (int i = 0; i < RELU_N; i++) {
        x[i] = (float)(i - 16);  /* -16, -15, ..., -1, 0, 1, ..., 15 */
    }

    hvx_relu2_f32(x, out, RELU_N);

    uint32_t pass = 1;
    for (int i = 0; i < RELU_N; i++) {
        float v = x[i] > 0.0f ? x[i] : 0.0f;
        float expected = v * v;
        float err = out[i] - expected;
        if (err < 0) err = -err;
        if (err > 0.01f) {
            FARF(HIGH, "  relu2[%d]: x=%d, got=%d/100, expected=%d/100 FAIL",
                 i, (int)x[i], (int)(out[i] * 100), (int)(expected * 100));
            pass = 0;
        }
    }

    /* Check: negatives should be exactly 0 */
    for (int i = 0; i < 16; i++) {
        if (out[i] != 0.0f) {
            FARF(HIGH, "  relu2[%d]: negative input but out=%d/10000 (not 0!)",
                 i, (int)(out[i] * 10000));
            pass = 0;
        }
    }
    /* Check: zero should be 0 */
    if (out[16] != 0.0f) {
        FARF(HIGH, "  relu2[16]: zero input but out=%d/10000", (int)(out[16] * 10000));
        pass = 0;
    }

    FARF(HIGH, "  relu2 samples: out[0]=%d(exp 0), out[17]=%d(exp 100), out[31]=%d(exp 22500)",
         (int)(out[0] * 100), (int)(out[17] * 100), (int)(out[31] * 100));
    FARF(HIGH, "ReLU2 Test: %s", pass ? "PASS" : "FAIL");
    return pass;
}

static uint32_t run_mul_test(void) {
    FARF(HIGH, "=== Mul Test ===");
    #define MUL_N 32
    float __attribute__((aligned(128))) a[MUL_N];
    float __attribute__((aligned(128))) b[MUL_N];
    float __attribute__((aligned(128))) out[MUL_N];

    for (int i = 0; i < MUL_N; i++) {
        a[i] = (float)(i + 1);       /* 1, 2, ..., 32 */
        b[i] = (float)(i + 1) * 0.1f; /* 0.1, 0.2, ..., 3.2 */
    }

    hvx_mul_f32(a, b, out, MUL_N);

    uint32_t pass = 1;
    float max_err = 0;
    for (int i = 0; i < MUL_N; i++) {
        float expected = a[i] * b[i];
        float err = out[i] - expected;
        if (err < 0) err = -err;
        if (err > max_err) max_err = err;
    }
    FARF(HIGH, "  Mul max_err=%d/100000", (int)(max_err * 100000));
    FARF(HIGH, "  Mul samples: out[0]=%d/1000(exp 100), out[31]=%d/1000(exp 102400)",
         (int)(out[0] * 1000), (int)(out[31] * 1000));
    if (max_err > 0.1f) {
        FARF(HIGH, "  Mul FAIL: error too large");
        pass = 0;
    }
    FARF(HIGH, "Mul Test: %s", pass ? "PASS" : "FAIL");
    return pass;
}

static uint32_t run_add_test(void) {
    FARF(HIGH, "=== Add Test ===");
    #define ADD_N 32
    float __attribute__((aligned(128))) a[ADD_N];
    float __attribute__((aligned(128))) b[ADD_N];
    float __attribute__((aligned(128))) out[ADD_N];

    for (int i = 0; i < ADD_N; i++) {
        a[i] = (float)(i + 1);
        b[i] = (float)(ADD_N - i);  /* 32, 31, ..., 1 */
    }

    hvx_add_f32(a, b, out, ADD_N);

    uint32_t pass = 1;
    float max_err = 0;
    for (int i = 0; i < ADD_N; i++) {
        float expected = a[i] + b[i];  /* always 33 */
        float err = out[i] - expected;
        if (err < 0) err = -err;
        if (err > max_err) max_err = err;
    }
    FARF(HIGH, "  Add max_err=%d/100000", (int)(max_err * 100000));
    FARF(HIGH, "  Add samples: out[0]=%d/100(exp 3300), out[15]=%d/100(exp 3300)",
         (int)(out[0] * 100), (int)(out[15] * 100));
    if (max_err > 0.01f) {
        FARF(HIGH, "  Add FAIL: error too large");
        pass = 0;
    }
    FARF(HIGH, "Add Test: %s", pass ? "PASS" : "FAIL");
    return pass;
}

static uint32_t run_rope_test(void) {
    FARF(HIGH, "=== RoPE Test ===");
    /* head_dim=128, pos=0 and pos=1 */
    #define ROPE_DIM 128
    #define ROPE_HALF 64
    #define ROPE_MAX_POS 4

    float __attribute__((aligned(128))) x[ROPE_DIM];
    float __attribute__((aligned(128))) out[ROPE_DIM];
    float __attribute__((aligned(128))) cos_tbl[ROPE_MAX_POS * ROPE_HALF];
    float __attribute__((aligned(128))) sin_tbl[ROPE_MAX_POS * ROPE_HALF];

    /* Build cos/sin tables: theta_i = 1/(10000^(2i/dim)) for i=0..half-1 */
    for (int pos = 0; pos < ROPE_MAX_POS; pos++) {
        for (int i = 0; i < ROPE_HALF; i++) {
            float theta = (float)pos / powf(10000.0f, 2.0f * (float)i / (float)ROPE_DIM);
            cos_tbl[pos * ROPE_HALF + i] = cosf(theta);
            sin_tbl[pos * ROPE_HALF + i] = sinf(theta);
        }
    }

    /* Test input: x_r = [1,2,...,64], x_i = [65,66,...,128] */
    for (int i = 0; i < ROPE_DIM; i++) {
        x[i] = (float)(i + 1);
    }

    uint32_t pass = 1;

    /* Test pos=0: cos=1, sin=0 for all dims, so out should equal x */
    hvx_rope_f32(x, cos_tbl, sin_tbl, out, ROPE_DIM, 0);
    float max_err_p0 = 0;
    for (int i = 0; i < ROPE_DIM; i++) {
        float err = out[i] - x[i];
        if (err < 0) err = -err;
        if (err > max_err_p0) max_err_p0 = err;
    }
    FARF(HIGH, "  RoPE pos=0 max_err=%d/100000 (expect ~0, identity rotation)",
         (int)(max_err_p0 * 100000));
    if (max_err_p0 > 0.01f) {
        FARF(HIGH, "  RoPE pos=0 FAIL");
        pass = 0;
    }

    /* Test pos=1: verify against scalar reference */
    hvx_rope_f32(x, cos_tbl, sin_tbl, out, ROPE_DIM, 1);
    float max_err_p1 = 0;
    for (int i = 0; i < ROPE_HALF; i++) {
        float c = cos_tbl[1 * ROPE_HALF + i];
        float s = sin_tbl[1 * ROPE_HALF + i];
        float xr = x[i];
        float xi = x[ROPE_HALF + i];
        float ref_r = xr * c - xi * s;
        float ref_i = xr * s + xi * c;
        float err_r = out[i] - ref_r;
        float err_i = out[ROPE_HALF + i] - ref_i;
        if (err_r < 0) err_r = -err_r;
        if (err_i < 0) err_i = -err_i;
        if (err_r > max_err_p1) max_err_p1 = err_r;
        if (err_i > max_err_p1) max_err_p1 = err_i;
    }
    FARF(HIGH, "  RoPE pos=1 max_err=%d/100000", (int)(max_err_p1 * 100000));
    /* Sample values for inspection */
    FARF(HIGH, "  RoPE pos=1 out[0]=%d/100, out[64]=%d/100",
         (int)(out[0] * 100), (int)(out[64] * 100));
    if (max_err_p1 > 0.5f) {
        FARF(HIGH, "  RoPE pos=1 FAIL: error too large");
        pass = 0;
    }

    FARF(HIGH, "RoPE Test: %s", pass ? "PASS" : "FAIL");
    return pass;
}

static uint32_t run_ops_test(void) {
    FARF(HIGH, "========== F32 Operator Tests ==========");
    uint32_t p1 = run_rmsnorm_test();
    uint32_t p2 = run_relu2_test();
    uint32_t p3 = run_mul_test();
    uint32_t p4 = run_add_test();
    uint32_t p5 = run_rope_test();

    uint32_t pass = p1 & p2 & p3 & p4 & p5;
    FARF(HIGH, "Ops tests: norm=%u relu2=%u mul=%u add=%u rope=%u => %s",
         p1, p2, p3, p4, p5, pass ? "ALL PASS" : "SOME FAILED");
    return pass;
}

static void run_ops_test_dispatch(dspqueue_t queue) {
    uint32_t pass = run_ops_test();

    struct vlut16_test_rsp rsp;
    rsp.op = OP_OPS_TEST;
    rsp.status = 0;
    rsp.pass = pass;

    dspqueue_write(queue, 0, 0, NULL,
                   sizeof(rsp), (const uint8_t *)&rsp,
                   DSPQUEUE_TIMEOUT_NONE);
}

/* ========== Decoder Layer Test ========== */

/*
 * Run decoder layer test using weights from shared memory buffer.
 * The buffer starts with a WeightLayout header followed by all weight data.
 *
 * bufs[0].ptr points to the shared buffer.
 */
static void run_decoder_test(dspqueue_t queue, struct dspqueue_buffer *bufs, uint32_t n_bufs) {
    struct vlut16_test_rsp rsp;
    rsp.op = OP_DECODER_TEST;
    rsp.status = 0;
    rsp.pass = 0;

    if (n_bufs < 1 || !bufs[0].ptr) {
        FARF(ERROR, "Decoder test: no shared buffer received");
        rsp.status = 1;
        goto respond;
    }

    {
        uint8_t *base = (uint8_t *)bufs[0].ptr;
        const WeightLayout *layout = (const WeightLayout *)base;

        FARF(HIGH, "=== Decoder Layer Test ===");
        FARF(HIGH, "Buffer size: %u bytes, pos=%u, kv_seq_len=%u, kv_max_seq=%u",
             layout->total_size, layout->pos, layout->kv_seq_len, layout->kv_max_seq_len);

        /* Parse weight pointers from offsets */
        DecoderLayerWeights weights;
        weights.q_proj_w        = (const uint8_t *)(base + layout->q_proj_offset);
        weights.k_proj_w        = (const uint8_t *)(base + layout->k_proj_offset);
        weights.v_proj_w        = (const uint8_t *)(base + layout->v_proj_offset);
        weights.o_proj_w        = (const uint8_t *)(base + layout->o_proj_offset);
        weights.gate_proj_w     = (const uint8_t *)(base + layout->gate_proj_offset);
        weights.up_proj_w       = (const uint8_t *)(base + layout->up_proj_offset);
        weights.down_proj_w     = (const uint8_t *)(base + layout->down_proj_offset);
        weights.input_ln_w      = (const float *)(base + layout->input_ln_offset);
        weights.post_attn_ln_w  = (const float *)(base + layout->post_attn_ln_offset);
        weights.attn_sub_norm_w = (const float *)(base + layout->attn_sub_norm_offset);
        weights.ffn_sub_norm_w  = (const float *)(base + layout->ffn_sub_norm_offset);
        weights.rope_cos        = (const float *)(base + layout->rope_cos_offset);
        weights.rope_sin        = (const float *)(base + layout->rope_sin_offset);

        /* Copy input (we will modify it in-place) */
        const float *input_src = (const float *)(base + layout->input_offset);
        const float *ref_output = (const float *)(base + layout->ref_output_offset);

        float *x = (float *)memalign(128, HIDDEN_SIZE * sizeof(float));
        if (!x) {
            FARF(ERROR, "Decoder test: malloc failed for input");
            rsp.status = 2;
            goto respond;
        }
        memcpy(x, input_src, HIDDEN_SIZE * sizeof(float));

        /* Set up KV cache -- points into the shared buffer (writable) */
        KVCache kv;
        kv.k_cache     = (float *)(base + layout->k_cache_offset);
        kv.v_cache     = (float *)(base + layout->v_cache_offset);
        kv.max_seq_len = (int)layout->kv_max_seq_len;
        kv.seq_len     = (int)layout->kv_seq_len;

        int pos = (int)layout->pos;

        FARF(HIGH, "Input[0..3]: %d %d %d %d (x1000)",
             (int)(x[0] * 1000), (int)(x[1] * 1000),
             (int)(x[2] * 1000), (int)(x[3] * 1000));

        /* Run decoder layer */
        uint64_t t_start = HAP_perf_get_time_us();
        bitnet_decoder_layer(x, pos, &weights, &kv);
        uint64_t t_end = HAP_perf_get_time_us();

        FARF(HIGH, "Decoder layer time: %llu us", t_end - t_start);

        /* Compare output with reference */
        float max_abs_err = 0;
        float max_abs_ref = 0;
        int mismatch_count = 0;

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            float err = x[i] - ref_output[i];
            if (err < 0) err = -err;
            if (err > max_abs_err) max_abs_err = err;

            float ra = ref_output[i] > 0 ? ref_output[i] : -ref_output[i];
            if (ra > max_abs_ref) max_abs_ref = ra;

            if (err > 1.0f) mismatch_count++;
        }

        float rel_err = max_abs_ref > 0 ? max_abs_err / max_abs_ref : 0;

        FARF(HIGH, "Output[0..3]:    %d %d %d %d (x1000)",
             (int)(x[0] * 1000), (int)(x[1] * 1000),
             (int)(x[2] * 1000), (int)(x[3] * 1000));
        FARF(HIGH, "Reference[0..3]: %d %d %d %d (x1000)",
             (int)(ref_output[0] * 1000), (int)(ref_output[1] * 1000),
             (int)(ref_output[2] * 1000), (int)(ref_output[3] * 1000));
        FARF(HIGH, "Max abs error: %d/10000, max ref: %d/10000",
             (int)(max_abs_err * 10000), (int)(max_abs_ref * 10000));
        FARF(HIGH, "Relative error: %d/10000", (int)(rel_err * 10000));
        FARF(HIGH, "Mismatches (>1.0): %d / %d", mismatch_count, HIDDEN_SIZE);

        /* Pass criterion: relative error < 10%
         * BitNet VLUT16 int16 quantization introduces ~1-5% error,
         * and qf32 adds another ~0.1% per op. With 7 GEMV + multiple
         * norms in a decoder layer, 10% total is reasonable. */
        uint32_t pass = (rel_err < 0.10f && mismatch_count == 0) ? 1 : 0;
        FARF(HIGH, "Decoder Layer Test: %s (rel_err=%.4f, time=%llu us)",
             pass ? "PASS" : "FAIL", (double)rel_err, t_end - t_start);

        rsp.pass = pass;
        free(x);
    }

respond:
    {
        /* Deref the shared buffer */
        struct dspqueue_buffer rsp_bufs[1];
        memset(rsp_bufs, 0, sizeof(rsp_bufs));
        int n_rsp_bufs = 0;
        if (n_bufs >= 1) {
            rsp_bufs[0].fd = bufs[0].fd;
            rsp_bufs[0].flags = DSPQUEUE_BUFFER_FLAG_DEREF;
            n_rsp_bufs = 1;
        }

        dspqueue_write(queue, 0, n_rsp_bufs, rsp_bufs,
                       sizeof(rsp), (const uint8_t *)&rsp,
                       DSPQUEUE_TIMEOUT_NONE);
    }
}

/* ========== Dispatch ========== */

static void run_gemv_test_dispatch(dspqueue_t queue) {
    uint32_t pass = run_gemv_test();

    struct vlut16_test_rsp rsp;
    rsp.op = OP_GEMV_TEST;
    rsp.status = 0;
    rsp.pass = pass;

    dspqueue_write(queue, 0, 0, NULL,
                   sizeof(rsp), (const uint8_t *)&rsp,
                   DSPQUEUE_TIMEOUT_NONE);
}

static void run_vlut16_test(dspqueue_t queue, uint32_t test_id) {
    FARF(HIGH, "Running VLUT16 tests...");

    uint32_t p1 = run_vlut16_test1();
    uint32_t p2 = run_vlut16_test2();
    uint32_t p3 = run_vlut16_test3();

    uint32_t pass = p1 & p2 & p3;
    FARF(HIGH, "Overall: test1=%u test2=%u test3=%u => %s",
         p1, p2, p3, pass ? "ALL PASS" : "SOME FAILED");

    struct vlut16_test_rsp rsp;
    rsp.op = OP_VLUT16_TEST;
    rsp.status = 0;
    rsp.pass = pass;

    dspqueue_write(queue, 0, 0, NULL,
                   sizeof(rsp), (const uint8_t *)&rsp,
                   DSPQUEUE_TIMEOUT_NONE);
}

/* ========== dspqueue callback ========== */

static void error_callback(dspqueue_t queue, int error, void *context) {
    FARF(ERROR, "dspqueue error: 0x%08x", (unsigned)error);
}

static void packet_callback(dspqueue_t queue, int error, void *context) {
    struct dsp_ctx *ctx = (struct dsp_ctx *)context;

    while (1) {
        /* Use a union to handle all possible message types */
        union {
            struct vlut16_test_req  vlut;
            struct decoder_test_req decoder;
        } msg;
        uint32_t flags, msg_len, n_bufs;
        struct dspqueue_buffer bufs[MAX_BUFFERS];

        int err = dspqueue_read_noblock(queue, &flags,
                                        MAX_BUFFERS, &n_bufs, bufs,
                                        sizeof(msg), &msg_len, (uint8_t *)&msg);
        if (err == AEE_EWOULDBLOCK) return;
        if (err != 0) {
            FARF(ERROR, "dspqueue_read failed: 0x%08x", (unsigned)err);
            return;
        }

        uint32_t op = msg.vlut.op;  /* op is always the first field */
        uint64_t t1 = HAP_perf_get_time_us();

        if (op == OP_VLUT16_TEST) {
            run_vlut16_test(queue, msg.vlut.test_id);
        } else if (op == OP_GEMV_TEST) {
            run_gemv_test_dispatch(queue);
        } else if (op == OP_ATTN_TEST) {
            run_attn_test_dispatch(queue);
        } else if (op == OP_OPS_TEST) {
            run_ops_test_dispatch(queue);
        } else if (op == OP_DECODER_TEST) {
            run_decoder_test(queue, bufs, n_bufs);
        } else if (op == OP_EXIT) {
            struct vlut16_test_rsp rsp = { OP_EXIT, 0, 0 };
            dspqueue_write(queue, 0, 0, NULL,
                           sizeof(rsp), (const uint8_t *)&rsp,
                           DSPQUEUE_TIMEOUT_NONE);
        } else {
            FARF(ERROR, "Unknown op %u", op);
            struct vlut16_test_rsp rsp = { op, (uint32_t)AEE_EBADPARM, 0 };
            dspqueue_write(queue, 0, 0, NULL,
                           sizeof(rsp), (const uint8_t *)&rsp,
                           DSPQUEUE_TIMEOUT_NONE);
        }

        uint64_t t2 = HAP_perf_get_time_us();
        ctx->total_us += (t2 - t1);
    }
}

/* ========== FastRPC lifecycle ========== */

AEEResult bitnet_test_open(const char *uri, remote_handle64 *handle) {
    struct dsp_ctx *ctx = (struct dsp_ctx *)calloc(1, sizeof(struct dsp_ctx));
    if (!ctx) return AEE_ENOMEMORY;
    *handle = (remote_handle64)ctx;

    /* Power config: lock to turbo, disable DCVS */
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_apptype;
    request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    HAP_power_set(ctx, &request);

    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_DCVS_v2;
    request.dcvs_v2.dcvs_enable = FALSE;
    request.dcvs_v2.set_dcvs_params = TRUE;
    request.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_DISABLE;
    request.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_DISABLE;
    request.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_TURBO;
    request.dcvs_v2.set_latency = TRUE;
    request.dcvs_v2.latency = 40;
    int pwr_err = HAP_power_set(ctx, &request);
    if (pwr_err != AEE_SUCCESS) {
        FARF(ERROR, "HAP_power_set failed: 0x%08x", (unsigned)pwr_err);
        free(ctx);
        return pwr_err;
    }

    FARF(HIGH, "bitnet_test_open: context allocated, turbo mode set");
    return AEE_SUCCESS;
}

AEEResult bitnet_test_close(remote_handle64 handle) {
    struct dsp_ctx *ctx = (struct dsp_ctx *)handle;
    if (!ctx) return AEE_EBADPARM;
    if (ctx->queue) return AEE_EITEMBUSY;
    free(ctx);
    return AEE_SUCCESS;
}

AEEResult bitnet_test_start(remote_handle64 handle, uint64 dsp_queue_id) {
    struct dsp_ctx *ctx = (struct dsp_ctx *)handle;
    if (!ctx) return AEE_EBADPARM;
    if (ctx->queue) return AEE_EITEMBUSY;
    ctx->total_us = 0;

    int err = dspqueue_import(dsp_queue_id, packet_callback, error_callback,
                              ctx, &ctx->queue);
    if (err) {
        FARF(ERROR, "dspqueue_import failed: 0x%08x", (unsigned)err);
        return err;
    }

    FARF(HIGH, "bitnet_test_start: dspqueue imported");
    return AEE_SUCCESS;
}

AEEResult bitnet_test_stop(remote_handle64 handle, uint64 *process_time) {
    struct dsp_ctx *ctx = (struct dsp_ctx *)handle;
    if (!ctx) return AEE_EBADPARM;
    if (!ctx->queue) return AEE_EBADSTATE;

    int err = dspqueue_close(ctx->queue);
    ctx->queue = NULL;
    if (err) return err;

    FARF(HIGH, "bitnet_test_stop: total DSP time = %llu us", ctx->total_us);
    *process_time = ctx->total_us;
    return AEE_SUCCESS;
}
