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

/* ========== DSP context ========== */

struct dsp_ctx {
    dspqueue_t queue;
    uint64_t   total_us;
};

/* ========== VLUT16 LUT shuffle ========== */

/*
 * Shuffle 16 LUT vectors into the interleaved format that VLUT16 expects.
 *
 * Input:  l_tmp[0..15] — 16 HVX vectors. l_tmp[i] has entry i's value
 *         at each of the 64 int16 positions.
 * Output: out[0..15] — 16 shuffled HVX vectors. Each can be passed
 *         directly to Q6_Wh_vlut16_VbVhR_nomatch as the LUT vector.
 *
 * The shuffle is a 4-step butterfly that interleaves entries at
 * 32-bit, 64-bit, 128-bit, and 256-bit granularity. This matches
 * the hardware's expected layout for 128B HVX mode.
 *
 * Reference: executorch TMANOpPackage/include/hvx_funcs.h, hvx_lut_ctor()
 */
static void vlut16_shuffle(HVX_Vector l_tmp[16], HVX_Vector out[16]) {
    HVX_VectorPair l_pa[8];
    HVX_VectorPair l_pb[8];

    /* Step 1: interleave at 32-bit (4 byte) granularity */
    for (int i = 0; i < 16; i += 2) {
        l_pa[i / 2] = Q6_W_vshuff_VVR(l_tmp[i + 1], l_tmp[i], -4);
    }

    /* Step 2: interleave at 64-bit (8 byte) granularity */
    for (int i = 0; i < 8; i += 2) {
        l_pb[i + 0] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pa[i + 1]), Q6_V_lo_W(l_pa[i + 0]), -8);
        l_pb[i + 1] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pa[i + 1]), Q6_V_hi_W(l_pa[i + 0]), -8);
    }

    /* Step 3: interleave at 128-bit (16 byte) granularity */
    for (int i = 0; i < 8; i += 4) {
        l_pa[i + 0] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pb[i + 2]), Q6_V_lo_W(l_pb[i + 0]), -16);
        l_pa[i + 1] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pb[i + 2]), Q6_V_hi_W(l_pb[i + 0]), -16);
        l_pa[i + 2] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pb[i + 3]), Q6_V_lo_W(l_pb[i + 1]), -16);
        l_pa[i + 3] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pb[i + 3]), Q6_V_hi_W(l_pb[i + 1]), -16);
    }

    /* Step 4: interleave at 256-bit (32 byte) granularity */
    for (int i = 0; i < 8; i += 8) {
        l_pb[i + 0] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pa[i + 4]), Q6_V_lo_W(l_pa[i + 0]), -32);
        l_pb[i + 1] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pa[i + 4]), Q6_V_hi_W(l_pa[i + 0]), -32);
        l_pb[i + 2] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pa[i + 5]), Q6_V_lo_W(l_pa[i + 1]), -32);
        l_pb[i + 3] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pa[i + 5]), Q6_V_hi_W(l_pa[i + 1]), -32);
        l_pb[i + 4] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pa[i + 6]), Q6_V_lo_W(l_pa[i + 2]), -32);
        l_pb[i + 5] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pa[i + 6]), Q6_V_hi_W(l_pa[i + 2]), -32);
        l_pb[i + 6] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pa[i + 7]), Q6_V_lo_W(l_pa[i + 3]), -32);
        l_pb[i + 7] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pa[i + 7]), Q6_V_hi_W(l_pa[i + 3]), -32);
    }

    /* Extract: each pair's lo and hi are consecutive output vectors */
    for (int i = 0; i < 8; i++) {
        out[i * 2]     = Q6_V_lo_W(l_pb[i]);
        out[i * 2 + 1] = Q6_V_hi_W(l_pb[i]);
    }
}

/*
 * Build a shuffled VLUT16 LUT vector from 16 int16 values.
 * All 64 output positions share the same 16-entry table.
 * Returns one shuffled HVX vector ready for VLUT16.
 */
static HVX_Vector vlut16_build_lut(const int16_t values[16]) {
    HVX_Vector l_tmp[16];
    for (int i = 0; i < 16; i++) {
        l_tmp[i] = Q6_Vh_vsplat_R(values[i]);
    }

    HVX_Vector shuffled[16];
    vlut16_shuffle(l_tmp, shuffled);

    /* All 16 output vectors are identical when input is splatted,
     * so just return the first one. */
    return shuffled[0];
}

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

/* ========== Dispatch ========== */

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
        struct vlut16_test_req msg;
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

        uint64_t t1 = HAP_perf_get_time_us();

        if (msg.op == OP_VLUT16_TEST) {
            run_vlut16_test(queue, msg.test_id);
        } else if (msg.op == OP_EXIT) {
            struct vlut16_test_rsp rsp = { OP_EXIT, 0, 0 };
            dspqueue_write(queue, 0, 0, NULL,
                           sizeof(rsp), (const uint8_t *)&rsp,
                           DSPQUEUE_TIMEOUT_NONE);
        } else {
            FARF(ERROR, "Unknown op %u", msg.op);
            struct vlut16_test_rsp rsp = { msg.op, (uint32_t)AEE_EBADPARM, 0 };
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
