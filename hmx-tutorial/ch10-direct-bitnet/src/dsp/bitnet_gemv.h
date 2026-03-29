/*
 * bitnet_gemv.h -- BitNet GEMV via VLUT16
 *
 * Computes y[M] = W[M,K] @ x[K] where W is ternary {-1, 0, +1}.
 *
 * Three kernels, progressively optimized:
 *   1. bitnet_gemv()     -- correct-first: per-Q scale, float accum
 *   2. bitnet_gemv_opt() -- global scale + int32 vector accum
 *   3. bitnet_gemv_4q()  -- 4Q packing + multi-segment VLUT16
 *
 * Common phases:
 *   1. LUT construction: for each group of 4 activations, precompute
 *      16 possible +/-x sums, quantize to int16, shuffle for VLUT16
 *   2. VLUT16 lookup + accumulation: use packed weight nibbles as
 *      indices, accumulate int16 results into int32
 *   3. Bit-serial combination: output = 0.5*partial_0 + partial_1 + lb
 *
 * Encoding: ternary w in {-1,0,+1} -> enc = w + 2 in {1,2,3}
 *   bit0 = enc & 1, bit1 = (enc >> 1) & 1
 *
 * LUT[index] = sum_j ((index>>j)&1 ? +x_j : -x_j)  for j=0..3
 *
 * Formula: y = 0.5 * partial_bit0 + partial_bit1 - 0.5 * sum(x)
 *
 * Weight layouts:
 *   Original: packed_w[bit_plane][q][m] = 1 byte per nibble (wasteful)
 *   4Q:       packed_4q[bit_plane][q_group_of_4][vec_lo/hi][m] = 2 nibbles/byte
 *
 * VLUT16 segment mapping (v75, verified empirically):
 *   seg 0 -> interleave position 0
 *   seg 1 -> interleave position 2 (swapped with seg 2!)
 *   seg 2 -> interleave position 1
 *   seg 3 -> interleave position 3
 */

#ifndef BITNET_GEMV_H
#define BITNET_GEMV_H

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

/* ========== VLUT16 shuffle (from experiment 1) ========== */

static inline void vlut16_shuffle(HVX_Vector l_tmp[16], HVX_Vector out[16]) {
    HVX_VectorPair l_pa[8], l_pb[8];

    for (int i = 0; i < 16; i += 2)
        l_pa[i/2] = Q6_W_vshuff_VVR(l_tmp[i+1], l_tmp[i], -4);

    for (int i = 0; i < 8; i += 2) {
        l_pb[i+0] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pa[i+1]), Q6_V_lo_W(l_pa[i+0]), -8);
        l_pb[i+1] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pa[i+1]), Q6_V_hi_W(l_pa[i+0]), -8);
    }

    for (int i = 0; i < 8; i += 4) {
        l_pa[i+0] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pb[i+2]), Q6_V_lo_W(l_pb[i+0]), -16);
        l_pa[i+1] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pb[i+2]), Q6_V_hi_W(l_pb[i+0]), -16);
        l_pa[i+2] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pb[i+3]), Q6_V_lo_W(l_pb[i+1]), -16);
        l_pa[i+3] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pb[i+3]), Q6_V_hi_W(l_pb[i+1]), -16);
    }

    l_pb[0] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pa[4]), Q6_V_lo_W(l_pa[0]), -32);
    l_pb[1] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pa[4]), Q6_V_hi_W(l_pa[0]), -32);
    l_pb[2] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pa[5]), Q6_V_lo_W(l_pa[1]), -32);
    l_pb[3] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pa[5]), Q6_V_hi_W(l_pa[1]), -32);
    l_pb[4] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pa[6]), Q6_V_lo_W(l_pa[2]), -32);
    l_pb[5] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pa[6]), Q6_V_hi_W(l_pa[2]), -32);
    l_pb[6] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pa[7]), Q6_V_lo_W(l_pa[3]), -32);
    l_pb[7] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pa[7]), Q6_V_hi_W(l_pa[3]), -32);

    for (int i = 0; i < 8; i++) {
        out[i*2]   = Q6_V_lo_W(l_pb[i]);
        out[i*2+1] = Q6_V_hi_W(l_pb[i]);
    }
}

static inline HVX_Vector vlut16_build_lut(const int16_t values[16]) {
    HVX_Vector l_tmp[16];
    for (int i = 0; i < 16; i++)
        l_tmp[i] = Q6_Vh_vsplat_R(values[i]);
    HVX_Vector shuffled[16];
    vlut16_shuffle(l_tmp, shuffled);
    return shuffled[0];
}

/* ========== Weight packing ========== */

/*
 * Pack ternary weights W[M][K] into VLUT16-ready nibble format.
 *
 * Output: packed_w[2 * Q * M] where Q = K/4
 *   Layout: packed_w[b * Q * M + q * M + m] = 4-bit nibble
 *
 * The nibble for bit plane b, Q-position q, output dim m is:
 *   nibble = Σ_{g=0}^{3} bit_b(W[m, q*4+g]) << g
 *
 * where bit_b extracts bit b of the encoded weight (enc = w + 2).
 */
static void bitnet_pack_weights(
    const int8_t *W,       /* [M * K] ternary weights, row-major */
    uint8_t *packed_w,     /* [2 * Q * M] output */
    int M, int K)
{
    int Q = K / 4;
    for (int b = 0; b < 2; b++) {
        for (int q = 0; q < Q; q++) {
            for (int m = 0; m < M; m++) {
                uint8_t nibble = 0;
                for (int g = 0; g < 4; g++) {
                    int8_t w = W[m * K + q * 4 + g];
                    uint8_t enc = (uint8_t)(w + 2);  /* {-1,0,+1} → {1,2,3} */
                    uint8_t bit = (enc >> b) & 1;
                    nibble |= (bit << g);
                }
                packed_w[b * Q * M + q * M + m] = nibble;
            }
        }
    }
}

/* ========== BitNet GEMV kernel ========== */

/*
 * Compute y[M] = W[M,K] @ x[K] using VLUT16.
 *
 * This is a correct-first implementation: one VLUT16 call per
 * (bit_plane, Q_position, output_chunk), with per-Q float accumulation.
 *
 * M must be a multiple of 128 (one VLUT16 pass per chunk).
 * K must be a multiple of 4.
 */
static void bitnet_gemv(
    const float *x,           /* [K] activations */
    const uint8_t *packed_w,  /* [2 * Q * M] packed weights */
    float *y,                 /* [M] output */
    int M, int K)
{
    int Q = K / 4;

    /* Per-bit-plane accumulators (float, interleaved even/odd) */
    float partial_even[2][64];  /* max M=128 → 64 even outputs */
    float partial_odd[2][64];

    /* Process each output chunk of 128 */
    for (int m_base = 0; m_base < M; m_base += 128) {

        for (int b = 0; b < 2; b++) {
            memset(partial_even[b], 0, 64 * sizeof(float));
            memset(partial_odd[b], 0, 64 * sizeof(float));
        }

        for (int b = 0; b < 2; b++) {
            for (int q = 0; q < Q; q++) {
                /* Build LUT for this group of 4 activations */
                float x0 = x[q*4+0], x1 = x[q*4+1];
                float x2 = x[q*4+2], x3 = x[q*4+3];

                float lut_f[16];
                for (int idx = 0; idx < 16; idx++) {
                    float sum = 0;
                    sum += (idx & 1) ? +x0 : -x0;
                    sum += (idx & 2) ? +x1 : -x1;
                    sum += (idx & 4) ? +x2 : -x2;
                    sum += (idx & 8) ? +x3 : -x3;
                    lut_f[idx] = sum;
                }

                /* Find scale for int16 quantization */
                float max_abs = 0;
                for (int i = 0; i < 16; i++) {
                    float a = lut_f[i] > 0 ? lut_f[i] : -lut_f[i];
                    if (a > max_abs) max_abs = a;
                }
                float ls = max_abs > 0 ? max_abs / 32767.0f : 1e-10f;

                /* Quantize to int16 */
                int16_t lut_q[16];
                for (int i = 0; i < 16; i++) {
                    float v = lut_f[i] / ls;
                    lut_q[i] = (int16_t)(v > 0 ? v + 0.5f : v - 0.5f);
                }

                /* Build shuffled VLUT16 vector */
                HVX_Vector lut_vec = vlut16_build_lut(lut_q);

                /* Load weight indices for this chunk */
                const uint8_t *w_ptr = packed_w + (b * Q + q) * M + m_base;
                HVX_Vector idx_vec = *(HVX_Vector *)w_ptr;

                /* VLUT16 lookup */
                HVX_VectorPair result = Q6_Wh_vlut16_VbVhR_nomatch(
                    idx_vec, lut_vec, 0);

                /* Extract int16 results */
                int16_t __attribute__((aligned(128))) out_lo[64];
                int16_t __attribute__((aligned(128))) out_hi[64];
                *(HVX_Vector *)out_lo = Q6_V_lo_W(result);
                *(HVX_Vector *)out_hi = Q6_V_hi_W(result);

                /* Dequantize and accumulate into float */
                for (int j = 0; j < 64; j++) {
                    partial_even[b][j] += (float)out_lo[j] * ls;
                    partial_odd[b][j]  += (float)out_hi[j] * ls;
                }
            }
        }

        /* Compute sum(x) for bias correction */
        float sum_x = 0;
        for (int k = 0; k < K; k++) sum_x += x[k];
        float lb = -0.5f * sum_x;

        /* Combine bit planes: y = 0.5*partial_0 + partial_1 + lb */
        for (int j = 0; j < 64; j++) {
            int m_even = m_base + j * 2;
            int m_odd  = m_base + j * 2 + 1;
            if (m_even < M)
                y[m_even] = 0.5f * partial_even[0][j] + partial_even[1][j] + lb;
            if (m_odd < M)
                y[m_odd]  = 0.5f * partial_odd[0][j]  + partial_odd[1][j]  + lb;
        }
    }
}

/* ========== Optimized BitNet GEMV kernel ========== */

/*
 * Compute y[M] = W[M,K] @ x[K] using VLUT16 with two key optimizations:
 *
 * 1. Single global scale factor — avoids per-Q max_abs and scale computation
 * 2. int32 vector accumulation — replaces per-Q float dequantize loop with
 *    HVX widening accumulate (Q6_Ww_vaddacc_WwVhVh), converting to float
 *    only once at the end of the Q loop
 *
 * Uses the same weight format as bitnet_pack_weights() (1 byte per Q per M).
 * M must be a multiple of 128. K must be a multiple of 4.
 */
static void bitnet_gemv_opt(
    const float *x,           /* [K] activations */
    const uint8_t *packed_w,  /* [2 * Q * M] packed weights (original format) */
    float *y,                 /* [M] output */
    int M, int K)
{
    int Q = K / 4;

    /* Compute global max|x| for a single quantization scale */
    float max_abs_x = 0;
    float sum_x = 0;
    for (int k = 0; k < K; k++) {
        float a = x[k] > 0 ? x[k] : -x[k];
        if (a > max_abs_x) max_abs_x = a;
        sum_x += x[k];
    }
    float lb = -0.5f * sum_x;

    /* Global scale: max possible LUT entry is 4 * max_abs_x.
     * Map to int16 range with some headroom. */
    float ls = max_abs_x > 0 ? (4.0f * max_abs_x) / 32767.0f : 1e-10f;
    float inv_ls = 1.0f / ls;

    /* Process each output chunk of 128 */
    for (int m_base = 0; m_base < M; m_base += 128) {

        float partial_f[2][128];  /* final float partial sums per bit-plane */

        for (int b = 0; b < 2; b++) {
            /* int32 accumulators for even-byte and odd-byte VLUT16 results.
             * Each HVX_VectorPair holds 64 int32 (32 lo + 32 hi). */
            HVX_VectorPair acc_even = Q6_W_vcombine_VV(
                Q6_V_vzero(), Q6_V_vzero());
            HVX_VectorPair acc_odd = Q6_W_vcombine_VV(
                Q6_V_vzero(), Q6_V_vzero());
            int first = 1;

            for (int q = 0; q < Q; q++) {
                /* Build LUT for this group of 4 activations */
                float x0 = x[q*4+0], x1 = x[q*4+1];
                float x2 = x[q*4+2], x3 = x[q*4+3];

                int16_t lut_q[16];
                for (int idx = 0; idx < 16; idx++) {
                    float sum = 0;
                    sum += (idx & 1) ? +x0 : -x0;
                    sum += (idx & 2) ? +x1 : -x1;
                    sum += (idx & 4) ? +x2 : -x2;
                    sum += (idx & 8) ? +x3 : -x3;
                    float v = sum * inv_ls;
                    lut_q[idx] = (int16_t)(v > 0 ? v + 0.5f : v - 0.5f);
                }
                HVX_Vector lut_vec = vlut16_build_lut(lut_q);

                /* Load weight indices for this chunk (original format) */
                const uint8_t *w_ptr = packed_w + (b * Q + q) * M + m_base;
                HVX_Vector idx_vec = *(HVX_Vector *)w_ptr;

                /* VLUT16 lookup */
                HVX_VectorPair result = Q6_Wh_vlut16_VbVhR_nomatch(
                    idx_vec, lut_vec, 0);
                HVX_Vector res_even = Q6_V_lo_W(result);
                HVX_Vector res_odd  = Q6_V_hi_W(result);

                /* Accumulate int16 -> int32 via widening add */
                if (first) {
                    acc_even = Q6_Ww_vadd_VhVh(res_even, Q6_V_vzero());
                    acc_odd  = Q6_Ww_vadd_VhVh(res_odd,  Q6_V_vzero());
                    first = 0;
                } else {
                    acc_even = Q6_Ww_vaddacc_WwVhVh(acc_even, res_even, Q6_V_vzero());
                    acc_odd  = Q6_Ww_vaddacc_WwVhVh(acc_odd,  res_odd,  Q6_V_vzero());
                }
            }

            /* Convert int32 accumulators to float.
             *
             * HVX widening ops deinterleave: Ww_vadd_VhVh(V,zero) produces
             *   lo[j] = int32(V.h[2*j])     for j=0..31 (even-indexed int16)
             *   hi[j] = int32(V.h[2*j+1])   for j=0..31 (odd-indexed int16)
             *
             * VLUT16 lo vector has results for even byte positions (0,2,...,126):
             *   lo.h[k] = result for byte position 2*k
             *
             * So after widening:
             *   acc_even_lo[j] = result for byte 2*(2j)   = byte 4j
             *   acc_even_hi[j] = result for byte 2*(2j+1) = byte 4j+2
             *   acc_odd_lo[j]  = result for byte 2*(2j)+1 = byte 4j+1
             *   acc_odd_hi[j]  = result for byte 2*(2j+1)+1 = byte 4j+3
             *
             * Output mapping: byte position p -> output m_base+p
             */
            int32_t __attribute__((aligned(128))) ae_lo[32]; /* even-byte, even-idx */
            int32_t __attribute__((aligned(128))) ae_hi[32]; /* even-byte, odd-idx */
            int32_t __attribute__((aligned(128))) ao_lo[32]; /* odd-byte, even-idx */
            int32_t __attribute__((aligned(128))) ao_hi[32]; /* odd-byte, odd-idx */
            *(HVX_Vector *)ae_lo = Q6_V_lo_W(acc_even);
            *(HVX_Vector *)ae_hi = Q6_V_hi_W(acc_even);
            *(HVX_Vector *)ao_lo = Q6_V_lo_W(acc_odd);
            *(HVX_Vector *)ao_hi = Q6_V_hi_W(acc_odd);

            for (int j = 0; j < 32; j++) {
                partial_f[b][4*j + 0] = (float)ae_lo[j] * ls;  /* byte 4j   */
                partial_f[b][4*j + 1] = (float)ao_lo[j] * ls;  /* byte 4j+1 */
                partial_f[b][4*j + 2] = (float)ae_hi[j] * ls;  /* byte 4j+2 */
                partial_f[b][4*j + 3] = (float)ao_hi[j] * ls;  /* byte 4j+3 */
            }
        }

        /* Combine bit planes: y = 0.5*partial_0 + partial_1 + lb */
        for (int i = 0; i < 128 && (m_base + i) < M; i++) {
            y[m_base + i] = 0.5f * partial_f[0][i] + partial_f[1][i] + lb;
        }
    }
}

/* ========== 4Q VLUT16: process 4 Q positions per cycle ========== */

/*
 * Build a single VLUT16 vector containing 4 interleaved LUTs.
 *
 * Each LUT has 16 int16 entries. The VLUT16 segments 0-3 select
 * different quarters of the 64-entry vector. We interleave the
 * 4 LUTs so that each segment retrieves its own LUT.
 *
 * The input vectors to vlut16_shuffle need the pattern:
 *   l_tmp[i][j] = luts[j % 4][i]  (repeating across 64 positions)
 *
 * After the shuffle, segment s reads the s-th interleaved LUT.
 */
static inline HVX_Vector vlut16_build_lut_4q(
    const int16_t lut0[16],
    const int16_t lut1[16],
    const int16_t lut2[16],
    const int16_t lut3[16])
{
    int16_t __attribute__((aligned(128))) buf[64];
    HVX_Vector l_tmp[16];

    for (int i = 0; i < 16; i++) {
        /* Fill 64 int16 positions with the 4 LUT values interleaved:
         * positions 0,4,8,...,60 get lut0[i]  (segment 0)
         * positions 1,5,9,...,61 get lut1[i]  (segment 1)
         * positions 2,6,10,...,62 get lut2[i] (segment 2)
         * positions 3,7,11,...,63 get lut3[i] (segment 3) */
        for (int j = 0; j < 64; j += 4) {
            buf[j + 0] = lut0[i];
            buf[j + 1] = lut1[i];
            buf[j + 2] = lut2[i];
            buf[j + 3] = lut3[i];
        }
        l_tmp[i] = *(HVX_Vector *)buf;
    }

    HVX_Vector shuffled[16];
    vlut16_shuffle(l_tmp, shuffled);
    return shuffled[0];
}

/*
 * Pack ternary weights for 4Q VLUT16: 2 nibbles per byte.
 *
 * Layout: for each bit plane b, for each group of 4 Q positions (qg),
 *   vec_lo[m] = nibble[qg*4+0][m] | (nibble[qg*4+1][m] << 4)
 *   vec_hi[m] = nibble[qg*4+2][m] | (nibble[qg*4+3][m] << 4)
 *
 * Storage order: packed_4q[b][qg][vec_idx][m]
 *   where b in {0,1}, qg in {0..Q/4-1}, vec_idx in {0,1}, m in {0..M-1}
 *
 * Total size: 2 * (Q/4) * 2 * M bytes = Q * M bytes
 *   (half of original format which uses 2 * Q * M bytes)
 *
 * K must be a multiple of 16 (4 Q positions * 4 activations each).
 * M must be a multiple of 128.
 */
static void bitnet_pack_weights_4q(
    const int8_t *W,       /* [M * K] ternary weights, row-major */
    uint8_t *packed_4q,    /* [2 * (Q/4) * 2 * M] output */
    int M, int K)
{
    int Q = K / 4;
    int QG = Q / 4;  /* number of 4-Q groups */

    for (int b = 0; b < 2; b++) {
        for (int qg = 0; qg < QG; qg++) {
            for (int m = 0; m < M; m++) {
                /* Compute nibbles for the 4 Q positions in this group */
                uint8_t nibbles[4];
                for (int qi = 0; qi < 4; qi++) {
                    int q = qg * 4 + qi;
                    uint8_t nibble = 0;
                    for (int g = 0; g < 4; g++) {
                        int8_t w = W[m * K + q * 4 + g];
                        uint8_t enc = (uint8_t)(w + 2);
                        uint8_t bit = (enc >> b) & 1;
                        nibble |= (bit << g);
                    }
                    nibbles[qi] = nibble;
                }
                /* Pack: lo byte = nibbles[0] | (nibbles[1] << 4)
                 *        hi byte = nibbles[2] | (nibbles[3] << 4) */
                int base = (b * QG + qg) * 2 * M;
                packed_4q[base + 0 * M + m] = nibbles[0] | (nibbles[1] << 4);
                packed_4q[base + 1 * M + m] = nibbles[2] | (nibbles[3] << 4);
            }
        }
    }
}

/*
 * BitNet GEMV with 4Q VLUT16 optimization.
 *
 * Processes 4 Q positions per VLUT16 cycle using all 4 segments.
 * Uses global scale + int32 accumulation (same as bitnet_gemv_opt).
 *
 * Weight format: bitnet_pack_weights_4q() output.
 * M must be a multiple of 128. K must be a multiple of 16.
 */
static void bitnet_gemv_4q(
    const float *x,            /* [K] activations */
    const uint8_t *packed_4q,  /* packed weights (4Q format) */
    float *y,                  /* [M] output */
    int M, int K)
{
    int Q = K / 4;
    int QG = Q / 4;

    /* Compute global max|x| for a single quantization scale */
    float max_abs_x = 0;
    float sum_x = 0;
    for (int k = 0; k < K; k++) {
        float a = x[k] > 0 ? x[k] : -x[k];
        if (a > max_abs_x) max_abs_x = a;
        sum_x += x[k];
    }
    float lb = -0.5f * sum_x;

    /* Global scale: max possible LUT entry is 4 * max_abs_x */
    float ls = max_abs_x > 0 ? (4.0f * max_abs_x) / 32767.0f : 1e-10f;
    float inv_ls = 1.0f / ls;

    /* Mask for extracting low nibble */
    HVX_Vector mask_0f = Q6_Vb_vsplat_R(0x0F);

    /* Process each output chunk of 128 */
    for (int m_base = 0; m_base < M; m_base += 128) {

        float partial_f[2][128];

        for (int b = 0; b < 2; b++) {
            /* int32 accumulators: VLUT16 produces int16 pairs (even/odd bytes).
             * We accumulate all 4 segments' results into these. */
            HVX_VectorPair acc_even = Q6_W_vcombine_VV(
                Q6_V_vzero(), Q6_V_vzero());
            HVX_VectorPair acc_odd = Q6_W_vcombine_VV(
                Q6_V_vzero(), Q6_V_vzero());
            int first = 1;

            for (int qg = 0; qg < QG; qg++) {
                /* Build 4 LUTs for the 4 Q positions in this group */
                int16_t lut_arr[4][16];
                for (int qi = 0; qi < 4; qi++) {
                    int q = qg * 4 + qi;
                    float x0 = x[q*4+0], x1 = x[q*4+1];
                    float x2 = x[q*4+2], x3 = x[q*4+3];
                    for (int idx = 0; idx < 16; idx++) {
                        float sum = 0;
                        sum += (idx & 1) ? +x0 : -x0;
                        sum += (idx & 2) ? +x1 : -x1;
                        sum += (idx & 4) ? +x2 : -x2;
                        sum += (idx & 8) ? +x3 : -x3;
                        float v = sum * inv_ls;
                        lut_arr[qi][idx] = (int16_t)(v > 0 ? v + 0.5f : v - 0.5f);
                    }
                }

                /* Build interleaved 4-LUT vector.
                 * Segment mapping (verified by test):
                 *   seg 0 reads position 0, seg 1 reads position 2,
                 *   seg 2 reads position 1, seg 3 reads position 3.
                 * We use: seg0 for q+0 (w_lo low nibble),
                 *         seg1 for q+2 (w_hi low nibble),
                 *         seg2 for q+1 (w_lo high nibble),
                 *         seg3 for q+3 (w_hi high nibble).
                 * Position 0=lut[q+0], 1=lut[q+1], 2=lut[q+2], 3=lut[q+3]. */
                HVX_Vector lut_vec = vlut16_build_lut_4q(
                    lut_arr[0], lut_arr[1], lut_arr[2], lut_arr[3]);

                /* Load packed weight vectors */
                int w_off = (b * QG + qg) * 2 * M + m_base;
                const uint8_t *w_ptr = packed_4q + w_off;
                HVX_Vector w_lo = *(HVX_Vector *)(w_ptr);          /* nibble[q+0]|(nibble[q+1]<<4) */
                HVX_Vector w_hi = *(HVX_Vector *)(w_ptr + M);      /* nibble[q+2]|(nibble[q+3]<<4) */

                /* Extract nibbles (byte-level shift for correct per-byte operation) */
                HVX_Vector w_lo_bot = Q6_V_vand_VV(w_lo, mask_0f);       /* q+0 indices */
                HVX_Vector w_lo_top = Q6_Vub_vlsr_VubR(w_lo, 4);         /* q+1 indices */
                HVX_Vector w_hi_bot = Q6_V_vand_VV(w_hi, mask_0f);       /* q+2 indices */
                HVX_Vector w_hi_top = Q6_Vub_vlsr_VubR(w_hi, 4);         /* q+3 indices */

                /* VLUT16 lookups with 4 segments of one LUT vector */
                HVX_VectorPair r0 = Q6_Wh_vlut16_VbVhR_nomatch(w_lo_bot, lut_vec, 0);
                HVX_VectorPair r1 = Q6_Wh_vlut16_VbVhR_nomatch(w_hi_bot, lut_vec, 1);
                HVX_VectorPair r2 = Q6_Wh_vlut16_VbVhR_nomatch(w_lo_top, lut_vec, 2);
                HVX_VectorPair r3 = Q6_Wh_vlut16_VbVhR_nomatch(w_hi_top, lut_vec, 3);

                /* Accumulate 4 segment results into int32 accumulators.
                 * Cannot sum in int16 first -- 4 * 32767 overflows int16!
                 * Use widening add-accumulate for each segment result. */
                if (first) {
                    /* r0 + r1 widened */
                    acc_even = Q6_Ww_vadd_VhVh(Q6_V_lo_W(r0), Q6_V_lo_W(r1));
                    acc_odd  = Q6_Ww_vadd_VhVh(Q6_V_hi_W(r0), Q6_V_hi_W(r1));
                    /* + r2 */
                    acc_even = Q6_Ww_vaddacc_WwVhVh(acc_even, Q6_V_lo_W(r2), Q6_V_vzero());
                    acc_odd  = Q6_Ww_vaddacc_WwVhVh(acc_odd,  Q6_V_hi_W(r2), Q6_V_vzero());
                    /* + r3 */
                    acc_even = Q6_Ww_vaddacc_WwVhVh(acc_even, Q6_V_lo_W(r3), Q6_V_vzero());
                    acc_odd  = Q6_Ww_vaddacc_WwVhVh(acc_odd,  Q6_V_hi_W(r3), Q6_V_vzero());
                    first = 0;
                } else {
                    /* Accumulate all 4 results: each widens int16 to int32 */
                    acc_even = Q6_Ww_vaddacc_WwVhVh(acc_even, Q6_V_lo_W(r0), Q6_V_lo_W(r1));
                    acc_odd  = Q6_Ww_vaddacc_WwVhVh(acc_odd,  Q6_V_hi_W(r0), Q6_V_hi_W(r1));
                    acc_even = Q6_Ww_vaddacc_WwVhVh(acc_even, Q6_V_lo_W(r2), Q6_V_lo_W(r3));
                    acc_odd  = Q6_Ww_vaddacc_WwVhVh(acc_odd,  Q6_V_hi_W(r2), Q6_V_hi_W(r3));
                }
            }

            /* Convert int32 accumulators to float (same layout as opt version) */
            int32_t __attribute__((aligned(128))) ae_lo[32];
            int32_t __attribute__((aligned(128))) ae_hi[32];
            int32_t __attribute__((aligned(128))) ao_lo[32];
            int32_t __attribute__((aligned(128))) ao_hi[32];
            *(HVX_Vector *)ae_lo = Q6_V_lo_W(acc_even);
            *(HVX_Vector *)ae_hi = Q6_V_hi_W(acc_even);
            *(HVX_Vector *)ao_lo = Q6_V_lo_W(acc_odd);
            *(HVX_Vector *)ao_hi = Q6_V_hi_W(acc_odd);

            for (int j = 0; j < 32; j++) {
                partial_f[b][4*j + 0] = (float)ae_lo[j] * ls;
                partial_f[b][4*j + 1] = (float)ao_lo[j] * ls;
                partial_f[b][4*j + 2] = (float)ae_hi[j] * ls;
                partial_f[b][4*j + 3] = (float)ao_hi[j] * ls;
            }
        }

        /* Combine bit planes: y = 0.5*partial_0 + partial_1 + lb */
        for (int i = 0; i < 128 && (m_base + i) < M; i++) {
            y[m_base + i] = 0.5f * partial_f[0][i] + partial_f[1][i] + lb;
        }
    }
}

/* ========== Reference scalar GEMV ========== */

static void bitnet_gemv_reference(
    const float *x,      /* [K] */
    const int8_t *W,     /* [M * K] ternary, row-major */
    float *y,            /* [M] */
    int M, int K)
{
    for (int m = 0; m < M; m++) {
        float sum = 0;
        for (int k = 0; k < K; k++) {
            sum += (float)W[m * K + k] * x[k];
        }
        y[m] = sum;
    }
}

#endif /* BITNET_GEMV_H */
