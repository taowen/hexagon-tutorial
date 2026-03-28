/*
 * bitnet_gemv.h -- BitNet GEMV via VLUT16
 *
 * Computes y[M] = W[M,K] @ x[K] where W is ternary {-1, 0, +1}.
 *
 * Three phases:
 *   1. LUT construction: for each group of 4 activations, precompute
 *      16 possible ±x sums, quantize to int16, shuffle for VLUT16
 *   2. VLUT16 lookup + accumulation: use packed weight nibbles as
 *      indices, accumulate int16 results into float
 *   3. Bit-serial combination: output = 0.5*partial_0 + partial_1 + lb
 *
 * Encoding: ternary w ∈ {-1,0,+1} → enc = w + 2 ∈ {1,2,3}
 *   bit0 = enc & 1, bit1 = (enc >> 1) & 1
 *
 * LUT[index] = Σ_j ((index>>j)&1 ? +x_j : -x_j)  for j=0..3
 *
 * Formula: y = 0.5 * partial_bit0 + partial_bit1 - 0.5 * sum(x)
 *
 * Weight layout (simple, not optimized):
 *   packed_w[bit_plane][q][m] = 4-bit nibble index
 *   Total: 2 * Q * M bytes, where Q = K/4
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
