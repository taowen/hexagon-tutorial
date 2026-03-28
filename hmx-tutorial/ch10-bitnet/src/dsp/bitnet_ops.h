/*
 * bitnet_ops.h -- HVX f32 operators for BitNet decoder layers
 *
 * Implements:
 *   - hvx_add_f32()               : element-wise add (residual)
 *   - hvx_mul_f32()               : element-wise multiply (gating)
 *   - hvx_relu2_f32()             : ReLU squared activation
 *   - hvx_rmsnorm_f32()           : RMSNorm normalization
 *   - hvx_rope_f32()              : Rotary Position Embedding
 *   - hvx_softmax_f32()           : softmax over f32 vector
 *   - hvx_dot_f32()               : dot product of two f32 vectors
 *   - hvx_attention_decode_f32()  : single-head scaled dot-product attention (decode)
 *   - hvx_mha_decode_f32()        : multi-head attention with GQA (decode)
 *
 * HVX v75 constraints:
 *   - No direct IEEE f32 HVX ops; must use qf32 for multiply/add/sub
 *   - Scalar expf()/sqrtf() for transcendentals
 *   - All HVX load/store pointers must be 128-byte aligned
 */

#ifndef BITNET_OPS_H
#define BITNET_OPS_H

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>
#include <math.h>

#define HVX_FLOATS 32  /* 1024-bit / 32-bit = 32 floats per HVX vector */

/* ========== Element-wise add (residual) ========== */

/*
 * out[i] = a[i] + b[i] for i in [0, n)
 */
static void hvx_add_f32(const float *a, const float *b, float *out, int n) {
    int i = 0;
    int n_vec = n & ~(HVX_FLOATS - 1);
    for (; i < n_vec; i += HVX_FLOATS) {
        HVX_Vector va = *(HVX_Vector *)(a + i);
        HVX_Vector vb = *(HVX_Vector *)(b + i);
        HVX_Vector sum = Q6_Vqf32_vadd_VsfVsf(va, vb);
        *(HVX_Vector *)(out + i) = Q6_Vsf_equals_Vqf32(sum);
    }
    for (; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

/* ========== Element-wise multiply ========== */

/*
 * out[i] = a[i] * b[i] for i in [0, n)
 */
static void hvx_mul_f32(const float *a, const float *b, float *out, int n) {
    int i = 0;
    int n_vec = n & ~(HVX_FLOATS - 1);
    for (; i < n_vec; i += HVX_FLOATS) {
        HVX_Vector va = *(HVX_Vector *)(a + i);
        HVX_Vector vb = *(HVX_Vector *)(b + i);
        HVX_Vector prod = Q6_Vqf32_vmpy_VsfVsf(va, vb);
        *(HVX_Vector *)(out + i) = Q6_Vsf_equals_Vqf32(prod);
    }
    for (; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

/* ========== ReLU squared ========== */

/*
 * out[i] = max(0, x[i])^2
 *
 * ReLU via integer comparison: positive IEEE f32 has sign bit = 0,
 * so vcmp_gt_VwVw against -1 (0xFFFFFFFF) selects positive values.
 * Then square via qf32 multiply.
 */
static void hvx_relu2_f32(const float *x, float *out, int n) {
    HVX_Vector zero = Q6_V_vzero();
    HVX_Vector neg1 = Q6_V_vsplat_R(-1);
    int i = 0;
    int n_vec = n & ~(HVX_FLOATS - 1);
    for (; i < n_vec; i += HVX_FLOATS) {
        HVX_Vector v = *(HVX_Vector *)(x + i);
        HVX_VectorPred pos = Q6_Q_vcmp_gt_VwVw(v, neg1);
        HVX_Vector relu_v = Q6_V_vmux_QVV(pos, v, zero);
        HVX_Vector sq = Q6_Vqf32_vmpy_VsfVsf(relu_v, relu_v);
        *(HVX_Vector *)(out + i) = Q6_Vsf_equals_Vqf32(sq);
    }
    for (; i < n; i++) {
        float v = x[i] > 0.0f ? x[i] : 0.0f;
        out[i] = v * v;
    }
}

/* ========== RMSNorm ========== */

/*
 * out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
 *
 * 1. Sum of squares using HVX qf32 multiply, scalar horizontal reduction
 * 2. rms = sqrtf(sum_sq / n + eps) via scalar math
 * 3. Broadcast inv_rms and element-wise: out = x * inv_rms * weight
 */
static void hvx_rmsnorm_f32(const float *x, const float *weight,
                             float *out, int n, float eps) {
    int n_vec = n & ~(HVX_FLOATS - 1);

    /* Step 1: Sum of squares.
     * For each HVX vector, compute x*x, store to temp, sum scalar.
     * n is typically 2560 = 80 vectors -- scalar reduction is fine. */
    float sum_sq = 0.0f;
    float __attribute__((aligned(128))) tmp[HVX_FLOATS];

    for (int i = 0; i < n_vec; i += HVX_FLOATS) {
        HVX_Vector vx = *(HVX_Vector *)(x + i);
        HVX_Vector sq = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vx, vx));
        *(HVX_Vector *)tmp = sq;
        for (int j = 0; j < HVX_FLOATS; j++) {
            sum_sq += tmp[j];
        }
    }
    for (int i = n_vec; i < n; i++) {
        sum_sq += x[i] * x[i];
    }

    /* Step 2: Compute inv_rms */
    float rms = sqrtf(sum_sq / (float)n + eps);
    float inv_rms = 1.0f / rms;

    /* Step 3: out = x * inv_rms * weight (two qf32 multiplies) */
    HVX_Vector v_inv_rms = Q6_V_vsplat_R(*(int32_t *)&inv_rms);

    for (int i = 0; i < n_vec; i += HVX_FLOATS) {
        HVX_Vector vx = *(HVX_Vector *)(x + i);
        HVX_Vector vw = *(HVX_Vector *)(weight + i);
        HVX_Vector normed = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vmpy_VsfVsf(vx, v_inv_rms));
        HVX_Vector result = Q6_Vqf32_vmpy_VsfVsf(normed, vw);
        *(HVX_Vector *)(out + i) = Q6_Vsf_equals_Vqf32(result);
    }
    for (int i = n_vec; i < n; i++) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

/* ========== RoPE (Rotary Position Embedding) ========== */

/*
 * "Two-half" split RoPE applied to a single head:
 *   x_r = x[0 .. head_dim/2 - 1]
 *   x_i = x[head_dim/2 .. head_dim - 1]
 *   out[0..half-1]     = x_r * cos - x_i * sin
 *   out[half..dim-1]   = x_r * sin + x_i * cos
 *
 * cos_table, sin_table: [max_seq_len, head_dim/2]
 *   Indexed by pos * (head_dim / 2).
 *
 * head_dim=128: half=64 = 2 HVX vectors of f32.
 */
static void hvx_rope_f32(const float *x, const float *cos_table,
                          const float *sin_table, float *out,
                          int head_dim, int pos) {
    int half = head_dim / 2;
    const float *cos_ptr = cos_table + pos * half;
    const float *sin_ptr = sin_table + pos * half;
    const float *x_r = x;
    const float *x_i = x + half;
    float *out_r = out;
    float *out_i = out + half;

    int i = 0;
    int half_vec = half & ~(HVX_FLOATS - 1);

    for (; i < half_vec; i += HVX_FLOATS) {
        HVX_Vector vxr = *(HVX_Vector *)(x_r + i);
        HVX_Vector vxi = *(HVX_Vector *)(x_i + i);
        HVX_Vector vc  = *(HVX_Vector *)(cos_ptr + i);
        HVX_Vector vs  = *(HVX_Vector *)(sin_ptr + i);

        /* out_r = x_r * cos - x_i * sin */
        HVX_Vector rc = Q6_Vqf32_vmpy_VsfVsf(vxr, vc);
        HVX_Vector is = Q6_Vqf32_vmpy_VsfVsf(vxi, vs);
        HVX_Vector vout_r = Q6_Vqf32_vsub_Vqf32Vqf32(rc, is);
        *(HVX_Vector *)(out_r + i) = Q6_Vsf_equals_Vqf32(vout_r);

        /* out_i = x_r * sin + x_i * cos */
        HVX_Vector rs = Q6_Vqf32_vmpy_VsfVsf(vxr, vs);
        HVX_Vector ic = Q6_Vqf32_vmpy_VsfVsf(vxi, vc);
        HVX_Vector vout_i = Q6_Vqf32_vadd_Vqf32Vqf32(rs, ic);
        *(HVX_Vector *)(out_i + i) = Q6_Vsf_equals_Vqf32(vout_i);
    }
    for (; i < half; i++) {
        float c = cos_ptr[i];
        float s = sin_ptr[i];
        out_r[i] = x_r[i] * c - x_i[i] * s;
        out_i[i] = x_r[i] * s + x_i[i] * c;
    }
}

/* ========== Dot product (f32, HVX qf32) ========== */

/*
 * Compute dot product of two f32 vectors using HVX qf32 multiply + accumulate.
 * n must be a multiple of 32. Pointers must be 128-byte aligned.
 */
static float hvx_dot_f32(const float *a, const float *b, int n) {
    /* Accumulate in qf32 using HVX for the multiply, then reduce scalar */
    float __attribute__((aligned(128))) tmp[HVX_FLOATS];
    HVX_Vector acc = Q6_V_vzero();
    int first = 1;

    for (int i = 0; i < n; i += HVX_FLOATS) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);

        /* qf32 multiply: va * vb */
        HVX_Vector prod_qf32 = Q6_Vqf32_vmpy_VsfVsf(va, vb);

        if (first) {
            acc = Q6_Vsf_equals_Vqf32(prod_qf32);
            first = 0;
        } else {
            /* Accumulate: acc(sf) + prod(qf32) -> qf32, then back to sf */
            HVX_Vector sum_qf32 = Q6_Vqf32_vadd_VsfVsf(
                acc, Q6_Vsf_equals_Vqf32(prod_qf32));
            acc = Q6_Vsf_equals_Vqf32(sum_qf32);
        }
    }

    /* Reduce 32 floats to scalar */
    *(HVX_Vector *)tmp = acc;
    float sum = 0.0f;
    for (int i = 0; i < HVX_FLOATS; i++) {
        sum += tmp[i];
    }
    return sum;
}

/* ========== Softmax (f32) ========== */

/*
 * Compute softmax: out[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
 *
 * Uses scalar expf() for correctness. HVX is used for the final
 * normalization (multiply by 1/sum).
 *
 * n can be any positive integer (handles tail elements).
 * x and out need not be HVX-aligned for small n; the HVX normalization
 * only runs on full 32-float chunks.
 */
static void hvx_softmax_f32(const float *x, float *out, int n) {
    /* Step 1: find max */
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* Step 2: exp(x - max) and sum */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = expf(x[i] - max_val);
        sum += out[i];
    }

    /* Step 3: normalize */
    float inv_sum = 1.0f / sum;

    /* HVX-accelerated normalization for aligned, full-vector chunks */
    int n_hvx = (n / HVX_FLOATS) * HVX_FLOATS;

    if (n_hvx > 0 && ((uintptr_t)out % 128) == 0) {
        HVX_Vector v_inv = Q6_V_vsplat_R(*(int *)&inv_sum);
        for (int i = 0; i < n_hvx; i += HVX_FLOATS) {
            HVX_Vector v = *(HVX_Vector *)(out + i);
            HVX_Vector r = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v, v_inv));
            *(HVX_Vector *)(out + i) = r;
        }
        /* Scalar tail */
        for (int i = n_hvx; i < n; i++) {
            out[i] *= inv_sum;
        }
    } else {
        /* Fully scalar path for unaligned or small buffers */
        for (int i = 0; i < n; i++) {
            out[i] *= inv_sum;
        }
    }
}

/* ========== Scaled dot-product attention (single head, decode) ========== */

/*
 * Single-head attention for autoregressive decode (seq_len=1 query).
 *
 * q:       [head_dim]               -- current query vector
 * k_cache: [seq_len * head_dim]     -- cached keys (row-major)
 * v_cache: [seq_len * head_dim]     -- cached values (row-major)
 * out:     [head_dim]               -- output vector
 * head_dim: dimension per head (must be multiple of 32 for HVX dot)
 * seq_len:  number of cached positions
 * scale:    typically 1/sqrt(head_dim)
 *
 * Steps:
 *   1. scores[t] = dot(q, k_cache[t]) * scale   for t in [0, seq_len)
 *   2. attn = softmax(scores)
 *   3. out = sum_t attn[t] * v_cache[t]
 */
static void hvx_attention_decode_f32(
    const float *q,
    const float *k_cache,
    const float *v_cache,
    float *out,
    int head_dim,
    int seq_len,
    float scale)
{
    /* Allocate scores buffer -- aligned for potential HVX softmax use.
     * Pad to multiple of 32 for HVX alignment. */
    int scores_padded = ((seq_len + HVX_FLOATS - 1) / HVX_FLOATS) * HVX_FLOATS;
    float *scores = (float *)memalign(128, scores_padded * sizeof(float));
    float *attn   = (float *)memalign(128, scores_padded * sizeof(float));
    if (!scores || !attn) {
        /* Fallback: zero output */
        memset(out, 0, head_dim * sizeof(float));
        free(scores);
        free(attn);
        return;
    }
    memset(scores, 0, scores_padded * sizeof(float));
    memset(attn, 0, scores_padded * sizeof(float));

    /* Step 1: Compute attention scores */
    if (head_dim >= HVX_FLOATS && (head_dim % HVX_FLOATS) == 0 &&
        ((uintptr_t)q % 128) == 0 && ((uintptr_t)k_cache % 128) == 0) {
        /* HVX dot product path */
        for (int t = 0; t < seq_len; t++) {
            scores[t] = hvx_dot_f32(q, k_cache + t * head_dim, head_dim) * scale;
        }
    } else {
        /* Scalar fallback for unaligned or small head_dim */
        for (int t = 0; t < seq_len; t++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q[d] * k_cache[t * head_dim + d];
            }
            scores[t] = dot * scale;
        }
    }

    /* Step 2: Softmax over scores[0..seq_len-1] */
    hvx_softmax_f32(scores, attn, seq_len);

    /* Step 3: Weighted sum of values: out[d] = sum_t attn[t] * v_cache[t][d] */
    /* Initialize output to zero */
    int dim_hvx = (head_dim / HVX_FLOATS) * HVX_FLOATS;

    if (dim_hvx == head_dim && ((uintptr_t)out % 128) == 0 &&
        ((uintptr_t)v_cache % 128) == 0) {
        /* HVX path: accumulate attn[t] * v_cache[t] using qf32 */
        /* Zero the output */
        for (int d = 0; d < head_dim; d += HVX_FLOATS) {
            *(HVX_Vector *)(out + d) = Q6_V_vzero();
        }

        for (int t = 0; t < seq_len; t++) {
            float w = attn[t];
            if (w == 0.0f) continue;

            HVX_Vector v_w = Q6_V_vsplat_R(*(int *)&w);
            const float *v_row = v_cache + t * head_dim;

            for (int d = 0; d < head_dim; d += HVX_FLOATS) {
                HVX_Vector v_val = *(const HVX_Vector *)(v_row + d);
                HVX_Vector v_out = *(HVX_Vector *)(out + d);

                /* product = w * v_val (qf32) */
                HVX_Vector prod_qf32 = Q6_Vqf32_vmpy_VsfVsf(v_w, v_val);
                HVX_Vector prod_sf = Q6_Vsf_equals_Vqf32(prod_qf32);

                /* accumulate: out += prod */
                HVX_Vector sum_qf32 = Q6_Vqf32_vadd_VsfVsf(v_out, prod_sf);
                *(HVX_Vector *)(out + d) = Q6_Vsf_equals_Vqf32(sum_qf32);
            }
        }
    } else {
        /* Scalar fallback */
        memset(out, 0, head_dim * sizeof(float));
        for (int t = 0; t < seq_len; t++) {
            float w = attn[t];
            const float *v_row = v_cache + t * head_dim;
            for (int d = 0; d < head_dim; d++) {
                out[d] += w * v_row[d];
            }
        }
    }

    free(scores);
    free(attn);
}

/* ========== Multi-head attention with GQA (decode) ========== */

/*
 * Multi-head attention with grouped query attention (GQA) for decode.
 *
 * q_all:    [num_heads * head_dim]                    -- all query heads
 * k_cache:  [num_kv_heads * seq_len * head_dim]       -- KV cache keys
 * v_cache:  [num_kv_heads * seq_len * head_dim]       -- KV cache values
 * out:      [num_heads * head_dim]                    -- all output heads
 *
 * GQA: each KV head serves (num_heads / num_kv_heads) query heads.
 *   heads 0..(ratio-1) use KV head 0
 *   heads ratio..(2*ratio-1) use KV head 1
 *   etc.
 */
static void hvx_mha_decode_f32(
    const float *q_all,
    const float *k_cache,
    const float *v_cache,
    float *out,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    float scale)
{
    int gqa_ratio = num_heads / num_kv_heads;

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / gqa_ratio;

        const float *q_head = q_all + h * head_dim;
        const float *k_head = k_cache + kv_h * seq_len * head_dim;
        const float *v_head = v_cache + kv_h * seq_len * head_dim;
        float *out_head = out + h * head_dim;

        hvx_attention_decode_f32(q_head, k_head, v_head, out_head,
                                 head_dim, seq_len, scale);
    }
}

#endif /* BITNET_OPS_H */
