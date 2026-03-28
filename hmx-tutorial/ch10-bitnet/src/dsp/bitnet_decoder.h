/*
 * bitnet_decoder.h -- BitNet decoder layer for single-token decode
 *
 * Implements one transformer decoder layer with:
 *   - RMSNorm (input + post-attention + BitNet sub-norms)
 *   - BitNet ternary GEMV projections (Q/K/V/O, gate/up/down)
 *   - RoPE (two-half split)
 *   - Multi-head attention with GQA
 *   - ReLU^2 gated MLP
 *   - Residual connections
 *
 * All projections use bitnet_gemv_opt() with packed ternary weights.
 * All HVX ops use qf32 arithmetic (v75 compatible).
 */

#ifndef BITNET_DECODER_H
#define BITNET_DECODER_H

#include "dsp/bitnet_gemv.h"
#include "dsp/bitnet_ops.h"
#include <stdlib.h>

/* Model dimensions (BitNet b1.58 2.4B / 3B class) */
#define HIDDEN_SIZE         2560
#define NUM_HEADS           20
#define NUM_KV_HEADS        5
#define HEAD_DIM            128
#define INTERMEDIATE_SIZE   6912
#define RMS_NORM_EPS        1e-5f

/* Derived constants */
#define KV_DIM              (NUM_KV_HEADS * HEAD_DIM)   /* 640  */
#define GQA_RATIO           (NUM_HEADS / NUM_KV_HEADS)  /* 4    */

/* Decoder layer weights (all pointers to weight buffers) */
typedef struct {
    /* RMSNorm weights (f32) */
    const float *input_ln_w;        /* [HIDDEN_SIZE]         */
    const float *post_attn_ln_w;    /* [HIDDEN_SIZE]         */
    const float *attn_sub_norm_w;   /* [HIDDEN_SIZE]         */
    const float *ffn_sub_norm_w;    /* [INTERMEDIATE_SIZE]   */

    /* Attention projection weights (packed for bitnet_gemv_opt)
     * Pack format: packed_w[2 * Q * M] where Q = K_in / 4
     * bitnet_gemv_opt(x, packed_w, y, M_out, K_in)           */
    const uint8_t *q_proj_w;   /* M=HIDDEN_SIZE, K=HIDDEN_SIZE => [2*(HIDDEN_SIZE/4)*HIDDEN_SIZE]  */
    const uint8_t *k_proj_w;   /* M=KV_DIM,      K=HIDDEN_SIZE => [2*(HIDDEN_SIZE/4)*KV_DIM]      */
    const uint8_t *v_proj_w;   /* M=KV_DIM,      K=HIDDEN_SIZE => [2*(HIDDEN_SIZE/4)*KV_DIM]      */
    const uint8_t *o_proj_w;   /* M=HIDDEN_SIZE, K=HIDDEN_SIZE => [2*(HIDDEN_SIZE/4)*HIDDEN_SIZE]  */

    /* MLP projection weights (packed) */
    const uint8_t *gate_proj_w; /* M=INTERMEDIATE_SIZE, K=HIDDEN_SIZE      */
    const uint8_t *up_proj_w;   /* M=INTERMEDIATE_SIZE, K=HIDDEN_SIZE      */
    const uint8_t *down_proj_w; /* M=HIDDEN_SIZE,       K=INTERMEDIATE_SIZE */

    /* RoPE tables (f32), indexed by hvx_rope_f32 as table[pos * half] */
    const float *rope_cos;      /* [max_pos * HEAD_DIM/2] */
    const float *rope_sin;      /* [max_pos * HEAD_DIM/2] */
} DecoderLayerWeights;

/*
 * KV cache for one layer.
 *
 * Layout: [NUM_KV_HEADS][max_seq_len][HEAD_DIM]
 *   k_cache[kv_h * max_seq_len * HEAD_DIM + t * HEAD_DIM + d]
 *
 * This matches hvx_mha_decode_f32 which indexes:
 *   k_head = k_cache + kv_h * seq_len * head_dim
 */
typedef struct {
    float *k_cache;     /* [NUM_KV_HEADS * max_seq_len * HEAD_DIM] */
    float *v_cache;     /* [NUM_KV_HEADS * max_seq_len * HEAD_DIM] */
    int    max_seq_len; /* allocated capacity */
    int    seq_len;     /* current filled length */
} KVCache;

/*
 * Run one decoder layer (decode mode: single token).
 *
 * x: input/output [HIDDEN_SIZE] f32 (modified in-place)
 * pos: token position in the sequence
 * weights: all layer weights
 * kv: KV cache for this layer
 *
 * All scratch buffers are heap-allocated (DSP stack is limited).
 */
static void bitnet_decoder_layer(
    float *x,                          /* [HIDDEN_SIZE] in/out */
    int pos,
    const DecoderLayerWeights *weights,
    KVCache *kv)
{
    /* ----- Allocate scratch buffers (128-byte aligned for HVX) ----- */

    float *norm_out = (float *)memalign(128, HIDDEN_SIZE * sizeof(float));
    float *q       = (float *)memalign(128, HIDDEN_SIZE * sizeof(float));
    float *k       = (float *)memalign(128, KV_DIM * sizeof(float));
    float *v       = (float *)memalign(128, KV_DIM * sizeof(float));
    float *attn_out = (float *)memalign(128, HIDDEN_SIZE * sizeof(float));

    /* MLP scratch -- INTERMEDIATE_SIZE = 6912, ~27 KB each */
    float *gate_buf = (float *)memalign(128, INTERMEDIATE_SIZE * sizeof(float));
    float *up_buf   = (float *)memalign(128, INTERMEDIATE_SIZE * sizeof(float));
    float *mlp_out  = (float *)memalign(128, HIDDEN_SIZE * sizeof(float));

    if (!norm_out || !q || !k || !v || !attn_out ||
        !gate_buf || !up_buf || !mlp_out) {
        /* Fatal: out of memory -- silently skip layer */
        free(norm_out); free(q); free(k); free(v);
        free(attn_out); free(gate_buf); free(up_buf); free(mlp_out);
        return;
    }

    /* ===== Self-Attention Block ===== */

    /* 1. Input LayerNorm */
    hvx_rmsnorm_f32(x, weights->input_ln_w, norm_out, HIDDEN_SIZE, RMS_NORM_EPS);

    /* 2. Q/K/V projections (BitNet GEMV)
     *    bitnet_gemv_opt(input, packed_w, output, M_out, K_in)
     *    q_proj: [HIDDEN_SIZE, HIDDEN_SIZE] @ norm_out -> q
     *    k_proj: [KV_DIM, HIDDEN_SIZE]      @ norm_out -> k
     *    v_proj: [KV_DIM, HIDDEN_SIZE]      @ norm_out -> v */
    bitnet_gemv_opt(norm_out, weights->q_proj_w, q, HIDDEN_SIZE, HIDDEN_SIZE);
    bitnet_gemv_opt(norm_out, weights->k_proj_w, k, KV_DIM, HIDDEN_SIZE);
    bitnet_gemv_opt(norm_out, weights->v_proj_w, v, KV_DIM, HIDDEN_SIZE);

    /* 3. RoPE on Q and K
     *    hvx_rope_f32(x, cos_table, sin_table, out, head_dim, pos)
     *    The function internally indexes: cos_ptr = cos_table + pos * half
     *    So we pass the base table pointer and the position. */
    for (int h = 0; h < NUM_HEADS; h++) {
        hvx_rope_f32(q + h * HEAD_DIM,
                     weights->rope_cos,
                     weights->rope_sin,
                     q + h * HEAD_DIM,   /* in-place */
                     HEAD_DIM, pos);
    }
    for (int h = 0; h < NUM_KV_HEADS; h++) {
        hvx_rope_f32(k + h * HEAD_DIM,
                     weights->rope_cos,
                     weights->rope_sin,
                     k + h * HEAD_DIM,   /* in-place */
                     HEAD_DIM, pos);
    }

    /* 4. Append K, V to cache
     *    Cache layout: [NUM_KV_HEADS][max_seq_len][HEAD_DIM]
     *    For each KV head, write HEAD_DIM floats at position pos. */
    for (int kv_h = 0; kv_h < NUM_KV_HEADS; kv_h++) {
        memcpy(kv->k_cache + kv_h * kv->max_seq_len * HEAD_DIM + pos * HEAD_DIM,
               k + kv_h * HEAD_DIM,
               HEAD_DIM * sizeof(float));
        memcpy(kv->v_cache + kv_h * kv->max_seq_len * HEAD_DIM + pos * HEAD_DIM,
               v + kv_h * HEAD_DIM,
               HEAD_DIM * sizeof(float));
    }
    kv->seq_len = pos + 1;

    /* 5. Multi-head attention with GQA
     *    hvx_mha_decode_f32 expects KV cache layout:
     *      [num_kv_heads][seq_len][head_dim]  (stride = seq_len * head_dim)
     *
     *    Our cache uses max_seq_len as stride.  When max_seq_len > seq_len
     *    we need compact copies so the stride matches.  When they are equal
     *    (which is the common case for the initial test), we can pass directly.
     *
     *    scale = 1.0 / sqrt(HEAD_DIM) */
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    if (kv->max_seq_len == kv->seq_len) {
        /* Stride matches -- pass cache directly */
        hvx_mha_decode_f32(q, kv->k_cache, kv->v_cache, attn_out,
                           NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, kv->seq_len, scale);
    } else {
        /* Compact the KV cache: [kv_heads][seq_len][head_dim] */
        int compact_size = NUM_KV_HEADS * kv->seq_len * HEAD_DIM;
        float *k_compact = (float *)memalign(128, compact_size * sizeof(float));
        float *v_compact = (float *)memalign(128, compact_size * sizeof(float));
        if (k_compact && v_compact) {
            for (int kv_h = 0; kv_h < NUM_KV_HEADS; kv_h++) {
                memcpy(k_compact + kv_h * kv->seq_len * HEAD_DIM,
                       kv->k_cache + kv_h * kv->max_seq_len * HEAD_DIM,
                       kv->seq_len * HEAD_DIM * sizeof(float));
                memcpy(v_compact + kv_h * kv->seq_len * HEAD_DIM,
                       kv->v_cache + kv_h * kv->max_seq_len * HEAD_DIM,
                       kv->seq_len * HEAD_DIM * sizeof(float));
            }
            hvx_mha_decode_f32(q, k_compact, v_compact, attn_out,
                               NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, kv->seq_len, scale);
        }
        free(k_compact);
        free(v_compact);
    }

    /* 6. attn_sub_norm (BitNet-specific: RMSNorm before o_proj) */
    hvx_rmsnorm_f32(attn_out, weights->attn_sub_norm_w, attn_out,
                    HIDDEN_SIZE, RMS_NORM_EPS);

    /* 7. O projection: reuse norm_out as temp for o_proj output */
    bitnet_gemv_opt(attn_out, weights->o_proj_w, norm_out, HIDDEN_SIZE, HIDDEN_SIZE);

    /* 8. Residual add: x = x + o_proj_out */
    hvx_add_f32(x, norm_out, x, HIDDEN_SIZE);

    /* ===== MLP Block ===== */

    /* 9. Post-attention LayerNorm */
    hvx_rmsnorm_f32(x, weights->post_attn_ln_w, norm_out, HIDDEN_SIZE, RMS_NORM_EPS);

    /* 10. Gate and Up projections */
    bitnet_gemv_opt(norm_out, weights->gate_proj_w, gate_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE);
    bitnet_gemv_opt(norm_out, weights->up_proj_w, up_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE);

    /* 11. ReLU^2 + element-wise multiply: gate = relu2(gate) * up */
    hvx_relu2_f32(gate_buf, gate_buf, INTERMEDIATE_SIZE);
    hvx_mul_f32(gate_buf, up_buf, gate_buf, INTERMEDIATE_SIZE);

    /* 12. ffn_sub_norm (BitNet-specific: RMSNorm before down_proj) */
    hvx_rmsnorm_f32(gate_buf, weights->ffn_sub_norm_w, gate_buf,
                    INTERMEDIATE_SIZE, RMS_NORM_EPS);

    /* 13. Down projection */
    bitnet_gemv_opt(gate_buf, weights->down_proj_w, mlp_out, HIDDEN_SIZE, INTERMEDIATE_SIZE);

    /* 14. Residual add: x = x + mlp_out */
    hvx_add_f32(x, mlp_out, x, HIDDEN_SIZE);

    /* ----- Free scratch ----- */
    free(norm_out);
    free(q);
    free(k);
    free(v);
    free(attn_out);
    free(gate_buf);
    free(up_buf);
    free(mlp_out);
}

#endif /* BITNET_DECODER_H */
