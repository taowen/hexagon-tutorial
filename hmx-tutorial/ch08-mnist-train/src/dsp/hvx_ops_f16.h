#ifndef HVX_OPS_F16_H
#define HVX_OPS_F16_H

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>
#include <string.h>
#include <math.h>
#include "common/protocol.h"

/* 64 half-floats per HVX vector (128 bytes / 2 bytes) */
#define HVX_HALFS 64

/* Fast exp approximation (Schraudolph's algorithm) */
static inline float fast_expf(float x) {
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) return 3.4e38f;
    union { float f; int32_t i; } u;
    u.i = (int32_t)(12102203.0f * x + 1065353216.0f);
    return u.f;
}

static inline float fast_logf(float x) {
    if (x <= 0.0f) return -88.0f;
    union { float f; int32_t i; } u;
    u.f = x;
    return (u.i - 1065353216.0f) / 12102203.0f;
}

/* ReLU forward f16: out = max(0, x)
 * Uses HVX integer sign-bit check: f16 positive iff bit[15]=0,
 * which means positive as signed int16. */
static void hvx_relu_forward_f16(_Float16 *out, const _Float16 *in, uint32_t n) {
    HVX_Vector vzero = Q6_V_vzero();
    uint32_t i = 0;
    for (; i + HVX_HALFS <= n; i += HVX_HALFS) {
        HVX_Vector v = *(HVX_Vector *)(in + i);
        HVX_VectorPred p = Q6_Q_vcmp_gt_VhVh(v, vzero);
        *(HVX_Vector *)(out + i) = Q6_V_vmux_QVV(p, v, vzero);
    }
    for (; i < n; i++)
        out[i] = in[i] > (_Float16)0 ? in[i] : (_Float16)0;
}

/* ReLU backward f16: dx = (pre_relu > 0) ? dout : 0 */
static void hvx_relu_backward_f16(_Float16 *dx, const _Float16 *dout,
                                   const _Float16 *pre_relu, uint32_t n) {
    HVX_Vector vzero = Q6_V_vzero();
    uint32_t i = 0;
    for (; i + HVX_HALFS <= n; i += HVX_HALFS) {
        HVX_Vector pr = *(HVX_Vector *)(pre_relu + i);
        HVX_Vector dv = *(HVX_Vector *)(dout + i);
        HVX_VectorPred p = Q6_Q_vcmp_gt_VhVh(pr, vzero);
        *(HVX_Vector *)(dx + i) = Q6_V_vmux_QVV(p, dv, vzero);
    }
    for (; i < n; i++)
        dx[i] = pre_relu[i] > (_Float16)0 ? dout[i] : (_Float16)0;
}

/* Bias add f16: out[i][j] += bias[j] for i in [0,rows)
 * HVX qf16 vectorized */
static void hvx_add_bias_f16(_Float16 *out, const _Float16 *bias,
                              uint32_t rows, uint32_t cols) {
    uint16_t one_bits = 0x3C00;
    HVX_Vector one_vec = Q6_Vh_vsplat_R((int32_t)one_bits);
    for (uint32_t i = 0; i < rows; i++) {
        _Float16 *row = out + i * cols;
        uint32_t j = 0;
        for (; j + HVX_HALFS <= cols; j += HVX_HALFS) {
            HVX_Vector ov = *(HVX_Vector *)(row + j);
            HVX_Vector bv = *(HVX_Vector *)(bias + j);
            HVX_Vector oq = Q6_Vqf16_vmpy_VhfVhf(ov, one_vec);
            HVX_Vector bq = Q6_Vqf16_vmpy_VhfVhf(bv, one_vec);
            HVX_Vector rq = Q6_Vqf16_vadd_Vqf16Vqf16(oq, bq);
            *(HVX_Vector *)(row + j) = Q6_Vhf_equals_Vqf16(rq);
        }
        for (; j < cols; j++)
            row[j] = (_Float16)((float)row[j] + (float)bias[j]);
    }
}

/* Bias backward f16: db[j] = sum_i(dout[i][j])
 * HVX qf16 vectorized, column-major accumulation for cache friendliness */
static void hvx_bias_backward_f16(_Float16 *db, const _Float16 *dout,
                                   uint32_t rows, uint32_t cols) {
    uint16_t one_bits = 0x3C00;
    HVX_Vector one_vec = Q6_Vh_vsplat_R((int32_t)one_bits);
    uint32_t n_vecs = cols / HVX_HALFS;
    for (uint32_t v = 0; v < n_vecs; v++) {
        HVX_Vector acc = Q6_V_vzero();
        for (uint32_t b = 0; b < rows; b++) {
            HVX_Vector row = *(HVX_Vector *)(dout + b * cols + v * HVX_HALFS);
            acc = Q6_Vqf16_vadd_Vqf16Vqf16(acc, Q6_Vqf16_vmpy_VhfVhf(row, one_vec));
        }
        *(HVX_Vector *)(db + v * HVX_HALFS) = Q6_Vhf_equals_Vqf16(acc);
    }
    /* Scalar tail for remaining columns */
    for (uint32_t j = n_vecs * HVX_HALFS; j < cols; j++) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < rows; i++)
            sum += (float)dout[i * cols + j];
        db[j] = (_Float16)sum;
    }
}

/* SGD update f16: w -= lr * grad
 * HVX qf16 vectorized */
static void hvx_sgd_update_f16(_Float16 *w, const _Float16 *grad, float lr, uint32_t n) {
    _Float16 neg_lr_h = (_Float16)(-lr);
    uint16_t neg_lr_bits;
    memcpy(&neg_lr_bits, &neg_lr_h, sizeof(neg_lr_bits));
    HVX_Vector neg_lr_vec = Q6_Vh_vsplat_R((int32_t)neg_lr_bits);
    uint16_t one_bits = 0x3C00; /* f16 1.0 */
    HVX_Vector one_vec = Q6_Vh_vsplat_R((int32_t)one_bits);

    uint32_t i = 0;
    for (; i + HVX_HALFS <= n; i += HVX_HALFS) {
        HVX_Vector wv = *(HVX_Vector *)(w + i);
        HVX_Vector gv = *(HVX_Vector *)(grad + i);
        /* delta = (-lr) * grad in qf16 */
        HVX_Vector delta_q = Q6_Vqf16_vmpy_VhfVhf(neg_lr_vec, gv);
        /* w_qf16 = w * 1.0 (promote to qf16) */
        HVX_Vector w_q = Q6_Vqf16_vmpy_VhfVhf(wv, one_vec);
        /* result = w + delta = w - lr*grad */
        HVX_Vector result_q = Q6_Vqf16_vadd_Vqf16Vqf16(w_q, delta_q);
        *(HVX_Vector *)(w + i) = Q6_Vhf_equals_Vqf16(result_q);
    }
    /* Scalar tail */
    for (; i < n; i++)
        w[i] = (_Float16)((float)w[i] - lr * (float)grad[i]);
}

/* Compute dlogits f16: dlogits = (probs - one_hot) / batch
 * Partially HVX vectorized: memcpy + scalar label subtract + HVX scale */
static void hvx_compute_dlogits_f16(_Float16 *dlogits, const _Float16 *probs,
                                     const uint8_t *labels,
                                     uint32_t batch, uint32_t classes) {
    /* Copy probs to dlogits first, then subtract 1.0 at label positions and scale */
    memcpy(dlogits, probs, batch * classes * sizeof(_Float16));

    _Float16 inv_batch_h = (_Float16)(1.0f / (float)batch);
    uint16_t inv_bits;
    memcpy(&inv_bits, &inv_batch_h, sizeof(inv_bits));
    HVX_Vector inv_vec = Q6_Vh_vsplat_R((int32_t)inv_bits);

    for (uint32_t i = 0; i < batch; i++) {
        _Float16 *row = dlogits + i * classes;
        /* Subtract 1.0 at label position (scalar, one element) */
        row[labels[i]] = (_Float16)((float)row[labels[i]] - 1.0f);
        /* Scale entire row by 1/batch using HVX qf16 */
        uint32_t j = 0;
        for (; j + HVX_HALFS <= classes; j += HVX_HALFS) {
            HVX_Vector v = *(HVX_Vector *)(row + j);
            HVX_Vector vq = Q6_Vqf16_vmpy_VhfVhf(v, inv_vec);
            *(HVX_Vector *)(row + j) = Q6_Vhf_equals_Vqf16(vq);
        }
        for (; j < classes; j++)
            row[j] = (_Float16)((float)row[j] * (1.0f / (float)batch));
    }
}

/* Softmax cross-entropy f16
 * Input logits are f16, output probs are f16
 * Computation done in f32 for numerical stability (exp/log need range)
 * Returns loss as float */
static float hvx_softmax_cross_entropy_f16(
    _Float16 *probs, const _Float16 *logits,
    const uint8_t *labels, uint32_t batch, uint32_t classes)
{
    float total_loss = 0.0f;
    for (uint32_t i = 0; i < batch; i++) {
        /* Find max for numerical stability */
        float max_val = (float)logits[i * classes];
        for (uint32_t j = 1; j < classes; j++) {
            float v = (float)logits[i * classes + j];
            if (v > max_val) max_val = v;
        }
        /* Compute exp and sum */
        float sum_exp = 0.0f;
        for (uint32_t j = 0; j < classes; j++) {
            float e = fast_expf((float)logits[i * classes + j] - max_val);
            probs[i * classes + j] = (_Float16)e;  /* temp store */
            sum_exp += e;
        }
        /* Normalize and compute loss */
        float inv_sum = 1.0f / sum_exp;
        for (uint32_t j = 0; j < classes; j++) {
            float p = (float)probs[i * classes + j] * inv_sum;
            probs[i * classes + j] = (_Float16)p;
        }
        float p_correct = (float)probs[i * classes + labels[i]];
        total_loss -= fast_logf(p_correct > 1e-7f ? p_correct : 1e-7f);
    }
    return total_loss / (float)batch;
}

#endif /* HVX_OPS_F16_H */
