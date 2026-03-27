/*
 * HVX training operator helpers (activation, loss, etc.)
 *
 * Extracted from mnist_train_dsp.c for readability.
 * All functions are static -- this header is intended to be included
 * by exactly one translation unit.
 */
#ifndef HVX_OPS_H
#define HVX_OPS_H

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include "common/protocol.h"

#ifndef HVX_FLOATS
#define HVX_FLOATS 32  /* 1024-bit HVX vector / 32-bit float */
#endif

/* Fast exp approximation (Schraudolph's algorithm, good enough for softmax) */
static float hvx_expf(float x) {
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) return 3.4e38f;
    union { float f; int32_t i; } u;
    u.i = (int32_t)(12102203.0f * x + 1065353216.0f);
    return u.f;
}

/* Fast log approximation */
static float hvx_logf(float x) {
    if (x <= 0.0f) return -88.0f;
    union { float f; int32_t i; } u;
    u.f = x;
    return (u.i - 1065353216.0f) / 12102203.0f;
}

static void hvx_add_bias(float *out, const float *bias, uint32_t batch, uint32_t dim) {
    uint32_t dim_vec = dim & ~(HVX_FLOATS - 1);
    for (uint32_t b = 0; b < batch; b++) {
        float *row = out + b * dim;
        uint32_t j = 0;
        for (; j < dim_vec; j += HVX_FLOATS) {
            HVX_Vector o = *(HVX_Vector *)(row + j);
            HVX_Vector bv = *(HVX_Vector *)(bias + j);
            *(HVX_Vector *)(row + j) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(o, bv));
        }
        for (; j < dim; j++)
            row[j] += bias[j];
    }
}

static void hvx_relu_forward(float *x, uint32_t n) {
    /* ReLU: max(x, 0). Use integer vmax since positive IEEE floats
       have the same ordering as unsigned integers, and zero = 0x00000000 */
    HVX_Vector zero = Q6_V_vzero();
    uint32_t n_vec = n & ~(HVX_FLOATS - 1);
    uint32_t i = 0;
    for (; i < n_vec; i += HVX_FLOATS) {
        HVX_Vector v = *(HVX_Vector *)(x + i);
        /* For IEEE f32: if sign bit set (negative), replace with 0 */
        HVX_VectorPred pos = Q6_Q_vcmp_gt_VwVw(v, Q6_V_vsplat_R(-1));
        *(HVX_Vector *)(x + i) = Q6_V_vmux_QVV(pos, v, zero);
    }
    for (; i < n; i++)
        if (x[i] < 0.0f) x[i] = 0.0f;
}

static void hvx_relu_backward(float *dx, const float *pre_relu, uint32_t n) {
    HVX_Vector zero = Q6_V_vzero();
    uint32_t n_vec = n & ~(HVX_FLOATS - 1);
    uint32_t i = 0;
    for (; i < n_vec; i += HVX_FLOATS) {
        HVX_Vector d = *(HVX_Vector *)(dx + i);
        HVX_Vector p = *(HVX_Vector *)(pre_relu + i);
        /* mask = pre_relu > 0 (using integer comparison on IEEE floats) */
        HVX_VectorPred mask = Q6_Q_vcmp_gt_VwVw(p, Q6_V_vsplat_R(-1));
        *(HVX_Vector *)(dx + i) = Q6_V_vmux_QVV(mask, d, zero);
    }
    for (; i < n; i++)
        if (pre_relu[i] <= 0.0f) dx[i] = 0.0f;
}

static void hvx_bias_backward(float *db, const float *dout, uint32_t batch, uint32_t dim) {
    HVX_Vector one_v = Q6_V_vsplat_R(0x3f800000);
    uint32_t n_vecs = dim / HVX_FLOATS;
    for (uint32_t v = 0; v < n_vecs; v++) {
        HVX_Vector acc = Q6_V_vzero();
        for (uint32_t b = 0; b < batch; b++) {
            HVX_Vector row = *(HVX_Vector *)(dout + b * dim + v * HVX_FLOATS);
            acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(row, one_v));
        }
        *(HVX_Vector *)(db + v * HVX_FLOATS) = Q6_Vsf_equals_Vqf32(acc);
    }
}

static void hvx_compute_dlogits(float *dlogits, const float *probs,
                                 const uint8_t *labels, uint32_t batch) {
    float inv_bs = 1.0f / (float)batch;
    HVX_Vector inv_bs_vec = Q6_V_vsplat_R(*(int32_t *)&inv_bs);
    memcpy(dlogits, probs, batch * NET_OUTPUT_DIM_PAD * sizeof(float));
    for (uint32_t b = 0; b < batch; b++) {
        float *row = dlogits + b * NET_OUTPUT_DIM_PAD;
        row[labels[b]] -= 1.0f;
        HVX_Vector v = *(HVX_Vector *)row;
        *(HVX_Vector *)row = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v, inv_bs_vec));
    }
}

static float hvx_softmax_cross_entropy(const float *logits, const uint8_t *labels,
                                        float *probs, uint32_t batch) {
    float total_loss = 0.0f;
    for (uint32_t b = 0; b < batch; b++) {
        const float *row = logits + b * NET_OUTPUT_DIM_PAD;
        float *prob = probs + b * NET_OUTPUT_DIM_PAD;

        float max_val = row[0];
        for (int j = 1; j < NET_OUTPUT_DIM; j++)
            if (row[j] > max_val) max_val = row[j];

        float sum_exp = 0.0f;
        for (int j = 0; j < NET_OUTPUT_DIM; j++) {
            float val = hvx_expf(row[j] - max_val);
            if (val < 1e-10f) val = 1e-10f;
            prob[j] = val;
            sum_exp += val;
        }
        for (int j = 0; j < NET_OUTPUT_DIM; j++)
            prob[j] /= sum_exp;
        for (int j = NET_OUTPUT_DIM; j < NET_OUTPUT_DIM_PAD; j++)
            prob[j] = 0.0f;

        int label = labels[b];
        float p = prob[label];
        if (p < 1e-7f) p = 1e-7f;
        total_loss += -hvx_logf(p);
    }
    return total_loss / (float)batch;
}

static void hvx_sgd_update(float *w, const float *grad, float lr, uint32_t n) {
    HVX_Vector lr_vec = Q6_V_vsplat_R(*(int32_t *)&lr);
    HVX_Vector one = Q6_V_vsplat_R(0x3f800000);
    uint32_t n_vec = n & ~(HVX_FLOATS - 1);
    uint32_t i = 0;
    for (; i < n_vec; i += HVX_FLOATS) {
        HVX_Vector wv = *(HVX_Vector *)(w + i);
        HVX_Vector gv = *(HVX_Vector *)(grad + i);
        HVX_Vector prod = Q6_Vqf32_vmpy_VsfVsf(lr_vec, gv);
        HVX_Vector wq = Q6_Vqf32_vmpy_VsfVsf(wv, one);
        *(HVX_Vector *)(w + i) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(wq, prod));
    }
    for (; i < n; i++)
        w[i] -= lr * grad[i];
}

/* f32 -> f16 conversion (scalar, used by HMX path)
 * Note: Q6_Vhf_vcvt_VsfVsf is v79+ only, and Q6_Vhf_equals_Wqf32
 * produces incorrect element ordering on v75. Scalar fallback is correct. */
static void hvx_f32_to_f16(_Float16 *dst, const float *src, uint32_t n) {
    for (uint32_t i = 0; i < n; i++)
        dst[i] = (_Float16)src[i];
}

/* f16 -> f32 conversion (scalar, used by HMX path) */
static void hvx_f16_to_f32(float *dst, const _Float16 *src, uint32_t n) {
    for (uint32_t i = 0; i < n; i++)
        dst[i] = (float)src[i];
}

#endif /* HVX_OPS_H */
