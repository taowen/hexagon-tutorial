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

/* =========================================================================
 * HVX polynomial softmax + cross-entropy (vectorized, f16 in/out)
 *
 * Each row is one HVX vector (64 x f16, NET_OUTPUT_DIM_PAD=64).
 * Only elements [0..NET_OUTPUT_DIM-1] are active; the rest are masked to -inf.
 *
 * Polynomial exp approximation ported from QNN SDK ExampleOpPackageSoftmax.cpp:
 *   x_scaled = (logit - max) / ln(2)
 *   Split into integer exponent n and normalized mantissa m (|m| in [1,2))
 *   exp(r) ~ c0 + c1*m + c2*m^2 + c3*m^3   (where r = m * ln(2))
 *   2^n computed via repeated squaring + vmux select
 *   exp(x_scaled) = poly^(2^n)
 * ========================================================================= */

static inline void hvx_softmax_cross_entropy_f16_vec(
    const uint16_t* logits,    /* [batch_size * NET_OUTPUT_DIM_PAD] in VTCM, 128B-aligned rows */
    uint16_t* probs,           /* [batch_size * NET_OUTPUT_DIM_PAD] output probs */
    const uint8_t* labels,     /* [batch_size] ground truth labels */
    int batch_size,
    float* total_loss,         /* output: sum of cross-entropy losses */
    int* total_correct         /* output: count of correct predictions */
)
{
    /* Polynomial coefficients for exp2 approximation on mantissa range:
     * exp2(m) ~ c0 + c1*m + c2*m^2 + c3*m^3  for m in [1,2) */
    union { float f; int32_t i; } scaleu, uc0, uc1, uc2, uc3;
    scaleu.f = 1.0f / 0.693147180559945f;  /* 1/ln(2) = log2(e) */
    uc0.f = 1.0f;
    uc1.f = 0.692850309695840f;
    uc2.f = 0.237504551482093f;
    uc3.f = 0.046751431261525f;

    /* Precompute HVX constant vectors */
    HVX_Vector vzero     = Q6_V_vzero();
    HVX_Vector voneh     = Q6_Vh_vsplat_R(0x3C00);         /* f16 1.0 */
    HVX_Vector vneginf_h = Q6_Vh_vsplat_R(0xFC00);         /* f16 -inf */
    HVX_Vector f0        = Q6_V_vsplat_R(uc0.i);
    HVX_Vector f1        = Q6_V_vsplat_R(uc1.i);
    HVX_Vector f2        = Q6_V_vsplat_R(uc2.i);
    HVX_Vector f3        = Q6_V_vsplat_R(uc3.i);
    HVX_Vector c7f800000 = Q6_V_vsplat_R(0x7f800000);      /* f32 exponent mask */
    HVX_Vector c807fffff = Q6_V_vsplat_R(0x807fffff);      /* f32 sign+mantissa mask */
    HVX_Vector vbeta     = Q6_Vqf32_vadd_VsfVsf(Q6_V_vsplat_R(scaleu.i), vzero); /* 1/ln(2) in qf32 */
    HVX_Vector c126      = Q6_V_vsplat_R(126 << 23);
    HVX_Vector c1w       = Q6_V_vsplat_R(1 << 23);
    HVX_Vector c2w       = Q6_Vw_vadd_VwVw(c1w, c1w);
    HVX_Vector c3w       = Q6_Vw_vadd_VwVw(c2w, c1w);
    HVX_Vector c4w       = Q6_Vw_vadd_VwVw(c3w, c1w);
    HVX_Vector c5w       = Q6_Vw_vadd_VwVw(c4w, c1w);

    /* Mask: set elements [NET_OUTPUT_DIM..63] to -inf.
     * Q6_Q_vsetq2_R(N) sets predicate bits for bytes [0..N-1].
     * Each f16 = 2 bytes, so N = NET_OUTPUT_DIM * 2. */
    HVX_VectorPred active_mask = Q6_Q_vsetq2_R(NET_OUTPUT_DIM * 2);

    float loss_acc = 0.0f;
    int correct_acc = 0;

    for (int b = 0; b < batch_size; b++) {
        const HVX_Vector *inp = (const HVX_Vector *)(logits + b * NET_OUTPUT_DIM_PAD);
        HVX_Vector *outp      = (HVX_Vector *)(probs + b * NET_OUTPUT_DIM_PAD);

        /* --- Load and mask --- */
        HVX_Vector x = *inp;
        x = Q6_V_vmux_QVV(active_mask, x, vneginf_h);

        /* --- Find row max via shuffle-reduce (6 rounds for 64 f16 elements) --- */
        HVX_Vector xmax = x;
        {
            int nshift = 2;  /* start at 2 bytes = 1 f16 element */
            for (int i = 0; i < 6; i++) {
                HVX_VectorPair temps = Q6_W_vshuff_VVR(xmax, xmax, nshift);
                xmax = Q6_Vhf_vmax_VhfVhf(Q6_V_lo_W(temps), Q6_V_hi_W(temps));
                nshift <<= 1;
            }
        }
        /* xmax now has the row max splatted across all lanes */

        /* --- Subtract max and widen to qf32 pair --- */
        HVX_Vector xd = Q6_Vqf16_vsub_VhfVhf(x, xmax);
        /* Widen qf16 -> two qf32 vectors (lo = elements 0..31, hi = elements 32..63) */
        HVX_VectorPair xdiff = Q6_Wqf32_vmpy_Vqf16Vhf(xd, voneh);

        /* --- Process lo half (elements 0..31 as f32) --- */
        HVX_Vector x0 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(xdiff), vbeta);
        x0 = Q6_Vsf_equals_Vqf32(x0);
        /* Extract and limit exponent */
        HVX_Vector x0exp      = Q6_V_vand_VV(x0, c7f800000);
        HVX_Vector x0explimit = Q6_Vw_vmin_VwVw(x0exp, c126);
        x0exp                 = Q6_Vw_vsub_VwVw(x0exp, x0explimit);
        HVX_Vector x0norm     = Q6_V_vor_VV(Q6_V_vand_VV(c807fffff, x0), x0explimit);

        /* --- Process hi half (elements 32..63 as f32) --- */
        HVX_Vector x1 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(xdiff), vbeta);
        x1 = Q6_Vsf_equals_Vqf32(x1);
        HVX_Vector x1exp      = Q6_V_vand_VV(x1, c7f800000);
        HVX_Vector x1explimit = Q6_Vw_vmin_VwVw(x1exp, c126);
        x1exp                 = Q6_Vw_vsub_VwVw(x1exp, x1explimit);
        HVX_Vector x1norm     = Q6_V_vor_VV(Q6_V_vand_VV(c807fffff, x1), x1explimit);

        /* --- Polynomial: p = c0 + c1*m + c2*m^2 + c3*m^3  (Horner form) --- */
        /* Lo half */
        HVX_Vector p0 = Q6_Vqf32_vmpy_VsfVsf(x0norm, f3);
        p0 = Q6_Vqf32_vadd_Vqf32Vsf(p0, f2);
        x0norm = Q6_Vqf32_vadd_VsfVsf(x0norm, vzero);  /* promote to qf32 for multiply */
        p0 = Q6_Vqf32_vmpy_Vqf32Vqf32(p0, x0norm);
        p0 = Q6_Vqf32_vadd_Vqf32Vsf(p0, f1);
        p0 = Q6_Vqf32_vmpy_Vqf32Vqf32(p0, x0norm);
        p0 = Q6_Vqf32_vadd_Vqf32Vsf(p0, f0);

        /* Hi half */
        HVX_Vector p1 = Q6_Vqf32_vmpy_VsfVsf(x1norm, f3);
        p1 = Q6_Vqf32_vadd_Vqf32Vsf(p1, f2);
        x1norm = Q6_Vqf32_vadd_VsfVsf(x1norm, vzero);
        p1 = Q6_Vqf32_vmpy_Vqf32Vqf32(p1, x1norm);
        p1 = Q6_Vqf32_vadd_Vqf32Vsf(p1, f1);
        p1 = Q6_Vqf32_vmpy_Vqf32Vqf32(p1, x1norm);
        p1 = Q6_Vqf32_vadd_Vqf32Vsf(p1, f0);

        /* --- Repeated squaring to compute 2^n --- */
        /* Lo: p^2, p^4, ... p^64 */
        HVX_Vector p0_2  = Q6_Vqf32_vmpy_Vqf32Vqf32(p0, p0);
        p0_2             = Q6_Vqf32_vadd_Vqf32Vsf(p0_2, vzero);
        HVX_Vector p0_4  = Q6_Vqf32_vmpy_Vqf32Vqf32(p0_2, p0_2);
        p0_4             = Q6_Vqf32_vadd_Vqf32Vsf(p0_4, vzero);
        HVX_Vector p0_8  = Q6_Vqf32_vmpy_Vqf32Vqf32(p0_4, p0_4);
        p0_8             = Q6_Vqf32_vadd_Vqf32Vsf(p0_8, vzero);
        HVX_Vector p0_16 = Q6_Vqf32_vmpy_Vqf32Vqf32(p0_8, p0_8);
        p0_16            = Q6_Vqf32_vadd_Vqf32Vsf(p0_16, vzero);
        HVX_Vector p0_32 = Q6_Vqf32_vmpy_Vqf32Vqf32(p0_16, p0_16);
        p0_32            = Q6_Vqf32_vadd_Vqf32Vsf(p0_32, vzero);
        HVX_Vector p0_64 = Q6_Vqf32_vmpy_Vqf32Vqf32(p0_32, p0_32);

        /* Hi: p^2, p^4, ... p^64 */
        HVX_Vector p1_2  = Q6_Vqf32_vmpy_Vqf32Vqf32(p1, p1);
        p1_2             = Q6_Vqf32_vadd_Vqf32Vsf(p1_2, vzero);
        HVX_Vector p1_4  = Q6_Vqf32_vmpy_Vqf32Vqf32(p1_2, p1_2);
        p1_4             = Q6_Vqf32_vadd_Vqf32Vsf(p1_4, vzero);
        HVX_Vector p1_8  = Q6_Vqf32_vmpy_Vqf32Vqf32(p1_4, p1_4);
        p1_8             = Q6_Vqf32_vadd_Vqf32Vsf(p1_8, vzero);
        HVX_Vector p1_16 = Q6_Vqf32_vmpy_Vqf32Vqf32(p1_8, p1_8);
        p1_16            = Q6_Vqf32_vadd_Vqf32Vsf(p1_16, vzero);
        HVX_Vector p1_32 = Q6_Vqf32_vmpy_Vqf32Vqf32(p1_16, p1_16);
        p1_32            = Q6_Vqf32_vadd_Vqf32Vsf(p1_32, vzero);
        HVX_Vector p1_64 = Q6_Vqf32_vmpy_Vqf32Vqf32(p1_32, p1_32);

        /* --- Select correct power based on integer exponent via vmux --- */
        HVX_VectorPred q0, q1;

        q0 = Q6_Q_vcmp_eq_VwVw(x0exp, c1w);
        q1 = Q6_Q_vcmp_eq_VwVw(x1exp, c1w);
        p0 = Q6_V_vmux_QVV(q0, p0_2, p0);
        p1 = Q6_V_vmux_QVV(q1, p1_2, p1);

        q0 = Q6_Q_vcmp_eq_VwVw(x0exp, c2w);
        q1 = Q6_Q_vcmp_eq_VwVw(x1exp, c2w);
        p0 = Q6_V_vmux_QVV(q0, p0_4, p0);
        p1 = Q6_V_vmux_QVV(q1, p1_4, p1);

        q0 = Q6_Q_vcmp_eq_VwVw(x0exp, c3w);
        q1 = Q6_Q_vcmp_eq_VwVw(x1exp, c3w);
        p0 = Q6_V_vmux_QVV(q0, p0_8, p0);
        p1 = Q6_V_vmux_QVV(q1, p1_8, p1);

        q0 = Q6_Q_vcmp_eq_VwVw(x0exp, c4w);
        q1 = Q6_Q_vcmp_eq_VwVw(x1exp, c4w);
        p0 = Q6_V_vmux_QVV(q0, p0_16, p0);
        p1 = Q6_V_vmux_QVV(q1, p1_16, p1);

        q0 = Q6_Q_vcmp_eq_VwVw(x0exp, c5w);
        q1 = Q6_Q_vcmp_eq_VwVw(x1exp, c5w);
        p0 = Q6_V_vmux_QVV(q0, p0_32, p0);
        p1 = Q6_V_vmux_QVV(q1, p1_32, p1);

        q0 = Q6_Q_vcmp_gt_VwVw(x0exp, c5w);
        q1 = Q6_Q_vcmp_gt_VwVw(x1exp, c5w);
        p0 = Q6_V_vmux_QVV(q0, p0_64, p0);
        p1 = Q6_V_vmux_QVV(q1, p1_64, p1);

        /* p0, p1 now hold exp(x_scaled) in qf32 for lo and hi halves */

        /* --- Sum reduction in qf32 --- */
        /* Accumulate both halves into one vector */
        HVX_Vector vsumf = Q6_Vqf32_vadd_Vqf32Vqf32(p0, p1);
        {
            int nshift = 4;  /* start at 4 bytes = 1 f32 element */
            for (int i = 0; i < 5; i++) {
                HVX_VectorPair temps = Q6_W_vshuff_VVR(vsumf, vsumf, nshift);
                vsumf = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(temps), Q6_V_hi_W(temps));
                nshift <<= 1;
            }
        }
        vsumf = Q6_Vsf_equals_Vqf32(vsumf);

        /* --- Compute reciprocal and scale --- */
        union { float f; int32_t i; } sum_val, recip_val;
        sum_val.i = Q6_R_vextract_VR(vsumf, 0);
        recip_val.f = 1.0f / sum_val.f;
        HVX_Vector vrecip = Q6_Vqf32_vadd_VsfVsf(Q6_V_vsplat_R(recip_val.i), vzero);

        /* Scale: prob = exp_val * (1/sum) */
        HVX_Vector xl = Q6_Vqf32_vmpy_Vqf32Vqf32(
            Q6_Vqf32_vadd_Vqf32Vsf(p0, vzero), vrecip);
        HVX_Vector xh = Q6_Vqf32_vmpy_Vqf32Vqf32(
            Q6_Vqf32_vadd_Vqf32Vsf(p1, vzero), vrecip);

        /* Pack back to f16 */
        HVX_VectorPair scaled_pair = Q6_W_vcombine_VV(xh, xl);
        HVX_Vector prob_h = Q6_Vhf_equals_Wqf32(scaled_pair);
        *outp = prob_h;

        /* --- Extract loss and accuracy (scalar, only 10 classes) --- */
        uint8_t label = labels[b];
        const uint16_t *prob_ptr = probs + b * NET_OUTPUT_DIM_PAD;

        /* Cross-entropy loss: -log(prob[label]) */
        _Float16 p_correct_h;
        memcpy(&p_correct_h, &prob_ptr[label], sizeof(uint16_t));
        float p_correct = (float)p_correct_h;
        if (p_correct < 1e-7f) p_correct = 1e-7f;
        loss_acc -= fast_logf(p_correct);

        /* Argmax for accuracy */
        int argmax = 0;
        _Float16 max_p;
        memcpy(&max_p, &prob_ptr[0], sizeof(uint16_t));
        for (int j = 1; j < NET_OUTPUT_DIM; j++) {
            _Float16 pj;
            memcpy(&pj, &prob_ptr[j], sizeof(uint16_t));
            if (pj > max_p) {
                max_p = pj;
                argmax = j;
            }
        }
        if (argmax == (int)label) correct_acc++;
    }

    *total_loss = loss_acc / (float)batch_size;
    *total_correct = correct_acc;
}

#endif /* HVX_OPS_F16_H */
