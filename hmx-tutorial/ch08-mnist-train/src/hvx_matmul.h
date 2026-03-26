/*
 * HVX f32 Matmul implementations (v75+)
 *
 * Extracted from mnist_train_dsp.c for readability.
 * All functions are static -- this header is intended to be included
 * by exactly one translation unit.
 */
#ifndef HVX_MATMUL_H
#define HVX_MATMUL_H

#include <hexagon_types.h>
#include <hexagon_protos.h>

#define HVX_FLOATS 32  /* 1024-bit HVX vector / 32-bit float */

/* Static scratch buffer for transposing B in matmul_nt.
 * Largest weight matrix in our training: W1[128x800] = 102400 floats = 400KB. */
#define MAX_SCRATCH (128 * 800)
static float g_scratch[MAX_SCRATCH] __attribute__((aligned(128)));

/*
 * C[m x n] = A[m x k] @ B[k x n]  (standard matmul, 4x HVX unrolled)
 *
 * Process 4 HVX vectors (128 floats) per inner iteration.
 * Each accumulator is independent so the pipeline can overlap
 * multiply+add across accumulators.
 */
static void matmul_nn(float *C, const float *A, const float *B,
                      uint32_t m, uint32_t n, uint32_t k) {
    const uint32_t VEC4 = HVX_FLOATS * 4;
    uint32_t n_vec4 = n & ~(VEC4 - 1);
    uint32_t n_vec  = n & ~(HVX_FLOATS - 1);

    for (uint32_t i = 0; i < m; i++) {
        const float *a_row = A + i * k;
        float *c_row = C + i * n;
        uint32_t j = 0;

        for (; j < n_vec4; j += VEC4) {
            HVX_Vector acc0 = Q6_V_vzero();
            HVX_Vector acc1 = Q6_V_vzero();
            HVX_Vector acc2 = Q6_V_vzero();
            HVX_Vector acc3 = Q6_V_vzero();
            for (uint32_t p = 0; p < k; p++) {
                float a_val = a_row[p];
                HVX_Vector a_splat = Q6_V_vsplat_R(*(int32_t *)&a_val);
                const float *b_ptr = B + p * n + j;
                HVX_Vector b0 = *(HVX_Vector *)(b_ptr);
                HVX_Vector b1 = *(HVX_Vector *)(b_ptr + HVX_FLOATS);
                HVX_Vector b2 = *(HVX_Vector *)(b_ptr + HVX_FLOATS * 2);
                HVX_Vector b3 = *(HVX_Vector *)(b_ptr + HVX_FLOATS * 3);
                acc0 = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, Q6_Vqf32_vmpy_VsfVsf(a_splat, b0));
                acc1 = Q6_Vqf32_vadd_Vqf32Vqf32(acc1, Q6_Vqf32_vmpy_VsfVsf(a_splat, b1));
                acc2 = Q6_Vqf32_vadd_Vqf32Vqf32(acc2, Q6_Vqf32_vmpy_VsfVsf(a_splat, b2));
                acc3 = Q6_Vqf32_vadd_Vqf32Vqf32(acc3, Q6_Vqf32_vmpy_VsfVsf(a_splat, b3));
            }
            *(HVX_Vector *)(c_row + j) = Q6_Vsf_equals_Vqf32(acc0);
            *(HVX_Vector *)(c_row + j + HVX_FLOATS) = Q6_Vsf_equals_Vqf32(acc1);
            *(HVX_Vector *)(c_row + j + HVX_FLOATS * 2) = Q6_Vsf_equals_Vqf32(acc2);
            *(HVX_Vector *)(c_row + j + HVX_FLOATS * 3) = Q6_Vsf_equals_Vqf32(acc3);
        }

        for (; j < n_vec; j += HVX_FLOATS) {
            HVX_Vector acc = Q6_V_vzero();
            for (uint32_t p = 0; p < k; p++) {
                float a_val = a_row[p];
                HVX_Vector a_splat = Q6_V_vsplat_R(*(int32_t *)&a_val);
                HVX_Vector b_vec = *(HVX_Vector *)(B + p * n + j);
                acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(a_splat, b_vec));
            }
            *(HVX_Vector *)(c_row + j) = Q6_Vsf_equals_Vqf32(acc);
        }

        for (; j < n; j++) {
            float acc = 0.0f;
            for (uint32_t p = 0; p < k; p++)
                acc += a_row[p] * B[p * n + j];
            c_row[j] = acc;
        }
    }
}

/*
 * C[m x n] += A[m x k] @ B[k x n]  (accumulating variant, 4x HVX unrolled)
 */
static void matmul_nn_acc(float *C, const float *A, const float *B,
                          uint32_t m, uint32_t n, uint32_t k) {
    const uint32_t VEC4 = HVX_FLOATS * 4;
    uint32_t n_vec4 = n & ~(VEC4 - 1);
    uint32_t n_vec  = n & ~(HVX_FLOATS - 1);
    HVX_Vector one = Q6_V_vsplat_R(0x3f800000);

    for (uint32_t i = 0; i < m; i++) {
        const float *a_row = A + i * k;
        float *c_row = C + i * n;
        uint32_t j = 0;

        for (; j < n_vec4; j += VEC4) {
            HVX_Vector acc0 = Q6_Vqf32_vmpy_VsfVsf(*(HVX_Vector *)(c_row + j), one);
            HVX_Vector acc1 = Q6_Vqf32_vmpy_VsfVsf(*(HVX_Vector *)(c_row + j + HVX_FLOATS), one);
            HVX_Vector acc2 = Q6_Vqf32_vmpy_VsfVsf(*(HVX_Vector *)(c_row + j + HVX_FLOATS * 2), one);
            HVX_Vector acc3 = Q6_Vqf32_vmpy_VsfVsf(*(HVX_Vector *)(c_row + j + HVX_FLOATS * 3), one);
            for (uint32_t p = 0; p < k; p++) {
                float a_val = a_row[p];
                HVX_Vector a_splat = Q6_V_vsplat_R(*(int32_t *)&a_val);
                const float *b_ptr = B + p * n + j;
                acc0 = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, Q6_Vqf32_vmpy_VsfVsf(a_splat, *(HVX_Vector *)(b_ptr)));
                acc1 = Q6_Vqf32_vadd_Vqf32Vqf32(acc1, Q6_Vqf32_vmpy_VsfVsf(a_splat, *(HVX_Vector *)(b_ptr + HVX_FLOATS)));
                acc2 = Q6_Vqf32_vadd_Vqf32Vqf32(acc2, Q6_Vqf32_vmpy_VsfVsf(a_splat, *(HVX_Vector *)(b_ptr + HVX_FLOATS * 2)));
                acc3 = Q6_Vqf32_vadd_Vqf32Vqf32(acc3, Q6_Vqf32_vmpy_VsfVsf(a_splat, *(HVX_Vector *)(b_ptr + HVX_FLOATS * 3)));
            }
            *(HVX_Vector *)(c_row + j) = Q6_Vsf_equals_Vqf32(acc0);
            *(HVX_Vector *)(c_row + j + HVX_FLOATS) = Q6_Vsf_equals_Vqf32(acc1);
            *(HVX_Vector *)(c_row + j + HVX_FLOATS * 2) = Q6_Vsf_equals_Vqf32(acc2);
            *(HVX_Vector *)(c_row + j + HVX_FLOATS * 3) = Q6_Vsf_equals_Vqf32(acc3);
        }

        for (; j < n_vec; j += HVX_FLOATS) {
            HVX_Vector qacc = Q6_Vqf32_vmpy_VsfVsf(*(HVX_Vector *)(c_row + j), one);
            for (uint32_t p = 0; p < k; p++) {
                float a_val = a_row[p];
                HVX_Vector a_splat = Q6_V_vsplat_R(*(int32_t *)&a_val);
                qacc = Q6_Vqf32_vadd_Vqf32Vqf32(qacc, Q6_Vqf32_vmpy_VsfVsf(a_splat, *(HVX_Vector *)(B + p * n + j)));
            }
            *(HVX_Vector *)(c_row + j) = Q6_Vsf_equals_Vqf32(qacc);
        }

        for (; j < n; j++) {
            float acc = c_row[j];
            for (uint32_t p = 0; p < k; p++)
                acc += a_row[p] * B[p * n + j];
            c_row[j] = acc;
        }
    }
}

/*
 * C[m x n] = A[m x k] @ B^T[n x k]  (transpose B then call matmul_nn)
 */
static void matmul_nt(float *C, const float *A, const float *B,
                      uint32_t m, uint32_t n, uint32_t k) {
    /* Transpose B[n x k] -> B_T[k x n] in scratch */
    float *B_T = g_scratch;
    for (uint32_t p = 0; p < k; p++)
        for (uint32_t j = 0; j < n; j++)
            B_T[p * n + j] = B[j * k + p];
    matmul_nn(C, A, B_T, m, n, k);
}

/*
 * C[m x n] += A^T[k x m] @ B[k x n]  (transpose A then call matmul_nn_acc)
 */
static void matmul_tn_acc(float *C, const float *A, const float *B,
                          uint32_t m, uint32_t n, uint32_t k) {
    /* Transpose A[k x m] -> A_T[m x k] in scratch */
    float *A_T = g_scratch;
    for (uint32_t i = 0; i < m; i++)
        for (uint32_t p = 0; p < k; p++)
            A_T[i * k + p] = A[p * m + i];
    matmul_nn_acc(C, A_T, B, m, n, k);
}

/*
 * C[m x n] = A^T[k x m] @ B[k x n]  (transpose A then call matmul_nn)
 */
static void matmul_tn(float *C, const float *A, const float *B,
                      uint32_t m, uint32_t n, uint32_t k) {
    /* Transpose A[k x m] -> A_T[m x k] in scratch */
    float *A_T = g_scratch;
    for (uint32_t i = 0; i < m; i++)
        for (uint32_t p = 0; p < k; p++)
            A_T[i * k + p] = A[p * m + i];
    matmul_nn(C, A_T, B, m, n, k);
}

/* Dispatch matmul based on transpose flag */
static void do_matmul(float *C, const float *A, const float *B,
                      uint32_t m, uint32_t n, uint32_t k,
                      uint32_t transpose, uint32_t accumulate) {
    switch (transpose) {
        case 0:  /* C = A @ B */
            if (accumulate) {
                matmul_nn_acc(C, A, B, m, n, k);
            } else {
                matmul_nn(C, A, B, m, n, k);
            }
            break;
        case 1:  /* C = A @ B^T */
            matmul_nt(C, A, B, m, n, k);
            break;
        case 2:  /* C += A^T @ B */
            matmul_tn_acc(C, A, B, m, n, k);
            break;
        default:
            FARF(ERROR, "Unknown transpose mode %u", transpose);
            break;
    }
}

#endif /* HVX_MATMUL_H */
