/*
 * mnist_cpu_matmul.h -- CPU matmul implementations
 */

#ifndef MNIST_CPU_MATMUL_H
#define MNIST_CPU_MATMUL_H

#include "common/common.h"

/* ====================================================================
 * CPU matmul implementations
 * ==================================================================== */

/* C[m x n] = A[m x k] @ B[k x n] */
static void cpu_matmul_nn(int m, int n, int k,
                           float *C, const float *A, const float *B) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int p = 0; p < k; p++)
                acc += A[i * k + p] * B[p * n + j];
            C[i * n + j] = acc;
        }
    }
}

/* C[m x n] = A[m x k] @ B^T[n x k] */
static void cpu_matmul_nt(int m, int n, int k,
                           float *C, const float *A, const float *B) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int p = 0; p < k; p++)
                acc += A[i * k + p] * B[j * k + p];
            C[i * n + j] = acc;
        }
    }
}

/* C[m x n] += A^T[k x m]^T @ B[k x n] */
static void cpu_matmul_tn_acc(int m, int n, int k,
                               float *C, const float *A, const float *B) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int p = 0; p < k; p++)
                acc += A[p * m + i] * B[p * n + j];
            C[i * n + j] += acc;
        }
    }
}

/* CPU matmul dispatch (used as matmul_fn_t) */
static void cpu_matmul_dispatch(float *C, const float *A, const float *B,
                                int m, int n, int k, int transpose) {
    switch (transpose) {
        case 0: cpu_matmul_nn(m, n, k, C, A, B); break;
        case 1: cpu_matmul_nt(m, n, k, C, A, B); break;
        case 2: cpu_matmul_tn_acc(m, n, k, C, A, B); break;
    }
}

#endif /* MNIST_CPU_MATMUL_H */
