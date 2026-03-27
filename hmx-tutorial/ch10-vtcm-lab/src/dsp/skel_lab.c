/*
 * ch10: VTCM lab -- systematic experiments to understand VTCM behavior
 *
 * Uses dspqueue (persistent DSP thread) to keep VTCM alive across messages.
 * Experiments progress from trivial (write/read) to complex (training).
 */

#include <stdlib.h>
#include <string.h>

#include "dsp/skel_common.h"     /* mnist_train.h, dspqueue.h, HAP_farf, power */
#include "common/lab_protocol.h"

#include "HAP_compute_res.h"
#include <qurt.h>

/* HVX intrinsics */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#define HVX_FLOATS 32

/* ================================================================
 * VTCM layout: 4MB total
 *
 * Region A: Test area (64KB) -- for basic/persist/hvx tests
 * Region B: SGD weights (400KB) + gradients (400KB) + scratch
 * Region C: Training buffers (full network)
 * ================================================================ */

#define VTCM_SIZE       (4 * 1024 * 1024)
#define TEST_AREA_SIZE  (64 * 1024)         /* 16K floats */
#define TEST_FLOATS     (TEST_AREA_SIZE / 4)

/* Network dims (same as ch08/ch09) */
#define NET_INPUT_DIM_PAD  800
#define NET_HIDDEN_DIM     128
#define NET_OUTPUT_DIM     10
#define NET_OUTPUT_DIM_PAD 32
#define MAX_BATCH          256

#define W1_FLOATS  (NET_HIDDEN_DIM * NET_INPUT_DIM_PAD)  /* 102400 */
#define W2_FLOATS  (NET_OUTPUT_DIM_PAD * NET_HIDDEN_DIM) /* 4096 */

/* ================================================================ */

struct lab_context {
    dspqueue_t queue;

    /* VTCM resource */
    unsigned int vtcm_res_id;
    void        *vtcm_base;

    /* VTCM regions (bump-allocated) */
    float *test_area;       /* 16K floats for basic experiments */

    /* SGD experiment */
    float *sgd_w;           /* W1-sized weight buffer in VTCM */
    float *sgd_grad;        /* W1-sized gradient buffer in VTCM */

    /* Training experiment (full network in VTCM) */
    float *v_w1, *v_b1, *v_w2, *v_b2;
    float *v_dw1, *v_dw2;
    float *v_hidden, *v_logits;
    float *v_dhidden, *v_dlogits;
    float *v_hidden_pre, *v_probs;
    float *v_input;
    float *v_scratch;  /* matmul transpose scratch */

    /* Persistence test state */
    uint32_t persist_magic;  /* last written magic value */

    /* DDR pointers saved from TRAIN_INIT (for zero-REF sync) */
    void *ddr_w1, *ddr_b1, *ddr_w2, *ddr_b2;

    int vtcm_ready;
};


/* ================================================================
 * Bump allocator
 * ================================================================ */

static float *bump_alloc(uint8_t **bump, uint32_t bytes) {
    uintptr_t addr = (uintptr_t)*bump;
    addr = (addr + 127) & ~(uintptr_t)127;  /* 128-byte align */
    float *ptr = (float *)addr;
    *bump = (uint8_t *)(addr + bytes);
    return ptr;
}


/* ================================================================
 * FastRPC lifecycle
 * ================================================================ */

AEEResult mnist_train_open(const char *uri, remote_handle64 *handle) {
    return mnist_train_open_impl(uri, handle, sizeof(struct lab_context));
}

AEEResult mnist_train_close(remote_handle64 handle) {
    struct lab_context *ctx = (struct lab_context *)handle;
    if (!ctx) return AEE_EBADPARM;
    if (ctx->queue) return AEE_EITEMBUSY;
    if (ctx->vtcm_res_id) {
        HAP_compute_res_release(ctx->vtcm_res_id);
        FARF(HIGH, "VTCM released in close");
    }
    free(ctx);
    return AEE_SUCCESS;
}


/* ================================================================
 * VTCM allocation
 * ================================================================ */

static int alloc_vtcm(struct lab_context *ctx) {
    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_vtcm_param(&attr, VTCM_SIZE, 1);

    ctx->vtcm_res_id = HAP_compute_res_acquire(&attr, 100000);
    if (!ctx->vtcm_res_id) {
        FARF(ERROR, "VTCM acquire failed (%d bytes)", VTCM_SIZE);
        return AEE_ENOMEMORY;
    }
    ctx->vtcm_base = HAP_compute_res_attr_get_vtcm_ptr(&attr);
    FARF(HIGH, "VTCM acquired: %d KB at %p", VTCM_SIZE / 1024, ctx->vtcm_base);

    /* Bump-allocate all regions */
    uint8_t *bump = (uint8_t *)ctx->vtcm_base;

    /* Region A: test area */
    ctx->test_area = bump_alloc(&bump, TEST_AREA_SIZE);

    /* Region B: SGD experiment */
    ctx->sgd_w    = bump_alloc(&bump, W1_FLOATS * 4);
    ctx->sgd_grad = bump_alloc(&bump, W1_FLOATS * 4);

    /* Region C: training */
    ctx->v_w1         = bump_alloc(&bump, W1_FLOATS * 4);
    ctx->v_b1         = bump_alloc(&bump, NET_HIDDEN_DIM * 4);
    ctx->v_w2         = bump_alloc(&bump, W2_FLOATS * 4);
    ctx->v_b2         = bump_alloc(&bump, NET_OUTPUT_DIM_PAD * 4);
    ctx->v_dw1        = bump_alloc(&bump, W1_FLOATS * 4);
    ctx->v_dw2        = bump_alloc(&bump, W2_FLOATS * 4);
    ctx->v_hidden     = bump_alloc(&bump, MAX_BATCH * NET_HIDDEN_DIM * 4);
    ctx->v_logits     = bump_alloc(&bump, MAX_BATCH * NET_OUTPUT_DIM_PAD * 4);
    ctx->v_dhidden    = bump_alloc(&bump, MAX_BATCH * NET_HIDDEN_DIM * 4);
    ctx->v_dlogits    = bump_alloc(&bump, MAX_BATCH * NET_OUTPUT_DIM_PAD * 4);
    ctx->v_hidden_pre = bump_alloc(&bump, MAX_BATCH * NET_HIDDEN_DIM * 4);
    ctx->v_probs      = bump_alloc(&bump, MAX_BATCH * NET_OUTPUT_DIM_PAD * 4);
    ctx->v_input      = bump_alloc(&bump, MAX_BATCH * NET_INPUT_DIM_PAD * 4);
    ctx->v_scratch    = bump_alloc(&bump, W1_FLOATS * 4);

    uint32_t used = (uint32_t)((uintptr_t)bump - (uintptr_t)ctx->vtcm_base);
    FARF(HIGH, "VTCM bump: %u KB / %u KB", used / 1024, VTCM_SIZE / 1024);
    if (used > VTCM_SIZE) {
        FARF(ERROR, "VTCM overflow! need %u KB", used / 1024);
        HAP_compute_res_release(ctx->vtcm_res_id);
        ctx->vtcm_res_id = 0;
        return AEE_ENOMEMORY;
    }

    ctx->vtcm_ready = 1;
    return AEE_SUCCESS;
}


/* ================================================================
 * Experiment: VTCM basic write/read
 *
 * Write a known pattern (iter-derived) to test_area, read back
 * immediately. Tests that VTCM is physically accessible.
 * ================================================================ */

static void exp_vtcm_basic(struct lab_context *ctx, uint32_t iter,
                            uint32_t size, struct lab_rsp *rsp) {
    if (size > TEST_FLOATS) size = TEST_FLOATS;
    uint32_t errors = 0;

    /* Write pattern: each element = iter * 1000 + index */
    for (uint32_t i = 0; i < size; i++) {
        float val = (float)(iter * 1000 + i);
        ctx->test_area[i] = val;
    }

    /* Read back and verify */
    for (uint32_t i = 0; i < size; i++) {
        float expected = (float)(iter * 1000 + i);
        float got = ctx->test_area[i];
        if (got != expected) {
            if (errors < 3) {
                FARF(ERROR, "BASIC[%u] idx=%u expect=%.0f got=%.0f",
                     iter, i, expected, got);
            }
            errors++;
        }
    }

    rsp->op = OP_LAB_VTCM_BASIC;
    rsp->status = (errors == 0) ? 0 : 1;
    rsp->errors = errors;
    if (errors == 0) {
        FARF(HIGH, "BASIC[%u] PASS (%u floats)", iter, size);
    } else {
        FARF(ERROR, "BASIC[%u] FAIL %u/%u errors", iter, errors, size);
    }
}


/* ================================================================
 * Experiment: VTCM persistence across messages
 *
 * Message N writes a magic value. Message N+1 verifies it's still
 * there, then writes a new value. Tests VTCM data survives between
 * dspqueue callbacks.
 * ================================================================ */

static void exp_vtcm_persist(struct lab_context *ctx, uint32_t iter,
                              struct lab_rsp *rsp) {
    uint32_t errors = 0;

    if (iter == 0) {
        /* First message: just write */
        ctx->persist_magic = 0xCAFE0000;
        for (uint32_t i = 0; i < 1024; i++) {
            ctx->test_area[i] = (float)(ctx->persist_magic + i);
        }
        FARF(HIGH, "PERSIST[0] wrote initial pattern 0x%08x", ctx->persist_magic);
    } else {
        /* Verify previous write */
        for (uint32_t i = 0; i < 1024; i++) {
            float expected = (float)(ctx->persist_magic + i);
            float got = ctx->test_area[i];
            if (got != expected) {
                if (errors < 3) {
                    FARF(ERROR, "PERSIST[%u] idx=%u expect=%.0f got=%.0f",
                         iter, i, expected, got);
                }
                errors++;
            }
        }

        /* Write new pattern for next message */
        ctx->persist_magic = 0xCAFE0000 + iter * 1024;
        for (uint32_t i = 0; i < 1024; i++) {
            ctx->test_area[i] = (float)(ctx->persist_magic + i);
        }
    }

    rsp->op = OP_LAB_VTCM_PERSIST;
    rsp->status = (errors == 0) ? 0 : 1;
    rsp->errors = errors;
    if (iter > 0 && iter % 1000 == 0) {
        FARF(HIGH, "PERSIST[%u] %s", iter, errors ? "FAIL" : "PASS");
    }
}


/* ================================================================
 * Experiment: HVX write to VTCM, then scalar read
 *
 * Tests coherency: HVX vector stores -> scalar loads on same VTCM.
 * This is the pattern used by hvx_sgd_update (HVX writes weights)
 * followed by matmul transpose (scalar reads weights).
 * ================================================================ */

static void exp_hvx_write(struct lab_context *ctx, uint32_t iter,
                           uint32_t size, struct lab_rsp *rsp) {
    if (size > TEST_FLOATS) size = TEST_FLOATS;
    uint32_t errors = 0;

    /* HVX write: fill test_area with iter-based pattern */
    float pattern_val = (float)(iter + 1) * 3.14f;
    HVX_Vector v_pat = Q6_V_vsplat_R(*(int32_t *)&pattern_val);

    uint32_t n_vec = size & ~(HVX_FLOATS - 1);
    for (uint32_t i = 0; i < n_vec; i += HVX_FLOATS) {
        /* Add index offset: pattern_val + i */
        float base = (float)i;
        HVX_Vector v_base = Q6_V_vsplat_R(*(int32_t *)&base);
        HVX_Vector v_one = Q6_V_vsplat_R(0x3f800000);
        /* v_result = pattern_val + i (approximately, using qf32) */
        HVX_Vector v_result = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vadd_VsfVsf(v_pat, v_base));
        *(HVX_Vector *)(ctx->test_area + i) = v_result;
    }

    /* Scalar read back and verify */
    for (uint32_t i = 0; i < n_vec; i++) {
        /* Reconstruct expected value: qf32 of (pattern_val + base_for_this_vec) */
        uint32_t vec_idx = i / HVX_FLOATS;
        float base = (float)(vec_idx * HVX_FLOATS);
        /* qf32 add: may not be exact IEEE, so compare with tolerance */
        float expected = pattern_val + base;
        float got = ctx->test_area[i];
        float diff = got - expected;
        if (diff < 0) diff = -diff;
        if (diff > 0.01f * (expected < 0 ? -expected : expected) + 1e-6f) {
            if (errors < 3) {
                FARF(ERROR, "HVX_WRITE[%u] idx=%u expect=%.4f got=%.4f diff=%.4f",
                     iter, i, expected, got, diff);
            }
            errors++;
        }
    }

    rsp->op = OP_LAB_HVX_WRITE;
    rsp->status = (errors == 0) ? 0 : 1;
    rsp->errors = errors;
    if (iter % 100 == 0) {
        FARF(HIGH, "HVX_WRITE[%u] %s (%u floats checked)", iter,
             errors ? "FAIL" : "PASS", n_vec);
    }
}


/* ================================================================
 * Experiment: Scalar write to VTCM, then HVX read
 *
 * Tests the reverse: scalar stores -> HVX loads.
 * This is the pattern used by matmul transpose (scalar writes to
 * scratch in VTCM) -> matmul_nn (HVX reads from scratch).
 * ================================================================ */

static void exp_scalar_write(struct lab_context *ctx, uint32_t iter,
                              uint32_t size, struct lab_rsp *rsp) {
    if (size > TEST_FLOATS) size = TEST_FLOATS;
    uint32_t errors = 0;
    uint32_t n_vec = size & ~(HVX_FLOATS - 1);

    /* Scalar write pattern */
    float base_val = (float)(iter * 7 + 42);
    for (uint32_t i = 0; i < n_vec; i++) {
        ctx->test_area[i] = base_val + (float)i;
    }

    /* HVX read back and check via scalar comparison */
    for (uint32_t i = 0; i < n_vec; i += HVX_FLOATS) {
        HVX_Vector v = *(HVX_Vector *)(ctx->test_area + i);
        /* Store to stack for scalar comparison */
        float tmp[HVX_FLOATS] __attribute__((aligned(128)));
        *(HVX_Vector *)tmp = v;
        for (uint32_t j = 0; j < HVX_FLOATS; j++) {
            float expected = base_val + (float)(i + j);
            if (tmp[j] != expected) {
                if (errors < 3) {
                    FARF(ERROR, "SCALAR_WRITE[%u] idx=%u expect=%.1f got=%.1f",
                         iter, i + j, expected, tmp[j]);
                }
                errors++;
            }
        }
    }

    rsp->op = OP_LAB_SCALAR_WRITE;
    rsp->status = (errors == 0) ? 0 : 1;
    rsp->errors = errors;
    if (iter % 100 == 0) {
        FARF(HIGH, "SCALAR_WRITE[%u] %s (%u floats)", iter,
             errors ? "FAIL" : "PASS", n_vec);
    }
}


/* ================================================================
 * Experiment: HVX SGD on VTCM weights
 *
 * Simulates the core training pattern:
 *   w -= lr * grad  (HVX vectorized, both w and grad in VTCM)
 *
 * The ARM side maintains a CPU reference and compares after N iters.
 * ================================================================ */

static void exp_sgd_iter(struct lab_context *ctx, uint32_t iter,
                          float lr, struct lab_rsp *rsp) {
    /* On iter 0: initialize weights and gradients in VTCM */
    if (iter == 0) {
        for (uint32_t i = 0; i < W1_FLOATS; i++) {
            ctx->sgd_w[i] = (float)(i % 100) * 0.01f;
            ctx->sgd_grad[i] = 0.001f;  /* constant gradient */
        }
        FARF(HIGH, "SGD_ITER[0] initialized %u floats, lr=%.4f",
             W1_FLOATS, lr);
    }

    /* HVX SGD update: w -= lr * grad */
    HVX_Vector lr_vec = Q6_V_vsplat_R(*(int32_t *)&lr);
    HVX_Vector one = Q6_V_vsplat_R(0x3f800000);
    uint32_t n_vec = W1_FLOATS & ~(HVX_FLOATS - 1);

    for (uint32_t i = 0; i < n_vec; i += HVX_FLOATS) {
        HVX_Vector wv = *(HVX_Vector *)(ctx->sgd_w + i);
        HVX_Vector gv = *(HVX_Vector *)(ctx->sgd_grad + i);
        HVX_Vector prod = Q6_Vqf32_vmpy_VsfVsf(lr_vec, gv);
        HVX_Vector wq = Q6_Vqf32_vmpy_VsfVsf(wv, one);
        *(HVX_Vector *)(ctx->sgd_w + i) =
            Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(wq, prod));
    }

    rsp->op = OP_LAB_SGD_ITER;
    rsp->status = 0;
    rsp->errors = 0;
    if (iter % 1000 == 0) {
        FARF(HIGH, "SGD_ITER[%u] w[0]=%.6f w[99]=%.6f",
             iter, ctx->sgd_w[0], ctx->sgd_w[99]);
    }
}


/* ================================================================
 * Experiment: SGD verify -- copy weights to DDR buffer for ARM check
 * ================================================================ */

static void exp_sgd_verify(struct lab_context *ctx, uint32_t iter,
                            struct dspqueue_buffer *bufs, uint32_t n_bufs,
                            struct lab_rsp *rsp) {
    if (n_bufs < 1 || !bufs[0].ptr) {
        rsp->op = OP_LAB_SGD_VERIFY;
        rsp->status = AEE_EBADPARM;
        return;
    }

    /* Copy VTCM weights to DDR output buffer */
    memcpy(bufs[0].ptr, ctx->sgd_w, W1_FLOATS * 4);

    rsp->op = OP_LAB_SGD_VERIFY;
    rsp->status = 0;
    rsp->errors = 0;
    FARF(HIGH, "SGD_VERIFY[%u] copied %u KB to DDR", iter, W1_FLOATS * 4 / 1024);
}


/* ================================================================
 * Training experiment helpers
 *
 * Minimal matmul implementations (inline, no separate header needed).
 * Use v_scratch (VTCM) for transpose buffer.
 * ================================================================ */

static void lab_matmul_nn(float *C, const float *A, const float *B,
                           uint32_t m, uint32_t n, uint32_t k) {
    const uint32_t VEC4 = HVX_FLOATS * 4;
    uint32_t n_vec4 = n & ~(VEC4 - 1);

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
                acc0 = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, Q6_Vqf32_vmpy_VsfVsf(a_splat, *(HVX_Vector *)(b_ptr)));
                acc1 = Q6_Vqf32_vadd_Vqf32Vqf32(acc1, Q6_Vqf32_vmpy_VsfVsf(a_splat, *(HVX_Vector *)(b_ptr + HVX_FLOATS)));
                acc2 = Q6_Vqf32_vadd_Vqf32Vqf32(acc2, Q6_Vqf32_vmpy_VsfVsf(a_splat, *(HVX_Vector *)(b_ptr + HVX_FLOATS*2)));
                acc3 = Q6_Vqf32_vadd_Vqf32Vqf32(acc3, Q6_Vqf32_vmpy_VsfVsf(a_splat, *(HVX_Vector *)(b_ptr + HVX_FLOATS*3)));
            }
            *(HVX_Vector *)(c_row + j) = Q6_Vsf_equals_Vqf32(acc0);
            *(HVX_Vector *)(c_row + j + HVX_FLOATS) = Q6_Vsf_equals_Vqf32(acc1);
            *(HVX_Vector *)(c_row + j + HVX_FLOATS*2) = Q6_Vsf_equals_Vqf32(acc2);
            *(HVX_Vector *)(c_row + j + HVX_FLOATS*3) = Q6_Vsf_equals_Vqf32(acc3);
        }
        for (; j < n; j++) {
            float acc = 0.0f;
            for (uint32_t p = 0; p < k; p++)
                acc += a_row[p] * B[p * n + j];
            c_row[j] = acc;
        }
    }
}

/* C = A @ B^T, transpose B into v_scratch (VTCM) */
static void lab_matmul_nt(struct lab_context *ctx,
                           float *C, const float *A, const float *B,
                           uint32_t m, uint32_t n, uint32_t k) {
    float *B_T = ctx->v_scratch;
    for (uint32_t p = 0; p < k; p++)
        for (uint32_t j = 0; j < n; j++)
            B_T[p * n + j] = B[j * k + p];
    lab_matmul_nn(C, A, B_T, m, n, k);
}

/* C = A^T @ B, transpose A into v_scratch (VTCM) */
static void lab_matmul_tn(struct lab_context *ctx,
                           float *C, const float *A, const float *B,
                           uint32_t m, uint32_t n, uint32_t k) {
    float *A_T = ctx->v_scratch;
    for (uint32_t i = 0; i < m; i++)
        for (uint32_t p = 0; p < k; p++)
            A_T[i * k + p] = A[p * m + i];
    lab_matmul_nn(C, A_T, B, m, n, k);
}

/* C += A^T @ B */
static void lab_matmul_nn_acc(float *C, const float *A, const float *B,
                               uint32_t m, uint32_t n, uint32_t k) {
    HVX_Vector one = Q6_V_vsplat_R(0x3f800000);
    const uint32_t VEC4 = HVX_FLOATS * 4;
    uint32_t n_vec4 = n & ~(VEC4 - 1);

    for (uint32_t i = 0; i < m; i++) {
        const float *a_row = A + i * k;
        float *c_row = C + i * n;
        uint32_t j = 0;
        for (; j < n_vec4; j += VEC4) {
            HVX_Vector acc0 = Q6_Vqf32_vmpy_VsfVsf(*(HVX_Vector *)(c_row + j), one);
            HVX_Vector acc1 = Q6_Vqf32_vmpy_VsfVsf(*(HVX_Vector *)(c_row + j + HVX_FLOATS), one);
            HVX_Vector acc2 = Q6_Vqf32_vmpy_VsfVsf(*(HVX_Vector *)(c_row + j + HVX_FLOATS*2), one);
            HVX_Vector acc3 = Q6_Vqf32_vmpy_VsfVsf(*(HVX_Vector *)(c_row + j + HVX_FLOATS*3), one);
            for (uint32_t p = 0; p < k; p++) {
                float a_val = a_row[p];
                HVX_Vector a_splat = Q6_V_vsplat_R(*(int32_t *)&a_val);
                const float *b_ptr = B + p * n + j;
                acc0 = Q6_Vqf32_vadd_Vqf32Vqf32(acc0, Q6_Vqf32_vmpy_VsfVsf(a_splat, *(HVX_Vector *)(b_ptr)));
                acc1 = Q6_Vqf32_vadd_Vqf32Vqf32(acc1, Q6_Vqf32_vmpy_VsfVsf(a_splat, *(HVX_Vector *)(b_ptr + HVX_FLOATS)));
                acc2 = Q6_Vqf32_vadd_Vqf32Vqf32(acc2, Q6_Vqf32_vmpy_VsfVsf(a_splat, *(HVX_Vector *)(b_ptr + HVX_FLOATS*2)));
                acc3 = Q6_Vqf32_vadd_Vqf32Vqf32(acc3, Q6_Vqf32_vmpy_VsfVsf(a_splat, *(HVX_Vector *)(b_ptr + HVX_FLOATS*3)));
            }
            *(HVX_Vector *)(c_row + j) = Q6_Vsf_equals_Vqf32(acc0);
            *(HVX_Vector *)(c_row + j + HVX_FLOATS) = Q6_Vsf_equals_Vqf32(acc1);
            *(HVX_Vector *)(c_row + j + HVX_FLOATS*2) = Q6_Vsf_equals_Vqf32(acc2);
            *(HVX_Vector *)(c_row + j + HVX_FLOATS*3) = Q6_Vsf_equals_Vqf32(acc3);
        }
        for (; j < n; j++) {
            float acc = c_row[j];
            for (uint32_t p = 0; p < k; p++)
                acc += a_row[p] * B[p * n + j];
            c_row[j] = acc;
        }
    }
}

/* C = A^T @ B (non-accumulating), transpose via scratch */
/* Actually matmul_tn is non-accumulating, use separate function */

/* HVX elementwise ops (inlined to avoid header deps) */

static void lab_add_bias(float *out, const float *bias, uint32_t batch, uint32_t dim) {
    uint32_t dim_vec = dim & ~(HVX_FLOATS - 1);
    for (uint32_t b = 0; b < batch; b++) {
        float *row = out + b * dim;
        for (uint32_t j = 0; j < dim_vec; j += HVX_FLOATS) {
            HVX_Vector o = *(HVX_Vector *)(row + j);
            HVX_Vector bv = *(HVX_Vector *)(bias + j);
            *(HVX_Vector *)(row + j) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(o, bv));
        }
    }
}

static void lab_relu_forward(float *x, uint32_t n) {
    HVX_Vector zero = Q6_V_vzero();
    uint32_t n_vec = n & ~(HVX_FLOATS - 1);
    for (uint32_t i = 0; i < n_vec; i += HVX_FLOATS) {
        HVX_Vector v = *(HVX_Vector *)(x + i);
        HVX_VectorPred pos = Q6_Q_vcmp_gt_VwVw(v, Q6_V_vsplat_R(-1));
        *(HVX_Vector *)(x + i) = Q6_V_vmux_QVV(pos, v, zero);
    }
}

static void lab_relu_backward(float *dx, const float *pre_relu, uint32_t n) {
    HVX_Vector zero = Q6_V_vzero();
    uint32_t n_vec = n & ~(HVX_FLOATS - 1);
    for (uint32_t i = 0; i < n_vec; i += HVX_FLOATS) {
        HVX_Vector d = *(HVX_Vector *)(dx + i);
        HVX_Vector p = *(HVX_Vector *)(pre_relu + i);
        HVX_VectorPred mask = Q6_Q_vcmp_gt_VwVw(p, Q6_V_vsplat_R(-1));
        *(HVX_Vector *)(dx + i) = Q6_V_vmux_QVV(mask, d, zero);
    }
}

static float lab_expf(float x) {
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) return 3.4e38f;
    union { float f; int32_t i; } u;
    u.i = (int32_t)(12102203.0f * x + 1065353216.0f);
    return u.f;
}

static float lab_logf(float x) {
    if (x <= 0.0f) return -88.0f;
    union { float f; int32_t i; } u;
    u.f = x;
    return (u.i - 1065353216.0f) / 12102203.0f;
}

static float lab_softmax_ce(const float *logits, const uint8_t *labels,
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
            float val = lab_expf(row[j] - max_val);
            if (val < 1e-10f) val = 1e-10f;
            prob[j] = val;
            sum_exp += val;
        }
        for (int j = 0; j < NET_OUTPUT_DIM; j++)
            prob[j] /= sum_exp;
        for (int j = NET_OUTPUT_DIM; j < NET_OUTPUT_DIM_PAD; j++)
            prob[j] = 0.0f;
        float p = prob[labels[b]];
        if (p < 1e-7f) p = 1e-7f;
        total_loss += -lab_logf(p);
    }
    return total_loss / (float)batch;
}

static void lab_compute_dlogits(float *dlogits, const float *probs,
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

static void lab_bias_backward(float *db, const float *dout,
                                uint32_t batch, uint32_t dim) {
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

static void lab_sgd_update(float *w, const float *grad, float lr, uint32_t n) {
    HVX_Vector lr_vec = Q6_V_vsplat_R(*(int32_t *)&lr);
    HVX_Vector one = Q6_V_vsplat_R(0x3f800000);
    uint32_t n_vec = n & ~(HVX_FLOATS - 1);
    for (uint32_t i = 0; i < n_vec; i += HVX_FLOATS) {
        HVX_Vector wv = *(HVX_Vector *)(w + i);
        HVX_Vector gv = *(HVX_Vector *)(grad + i);
        HVX_Vector prod = Q6_Vqf32_vmpy_VsfVsf(lr_vec, gv);
        HVX_Vector wq = Q6_Vqf32_vmpy_VsfVsf(wv, one);
        *(HVX_Vector *)(w + i) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(wq, prod));
    }
    for (uint32_t i = n_vec; i < n; i++)
        w[i] -= lr * grad[i];
}


/* ================================================================
 * Training: init weights from DDR
 * ================================================================ */

static void exp_train_init(struct lab_context *ctx,
                            struct dspqueue_buffer *bufs, uint32_t n_bufs,
                            struct lab_rsp *rsp) {
    if (n_bufs < 4) {
        rsp->op = OP_LAB_TRAIN_INIT;
        rsp->status = AEE_EBADPARM;
        return;
    }

    /* bufs: [w1, b1, w2, b2] -- save DDR pointers for later SYNC */
    ctx->ddr_w1 = bufs[0].ptr;
    ctx->ddr_b1 = bufs[1].ptr;
    ctx->ddr_w2 = bufs[2].ptr;
    ctx->ddr_b2 = bufs[3].ptr;

    memcpy(ctx->v_w1, bufs[0].ptr, W1_FLOATS * 4);
    memcpy(ctx->v_b1, bufs[1].ptr, NET_HIDDEN_DIM * 4);
    memcpy(ctx->v_w2, bufs[2].ptr, W2_FLOATS * 4);
    memcpy(ctx->v_b2, bufs[3].ptr, NET_OUTPUT_DIM_PAD * 4);

    FARF(HIGH, "TRAIN_INIT: copied weights to VTCM, saved DDR ptrs (w1[0]=%.4f w2[0]=%.4f)",
         ctx->v_w1[0], ctx->v_w2[0]);

    rsp->op = OP_LAB_TRAIN_INIT;
    rsp->status = 0;
}


/* ================================================================
 * Training: one batch forward+backward+SGD (all VTCM)
 * ================================================================ */

static void exp_train_batch(struct lab_context *ctx, struct lab_req *req,
                             struct dspqueue_buffer *bufs, uint32_t n_bufs,
                             struct lab_rsp *rsp) {
    uint32_t bs = req->batch_size;
    float lr = req->learning_rate;

    if (n_bufs < 1 || !bufs[0].ptr) {
        rsp->op = OP_LAB_TRAIN_BATCH;
        rsp->status = AEE_EBADPARM;
        return;
    }

    /* Copy input DDR -> VTCM */
    memcpy(ctx->v_input, bufs[0].ptr, bs * NET_INPUT_DIM_PAD * sizeof(float));

    float *inp        = ctx->v_input;
    float *w1         = ctx->v_w1;
    float *b1         = ctx->v_b1;
    float *w2         = ctx->v_w2;
    float *b2         = ctx->v_b2;
    float *hidden     = ctx->v_hidden;
    float *logits     = ctx->v_logits;
    float *hidden_pre = ctx->v_hidden_pre;
    float *probs      = ctx->v_probs;
    float *dw1        = ctx->v_dw1;
    float *dw2        = ctx->v_dw2;
    float *dhidden    = ctx->v_dhidden;
    float *dlogits    = ctx->v_dlogits;

    /* Diagnostic: log w1[0] at key points */
    static uint32_t s_batch_num = 0;
    s_batch_num++;
    if (s_batch_num <= 3 || s_batch_num % 468 == 0 || s_batch_num % 468 == 1) {
        FARF(HIGH, "BATCH[%u] START w1[0]=%.6f w1[1]=%.6f w2[0]=%.6f b1[0]=%.6f",
             s_batch_num, w1[0], w1[1], w2[0], b1[0]);
    }

    /* Forward */
    lab_matmul_nt(ctx, hidden, inp, w1, bs, NET_HIDDEN_DIM, NET_INPUT_DIM_PAD);
    lab_add_bias(hidden, b1, bs, NET_HIDDEN_DIM);
    memcpy(hidden_pre, hidden, bs * NET_HIDDEN_DIM * sizeof(float));
    lab_relu_forward(hidden, bs * NET_HIDDEN_DIM);

    lab_matmul_nt(ctx, logits, hidden, w2, bs, NET_OUTPUT_DIM_PAD, NET_HIDDEN_DIM);
    lab_add_bias(logits, b2, bs, NET_OUTPUT_DIM_PAD);

    float loss = lab_softmax_ce(logits, req->labels, probs, bs);

    uint32_t correct = 0;
    for (uint32_t bi = 0; bi < bs; bi++) {
        const float *row = logits + bi * NET_OUTPUT_DIM_PAD;
        float mv = row[0]; int mj = 0;
        for (int j = 1; j < NET_OUTPUT_DIM; j++)
            if (row[j] > mv) { mv = row[j]; mj = j; }
        if (mj == req->labels[bi]) correct++;
    }

    /* Backward */
    lab_compute_dlogits(dlogits, probs, req->labels, bs);
    lab_matmul_tn(ctx, dw2, dlogits, hidden, NET_OUTPUT_DIM_PAD, NET_HIDDEN_DIM, bs);
    float db2_local[NET_OUTPUT_DIM_PAD] __attribute__((aligned(128)));
    lab_bias_backward(db2_local, dlogits, bs, NET_OUTPUT_DIM_PAD);

    lab_matmul_nn(dhidden, dlogits, w2, bs, NET_HIDDEN_DIM, NET_OUTPUT_DIM_PAD);
    lab_relu_backward(dhidden, hidden_pre, bs * NET_HIDDEN_DIM);
    lab_matmul_tn(ctx, dw1, dhidden, inp, NET_HIDDEN_DIM, NET_INPUT_DIM_PAD, bs);
    float db1_local[NET_HIDDEN_DIM] __attribute__((aligned(128)));
    lab_bias_backward(db1_local, dhidden, bs, NET_HIDDEN_DIM);

    /* SGD */
    lab_sgd_update(w1, dw1, lr, W1_FLOATS);
    for (int i = 0; i < NET_HIDDEN_DIM; i++) b1[i] -= lr * db1_local[i];
    lab_sgd_update(w2, dw2, lr, W2_FLOATS);
    for (int i = 0; i < NET_OUTPUT_DIM_PAD; i++) b2[i] -= lr * db2_local[i];

    if (s_batch_num <= 3 || s_batch_num % 468 == 0 || s_batch_num % 468 == 1) {
        FARF(HIGH, "BATCH[%u] AFTER_SGD w1[0]=%.6f w1[1]=%.6f loss=%.4f correct=%u",
             s_batch_num, w1[0], w1[1], loss, correct);
    }

    rsp->op = OP_LAB_TRAIN_BATCH;
    rsp->status = 0;
    rsp->errors = correct;
    rsp->detail = loss;
}


/* ================================================================
 * Training: sync weights VTCM -> DDR
 * ================================================================ */

static void exp_train_sync(struct lab_context *ctx,
                            struct dspqueue_buffer *bufs, uint32_t n_bufs,
                            struct lab_rsp *rsp) {
    /*
     * Copy VTCM weights to DDR using saved pointers from INIT.
     * NO dspqueue buffer REF/DEREF needed -- avoids the cache
     * maintenance that dspqueue does on REF/DEREF which corrupts VTCM.
     */
    if (!ctx->ddr_w1) {
        rsp->op = OP_LAB_TRAIN_SYNC;
        rsp->status = AEE_EBADSTATE;
        return;
    }

    memcpy(ctx->ddr_w1, ctx->v_w1, W1_FLOATS * 4);
    memcpy(ctx->ddr_b1, ctx->v_b1, NET_HIDDEN_DIM * 4);
    memcpy(ctx->ddr_w2, ctx->v_w2, W2_FLOATS * 4);
    memcpy(ctx->ddr_b2, ctx->v_b2, NET_OUTPUT_DIM_PAD * 4);

    /* Flush DSP L2 cache for DDR buffers so ARM can read fresh data */
    qurt_mem_cache_clean((qurt_addr_t)ctx->ddr_w1, W1_FLOATS * 4,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    qurt_mem_cache_clean((qurt_addr_t)ctx->ddr_b1, NET_HIDDEN_DIM * 4,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    qurt_mem_cache_clean((qurt_addr_t)ctx->ddr_w2, W2_FLOATS * 4,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    qurt_mem_cache_clean((qurt_addr_t)ctx->ddr_b2, NET_OUTPUT_DIM_PAD * 4,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

    /*
     * CRITICAL: memcpy above read VTCM data, which may have pulled
     * VTCM addresses into L2 cache. Invalidate the ENTIRE VTCM block
     * to prevent stale L2 hits on future scalar reads from VTCM.
     */
    qurt_mem_cache_clean((qurt_addr_t)ctx->vtcm_base, VTCM_SIZE,
                         QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);

    FARF(HIGH, "TRAIN_SYNC: copied VTCM->DDR via saved ptrs (w1[0]=%.4f)", ctx->v_w1[0]);

    rsp->op = OP_LAB_TRAIN_SYNC;
    rsp->status = 0;
}


/* ================================================================
 * Training: evaluate test batch on DSP (no VTCM->DDR copy needed)
 * ================================================================ */

static void exp_train_eval(struct lab_context *ctx, struct lab_req *req,
                            struct dspqueue_buffer *bufs, uint32_t n_bufs,
                            struct lab_rsp *rsp) {
    uint32_t bs = req->batch_size;
    if (n_bufs < 1 || !bufs[0].ptr || bs == 0) {
        rsp->op = OP_LAB_TRAIN_EVAL;
        rsp->status = AEE_EBADPARM;
        return;
    }

    /* Copy test batch input from DDR to VTCM (reuse v_input) */
    memcpy(ctx->v_input, bufs[0].ptr, bs * NET_INPUT_DIM_PAD * sizeof(float));

    /* Forward pass only (no backward/SGD) */
    float *inp    = ctx->v_input;
    float *hidden = ctx->v_hidden;
    float *logits = ctx->v_logits;

    lab_matmul_nt(ctx, hidden, inp, ctx->v_w1, bs, NET_HIDDEN_DIM, NET_INPUT_DIM_PAD);
    lab_add_bias(hidden, ctx->v_b1, bs, NET_HIDDEN_DIM);
    lab_relu_forward(hidden, bs * NET_HIDDEN_DIM);
    lab_matmul_nt(ctx, logits, hidden, ctx->v_w2, bs, NET_OUTPUT_DIM_PAD, NET_HIDDEN_DIM);
    lab_add_bias(logits, ctx->v_b2, bs, NET_OUTPUT_DIM_PAD);

    /* Count correct */
    uint32_t correct = 0;
    for (uint32_t bi = 0; bi < bs; bi++) {
        const float *row = logits + bi * NET_OUTPUT_DIM_PAD;
        float mv = row[0]; int mj = 0;
        for (int j = 1; j < NET_OUTPUT_DIM; j++)
            if (row[j] > mv) { mv = row[j]; mj = j; }
        if (mj == req->labels[bi]) correct++;
    }

    rsp->op = OP_LAB_TRAIN_EVAL;
    rsp->status = 0;
    rsp->errors = correct;  /* reuse errors field for correct count */
}


/* ================================================================
 * Experiment 7: Does memcpy from VTCM cause L2 stale reads?
 *
 * Step 1: HVX writes pattern A (value 1.0) to test_area
 * Step 2: scalar reads test_area[0..31] — should be 1.0  (PASS)
 * Step 3: memcpy(heap_buf, test_area, 64KB) — pulls VTCM into L2 cache
 * Step 4: HVX writes pattern B (value 2.0) to test_area
 * Step 5: scalar reads test_area[0..31] — if L2 stale, gets 1.0 instead of 2.0
 *
 * Returns: errors = number of scalar reads that returned wrong value
 *          detail = first bad value (should be 2.0, if stale returns 1.0)
 * ================================================================ */

static void exp_l2_stale_test(struct lab_context *ctx, struct lab_rsp *rsp) {
    uint32_t errors = 0;
    float first_bad = 0.0f;
    uint32_t n_floats = TEST_FLOATS;  /* 16K floats = 64KB */
    uint32_t n_vecs = n_floats / HVX_FLOATS;

    /* Step 1: HVX writes pattern A (1.0) */
    float val_a = 1.0f;
    HVX_Vector va = Q6_V_vsplat_R(*(int32_t *)&val_a);
    for (uint32_t i = 0; i < n_vecs; i++) {
        ((HVX_Vector *)ctx->test_area)[i] = va;
    }

    /* Step 2: scalar verify — should be 1.0 */
    uint32_t step2_errors = 0;
    for (uint32_t i = 0; i < 32; i++) {
        if (ctx->test_area[i] != val_a) step2_errors++;
    }
    if (step2_errors > 0) {
        FARF(ERROR, "L2_STALE step2 failed: %u errors (basic HVX->scalar broken)", step2_errors);
        rsp->op = OP_LAB_L2_STALE_TEST;
        rsp->status = 2;  /* infrastructure failure */
        rsp->errors = step2_errors;
        return;
    }

    /* Step 3: memcpy to heap — this is what pulls VTCM into L2 */
    static float heap_buf[16384] __attribute__((aligned(128)));
    memcpy(heap_buf, ctx->test_area, n_floats * sizeof(float));

    /* Step 4: HVX writes pattern B (2.0) */
    float val_b = 2.0f;
    HVX_Vector vb = Q6_V_vsplat_R(*(int32_t *)&val_b);
    for (uint32_t i = 0; i < n_vecs; i++) {
        ((HVX_Vector *)ctx->test_area)[i] = vb;
    }

    /* Step 5: scalar reads — should be 2.0, but if L2 is stale, gets 1.0 */
    for (uint32_t i = 0; i < n_floats; i++) {
        float got = ctx->test_area[i];
        if (got != val_b) {
            if (errors == 0) first_bad = got;
            errors++;
        }
    }

    rsp->op = OP_LAB_L2_STALE_TEST;
    rsp->status = (errors == 0) ? 0 : 1;
    rsp->errors = errors;
    rsp->detail = first_bad;

    FARF(HIGH, "L2_STALE: %s — %u/%u stale reads (first_bad=%.1f, expected=%.1f)",
         errors ? "STALE DETECTED" : "NO STALE", errors, n_floats, first_bad, val_b);
}


/* ================================================================
 * Experiment 8: Does cache invalidation fix the stale reads?
 *
 * Same as Exp 7, but between step 4 (HVX writes B) and step 5 (scalar reads),
 * we call qurt_mem_cache_clean(INVALIDATE) on the VTCM test area.
 * ================================================================ */

static void exp_l2_inval_test(struct lab_context *ctx, struct lab_rsp *rsp) {
    uint32_t errors = 0;
    float first_bad = 0.0f;
    uint32_t n_floats = TEST_FLOATS;
    uint32_t n_vecs = n_floats / HVX_FLOATS;

    /* Step 1: HVX writes pattern A (3.0) */
    float val_a = 3.0f;
    HVX_Vector va = Q6_V_vsplat_R(*(int32_t *)&val_a);
    for (uint32_t i = 0; i < n_vecs; i++) {
        ((HVX_Vector *)ctx->test_area)[i] = va;
    }

    /* Step 2: memcpy to heap */
    static float heap_buf2[16384] __attribute__((aligned(128)));
    memcpy(heap_buf2, ctx->test_area, n_floats * sizeof(float));

    /* Step 3: HVX writes pattern B (4.0) */
    float val_b = 4.0f;
    HVX_Vector vb = Q6_V_vsplat_R(*(int32_t *)&val_b);
    for (uint32_t i = 0; i < n_vecs; i++) {
        ((HVX_Vector *)ctx->test_area)[i] = vb;
    }

    /* Step 4: Cache invalidation */
    qurt_mem_cache_clean((qurt_addr_t)ctx->test_area, n_floats * sizeof(float),
                         QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);

    /* Step 5: scalar reads — should be 4.0 if invalidation worked */
    for (uint32_t i = 0; i < n_floats; i++) {
        float got = ctx->test_area[i];
        if (got != val_b) {
            if (errors == 0) first_bad = got;
            errors++;
        }
    }

    rsp->op = OP_LAB_L2_INVAL_TEST;
    rsp->status = (errors == 0) ? 0 : 1;
    rsp->errors = errors;
    rsp->detail = first_bad;

    FARF(HIGH, "L2_INVAL: %s — %u/%u stale reads after invalidate (first_bad=%.1f, expected=%.1f)",
         errors ? "INVALIDATION FAILED" : "INVALIDATION WORKED", errors, n_floats, first_bad, val_b);
}


/* ================================================================
 * Experiment 9: SYNC with L2 invalidation
 *
 * Same as exp_train_sync but after memcpy, invalidate the entire VTCM
 * allocation to flush stale L2 lines.
 * ================================================================ */

static void exp_train_sync_v2(struct lab_context *ctx,
                                struct dspqueue_buffer *bufs, uint32_t n_bufs,
                                struct lab_rsp *rsp) {
    if (!ctx->ddr_w1) {
        rsp->op = OP_LAB_TRAIN_SYNC_V2;
        rsp->status = AEE_EBADSTATE;
        return;
    }

    /* Copy VTCM → DDR */
    memcpy(ctx->ddr_w1, ctx->v_w1, W1_FLOATS * 4);
    memcpy(ctx->ddr_b1, ctx->v_b1, NET_HIDDEN_DIM * 4);
    memcpy(ctx->ddr_w2, ctx->v_w2, W2_FLOATS * 4);
    memcpy(ctx->ddr_b2, ctx->v_b2, NET_OUTPUT_DIM_PAD * 4);

    /* Flush DDR so ARM can read */
    qurt_mem_cache_clean((qurt_addr_t)ctx->ddr_w1, W1_FLOATS * 4,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    qurt_mem_cache_clean((qurt_addr_t)ctx->ddr_b1, NET_HIDDEN_DIM * 4,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    qurt_mem_cache_clean((qurt_addr_t)ctx->ddr_w2, W2_FLOATS * 4,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    qurt_mem_cache_clean((qurt_addr_t)ctx->ddr_b2, NET_OUTPUT_DIM_PAD * 4,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

    /* Invalidate L2 for ALL VTCM allocations (not just weights).
     * memcpy above may have pulled VTCM into L2; invalidate ensures
     * future scalar reads go back to VTCM physical memory. */
    qurt_mem_cache_clean((qurt_addr_t)ctx->vtcm_base, VTCM_SIZE,
                         QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);

    FARF(HIGH, "TRAIN_SYNC_V2: VTCM->DDR + full L2 invalidate (w1[0]=%.4f)", ctx->v_w1[0]);

    rsp->op = OP_LAB_TRAIN_SYNC_V2;
    rsp->status = 0;
}


/* ================================================================
 * Experiment 12a: SYNC memcpy only (no cache ops at all)
 * ================================================================ */

static void exp_train_sync_mconly(struct lab_context *ctx,
                                   struct dspqueue_buffer *bufs, uint32_t n_bufs,
                                   struct lab_rsp *rsp) {
    if (!ctx->ddr_w1) {
        rsp->op = OP_LAB_TRAIN_SYNC_MCONLY;
        rsp->status = AEE_EBADSTATE;
        return;
    }

    /* Snapshot w1[0] BEFORE memcpy */
    float w1_before = ctx->v_w1[0];

    /* Copy VTCM → DSP HEAP (not shared DDR!) to test if DDR destination is the issue */
    static float *heap_w1 = NULL;
    if (!heap_w1) heap_w1 = (float *)malloc(W1_FLOATS * 4);
    if (heap_w1) {
        memcpy(heap_w1, ctx->v_w1, W1_FLOATS * 4);
    }

    /* Check w1[0] AFTER memcpy to heap */
    float w1_after = ctx->v_w1[0];

    FARF(HIGH, "SYNC_MCONLY(heap): w1_before=%.6f w1_after=%.6f %s",
         w1_before, w1_after,
         (w1_before != w1_after) ? "*** CORRUPTED ***" : "OK");

    rsp->op = OP_LAB_TRAIN_SYNC_MCONLY;
    rsp->status = 0;
    rsp->detail = w1_after;
    rsp->errors = (w1_before != w1_after) ? 1 : 0;
}


/* ================================================================
 * Experiment 12b: SYNC memcpy + flush DDR (no VTCM invalidate)
 * ================================================================ */

static void exp_train_sync_flush(struct lab_context *ctx,
                                  struct dspqueue_buffer *bufs, uint32_t n_bufs,
                                  struct lab_rsp *rsp) {
    if (!ctx->ddr_w1) {
        rsp->op = OP_LAB_TRAIN_SYNC_FLUSH;
        rsp->status = AEE_EBADSTATE;
        return;
    }

    memcpy(ctx->ddr_w1, ctx->v_w1, W1_FLOATS * 4);
    memcpy(ctx->ddr_b1, ctx->v_b1, NET_HIDDEN_DIM * 4);
    memcpy(ctx->ddr_w2, ctx->v_w2, W2_FLOATS * 4);
    memcpy(ctx->ddr_b2, ctx->v_b2, NET_OUTPUT_DIM_PAD * 4);

    /* Flush DDR only, DO NOT touch VTCM cache */
    qurt_mem_cache_clean((qurt_addr_t)ctx->ddr_w1, W1_FLOATS * 4,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    qurt_mem_cache_clean((qurt_addr_t)ctx->ddr_b1, NET_HIDDEN_DIM * 4,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    qurt_mem_cache_clean((qurt_addr_t)ctx->ddr_w2, W2_FLOATS * 4,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    qurt_mem_cache_clean((qurt_addr_t)ctx->ddr_b2, NET_OUTPUT_DIM_PAD * 4,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

    FARF(HIGH, "SYNC_FLUSH: memcpy + DDR flush (w1[0]=%.4f b1[0]=%.4f)", ctx->v_w1[0], ctx->v_b1[0]);

    rsp->op = OP_LAB_TRAIN_SYNC_FLUSH;
    rsp->status = 0;
}


/* ================================================================
 * Experiment 12c: SYNC no-op — just verify w1[0], no memcpy at all
 * ================================================================ */

static void exp_train_sync_noop(struct lab_context *ctx,
                                 struct lab_rsp *rsp) {
    rsp->op = OP_LAB_TRAIN_SYNC_NOOP;
    rsp->status = 0;
    rsp->detail = ctx->v_w1 ? ctx->v_w1[0] : -999.0f;
    rsp->errors = 0;

    FARF(HIGH, "SYNC_NOOP: w1[0]=%.6f (no memcpy)", rsp->detail);
}


/* ================================================================
 * Experiment 12d: SYNC memcpy VTCM→DDR (shared memory destination)
 * ================================================================ */

static void exp_train_sync_ddr(struct lab_context *ctx,
                                struct dspqueue_buffer *bufs, uint32_t n_bufs,
                                struct lab_rsp *rsp) {
    if (!ctx->ddr_w1) {
        rsp->op = OP_LAB_TRAIN_SYNC_DDR;
        rsp->status = AEE_EBADSTATE;
        return;
    }

    float w1_before = ctx->v_w1[0];

    /* Copy VTCM → DDR (shared rpcmem) */
    memcpy(ctx->ddr_w1, ctx->v_w1, W1_FLOATS * 4);
    memcpy(ctx->ddr_b1, ctx->v_b1, NET_HIDDEN_DIM * 4);
    memcpy(ctx->ddr_w2, ctx->v_w2, W2_FLOATS * 4);
    memcpy(ctx->ddr_b2, ctx->v_b2, NET_OUTPUT_DIM_PAD * 4);

    float w1_after = ctx->v_w1[0];

    FARF(HIGH, "SYNC_DDR: w1_before=%.6f w1_after=%.6f %s",
         w1_before, w1_after,
         (w1_before != w1_after) ? "*** CORRUPTED ***" : "OK");

    rsp->op = OP_LAB_TRAIN_SYNC_DDR;
    rsp->status = 0;
    rsp->detail = w1_after;
    rsp->errors = (w1_before != w1_after) ? 1 : 0;
}


/* ================================================================
 * Experiment 12e: SYNC only w2+b2 (16KB) to test size dependency
 * ================================================================ */

static void exp_train_sync_small(struct lab_context *ctx,
                                  struct dspqueue_buffer *bufs, uint32_t n_bufs,
                                  struct lab_rsp *rsp) {
    float w1_before = ctx->v_w1[0];

    /* Only copy w2 (16KB) and b2 (128B) — skip the large w1 (400KB) */
    static float heap_w2[4096] __attribute__((aligned(128)));
    static float heap_b2[32] __attribute__((aligned(128)));
    memcpy(heap_w2, ctx->v_w2, W2_FLOATS * 4);
    memcpy(heap_b2, ctx->v_b2, NET_OUTPUT_DIM_PAD * 4);

    float w1_after = ctx->v_w1[0];

    FARF(HIGH, "SYNC_SMALL: w1_before=%.6f w1_after=%.6f (copied 16KB only)",
         w1_before, w1_after);

    rsp->op = OP_LAB_TRAIN_SYNC_SMALL;
    rsp->status = 0;
    rsp->detail = w1_after;
    rsp->errors = 0;
}


/* ================================================================
 * Experiment 13a: SYNC no-op but ARM will do cpu_evaluate
 * Identical to NOOP on DSP side; ARM forces cpu_evaluate anyway.
 * ================================================================ */

static void exp_train_sync_noop_eval(struct lab_context *ctx,
                                      struct lab_rsp *rsp) {
    rsp->op = OP_LAB_TRAIN_SYNC_NOOP_EVAL;
    rsp->status = 0;
    rsp->detail = ctx->v_w1 ? ctx->v_w1[0] : -999.0f;
    rsp->errors = 0;

    FARF(HIGH, "SYNC_NOOP_EVAL: w1[0]=%.6f (ARM will do cpu_evaluate)", rsp->detail);
}


/* ================================================================
 * Experiment 13b: SYNC memcpy + DSP busy-work (keep DSP active)
 *
 * Does the same memcpy as MCONLY, then runs dummy matmuls to keep
 * the DSP thread busy for ~5ms (simulating the time ARM cpu_evaluate
 * takes).  If corruption disappears, it proves the root cause is
 * DSP idle time, not memcpy or L2 cache effects.
 * ================================================================ */

static void exp_train_sync_dspeval(struct lab_context *ctx,
                                    struct lab_rsp *rsp) {
    if (!ctx->ddr_w1) {
        rsp->op = OP_LAB_TRAIN_SYNC_DSPEVAL;
        rsp->status = AEE_EBADSTATE;
        return;
    }

    /* Same memcpy as MCONLY */
    static float *heap_w1 = NULL;
    if (!heap_w1) heap_w1 = (float *)malloc(W1_FLOATS * 4);
    if (heap_w1) {
        memcpy(heap_w1, ctx->v_w1, W1_FLOATS * sizeof(float));
    }

    /* Keep DSP busy: run forward pass on dummy data using VTCM weights.
     * This simulates the time ARM cpu_evaluate would take, but keeps
     * DSP thread active (preventing VTCM resource preemption). */
    float *dummy_in  = ctx->v_input;   /* reuse input buffer */
    float *dummy_out = ctx->v_hidden;  /* reuse hidden buffer */
    /* Run 50 matmuls (each ~0.1ms) ~ 5ms of DSP work */
    for (int i = 0; i < 50; i++) {
        lab_matmul_nt(ctx, dummy_out, dummy_in, ctx->v_w1,
                      128, NET_HIDDEN_DIM, NET_INPUT_DIM_PAD);
    }

    FARF(HIGH, "SYNC_DSPEVAL: memcpy + 50 dummy matmuls (w1[0]=%.4f)", ctx->v_w1[0]);

    rsp->op = OP_LAB_TRAIN_SYNC_DSPEVAL;
    rsp->status = 0;
    rsp->detail = ctx->v_w1[0];
}


/* ================================================================
 * Experiment 10a-d: Incremental accumulation tests
 *
 * Ultra-simple: HVX adds 1.0 to a VTCM buffer each message.
 * Scalar verifies the running total. Tests whether various cache
 * operations between increments cause stale reads.
 * ================================================================ */

static void exp_increment(struct lab_context *ctx, uint32_t iter,
                           struct lab_rsp *rsp) {
    uint32_t n_floats = 4096;
    uint32_t n_vecs = n_floats / HVX_FLOATS;

    if (iter == 0) {
        HVX_Vector zero = Q6_V_vzero();
        for (uint32_t i = 0; i < n_vecs; i++)
            ((HVX_Vector *)ctx->test_area)[i] = zero;
    }

    /* HVX: add 1.0 to every element */
    float one_f = 1.0f;
    HVX_Vector one_v = Q6_V_vsplat_R(*(int32_t *)&one_f);
    for (uint32_t i = 0; i < n_vecs; i++) {
        HVX_Vector v = ((HVX_Vector *)ctx->test_area)[i];
        HVX_Vector result = Q6_Vqf32_vadd_VsfVsf(v, one_v);
        ((HVX_Vector *)ctx->test_area)[i] = Q6_Vsf_equals_Vqf32(result);
    }

    /* Scalar verify: each element should be (iter + 1) */
    float expected = (float)(iter + 1);
    uint32_t errors = 0;
    float first_bad = 0.0f;
    for (uint32_t i = 0; i < n_floats; i++) {
        float got = ctx->test_area[i];
        float diff = got - expected;
        if (diff < 0) diff = -diff;
        if (diff > 0.01f) {
            if (errors == 0) first_bad = got;
            errors++;
        }
    }

    rsp->op = OP_LAB_INCREMENT;
    rsp->status = (errors == 0) ? 0 : 1;
    rsp->errors = errors;
    rsp->detail = first_bad;

    if (iter % 100 == 0 || errors > 0) {
        FARF(HIGH, "INCREMENT[%u] expected=%.1f errors=%u first_bad=%.4f",
             iter, expected, errors, first_bad);
    }
}

static void exp_increment_sync(struct lab_context *ctx, uint32_t iter,
                                struct lab_rsp *rsp) {
    uint32_t n_floats = 4096;
    uint32_t n_vecs = n_floats / HVX_FLOATS;

    if (iter == 0) {
        HVX_Vector zero = Q6_V_vzero();
        for (uint32_t i = 0; i < n_vecs; i++)
            ((HVX_Vector *)ctx->test_area)[i] = zero;
    }

    float one_f = 1.0f;
    HVX_Vector one_v = Q6_V_vsplat_R(*(int32_t *)&one_f);
    for (uint32_t i = 0; i < n_vecs; i++) {
        HVX_Vector v = ((HVX_Vector *)ctx->test_area)[i];
        ((HVX_Vector *)ctx->test_area)[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, one_v));
    }

    /* Simulate SYNC: memcpy VTCM → heap */
    static float sync_buf[4096] __attribute__((aligned(128)));
    memcpy(sync_buf, ctx->test_area, n_floats * sizeof(float));

    /* INVALIDATE L2 for test area */
    qurt_mem_cache_clean((qurt_addr_t)ctx->test_area, n_floats * sizeof(float),
                         QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);

    float expected = (float)(iter + 1);
    uint32_t errors = 0;
    float first_bad = 0.0f;
    for (uint32_t i = 0; i < n_floats; i++) {
        float got = ctx->test_area[i];
        float diff = got - expected;
        if (diff < 0) diff = -diff;
        if (diff > 0.01f) {
            if (errors == 0) first_bad = got;
            errors++;
        }
    }

    rsp->op = OP_LAB_INCREMENT_SYNC;
    rsp->status = (errors == 0) ? 0 : 1;
    rsp->errors = errors;
    rsp->detail = first_bad;

    if (iter % 100 == 0 || errors > 0) {
        FARF(HIGH, "INCREMENT_SYNC[%u] expected=%.1f errors=%u first_bad=%.4f",
             iter, expected, errors, first_bad);
    }
}

static void exp_increment_flush(struct lab_context *ctx, uint32_t iter,
                                 struct lab_rsp *rsp) {
    uint32_t n_floats = 4096;
    uint32_t n_vecs = n_floats / HVX_FLOATS;

    if (iter == 0) {
        HVX_Vector zero = Q6_V_vzero();
        for (uint32_t i = 0; i < n_vecs; i++)
            ((HVX_Vector *)ctx->test_area)[i] = zero;
    }

    float one_f = 1.0f;
    HVX_Vector one_v = Q6_V_vsplat_R(*(int32_t *)&one_f);
    for (uint32_t i = 0; i < n_vecs; i++) {
        HVX_Vector v = ((HVX_Vector *)ctx->test_area)[i];
        ((HVX_Vector *)ctx->test_area)[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, one_v));
    }

    static float sync_buf3[4096] __attribute__((aligned(128)));
    memcpy(sync_buf3, ctx->test_area, n_floats * sizeof(float));

    /* FLUSH_INVALIDATE instead of just INVALIDATE */
    qurt_mem_cache_clean((qurt_addr_t)ctx->test_area, n_floats * sizeof(float),
                         QURT_MEM_CACHE_FLUSH_INVALIDATE, QURT_MEM_DCACHE);

    float expected = (float)(iter + 1);
    uint32_t errors = 0;
    float first_bad = 0.0f;
    for (uint32_t i = 0; i < n_floats; i++) {
        float got = ctx->test_area[i];
        float diff = got - expected;
        if (diff < 0) diff = -diff;
        if (diff > 0.01f) {
            if (errors == 0) first_bad = got;
            errors++;
        }
    }

    rsp->op = OP_LAB_INCREMENT_FLUSH;
    rsp->status = (errors == 0) ? 0 : 1;
    rsp->errors = errors;
    rsp->detail = first_bad;

    if (iter % 100 == 0 || errors > 0) {
        FARF(HIGH, "INCREMENT_FLUSH[%u] expected=%.1f errors=%u first_bad=%.4f",
             iter, expected, errors, first_bad);
    }
}

static void exp_increment_nocache(struct lab_context *ctx, uint32_t iter,
                                   struct lab_rsp *rsp) {
    uint32_t n_floats = 4096;
    uint32_t n_vecs = n_floats / HVX_FLOATS;

    if (iter == 0) {
        HVX_Vector zero = Q6_V_vzero();
        for (uint32_t i = 0; i < n_vecs; i++)
            ((HVX_Vector *)ctx->test_area)[i] = zero;
    }

    float one_f = 1.0f;
    HVX_Vector one_v = Q6_V_vsplat_R(*(int32_t *)&one_f);
    for (uint32_t i = 0; i < n_vecs; i++) {
        HVX_Vector v = ((HVX_Vector *)ctx->test_area)[i];
        ((HVX_Vector *)ctx->test_area)[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, one_v));
    }

    static float sync_buf4[4096] __attribute__((aligned(128)));
    memcpy(sync_buf4, ctx->test_area, n_floats * sizeof(float));

    /* NO cache operations at all */

    float expected = (float)(iter + 1);
    uint32_t errors = 0;
    float first_bad = 0.0f;
    for (uint32_t i = 0; i < n_floats; i++) {
        float got = ctx->test_area[i];
        float diff = got - expected;
        if (diff < 0) diff = -diff;
        if (diff > 0.01f) {
            if (errors == 0) first_bad = got;
            errors++;
        }
    }

    rsp->op = OP_LAB_INCREMENT_NOCACHE;
    rsp->status = (errors == 0) ? 0 : 1;
    rsp->errors = errors;
    rsp->detail = first_bad;

    if (iter % 100 == 0 || errors > 0) {
        FARF(HIGH, "INCREMENT_NOCACHE[%u] expected=%.1f errors=%u first_bad=%.4f",
             iter, expected, errors, first_bad);
    }
}


/* ================================================================
 * Experiment 11: Scalar write VTCM → INVALIDATE → scalar read
 *
 * Tests the theory: scalar writes to VTCM go through L2 (write-back).
 * If INVALIDATE drops dirty L2 lines, the scalar-written data is LOST.
 *
 * Step 1: Scalar write pattern A to test_area (creates dirty L2 lines)
 * Step 2: INVALIDATE the test_area range in L2
 * Step 3: Scalar read test_area — if L2 was write-back, data is lost
 *
 * Compare with Exp 3 (HVX write → scalar read, no INVALIDATE) which PASSES.
 * ================================================================ */

static void exp_scalar_write_inval(struct lab_context *ctx, uint32_t iter,
                                    struct lab_rsp *rsp) {
    uint32_t n_floats = 4096;  /* 16KB */

    /* Step 1: Scalar write unique pattern to test_area */
    float base = (float)(iter + 1) * 100.0f;
    for (uint32_t i = 0; i < n_floats; i++) {
        ctx->test_area[i] = base + (float)(i % 100);
    }

    /* Step 2: INVALIDATE L2 for test_area range */
    qurt_mem_cache_clean((qurt_addr_t)ctx->test_area, n_floats * sizeof(float),
                         QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);

    /* Step 3: Scalar read back — should be same values if writes went to VTCM */
    uint32_t errors = 0;
    float first_bad = 0.0f;
    float first_expected = 0.0f;
    for (uint32_t i = 0; i < n_floats; i++) {
        float expected = base + (float)(i % 100);
        float got = ctx->test_area[i];
        float diff = got - expected;
        if (diff < 0) diff = -diff;
        if (diff > 0.01f) {
            if (errors == 0) { first_bad = got; first_expected = expected; }
            errors++;
        }
    }

    rsp->op = OP_LAB_SCALAR_WRITE_INVAL;
    rsp->status = (errors == 0) ? 0 : 1;
    rsp->errors = errors;
    rsp->detail = first_bad;

    FARF(HIGH, "SCALAR_WRITE_INVAL[%u] errors=%u/%u first: got=%.1f expected=%.1f",
         iter, errors, n_floats, first_bad, first_expected);
}


/* ================================================================
 * dspqueue message dispatch
 * ================================================================ */

static void packet_callback(dspqueue_t queue, int error, void *context) {
    struct lab_context *ctx = (struct lab_context *)context;

    while (1) {
        struct lab_req msg;
        uint32_t flags, msg_len, n_bufs;
        struct dspqueue_buffer bufs[LAB_MAX_BUFFERS];

        int err = dspqueue_read_noblock(queue, &flags,
                                        LAB_MAX_BUFFERS, &n_bufs, bufs,
                                        sizeof(msg), &msg_len, (uint8_t *)&msg);
        if (err == AEE_EWOULDBLOCK) return;
        if (err != 0) {
            FARF(ERROR, "dspqueue_read failed: 0x%08x", (unsigned)err);
            return;
        }

        struct lab_rsp rsp;
        memset(&rsp, 0, sizeof(rsp));

        switch (msg.op) {
        case OP_LAB_VTCM_BASIC:
            exp_vtcm_basic(ctx, msg.iter, msg.size, &rsp);
            break;

        case OP_LAB_VTCM_PERSIST:
            exp_vtcm_persist(ctx, msg.iter, &rsp);
            break;

        case OP_LAB_HVX_WRITE:
            exp_hvx_write(ctx, msg.iter, msg.size, &rsp);
            break;

        case OP_LAB_SCALAR_WRITE:
            exp_scalar_write(ctx, msg.iter, msg.size, &rsp);
            break;

        case OP_LAB_SGD_ITER:
            exp_sgd_iter(ctx, msg.iter, msg.learning_rate, &rsp);
            break;

        case OP_LAB_SGD_VERIFY:
            exp_sgd_verify(ctx, msg.iter, bufs, n_bufs, &rsp);
            break;

        case OP_LAB_TRAIN_INIT:
            exp_train_init(ctx, bufs, n_bufs, &rsp);
            break;

        case OP_LAB_TRAIN_BATCH:
            exp_train_batch(ctx, &msg, bufs, n_bufs, &rsp);
            break;

        case OP_LAB_TRAIN_SYNC:
            exp_train_sync(ctx, bufs, n_bufs, &rsp);
            break;

        case OP_LAB_TRAIN_EVAL:
            exp_train_eval(ctx, &msg, bufs, n_bufs, &rsp);
            break;

        case OP_LAB_L2_STALE_TEST:
            exp_l2_stale_test(ctx, &rsp);
            break;

        case OP_LAB_L2_INVAL_TEST:
            exp_l2_inval_test(ctx, &rsp);
            break;

        case OP_LAB_TRAIN_SYNC_V2:
            exp_train_sync_v2(ctx, bufs, n_bufs, &rsp);
            break;

        case OP_LAB_INCREMENT:
            exp_increment(ctx, msg.iter, &rsp);
            break;

        case OP_LAB_INCREMENT_SYNC:
            exp_increment_sync(ctx, msg.iter, &rsp);
            break;

        case OP_LAB_INCREMENT_FLUSH:
            exp_increment_flush(ctx, msg.iter, &rsp);
            break;

        case OP_LAB_INCREMENT_NOCACHE:
            exp_increment_nocache(ctx, msg.iter, &rsp);
            break;

        case OP_LAB_SCALAR_WRITE_INVAL:
            exp_scalar_write_inval(ctx, msg.iter, &rsp);
            break;

        case OP_LAB_TRAIN_SYNC_MCONLY:
            exp_train_sync_mconly(ctx, bufs, n_bufs, &rsp);
            break;

        case OP_LAB_TRAIN_SYNC_FLUSH:
            exp_train_sync_flush(ctx, bufs, n_bufs, &rsp);
            break;

        case OP_LAB_TRAIN_SYNC_NOOP:
            exp_train_sync_noop(ctx, &rsp);
            break;

        case OP_LAB_TRAIN_SYNC_DDR:
            exp_train_sync_ddr(ctx, bufs, n_bufs, &rsp);
            break;

        case OP_LAB_TRAIN_SYNC_SMALL:
            exp_train_sync_small(ctx, bufs, n_bufs, &rsp);
            break;

        case OP_LAB_TRAIN_SYNC_NOOP_EVAL:
            exp_train_sync_noop_eval(ctx, &rsp);
            break;

        case OP_LAB_TRAIN_SYNC_DSPEVAL:
            exp_train_sync_dspeval(ctx, &rsp);
            break;

        default:
            FARF(ERROR, "Unknown op %u", msg.op);
            rsp.op = msg.op;
            rsp.status = AEE_EBADPARM;
            break;
        }

        /* Deref all buffers in response */
        struct dspqueue_buffer rsp_bufs[LAB_MAX_BUFFERS];
        memset(rsp_bufs, 0, sizeof(rsp_bufs));
        for (uint32_t i = 0; i < n_bufs; i++) {
            rsp_bufs[i].fd = bufs[i].fd;
            rsp_bufs[i].flags = DSPQUEUE_BUFFER_FLAG_DEREF
                              | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
                              | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
        }

        dspqueue_write(queue, 0, n_bufs, rsp_bufs,
                       sizeof(rsp), (const uint8_t *)&rsp,
                       DSPQUEUE_TIMEOUT_NONE);
    }
}


/* ================================================================
 * dspqueue start/stop
 * ================================================================ */

AEEResult mnist_train_start(remote_handle64 handle, uint64 dsp_queue_id) {
    struct lab_context *ctx = (struct lab_context *)handle;
    if (!ctx) return AEE_EBADPARM;
    if (ctx->queue) return AEE_EITEMBUSY;

    /* Allocate VTCM */
    int vtcm_err = alloc_vtcm(ctx);
    if (vtcm_err != AEE_SUCCESS) return vtcm_err;

    int err = dspqueue_import(dsp_queue_id, packet_callback, error_callback,
                              ctx, &ctx->queue);
    if (err) {
        FARF(ERROR, "dspqueue_import failed: 0x%08x", (unsigned)err);
        return err;
    }

    return AEE_SUCCESS;
}

AEEResult mnist_train_stop(remote_handle64 handle, uint64 *process_time) {
    struct lab_context *ctx = (struct lab_context *)handle;
    if (!ctx) return AEE_EBADPARM;
    if (!ctx->queue) return AEE_EBADSTATE;

    int err = dspqueue_close(ctx->queue);
    ctx->queue = NULL;
    if (err) return err;

    if (ctx->vtcm_res_id) {
        HAP_compute_res_release(ctx->vtcm_res_id);
        ctx->vtcm_res_id = 0;
        ctx->vtcm_ready = 0;
        FARF(HIGH, "VTCM released in stop");
    }

    *process_time = 0;
    return AEE_SUCCESS;
}


/* Stub -- not used */
AEEResult mnist_train_do_matmul(remote_handle64 handle,
                                 const uint8 *a_buf, int a_buf_len,
                                 const uint8 *b_buf, int b_buf_len,
                                 uint8 *c_buf, int c_buf_len,
                                 uint32 m, uint32 n, uint32 k, uint32 transpose,
                                 uint64 *process_time) {
    return AEE_EUNSUPPORTED;
}
