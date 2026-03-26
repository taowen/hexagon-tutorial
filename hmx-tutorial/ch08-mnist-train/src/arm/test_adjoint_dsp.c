/*
 * test_adjoint_dsp.c -- Adjoint dot-product test for DSP HVX backward pass
 *
 * Mirrors test_adjoint.c but runs ops on the DSP via dspqueue.
 * All buffers are in shared memory (rpcmem) so the DSP can access them
 * directly via zero-copy fd references.
 *
 * Build:
 *   $NDK_CC -O2 ${ARM_INCS[@]} src/test_adjoint_dsp.c ${ARM_LIBS[@]} -o build/test_adjoint_dsp
 */

#include "common/common.h"
#include "arm/cpu_matmul.h"
#include "arm/network.h"
#include "arm/dspqueue_mgr.h"

#define CDSP_DOMAIN_ID 3
#define TEST_BATCH     4
#define REL_TOL        1e-3f
#define REL_TOL_SOFTMAX 2e-1f  /* DSP uses hvx_expf (Schraudolph's piecewise-linear approx) */

static int g_total_pass = 0;
static int g_total_fail = 0;

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */
static double dot_f(const float *a, const float *b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += (double)a[i] * (double)b[i];
    return sum;
}

static void fill_random(float *buf, int n) {
    for (int i = 0; i < n; i++)
        buf[i] = rand_uniform() * 2.0f - 1.0f;
}

static void report_tol(const char *name, double lhs, double rhs, double tol) {
    double denom = fmax(fabs(lhs), fabs(rhs));
    double rel_err = (denom < 1e-12) ? 0.0 : fabs(lhs - rhs) / denom;
    int ok = (rel_err < tol);
    if (ok) g_total_pass++; else g_total_fail++;
    printf("  %s %-50s lhs=%.8e  rhs=%.8e  rel_err=%.2e\n",
           ok ? "PASS" : "FAIL", name, lhs, rhs, rel_err);
}

static void report(const char *name, double lhs, double rhs) {
    report_tol(name, lhs, rhs, (double)REL_TOL);
}

/* ------------------------------------------------------------------ */
/*  send_test_op: send a test op to DSP via dspqueue                   */
/* ------------------------------------------------------------------ */
static void send_test_op(uint32_t op, void **bufs, int n_bufs,
                         uint32_t param1, uint32_t param2,
                         const uint8_t *labels) {
    struct test_op_req req;
    memset(&req, 0, sizeof(req));
    req.op = op;
    req.param1 = param1;
    req.param2 = param2;
    if (labels) {
        int n_labels = (param1 < 256) ? param1 : 256;
        memcpy(req.labels, labels, n_labels);
    }

    struct dspqueue_buffer dbufs[MATMUL_MAX_BUFFERS];
    memset(dbufs, 0, sizeof(dbufs));

    for (int i = 0; i < n_bufs; i++) {
        int fd = dspq_find_fd(bufs[i]);
        if (fd < 0) {
            fprintf(stderr, "[ERROR] send_test_op: buf %d not in shared memory\n", i);
            return;
        }
        dbufs[i].fd = fd;
        dbufs[i].flags = DSPQUEUE_BUFFER_FLAG_REF
                       | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
                       | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
    }

    g_dspq.ops_done = 0;
    g_dspq.ops_total = 1;

    int err = dspqueue_write(g_dspq.queue, 0, n_bufs, dbufs,
                              sizeof(req), (const uint8_t *)&req, 1000000);
    if (err != 0) {
        fprintf(stderr, "[ERROR] send_test_op: dspqueue_write failed: 0x%08x\n",
                (unsigned)err);
        return;
    }

    sem_wait(&g_dspq.done_sem);
}

/* ================================================================== */
/*  Test 1: relu adjoint on DSP                                        */
/*                                                                     */
/*  dot(ReLU(x), dy) == dot(x, relu_bwd(dy, x))                       */
/* ================================================================== */
static void test_relu_adjoint_dsp(void) {
    printf("\n[Test: relu adjoint (DSP HVX)]\n");

    const int n = TEST_BATCH * HIDDEN_DIM;

    float *x      = (float *)net_alloc(n * sizeof(float));
    float *dy     = (float *)net_alloc(n * sizeof(float));
    float *dx     = (float *)net_alloc(n * sizeof(float));
    float *x_relu = (float *)net_alloc(n * sizeof(float));

    fill_random(x, n);
    fill_random(dy, n);

    /* Forward: ReLU in-place on x_relu */
    memcpy(x_relu, x, n * sizeof(float));
    void *fwd_bufs[] = { x_relu };
    send_test_op(OP_TEST_RELU_FWD, fwd_bufs, 1, (uint32_t)n, 0, NULL);

    /* Backward: dx = relu_bwd(dy, x) */
    memcpy(dx, dy, n * sizeof(float));
    void *bwd_bufs[] = { dx, x };
    send_test_op(OP_TEST_RELU_BWD, bwd_bufs, 2, (uint32_t)n, 0, NULL);

    /* ReLU is self-adjoint: dot(ReLU(x), dy) == dot(x, relu_bwd(dy)) */
    double lhs = dot_f(x_relu, dy, n);
    double rhs = dot_f(x, dx, n);
    report("relu: dot(ReLU(x), dy) == dot(x, relu_bwd(dy))", lhs, rhs);

    net_free(x_relu); net_free(dx); net_free(dy); net_free(x);
}

/* ================================================================== */
/*  Test 2: bias_backward adjoint on DSP                               */
/*                                                                     */
/*  sum_b sum_j bias[j] * dy[b,j] == dot(bias, db)                    */
/* ================================================================== */
static void test_bias_adjoint_dsp(void) {
    printf("\n[Test: bias_backward adjoint (DSP HVX)]\n");

    const int batch = TEST_BATCH, dim = HIDDEN_DIM;

    float *bias = (float *)net_alloc(dim * sizeof(float));
    float *dy   = (float *)net_alloc(batch * dim * sizeof(float));
    float *db   = (float *)net_alloc(dim * sizeof(float));

    fill_random(bias, dim);
    fill_random(dy, batch * dim);

    /* Forward: bias broadcast contribution = sum_b sum_j bias[j] * dy[b,j] */
    double lhs = 0.0;
    for (int b = 0; b < batch; b++)
        for (int j = 0; j < dim; j++)
            lhs += (double)bias[j] * (double)dy[b * dim + j];

    /* Backward: db = sum_rows(dy) on DSP */
    void *bufs[] = { dy, db };
    send_test_op(OP_TEST_BIAS_BWD, bufs, 2, (uint32_t)batch, (uint32_t)dim, NULL);

    double rhs = dot_f(bias, db, dim);
    report("bias: dot(bias_broadcast, dy) == dot(bias, db)", lhs, rhs);

    net_free(db); net_free(dy); net_free(bias);
}

/* ================================================================== */
/*  Test 3: add_bias adjoint on DSP                                    */
/*                                                                     */
/*  dot(x + broadcast(bias), dy) - dot(x, dy) == dot(bias, db)        */
/* ================================================================== */
static void test_add_bias_adjoint_dsp(void) {
    printf("\n[Test: add_bias adjoint (DSP HVX)]\n");

    const int batch = TEST_BATCH, dim = HIDDEN_DIM;
    const int total = batch * dim;

    float *x_in  = (float *)net_alloc(total * sizeof(float));
    float *bias  = (float *)net_alloc(dim * sizeof(float));
    float *dy    = (float *)net_alloc(total * sizeof(float));
    float *out   = (float *)net_alloc(total * sizeof(float));

    fill_random(x_in, total);
    fill_random(bias, dim);
    fill_random(dy, total);

    /* Forward: out = x_in + broadcast(bias) on DSP */
    memcpy(out, x_in, total * sizeof(float));
    void *fwd_bufs[] = { out, bias };
    send_test_op(OP_TEST_ADD_BIAS, fwd_bufs, 2, (uint32_t)batch, (uint32_t)dim, NULL);

    /* lhs = dot(out, dy) - dot(x_in, dy) = dot(broadcast(bias), dy) */
    double lhs = dot_f(out, dy, total) - dot_f(x_in, dy, total);

    /* Compute bias gradient on CPU: db[j] = sum_b dy[b*dim+j] */
    float *db_cpu = (float *)malloc(dim * sizeof(float));
    bias_backward(db_cpu, dy, batch, dim);

    /* rhs = dot(bias, db) */
    double rhs = dot_f(bias, db_cpu, dim);
    report("add_bias: dot(broadcast(bias), dy) == dot(bias, db)", lhs, rhs);

    free(db_cpu);
    net_free(out); net_free(dy); net_free(bias); net_free(x_in);
}

/* ================================================================== */
/*  Test 4: softmax+cross-entropy dlogits gradient on DSP              */
/*                                                                     */
/*  Finite difference: perturb logits, recompute softmax on DSP,       */
/*  check directional derivative matches dot(dlogits, perturbation).   */
/* ================================================================== */
static void test_softmax_dlogits_dsp(void) {
    printf("\n[Test: softmax dlogits gradient (DSP HVX)]\n");

    const int batch = TEST_BATCH, dim = OUTPUT_DIM_PAD;

    float *logits  = (float *)net_alloc(batch * dim * sizeof(float));
    float *probs   = (float *)net_alloc(batch * dim * sizeof(float));
    float *dlogits = (float *)net_alloc(batch * dim * sizeof(float));
    float *perturb = (float *)malloc(batch * dim * sizeof(float));
    float *logits2 = (float *)net_alloc(batch * dim * sizeof(float));
    float *probs2  = (float *)net_alloc(batch * dim * sizeof(float));
    uint8_t *labels = (uint8_t *)malloc(batch);

    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < OUTPUT_DIM; j++)
            logits[b * dim + j] = rand_uniform() * 2.0f - 1.0f;
        for (int j = OUTPUT_DIM; j < dim; j++)
            logits[b * dim + j] = 0.0f;
        labels[b] = (uint8_t)(lcg_rand() % OUTPUT_DIM);
    }

    /* Forward: softmax on DSP */
    void *fwd_bufs[] = { logits, probs };
    send_test_op(OP_TEST_SOFTMAX_CE, fwd_bufs, 2,
                 (uint32_t)batch, 0, labels);

    /* Backward: compute dlogits on DSP */
    void *bwd_bufs[] = { dlogits, probs };
    send_test_op(OP_TEST_DLOGITS, bwd_bufs, 2,
                 (uint32_t)batch, 0, labels);

    /* Verify via directional finite differences */
    fill_random(perturb, batch * dim);
    for (int b = 0; b < batch; b++)
        for (int j = OUTPUT_DIM; j < dim; j++)
            perturb[b * dim + j] = 0.0f;

    float eps = 1e-3f;

    /* loss(logits + eps*perturb) via DSP softmax */
    for (int i = 0; i < batch * dim; i++)
        logits2[i] = logits[i] + eps * perturb[i];
    void *plus_bufs[] = { logits2, probs2 };
    send_test_op(OP_TEST_SOFTMAX_CE, plus_bufs, 2,
                 (uint32_t)batch, 0, labels);
    /* Compute loss from probs on ARM */
    float loss_plus = 0.0f;
    for (int b = 0; b < batch; b++) {
        float p = probs2[b * dim + labels[b]];
        if (p < 1e-7f) p = 1e-7f;
        loss_plus += -logf(p);
    }
    loss_plus /= (float)batch;

    /* loss(logits - eps*perturb) via DSP softmax */
    for (int i = 0; i < batch * dim; i++)
        logits2[i] = logits[i] - eps * perturb[i];
    void *minus_bufs[] = { logits2, probs2 };
    send_test_op(OP_TEST_SOFTMAX_CE, minus_bufs, 2,
                 (uint32_t)batch, 0, labels);
    float loss_minus = 0.0f;
    for (int b = 0; b < batch; b++) {
        float p = probs2[b * dim + labels[b]];
        if (p < 1e-7f) p = 1e-7f;
        loss_minus += -logf(p);
    }
    loss_minus /= (float)batch;

    double numerical = (double)(loss_plus - loss_minus) / (2.0 * eps);
    double analytical = dot_f(dlogits, perturb, batch * dim);

    report_tol("dlogits: numerical_dir_deriv == dot(dlogits, v)",
               numerical, analytical, (double)REL_TOL_SOFTMAX);

    free(labels); free(perturb);
    net_free(probs2); net_free(logits2);
    net_free(dlogits); net_free(probs); net_free(logits);
}

/* ================================================================== */
/*  Test 5: matmul adjoint on DSP (via dspqueue_matmul_dispatch)       */
/*                                                                     */
/*  dot(x @ W^T, dy) == dot(x, dy @ W) == dot(W, dW)                  */
/* ================================================================== */
static void test_matmul_adjoint_dsp(void) {
    printf("\n[Test: matmul adjoint (DSP HVX via dspqueue_matmul_dispatch)]\n");

    const int batch = TEST_BATCH, in_dim = INPUT_DIM_PAD, out_dim = HIDDEN_DIM;

    float *W    = (float *)net_alloc(out_dim * in_dim * sizeof(float));
    float *x    = (float *)net_alloc(batch * in_dim * sizeof(float));
    float *dy   = (float *)net_alloc(batch * out_dim * sizeof(float));
    float *Ax   = (float *)net_alloc(batch * out_dim * sizeof(float));
    float *ATdy = (float *)net_alloc(batch * in_dim * sizeof(float));
    float *dW   = (float *)net_alloc(out_dim * in_dim * sizeof(float));

    fill_random(W, out_dim * in_dim);
    fill_random(x, batch * in_dim);
    fill_random(dy, batch * out_dim);

    /* Forward: Ax[batch x out_dim] = x[batch x in_dim] @ W^T[out_dim x in_dim] */
    dspqueue_matmul_dispatch(Ax, x, W, batch, out_dim, in_dim, /*transpose=*/1);

    /* Backward input: ATdy[batch x in_dim] = dy[batch x out_dim] @ W[out_dim x in_dim] */
    dspqueue_matmul_dispatch(ATdy, dy, W, batch, in_dim, out_dim, /*transpose=*/0);

    double lhs = dot_f(Ax, dy, batch * out_dim);
    double rhs = dot_f(x, ATdy, batch * in_dim);
    report("input: dot(x@W^T, dy) == dot(x, dy@W)", lhs, rhs);

    /* Backward weight: dW[out_dim x in_dim] += dy^T @ x */
    memset(dW, 0, out_dim * in_dim * sizeof(float));
    dspqueue_matmul_dispatch(dW, dy, x, out_dim, in_dim, batch, /*transpose=*/2);

    double rhs_w = dot_f(W, dW, out_dim * in_dim);
    report("weight: dot(x@W^T, dy) == dot(W, dW)", lhs, rhs_w);

    net_free(dW); net_free(ATdy); net_free(Ax);
    net_free(dy); net_free(x); net_free(W);
}

/* ================================================================== */
/*  Test 6: SGD update on DSP                                          */
/*                                                                     */
/*  After SGD: w_new = w - lr * grad                                   */
/*  Property: dot(w_new, v) = dot(w, v) - lr * dot(grad, v)           */
/* ================================================================== */
static void test_sgd_dsp(void) {
    printf("\n[Test: SGD update (DSP HVX)]\n");

    const int n = HIDDEN_DIM * 32; /* use a non-trivial size */
    const float lr = 0.01f;

    float *w    = (float *)net_alloc(n * sizeof(float));
    float *grad = (float *)net_alloc(n * sizeof(float));
    float *v    = (float *)malloc(n * sizeof(float));

    fill_random(w, n);
    fill_random(grad, n);
    fill_random(v, n);

    /* Expected: dot(w - lr*grad, v) = dot(w,v) - lr*dot(grad,v) */
    double dot_w_v = dot_f(w, v, n);
    double dot_g_v = dot_f(grad, v, n);
    double expected = dot_w_v - (double)lr * dot_g_v;

    /* Run SGD on DSP */
    uint32_t lr_bits = *(uint32_t *)&lr;
    void *bufs[] = { w, grad };
    send_test_op(OP_TEST_SGD, bufs, 2, (uint32_t)n, lr_bits, NULL);

    double actual = dot_f(w, v, n);
    report("sgd: dot(w-lr*g, v) == dot(w,v)-lr*dot(g,v)", actual, expected);

    free(v);
    net_free(grad); net_free(w);
}

/* ================================================================== */
/*  Test 7: Linearity of DSP backward functions                        */
/*                                                                     */
/*  backward(alpha*dy) == alpha*backward(dy)                           */
/* ================================================================== */
static void test_linearity_dsp(void) {
    printf("\n[Test: Linearity of DSP backward functions]\n");

    const int batch = TEST_BATCH, dim = HIDDEN_DIM;
    const float alpha = 3.7f;

    /* bias_backward linearity */
    float *dy        = (float *)net_alloc(batch * dim * sizeof(float));
    float *dy_scaled = (float *)net_alloc(batch * dim * sizeof(float));
    float *db1       = (float *)net_alloc(dim * sizeof(float));
    float *db2       = (float *)net_alloc(dim * sizeof(float));

    fill_random(dy, batch * dim);
    for (int i = 0; i < batch * dim; i++)
        dy_scaled[i] = alpha * dy[i];

    void *bufs1[] = { dy, db1 };
    send_test_op(OP_TEST_BIAS_BWD, bufs1, 2, (uint32_t)batch, (uint32_t)dim, NULL);

    void *bufs2[] = { dy_scaled, db2 };
    send_test_op(OP_TEST_BIAS_BWD, bufs2, 2, (uint32_t)batch, (uint32_t)dim, NULL);

    float *v = (float *)malloc(dim * sizeof(float));
    fill_random(v, dim);
    double lhs = dot_f(db2, v, dim);
    double rhs = (double)alpha * dot_f(db1, v, dim);
    report("bias_backward: bwd(a*dy) == a*bwd(dy)", lhs, rhs);

    /* relu_backward linearity */
    const int n = batch * dim;
    float *x         = (float *)net_alloc(n * sizeof(float));
    float *relu_dy   = (float *)net_alloc(n * sizeof(float));
    float *relu_dy_s = (float *)net_alloc(n * sizeof(float));
    float *rdx1      = (float *)net_alloc(n * sizeof(float));
    float *rdx2      = (float *)net_alloc(n * sizeof(float));

    fill_random(x, n);
    fill_random(relu_dy, n);
    for (int i = 0; i < n; i++)
        relu_dy_s[i] = alpha * relu_dy[i];

    memcpy(rdx1, relu_dy, n * sizeof(float));
    void *rbufs1[] = { rdx1, x };
    send_test_op(OP_TEST_RELU_BWD, rbufs1, 2, (uint32_t)n, 0, NULL);

    memcpy(rdx2, relu_dy_s, n * sizeof(float));
    void *rbufs2[] = { rdx2, x };
    send_test_op(OP_TEST_RELU_BWD, rbufs2, 2, (uint32_t)n, 0, NULL);

    float *rv = (float *)malloc(n * sizeof(float));
    fill_random(rv, n);
    double lhs_r = dot_f(rdx2, rv, n);
    double rhs_r = (double)alpha * dot_f(rdx1, rv, n);
    report("relu_backward: bwd(a*dy) == a*bwd(dy)", lhs_r, rhs_r);

    free(rv); free(v);
    net_free(rdx2); net_free(rdx1); net_free(relu_dy_s);
    net_free(relu_dy); net_free(x);
    net_free(db2); net_free(db1); net_free(dy_scaled); net_free(dy);
}

/* ================================================================== */
/*  Test 8: Zero propagation on DSP                                    */
/*                                                                     */
/*  backward(0) == 0                                                   */
/* ================================================================== */
static void test_zero_propagation_dsp(void) {
    printf("\n[Test: Zero propagation of DSP backward functions]\n");

    const int batch = TEST_BATCH, dim = HIDDEN_DIM;

    /* bias_backward(0) == 0 */
    float *dy = (float *)net_alloc(batch * dim * sizeof(float));
    float *db = (float *)net_alloc(dim * sizeof(float));
    memset(dy, 0, batch * dim * sizeof(float));

    void *bufs[] = { dy, db };
    send_test_op(OP_TEST_BIAS_BWD, bufs, 2, (uint32_t)batch, (uint32_t)dim, NULL);

    double norm = dot_f(db, db, dim);
    report("bias_backward(0) == 0: ||db||^2", norm, 0.0);

    /* relu_backward(0) == 0 */
    float *x  = (float *)net_alloc(batch * dim * sizeof(float));
    float *dx = (float *)net_alloc(batch * dim * sizeof(float));
    fill_random(x, batch * dim);
    memset(dx, 0, batch * dim * sizeof(float));

    void *rbufs[] = { dx, x };
    send_test_op(OP_TEST_RELU_BWD, rbufs, 2, (uint32_t)(batch * dim), 0, NULL);

    double norm_r = dot_f(dx, dx, batch * dim);
    report("relu_backward(0) == 0: ||dx||^2", norm_r, 0.0);

    net_free(dx); net_free(x); net_free(db); net_free(dy);
}

/* ================================================================== */
/*  main                                                               */
/* ================================================================== */
int main(void) {
    setbuf(stdout, NULL);
    g_batch_size = TEST_BATCH;
    g_rng_state = 42;

    printf("=== Adjoint Dot-Product Test (DSP HVX) ===\n");
    printf("All ops run on DSP via dspqueue (HVX implementations)\n");
    printf("batch=%d, rel_tol=%.0e\n", TEST_BATCH, (double)REL_TOL);
    printf("Network dims: input=%d, hidden=%d, output=%d (padded: %d, %d)\n",
           INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, INPUT_DIM_PAD, OUTPUT_DIM_PAD);

    /* Initialize dspqueue (sets g_alloc_fn = dspq_alloc for shared memory) */
    if (dspqueue_init(CDSP_DOMAIN_ID) != 0) {
        printf("[ERROR] dspqueue_init failed\n");
        return 1;
    }

    test_relu_adjoint_dsp();       /* ReLU forward+backward adjoint */
    test_bias_adjoint_dsp();       /* bias_backward adjoint */
    test_add_bias_adjoint_dsp();   /* add_bias adjoint */
    test_softmax_dlogits_dsp();    /* softmax+CE dlogits gradient */
    test_matmul_adjoint_dsp();     /* HVX matmul transpose modes */
    test_sgd_dsp();                /* SGD update property */
    test_linearity_dsp();          /* linearity property */
    test_zero_propagation_dsp();   /* zero propagation */

    printf("\n=== Summary: %d passed, %d failed ===\n",
           g_total_pass, g_total_fail);

    if (g_total_fail == 0)
        printf("=== ALL PASSED ===\n");
    else
        printf("=== FAILED ===\n");

    dspqueue_cleanup();

    return g_total_fail > 0 ? 1 : 0;
}
