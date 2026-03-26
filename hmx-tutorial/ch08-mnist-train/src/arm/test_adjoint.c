/*
 * test_adjoint.c -- Adjoint dot-product test for backward pass correctness
 *
 * Tests the REAL backward functions from mnist_network.h, not reimplementations.
 * For a linear operator A and its adjoint A*, the identity
 *   dot(A(x), y) == dot(x, A*(y))
 * must hold.
 *
 * Build (CPU only, no DSP needed):
 *   cc -O2 -Isrc -o test_adjoint src/test_adjoint.c -lm
 */

#include "common/common.h"
#include "arm/cpu_matmul.h"
#include "arm/network.h"

#define TEST_BATCH  4
#define REL_TOL         1e-4f
#define REL_TOL_SOFTMAX 2e-3f

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

/* ================================================================== */
/*  Test 1: matmul adjoint (transpose=1 forward, transpose=0 backward)*/
/*                                                                     */
/*  Forward: out = A @ W^T       (matmul transpose=1)                 */
/*  Backward input: dx = dout @ W (matmul transpose=0)                */
/*  Identity: dot(A @ W^T, dy) == dot(A, dy @ W)                     */
/* ================================================================== */
static void test_matmul_adjoint(void) {
    printf("\n[Test: matmul adjoint (real cpu_matmul_dispatch)]\n");

    const int batch = TEST_BATCH, in_dim = INPUT_DIM_PAD, out_dim = HIDDEN_DIM;

    float *W   = malloc(out_dim * in_dim * sizeof(float));
    float *x   = malloc(batch * in_dim * sizeof(float));
    float *dy  = malloc(batch * out_dim * sizeof(float));
    float *Ax  = malloc(batch * out_dim * sizeof(float));
    float *ATdy = malloc(batch * in_dim * sizeof(float));
    float *dW  = malloc(out_dim * in_dim * sizeof(float));

    fill_random(W, out_dim * in_dim);
    fill_random(x, batch * in_dim);
    fill_random(dy, batch * out_dim);

    /* Forward: out[batch x out_dim] = x[batch x in_dim] @ W^T[out_dim x in_dim] */
    cpu_matmul_dispatch(Ax, x, W, batch, out_dim, in_dim, /*transpose=*/1);

    /* Backward input: dx[batch x in_dim] = dy[batch x out_dim] @ W[out_dim x in_dim] */
    cpu_matmul_dispatch(ATdy, dy, W, batch, in_dim, out_dim, /*transpose=*/0);

    double lhs = dot_f(Ax, dy, batch * out_dim);
    double rhs = dot_f(x, ATdy, batch * in_dim);
    report("input: dot(x@W^T, dy) == dot(x, dy@W)", lhs, rhs);

    /* Backward weight: dW[out_dim x in_dim] += dy^T @ x */
    memset(dW, 0, out_dim * in_dim * sizeof(float));
    cpu_matmul_dispatch(dW, dy, x, out_dim, in_dim, batch, /*transpose=*/2);

    double rhs_w = dot_f(W, dW, out_dim * in_dim);
    report("weight: dot(x@W^T, dy) == dot(W, dW)", lhs, rhs_w);

    free(dW); free(ATdy); free(Ax); free(dy); free(x); free(W);
}

/* ================================================================== */
/*  Test 2: bias_backward adjoint (calls real bias_backward)           */
/* ================================================================== */
static void test_bias_adjoint(void) {
    printf("\n[Test: bias_backward adjoint (real function)]\n");

    const int batch = TEST_BATCH, dim = HIDDEN_DIM;

    float *bias = malloc(dim * sizeof(float));
    float *dy   = malloc(batch * dim * sizeof(float));
    float *db   = malloc(dim * sizeof(float));

    fill_random(bias, dim);
    fill_random(dy, batch * dim);

    /* Forward: out[b,j] += bias[j] */
    /* The bias broadcast contributes: sum_b sum_j bias[j] * dy[b,j] */
    double lhs = 0.0;
    for (int b = 0; b < batch; b++)
        for (int j = 0; j < dim; j++)
            lhs += (double)bias[j] * (double)dy[b * dim + j];

    /* Backward: db = sum_rows(dy) -- calls REAL function */
    bias_backward(db, dy, batch, dim);

    double rhs = dot_f(bias, db, dim);
    report("bias: dot(bias_broadcast, dy) == dot(bias, db)", lhs, rhs);

    free(db); free(dy); free(bias);
}

/* ================================================================== */
/*  Test 3: relu_backward adjoint (calls real relu_backward)           */
/* ================================================================== */
static void test_relu_adjoint(void) {
    printf("\n[Test: relu_backward adjoint (real function)]\n");

    const int n = TEST_BATCH * HIDDEN_DIM;

    float *x      = malloc(n * sizeof(float));
    float *x_relu = malloc(n * sizeof(float));
    float *dy     = malloc(n * sizeof(float));
    float *dx     = malloc(n * sizeof(float));

    fill_random(x, n);
    fill_random(dy, n);

    /* Forward: ReLU */
    memcpy(x_relu, x, n * sizeof(float));
    relu_forward(x_relu, n);

    /* Backward: calls REAL relu_backward from mnist_network.h */
    memcpy(dx, dy, n * sizeof(float));
    relu_backward(dx, x, n);

    /* ReLU is self-adjoint: dot(ReLU(x), dy) == dot(x, relu_bwd(dy)) */
    double lhs = dot_f(x_relu, dy, n);
    double rhs = dot_f(x, dx, n);
    report("relu: dot(ReLU(x), dy) == dot(x, relu_bwd(dy))", lhs, rhs);

    free(dx); free(dy); free(x_relu); free(x);
}

/* ================================================================== */
/*  Test 4: compute_dlogits gradient (calls real compute_dlogits)      */
/* ================================================================== */
static void test_dlogits(void) {
    printf("\n[Test: compute_dlogits gradient (real function)]\n");

    const int batch = TEST_BATCH, dim = OUTPUT_DIM_PAD;

    float *logits  = malloc(batch * dim * sizeof(float));
    float *probs   = malloc(batch * dim * sizeof(float));
    float *dlogits = malloc(batch * dim * sizeof(float));
    float *perturb = malloc(batch * dim * sizeof(float));
    float *logits2 = malloc(batch * dim * sizeof(float));
    float *probs2  = malloc(batch * dim * sizeof(float));
    uint8_t *labels = malloc(batch);

    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < OUTPUT_DIM; j++)
            logits[b * dim + j] = rand_uniform() * 2.0f - 1.0f;
        for (int j = OUTPUT_DIM; j < dim; j++)
            logits[b * dim + j] = 0.0f;
        labels[b] = (uint8_t)(lcg_rand() % OUTPUT_DIM);
    }

    /* Forward: loss + probs */
    softmax_cross_entropy(logits, labels, probs, batch);

    /* Backward: calls REAL compute_dlogits */
    compute_dlogits(dlogits, probs, labels, batch, dim);

    /* Verify via directional finite differences */
    fill_random(perturb, batch * dim);
    for (int b = 0; b < batch; b++)
        for (int j = OUTPUT_DIM; j < dim; j++)
            perturb[b * dim + j] = 0.0f;

    float eps = 1e-3f;
    for (int i = 0; i < batch * dim; i++)
        logits2[i] = logits[i] + eps * perturb[i];
    float loss_plus = softmax_cross_entropy(logits2, labels, probs2, batch);

    for (int i = 0; i < batch * dim; i++)
        logits2[i] = logits[i] - eps * perturb[i];
    float loss_minus = softmax_cross_entropy(logits2, labels, probs2, batch);

    double numerical = (double)(loss_plus - loss_minus) / (2.0 * eps);
    double analytical = dot_f(dlogits, perturb, batch * dim);

    report_tol("dlogits: numerical_dir_deriv == dot(dlogits, v)",
               numerical, analytical, (double)REL_TOL_SOFTMAX);

    free(labels); free(probs2); free(logits2);
    free(perturb); free(dlogits); free(probs); free(logits);
}

/* ================================================================== */
/*  Test 5: Full backward() end-to-end                                 */
/*                                                                     */
/*  Run the REAL forward() + backward() from mnist_network.h, then     */
/*  verify each gradient via finite-difference directional derivative. */
/* ================================================================== */
static void test_full_backward(void) {
    printf("\n[Test: Full backward() end-to-end (real forward + backward)]\n");

    const int batch = TEST_BATCH;
    network_t net;
    network_init(&net);

    float *input = malloc(batch * INPUT_DIM_PAD * sizeof(float));
    uint8_t *labels = malloc(batch);
    fill_random(input, batch * INPUT_DIM_PAD);
    for (int b = 0; b < batch; b++)
        labels[b] = (uint8_t)(lcg_rand() % OUTPUT_DIM);

    /* Run real forward + loss + backward */
    forward(&net, input, batch, cpu_matmul_dispatch);
    softmax_cross_entropy(net.logits, labels, net.probs, batch);

    memset(net.dw1, 0, HIDDEN_DIM * INPUT_DIM_PAD * sizeof(float));
    memset(net.db1, 0, HIDDEN_DIM * sizeof(float));
    memset(net.dw2, 0, OUTPUT_DIM_PAD * HIDDEN_DIM * sizeof(float));
    memset(net.db2, 0, OUTPUT_DIM_PAD * sizeof(float));
    backward(&net, input, labels, batch, cpu_matmul_dispatch);

    /* Verify dW2 via directional derivative:
     * loss(W2 + eps*V) - loss(W2 - eps*V) ~ 2*eps * dot(dW2, V) */
    float *V = malloc(OUTPUT_DIM_PAD * HIDDEN_DIM * sizeof(float));
    float *W2_save = malloc(OUTPUT_DIM_PAD * HIDDEN_DIM * sizeof(float));
    memcpy(W2_save, net.w2, OUTPUT_DIM_PAD * HIDDEN_DIM * sizeof(float));
    fill_random(V, OUTPUT_DIM_PAD * HIDDEN_DIM);

    float eps = 1e-3f;

    /* loss(W2 + eps*V) */
    for (int i = 0; i < OUTPUT_DIM_PAD * HIDDEN_DIM; i++)
        net.w2[i] = W2_save[i] + eps * V[i];
    forward(&net, input, batch, cpu_matmul_dispatch);
    float loss_plus = softmax_cross_entropy(net.logits, labels, net.probs, batch);

    /* loss(W2 - eps*V) */
    for (int i = 0; i < OUTPUT_DIM_PAD * HIDDEN_DIM; i++)
        net.w2[i] = W2_save[i] - eps * V[i];
    forward(&net, input, batch, cpu_matmul_dispatch);
    float loss_minus = softmax_cross_entropy(net.logits, labels, net.probs, batch);

    memcpy(net.w2, W2_save, OUTPUT_DIM_PAD * HIDDEN_DIM * sizeof(float));

    double numerical = (double)(loss_plus - loss_minus) / (2.0 * eps);
    double analytical = dot_f(net.dw2, V, OUTPUT_DIM_PAD * HIDDEN_DIM);
    report_tol("dW2: numerical == dot(dW2, V)", numerical, analytical, (double)REL_TOL_SOFTMAX);

    /* Verify dW1 via directional derivative.
     * W1 gradients pass through ReLU, so finite-difference is less precise.
     * Use smaller eps to reduce ReLU kink crossings. */
    float eps1 = 1e-4f;
    float *V1 = malloc(HIDDEN_DIM * INPUT_DIM_PAD * sizeof(float));
    float *W1_save = malloc(HIDDEN_DIM * INPUT_DIM_PAD * sizeof(float));
    memcpy(W1_save, net.w1, HIDDEN_DIM * INPUT_DIM_PAD * sizeof(float));
    fill_random(V1, HIDDEN_DIM * INPUT_DIM_PAD);

    for (int i = 0; i < HIDDEN_DIM * INPUT_DIM_PAD; i++)
        net.w1[i] = W1_save[i] + eps1 * V1[i];
    forward(&net, input, batch, cpu_matmul_dispatch);
    loss_plus = softmax_cross_entropy(net.logits, labels, net.probs, batch);

    for (int i = 0; i < HIDDEN_DIM * INPUT_DIM_PAD; i++)
        net.w1[i] = W1_save[i] - eps1 * V1[i];
    forward(&net, input, batch, cpu_matmul_dispatch);
    loss_minus = softmax_cross_entropy(net.logits, labels, net.probs, batch);

    memcpy(net.w1, W1_save, HIDDEN_DIM * INPUT_DIM_PAD * sizeof(float));

    numerical = (double)(loss_plus - loss_minus) / (2.0 * eps1);
    analytical = dot_f(net.dw1, V1, HIDDEN_DIM * INPUT_DIM_PAD);
    /* ReLU kink + float32 → expect ~2-5% error (PyTorch uses float64 for gradcheck) */
    report_tol("dW1: numerical == dot(dW1, V) [through ReLU]", numerical, analytical, 5e-2);

    /* Verify db2 */
    float *b2_save = malloc(OUTPUT_DIM_PAD * sizeof(float));
    float *Vb2 = malloc(OUTPUT_DIM_PAD * sizeof(float));
    memcpy(b2_save, net.b2, OUTPUT_DIM_PAD * sizeof(float));
    fill_random(Vb2, OUTPUT_DIM_PAD);

    for (int i = 0; i < OUTPUT_DIM_PAD; i++)
        net.b2[i] = b2_save[i] + eps * Vb2[i];
    forward(&net, input, batch, cpu_matmul_dispatch);
    loss_plus = softmax_cross_entropy(net.logits, labels, net.probs, batch);

    for (int i = 0; i < OUTPUT_DIM_PAD; i++)
        net.b2[i] = b2_save[i] - eps * Vb2[i];
    forward(&net, input, batch, cpu_matmul_dispatch);
    loss_minus = softmax_cross_entropy(net.logits, labels, net.probs, batch);

    memcpy(net.b2, b2_save, OUTPUT_DIM_PAD * sizeof(float));

    numerical = (double)(loss_plus - loss_minus) / (2.0 * eps);
    analytical = dot_f(net.db2, Vb2, OUTPUT_DIM_PAD);
    report_tol("db2: numerical == dot(db2, V)", numerical, analytical, (double)REL_TOL_SOFTMAX);

    /* Verify db1 (passes through ReLU, use smaller eps) */
    float *b1_save = malloc(HIDDEN_DIM * sizeof(float));
    float *Vb1 = malloc(HIDDEN_DIM * sizeof(float));
    memcpy(b1_save, net.b1, HIDDEN_DIM * sizeof(float));
    fill_random(Vb1, HIDDEN_DIM);

    for (int i = 0; i < HIDDEN_DIM; i++)
        net.b1[i] = b1_save[i] + eps1 * Vb1[i];
    forward(&net, input, batch, cpu_matmul_dispatch);
    loss_plus = softmax_cross_entropy(net.logits, labels, net.probs, batch);

    for (int i = 0; i < HIDDEN_DIM; i++)
        net.b1[i] = b1_save[i] - eps1 * Vb1[i];
    forward(&net, input, batch, cpu_matmul_dispatch);
    loss_minus = softmax_cross_entropy(net.logits, labels, net.probs, batch);

    memcpy(net.b1, b1_save, HIDDEN_DIM * sizeof(float));

    numerical = (double)(loss_plus - loss_minus) / (2.0 * eps1);
    analytical = dot_f(net.db1, Vb1, HIDDEN_DIM);
    report_tol("db1: numerical == dot(db1, V) [through ReLU]", numerical, analytical, 5e-2);

    free(Vb1); free(b1_save); free(Vb2); free(b2_save);
    free(V1); free(W1_save); free(V); free(W2_save);
    free(labels); free(input);
    network_free(&net);
}

/* ================================================================== */
/*  Test 6: SGD update (CPU)                                           */
/*                                                                     */
/*  After SGD: w_new = w - lr * grad                                   */
/*  Property: dot(w_new, v) = dot(w, v) - lr * dot(grad, v)           */
/* ================================================================== */
static void test_sgd_cpu(void) {
    printf("\n[Test: SGD update (CPU)]\n");

    const int n = HIDDEN_DIM * 32;
    const float lr = 0.01f;

    float *w    = (float *)malloc(n * sizeof(float));
    float *grad = (float *)malloc(n * sizeof(float));
    float *v    = (float *)malloc(n * sizeof(float));

    g_rng_state = 99;
    fill_random(w, n);
    fill_random(grad, n);
    fill_random(v, n);

    double dot_w_v = dot_f(w, v, n);
    double dot_g_v = dot_f(grad, v, n);
    double expected = dot_w_v - (double)lr * dot_g_v;

    /* CPU SGD: w -= lr * grad */
    for (int i = 0; i < n; i++)
        w[i] -= lr * grad[i];

    double actual = dot_f(w, v, n);
    report("sgd: dot(w-lr*g, v) == dot(w,v)-lr*dot(g,v)", actual, expected);

    free(v); free(grad); free(w);
}

/* ================================================================== */
/*  Test 7: Linearity -- backward(alpha*dy) == alpha*backward(dy)      */
/* ================================================================== */
static void test_linearity(void) {
    printf("\n[Test: Linearity of real backward functions]\n");

    const int batch = TEST_BATCH, dim = HIDDEN_DIM;
    const float alpha = 3.7f;

    /* bias_backward linearity */
    float *dy = malloc(batch * dim * sizeof(float));
    float *dy_scaled = malloc(batch * dim * sizeof(float));
    float *db1 = malloc(dim * sizeof(float));
    float *db2 = malloc(dim * sizeof(float));

    fill_random(dy, batch * dim);
    for (int i = 0; i < batch * dim; i++)
        dy_scaled[i] = alpha * dy[i];

    bias_backward(db1, dy, batch, dim);
    bias_backward(db2, dy_scaled, batch, dim);

    float *v = malloc(dim * sizeof(float));
    fill_random(v, dim);
    double lhs = dot_f(db2, v, dim);
    double rhs = (double)alpha * dot_f(db1, v, dim);
    report("bias_backward: bwd(a*dy) == a*bwd(dy)", lhs, rhs);

    /* relu_backward linearity */
    const int n = batch * dim;
    float *x = malloc(n * sizeof(float));
    float *relu_dy = malloc(n * sizeof(float));
    float *relu_dy_s = malloc(n * sizeof(float));
    float *rdx1 = malloc(n * sizeof(float));
    float *rdx2 = malloc(n * sizeof(float));

    fill_random(x, n);
    fill_random(relu_dy, n);
    for (int i = 0; i < n; i++)
        relu_dy_s[i] = alpha * relu_dy[i];

    memcpy(rdx1, relu_dy, n * sizeof(float));
    relu_backward(rdx1, x, n);

    memcpy(rdx2, relu_dy_s, n * sizeof(float));
    relu_backward(rdx2, x, n);

    float *rv = malloc(n * sizeof(float));
    fill_random(rv, n);
    double lhs_r = dot_f(rdx2, rv, n);
    double rhs_r = (double)alpha * dot_f(rdx1, rv, n);
    report("relu_backward: bwd(a*dy) == a*bwd(dy)", lhs_r, rhs_r);

    free(rv); free(rdx2); free(rdx1); free(relu_dy_s); free(relu_dy); free(x);
    free(v); free(db2); free(db1); free(dy_scaled); free(dy);
}

/* ================================================================== */
/*  Test 8: Zero propagation -- backward(0) == 0                       */
/* ================================================================== */
static void test_zero_propagation(void) {
    printf("\n[Test: Zero propagation of real backward functions]\n");

    const int batch = TEST_BATCH, dim = HIDDEN_DIM;

    /* bias_backward(0) == 0 */
    float *dy = calloc(batch * dim, sizeof(float));
    float *db = malloc(dim * sizeof(float));
    bias_backward(db, dy, batch, dim);
    double norm = dot_f(db, db, dim);
    report("bias_backward(0) == 0: ||db||^2", norm, 0.0);

    /* relu_backward(0) == 0 */
    float *x = malloc(batch * dim * sizeof(float));
    float *dx = calloc(batch * dim, sizeof(float));
    fill_random(x, batch * dim);
    relu_backward(dx, x, batch * dim);
    double norm_r = dot_f(dx, dx, batch * dim);
    report("relu_backward(0) == 0: ||dx||^2", norm_r, 0.0);

    free(dx); free(x); free(db); free(dy);
}

/* ================================================================== */
/*  main                                                               */
/* ================================================================== */
int main(void) {
    setbuf(stdout, NULL);
    g_batch_size = TEST_BATCH;
    g_rng_state = 42;

    printf("=== Adjoint Dot-Product Test ===\n");
    printf("All backward functions are called from mnist_network.h (real code)\n");
    printf("batch=%d, rel_tol=%.0e\n", TEST_BATCH, (double)REL_TOL);
    printf("Network dims: input=%d, hidden=%d, output=%d (padded: %d, %d)\n",
           INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, INPUT_DIM_PAD, OUTPUT_DIM_PAD);

    test_matmul_adjoint();        /* matmul transpose modes */
    test_bias_adjoint();          /* real bias_backward() */
    test_relu_adjoint();          /* real relu_backward() */
    test_dlogits();               /* real compute_dlogits() */
    test_full_backward();         /* real forward() + backward() end-to-end */
    test_sgd_cpu();               /* SGD update property */
    test_linearity();             /* linearity property */
    test_zero_propagation();      /* zero propagation */

    printf("\n=== Summary: %d passed, %d failed ===\n",
           g_total_pass, g_total_fail);

    if (g_total_fail == 0)
        printf("=== ALL PASSED ===\n");
    else
        printf("=== FAILED ===\n");

    return g_total_fail > 0 ? 1 : 0;
}
