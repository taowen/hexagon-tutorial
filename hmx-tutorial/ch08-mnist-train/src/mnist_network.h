/*
 * mnist_network.h -- Network init/free, forward, backward, SGD, evaluate, train
 */

#ifndef MNIST_NETWORK_H
#define MNIST_NETWORK_H

#include "mnist_common.h"
#include "mnist_cpu_matmul.h"

/* ====================================================================
 * Network init / free
 * ==================================================================== */

static int network_init(network_t *net) {
    memset(net, 0, sizeof(*net));

    /* Buffers used by DSP matmul use net_alloc (shared memory in dspqueue mode) */
    net->w1      = (float *)net_alloc(HIDDEN_DIM * INPUT_DIM_PAD * sizeof(float));
    net->w2      = (float *)net_alloc(OUTPUT_DIM_PAD * HIDDEN_DIM * sizeof(float));
    net->dw1     = (float *)net_alloc(HIDDEN_DIM * INPUT_DIM_PAD * sizeof(float));
    net->dw2     = (float *)net_alloc(OUTPUT_DIM_PAD * HIDDEN_DIM * sizeof(float));
    net->hidden  = (float *)net_alloc(g_batch_size * HIDDEN_DIM * sizeof(float));
    net->logits  = (float *)net_alloc(g_batch_size * OUTPUT_DIM_PAD * sizeof(float));
    net->dlogits = (float *)net_alloc(g_batch_size * OUTPUT_DIM_PAD * sizeof(float));
    net->dhidden = (float *)net_alloc(g_batch_size * HIDDEN_DIM * sizeof(float));

    /* All buffers use net_alloc (shared memory in dspqueue mode, regular malloc in CPU mode) */
    net->b1  = (float *)net_calloc(HIDDEN_DIM, sizeof(float));
    net->b2  = (float *)net_calloc(OUTPUT_DIM_PAD, sizeof(float));
    net->db1 = (float *)net_alloc(HIDDEN_DIM * sizeof(float));
    net->db2 = (float *)net_alloc(OUTPUT_DIM_PAD * sizeof(float));
    net->hidden_pre_relu = (float *)net_alloc(g_batch_size * HIDDEN_DIM * sizeof(float));
    net->probs           = (float *)net_alloc(g_batch_size * OUTPUT_DIM_PAD * sizeof(float));

    if (!net->w1 || !net->b1 || !net->w2 || !net->b2 ||
        !net->dw1 || !net->db1 || !net->dw2 || !net->db2 ||
        !net->hidden || !net->hidden_pre_relu || !net->logits ||
        !net->probs || !net->dlogits || !net->dhidden) {
        printf("[ERROR] malloc failed\n");
        return -1;
    }

    he_init(net->w1, HIDDEN_DIM, INPUT_DIM_PAD, INPUT_DIM_PAD);
    he_init(net->w2, OUTPUT_DIM_PAD, HIDDEN_DIM, HIDDEN_DIM);
    /* Zero padding rows of W2 */
    for (int i = OUTPUT_DIM; i < OUTPUT_DIM_PAD; i++)
        for (int j = 0; j < HIDDEN_DIM; j++)
            net->w2[i * HIDDEN_DIM + j] = 0.0f;

    return 0;
}

static void network_free(network_t *net) {
    /* Free shared memory buffers via net_free */
    net_free(net->dhidden);
    net_free(net->dlogits);
    net_free(net->logits);
    net_free(net->hidden);
    net_free(net->dw2); net_free(net->dw1);
    net_free(net->w2); net_free(net->w1);
    /* Free remaining buffers via net_free */
    net_free(net->probs);
    net_free(net->hidden_pre_relu);
    net_free(net->db2); net_free(net->db1);
    net_free(net->b2); net_free(net->b1);
}

/* ====================================================================
 * Activation and loss functions
 * ==================================================================== */

static void add_bias(float *out, const float *bias, int batch, int dim) {
    for (int b = 0; b < batch; b++)
        for (int j = 0; j < dim; j++)
            out[b * dim + j] += bias[j];
}

static void relu_forward(float *x, int n) {
    for (int i = 0; i < n; i++)
        if (x[i] < 0.0f) x[i] = 0.0f;
}

static float softmax_cross_entropy(const float *logits, const uint8_t *labels,
                                   float *probs, int batch) {
    float total_loss = 0.0f;
    for (int b = 0; b < batch; b++) {
        const float *row = logits + b * OUTPUT_DIM_PAD;
        float *prob = probs + b * OUTPUT_DIM_PAD;

        float max_val = row[0];
        for (int j = 1; j < OUTPUT_DIM; j++)
            if (row[j] > max_val) max_val = row[j];

        float sum_exp = 0.0f;
        for (int j = 0; j < OUTPUT_DIM; j++) {
            prob[j] = expf(row[j] - max_val);
            sum_exp += prob[j];
        }
        for (int j = 0; j < OUTPUT_DIM; j++)
            prob[j] /= sum_exp;
        for (int j = OUTPUT_DIM; j < OUTPUT_DIM_PAD; j++)
            prob[j] = 0.0f;

        int label = labels[b];
        float p = prob[label];
        if (p < 1e-7f) p = 1e-7f;
        total_loss += -logf(p);
    }
    return total_loss / (float)batch;
}

/* ====================================================================
 * Forward pass (uses matmul function pointer)
 * ==================================================================== */

static void forward(network_t *net, const float *input, int batch,
                    matmul_fn_t matmul) {
    /* Layer 1: hidden = input @ W1^T + b1 */
    matmul(net->hidden, input, net->w1,
           batch, HIDDEN_DIM, INPUT_DIM_PAD, 1);
    add_bias(net->hidden, net->b1, batch, HIDDEN_DIM);

    memcpy(net->hidden_pre_relu, net->hidden, batch * HIDDEN_DIM * sizeof(float));
    relu_forward(net->hidden, batch * HIDDEN_DIM);

    /* Layer 2: logits = hidden @ W2^T + b2 */
    matmul(net->logits, net->hidden, net->w2,
           batch, OUTPUT_DIM_PAD, HIDDEN_DIM, 1);
    add_bias(net->logits, net->b2, batch, OUTPUT_DIM_PAD);
}

/* ====================================================================
 * Backward pass (uses matmul function pointer)
 * ==================================================================== */

static void backward(network_t *net, const float *input, const uint8_t *labels,
                     int batch, matmul_fn_t matmul) {
    /* dlogits = probs - one_hot(labels) / batch */
    memcpy(net->dlogits, net->probs, batch * OUTPUT_DIM_PAD * sizeof(float));
    for (int b = 0; b < batch; b++) {
        net->dlogits[b * OUTPUT_DIM_PAD + labels[b]] -= 1.0f;
        for (int j = 0; j < OUTPUT_DIM_PAD; j++)
            net->dlogits[b * OUTPUT_DIM_PAD + j] /= (float)batch;
    }

    /* dW2 += dlogits^T @ hidden (transpose=2: C += A^T @ B) */
    matmul(net->dw2, net->dlogits, net->hidden,
           OUTPUT_DIM_PAD, HIDDEN_DIM, batch, 2);

    /* db2 += sum_rows(dlogits) */
    for (int j = 0; j < OUTPUT_DIM_PAD; j++) {
        float acc = 0.0f;
        for (int b = 0; b < batch; b++)
            acc += net->dlogits[b * OUTPUT_DIM_PAD + j];
        net->db2[j] += acc;
    }

    /* dhidden = dlogits @ W2 (transpose=0: C = A @ B) */
    matmul(net->dhidden, net->dlogits, net->w2,
           batch, HIDDEN_DIM, OUTPUT_DIM_PAD, 0);

    /* ReLU backward */
    for (int i = 0; i < batch * HIDDEN_DIM; i++)
        if (net->hidden_pre_relu[i] <= 0.0f)
            net->dhidden[i] = 0.0f;

    /* dW1 += dhidden^T @ input (transpose=2: C += A^T @ B) */
    matmul(net->dw1, net->dhidden, input,
           HIDDEN_DIM, INPUT_DIM_PAD, batch, 2);

    /* db1 += sum_rows(dhidden) */
    for (int j = 0; j < HIDDEN_DIM; j++) {
        float acc = 0.0f;
        for (int b = 0; b < batch; b++)
            acc += net->dhidden[b * HIDDEN_DIM + j];
        net->db1[j] += acc;
    }
}

/* ====================================================================
 * SGD update
 * ==================================================================== */

static void sgd_update(float *w, float *grad, int n, float lr) {
    for (int i = 0; i < n; i++) {
        w[i] -= lr * grad[i];
        grad[i] = 0.0f;
    }
}

static void update_weights(network_t *net, float lr) {
    sgd_update(net->w1, net->dw1, HIDDEN_DIM * INPUT_DIM_PAD, lr);
    sgd_update(net->b1, net->db1, HIDDEN_DIM, lr);
    sgd_update(net->w2, net->dw2, OUTPUT_DIM_PAD * HIDDEN_DIM, lr);
    sgd_update(net->b2, net->db2, OUTPUT_DIM_PAD, lr);
}

/* ====================================================================
 * Evaluation
 * ==================================================================== */

static float evaluate(network_t *net, const float *images, const uint8_t *labels,
                      int count, matmul_fn_t matmul) {
    int correct = 0, total = 0;
    /* Use a temporary input buffer for evaluation batches */
    for (int start = 0; start + g_batch_size <= count; start += g_batch_size) {
        const float *batch_input = images + (size_t)start * INPUT_DIM_PAD;
        forward(net, batch_input, g_batch_size, matmul);

        for (int b = 0; b < g_batch_size; b++) {
            float max_val = net->logits[b * OUTPUT_DIM_PAD];
            int max_idx = 0;
            for (int j = 1; j < OUTPUT_DIM; j++) {
                if (net->logits[b * OUTPUT_DIM_PAD + j] > max_val) {
                    max_val = net->logits[b * OUTPUT_DIM_PAD + j];
                    max_idx = j;
                }
            }
            if (max_idx == labels[start + b]) correct++;
            total++;
        }
    }
    return (total > 0) ? (float)correct / (float)total : 0.0f;
}

/* ====================================================================
 * Training loop
 * ==================================================================== */

static int train(network_t *net, int epochs, matmul_fn_t matmul,
                 const char *mode_str,
                 float *train_images, uint8_t *train_labels, int train_count,
                 float *test_images, uint8_t *test_labels, int test_count) {
    int n_batches = train_count / g_batch_size;

    int *indices = (int *)malloc(train_count * sizeof(int));
    /* batch_buf is passed to DSP matmul, so use net_alloc (shared memory) */
    float *batch_buf = (float *)net_alloc(g_batch_size * INPUT_DIM_PAD * sizeof(float));
    uint8_t *label_buf = (uint8_t *)malloc(g_batch_size * sizeof(uint8_t));
    if (!indices || !batch_buf || !label_buf) {
        printf("[ERROR] malloc failed\n");
        free(indices); net_free(batch_buf); free(label_buf);
        return -1;
    }
    for (int i = 0; i < train_count; i++) indices[i] = i;

    printf("\n");
    printf("==========================================================\n");
    printf(" MNIST Training: mode=%s, epochs=%d, batch=%d, lr=%.3f\n",
           mode_str, epochs, g_batch_size, LEARNING_RATE);
    printf(" Network: %d -> %d (ReLU) -> %d (Softmax)\n",
           INPUT_DIM_PAD, HIDDEN_DIM, OUTPUT_DIM);
    printf("==========================================================\n\n");

    for (int epoch = 0; epoch < epochs; epoch++) {
        struct timespec t_epoch_start, t_epoch_end;
        struct timespec t_fwd_start, t_fwd_end;
        struct timespec t_bwd_start, t_bwd_end;
        struct timespec t_upd_start, t_upd_end;

        double fwd_ms_total = 0.0, bwd_ms_total = 0.0, upd_ms_total = 0.0;
        float epoch_loss = 0.0f;
        int epoch_correct = 0, epoch_total = 0;

        shuffle_indices(indices, train_count);
        clock_gettime(CLOCK_MONOTONIC, &t_epoch_start);

        for (int batch_idx = 0; batch_idx < n_batches; batch_idx++) {
            /* Assemble batch */
            for (int b = 0; b < g_batch_size; b++) {
                int idx = indices[batch_idx * g_batch_size + b];
                memcpy(batch_buf + b * INPUT_DIM_PAD,
                       train_images + (size_t)idx * INPUT_DIM_PAD,
                       INPUT_DIM_PAD * sizeof(float));
                label_buf[b] = train_labels[idx];
            }

            /* Forward */
            clock_gettime(CLOCK_MONOTONIC, &t_fwd_start);
            forward(net, batch_buf, g_batch_size, matmul);
            clock_gettime(CLOCK_MONOTONIC, &t_fwd_end);
            fwd_ms_total += time_ms(&t_fwd_start, &t_fwd_end);

            /* Loss */
            float loss = softmax_cross_entropy(net->logits, label_buf,
                                               net->probs, g_batch_size);
            epoch_loss += loss;

            /* Training accuracy */
            for (int b = 0; b < g_batch_size; b++) {
                float max_val = net->logits[b * OUTPUT_DIM_PAD];
                int max_idx = 0;
                for (int j = 1; j < OUTPUT_DIM; j++) {
                    if (net->logits[b * OUTPUT_DIM_PAD + j] > max_val) {
                        max_val = net->logits[b * OUTPUT_DIM_PAD + j];
                        max_idx = j;
                    }
                }
                if (max_idx == label_buf[b]) epoch_correct++;
                epoch_total++;
            }

            /* Backward */
            clock_gettime(CLOCK_MONOTONIC, &t_bwd_start);
            memset(net->dw1, 0, HIDDEN_DIM * INPUT_DIM_PAD * sizeof(float));
            memset(net->db1, 0, HIDDEN_DIM * sizeof(float));
            memset(net->dw2, 0, OUTPUT_DIM_PAD * HIDDEN_DIM * sizeof(float));
            memset(net->db2, 0, OUTPUT_DIM_PAD * sizeof(float));
            backward(net, batch_buf, label_buf, g_batch_size, matmul);
            clock_gettime(CLOCK_MONOTONIC, &t_bwd_end);
            bwd_ms_total += time_ms(&t_bwd_start, &t_bwd_end);

            /* Update */
            clock_gettime(CLOCK_MONOTONIC, &t_upd_start);
            update_weights(net, LEARNING_RATE);
            clock_gettime(CLOCK_MONOTONIC, &t_upd_end);
            upd_ms_total += time_ms(&t_upd_start, &t_upd_end);
        }

        clock_gettime(CLOCK_MONOTONIC, &t_epoch_end);
        double epoch_ms = time_ms(&t_epoch_start, &t_epoch_end);
        float avg_loss = epoch_loss / (float)n_batches;
        float train_acc = (float)epoch_correct / (float)epoch_total;

        /* Use CPU matmul for evaluation to avoid overhead measurement noise */
        float test_acc = evaluate(net, test_images, test_labels, test_count,
                                  cpu_matmul_dispatch);

        printf("Epoch %d/%d: loss=%.4f  train_acc=%.4f  test_acc=%.4f  "
               "time=%.1fms\n",
               epoch + 1, epochs, avg_loss, train_acc, test_acc, epoch_ms);
        printf("  [timing] forward=%.1fms  backward=%.1fms  update=%.1fms\n",
               fwd_ms_total, bwd_ms_total, upd_ms_total);
    }

    free(label_buf);
    net_free(batch_buf);
    free(indices);
    return 0;
}

#endif /* MNIST_NETWORK_H */
