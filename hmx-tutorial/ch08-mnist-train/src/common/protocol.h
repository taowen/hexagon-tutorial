/*
 * ch12: ARM <-> DSP shared message protocol for MNIST training
 */

#ifndef MNIST_TRAIN_SHARED_H
#define MNIST_TRAIN_SHARED_H

#include <stdint.h>

/* Op codes */
#define OP_MATMUL          1   /* single matmul (legacy, for FastRPC mode) */
#define OP_REGISTER_NET    2   /* register all network buffer pointers on DSP */
#define OP_TRAIN_BATCH     3   /* fused forward+backward+SGD */
#define OP_SYNC            4   /* flush DSP caches for weight buffers */
#define OP_EVAL            5   /* forward-only evaluation on DSP */
#define OP_TRAIN_ALL       6   /* run all epochs on DSP, single message */

/* Test op codes for adjoint testing on DSP */
#define OP_TEST_RELU_FWD   10  /* bufs: [x_inout], params: n */
#define OP_TEST_RELU_BWD   11  /* bufs: [dx_inout, pre_relu], params: n */
#define OP_TEST_BIAS_BWD   12  /* bufs: [dout, db_out], params: batch, dim */
#define OP_TEST_ADD_BIAS   13  /* bufs: [out_inout, bias], params: batch, dim */
#define OP_TEST_SOFTMAX_CE 14  /* bufs: [logits, probs_out], params: batch, labels[] */
#define OP_TEST_DLOGITS    15  /* bufs: [dlogits_out, probs], params: batch, labels[] */
#define OP_TEST_SGD        16  /* bufs: [w_inout, grad], params: n, lr_bits */

/* Buffer indices for OP_REGISTER_NET (12 buffers) */
#define NET_BUF_W1          0
#define NET_BUF_B1          1
#define NET_BUF_W2          2
#define NET_BUF_B2          3
#define NET_BUF_DW1         4
#define NET_BUF_DW2         5
#define NET_BUF_HIDDEN      6
#define NET_BUF_LOGITS      7
#define NET_BUF_DHIDDEN     8
#define NET_BUF_DLOGITS     9
#define NET_BUF_HIDDEN_PRE 10
#define NET_BUF_PROBS      11
#define NET_BUF_COUNT      12

/* Network dimensions (must match mnist_common.h) */
#define NET_INPUT_DIM_PAD   832
#define NET_HIDDEN_DIM      128
#define NET_OUTPUT_DIM      10
#define NET_OUTPUT_DIM_PAD  64

/* Matmul request (OP_MATMUL) -- unchanged */
struct matmul_req {
    uint32_t op;
    uint32_t m, n, k;
    uint32_t transpose;
    uint32_t accumulate;
    uint32_t reserved[2];
};

/* Register network request (OP_REGISTER_NET)
 * Sent with NET_BUF_COUNT dspqueue buffer references */
struct register_net_req {
    uint32_t op;
    uint32_t reserved[3];
};

/* Train batch request (OP_TRAIN_BATCH)
 * Sent with 1 buffer: input [batch x INPUT_DIM_PAD] */
struct train_batch_req {
    uint32_t op;
    uint32_t batch_size;
    float    learning_rate;
    uint32_t reserved;
    uint8_t  labels[256];
};

/* Sync request (OP_SYNC)
 * Sent with 4 buffers: w1, b1, w2, b2 */
struct sync_req {
    uint32_t op;
    uint32_t reserved[3];
};

/* Train-all request (OP_TRAIN_ALL)
 * Sent with 5 buffers:
 *   [0] = train_all_req (this struct, in rpcmem)
 *   [1] = train_images [num_train_samples x NET_INPUT_DIM_PAD] f16
 *   [2] = train_labels [num_train_samples] uint8
 *   [3] = test_images  [num_test_samples x NET_INPUT_DIM_PAD] f16
 *   [4] = test_labels  [num_test_samples] uint8
 * Response buffer:
 *   [0] = train_all_rsp
 */
#define MAX_EPOCHS 20

struct train_all_req {
    uint32_t opcode;           /* OP_TRAIN_ALL */
    uint32_t num_epochs;
    uint32_t num_train_samples;
    uint32_t num_test_samples;
    uint32_t batch_size;
    float    learning_rate;
    uint32_t reserved[2];
};

struct train_all_rsp {
    float    epoch_losses[MAX_EPOCHS];      /* average loss per epoch */
    float    epoch_train_acc[MAX_EPOCHS];   /* train accuracy per epoch */
    float    epoch_test_acc[MAX_EPOCHS];    /* test accuracy per epoch */
    uint32_t num_epochs_done;
};

/* Response for OP_MATMUL, OP_REGISTER_NET, OP_SYNC */
struct matmul_rsp {
    uint32_t op;
    uint32_t status;
};

/* Response for OP_TRAIN_BATCH */
struct train_batch_rsp {
    uint32_t op;
    uint32_t status;
    float    loss;
    uint32_t correct;
};

/* Test op request (OP_TEST_*) */
struct test_op_req {
    uint32_t op;
    uint32_t param1;       /* n or batch_size */
    uint32_t param2;       /* dim (for bias ops) */
    uint32_t reserved;
    uint8_t  labels[256];  /* for softmax/dlogits */
};

/* Test op response */
struct test_op_rsp {
    uint32_t op;
    uint32_t status;
    float    result;       /* loss for softmax_ce */
    uint32_t reserved;
};

#define MATMUL_MAX_MESSAGE_SIZE  sizeof(struct test_op_req)
#define MATMUL_MAX_RESPONSE_SIZE sizeof(struct train_all_rsp)
#define MATMUL_MAX_BUFFERS       12

#endif /* MNIST_TRAIN_SHARED_H */
