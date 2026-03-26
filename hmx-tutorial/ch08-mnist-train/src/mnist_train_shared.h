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
#define NET_INPUT_DIM_PAD   800
#define NET_HIDDEN_DIM      128
#define NET_OUTPUT_DIM      10
#define NET_OUTPUT_DIM_PAD  32

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

#define MATMUL_MAX_MESSAGE_SIZE  sizeof(struct train_batch_req)
#define MATMUL_MAX_BUFFERS       12

#endif /* MNIST_TRAIN_SHARED_H */
