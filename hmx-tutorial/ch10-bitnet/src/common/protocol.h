/*
 * ch10: ARM <-> DSP shared message protocol for BitNet VLUT16 tests
 */

#ifndef BITNET_PROTOCOL_H
#define BITNET_PROTOCOL_H

#include <stdint.h>

/* Op codes */
#define OP_VLUT16_TEST  1   /* run VLUT16 test on DSP */
#define OP_GEMV_TEST    2   /* run BitNet GEMV test on DSP */
#define OP_ATTN_TEST    3   /* run softmax + attention test on DSP */
#define OP_OPS_TEST     4   /* run f32 operator tests (rmsnorm, relu2, etc.) */
#define OP_EXIT         99

/* Request: run a VLUT16 test */
struct vlut16_test_req {
    uint32_t op;
    uint32_t test_id;
};

/* Response: test result */
struct vlut16_test_rsp {
    uint32_t op;
    uint32_t status;
    uint32_t pass;
};

#define MAX_MESSAGE_SIZE   sizeof(struct vlut16_test_req)
#define MAX_RESPONSE_SIZE  sizeof(struct vlut16_test_rsp)
#define MAX_BUFFERS        4

#endif /* BITNET_PROTOCOL_H */
