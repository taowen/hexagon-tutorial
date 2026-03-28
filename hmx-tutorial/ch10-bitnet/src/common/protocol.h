/*
 * ch10: ARM <-> DSP shared message protocol for BitNet VLUT16 tests
 */

#ifndef BITNET_PROTOCOL_H
#define BITNET_PROTOCOL_H

#include <stdint.h>

/* Op codes */
#define OP_VLUT16_TEST      1   /* run VLUT16 test on DSP */
#define OP_GEMV_TEST        2   /* run BitNet GEMV test on DSP */
#define OP_ATTN_TEST        3   /* run softmax + attention test on DSP */
#define OP_OPS_TEST         4   /* run f32 operator tests (rmsnorm, relu2, etc.) */
#define OP_DECODER_TEST     5   /* run decoder layer test against PyTorch reference */
#define OP_EXIT             99

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

/*
 * Weight layout header for decoder layer test.
 *
 * All weights and test data are packed into a single contiguous shared
 * memory buffer allocated via rpcmem on the ARM side.  Each offset is
 * a byte offset from the start of the buffer.  The data at each offset
 * is 128-byte aligned for HVX access.
 *
 * The Python prep script writes this header at offset 0, followed by
 * the actual weight/data payloads at the specified offsets.
 */
typedef struct {
    uint32_t total_size;            /* total buffer size in bytes */

    /* Test input/output */
    uint32_t input_offset;          /* [HIDDEN_SIZE] f32 -- decoder input */
    uint32_t ref_output_offset;     /* [HIDDEN_SIZE] f32 -- PyTorch reference output */

    /* Position for RoPE */
    uint32_t pos;                   /* token position in the sequence */

    /* Packed ternary projection weights (bitnet_gemv_opt format) */
    uint32_t q_proj_offset;         /* [2 * (HIDDEN/4) * HIDDEN]           bytes */
    uint32_t k_proj_offset;         /* [2 * (HIDDEN/4) * KV_DIM]           bytes */
    uint32_t v_proj_offset;         /* [2 * (HIDDEN/4) * KV_DIM]           bytes */
    uint32_t o_proj_offset;         /* [2 * (HIDDEN/4) * HIDDEN]           bytes */
    uint32_t gate_proj_offset;      /* [2 * (HIDDEN/4) * INTERMEDIATE]     bytes */
    uint32_t up_proj_offset;        /* [2 * (HIDDEN/4) * INTERMEDIATE]     bytes */
    uint32_t down_proj_offset;      /* [2 * (INTER/4)  * HIDDEN]           bytes */

    /* RMSNorm weights (f32) */
    uint32_t input_ln_offset;       /* [HIDDEN_SIZE] f32 */
    uint32_t post_attn_ln_offset;   /* [HIDDEN_SIZE] f32 */
    uint32_t attn_sub_norm_offset;  /* [HIDDEN_SIZE] f32 */
    uint32_t ffn_sub_norm_offset;   /* [INTERMEDIATE_SIZE] f32 */

    /* RoPE tables (f32) */
    uint32_t rope_cos_offset;       /* [max_pos * HEAD_DIM/2] f32 */
    uint32_t rope_sin_offset;       /* [max_pos * HEAD_DIM/2] f32 */
    uint32_t rope_max_pos;          /* max position entries in RoPE tables */

    /* KV cache initial state (f32, may be zero-filled for pos=0) */
    uint32_t k_cache_offset;        /* [NUM_KV_HEADS * max_seq_len * HEAD_DIM] f32 */
    uint32_t v_cache_offset;        /* [NUM_KV_HEADS * max_seq_len * HEAD_DIM] f32 */
    uint32_t kv_seq_len;            /* current filled positions in KV cache */
    uint32_t kv_max_seq_len;        /* allocated KV cache capacity */
} WeightLayout;

/*
 * Decoder test request message.
 *
 * The shared buffer is passed as dspqueue_buffer[0] in the dspqueue_write
 * call.  The DSP side reads bufs[0].ptr to access the buffer, then
 * interprets it via the WeightLayout at the start.
 */
struct decoder_test_req {
    uint32_t op;                    /* OP_DECODER_TEST */
};

#define MAX_MESSAGE_SIZE   sizeof(struct vlut16_test_req)
#define MAX_RESPONSE_SIZE  sizeof(struct vlut16_test_rsp)
#define MAX_BUFFERS        4

#endif /* BITNET_PROTOCOL_H */
