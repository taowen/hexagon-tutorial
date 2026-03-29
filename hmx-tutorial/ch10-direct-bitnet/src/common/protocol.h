/*
 * ch10: ARM <-> DSP shared message protocol for BitNet VLUT16 tests
 *       and full model inference (experiment 5)
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
#define OP_LOAD_MODEL       6   /* load DSP weights (layers+norms+rope) from shared memory */
#define OP_FORWARD          7   /* run decoder layers + final norm on hidden state */
#define OP_RESET_KV         8   /* reset KV caches (start new sequence) */
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
 * Weight layout header for decoder layer test (experiment 4).
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
 * Decoder test request message (experiment 4).
 *
 * The shared buffer is passed as dspqueue_buffer[0] in the dspqueue_write
 * call.  The DSP side reads bufs[0].ptr to access the buffer, then
 * interprets it via the WeightLayout at the start.
 */
struct decoder_test_req {
    uint32_t op;                    /* OP_DECODER_TEST */
};

/* ========== Experiment 5: Full model inference ========== */

/*
 * Full model header for OP_LOAD_MODEL (DSP weights only).
 *
 * Placed at offset 0 of the shared DSP weight buffer.  Contains model
 * hyperparameters and byte offsets to each component within the buffer.
 *
 * The embedding table is NOT included -- it stays on the ARM side in
 * regular malloc to avoid exceeding DSP virtual address space (~2GB).
 * ARM handles embedding lookup and LM head (tied weights) locally.
 *
 * Memory map (DSP buffer only):
 *   [0 .. sizeof(ModelHeader)-1]             : ModelHeader
 *   [final_norm_offset .. +hidden*4]         : f32 final norm [hidden]
 *   [rope_cos_offset .. +max_pos*half*4]     : f32 rope cos
 *   [rope_sin_offset .. +max_pos*half*4]     : f32 rope sin
 *   [layer_offsets[0] .. +layer_sizes[0]]    : layer 0 data
 *   ...
 *   [layer_offsets[N-1] .. +layer_sizes[N-1]]: layer N-1 data
 */

/*
 * Per-layer weight layout within the model buffer.
 *
 * Each layer_offsets[i] points to a block of memory starting with this
 * header, followed by the actual weight data at the listed sub-offsets
 * (relative to the start of this layer block, NOT the model buffer).
 */
typedef struct {
    /* Packed ternary projection weights (bitnet_gemv_opt format).
     * Sub-offsets are relative to layer_offsets[i]. */
    uint32_t q_proj_offset;
    uint32_t k_proj_offset;
    uint32_t v_proj_offset;
    uint32_t o_proj_offset;
    uint32_t gate_proj_offset;
    uint32_t up_proj_offset;
    uint32_t down_proj_offset;

    /* RMSNorm weights (f32).  Sub-offsets relative to layer_offsets[i]. */
    uint32_t input_ln_offset;
    uint32_t post_attn_ln_offset;
    uint32_t attn_sub_norm_offset;
    uint32_t ffn_sub_norm_offset;
} LayerWeightLayout;

typedef struct {
    uint32_t magic;                 /* 0x424E4554 ("BNET") */
    uint32_t version;               /* 1 */
    uint32_t num_layers;
    uint32_t max_seq_len;           /* max sequence length for KV cache */
    uint32_t hidden_size;           /* 2560 */
    uint32_t intermediate_size;     /* 6912 */
    uint32_t num_heads;             /* 20 */
    uint32_t num_kv_heads;          /* 5 */
    uint32_t head_dim;              /* 128 */
    float    rms_norm_eps;          /* 1e-5 */
    uint32_t total_size;            /* total buffer size in bytes */

    /* Byte offsets from the start of the DSP weight buffer */
    uint64_t final_norm_offset;     /* f32 [hidden_size] float */
    uint64_t rope_cos_offset;       /* f32 [max_seq_len * head_dim/2] float */
    uint64_t rope_sin_offset;       /* f32 [max_seq_len * head_dim/2] float */

    /* Per-layer: offset into the model buffer where LayerWeightLayout begins,
     * followed by the actual weight data. */
    uint64_t layer_offsets[30];     /* byte offset to LayerWeightLayout */
    uint32_t layer_sizes[30];       /* total bytes for each layer block */
} ModelHeader;

#define MODEL_MAGIC  0x424E4554     /* "BNET" */
#define MODEL_VERSION 1

/*
 * OP_LOAD_MODEL request.
 *
 * The shared buffer containing the ModelHeader + DSP weights (layers,
 * norms, RoPE -- NO embedding) is passed as dspqueue_buffer[0].
 * The DSP keeps the buffer referenced so weight pointers remain valid
 * for subsequent OP_FORWARD calls.
 */
struct load_model_req {
    uint32_t op;                    /* OP_LOAD_MODEL */
};

/* OP_LOAD_MODEL response */
struct load_model_rsp {
    uint32_t op;                    /* OP_LOAD_MODEL */
    uint32_t status;                /* 0 = success */
};

/*
 * OP_FORWARD request -- run decoder layers + final norm on hidden state.
 *
 * ARM sends position; the hidden state (HIDDEN_SIZE floats = 10KB) is
 * passed as dspqueue_buffer[0] (small shared buffer).  DSP reads the
 * hidden state, runs all decoder layers + final norm in-place, and
 * writes the result back to the same buffer.
 *
 * The ARM side handles embedding lookup (before) and LM head argmax
 * (after) locally using regular malloc'd embedding table.
 */
struct forward_req {
    uint32_t op;                    /* OP_FORWARD */
    int32_t  position;              /* token position for RoPE/KV cache */
};

/* OP_FORWARD response */
struct forward_rsp {
    uint32_t op;                    /* OP_FORWARD */
    uint32_t status;                /* 0 = success */
    uint32_t total_time_us;         /* total time for layers + norm (us) */
    uint32_t layers_time_us;        /* time for decoder layers only (us) */
};

/* OP_RESET_KV request -- reset KV caches for new sequence */
struct reset_kv_req {
    uint32_t op;                    /* OP_RESET_KV */
};

/* OP_RESET_KV response */
struct reset_kv_rsp {
    uint32_t op;                    /* OP_RESET_KV */
    uint32_t status;                /* 0 = success */
};

/*
 * MAX_MESSAGE_SIZE must cover the largest request struct.
 * forward_req is 8 bytes, but we keep 16 for safety.
 */
#define MAX_MESSAGE_SIZE   16
#define MAX_RESPONSE_SIZE  sizeof(struct forward_rsp)
#define MAX_BUFFERS        4

#endif /* BITNET_PROTOCOL_H */
