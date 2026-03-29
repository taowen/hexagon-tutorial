/*
 * main.c -- ARM driver for ch10 BitNet VLUT16 tests & inference
 *
 * Sets up FastRPC + dspqueue, sends test/inference requests to DSP.
 *
 * Experiments 1-4: unit tests (VLUT16, GEMV, attention, ops, decoder layer)
 * Experiment 5:    full model inference -- loads model weights from files
 *                  into shared memory, runs autoregressive generation.
 *
 * Usage:
 *   ./bitnet_arm                   -- run all unit tests
 *   ./bitnet_arm decoder           -- run only decoder test
 *   ./bitnet_arm generate [N]      -- run inference, generate up to N tokens
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <semaphore.h>
#include <inttypes.h>
#include <sys/time.h>
#include <rpcmem.h>
#include <AEEStdErr.h>
#include "dspqueue.h"
#include "bitnet_test.h"
#include "common/protocol.h"

#define CDSP_DOMAIN_ID_ID 3

/* ========== Model constants ========== */

#define MODEL_DIR        "/data/local/tmp/bitnet_model"
#define NUM_LAYERS       30
#define MAX_SEQ_LEN      128     /* KV cache capacity */
#define VOCAB_SIZE       128256
#define HIDDEN_SIZE_VAL  2560
#define INTERMEDIATE_SIZE_VAL 6912
#define NUM_HEADS_VAL    20
#define NUM_KV_HEADS_VAL 5
#define HEAD_DIM_VAL     128
#define RMS_NORM_EPS     1e-5f

/* Embedding: f16, [vocab_size * hidden_size] */
#define EMBEDDING_SIZE   ((uint64_t)VOCAB_SIZE * HIDDEN_SIZE_VAL * 2)
/* Final norm: f32, [hidden_size] */
#define FINAL_NORM_SIZE  ((uint64_t)HIDDEN_SIZE_VAL * 4)
/* RoPE tables: f32, [max_seq_len * head_dim/2] each */
#define ROPE_TABLE_SIZE  ((uint64_t)MAX_SEQ_LEN * (HEAD_DIM_VAL / 2) * 4)

/* Alignment helper */
#define ALIGN_128(x) (((x) + 127ULL) & ~127ULL)

/* Llama-3 EOS token IDs */
#define EOS_TOKEN_1  128001
#define EOS_TOKEN_2  128009

/* ========== Timing ========== */

static uint64_t get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

/* ========== Shared memory helpers ========== */

/*
 * Allocate shared memory visible to both ARM and DSP.
 * Returns pointer (NULL on failure), sets *out_fd.
 */
static void *shared_alloc(size_t size, int *out_fd) {
    size_t aligned = (size + 4095) & ~4095UL;
    void *p = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                           RPCMEM_DEFAULT_FLAGS | RPCMEM_HEAP_NOREG,
                           aligned);
    if (!p) {
        fprintf(stderr, "[ARM] rpcmem_alloc failed for %zu bytes\n", aligned);
        return NULL;
    }
    int fd = rpcmem_to_fd(p);
    if (fd <= 0) {
        fprintf(stderr, "[ARM] rpcmem_to_fd failed\n");
        rpcmem_free(p);
        return NULL;
    }
    int err = fastrpc_mmap(CDSP_DOMAIN_ID, fd, p, 0, aligned,
                           FASTRPC_MAP_FD);
    if (err != 0) {
        fprintf(stderr, "[ARM] fastrpc_mmap failed: 0x%08x\n", (unsigned)err);
        rpcmem_free(p);
        return NULL;
    }
    *out_fd = fd;
    return p;
}

static void shared_free(void *p, int fd) {
    if (!p) return;
    fastrpc_munmap(CDSP_DOMAIN_ID, fd, NULL, 0);
    rpcmem_free(p);
}

/*
 * Load a binary file into a buffer.
 * Returns bytes read, or -1 on error.
 */
static long load_binary_file(const char *path, void *dst, size_t max_size) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[ARM] Cannot open %s\n", path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size < 0 || (size_t)size > max_size) {
        fprintf(stderr, "[ARM] File %s too large: %ld bytes (max %zu)\n",
                path, size, max_size);
        fclose(f);
        return -1;
    }
    size_t nread = fread(dst, 1, (size_t)size, f);
    fclose(f);
    if ((long)nread != size) {
        fprintf(stderr, "[ARM] Short read on %s: got %zu, expected %ld\n",
                path, nread, size);
        return -1;
    }
    return size;
}

/*
 * Get file size without loading. Returns -1 on error.
 */
static long get_file_size(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fclose(f);
    return size;
}

/* ========== Response handling ========== */

/*
 * Response buffer -- a union large enough for any response type.
 * The callback writes into this, then posts the semaphore.
 */
static sem_t g_done;

static union {
    struct vlut16_test_rsp  test;
    struct load_model_rsp   load;
    struct forward_rsp      fwd;
    uint8_t                 raw[64];
} g_rsp_buf;

/* Convenience aliases for reading responses after sem_wait */
#define g_rsp       g_rsp_buf.test
#define g_load_rsp  g_rsp_buf.load
#define g_fwd_rsp   g_rsp_buf.fwd

static void pkt_cb(dspqueue_t q, AEEResult err, void *ctx) {
    uint32_t flags, msg_len, n_bufs;
    struct dspqueue_buffer bufs[MAX_BUFFERS];

    while (dspqueue_read_noblock(q, &flags, MAX_BUFFERS, &n_bufs, bufs,
                                  sizeof(g_rsp_buf), &msg_len,
                                  g_rsp_buf.raw) == 0) {
        sem_post(&g_done);
    }
}

static void err_cb(dspqueue_t q, AEEResult err, void *ctx) {
    fprintf(stderr, "dspqueue error: 0x%x\n", err);
}

/* ========== Send simple test request (no buffers) ========== */

static int send_simple_test(dspqueue_t queue, uint32_t op, const char *name) {
    printf("[ARM] Sending %s request...\n", name);
    struct vlut16_test_req req;
    memset(&req, 0, sizeof(req));
    req.op = op;
    req.test_id = 0;

    int ret = dspqueue_write(queue, 0, 0, NULL,
                              sizeof(req), (const uint8_t *)&req, 1000000);
    if (ret != 0) {
        fprintf(stderr, "dspqueue_write (%s) failed: 0x%08x\n", name, (unsigned)ret);
        return -1;
    }

    sem_wait(&g_done);
    printf("\n[ARM] %s Response: op=%u status=%u pass=%u\n",
           name, g_rsp.op, g_rsp.status, g_rsp.pass);
    return g_rsp.pass ? 0 : 1;
}

/* ========== Decoder layer test (experiment 4) ========== */

/*
 * Load the decoder_layer.bin file (which contains the WeightLayout header
 * followed by all weight data) into shared memory, then send to DSP.
 *
 * The binary file is prepared by the Python script and pushed to
 * /data/local/tmp/bitnet_weights/decoder_layer.bin
 */
static int run_decoder_layer_test(dspqueue_t queue) {
    const char *weight_path = "/data/local/tmp/bitnet_weights/decoder_layer.bin";
    int ret = -1;
    void *shared_buf = NULL;
    int shared_fd = -1;

    printf("[ARM] === Decoder Layer Test ===\n");

    /* First, get file size */
    FILE *f = fopen(weight_path, "rb");
    if (!f) {
        fprintf(stderr, "[ARM] Cannot open %s -- skipping decoder test\n", weight_path);
        fprintf(stderr, "[ARM] (Run the Python prep script first)\n");
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fclose(f);

    if (file_size <= 0 || file_size > 256 * 1024 * 1024) {
        fprintf(stderr, "[ARM] Invalid file size: %ld\n", file_size);
        return -1;
    }

    printf("[ARM] Weight file: %s (%ld bytes = %.1f MB)\n",
           weight_path, file_size, (double)file_size / (1024.0 * 1024.0));

    /* Allocate shared memory */
    shared_buf = shared_alloc((size_t)file_size, &shared_fd);
    if (!shared_buf) {
        fprintf(stderr, "[ARM] Failed to allocate shared memory\n");
        return -1;
    }

    /* Load file into shared memory */
    long loaded = load_binary_file(weight_path, shared_buf, (size_t)file_size);
    if (loaded != file_size) {
        fprintf(stderr, "[ARM] Failed to load weight file\n");
        goto cleanup;
    }

    /* Verify header */
    const WeightLayout *layout = (const WeightLayout *)shared_buf;
    printf("[ARM] WeightLayout: total_size=%u, pos=%u, kv_seq_len=%u\n",
           layout->total_size, layout->pos, layout->kv_seq_len);

    if (layout->total_size != (uint32_t)file_size) {
        fprintf(stderr, "[ARM] WARNING: header total_size (%u) != file_size (%ld)\n",
                layout->total_size, file_size);
    }

    /* Send decoder test request with shared buffer */
    printf("[ARM] Sending decoder test to DSP...\n");

    struct decoder_test_req req;
    memset(&req, 0, sizeof(req));
    req.op = OP_DECODER_TEST;

    struct dspqueue_buffer bufs[1];
    memset(bufs, 0, sizeof(bufs));
    bufs[0].fd = shared_fd;
    bufs[0].flags = DSPQUEUE_BUFFER_FLAG_REF
                  | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER;

    ret = dspqueue_write(queue, 0, 1, bufs,
                          sizeof(req), (const uint8_t *)&req, 30000000);
    if (ret != 0) {
        fprintf(stderr, "[ARM] dspqueue_write (decoder) failed: 0x%08x\n", (unsigned)ret);
        goto cleanup;
    }

    /* Wait for response (decoder layer can take a while) */
    sem_wait(&g_done);

    printf("[ARM] Decoder Response: op=%u status=%u pass=%u\n",
           g_rsp.op, g_rsp.status, g_rsp.pass);
    ret = g_rsp.pass ? 0 : 1;

cleanup:
    shared_free(shared_buf, shared_fd);
    return ret;
}

/* ========== Full model inference (experiment 5) ========== */

/* ---- f16 -> f32 conversion (for ARM-side embedding/LM head) ---- */

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            float r; memcpy(&r, &sign, 4); return r;
        }
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= 0x3FF;
    } else if (exp == 31) {
        uint32_t r = sign | 0x7F800000 | (mant << 13);
        float rf; memcpy(&rf, &r, 4); return rf;
    }

    uint32_t r = sign | ((exp + 112) << 23) | (mant << 13);
    float rf; memcpy(&rf, &r, 4); return rf;
}

/* ---- ARM-side embedding lookup: f16 -> f32 ---- */

static void embedding_lookup_arm(const uint16_t *embedding, int token_id,
                                  float *out) {
    const uint16_t *row = embedding + (size_t)token_id * HIDDEN_SIZE_VAL;
    for (int i = 0; i < HIDDEN_SIZE_VAL; i++) {
        out[i] = f16_to_f32(row[i]);
    }
}

/* ---- ARM-side LM head: greedy argmax over tied embedding ---- */

static int lm_head_greedy_arm(const uint16_t *embedding, const float *hidden) {
    float best = -1e30f;
    int best_id = 0;
    for (int v = 0; v < VOCAB_SIZE; v++) {
        const uint16_t *row = embedding + (size_t)v * HIDDEN_SIZE_VAL;
        float score = 0;
        for (int i = 0; i < HIDDEN_SIZE_VAL; i++) {
            score += f16_to_f32(row[i]) * hidden[i];
        }
        if (score > best) { best = score; best_id = v; }
    }
    return best_id;
}

/* ---- Load embedding into regular malloc (ARM-only, not shared with DSP) ---- */

static uint16_t *load_embedding(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/embedding.bin", MODEL_DIR);

    long size = get_file_size(path);
    if (size < 0) {
        fprintf(stderr, "[ARM] Cannot find %s\n", path);
        fprintf(stderr, "[ARM] Make sure model files are pushed to %s/\n", MODEL_DIR);
        return NULL;
    }

    printf("[ARM] Loading embedding into regular malloc (%ld bytes = %.1f MB)...\n",
           size, size / (1024.0 * 1024.0));

    uint16_t *emb = (uint16_t *)malloc((size_t)size);
    if (!emb) {
        fprintf(stderr, "[ARM] malloc failed for embedding (%ld bytes)\n", size);
        return NULL;
    }

    if (load_binary_file(path, emb, (size_t)size) < 0) {
        free(emb);
        return NULL;
    }

    printf("[ARM] Embedding loaded (regular malloc, not shared with DSP).\n");
    return emb;
}

/*
 * Load DSP weights (layers + norms + RoPE) into shared memory (rpcmem).
 * Does NOT include embedding -- that stays on ARM side.
 *
 * File layout on device:
 *   MODEL_DIR/final_norm.bin      -- f32, [2560 * 4] bytes
 *   MODEL_DIR/rope_cos.bin        -- f32, [max_seq_len * 64 * 4] bytes
 *   MODEL_DIR/rope_sin.bin        -- f32, [max_seq_len * 64 * 4] bytes
 *   MODEL_DIR/layer_00.bin .. layer_29.bin -- packed layer weights
 *
 * Returns shared memory pointer (caller must shared_free), or NULL on error.
 * Sets *out_fd, *out_total_size.
 */
static void *load_dsp_weights(int *out_fd, uint64_t *out_total_size) {
    char path[512];
    int i;

    printf("[ARM] === Loading DSP Weights from %s ===\n", MODEL_DIR);

    /* --- Phase 1: compute total buffer size (NO embedding) --- */

    snprintf(path, sizeof(path), "%s/final_norm.bin", MODEL_DIR);
    long norm_size = get_file_size(path);
    if (norm_size < 0) {
        fprintf(stderr, "[ARM] Cannot find %s\n", path);
        return NULL;
    }

    snprintf(path, sizeof(path), "%s/rope_cos.bin", MODEL_DIR);
    long rcos_size = get_file_size(path);
    if (rcos_size < 0) {
        fprintf(stderr, "[ARM] Cannot find %s\n", path);
        return NULL;
    }

    snprintf(path, sizeof(path), "%s/rope_sin.bin", MODEL_DIR);
    long rsin_size = get_file_size(path);
    if (rsin_size < 0) {
        fprintf(stderr, "[ARM] Cannot find %s\n", path);
        return NULL;
    }

    long layer_file_sizes[NUM_LAYERS];
    for (i = 0; i < NUM_LAYERS; i++) {
        snprintf(path, sizeof(path), "%s/layer_%02d.bin", MODEL_DIR, i);
        layer_file_sizes[i] = get_file_size(path);
        if (layer_file_sizes[i] < 0) {
            fprintf(stderr, "[ARM] Cannot find %s\n", path);
            return NULL;
        }
    }

    /* Compute offsets (header first, then each component aligned to 128) */
    uint64_t offset = ALIGN_128(sizeof(ModelHeader));

    uint64_t norm_off = offset;
    offset += ALIGN_128((uint64_t)norm_size);

    uint64_t rcos_off = offset;
    offset += ALIGN_128((uint64_t)rcos_size);

    uint64_t rsin_off = offset;
    offset += ALIGN_128((uint64_t)rsin_size);

    uint64_t layer_offs[NUM_LAYERS];
    for (i = 0; i < NUM_LAYERS; i++) {
        layer_offs[i] = offset;
        offset += ALIGN_128((uint64_t)layer_file_sizes[i]);
    }

    uint64_t total_size = offset;

    printf("[ARM] DSP weight buffer layout (NO embedding):\n");
    printf("[ARM]   Header:         0 .. %llu\n",
           (unsigned long long)ALIGN_128(sizeof(ModelHeader)));
    printf("[ARM]   Final norm:     %llu (%ld bytes)\n",
           (unsigned long long)norm_off, norm_size);
    printf("[ARM]   RoPE cos:       %llu (%ld bytes)\n",
           (unsigned long long)rcos_off, rcos_size);
    printf("[ARM]   RoPE sin:       %llu (%ld bytes)\n",
           (unsigned long long)rsin_off, rsin_size);
    for (i = 0; i < NUM_LAYERS; i++) {
        printf("[ARM]   Layer %02d:       %llu (%ld bytes = %.1f MB)\n",
               i, (unsigned long long)layer_offs[i],
               layer_file_sizes[i], layer_file_sizes[i] / (1024.0 * 1024.0));
    }
    printf("[ARM]   Total:          %llu bytes = %.1f MB\n",
           (unsigned long long)total_size, total_size / (1024.0 * 1024.0));

    /* --- Phase 2: allocate shared memory --- */

    int shared_fd = -1;
    void *buf = shared_alloc((size_t)total_size, &shared_fd);
    if (!buf) {
        fprintf(stderr, "[ARM] Failed to allocate %.1f MB of shared memory\n",
                total_size / (1024.0 * 1024.0));
        fprintf(stderr, "[ARM] This may exceed the device's ION/dmabuf limit.\n");
        return NULL;
    }
    /* --- Phase 3: fill the header --- */

    ModelHeader *hdr = (ModelHeader *)buf;
    hdr->magic              = MODEL_MAGIC;
    hdr->version            = MODEL_VERSION;
    hdr->num_layers         = NUM_LAYERS;
    hdr->max_seq_len        = MAX_SEQ_LEN;
    hdr->hidden_size        = HIDDEN_SIZE_VAL;
    hdr->intermediate_size  = INTERMEDIATE_SIZE_VAL;
    hdr->num_heads          = NUM_HEADS_VAL;
    hdr->num_kv_heads       = NUM_KV_HEADS_VAL;
    hdr->head_dim           = HEAD_DIM_VAL;
    hdr->rms_norm_eps       = RMS_NORM_EPS;
    hdr->total_size         = (uint32_t)total_size;

    hdr->final_norm_offset  = norm_off;
    hdr->rope_cos_offset    = rcos_off;
    hdr->rope_sin_offset    = rsin_off;

    for (i = 0; i < NUM_LAYERS; i++) {
        hdr->layer_offsets[i] = layer_offs[i];
        hdr->layer_sizes[i]   = (uint32_t)layer_file_sizes[i];
    }

    /* --- Phase 4: load files into staging buffer (regular malloc),
     * then memcpy to shared memory.
     *
     * Loading directly into shared memory via fread() is very slow due to
     * the rpcmem write penalty (~23x slower than regular malloc).  By staging
     * through regular malloc first, we concentrate shared memory writes into
     * fast sequential memcpy calls and keep the dspqueue from timing out. */

    printf("[ARM] Loading DSP weight files (staging via regular malloc)...\n");
    uint64_t t_stage_start = get_time_us();

    /* Allocate staging buffer in regular malloc */
    uint8_t *stage = (uint8_t *)malloc((size_t)total_size);
    if (!stage) {
        fprintf(stderr, "[ARM] Failed to allocate %.1f MB staging buffer\n",
                total_size / (1024.0 * 1024.0));
        goto fail;
    }

    /* Copy header from shared memory to staging (already filled) */
    memcpy(stage, buf, ALIGN_128(sizeof(ModelHeader)));

    snprintf(path, sizeof(path), "%s/final_norm.bin", MODEL_DIR);
    printf("[ARM]   Loading final_norm (%ld bytes)...\n", norm_size);
    if (load_binary_file(path, stage + norm_off, (size_t)norm_size) < 0) {
        free(stage);
        goto fail;
    }

    snprintf(path, sizeof(path), "%s/rope_cos.bin", MODEL_DIR);
    printf("[ARM]   Loading rope_cos (%ld bytes)...\n", rcos_size);
    if (load_binary_file(path, stage + rcos_off, (size_t)rcos_size) < 0) {
        free(stage);
        goto fail;
    }

    snprintf(path, sizeof(path), "%s/rope_sin.bin", MODEL_DIR);
    printf("[ARM]   Loading rope_sin (%ld bytes)...\n", rsin_size);
    if (load_binary_file(path, stage + rsin_off, (size_t)rsin_size) < 0) {
        free(stage);
        goto fail;
    }

    for (i = 0; i < NUM_LAYERS; i++) {
        snprintf(path, sizeof(path), "%s/layer_%02d.bin", MODEL_DIR, i);
        printf("[ARM]   Loading layer_%02d (%ld bytes)...\n", i, layer_file_sizes[i]);
        if (load_binary_file(path, stage + layer_offs[i],
                             (size_t)layer_file_sizes[i]) < 0) {
            free(stage);
            goto fail;
        }
    }

    uint64_t t_stage_end = get_time_us();
    printf("[ARM] Files loaded into staging buffer in %.1f seconds.\n",
           (t_stage_end - t_stage_start) / 1000000.0);

    /* Now copy from staging to shared memory in one shot */
    printf("[ARM] Copying %.1f MB from staging to shared memory...\n",
           total_size / (1024.0 * 1024.0));
    uint64_t t_copy_start = get_time_us();
    memcpy(buf, stage, (size_t)total_size);
    uint64_t t_copy_end = get_time_us();
    printf("[ARM] Shared memory copy done in %.1f seconds (%.0f MB/s).\n",
           (t_copy_end - t_copy_start) / 1000000.0,
           total_size / ((t_copy_end - t_copy_start) / 1000000.0) / (1024.0 * 1024.0));

    free(stage);
    printf("[ARM] All DSP weight files loaded successfully.\n");

    *out_fd = shared_fd;
    *out_total_size = total_size;
    return buf;

fail:
    shared_free(buf, shared_fd);
    return NULL;
}

/*
 * Load pre-tokenized input from a file (array of int32 token IDs).
 * Returns malloc'd array, sets *out_count. Caller must free().
 */
static int32_t *load_input_tokens(int *out_count) {
    char path[512];
    snprintf(path, sizeof(path), "%s/input_tokens.bin", MODEL_DIR);

    long fsize = get_file_size(path);
    if (fsize < 0) {
        fprintf(stderr, "[ARM] Cannot find %s\n", path);
        return NULL;
    }
    if (fsize == 0 || fsize % 4 != 0) {
        fprintf(stderr, "[ARM] Invalid input_tokens.bin size: %ld (must be multiple of 4)\n", fsize);
        return NULL;
    }

    int count = (int)(fsize / 4);
    int32_t *tokens = (int32_t *)malloc((size_t)fsize);
    if (!tokens) {
        fprintf(stderr, "[ARM] malloc failed for input tokens\n");
        return NULL;
    }

    if (load_binary_file(path, tokens, (size_t)fsize) < 0) {
        free(tokens);
        return NULL;
    }

    *out_count = count;
    return tokens;
}

/*
 * Save output token IDs to a file.
 */
static int save_output_tokens(const int32_t *tokens, int count) {
    char path[512];
    snprintf(path, sizeof(path), "%s/output_tokens.bin", MODEL_DIR);

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[ARM] Cannot create %s\n", path);
        return -1;
    }
    size_t written = fwrite(tokens, sizeof(int32_t), (size_t)count, f);
    fclose(f);

    if ((int)written != count) {
        fprintf(stderr, "[ARM] Short write on %s\n", path);
        return -1;
    }

    printf("[ARM] Saved %d output tokens to %s\n", count, path);
    return 0;
}

/*
 * Run full model inference with ARM/DSP split.
 *
 * All data (embedding, DSP weights, hidden buffer) must be pre-loaded before
 * calling this function.  This avoids dspqueue timeout during slow loading.
 *
 * Steps:
 * 1. Send OP_LOAD_MODEL to DSP (DSP weights only)
 * 2. Load pre-tokenized input
 * 3. For each token:
 *    a. ARM: embedding_lookup(embedding, token_id, hidden_buf)
 *    b. Send OP_FORWARD with hidden_buf ref
 *    c. DSP runs 30 layers + final norm, writes to hidden_buf
 *    d. ARM: lm_head_greedy(embedding, hidden_buf) -> next_token_id
 * 4. Save output tokens
 */
static int run_generate(dspqueue_t queue, int max_gen_tokens,
                         uint16_t *embedding,
                         void *dsp_buf, int dsp_fd,
                         float *hidden_buf, int hidden_fd) {
    int ret = -1;
    int32_t *input_tokens = NULL;
    int32_t *output_tokens = NULL;

    /* Step 1: Send OP_LOAD_MODEL to DSP (DSP weights only) */
    printf("[ARM] Sending OP_LOAD_MODEL to DSP...\n");
    {
        struct load_model_req req;
        memset(&req, 0, sizeof(req));
        req.op = OP_LOAD_MODEL;

        struct dspqueue_buffer bufs[1];
        memset(bufs, 0, sizeof(bufs));
        bufs[0].fd = dsp_fd;
        bufs[0].flags = DSPQUEUE_BUFFER_FLAG_REF
                      | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER;

        int err = dspqueue_write(queue, 0, 1, bufs,
                                 sizeof(req), (const uint8_t *)&req,
                                 60000000);  /* 60s timeout */
        if (err != 0) {
            fprintf(stderr, "[ARM] dspqueue_write (OP_LOAD_MODEL) failed: 0x%08x\n",
                    (unsigned)err);
            goto cleanup;
        }
    }

    sem_wait(&g_done);
    if (g_load_rsp.status != 0) {
        fprintf(stderr, "[ARM] OP_LOAD_MODEL failed on DSP: status=%u\n",
                g_load_rsp.status);
        goto cleanup;
    }
    printf("[ARM] Model loaded on DSP successfully.\n\n");

    /* Step 5: Load pre-tokenized input */
    int num_input = 0;
    input_tokens = load_input_tokens(&num_input);
    if (!input_tokens || num_input == 0) {
        fprintf(stderr, "[ARM] No input tokens found. Create %s/input_tokens.bin "
                "using tokenize.py\n", MODEL_DIR);
        goto cleanup;
    }
    printf("[ARM] Input tokens (%d):", num_input);
    for (int i = 0; i < num_input && i < 16; i++)
        printf(" %d", input_tokens[i]);
    if (num_input > 16) printf(" ...");
    printf("\n\n");

    /* Step 6: Prefill -- process all input tokens */
    printf("[ARM] --- Prefill phase (%d tokens) ---\n", num_input);
    uint64_t t_prefill_start = get_time_us();

    int pos = 0;
    int32_t last_token = 0;

    for (int i = 0; i < num_input; i++) {
        /* a. ARM: embedding lookup */
        uint64_t t_emb_start = get_time_us();
        embedding_lookup_arm(embedding, input_tokens[i], hidden_buf);
        uint64_t t_emb_end = get_time_us();

        /* b. Send OP_FORWARD with hidden_buf ref */
        struct forward_req req;
        memset(&req, 0, sizeof(req));
        req.op = OP_FORWARD;
        req.position = pos;

        struct dspqueue_buffer bufs[1];
        memset(bufs, 0, sizeof(bufs));
        bufs[0].fd = hidden_fd;
        bufs[0].flags = DSPQUEUE_BUFFER_FLAG_REF
                      | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER;

        int err = dspqueue_write(queue, 0, 1, bufs,
                                 sizeof(req), (const uint8_t *)&req,
                                 60000000);
        if (err != 0) {
            fprintf(stderr, "[ARM] dspqueue_write (prefill %d) failed: 0x%08x\n",
                    i, (unsigned)err);
            goto cleanup;
        }

        sem_wait(&g_done);

        if (g_fwd_rsp.status != 0) {
            fprintf(stderr, "[ARM] OP_FORWARD failed at prefill %d: status=%u\n",
                    i, g_fwd_rsp.status);
            goto cleanup;
        }

        /* c. ARM: LM head greedy argmax */
        uint64_t t_lm_start = get_time_us();
        last_token = lm_head_greedy_arm(embedding, hidden_buf);
        uint64_t t_lm_end = get_time_us();

        printf("[ARM] Prefill [%d/%d] pos=%d token=%d -> next=%d "
               "(emb=%.1f ms, dsp=%.1f ms, lmhead=%.1f ms)\n",
               i + 1, num_input, pos, input_tokens[i], last_token,
               (t_emb_end - t_emb_start) / 1000.0f,
               g_fwd_rsp.total_time_us / 1000.0f,
               (t_lm_end - t_lm_start) / 1000.0f);
        pos++;
    }

    uint64_t t_prefill_end = get_time_us();
    double prefill_ms = (t_prefill_end - t_prefill_start) / 1000.0;
    printf("[ARM] Prefill done: %d tokens in %.1f ms (%.1f ms/token)\n\n",
           num_input, prefill_ms, prefill_ms / num_input);

    /* Step 7: Generate -- autoregressive decoding */
    printf("[ARM] --- Generation phase (up to %d tokens) ---\n", max_gen_tokens);

    output_tokens = (int32_t *)malloc((size_t)max_gen_tokens * sizeof(int32_t));
    if (!output_tokens) {
        fprintf(stderr, "[ARM] malloc failed for output tokens\n");
        goto cleanup;
    }

    int num_output = 0;
    uint64_t t_gen_start = get_time_us();
    uint64_t total_dsp_us = 0;
    uint64_t total_lm_us = 0;

    for (int i = 0; i < max_gen_tokens; i++) {
        /* a. ARM: embedding lookup */
        uint64_t t_emb_start = get_time_us();
        embedding_lookup_arm(embedding, last_token, hidden_buf);
        uint64_t t_emb_end = get_time_us();

        /* b. Send OP_FORWARD with hidden_buf ref */
        struct forward_req req;
        memset(&req, 0, sizeof(req));
        req.op = OP_FORWARD;
        req.position = pos;

        struct dspqueue_buffer bufs[1];
        memset(bufs, 0, sizeof(bufs));
        bufs[0].fd = hidden_fd;
        bufs[0].flags = DSPQUEUE_BUFFER_FLAG_REF
                      | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER;

        int err = dspqueue_write(queue, 0, 1, bufs,
                                 sizeof(req), (const uint8_t *)&req,
                                 60000000);
        if (err != 0) {
            fprintf(stderr, "[ARM] dspqueue_write (gen %d) failed: 0x%08x\n",
                    i, (unsigned)err);
            break;
        }

        sem_wait(&g_done);

        if (g_fwd_rsp.status != 0) {
            fprintf(stderr, "[ARM] OP_FORWARD failed at gen %d: status=%u\n",
                    i, g_fwd_rsp.status);
            break;
        }

        /* c. ARM: LM head greedy argmax */
        uint64_t t_lm_start = get_time_us();
        int next_token = lm_head_greedy_arm(embedding, hidden_buf);
        uint64_t t_lm_end = get_time_us();

        output_tokens[num_output] = next_token;
        num_output++;
        total_dsp_us += g_fwd_rsp.total_time_us;
        total_lm_us += (t_lm_end - t_lm_start);

        printf("[ARM] Gen [%d] pos=%d token=%d "
               "(emb=%.1f ms, dsp=%.1f ms, lmhead=%.1f ms)\n",
               i, pos, next_token,
               (t_emb_end - t_emb_start) / 1000.0f,
               g_fwd_rsp.total_time_us / 1000.0f,
               (t_lm_end - t_lm_start) / 1000.0f);

        last_token = next_token;
        pos++;

        /* Check for EOS */
        if (last_token == EOS_TOKEN_1 || last_token == EOS_TOKEN_2) {
            printf("[ARM] EOS token encountered, stopping generation.\n");
            break;
        }

        /* Safety: stop if position exceeds KV cache capacity */
        if (pos >= MAX_SEQ_LEN) {
            printf("[ARM] Reached max sequence length (%d), stopping.\n", MAX_SEQ_LEN);
            break;
        }
    }

    uint64_t t_gen_end = get_time_us();
    double gen_ms = (t_gen_end - t_gen_start) / 1000.0;

    printf("\n[ARM] === Generation Summary ===\n");
    printf("[ARM] Prefill: %d tokens in %.1f ms (%.1f ms/token)\n",
           num_input, prefill_ms, prefill_ms / num_input);
    printf("[ARM] Generated: %d tokens in %.1f ms", num_output, gen_ms);
    if (num_output > 0) {
        printf(" (%.1f ms/token, %.1f tokens/sec)\n",
               gen_ms / num_output, num_output * 1000.0 / gen_ms);
        printf("[ARM] DSP layers: %.1f ms avg/token\n",
               total_dsp_us / 1000.0 / num_output);
        printf("[ARM] ARM lmhead: %.1f ms avg/token\n",
               total_lm_us / 1000.0 / num_output);
    } else {
        printf("\n");
    }

    printf("[ARM] Output tokens (%d):", num_output);
    for (int i = 0; i < num_output && i < 32; i++)
        printf(" %d", output_tokens[i]);
    if (num_output > 32) printf(" ...");
    printf("\n");

    /* Step 8: Save output tokens */
    if (num_output > 0) {
        save_output_tokens(output_tokens, num_output);
    }

    ret = 0;

cleanup:
    free(input_tokens);
    free(output_tokens);

    /* Note: embedding, dsp_buf, hidden_buf are owned by the caller (main). */
    return ret;
}

/* ========== Main ========== */

static void print_usage(const char *progname) {
    printf("Usage:\n");
    printf("  %s                    -- run all unit tests (exp 1-4)\n", progname);
    printf("  %s vlut16             -- run VLUT16 test only\n", progname);
    printf("  %s gemv               -- run GEMV test only\n", progname);
    printf("  %s attn               -- run attention test only\n", progname);
    printf("  %s ops                -- run operator tests only\n", progname);
    printf("  %s decoder            -- run decoder layer test only\n", progname);
    printf("  %s generate [N]       -- run full inference, generate up to N tokens (default 32)\n", progname);
    printf("\n");
}

int main(int argc, char **argv) {
    int ret = 0;
    remote_handle64 handle = 0;
    dspqueue_t queue = NULL;

    printf("=== ch10: BitNet VLUT16 Exploration ===\n\n");

    /* Parse command */
    enum { CMD_ALL_TESTS, CMD_DECODER, CMD_GENERATE,
           CMD_VLUT16, CMD_GEMV, CMD_ATTN, CMD_OPS } command = CMD_ALL_TESTS;
    int max_gen_tokens = 32;

    if (argc >= 2) {
        if (strcmp(argv[1], "decoder") == 0) {
            command = CMD_DECODER;
        } else if (strcmp(argv[1], "generate") == 0) {
            command = CMD_GENERATE;
            if (argc >= 3) {
                max_gen_tokens = atoi(argv[2]);
                if (max_gen_tokens <= 0) max_gen_tokens = 32;
            }
        } else if (strcmp(argv[1], "vlut16") == 0) {
            command = CMD_VLUT16;
        } else if (strcmp(argv[1], "gemv") == 0) {
            command = CMD_GEMV;
        } else if (strcmp(argv[1], "attn") == 0) {
            command = CMD_ATTN;
        } else if (strcmp(argv[1], "ops") == 0) {
            command = CMD_OPS;
        } else if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown command: %s\n\n", argv[1]);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Enable unsigned module loading */
    struct remote_rpc_control_unsigned_module udata;
    udata.domain = CDSP_DOMAIN_ID;
    udata.enable = 1;
    remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&udata, sizeof(udata));

    /* Open FastRPC session */
    char uri[256];
    snprintf(uri, sizeof(uri), "%s&_dom=cdsp", bitnet_test_URI);
    ret = bitnet_test_open(uri, &handle);
    if (ret != 0) {
        fprintf(stderr, "bitnet_test_open failed: 0x%08x\n", (unsigned)ret);
        return 1;
    }
    printf("[ARM] FastRPC session opened\n");

    /* Enable FastRPC QoS */
    struct remote_rpc_control_latency ldata = { .enable = 1 };
    remote_handle64_control(handle, DSPRPC_CONTROL_LATENCY, &ldata, sizeof(ldata));

    /* Init shared memory */
    rpcmem_init();

    /*
     * For generate mode: load all model data BEFORE creating dspqueue.
     * Loading ~1.6 GB from flash into shared memory takes a long time
     * (rpcmem write penalty), and the dspqueue connection will time out
     * if idle for too long (~60 seconds).  By loading first, we ensure
     * the queue is created right before we start sending messages.
     */
    uint16_t *gen_embedding = NULL;
    void *gen_dsp_buf = NULL;
    int gen_dsp_fd = -1;
    uint64_t gen_dsp_total = 0;
    float *gen_hidden_buf = NULL;
    int gen_hidden_fd = -1;

    if (command == CMD_GENERATE) {
        printf("[ARM] === Full Model Inference (experiment 5) ===\n");
        printf("[ARM] === ARM/DSP split: embedding on ARM, layers on DSP ===\n");
        printf("[ARM] Max generation tokens: %d\n", max_gen_tokens);

        uint64_t t_load_start = get_time_us();

        /* Load embedding into regular malloc (ARM-only, 626 MB) */
        gen_embedding = load_embedding();
        if (!gen_embedding) {
            fprintf(stderr, "[ARM] Failed to load embedding -- aborting\n");
            goto cleanup;
        }

        /* Load DSP weights into shared memory (~1 GB, no embedding) */
        gen_dsp_buf = load_dsp_weights(&gen_dsp_fd, &gen_dsp_total);
        if (!gen_dsp_buf) {
            fprintf(stderr, "[ARM] Failed to load DSP weights -- aborting\n");
            goto cleanup;
        }

        /* Allocate small shared buffer for hidden state I/O (10 KB) */
        gen_hidden_buf = (float *)shared_alloc(HIDDEN_SIZE_VAL * sizeof(float),
                                                &gen_hidden_fd);
        if (!gen_hidden_buf) {
            fprintf(stderr, "[ARM] Failed to allocate hidden state buffer\n");
            goto cleanup;
        }

        uint64_t t_load_end = get_time_us();
        printf("[ARM] All data loaded in %.1f seconds\n\n",
               (t_load_end - t_load_start) / 1000000.0);
    }

    /* Create dspqueue -- now that data is loaded, queue won't time out */
    sem_init(&g_done, 0, 0);
    uint32_t queue_size = (command == CMD_GENERATE) ? 8192 : 4096;
    ret = dspqueue_create(CDSP_DOMAIN_ID, 0,
                          queue_size, queue_size,
                          pkt_cb, err_cb, NULL,
                          &queue);
    if (ret != 0) {
        fprintf(stderr, "dspqueue_create failed: 0x%08x\n", (unsigned)ret);
        goto cleanup;
    }
    printf("[ARM] dspqueue created\n");

    /* Export and pass queue to DSP */
    uint64_t dsp_queue_id;
    ret = dspqueue_export(queue, &dsp_queue_id);
    if (ret != 0) {
        fprintf(stderr, "dspqueue_export failed: 0x%08x\n", (unsigned)ret);
        goto cleanup;
    }

    ret = bitnet_test_start(handle, dsp_queue_id);
    if (ret != 0) {
        fprintf(stderr, "bitnet_test_start failed: 0x%08x\n", (unsigned)ret);
        goto cleanup;
    }
    printf("[ARM] DSP started, queue connected\n\n");

    int all_pass = 1;

    switch (command) {
    case CMD_GENERATE:
        /* Full inference mode -- data already pre-loaded above */
        if (run_generate(queue, max_gen_tokens,
                          gen_embedding, gen_dsp_buf, gen_dsp_fd,
                          gen_hidden_buf, gen_hidden_fd) != 0) {
            all_pass = 0;
        }
        break;

    case CMD_DECODER:
        /* Decoder test only */
        {
            int dec_ret = run_decoder_layer_test(queue);
            if (dec_ret < 0) {
                printf("[ARM] Decoder test skipped (weight file not found)\n");
            } else if (dec_ret != 0) {
                all_pass = 0;
            }
        }
        break;

    case CMD_VLUT16:
        if (send_simple_test(queue, OP_VLUT16_TEST, "VLUT16") != 0) all_pass = 0;
        break;

    case CMD_GEMV:
        if (send_simple_test(queue, OP_GEMV_TEST, "GEMV") != 0) all_pass = 0;
        break;

    case CMD_ATTN:
        if (send_simple_test(queue, OP_ATTN_TEST, "Attention") != 0) all_pass = 0;
        break;

    case CMD_OPS:
        if (send_simple_test(queue, OP_OPS_TEST, "Ops") != 0) all_pass = 0;
        break;

    case CMD_ALL_TESTS:
    default:
        /* Run all unit tests */
        if (send_simple_test(queue, OP_VLUT16_TEST, "VLUT16") != 0)
            all_pass = 0;
        if (send_simple_test(queue, OP_GEMV_TEST, "GEMV") != 0)
            all_pass = 0;
        if (send_simple_test(queue, OP_ATTN_TEST, "Attention") != 0)
            all_pass = 0;
        if (send_simple_test(queue, OP_OPS_TEST, "Ops") != 0)
            all_pass = 0;

        /* Decoder layer test (experiment 4) */
        {
            int dec_ret = run_decoder_layer_test(queue);
            if (dec_ret < 0) {
                printf("[ARM] Decoder test skipped (weight file not found)\n");
            } else if (dec_ret != 0) {
                all_pass = 0;
            }
        }
        break;
    }

    if (command != CMD_GENERATE) {
        if (all_pass) {
            printf("[ARM] *** ALL TESTS PASSED ***\n");
        } else {
            printf("[ARM] *** SOME TESTS FAILED -- check logcat for details ***\n");
        }
    }

cleanup:
    printf("\n[ARM] Cleaning up...\n");

    if (handle) {
        uint64 process_time = 0;
        bitnet_test_stop(handle, &process_time);
        printf("[ARM] DSP total processing time: %llu us\n", (unsigned long long)process_time);
    }

    if (queue) {
        dspqueue_close(queue);
    }

    /* Free generate-mode data (pre-loaded before dspqueue) */
    if (gen_hidden_buf) shared_free(gen_hidden_buf, gen_hidden_fd);
    if (gen_dsp_buf) shared_free(gen_dsp_buf, gen_dsp_fd);
    free(gen_embedding);

    rpcmem_deinit();

    if (handle) {
        bitnet_test_close(handle);
    }

    sem_destroy(&g_done);

    printf("[ARM] Done.\n");
    return all_pass ? 0 : 1;
}
