/*
 * main.c -- ARM driver for ch10 BitNet VLUT16 tests
 *
 * Sets up FastRPC + dspqueue, sends VLUT16 test request to DSP,
 * waits for result, prints pass/fail.
 *
 * Experiment 4 (decoder layer test): loads weight binary from
 * /data/local/tmp/bitnet_weights/decoder_layer.bin into rpcmem
 * shared memory, sends to DSP for verification against PyTorch
 * reference output.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <semaphore.h>
#include <inttypes.h>
#include <rpcmem.h>
#include <AEEStdErr.h>
#include "dspqueue.h"
#include "bitnet_test.h"
#include "common/protocol.h"

#define CDSP_DOMAIN_ID_ID 3

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

/* ========== Response handling ========== */

static sem_t g_done;
static struct vlut16_test_rsp g_rsp;

static void pkt_cb(dspqueue_t q, AEEResult err, void *ctx) {
    uint32_t flags, msg_len, n_bufs;
    struct dspqueue_buffer bufs[MAX_BUFFERS];
    struct vlut16_test_rsp rsp;

    while (dspqueue_read_noblock(q, &flags, MAX_BUFFERS, &n_bufs, bufs,
                                  sizeof(rsp), &msg_len, (uint8_t *)&rsp) == 0) {
        g_rsp = rsp;
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

/* ========== Decoder layer test ========== */

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

/* ========== Main ========== */

int main(int argc, char **argv) {
    int ret = 0;
    remote_handle64 handle = 0;
    dspqueue_t queue = NULL;

    printf("=== ch10: BitNet VLUT16 Exploration ===\n\n");

    /* Parse args: if "decoder" is passed, only run decoder test */
    int decoder_only = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "decoder") == 0) {
            decoder_only = 1;
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

    /* Create dspqueue */
    sem_init(&g_done, 0, 0);
    ret = dspqueue_create(CDSP_DOMAIN_ID, 0,
                          4096, 4096,
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

    if (!decoder_only) {
        /* Run existing tests */
        if (send_simple_test(queue, OP_VLUT16_TEST, "VLUT16") != 0)
            all_pass = 0;

        if (send_simple_test(queue, OP_GEMV_TEST, "GEMV") != 0)
            all_pass = 0;

        if (send_simple_test(queue, OP_ATTN_TEST, "Attention") != 0)
            all_pass = 0;

        if (send_simple_test(queue, OP_OPS_TEST, "Ops") != 0)
            all_pass = 0;
    }

    /* Run decoder layer test (experiment 4) */
    {
        int dec_ret = run_decoder_layer_test(queue);
        if (dec_ret < 0) {
            printf("[ARM] Decoder test skipped (weight file not found)\n");
        } else if (dec_ret != 0) {
            all_pass = 0;
        }
    }

    if (all_pass) {
        printf("[ARM] *** ALL TESTS PASSED ***\n");
    } else {
        printf("[ARM] *** SOME TESTS FAILED -- check logcat for details ***\n");
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

    rpcmem_deinit();

    if (handle) {
        bitnet_test_close(handle);
    }

    sem_destroy(&g_done);

    printf("[ARM] Done.\n");
    return all_pass ? 0 : 1;
}
