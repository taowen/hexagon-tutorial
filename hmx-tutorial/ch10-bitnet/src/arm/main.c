/*
 * main.c -- ARM driver for ch10 BitNet VLUT16 tests
 *
 * Sets up FastRPC + dspqueue, sends VLUT16 test request to DSP,
 * waits for result, prints pass/fail.
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

/* ========== Main ========== */

int main(int argc, char **argv) {
    int ret = 0;
    remote_handle64 handle = 0;
    dspqueue_t queue = NULL;

    printf("=== ch10: BitNet VLUT16 Exploration ===\n\n");

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

    /* Send VLUT16 test request */
    printf("[ARM] Sending VLUT16 test request...\n");
    struct vlut16_test_req req;
    memset(&req, 0, sizeof(req));
    req.op = OP_VLUT16_TEST;
    req.test_id = 1;

    ret = dspqueue_write(queue, 0, 0, NULL,
                          sizeof(req), (const uint8_t *)&req, 1000000);
    if (ret != 0) {
        fprintf(stderr, "dspqueue_write failed: 0x%08x\n", (unsigned)ret);
        goto cleanup;
    }

    /* Wait for response */
    sem_wait(&g_done);

    printf("\n[ARM] VLUT16 Response: op=%u status=%u pass=%u\n",
           g_rsp.op, g_rsp.status, g_rsp.pass);
    int all_pass = g_rsp.pass;

    /* Send GEMV test request */
    printf("[ARM] Sending GEMV test request...\n");
    memset(&req, 0, sizeof(req));
    req.op = OP_GEMV_TEST;
    req.test_id = 0;

    ret = dspqueue_write(queue, 0, 0, NULL,
                          sizeof(req), (const uint8_t *)&req, 1000000);
    if (ret != 0) {
        fprintf(stderr, "dspqueue_write (GEMV) failed: 0x%08x\n", (unsigned)ret);
        goto cleanup;
    }

    sem_wait(&g_done);

    printf("\n[ARM] GEMV Response: op=%u status=%u pass=%u\n",
           g_rsp.op, g_rsp.status, g_rsp.pass);
    all_pass &= g_rsp.pass;

    /* Send Attention test request */
    printf("[ARM] Sending Attention test request...\n");
    memset(&req, 0, sizeof(req));
    req.op = OP_ATTN_TEST;
    req.test_id = 0;

    ret = dspqueue_write(queue, 0, 0, NULL,
                          sizeof(req), (const uint8_t *)&req, 1000000);
    if (ret != 0) {
        fprintf(stderr, "dspqueue_write (ATTN) failed: 0x%08x\n", (unsigned)ret);
        goto cleanup;
    }

    sem_wait(&g_done);

    printf("\n[ARM] Attention Response: op=%u status=%u pass=%u\n",
           g_rsp.op, g_rsp.status, g_rsp.pass);
    all_pass &= g_rsp.pass;

    /* Send Ops test request (rmsnorm, relu2, mul, add, rope) */
    printf("[ARM] Sending Ops test request...\n");
    memset(&req, 0, sizeof(req));
    req.op = OP_OPS_TEST;
    req.test_id = 0;

    ret = dspqueue_write(queue, 0, 0, NULL,
                          sizeof(req), (const uint8_t *)&req, 1000000);
    if (ret != 0) {
        fprintf(stderr, "dspqueue_write (OPS) failed: 0x%08x\n", (unsigned)ret);
        goto cleanup;
    }

    sem_wait(&g_done);

    printf("\n[ARM] Ops Response: op=%u status=%u pass=%u\n",
           g_rsp.op, g_rsp.status, g_rsp.pass);
    all_pass &= g_rsp.pass;

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
    return (g_rsp.pass == 1) ? 0 : 1;
}
