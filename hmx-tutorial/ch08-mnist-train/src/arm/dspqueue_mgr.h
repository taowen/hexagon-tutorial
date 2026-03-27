/*
 * dspqueue_mgr.h -- dspqueue communication for MNIST training
 *
 * Manages shared memory allocation, dspqueue lifecycle, and
 * zero-copy matmul dispatch to the Hexagon DSP.
 */

#ifndef DSPQUEUE_MGR_H
#define DSPQUEUE_MGR_H

#include "common/common.h"
#include "arm/network.h"

#include <semaphore.h>
#include <rpcmem.h>
#include <AEEStdErr.h>
#include "dspqueue.h"
#include "mnist_train.h"
#include "common/protocol.h"

/* ====================================================================
 * Shared memory registry
 *
 * All network buffers in dspqueue mode are allocated via rpcmem.
 * The registry maps pointer -> fd for zero-copy dspqueue dispatch.
 * ==================================================================== */

#define MAX_SHARED_BUFS 40

struct shared_buf_entry {
    void *ptr;
    int   fd;
    size_t size;
};

static struct shared_buf_entry g_shared_bufs[MAX_SHARED_BUFS];
static int g_n_shared_bufs = 0;
static int g_dspq_domain_id = -1;

/* Allocate shared memory, register in the fd lookup table */
static void *dspq_alloc(size_t size) {
    if (g_n_shared_bufs >= MAX_SHARED_BUFS) {
        printf("[ERROR] shared buf registry full\n");
        return NULL;
    }
    size_t aligned = (size + 4095) & ~4095UL;
    void *p = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                           RPCMEM_DEFAULT_FLAGS | RPCMEM_HEAP_NOREG,
                           aligned);
    if (!p) {
        printf("[ERROR] rpcmem_alloc failed for %zu bytes\n", aligned);
        return NULL;
    }
    int fd = rpcmem_to_fd(p);
    if (fd <= 0) {
        printf("[ERROR] rpcmem_to_fd failed\n");
        rpcmem_free(p);
        return NULL;
    }
    int err = fastrpc_mmap(g_dspq_domain_id, fd, p, 0, aligned, FASTRPC_MAP_FD);
    if (err != 0) {
        printf("[ERROR] fastrpc_mmap failed: 0x%08x\n", (unsigned)err);
        rpcmem_free(p);
        return NULL;
    }
    g_shared_bufs[g_n_shared_bufs++] = (struct shared_buf_entry){p, fd, aligned};
    return p;
}

/* Free a shared memory buffer and remove from registry */
static void dspq_free(void *p) {
    if (!p) return;
    for (int i = 0; i < g_n_shared_bufs; i++) {
        if (g_shared_bufs[i].ptr == p) {
            fastrpc_munmap(g_dspq_domain_id, g_shared_bufs[i].fd, NULL, 0);
            rpcmem_free(p);
            g_shared_bufs[i] = g_shared_bufs[--g_n_shared_bufs];
            return;
        }
    }
    free(p);  /* not in registry, use regular free */
}

/* Look up fd for a pointer (returns -1 if not found) */
static int dspq_find_fd(const void *ptr) {
    for (int i = 0; i < g_n_shared_bufs; i++) {
        if (ptr >= (const void *)g_shared_bufs[i].ptr &&
            (const char *)ptr < (const char *)g_shared_bufs[i].ptr + g_shared_bufs[i].size) {
            return g_shared_bufs[i].fd;
        }
    }
    return -1;
}

/* ====================================================================
 * DSP dspqueue context
 * ==================================================================== */

struct dspqueue_ctx {
    remote_handle64 handle;
    dspqueue_t queue;

    /* Synchronous response tracking */
    sem_t done_sem;
    uint32_t ops_done;
    uint32_t ops_total;

    /* Response data for OP_TRAIN_BATCH */
    float    last_loss;
    uint32_t last_correct;
};

static struct dspqueue_ctx g_dspq;

/* ARM-side response callback: post semaphore when response arrives */
static void arm_error_callback(dspqueue_t queue, AEEResult error, void *context) {
    fprintf(stderr, "[ERROR] dspqueue error: 0x%x\n", error);
}

static void arm_packet_callback(dspqueue_t queue, AEEResult error, void *context) {
    struct dspqueue_ctx *ctx = (struct dspqueue_ctx *)context;

    while (1) {
        union {
            struct matmul_rsp matmul;
            struct train_batch_rsp train;
        } rsp;
        uint32_t flags, msg_len, n_bufs;
        struct dspqueue_buffer bufs[MATMUL_MAX_BUFFERS];

        int err = dspqueue_read_noblock(queue, &flags,
                                        MATMUL_MAX_BUFFERS, &n_bufs, bufs,
                                        sizeof(rsp), &msg_len, (uint8_t *)&rsp);
        if (err == AEE_EWOULDBLOCK) return;
        if (err != 0) {
            fprintf(stderr, "[ERROR] dspqueue_read failed: 0x%08x\n", (unsigned)err);
            return;
        }

        if (rsp.matmul.op == OP_TRAIN_BATCH) {
            ctx->last_loss = rsp.train.loss;
            ctx->last_correct = rsp.train.correct;
        }

        ctx->ops_done++;
        if (ctx->ops_done >= ctx->ops_total) {
            sem_post(&ctx->done_sem);
        }
    }
}

/* Initialize dspqueue context */
static int dspqueue_init(int domain_id) {
    int err;
    memset(&g_dspq, 0, sizeof(g_dspq));
    g_dspq_domain_id = domain_id;

    sem_init(&g_dspq.done_sem, 0, 0);

    /* Open FastRPC session */
    struct remote_rpc_control_unsigned_module data;
    data.domain = domain_id;
    data.enable = 1;
    remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&data, sizeof(data));

    char uri[256];
    snprintf(uri, sizeof(uri), "%s&_dom=cdsp", mnist_train_URI);
    err = mnist_train_open(uri, &g_dspq.handle);
    if (err) {
        printf("[ERROR] mnist_train_open failed: 0x%08x\n", (unsigned)err);
        return -1;
    }

    /* Enable FastRPC QoS */
    struct remote_rpc_control_latency ldata = { .enable = 1 };
    remote_handle64_control(g_dspq.handle, DSPRPC_CONTROL_LATENCY, &ldata, sizeof(ldata));

    rpcmem_init();

    /* Set global allocator to use shared memory for network buffers */
    g_alloc_fn = dspq_alloc;
    g_free_fn  = dspq_free;

    /* Create dspqueue */
    err = dspqueue_create(domain_id, 0,
                          4096, 4096,
                          arm_packet_callback, arm_error_callback, &g_dspq,
                          &g_dspq.queue);
    if (err) {
        printf("[ERROR] dspqueue_create failed: 0x%08x\n", (unsigned)err);
        return -1;
    }

    uint64_t dsp_queue_id;
    err = dspqueue_export(g_dspq.queue, &dsp_queue_id);
    if (err) {
        printf("[ERROR] dspqueue_export failed: 0x%08x\n", (unsigned)err);
        return -1;
    }

    /* Pass queue ID to DSP via one-time FastRPC call */
    err = mnist_train_start(g_dspq.handle, dsp_queue_id);
    if (err) {
        printf("[ERROR] mnist_train_start failed: 0x%08x\n", (unsigned)err);
        return -1;
    }

    printf("[DSPQ] dspqueue initialized (%d shared buffers will be registered)\n",
           g_n_shared_bufs);
    return 0;
}

static void dspqueue_cleanup(void) {
    uint64 process_time = 0;
    mnist_train_stop(g_dspq.handle, &process_time);
    printf("[DSPQ] DSP total process time: %" PRIu64 " us\n", (uint64_t)process_time);

    dspqueue_close(g_dspq.queue);

    /* Reset allocator before freeing (network_free uses net_free -> dspq_free) */

    rpcmem_deinit();
    mnist_train_close(g_dspq.handle);
    sem_destroy(&g_dspq.done_sem);
}

/*
 * Register all network buffer fds with DSP.
 * This is called once before training starts. The DSP stores the pointers
 * and uses them for all subsequent OP_TRAIN_BATCH operations.
 */
static void dspqueue_register_net(network_t *net) {
    struct register_net_req req;
    memset(&req, 0, sizeof(req));
    req.op = OP_REGISTER_NET;

    struct dspqueue_buffer bufs[NET_BUF_COUNT];
    memset(bufs, 0, sizeof(bufs));

    /* Map buffer index -> network pointer */
    void *ptrs[NET_BUF_COUNT];
    ptrs[NET_BUF_W1]         = net->w1;
    ptrs[NET_BUF_B1]         = net->b1;
    ptrs[NET_BUF_W2]         = net->w2;
    ptrs[NET_BUF_B2]         = net->b2;
    ptrs[NET_BUF_DW1]        = net->dw1;
    ptrs[NET_BUF_DW2]        = net->dw2;
    ptrs[NET_BUF_HIDDEN]     = net->hidden;
    ptrs[NET_BUF_LOGITS]     = net->logits;
    ptrs[NET_BUF_DHIDDEN]    = net->dhidden;
    ptrs[NET_BUF_DLOGITS]    = net->dlogits;
    ptrs[NET_BUF_HIDDEN_PRE] = net->hidden_pre_relu;
    ptrs[NET_BUF_PROBS]      = net->probs;

    for (int i = 0; i < NET_BUF_COUNT; i++) {
        int fd = dspq_find_fd(ptrs[i]);
        if (fd < 0) {
            fprintf(stderr, "[ERROR] buffer %d not in shared memory\n", i);
            return;
        }
        bufs[i].fd = fd;
        bufs[i].flags = DSPQUEUE_BUFFER_FLAG_REF
                       | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER;
    }

    g_dspq.ops_done = 0;
    g_dspq.ops_total = 1;
    int err = dspqueue_write(g_dspq.queue, 0, NET_BUF_COUNT, bufs,
                              sizeof(req), (const uint8_t *)&req, 1000000);
    if (err != 0) {
        fprintf(stderr, "[ERROR] register_net write failed: 0x%08x\n", (unsigned)err);
        return;
    }
    sem_wait(&g_dspq.done_sem);
    printf("[DSPQ] Network buffers registered with DSP\n");
}

/*
 * Sync: flush DSP caches so ARM can read updated weights
 */
static void dspqueue_sync(network_t *net) {
    struct sync_req req;
    memset(&req, 0, sizeof(req));
    req.op = OP_SYNC;

    struct dspqueue_buffer bufs[4];
    memset(bufs, 0, sizeof(bufs));

    void *weight_ptrs[4] = { net->w1, net->b1, net->w2, net->b2 };
    for (int i = 0; i < 4; i++) {
        bufs[i].fd = dspq_find_fd(weight_ptrs[i]);
        bufs[i].flags = DSPQUEUE_BUFFER_FLAG_REF;
    }

    g_dspq.ops_done = 0;
    g_dspq.ops_total = 1;
    dspqueue_write(g_dspq.queue, 0, 4, bufs,
                   sizeof(req), (const uint8_t *)&req, 1000000);
    sem_wait(&g_dspq.done_sem);
}

#endif /* DSPQUEUE_MGR_H */
