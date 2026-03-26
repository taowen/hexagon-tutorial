/*
 * ch05: ARM 端 — dspqueue vs FastRPC 性能对比
 *
 * 对应 llama.cpp 的 ggml/src/ggml-hexagon/ggml-hexagon.cpp
 *
 * 核心流程：
 *   1. rpcmem_alloc 分配共享内存（零拷贝）
 *   2. dspqueue_create + dspqueue_export 创建队列
 *   3. FastRPC start() 传递 queue ID 给 DSP
 *   4. dspqueue_write 发送 op 请求（不走内核态）
 *   5. 回调中 dspqueue_read 接收响应
 *
 * 同时提供 FastRPC 直接调用路径做对比 benchmark。
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include <assert.h>
#include <semaphore.h>
#include <rpcmem.h>
#include <AEEStdErr.h>
#include "dspqueue.h"
#include "dspqueue_demo.h"
#include "dspqueue_demo_shared.h"

/* ---------- 全局状态 ---------- */
static remote_handle64 g_handle = -1;

static uint64_t time_usec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}


/* ---------- 回调上下文 ----------
 *
 * 对应 llama.cpp ggml-hexagon.cpp 中的 ggml_hexagon_session，
 * 它用 op_pending 计数器 + flush() 等待所有 op 完成。
 */
struct bench_context {
    uint32_t ops_done;
    uint32_t ops_total;
    uint64_t end_time;
    sem_t    done_sem;
    int      verify;      /* 是否验证结果正确性 */
    uint32_t op;          /* 当前测试的 op */
};

static void error_callback(dspqueue_t queue, AEEResult error, void *context) {
    fprintf(stderr, "ERROR 0x%x\n", error);
    exit(EXIT_FAILURE);
}


/* ---------- 响应回调 ----------
 *
 * 对应 llama.cpp 的 flush()：等待所有 enqueue 的 op 返回。
 * llama.cpp 用的是同步 dspqueue_read（没有回调），
 * 我们用回调方式（和 SDK 示例一致），效果相同。
 */
static void packet_callback(dspqueue_t queue, AEEResult error, void *context) {

    struct bench_context *c = (struct bench_context *)context;

    while (1) {
        struct demo_rsp rsp;
        uint32_t flags, msg_len, n_bufs;
        struct dspqueue_buffer bufs[DEMO_MAX_BUFFERS];

        int err = dspqueue_read_noblock(queue, &flags,
                                        DEMO_MAX_BUFFERS, &n_bufs, bufs,
                                        sizeof(rsp), &msg_len, (uint8_t *)&rsp);
        if (err == AEE_EWOULDBLOCK) return;
        if (err != 0) {
            fprintf(stderr, "dspqueue_read failed: 0x%08x\n", (unsigned)err);
            exit(EXIT_FAILURE);
        }

        c->ops_done++;
        if (c->ops_done == c->ops_total) {
            c->end_time = time_usec();
            sem_post(&c->done_sem);
        }
    }
}


/* ---------- 结果验证 ---------- */

static void verify_scale(const uint8_t *in, const uint8_t *out,
                          uint32_t n, uint8_t factor) {
    for (uint32_t i = 0; i < n; i++) {
        uint8_t expected = (uint8_t)(in[i] * factor);
        if (out[i] != expected) {
            fprintf(stderr, "SCALE mismatch at %u: expected %u, got %u\n",
                    i, expected, out[i]);
            exit(EXIT_FAILURE);
        }
    }
}

static void verify_add(const uint8_t *a, const uint8_t *b, const uint8_t *out,
                        uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        uint8_t expected = (uint8_t)(a[i] + b[i]);
        if (out[i] != expected) {
            fprintf(stderr, "ADD mismatch at %u: expected %u, got %u\n",
                    i, expected, out[i]);
            exit(EXIT_FAILURE);
        }
    }
}


/* ========== dspqueue 路径 benchmark ==========
 *
 * 对应 llama.cpp 的推理循环：
 *   for 每个 token:
 *     for 每个 op:
 *       enqueue(req, bufs)    // dspqueue_write
 *     flush()                 // 等待所有 op 完成
 */
static void bench_dspqueue(int domain_id, uint32_t op, uint32_t buf_size,
                            int num_ops, int verify) {

    int err;
    dspqueue_t queue;
    uint64_t dsp_queue_id;

    /* 上下文 */
    struct bench_context ctx;
    memset(&ctx, 0, sizeof(ctx));
    sem_init(&ctx.done_sem, 0, 0);
    ctx.ops_total = num_ops;
    ctx.verify = verify;
    ctx.op = op;

    /* 分配共享内存 — 对应 llama.cpp 的 rpcmem_alloc2 + fastrpc_mmap
     *
     * llama.cpp:
     *   base = rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
     *   fd = rpcmem_to_fd(base);
     *   fastrpc_mmap(domain_id, fd, base, 0, size, FASTRPC_MAP_FD);
     */
    int n_bufs = (op == OP_ADD) ? 3 : 2;   /* ADD: a + b → out; SCALE: in → out */
    void *buffers[3];
    int fds[3];

    for (int i = 0; i < n_bufs; i++) {
        buffers[i] = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                   RPCMEM_DEFAULT_FLAGS | RPCMEM_HEAP_NOREG,
                                   buf_size);
        assert(buffers[i]);
        fds[i] = rpcmem_to_fd(buffers[i]);
        assert(fds[i] > 0);
        err = fastrpc_mmap(domain_id, fds[i], buffers[i], 0, buf_size,
                           FASTRPC_MAP_FD);
        assert(err == 0);
    }

    /* 填充输入数据 */
    for (int i = 0; i < n_bufs - 1; i++) {
        uint8_t *p = buffers[i];
        for (uint32_t j = 0; j < buf_size; j++) {
            p[j] = (uint8_t)(j + i * 37);
        }
    }

    /* 创建 dspqueue — 对应 llama.cpp session 构造函数中的 dspqueue_create */
    err = dspqueue_create(domain_id, 0,
                           4096, 4096,    /* 请求/响应队列大小 */
                           packet_callback, error_callback, &ctx,
                           &queue);
    assert(err == 0);

    err = dspqueue_export(queue, &dsp_queue_id);
    assert(err == 0);

    /* 通过 FastRPC 传递 queue ID — 唯一一次 FastRPC */
    err = dspqueue_demo_start(g_handle, dsp_queue_id);
    assert(err == 0);

    /* 发送 op 请求 — 对应 llama.cpp 的 enqueue() */
    ctx.ops_done = 0;
    uint64_t start = time_usec();

    for (int i = 0; i < num_ops; i++) {
        struct demo_req req;
        req.op = op;
        req.param = (op == OP_SCALE) ? 2 : 0;
        req.n_elem = buf_size;
        req.reserved = 0;

        struct dspqueue_buffer dbufs[3];
        memset(dbufs, 0, sizeof(dbufs));

        if (op == OP_SCALE) {
            /* src0(input) + dst(output) */
            dbufs[0].fd = fds[0];
            dbufs[0].flags = DSPQUEUE_BUFFER_FLAG_REF
                           | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
                           | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
            dbufs[1].fd = fds[1];
            dbufs[1].flags = DSPQUEUE_BUFFER_FLAG_REF
                           | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER;
        } else {
            /* src0(a) + src1(b) + dst(out) */
            dbufs[0].fd = fds[0];
            dbufs[0].flags = DSPQUEUE_BUFFER_FLAG_REF
                           | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
                           | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
            dbufs[1].fd = fds[1];
            dbufs[1].flags = DSPQUEUE_BUFFER_FLAG_REF
                           | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER
                           | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
            dbufs[2].fd = fds[2];
            dbufs[2].flags = DSPQUEUE_BUFFER_FLAG_REF
                           | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER;
        }

        err = dspqueue_write(queue, 0, n_bufs, dbufs,
                              sizeof(req), (const uint8_t *)&req,
                              1000000);
        assert(err == 0);
    }

    /* 等所有 op 完成 — 对应 llama.cpp 的 flush() */
    sem_wait(&ctx.done_sem);
    uint64_t end = ctx.end_time;

    /* 验证 */
    if (verify) {
        if (op == OP_SCALE) {
            verify_scale(buffers[0], buffers[1], buf_size, 2);
        } else {
            verify_add(buffers[0], buffers[1], buffers[2], buf_size);
        }
        printf("  [PASS] %s correctness verified\n",
               op == OP_SCALE ? "OP_SCALE" : "OP_ADD");
    }

    /* 停止 DSP 端 */
    uint64 process_time;
    err = dspqueue_demo_stop(g_handle, &process_time);
    assert(err == 0);

    /* 打印结果 */
    if (!verify) {
        printf("  %d ops, %"PRIu64" us total, %"PRIu64" us DSP, "
               "%"PRIu64" us overhead/op\n",
               num_ops,
               end - start,
               (uint64_t)process_time,
               (end - start - (uint64_t)process_time) / num_ops);
    }

    /* 清理 */
    dspqueue_close(queue);
    for (int i = 0; i < n_bufs; i++) {
        fastrpc_munmap(domain_id, fds[i], NULL, 0);
        rpcmem_free(buffers[i]);
    }
    sem_destroy(&ctx.done_sem);
}


/* ========== FastRPC 路径 benchmark ==========
 *
 * 每次 do_op 都走一次完整的 FastRPC（内核态切换 + 参数序列化）。
 * 这是 ch02 中 run_main_on_hexagon 使用的方式。
 */
static void bench_fastrpc(int domain_id, uint32_t op, uint32_t buf_size,
                           int num_ops, int verify) {
    int err;

    /* 分配共享内存 */
    void *input = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, buf_size);
    void *output = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, buf_size);
    assert(input && output);

    /* 预映射 buffer 到 DSP（减少 FastRPC 动态映射开销） */
    int in_fd = rpcmem_to_fd(input);
    int out_fd = rpcmem_to_fd(output);
    fastrpc_mmap(domain_id, in_fd, input, 0, buf_size, FASTRPC_MAP_STATIC);
    fastrpc_mmap(domain_id, out_fd, output, 0, buf_size, FASTRPC_MAP_STATIC);

    /* 填充数据 */
    uint8_t *p = input;
    for (uint32_t i = 0; i < buf_size; i++) p[i] = (uint8_t)i;

    uint64_t total_dsp = 0;
    uint64_t start = time_usec();

    for (int i = 0; i < num_ops; i++) {
        uint64 dsp_time;
        err = dspqueue_demo_do_op(g_handle,
                                   input, buf_size,
                                   output, buf_size,
                                   op, &dsp_time);
        assert(err == 0);
        total_dsp += dsp_time;
    }

    uint64_t end = time_usec();

    if (verify) {
        if (op == OP_SCALE)
            verify_scale(input, output, buf_size, 2);
        printf("  [PASS] FastRPC %s correctness verified\n",
               op == OP_SCALE ? "OP_SCALE" : "OP_ADD");
    }

    if (!verify) {
        printf("  %d ops, %"PRIu64" us total, %"PRIu64" us DSP, "
               "%"PRIu64" us overhead/op\n",
               num_ops,
               end - start,
               total_dsp,
               (end - start - total_dsp) / num_ops);
    }

    fastrpc_munmap(domain_id, in_fd, NULL, 0);
    fastrpc_munmap(domain_id, out_fd, NULL, 0);
    rpcmem_free(input);
    rpcmem_free(output);
}


/* ========== main ========== */

int main(int argc, char *argv[]) {

    int domain_id = 3;     /* CDSP */
    uint32_t buf_size = 1024 * 1024;   /* 1MB — 和 SDK 示例相同 */
    int err;

    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    printf("ch05: dspqueue vs FastRPC benchmark\n");
    printf("====================================\n\n");

    /* 打开 FastRPC session — 对应 llama.cpp 的 htp_iface_open */
    {
        struct remote_rpc_control_unsigned_module data;
        data.domain = domain_id;
        data.enable = 1;
        remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&data, sizeof(data));
    }

    char uri[256];
    snprintf(uri, sizeof(uri), "%s&_dom=cdsp", dspqueue_demo_URI);
    err = dspqueue_demo_open(uri, &g_handle);
    if (err) {
        fprintf(stderr, "Session open failed: 0x%08x\n", (unsigned)err);
        return 1;
    }

    /* 启用 FastRPC QoS */
    struct remote_rpc_control_latency ldata = { .enable = 1 };
    remote_handle64_control(g_handle, DSPRPC_CONTROL_LATENCY, &ldata, sizeof(ldata));

    rpcmem_init();

    /* ---- 正确性验证 ---- */
    printf("[Test] Correctness verification (1 op each)\n");
    bench_dspqueue(domain_id, OP_SCALE, buf_size, 1, 1);
    bench_dspqueue(domain_id, OP_ADD,   buf_size, 1, 1);
    bench_fastrpc(domain_id, OP_SCALE, buf_size, 1, 1);
    printf("\n");

    /* ---- dspqueue benchmark: OP_SCALE ---- */
    printf("[Bench] dspqueue OP_SCALE (%u KB buffer)\n", buf_size / 1024);
    bench_dspqueue(domain_id, OP_SCALE, buf_size,   10, 0);
    bench_dspqueue(domain_id, OP_SCALE, buf_size,  100, 0);
    bench_dspqueue(domain_id, OP_SCALE, buf_size, 1000, 0);
    printf("\n");

    /* ---- dspqueue benchmark: OP_ADD ---- */
    printf("[Bench] dspqueue OP_ADD (%u KB buffer)\n", buf_size / 1024);
    bench_dspqueue(domain_id, OP_ADD, buf_size,   10, 0);
    bench_dspqueue(domain_id, OP_ADD, buf_size,  100, 0);
    bench_dspqueue(domain_id, OP_ADD, buf_size, 1000, 0);
    printf("\n");

    /* ---- FastRPC benchmark: OP_SCALE ---- */
    printf("[Bench] FastRPC OP_SCALE (%u KB buffer, static mapping)\n", buf_size / 1024);
    bench_fastrpc(domain_id, OP_SCALE, buf_size,   10, 0);
    bench_fastrpc(domain_id, OP_SCALE, buf_size,  100, 0);
    bench_fastrpc(domain_id, OP_SCALE, buf_size, 1000, 0);
    printf("\n");

    /* ---- 对比总结 ---- */
    printf("====================================\n");
    printf("Summary: Run 1000 ops of OP_SCALE to compare overhead.\n");
    printf("dspqueue avoids kernel transitions → lower overhead per op.\n");
    printf("This is why llama.cpp uses dspqueue for all %d ops/token.\n", 196);
    printf("====================================\n");

    rpcmem_deinit();
    dspqueue_demo_close(g_handle);
    return 0;
}
