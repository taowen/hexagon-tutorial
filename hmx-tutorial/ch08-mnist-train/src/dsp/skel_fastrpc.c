/*
 * Step 2: DSP receives matmul requests via FastRPC.
 * Each call goes through a kernel transition.
 *
 * This is the simplest DSP program -- only handles FastRPC matmul calls.
 * The dspqueue functions (start/stop) are stubbed out.
 */

#include "dsp/skel_common.h"
#include "dsp/hvx_matmul.h"

/* ---------- DSP context (minimal) ---------- */
struct mnist_context {
    int dummy;  /* placeholder */
};


/* ========== FastRPC lifecycle ========== */

AEEResult mnist_train_open(const char *uri, remote_handle64 *handle) {
    return mnist_train_open_impl(uri, handle, sizeof(struct mnist_context));
}

AEEResult mnist_train_close(remote_handle64 handle) {
    struct mnist_context *ctx = (struct mnist_context *)handle;
    if (!ctx) return AEE_EBADPARM;
    free(ctx);
    return AEE_SUCCESS;
}


/* ========== FastRPC matmul (the real implementation) ========== */

AEEResult mnist_train_do_matmul(remote_handle64 handle,
                                 const uint8 *a_buf, int a_buf_len,
                                 const uint8 *b_buf, int b_buf_len,
                                 uint8 *c_buf, int c_buf_len,
                                 uint32 m, uint32 n, uint32 k, uint32 transpose,
                                 uint64 *process_time) {
    uint64_t t1 = HAP_perf_get_time_us();

    do_matmul((float *)c_buf,
              (const float *)a_buf,
              (const float *)b_buf,
              m, n, k, transpose, 0);

    uint64_t t2 = HAP_perf_get_time_us();
    *process_time = t2 - t1;
    return AEE_SUCCESS;
}


/* ========== Stubs (not used by step 2) ========== */

AEEResult mnist_train_start(remote_handle64 handle, uint64 dsp_queue_id) {
    return AEE_EUNSUPPORTED;
}

AEEResult mnist_train_stop(remote_handle64 handle, uint64 *process_time) {
    return AEE_EUNSUPPORTED;
}
