/*
 * fastrpc_mgr.h -- FastRPC communication for MNIST training
 *
 * Manages FastRPC session lifecycle and matmul dispatch to the Hexagon DSP.
 */

#ifndef FASTRPC_MGR_H
#define FASTRPC_MGR_H

#include "mnist_common.h"

#include <rpcmem.h>
#include <AEEStdErr.h>
#include "mnist_train.h"
#include "mnist_train_shared.h"

/* --- FastRPC context --- */
static remote_handle64 g_fastrpc_handle = -1;

static int fastrpc_init(int domain_id) {
    int err;

    struct remote_rpc_control_unsigned_module data;
    data.domain = domain_id;
    data.enable = 1;
    remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&data, sizeof(data));

    char uri[256];
    snprintf(uri, sizeof(uri), "%s&_dom=cdsp", mnist_train_URI);
    err = mnist_train_open(uri, &g_fastrpc_handle);
    if (err) {
        printf("[ERROR] mnist_train_open failed: 0x%08x\n", (unsigned)err);
        return -1;
    }

    struct remote_rpc_control_latency ldata = { .enable = 1 };
    remote_handle64_control(g_fastrpc_handle, DSPRPC_CONTROL_LATENCY, &ldata, sizeof(ldata));

    rpcmem_init();

    printf("[FRPC] FastRPC initialized\n");
    return 0;
}

static void fastrpc_cleanup(void) {
    rpcmem_deinit();
    mnist_train_close(g_fastrpc_handle);
}

/* FastRPC matmul dispatch (used as matmul_fn_t) */
static void fastrpc_matmul_dispatch(float *C, const float *A, const float *B,
                                    int m, int n, int k, int transpose) {
    size_t a_bytes, b_bytes;

    if (transpose == 2) {
        a_bytes = (size_t)k * m * sizeof(float);
        b_bytes = (size_t)k * n * sizeof(float);
    } else if (transpose == 1) {
        a_bytes = (size_t)m * k * sizeof(float);
        b_bytes = (size_t)n * k * sizeof(float);
    } else {
        a_bytes = (size_t)m * k * sizeof(float);
        b_bytes = (size_t)k * n * sizeof(float);
    }
    size_t c_bytes = (size_t)m * n * sizeof(float);

    /* For transpose==2 (accumulate mode), we need to send existing C data and
     * handle accumulation on CPU side since FastRPC do_matmul always does C = result.
     * Save a copy and add after. */
    float *c_save = NULL;
    if (transpose == 2) {
        c_save = (float *)malloc(c_bytes);
        if (c_save) memcpy(c_save, C, c_bytes);
    }

    uint64 dsp_time;

    int err = mnist_train_do_matmul(g_fastrpc_handle,
                                     (const uint8_t *)A, (int)a_bytes,
                                     (const uint8_t *)B, (int)b_bytes,
                                     (uint8_t *)C, (int)c_bytes,
                                     (uint32_t)m, (uint32_t)n, (uint32_t)k,
                                     (uint32_t)transpose,
                                     &dsp_time);
    if (err != 0) {
        fprintf(stderr, "[ERROR] mnist_train_do_matmul failed: 0x%08x\n", (unsigned)err);
    }

    /* For accumulate mode, add saved values back */
    if (transpose == 2 && c_save) {
        int total = m * n;
        for (int i = 0; i < total; i++)
            C[i] += c_save[i];
        free(c_save);
    }
}

#endif /* FASTRPC_MGR_H */
