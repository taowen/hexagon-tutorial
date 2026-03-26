/*
 * Common DSP lifecycle functions shared by all step DSP programs.
 *
 * Provides: mnist_train_open, mnist_train_close, error_callback
 * Each step's .c file includes this header and implements the remaining
 * IDL functions (start, stop, do_matmul).
 */
#ifndef DSP_COMMON_H
#define DSP_COMMON_H

#define FARF_ERROR 1
#define FARF_HIGH 1
#include <HAP_farf.h>
#include <string.h>
#include <HAP_power.h>
#include <HAP_perf.h>
#include <AEEStdErr.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include "mnist_train.h"
#include "dspqueue.h"

/* ========== dspqueue error callback ========== */

static void error_callback(dspqueue_t queue, int error, void *context) {
    FARF(ERROR, "dspqueue error: 0x%08x", (unsigned)error);
}

/* ========== FastRPC lifecycle ========== */

static AEEResult mnist_train_open_impl(const char *uri, remote_handle64 *handle,
                                        size_t ctx_size) {
    void *ctx = calloc(1, ctx_size);
    if (!ctx) return AEE_ENOMEMORY;
    *handle = (remote_handle64)ctx;

    /* Power config: lock to turbo, disable DCVS */
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_apptype;
    request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    HAP_power_set(ctx, &request);

    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_DCVS_v2;
    request.dcvs_v2.dcvs_enable = FALSE;
    request.dcvs_v2.set_dcvs_params = TRUE;
    request.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_DISABLE;
    request.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_DISABLE;
    request.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_TURBO;
    request.dcvs_v2.set_latency = TRUE;
    request.dcvs_v2.latency = 40;
    int pwr_err = HAP_power_set(ctx, &request);
    if (pwr_err != AEE_SUCCESS) {
        free(ctx);
        return pwr_err;
    }

    return AEE_SUCCESS;
}

#endif /* DSP_COMMON_H */
