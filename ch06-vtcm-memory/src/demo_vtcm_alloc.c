/*
 * ch06: VTCM Memory Management — main entry point
 *
 * Uses run_main_on_hexagon to run directly on the DSP (same as ch02).
 * Single-threaded, no FastRPC cross-thread mapping issues.
 *
 * Demonstrates four levels of VTCM management:
 *   1. SDK basics: HAP_compute_res API
 *   2. llama.cpp bump allocator (vtcm_seq_alloc)
 *   3. TVM pool allocator (HexagonVtcmPool)
 *   4. DMA (DDR -> VTCM async transfer)
 *   5. HVX benchmark: VTCM vs DDR
 */

#include "common.h"

/* ---- VTCM globals (defined here, declared extern in common.h) ---- */
uint8_t  *g_vtcm_base;
uint32_t  g_vtcm_size;
uint32_t  g_vtcm_ctx;

/* ---- Power on implementation ---- */
static int power_ctx;

int power_on(void)
{
    HAP_power_request_t req;

    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_apptype;
    req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    if (HAP_power_set((void *)&power_ctx, &req) != 0) return -1;

    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_DCVS_v3;
    req.dcvs_v3.set_dcvs_enable = 1;
    req.dcvs_v3.dcvs_enable = 1;
    req.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
    req.dcvs_v3.set_bus_params = 1;
    req.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_core_params = 1;
    req.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_sleep_disable = 1;
    req.dcvs_v3.sleep_disable = 1;
    if (HAP_power_set((void *)&power_ctx, &req) != 0) return -2;

    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HVX;
    req.hvx.power_up = 1;
    if (HAP_power_set((void *)&power_ctx, &req) != 0) return -3;

    return 0;
}

/* ---- Part 1: VTCM allocation via HAP_compute_res API ---- */
void demo_vtcm_alloc(void)
{
    /* Step 1: Query VTCM total size
     *
     * VTCM size varies by chip:
     *   v68 (Snapdragon 888): 4 MB
     *   v73 (Snapdragon 8 Gen 2): 8 MB
     *   v75 (Snapdragon 8 Gen 3): 8 MB
     */
    unsigned int vtcm_size = 8 * 1024 * 1024;
    HAP_compute_res_query_VTCM(0, &vtcm_size, NULL, NULL, NULL);
    FARF(ALWAYS, "  VTCM total: %u KB (%u MB)",
         vtcm_size / 1024, vtcm_size / (1024 * 1024));

    /* Step 2: Configure resource attributes */
    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_vtcm_param(&attr, vtcm_size, 1);

    /* Step 3: Acquire resource (100ms timeout) */
    unsigned int ctx_id = HAP_compute_res_acquire(&attr, 100000);
    if (ctx_id == 0) {
        FARF(ALWAYS, "  FAILED: HAP_compute_res_acquire returned 0");
        return;
    }

    /* Step 4: Get VTCM virtual address */
    void *vtcm_ptr = HAP_compute_res_attr_get_vtcm_ptr(&attr);
    if (!vtcm_ptr) {
        FARF(ALWAYS, "  FAILED: VTCM pointer is NULL");
        HAP_compute_res_release(ctx_id);
        return;
    }

    g_vtcm_base = (uint8_t *)vtcm_ptr;
    g_vtcm_size = vtcm_size;
    g_vtcm_ctx  = ctx_id;

    FARF(ALWAYS, "  VTCM allocated: %u KB at %p", vtcm_size / 1024, vtcm_ptr);
}

/* ---- main ---- */
int main(int argc, char **argv)
{
    FARF(ALWAYS, "========================================");
    FARF(ALWAYS, "  Chapter 6: VTCM Memory Management");
    FARF(ALWAYS, "========================================");

    /* Step 1: Power on */
    FARF(ALWAYS, "");
    FARF(ALWAYS, "[1/6] Power on HVX...");
    if (power_on() != 0) {
        FARF(ALWAYS, "  FAILED");
        return 1;
    }
    FARF(ALWAYS, "  OK");

    /* Step 2: Allocate VTCM */
    FARF(ALWAYS, "");
    FARF(ALWAYS, "[2/6] VTCM allocation (HAP_compute_res API)...");
    demo_vtcm_alloc();
    if (!g_vtcm_base) {
        FARF(ALWAYS, "  FAILED");
        return 1;
    }

    /* Step 3: Bump allocator demo */
    FARF(ALWAYS, "");
    FARF(ALWAYS, "[3/6] Bump allocator (llama.cpp style)...");
    demo_bump_alloc();

    /* Step 4: Pool allocator demo */
    FARF(ALWAYS, "");
    FARF(ALWAYS, "[4/6] Pool allocator (TVM style)...");
    demo_pool_alloc();

    /* Step 5: DMA demo */
    FARF(ALWAYS, "");
    FARF(ALWAYS, "[5/6] DMA: DDR -> VTCM async transfer...");
    demo_dma();

    /* Step 6: HVX benchmark */
    FARF(ALWAYS, "");
    FARF(ALWAYS, "[6/6] HVX vadd benchmark: VTCM vs DDR...");
    demo_hvx_bench();

    /* Release */
    HAP_compute_res_release(g_vtcm_ctx);

    FARF(ALWAYS, "");
    FARF(ALWAYS, "========================================");
    FARF(ALWAYS, "  Done");
    FARF(ALWAYS, "========================================");
    return 0;
}
