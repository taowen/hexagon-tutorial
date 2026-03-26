/*
 * Part 4: DMA — DDR -> VTCM async transfer
 *
 * Hexagon's UDMA (User DMA) hardware transfers data in the background
 * while CPU/HVX/HMX can do other work.
 *
 * llama.cpp uses DMA to move quantized weights from DDR to VTCM
 * while HVX/HMX processes the previous chunk.
 */

#include "common.h"

/* UDMA type0 descriptor — inline definition to avoid path issues */
typedef struct {
    void *next;
    unsigned int length:24;
    unsigned int desctype:2;
    unsigned int dstcomp:1;
    unsigned int srccomp:1;
    unsigned int dstbypass:1;
    unsigned int srcbypass:1;
    unsigned int order:1;
    unsigned int dstate:1;
    void *src;
    void *dst;
} __attribute__((aligned(64))) dma_desc_type0_t;

#define DMA_SIZE (64 * 1024)   /* 64 KB — typical weight chunk size */

/* Forward declaration of hvx_vadd (defined in demo_hvx_bench.c) */
extern void hvx_vadd(const uint8_t *a, const uint8_t *b, uint8_t *c,
                      uint32_t n_bytes);

void demo_dma(void)
{
    /*
     * llama.cpp's hex-dma.c uses a similar pattern:
     *   1. Fill descriptor
     *   2. dmstart to launch
     *   3. dmwait to wait for completion
     *   4. Chained descriptors for ping-pong buffering
     *
     * We demo the simplest single DMA transfer.
     */

    /* 1. Allocate DDR source buffer (128-byte aligned) */
    uint8_t *ddr_buf = (uint8_t *)memalign(128, DMA_SIZE);
    if (!ddr_buf) {
        FARF(ALWAYS, "  DDR memalign failed");
        return;
    }
    /* Fill test data */
    for (int i = 0; i < DMA_SIZE; i++)
        ddr_buf[i] = (uint8_t)(i & 0xFF);

    /* 2. Allocate VTCM destination buffer */
    uint8_t *ptr = g_vtcm_base;
    uint8_t *vtcm_buf = vtcm_seq_alloc(&ptr, DMA_SIZE);
    memset(vtcm_buf, 0, DMA_SIZE);

    /* 3. Set up UDMA descriptor (must be 64-byte aligned) */
    dma_desc_type0_t desc __attribute__((aligned(64)));
    memset(&desc, 0, sizeof(desc));
    desc.next      = NULL;                    /* single descriptor, no chain */
    desc.length    = DMA_SIZE;                /* transfer length */
    desc.desctype  = 0;                       /* type0 = linear transfer */
    desc.dstcomp   = 0;                       /* no compression */
    desc.srccomp   = 0;
    desc.dstbypass = 0;                       /* VTCM doesn't need bypass */
    desc.srcbypass = 1;                       /* DDR source bypasses cache */
    desc.order     = 0;
    desc.dstate    = 0;                       /* 0 = not done */
    desc.src       = (void *)ddr_buf;
    desc.dst       = (void *)vtcm_buf;

    /* 4. Flush DDR source buffer cache (DMA reads from physical memory) */
    qurt_mem_cache_clean((qurt_addr_t)ddr_buf, DMA_SIZE,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

    /* 5. Start DMA and wait */
    Q6_dmstart_A((void *)&desc);
    int dma_status = Q6_R_dmwait();

    /* 6. Verify transfer result */
    int ok = 1;
    int first_mismatch = -1;
    for (int i = 0; i < DMA_SIZE; i++) {
        if (vtcm_buf[i] != (uint8_t)(i & 0xFF)) {
            ok = 0;
            first_mismatch = i;
            break;
        }
    }

    FARF(ALWAYS, "  DMA transfer: DDR -> VTCM, %d KB", DMA_SIZE / 1024);
    FARF(ALWAYS, "    dmwait status: %d (0 = success)", dma_status);
    FARF(ALWAYS, "    verify: %s%s", ok ? "OK" : "FAIL",
         first_mismatch >= 0 ? " (mismatch)" : "");

    /* 7. Benchmark: DMA's real advantage — freeing the CPU
     *
     * Blocking DMA (dmstart + dmwait) vs memcpy is meaningless because
     * dmwait blocks the CPU, losing DMA's core value.
     *
     * DMA's value: after dmstart, CPU can do HVX/HMX compute,
     * then dmwait when compute is done. So we compare:
     *
     * Plan A (memcpy): memcpy(64KB) + HVX_compute(64KB) = serial
     * Plan B (DMA):    dmstart(64KB) + HVX_compute(64KB) + dmwait = overlapped
     */
    uint32_t n_iters = 200;

    /* Prepare HVX compute buffers (simulate dequant-like work) */
    uint8_t *compute_a = vtcm_seq_alloc(&ptr, DMA_SIZE);
    uint8_t *compute_b = vtcm_seq_alloc(&ptr, DMA_SIZE);
    uint8_t *compute_c = vtcm_seq_alloc(&ptr, DMA_SIZE);
    memset(compute_a, 1, DMA_SIZE);
    memset(compute_b, 2, DMA_SIZE);

    /* Plan A: memcpy + HVX serial */
    qurt_mem_cache_clean((qurt_addr_t)ddr_buf, DMA_SIZE,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    uint64_t t1 = HAP_perf_get_time_us();
    for (uint32_t i = 0; i < n_iters; i++) {
        memcpy(vtcm_buf, ddr_buf, DMA_SIZE);            /* move data */
        hvx_vadd(compute_a, compute_b, compute_c, DMA_SIZE); /* compute */
    }
    uint64_t t_serial = HAP_perf_get_time_us() - t1;

    /* Plan B: DMA + HVX overlapped */
    qurt_mem_cache_clean((qurt_addr_t)ddr_buf, DMA_SIZE,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    uint64_t t2 = HAP_perf_get_time_us();
    for (uint32_t i = 0; i < n_iters; i++) {
        desc.dstate = 0;
        Q6_dmstart_A((void *)&desc);                         /* start DMA (non-blocking) */
        hvx_vadd(compute_a, compute_b, compute_c, DMA_SIZE); /* compute simultaneously */
        Q6_R_dmwait();                                        /* wait for DMA */
    }
    uint64_t t_overlap = HAP_perf_get_time_us() - t2;

    /* Pure memcpy time (for reference) */
    uint64_t t3 = HAP_perf_get_time_us();
    for (uint32_t i = 0; i < n_iters; i++) {
        memcpy(vtcm_buf, ddr_buf, DMA_SIZE);
    }
    uint64_t t_memcpy = HAP_perf_get_time_us() - t3;

    FARF(ALWAYS, "  Pipeline benchmark: %d KB x %u iters", DMA_SIZE / 1024, n_iters);
    FARF(ALWAYS, "    memcpy only:      %llu us",
         (unsigned long long)t_memcpy);
    FARF(ALWAYS, "    serial (memcpy+HVX): %llu us",
         (unsigned long long)t_serial);
    FARF(ALWAYS, "    overlap (DMA+HVX):   %llu us",
         (unsigned long long)t_overlap);
    if (t_overlap > 0)
        FARF(ALWAYS, "    speedup: %.2fx (serial/overlap)",
             (float)t_serial / (float)t_overlap);

    /*
     * DMA's real advantages are not about single transfer speed:
     * 1. During DMA transfer, CPU/HVX/HMX can do other compute
     * 2. Chained descriptors (dmlink) for automatic multi-transfer
     * 3. 2D transfer (stride) for tensor slicing
     *
     * llama.cpp's hex-dma.c uses double-buffer pattern:
     *   while (chunk < n_chunks) {
     *       dma_start(buf[chunk & 1], ddr_weight + chunk * size, size);
     *       hvx_dequant(buf[(chunk-1) & 1]);   // process previous chunk
     *       hmx_compute(...);
     *       dma_wait();
     *   }
     */

    free(ddr_buf);
}
