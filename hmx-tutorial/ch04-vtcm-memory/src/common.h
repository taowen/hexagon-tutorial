/*
 * common.h — shared declarations for ch06 VTCM demos
 */
#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>
#include "HAP_farf.h"
#include "HAP_compute_res.h"
#include "HAP_power.h"
#include "HAP_perf.h"
#include "qurt_memory.h"

/* ---- VTCM globals ---- */
extern uint8_t  *g_vtcm_base;
extern uint32_t  g_vtcm_size;
extern uint32_t  g_vtcm_ctx;

/* ---- Power on HVX (same as ch02) ---- */
int power_on(void);

/* ---- Bump allocator from llama.cpp ---- */
static inline uint8_t *vtcm_seq_alloc(uint8_t **ptr, size_t size) {
    uint8_t *p = *ptr;
    size = (size + 127) & ~(size_t)127;   /* 128 byte alignment */
    *ptr += size;
    return p;
}

/* ---- Demo entry points ---- */
void demo_vtcm_alloc(void);
void demo_bump_alloc(void);
void demo_pool_alloc(void);
void demo_dma(void);
void demo_hvx_bench(void);

#endif /* COMMON_H */
