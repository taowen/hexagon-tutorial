/*
 * Part 3: TVM-style pool allocator (simplified demo)
 *
 * TVM's HexagonVtcmPool implements:
 *   - free list management (multiple free blocks)
 *   - best-fit allocation (2K-aligned large blocks from front)
 *   - small blocks from back (reduce fragmentation)
 *   - thread safety (mutex)
 *
 * We demo a simplified version: single free list + first-fit.
 */

#include "common.h"

#define VTCM_POOL_MAX_BLOCKS 16

struct vtcm_block {
    uint8_t *addr;
    size_t   size;
    int      used;
};

struct vtcm_pool {
    struct vtcm_block blocks[VTCM_POOL_MAX_BLOCKS];
    int n_blocks;
};

static void pool_init(struct vtcm_pool *pool, uint8_t *base, size_t size) {
    pool->blocks[0].addr = base;
    pool->blocks[0].size = size;
    pool->blocks[0].used = 0;
    pool->n_blocks = 1;
}

static uint8_t *pool_alloc(struct vtcm_pool *pool, size_t size) {
    /* Align to 2048 bytes (TVM HMX tile alignment) */
    size = (size + 2047) & ~(size_t)2047;

    /* first-fit (TVM uses best-fit, we simplify) */
    for (int i = 0; i < pool->n_blocks; i++) {
        struct vtcm_block *b = &pool->blocks[i];
        if (!b->used && b->size >= size) {
            /* Split: carve out size, remainder becomes new block */
            if (b->size > size && pool->n_blocks < VTCM_POOL_MAX_BLOCKS) {
                for (int j = pool->n_blocks; j > i + 1; j--)
                    pool->blocks[j] = pool->blocks[j - 1];
                pool->blocks[i + 1].addr = b->addr + size;
                pool->blocks[i + 1].size = b->size - size;
                pool->blocks[i + 1].used = 0;
                pool->n_blocks++;
                b->size = size;
            }
            b->used = 1;
            return b->addr;
        }
    }
    return NULL;   /* allocation failed */
}

static void pool_free(struct vtcm_pool *pool, uint8_t *ptr) {
    for (int i = 0; i < pool->n_blocks; i++) {
        if (pool->blocks[i].addr == ptr) {
            pool->blocks[i].used = 0;
            /* Coalesce adjacent free blocks */
            if (i + 1 < pool->n_blocks && !pool->blocks[i + 1].used) {
                pool->blocks[i].size += pool->blocks[i + 1].size;
                for (int j = i + 1; j < pool->n_blocks - 1; j++)
                    pool->blocks[j] = pool->blocks[j + 1];
                pool->n_blocks--;
            }
            if (i > 0 && !pool->blocks[i - 1].used) {
                pool->blocks[i - 1].size += pool->blocks[i].size;
                for (int j = i; j < pool->n_blocks - 1; j++)
                    pool->blocks[j] = pool->blocks[j + 1];
                pool->n_blocks--;
            }
            return;
        }
    }
}

void demo_pool_alloc(void)
{
    /*
     * Simulate TVM scenario: compiler-generated op graph with dynamic VTCM needs.
     *
     * TVM code (simplified):
     *   HexagonVtcmPool pool;   // acquire all VTCM at construction
     *   void *buf_a = pool.Allocate(64 * 1024);   // conv2d input
     *   void *buf_b = pool.Allocate(32 * 1024);   // conv2d weight
     *   // ... compute ...
     *   pool.Free(buf_a);                          // release input
     *   void *buf_c = pool.Allocate(48 * 1024);   // next op
     *   // ... alternate alloc/free ...
     */
    struct vtcm_pool pool;
    pool_init(&pool, g_vtcm_base, g_vtcm_size);

    /* Allocate 3 blocks */
    uint8_t *a = pool_alloc(&pool, 64 * 1024);    /* 64 KB */
    uint8_t *b = pool_alloc(&pool, 32 * 1024);    /* 32 KB */
    uint8_t *c = pool_alloc(&pool, 128 * 1024);   /* 128 KB */

    FARF(ALWAYS, "  Pool allocator (TVM style):");
    FARF(ALWAYS, "    alloc A: 64 KB  at +0x%05X",
         a ? (unsigned)(a - g_vtcm_base) : 0);
    FARF(ALWAYS, "    alloc B: 32 KB  at +0x%05X",
         b ? (unsigned)(b - g_vtcm_base) : 0);
    FARF(ALWAYS, "    alloc C: 128 KB at +0x%05X",
         c ? (unsigned)(c - g_vtcm_base) : 0);
    FARF(ALWAYS, "    blocks: %d", pool.n_blocks);

    /* Free B (middle block) -> creates fragmentation */
    pool_free(&pool, b);
    FARF(ALWAYS, "    free B -> blocks: %d (hole at +0x%05X)",
         pool.n_blocks, (unsigned)(b - g_vtcm_base));

    /* Allocate 16 KB -> should fill into B's hole */
    uint8_t *d = pool_alloc(&pool, 16 * 1024);
    FARF(ALWAYS, "    alloc D: 16 KB  at +0x%05X (reuse B's hole)",
         d ? (unsigned)(d - g_vtcm_base) : 0);
    FARF(ALWAYS, "    blocks: %d", pool.n_blocks);

    /* Free all -> coalesce back to one block */
    pool_free(&pool, a);
    pool_free(&pool, c);
    pool_free(&pool, d);
    FARF(ALWAYS, "    free all -> blocks: %d (coalesced)",
         pool.n_blocks);

    /* Verify VTCM access */
    if (a) memset(a, 0x55, 64 * 1024);
    if (c) memset(c, 0x66, 128 * 1024);
    int ok = (a && c && a[0] == 0x55 && c[0] == 0x66);
    FARF(ALWAYS, "    read/write: %s", ok ? "OK" : "FAIL");
}
