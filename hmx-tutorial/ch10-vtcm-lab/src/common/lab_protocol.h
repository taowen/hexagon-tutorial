/*
 * ch10: VTCM lab experiment protocol
 *
 * Uses ch08's mnist_train IDL + dspqueue infrastructure.
 * Custom opcodes for VTCM experiments (must not conflict with ch08's 1-16).
 */
#ifndef LAB_PROTOCOL_H
#define LAB_PROTOCOL_H

#include <stdint.h>

/* Experiment opcodes */
#define OP_LAB_VTCM_BASIC      20  /* Write pattern to VTCM, read back immediately */
#define OP_LAB_VTCM_PERSIST    21  /* Verify VTCM value from previous message */
#define OP_LAB_HVX_WRITE       22  /* HVX vector store to VTCM, scalar verify */
#define OP_LAB_SCALAR_WRITE    23  /* Scalar store to VTCM, HVX verify */
#define OP_LAB_SGD_ITER        24  /* One HVX SGD iteration on VTCM weights */
#define OP_LAB_SGD_VERIFY      25  /* Read back VTCM weights for ARM comparison */
#define OP_LAB_TRAIN_INIT      26  /* Copy initial weights DDR->VTCM */
#define OP_LAB_TRAIN_BATCH     27  /* Forward+backward+SGD on VTCM */
#define OP_LAB_TRAIN_SYNC      28  /* Copy weights VTCM->DDR (DEPRECATED - corrupts L2) */
#define OP_LAB_TRAIN_EVAL      29  /* Evaluate test batch on DSP using VTCM weights */
#define OP_LAB_L2_STALE_TEST   30  /* Exp 7: Test if memcpy causes L2 stale reads */
#define OP_LAB_L2_INVAL_TEST   31  /* Exp 8: Test if cache invalidation fixes stale reads */
#define OP_LAB_TRAIN_SYNC_V2   32  /* Exp 9: SYNC with L2 invalidation for VTCM range */
#define OP_LAB_INCREMENT       33  /* Exp 10a: HVX increment VTCM buffer by 1.0 */
#define OP_LAB_INCREMENT_SYNC  34  /* Exp 10b: same + memcpy VTCM→heap + INVALIDATE */
#define OP_LAB_INCREMENT_FLUSH 35  /* Exp 10c: same + memcpy VTCM→heap + FLUSH_INVALIDATE */
#define OP_LAB_INCREMENT_NOCACHE 36 /* Exp 10d: same + memcpy VTCM→heap, no cache ops */
#define OP_LAB_SCALAR_WRITE_INVAL 37 /* Exp 11: scalar write VTCM → INVALIDATE → scalar read */
#define OP_LAB_TRAIN_SYNC_MCONLY  38 /* Exp 12a: SYNC memcpy only (no cache ops) */
#define OP_LAB_TRAIN_SYNC_FLUSH   39 /* Exp 12b: SYNC memcpy + flush DDR (no vtcm inval) */
#define OP_LAB_TRAIN_SYNC_NOOP    40 /* Exp 12c: SYNC no-op (just read w1[0], no memcpy) */
#define OP_LAB_TRAIN_SYNC_DDR     41 /* Exp 12d: SYNC memcpy VTCM→DDR only */
#define OP_LAB_TRAIN_SYNC_SMALL   42 /* Exp 12e: SYNC only w2+b2 (16KB, not w1 400KB) */
#define OP_LAB_TRAIN_SYNC_NOOP_EVAL 43 /* Exp 13a: no-op SYNC but ARM does cpu_evaluate */
#define OP_LAB_TRAIN_SYNC_DSPEVAL   44 /* Exp 13b: memcpy SYNC + DSP-side busy-work (no ARM idle) */

/* Lab request (generic) */
struct lab_req {
    uint32_t op;
    uint32_t iter;          /* iteration number (for persistence/SGD tests) */
    uint32_t size;          /* number of floats to test */
    uint32_t batch_size;    /* for training */
    float    learning_rate; /* for SGD/training */
    uint8_t  labels[256];   /* for training */
};

/* Lab response */
struct lab_rsp {
    uint32_t op;
    uint32_t status;        /* 0 = pass */
    uint32_t errors;        /* number of mismatches found */
    float    detail;        /* test-specific (e.g., loss) */
};

#define LAB_MAX_MESSAGE_SIZE  sizeof(struct lab_req)
#define LAB_MAX_BUFFERS       12

#endif /* LAB_PROTOCOL_H */
