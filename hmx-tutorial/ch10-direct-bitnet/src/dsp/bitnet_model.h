/*
 * bitnet_model.h -- BitNet model: decoder layers + final norm on DSP
 *
 * The DSP side only runs the decoder layers and final RMSNorm.
 * Embedding lookup and LM head (tied-weight argmax) are done on ARM.
 *
 * This avoids mapping the 626MB embedding table into DSP virtual
 * address space, keeping the DSP buffer under ~1GB.
 */

#ifndef BITNET_MODEL_H
#define BITNET_MODEL_H

#include "dsp/bitnet_decoder.h"
#include <string.h>
#include <stdlib.h>

#define NUM_LAYERS  30
#define MAX_SEQ_LEN 512  /* limit for KV cache memory */

/* ========== Full model state (DSP side) ========== */

typedef struct {
    /* Final RMSNorm weight: [HIDDEN_SIZE] f32 */
    const float *final_norm_w;

    /* Per-layer weights */
    DecoderLayerWeights layers[NUM_LAYERS];

    /* KV caches for all layers */
    KVCache kv_caches[NUM_LAYERS];

    /* Shared RoPE tables (all layers share the same tables) */
    const float *rope_cos;  /* [max_pos * HEAD_DIM/2] */
    const float *rope_sin;  /* [max_pos * HEAD_DIM/2] */

    /* Configuration */
    int num_layers;    /* actual number of layers (may differ from NUM_LAYERS) */
    int max_seq_len;   /* allocated KV cache capacity */
} BitNetModel;

/* ========== Decoder layers + final norm forward ========== */

/*
 * Run all decoder layers + final RMSNorm on the hidden state (in-place).
 *
 * hidden: [HIDDEN_SIZE] f32 input/output buffer
 * pos:    token position in the sequence (for RoPE / KV cache)
 *
 * This is called by the DSP OP_FORWARD handler.  The ARM side handles
 * embedding lookup before this call and LM head argmax after.
 */
static void bitnet_layers_forward(
    BitNetModel *model,
    float *hidden,     /* [HIDDEN_SIZE] in/out */
    int pos)
{
    /* Run all decoder layers */
    for (int l = 0; l < model->num_layers; l++) {
        bitnet_decoder_layer(hidden, pos, &model->layers[l],
                             &model->kv_caches[l]);
    }

    /* Final RMSNorm */
    hvx_rmsnorm_f32(hidden, model->final_norm_w, hidden,
                     HIDDEN_SIZE, RMS_NORM_EPS);
}

/* ========== KV cache management ========== */

/*
 * Allocate KV caches for all layers.
 *
 * Each layer gets [NUM_KV_HEADS * max_seq_len * HEAD_DIM] floats for
 * both K and V caches.  Layout: [NUM_KV_HEADS][max_seq_len][HEAD_DIM].
 *
 * Returns 0 on success, -1 on allocation failure (partial allocs are freed).
 */
static int bitnet_model_init_kv(BitNetModel *model, int max_seq_len) {
    size_t kv_size = (size_t)max_seq_len * NUM_KV_HEADS * HEAD_DIM * sizeof(float);

    for (int l = 0; l < model->num_layers; l++) {
        model->kv_caches[l].k_cache = (float *)memalign(128, kv_size);
        model->kv_caches[l].v_cache = (float *)memalign(128, kv_size);
        model->kv_caches[l].seq_len = 0;
        model->kv_caches[l].max_seq_len = max_seq_len;

        if (!model->kv_caches[l].k_cache || !model->kv_caches[l].v_cache) {
            FARF(HIGH, "Failed to allocate KV cache for layer %d "
                 "(need %u bytes each)", l, (unsigned)kv_size);
            /* Free everything allocated so far */
            for (int j = 0; j <= l; j++) {
                free(model->kv_caches[j].k_cache);
                free(model->kv_caches[j].v_cache);
                model->kv_caches[j].k_cache = NULL;
                model->kv_caches[j].v_cache = NULL;
            }
            return -1;
        }
        memset(model->kv_caches[l].k_cache, 0, kv_size);
        memset(model->kv_caches[l].v_cache, 0, kv_size);
    }

    model->max_seq_len = max_seq_len;
    return 0;
}

/*
 * Reset KV caches (clear all cached positions, keep allocations).
 */
static void bitnet_model_reset_kv(BitNetModel *model) {
    size_t kv_size = (size_t)model->max_seq_len * NUM_KV_HEADS *
                     HEAD_DIM * sizeof(float);
    for (int l = 0; l < model->num_layers; l++) {
        if (model->kv_caches[l].k_cache) {
            memset(model->kv_caches[l].k_cache, 0, kv_size);
            memset(model->kv_caches[l].v_cache, 0, kv_size);
            model->kv_caches[l].seq_len = 0;
        }
    }
}

/*
 * Free KV caches for all layers.
 */
static void bitnet_model_free_kv(BitNetModel *model) {
    for (int l = 0; l < model->num_layers; l++) {
        if (model->kv_caches[l].k_cache) {
            free(model->kv_caches[l].k_cache);
            model->kv_caches[l].k_cache = NULL;
        }
        if (model->kv_caches[l].v_cache) {
            free(model->kv_caches[l].v_cache);
            model->kv_caches[l].v_cache = NULL;
        }
        model->kv_caches[l].seq_len = 0;
        model->kv_caches[l].max_seq_len = 0;
    }
}

#endif /* BITNET_MODEL_H */
