/*
 * mnist_data.h -- MNIST IDX file loading functions
 */

#ifndef MNIST_DATA_H
#define MNIST_DATA_H

#include "mnist_common.h"

/* ====================================================================
 * MNIST IDX file loading
 * ==================================================================== */

static uint32_t read_be32(FILE *f) {
    uint8_t buf[4];
    if (fread(buf, 1, 4, f) != 4) return 0;
    return ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) |
           ((uint32_t)buf[2] << 8)  |  (uint32_t)buf[3];
}

static float *load_mnist_images(const char *path, int *out_count) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("[ERROR] Cannot open %s\n", path); return NULL; }

    uint32_t magic = read_be32(f);
    if (magic != 0x00000803) {
        printf("[ERROR] Invalid magic 0x%08x in %s\n", magic, path);
        fclose(f); return NULL;
    }

    uint32_t count = read_be32(f);
    uint32_t rows  = read_be32(f);
    uint32_t cols  = read_be32(f);
    if (rows != 28 || cols != 28) {
        printf("[ERROR] Unexpected image size %ux%u\n", rows, cols);
        fclose(f); return NULL;
    }

    float *data = (float *)calloc((size_t)count * INPUT_DIM_PAD, sizeof(float));
    if (!data) { fclose(f); return NULL; }

    uint8_t pixel_buf[784];
    for (uint32_t i = 0; i < count; i++) {
        if (fread(pixel_buf, 1, 784, f) != 784) {
            free(data); fclose(f); return NULL;
        }
        float *dst = data + (size_t)i * INPUT_DIM_PAD;
        for (int j = 0; j < 784; j++)
            dst[j] = (float)pixel_buf[j] / 255.0f;
    }

    fclose(f);
    *out_count = (int)count;
    printf("[DATA] Loaded %u images from %s (padded %d->%d)\n",
           count, path, INPUT_DIM, INPUT_DIM_PAD);
    return data;
}

static uint8_t *load_mnist_labels(const char *path, int *out_count) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("[ERROR] Cannot open %s\n", path); return NULL; }

    uint32_t magic = read_be32(f);
    if (magic != 0x00000801) {
        printf("[ERROR] Invalid magic 0x%08x in %s\n", magic, path);
        fclose(f); return NULL;
    }

    uint32_t count = read_be32(f);
    uint8_t *labels = (uint8_t *)malloc(count);
    if (!labels) { fclose(f); return NULL; }
    if (fread(labels, 1, count, f) != count) {
        free(labels); fclose(f); return NULL;
    }

    fclose(f);
    *out_count = (int)count;
    printf("[DATA] Loaded %u labels from %s\n", count, path);
    return labels;
}

#endif /* MNIST_DATA_H */
