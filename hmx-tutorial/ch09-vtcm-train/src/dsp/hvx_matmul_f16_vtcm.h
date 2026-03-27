#ifndef HVX_MATMUL_F16_VTCM_H
#define HVX_MATMUL_F16_VTCM_H

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>
#include <string.h>

/*
 * f16 HVX matmul with qf32 internal accumulation via widening multiply.
 *
 * Key instruction: Q6_Wqf32_vmpy_VhfVhf(a_f16, b_f16)
 *   - Takes two vectors of 64 f16 elements each
 *   - Produces a VectorPair of 2x32 qf32 outputs
 *   - Accumulate in qf32 for f32-precision dot products
 *   - Convert back to f16 via Q6_Vhf_equals_Wqf32
 *
 * No scalar f16<->f32 conversion needed. No DDR scratch buffers.
 * A rows preloaded from VTCM to stack via HVX for fast scalar access.
 * B rows read directly from VTCM via HVX wide port (1-cycle).
 *
 * All matrix dimensions (n) are multiples of 64, so row strides are
 * multiples of 128 bytes and naturally HVX-aligned in VTCM.
 * With NET_INPUT_DIM_PAD=832 (13*64), 832*2=1664=13*128, so all row
 * strides are 128-byte aligned and all paths use fast aligned loads.
 */

#define F16_VEC 64   /* f16 elements per HVX vector */

/*
 * Unaligned HVX vector load using valign.
 * Works for any byte-aligned address in VTCM or DDR.
 */
static inline HVX_Vector hvx_load_unaligned(const void *ptr) {
    uintptr_t addr = (uintptr_t)ptr;
    int offset = (int)(addr & 127);
    const HVX_Vector *base = (const HVX_Vector *)(addr & ~(uintptr_t)127);
    if (offset == 0) return base[0];
    /* valign(Vu, Vv, Rt): extracts 128 bytes from Vu:Vv starting at byte Rt
     * Vu = higher address vector (base[1]), Vv = lower address vector (base[0]) */
    return Q6_V_valign_VVR(base[1], base[0], offset);
}

/*
 * Preload a row of k f16 elements from potentially unaligned source
 * to a stack-aligned buffer.
 */
static inline void preload_row_f16(_Float16 *dst, const _Float16 *src, uint32_t k) {
    uint32_t bytes = k * sizeof(_Float16);
    uintptr_t addr = (uintptr_t)src;
    if ((addr & 127) == 0 && bytes >= 128) {
        /* Aligned fast path */
        uint32_t vecs = bytes / 128;
        const HVX_Vector *s = (const HVX_Vector *)src;
        HVX_Vector *d = (HVX_Vector *)dst;
        for (uint32_t v = 0; v < vecs; v++)
            d[v] = s[v];
        uint32_t done = vecs * F16_VEC;
        if (done < k) {
            HVX_Vector last = hvx_load_unaligned(src + done);
            _Float16 tmp[F16_VEC] __attribute__((aligned(128)));
            *(HVX_Vector *)tmp = last;
            for (uint32_t i = 0; i < k - done; i++)
                dst[done + i] = tmp[i];
        }
    } else if (bytes >= 128) {
        /* Unaligned path: use hvx_load_unaligned for each vector chunk */
        uint32_t vecs = bytes / 128;
        HVX_Vector *d = (HVX_Vector *)dst;
        for (uint32_t v = 0; v < vecs; v++)
            d[v] = hvx_load_unaligned((const char *)src + v * 128);
        uint32_t done = vecs * F16_VEC;
        if (done < k) {
            HVX_Vector last = hvx_load_unaligned(src + done);
            _Float16 tmp[F16_VEC] __attribute__((aligned(128)));
            *(HVX_Vector *)tmp = last;
            for (uint32_t i = 0; i < k - done; i++)
                dst[done + i] = tmp[i];
        }
    } else {
        /* Small row: scalar copy (only for VTCM reads < 128 bytes) */
        for (uint32_t i = 0; i < k; i++)
            dst[i] = src[i];
    }
}

/*
 * Write back a row of count f16 elements to a potentially unaligned destination.
 * Uses scalar writes for safety when dest is not 128-byte aligned.
 */
static inline void writeback_row_f16(_Float16 *dst, const _Float16 *src, uint32_t count) {
    uint32_t bytes = count * sizeof(_Float16);
    if (((uintptr_t)dst & 127) == 0 && bytes >= 128) {
        uint32_t vecs = bytes / 128;
        const HVX_Vector *s = (const HVX_Vector *)src;
        HVX_Vector *d = (HVX_Vector *)dst;
        for (uint32_t v = 0; v < vecs; v++)
            d[v] = s[v];
        uint32_t done = vecs * F16_VEC;
        for (uint32_t i = done; i < count; i++)
            dst[i] = src[i];
    } else {
        for (uint32_t i = 0; i < count; i++)
            dst[i] = src[i];
    }
}


/*
 * matmul_nn_f16: C[m,n] = A[m,k] @ B[k,n]
 *
 * REQUIRES: n and k are multiples of 64 (so all row strides are 128B-aligned).
 * Called for layer 2 forward: n=64, k=128.
 * A rows preloaded to stack for fast scalar access.
 * B and C rows accessed directly via aligned HVX loads/stores.
 */
static void matmul_nn_f16(
    _Float16 *C, const _Float16 *A, const _Float16 *B,
    uint32_t m, uint32_t n, uint32_t k)
{
    uint32_t n_vecs = n / F16_VEC;
    uint32_t k_vecs = k / F16_VEC;

    /* Stack buffer for A row preload from VTCM */
    _Float16 a_local[832] __attribute__((aligned(128)));

    for (uint32_t i = 0; i < m; i++) {
        const _Float16 *a_row = A + (size_t)i * k;

        /* Preload A row from VTCM to stack via aligned HVX copy */
        const HVX_Vector *a_src = (const HVX_Vector *)a_row;
        HVX_Vector *a_dst = (HVX_Vector *)a_local;
        for (uint32_t v = 0; v < k_vecs; v++)
            a_dst[v] = a_src[v];

        _Float16 *c_row = C + (size_t)i * n;

        for (uint32_t jb = 0; jb < n_vecs; jb++) {
            HVX_Vector acc_lo = Q6_V_vzero();
            HVX_Vector acc_hi = Q6_V_vzero();

            for (uint32_t p = 0; p < k; p++) {
                /* Scalar read from stack (L2, fast) */
                _Float16 a_val = a_local[p];
                uint32_t a_bits;
                memcpy(&a_bits, &a_val, 2);
                a_bits &= 0xFFFF;

                /* Splat f16 value to all 64 positions */
                HVX_Vector a_splat = Q6_Vh_vsplat_R(a_bits);

                /* Aligned load: B row stride = n*2 is multiple of 128 */
                const HVX_Vector *b_row = (const HVX_Vector *)(B + (size_t)p * n);
                HVX_Vector b_f16 = b_row[jb];

                /* Widening multiply: f16 x f16 -> qf32 pair */
                HVX_VectorPair prod = Q6_Wqf32_vmpy_VhfVhf(a_splat, b_f16);

                /* Accumulate in qf32 */
                acc_lo = Q6_Vqf32_vadd_Vqf32Vqf32(acc_lo, Q6_V_lo_W(prod));
                acc_hi = Q6_Vqf32_vadd_Vqf32Vqf32(acc_hi, Q6_V_hi_W(prod));
            }

            /* Convert qf32 pair back to f16 vector and store aligned */
            ((HVX_Vector *)c_row)[jb] = Q6_Vhf_equals_Wqf32(
                Q6_W_vcombine_VV(acc_hi, acc_lo));
        }
    }
}

/*
 * matmul_nn_f16_2x: C[m,n] = A[m,k] @ B[k,n]
 * 2x column-unrolled variant. REQUIRES n % 128 == 0 (rows 256-byte aligned).
 * A rows may be unaligned (preloaded to stack).
 * B rows are aligned since n % 128 == 0 implies stride is multiple of 256 bytes.
 */
static void matmul_nn_f16_2x(
    _Float16 *C, const _Float16 *A, const _Float16 *B,
    uint32_t m, uint32_t n, uint32_t k)
{
    uint32_t n_vecs = n / F16_VEC;
    _Float16 a_local[832] __attribute__((aligned(128)));

    for (uint32_t i = 0; i < m; i++) {
        const _Float16 *a_row = A + (size_t)i * k;

        /* Preload A row from VTCM to stack (handles unaligned) */
        preload_row_f16(a_local, a_row, k);

        _Float16 *c_row = C + (size_t)i * n;

        for (uint32_t jb = 0; jb < n_vecs; jb += 2) {
            HVX_Vector acc0_lo = Q6_V_vzero(), acc0_hi = Q6_V_vzero();
            HVX_Vector acc1_lo = Q6_V_vzero(), acc1_hi = Q6_V_vzero();

            for (uint32_t p = 0; p < k; p++) {
                _Float16 a_val = a_local[p];
                uint32_t a_bits;
                memcpy(&a_bits, &a_val, 2);
                a_bits &= 0xFFFF;
                HVX_Vector a_splat = Q6_Vh_vsplat_R(a_bits);

                /* B rows are aligned (n % 128 == 0) */
                const HVX_Vector *b_row = (const HVX_Vector *)(B + (size_t)p * n);

                HVX_VectorPair p0 = Q6_Wqf32_vmpy_VhfVhf(a_splat, b_row[jb]);
                acc0_lo = Q6_Vqf32_vadd_Vqf32Vqf32(acc0_lo, Q6_V_lo_W(p0));
                acc0_hi = Q6_Vqf32_vadd_Vqf32Vqf32(acc0_hi, Q6_V_hi_W(p0));

                HVX_VectorPair p1 = Q6_Wqf32_vmpy_VhfVhf(a_splat, b_row[jb + 1]);
                acc1_lo = Q6_Vqf32_vadd_Vqf32Vqf32(acc1_lo, Q6_V_lo_W(p1));
                acc1_hi = Q6_Vqf32_vadd_Vqf32Vqf32(acc1_hi, Q6_V_hi_W(p1));
            }

            /* C rows are aligned (n % 128 == 0) */
            ((HVX_Vector *)c_row)[jb] = Q6_Vhf_equals_Wqf32(
                Q6_W_vcombine_VV(acc0_hi, acc0_lo));
            ((HVX_Vector *)c_row)[jb + 1] = Q6_Vhf_equals_Wqf32(
                Q6_W_vcombine_VV(acc1_hi, acc1_lo));
        }
    }
}

/*
 * matmul_tn_f16: C[m,n] = A^T @ B
 * A stored as [k,m], B stored as [k,n].
 *
 * Preloads entire A[k,m] from VTCM to DDR static buffer for L2 access.
 *
 * Two paths:
 *   - Aligned (n % 64 == 0, e.g. n=128): direct aligned HVX loads/stores,
 *     no tail, no staging buffer. B row stride = n*2 is multiple of 128.
 *   - Unaligned (n not multiple of 64): Uses hvx_load_unaligned +
 *     writeback_row_f16 + tail handling.
 */
static _Float16 g_tn_a_buf[256 * 832] __attribute__((aligned(128)));

static void matmul_tn_f16(
    _Float16 *C, const _Float16 *A, const _Float16 *B,
    uint32_t m, uint32_t n, uint32_t k)
{
    uint32_t n_vecs = n / F16_VEC;

    /* Preload entire A[k,m] from VTCM to DDR static buffer */
    uint32_t a_total_vecs = (k * m * sizeof(_Float16) + 127) / 128;
    for (uint32_t v = 0; v < a_total_vecs; v++)
        ((HVX_Vector *)g_tn_a_buf)[v] = ((const HVX_Vector *)A)[v];

    /* Aligned fast-path: n is multiple of 64, so B/C row strides are 128B-aligned */
    if ((n % F16_VEC) == 0) {
        for (uint32_t j = 0; j < m; j++) {
            _Float16 *c_row = C + (size_t)j * n;

            for (uint32_t vb = 0; vb < n_vecs; vb++) {
                HVX_Vector acc_lo = Q6_V_vzero();
                HVX_Vector acc_hi = Q6_V_vzero();

                for (uint32_t i = 0; i < k; i++) {
                    _Float16 a_val = g_tn_a_buf[(size_t)i * m + j];
                    uint32_t a_bits;
                    memcpy(&a_bits, &a_val, 2);
                    a_bits &= 0xFFFF;
                    HVX_Vector a_splat = Q6_Vh_vsplat_R(a_bits);

                    const HVX_Vector *b_row = (const HVX_Vector *)(B + (size_t)i * n);
                    HVX_VectorPair prod = Q6_Wqf32_vmpy_VhfVhf(a_splat, b_row[vb]);
                    acc_lo = Q6_Vqf32_vadd_Vqf32Vqf32(acc_lo, Q6_V_lo_W(prod));
                    acc_hi = Q6_Vqf32_vadd_Vqf32Vqf32(acc_hi, Q6_V_hi_W(prod));
                }

                ((HVX_Vector *)c_row)[vb] = Q6_Vhf_equals_Wqf32(
                    Q6_W_vcombine_VV(acc_hi, acc_lo));
            }
        }
        return;
    }

    /* Unaligned path for n not a multiple of 64.
     * Uses staging buffer + unaligned loads + tail handling. */
    _Float16 c_local[832] __attribute__((aligned(128)));

    for (uint32_t j = 0; j < m; j++) {
        for (uint32_t vb = 0; vb < n_vecs; vb++) {
            HVX_Vector acc_lo = Q6_V_vzero();
            HVX_Vector acc_hi = Q6_V_vzero();

            for (uint32_t i = 0; i < k; i++) {
                _Float16 a_val = g_tn_a_buf[(size_t)i * m + j];
                uint32_t a_bits;
                memcpy(&a_bits, &a_val, 2);
                a_bits &= 0xFFFF;
                HVX_Vector a_splat = Q6_Vh_vsplat_R(a_bits);

                const _Float16 *b_ptr = B + (size_t)i * n + vb * F16_VEC;
                HVX_Vector b_f16 = hvx_load_unaligned(b_ptr);

                HVX_VectorPair prod = Q6_Wqf32_vmpy_VhfVhf(a_splat, b_f16);
                acc_lo = Q6_Vqf32_vadd_Vqf32Vqf32(acc_lo, Q6_V_lo_W(prod));
                acc_hi = Q6_Vqf32_vadd_Vqf32Vqf32(acc_hi, Q6_V_hi_W(prod));
            }

            ((HVX_Vector *)c_local)[vb] = Q6_Vhf_equals_Wqf32(
                Q6_W_vcombine_VV(acc_hi, acc_lo));
        }

        /* Tail: remaining elements when n is not a multiple of 64 */
        uint32_t tail_start = n_vecs * F16_VEC;
        if (tail_start < n) {
            HVX_Vector acc_lo = Q6_V_vzero();
            HVX_Vector acc_hi = Q6_V_vzero();

            for (uint32_t i = 0; i < k; i++) {
                _Float16 a_val = g_tn_a_buf[(size_t)i * m + j];
                uint32_t a_bits;
                memcpy(&a_bits, &a_val, 2);
                a_bits &= 0xFFFF;
                HVX_Vector a_splat = Q6_Vh_vsplat_R(a_bits);

                const _Float16 *b_ptr = B + (size_t)i * n + tail_start;
                HVX_Vector b_f16 = hvx_load_unaligned(b_ptr);

                HVX_VectorPair prod = Q6_Wqf32_vmpy_VhfVhf(a_splat, b_f16);
                acc_lo = Q6_Vqf32_vadd_Vqf32Vqf32(acc_lo, Q6_V_lo_W(prod));
                acc_hi = Q6_Vqf32_vadd_Vqf32Vqf32(acc_hi, Q6_V_hi_W(prod));
            }

            HVX_Vector result = Q6_Vhf_equals_Wqf32(
                Q6_W_vcombine_VV(acc_hi, acc_lo));
            _Float16 tmp[F16_VEC] __attribute__((aligned(128)));
            *(HVX_Vector *)tmp = result;
            for (uint32_t t = 0; t < n - tail_start; t++)
                c_local[tail_start + t] = tmp[t];
        }

        /* Write back C row (unaligned dest) */
        _Float16 *c_row = C + (size_t)j * n;
        writeback_row_f16(c_row, c_local, n);
    }
}


/*
 * matmul_nt_f16: C[m,n] = A[m,k] @ B^T, where B is stored as [n,k].
 *
 * C[i,j] = sum_p A[i,p] * B[j,p]
 *
 * Each output element is a dot product of two contiguous rows (length k).
 * REQUIRES: k is a multiple of 64 (HVX-aligned rows).
 *
 * Used for backward pass: dhidden = dlogits @ W2 when only W2^T is stored.
 *   A = dlogits[bs,64], B = W2_T[128,64] -> C = dhidden[bs,128]
 *
 * Strategy: For each output row i, preload A[i,:] to stack.
 * For each output column j, compute dot(A[i,:], B[j,:]) using
 * HVX widening multiply + horizontal reduction.
 *
 * With k=64: 1 HVX vmpy -> 64 products -> reduce to 1 scalar f16.
 * With k=128: 2 HVX vmpy -> accumulate -> reduce to 1 scalar f16.
 */
static void matmul_nt_f16(
    _Float16 *C, const _Float16 *A, const _Float16 *B,
    uint32_t m, uint32_t n, uint32_t k)
{
    uint32_t k_vecs = k / F16_VEC;

    /* Stack buffer for A row preload */
    _Float16 a_local[832] __attribute__((aligned(128)));

    for (uint32_t i = 0; i < m; i++) {
        const _Float16 *a_row = A + (size_t)i * k;

        /* Preload A row from VTCM to stack */
        const HVX_Vector *a_src = (const HVX_Vector *)a_row;
        HVX_Vector *a_dst = (HVX_Vector *)a_local;
        for (uint32_t v = 0; v < k_vecs; v++)
            a_dst[v] = a_src[v];

        _Float16 *c_row = C + (size_t)i * n;

        for (uint32_t j = 0; j < n; j++) {
            const _Float16 *b_row = B + (size_t)j * k;

            /* Accumulate dot product in qf32 */
            HVX_Vector acc_lo = Q6_V_vzero();
            HVX_Vector acc_hi = Q6_V_vzero();

            for (uint32_t v = 0; v < k_vecs; v++) {
                HVX_Vector a_vec = ((const HVX_Vector *)a_local)[v];
                HVX_Vector b_vec = ((const HVX_Vector *)b_row)[v];
                HVX_VectorPair prod = Q6_Wqf32_vmpy_VhfVhf(a_vec, b_vec);
                acc_lo = Q6_Vqf32_vadd_Vqf32Vqf32(acc_lo, Q6_V_lo_W(prod));
                acc_hi = Q6_Vqf32_vadd_Vqf32Vqf32(acc_hi, Q6_V_hi_W(prod));
            }

            /* Convert qf32 pair to f16 vector: 64 f16 products */
            HVX_Vector f16_vec = Q6_Vhf_equals_Wqf32(
                Q6_W_vcombine_VV(acc_hi, acc_lo));

            /* Horizontal reduction: sum 64 f16 elements to 1 scalar.
             * Use qf16 accumulation for precision:
             * 1) Convert f16 -> qf16 for addition
             * 2) Shuffle-add tree: 6 rounds to reduce 64 -> 1
             * 3) Extract lane 0 */
            HVX_Vector sum = Q6_Vqf16_vadd_VhfVhf(f16_vec, Q6_V_vzero());

            /* Round 1: add upper 32 to lower 32 (deal/shuffle by 64 bytes) */
            sum = Q6_Vqf16_vadd_Vqf16Vqf16(sum,
                Q6_V_vror_VR(sum, 64));
            /* Round 2: add upper 16 to lower 16 (rotate by 32 bytes) */
            sum = Q6_Vqf16_vadd_Vqf16Vqf16(sum,
                Q6_V_vror_VR(sum, 32));
            /* Round 3: add upper 8 to lower 8 (rotate by 16 bytes) */
            sum = Q6_Vqf16_vadd_Vqf16Vqf16(sum,
                Q6_V_vror_VR(sum, 16));
            /* Round 4: add upper 4 to lower 4 (rotate by 8 bytes) */
            sum = Q6_Vqf16_vadd_Vqf16Vqf16(sum,
                Q6_V_vror_VR(sum, 8));
            /* Round 5: add upper 2 to lower 2 (rotate by 4 bytes) */
            sum = Q6_Vqf16_vadd_Vqf16Vqf16(sum,
                Q6_V_vror_VR(sum, 4));
            /* Round 6: add upper 1 to lower 1 (rotate by 2 bytes) */
            sum = Q6_Vqf16_vadd_Vqf16Vqf16(sum,
                Q6_V_vror_VR(sum, 2));

            /* Convert qf16 back to f16 and extract lane 0 */
            HVX_Vector result_f16 = Q6_Vhf_equals_Vqf16(sum);
            _Float16 tmp[F16_VEC] __attribute__((aligned(128)));
            *(HVX_Vector *)tmp = result_f16;
            c_row[j] = tmp[0];
        }
    }
}


/*
 * blocked_transpose_f16_vtcm: dst[cols,rows] = transpose(src[rows,cols])
 * Both src and dst in VTCM.
 */
#define TRANS_BLK 32
static void blocked_transpose_f16_vtcm(
    _Float16 *dst, const _Float16 *src,
    uint32_t rows, uint32_t cols)
{
    _Float16 blk_in[TRANS_BLK * TRANS_BLK] __attribute__((aligned(128)));
    _Float16 blk_out[TRANS_BLK * TRANS_BLK] __attribute__((aligned(128)));

    for (uint32_t rb = 0; rb < rows; rb += TRANS_BLK) {
        uint32_t re = (rb + TRANS_BLK <= rows) ? rb + TRANS_BLK : rows;
        for (uint32_t cb = 0; cb < cols; cb += TRANS_BLK) {
            uint32_t ce = (cb + TRANS_BLK <= cols) ? cb + TRANS_BLK : cols;
            uint32_t bh = re - rb, bw = ce - cb;

            for (uint32_t r = 0; r < bh; r++) {
                const _Float16 *s = src + (size_t)(rb + r) * cols + cb;
                _Float16 *d = blk_in + r * TRANS_BLK;
                memcpy(d, s, bw * sizeof(_Float16));
            }

            for (uint32_t r = 0; r < bh; r++)
                for (uint32_t c = 0; c < bw; c++)
                    blk_out[c * TRANS_BLK + r] = blk_in[r * TRANS_BLK + c];

            for (uint32_t c = 0; c < bw; c++) {
                _Float16 *d = dst + (size_t)(cb + c) * rows + rb;
                _Float16 *s = blk_out + c * TRANS_BLK;
                memcpy(d, s, bh * sizeof(_Float16));
            }
        }
    }
}

#endif /* HVX_MATMUL_F16_VTCM_H */
