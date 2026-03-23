/*
 * QHPI Custom Op: HvxHmxMix — 混合 HVX + HMX 算子
 *
 * 演示在 QNN 自定义算子中同时使用 HVX 和 HMX:
 *   1. HVX: 对 activation 做 ReLU 预处理 (负值清零)
 *   2. HMX: 32x32 F16 矩阵乘法
 *   3. HVX: 对输出做 ReLU 后处理 (确保无负值)
 *
 * 输入:
 *   - activation: [1,1,32,32] F16 (VTCM) — 可能含负值
 *   - weight:     [1,1,32,32] F16 (VTCM)
 *   - bias_cfg:   [1,1,1,128] F16 (VTCM) — HMX scale 配置 (256 bytes)
 * 输出:
 *   - result:     [1,1,32,32] F16 (VTCM)
 *
 * 流水线: HVX ReLU → HMX matmul → HVX ReLU
 *
 * This file uses portable HVX/HMX intrinsics that work on BOTH platforms:
 *   - On hexagon: native compiler intrinsics
 *   - On x86: provided by libnative.a + headers in libnative/include/
 */

#include "HTP/core/qhpi.h"
#include <cstdint>
#include <cstring>

#ifdef __hexagon__
// Hexagon compiler provides these natively
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>
#include <hmx_hexagon_protos.h>
#else
// libnative provides x86 emulation of the same intrinsics
#include "hvx_hexagon_protos.h"
#include "hmx_hexagon_protos.h"
#endif

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

/* ============================================================
 * HMX matmul core
 * activation + weight loads MUST be in same function/VLIW packet
 * ============================================================ */
__attribute__((noinline))
static void hmx_matmul_f16(uintptr_t act_addr, uintptr_t wt_addr) {
    Q6_activation_hf_mxmem_RR(act_addr, 32767);
    Q6_weight_hf_mxmem_RR(wt_addr, 1920);
}

/* ============================================================
 * Kernel: HvxHmxMix (VTCM + HVX + HMX)
 * ============================================================ */
static uint32_t hvx_hmx_mix_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
    (void)handle;
    (void)num_outputs; (void)num_inputs;

    /* 获取 VTCM 指针 */
    uint16_t *act_ptr  = (uint16_t *)qhpi_tensor_raw_data(inputs[0]);
    uint16_t *wt_ptr   = (uint16_t *)qhpi_tensor_raw_data(inputs[1]);
    uint8_t  *bias_ptr = (uint8_t  *)qhpi_tensor_raw_data(inputs[2]);
    uint16_t *out_ptr  = (uint16_t *)qhpi_tensor_raw_data(outputs[0]);

    /* ---- Step 1: HVX ReLU 预处理 ---- */
    /* 对 activation 的每个 F16 元素做 max(x, 0) */
    {
        HVX_Vector v_zero = Q6_V_vzero();
        HVX_Vector *vp = (HVX_Vector *)act_ptr;
        /* 1024 F16 = 16 vectors of 64 F16 each */
        for (int i = 0; i < 16; i++) {
            vp[i] = Q6_Vh_vmax_VhVh(vp[i], v_zero);
        }
    }

    /* ---- Step 2: HMX matmul ---- */
    Q6_bias_mxmem2_A(bias_ptr);
    Q6_mxclracc_hf();
    hmx_matmul_f16(
        (uintptr_t)act_ptr,
        (uintptr_t)wt_ptr
    );
    Q6_mxmem_AR_after_hf(out_ptr, 0);

    /* ---- Step 3: HVX ReLU 后处理 ---- */
    /* 对 matmul 输出再做一次 ReLU, 确保无负值 */
    /* (这里主要为了演示 HMX → HVX 的后处理流水线) */
    {
        HVX_Vector v_zero = Q6_V_vzero();
        HVX_Vector *vp = (HVX_Vector *)out_ptr;
        for (int i = 0; i < 16; i++) {
            vp[i] = Q6_Vh_vmax_VhVh(vp[i], v_zero);
        }
    }

    return QHPI_Success;
}

/* ============================================================
 * Tensor Signatures — all TCM
 * ============================================================ */
static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_Float16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
    {QHPI_Float16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
    {QHPI_Float16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
};
static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_Float16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
};

/* ============================================================
 * Kernel + Op registration
 * ============================================================ */
static QHPI_Kernel_v1 kernels[] = {
    {
        .function_name = THIS_PKG_NAME_STR "::hvx_hmx_mix_f16",
        .function = hvx_hmx_mix_kernel,
        .resources = QHPI_RESOURCE_HMX,
        .source_destructive = true,   /* 我们修改了 activation (in-place ReLU) */
        .multithreaded = false,
        .variable_inputs = false,
        .variable_outputs = false,
        .min_inputs = 3,
        .input_signature = sig_inputs,
        .min_outputs = 1,
        .output_signature = sig_outputs,
        .cost_function = nullptr,
        .sync_block_size = 0,
        .precomputed_data_size = 0,
        .do_precomputation_function = nullptr,
        .function_with_precomputed_data = nullptr,
        .predicate = nullptr,
    },
};

static QHPI_OpInfo_v1 ops[] = {
    {
        .name = THIS_PKG_NAME_STR "::HvxHmxMix",
        .num_kernels = 1,
        .kernels = kernels,
        .early_rewrite = nullptr,
        .shape_required = nullptr,
        .shape_legalized = nullptr,
        .tile_output = 0,
        .build_tile = nullptr,
        .late_rewrite = nullptr,
    },
};

void register_hvx_hmx_mix_ops() {
    qhpi_register_ops_v1(sizeof(ops) / sizeof(ops[0]), ops, THIS_PKG_NAME_STR);
}
