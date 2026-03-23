/*
 * Chapter 4: QNN Custom Op on Simulator — HVX ReLU → HMX matmul → HVX scale
 *
 * 在 x86 QEMU 模拟器上运行 QNN 图, 包含 HvxHmxMix 自定义算子.
 * 无需真机, 通过 libQnnHtp.so (x86) + libQnnHtpQemu.so 实现 Hexagon 仿真.
 *
 * Test 1: act=1.0, wt=1.0 → ReLU(1.0)=1.0 → matmul=32.0 → ReLU=32.0
 * Test 2: act=-1.0, wt=1.0 → ReLU(-1.0)=0.0 → matmul=0.0 → ReLU=0.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <dlfcn.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "QnnInterface.h"
#include "QnnTypes.h"
#include "QnnCommon.h"
#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnLog.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpGraph.h"

/* F16 helpers */
static uint16_t f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint16_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x007FFFFF;
    if (exp > 15) return sign | 0x7C00;
    if (exp < -14) return sign;
    return sign | (uint16_t)((exp + 15) << 10) | (uint16_t)(mant >> 13);
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    if (exp == 0) {
        if (mant == 0) { float f; uint32_t r = sign; memcpy(&f, &r, 4); return f; }
        while (!(mant & 0x0400)) { mant <<= 1; exp--; }
        exp++; mant &= ~0x0400;
    } else if (exp == 31) {
        uint32_t r = sign | 0x7F800000 | (mant << 13);
        float f; memcpy(&f, &r, 4); return f;
    }
    uint32_t result = sign | ((exp + 112) << 23) | (mant << 13);
    float f; memcpy(&f, &result, 4); return f;
}

/* QNN interface */
typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn)(
    const QnnInterface_t*** providerList, uint32_t* numProviders);

static void* g_backendLib = NULL;
static QNN_INTERFACE_VER_TYPE g_qnn;

static int load_backend(const char* libPath) {
    g_backendLib = dlopen(libPath, RTLD_NOW | RTLD_LOCAL);
    if (!g_backendLib) { printf("ERROR: dlopen: %s\n", dlerror()); return -1; }
    QnnInterfaceGetProvidersFn fn =
        (QnnInterfaceGetProvidersFn)dlsym(g_backendLib, "QnnInterface_getProviders");
    if (!fn) return -1;
    const QnnInterface_t** list = NULL;
    uint32_t n = 0;
    if (fn(&list, &n) != QNN_SUCCESS || n == 0) return -1;
    g_qnn = list[0]->QNN_INTERFACE_VER_NAME;
    return 0;
}

static Qnn_Tensor_t make_tensor(const char* name, Qnn_TensorType_t type,
                                 Qnn_DataType_t dt, uint32_t rank, uint32_t* dims) {
    Qnn_Tensor_t t;
    memset(&t, 0, sizeof(t));
    t.version = QNN_TENSOR_VERSION_1;
    t.v1.name = name;
    t.v1.type = type;
    t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t.v1.dataType = dt;
    t.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    t.v1.rank = rank;
    t.v1.dimensions = dims;
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    return t;
}

static int run_test(Qnn_GraphHandle_t graph,
                    Qnn_Tensor_t actT, Qnn_Tensor_t wtT, Qnn_Tensor_t biasT, Qnn_Tensor_t outT,
                    uint16_t *actData, uint16_t *wtData, uint16_t *biasData, uint16_t *outData,
                    const char *test_name, float expected, float tolerance) {
    Qnn_Tensor_t eIn[3], eOut[1];
    memcpy(&eIn[0], &actT, sizeof(Qnn_Tensor_t));
    eIn[0].v1.clientBuf.data = actData;
    eIn[0].v1.clientBuf.dataSize = 1024 * 2;
    memcpy(&eIn[1], &wtT, sizeof(Qnn_Tensor_t));
    eIn[1].v1.clientBuf.data = wtData;
    eIn[1].v1.clientBuf.dataSize = 1024 * 2;
    memcpy(&eIn[2], &biasT, sizeof(Qnn_Tensor_t));
    eIn[2].v1.clientBuf.data = biasData;
    eIn[2].v1.clientBuf.dataSize = 128 * 2;
    memcpy(&eOut[0], &outT, sizeof(Qnn_Tensor_t));
    eOut[0].v1.clientBuf.data = outData;
    eOut[0].v1.clientBuf.dataSize = 1024 * 2;

    memset(outData, 0, 1024 * 2);
    Qnn_ErrorHandle_t err = g_qnn.graphExecute(graph, eIn, 3, eOut, 1, NULL, NULL);
    if (err != QNN_SUCCESS) {
        printf("  [FAIL] %s: graphExecute error %lu\n", test_name, (unsigned long)err);
        return 0;
    }

    /* Verify */
    float maxErr = 0;
    int nonzero = 0;
    for (int i = 0; i < 1024; i++) {
        float v = f16_to_f32(outData[i]);
        if (v != 0.0f) nonzero++;
        float e = fabsf(v - expected);
        if (e > maxErr) maxErr = e;
    }

    printf("  out[0..3]: 0x%04X(%.1f) 0x%04X(%.1f) 0x%04X(%.1f) 0x%04X(%.1f)\n",
           outData[0], f16_to_f32(outData[0]),
           outData[1], f16_to_f32(outData[1]),
           outData[2], f16_to_f32(outData[2]),
           outData[3], f16_to_f32(outData[3]));
    printf("  expected=%.1f  maxErr=%.2f  nonzero=%d/1024\n", expected, maxErr, nonzero);

    int pass = (maxErr < tolerance);
    printf("  [%s] %s\n", pass ? "PASS" : "FAIL", test_name);
    return pass;
}

int main(int argc, char** argv) {
    printf("========================================\n");
    printf("  Chapter 4: QNN Custom Op on Simulator\n");
    printf("========================================\n\n");

    const char* libPath = (argc > 1) ? argv[1] : "./libQnnHtp.so";
    printf("[Load] QNN backend: %s\n", libPath);
    if (load_backend(libPath) != 0) return 1;
    printf("[Load] OK\n\n");

    Qnn_ErrorHandle_t err;
    Qnn_BackendHandle_t backend = NULL;
    Qnn_ContextHandle_t context = NULL;
    Qnn_GraphHandle_t graph = NULL;
    uint16_t *actData = NULL, *wtData = NULL, *biasData = NULL, *outData = NULL;
    int ret = 1;
    int pass_count = 0;

    /* Backend */
    const QnnBackend_Config_t* beCfgs[] = {NULL};
    err = g_qnn.backendCreate(NULL, (const QnnBackend_Config_t**)beCfgs, &backend);
    if (err != QNN_SUCCESS) { printf("[FAIL] backendCreate=%lu\n", (unsigned long)err); goto done; }

    /* Register custom op package.
     * On x86 simulator, we register a single x86-compiled .so with NULL target.
     * The x86 HTP backend uses QEMU for hexagon execution internally.
     * On real device (ch03), we register CPU and HTP .so separately. */
    printf("[Register] Custom op packages...\n");
    err = g_qnn.backendRegisterOpPackage(backend,
        "./libHvxHmxMix_cpu.so", "HvxHmxMixInterfaceProvider", NULL);
    if (err != QNN_SUCCESS) { printf("[FAIL] reg=%lu\n", (unsigned long)err); goto done; }
    printf("[Register] OK\n\n");

    /* Context */
    {
        const QnnContext_Config_t* ctxCfgs[] = {NULL};
        err = g_qnn.contextCreate(backend, NULL, (const QnnContext_Config_t**)ctxCfgs, &context);
        if (err != QNN_SUCCESS) { printf("[FAIL] contextCreate=%lu\n", (unsigned long)err); goto done; }
    }

    /* Build graph */
    printf("[Graph] Building...\n");
    {
        const QnnGraph_Config_t* grCfgs[] = {NULL};
        err = g_qnn.graphCreate(context, "hvx_hmx_mix_graph",
            (const QnnGraph_Config_t**)grCfgs, &graph);
        if (err != QNN_SUCCESS) { printf("[FAIL] graphCreate=%lu\n", (unsigned long)err); goto done; }

        uint32_t matDims[] = {1, 1, 32, 32};
        uint32_t biasDims[] = {1, 1, 1, 128};

        Qnn_Tensor_t actT = make_tensor("activation", QNN_TENSOR_TYPE_APP_WRITE,
                                         QNN_DATATYPE_FLOAT_16, 4, matDims);
        Qnn_Tensor_t wtT  = make_tensor("weight", QNN_TENSOR_TYPE_APP_WRITE,
                                         QNN_DATATYPE_FLOAT_16, 4, matDims);
        Qnn_Tensor_t biasT = make_tensor("bias_cfg", QNN_TENSOR_TYPE_APP_WRITE,
                                          QNN_DATATYPE_FLOAT_16, 4, biasDims);
        Qnn_Tensor_t outT = make_tensor("output", QNN_TENSOR_TYPE_APP_READ,
                                          QNN_DATATYPE_FLOAT_16, 4, matDims);

        g_qnn.tensorCreateGraphTensor(graph, &actT);
        g_qnn.tensorCreateGraphTensor(graph, &wtT);
        g_qnn.tensorCreateGraphTensor(graph, &biasT);
        g_qnn.tensorCreateGraphTensor(graph, &outT);

        Qnn_Tensor_t inputTensors[] = {actT, wtT, biasT};
        Qnn_OpConfig_t op;
        memset(&op, 0, sizeof(op));
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = "hvx_hmx_mix_0";
        op.v1.packageName = "HvxHmxMixPackage";
        op.v1.typeName = "HvxHmxMix";
        op.v1.numOfParams = 0;
        op.v1.params = NULL;
        op.v1.numOfInputs = 3;
        op.v1.inputTensors = inputTensors;
        op.v1.numOfOutputs = 1;
        op.v1.outputTensors = &outT;

        err = g_qnn.graphAddNode(graph, op);
        if (err != QNN_SUCCESS) { printf("[FAIL] graphAddNode=%lu\n", (unsigned long)err); goto done; }

        err = g_qnn.graphFinalize(graph, NULL, NULL);
        if (err != QNN_SUCCESS) { printf("[FAIL] graphFinalize=%lu\n", (unsigned long)err); goto done; }
        printf("[Graph] OK\n\n");

        /* Allocate data buffers */
        actData  = (uint16_t*)malloc(1024 * 2);
        wtData   = (uint16_t*)malloc(1024 * 2);
        biasData = (uint16_t*)calloc(128, 2);
        outData  = (uint16_t*)calloc(1024, 2);

        /* Bias config: scale=1.0 */
        {
            uint8_t *bp = (uint8_t *)biasData;
            for (int i = 0; i < 32; i++)
                bp[i * 4 + 1] = 0x3C;  /* F16 1.0 = 0x3C00 */
        }

        /* ---- Test 1: act=1.0, wt=1.0 ---- */
        printf("-- Test 1: HVX ReLU(1.0) -> HMX matmul -> HVX ReLU --\n");
        printf("  act=1.0, wt=1.0\n");
        printf("  ReLU(1.0)=1.0, matmul=1*1*32~=32, ReLU(32)=32\n");
        for (int i = 0; i < 1024; i++) {
            actData[i] = f32_to_f16(1.0f);
            wtData[i]  = f32_to_f16(1.0f);
        }
        pass_count += run_test(graph, actT, wtT, biasT, outT,
                               actData, wtData, biasData, outData,
                               "ReLU(+) -> matmul -> ReLU", 32.0f, 2.0f);

        /* ---- Test 2: act=-1.0, wt=1.0 ---- */
        printf("\n-- Test 2: HVX ReLU(-1.0) -> HMX matmul -> HVX ReLU --\n");
        printf("  act=-1.0, wt=1.0\n");
        printf("  ReLU(-1.0)=0.0, matmul=0*1*32=0, ReLU(0)=0\n");
        for (int i = 0; i < 1024; i++) {
            actData[i] = f32_to_f16(-1.0f);
            wtData[i]  = f32_to_f16(1.0f);
        }
        pass_count += run_test(graph, actT, wtT, biasT, outT,
                               actData, wtData, biasData, outData,
                               "ReLU(-) -> matmul -> ReLU", 0.0f, 0.5f);

        printf("\n========================================\n");
        printf("  Results: %d PASS / %d FAIL\n", pass_count, 2 - pass_count);
        printf("========================================\n");
        if (pass_count == 2) ret = 0;
    }

done:
    free(actData); free(wtData); free(biasData); free(outData);
    if (context) g_qnn.contextFree(context, NULL);
    if (backend) g_qnn.backendFree(backend);
    return ret;
}
