/*
 * Chapter 7: QNN VTCM 实验
 *
 * 实验 1+2: VTCM 大小配置 + QNN Profiling
 * 实验 3: 多图 VTCM 共享（两个独立 context）
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
#include "QnnProfile.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpGraph.h"
#include "HTP/QnnHtpProfile.h"

/* ---- F16 helpers ---- */
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

/* ---- QNN interface ---- */
typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn)(
    const QnnInterface_t*** providerList, uint32_t* numProviders);

static QNN_INTERFACE_VER_TYPE g_qnn;
static void* g_backendLib = NULL;

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

/* ---- Profiling ---- */
static const char* event_type_name(uint32_t type) {
    switch (type) {
        case 100: return "INIT";
        case 300: return "FINALIZE";
        case 400: return "EXECUTE";
        case 404: return "NODE";
        case 405: return "EXECUTE_QUEUE_WAIT";
        case 406: return "EXECUTE_PREPROCESS";
        case 407: return "EXECUTE_DEVICE";
        case 408: return "EXECUTE_POSTPROCESS";
        case 3003: return "HTP_ACCEL_TIME_CYCLE";
        case 3004: return "HTP_ACCEL_TIME_MICROSEC";
        case 3010: return "HTP_VTCM_ACQUIRE_TIME";
        case 3012: return "HTP_ACCEL_EXCL_WAIT_TIME_MICROSEC";
        case 8001: return "HTP_NUM_HVX_THREADS";
        default: return NULL;
    }
}

static const char* event_unit_name(uint32_t unit) {
    switch (unit) {
        case 1: return "us";
        case 2: return "bytes";
        case 3: return "cycles";
        case 4: return "count";
        case 6: return "";
        default: return "?";
    }
}

static void print_profile_events(int depth, const QnnProfile_EventId_t* events, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        QnnProfile_EventData_t data;
        memset(&data, 0, sizeof(data));
        Qnn_ErrorHandle_t err = g_qnn.profileGetEventData(events[i], &data);
        if (err != QNN_SUCCESS) continue;
        const char* name = event_type_name(data.type);
        const char* unit = event_unit_name(data.unit);
        for (int d = 0; d < depth; d++) printf("  ");
        if (name)
            printf("%-35s %8lu %s", name, (unsigned long)data.value, unit);
        else
            printf("event_type=%-6u                    %8lu %s", data.type, (unsigned long)data.value, unit);
        if (data.identifier) printf("  [%s]", data.identifier);
        printf("\n");
        const QnnProfile_EventId_t* subEvents = NULL;
        uint32_t numSub = 0;
        err = g_qnn.profileGetSubEvents(events[i], &subEvents, &numSub);
        if (err == QNN_SUCCESS && numSub > 0)
            print_profile_events(depth + 1, subEvents, numSub);
    }
}

static void dump_profile(Qnn_ProfileHandle_t profile, const char* label) {
    const QnnProfile_EventId_t* events = NULL;
    uint32_t numEvents = 0;
    Qnn_ErrorHandle_t err = g_qnn.profileGetEvents(profile, &events, &numEvents);
    if (err != QNN_SUCCESS || numEvents == 0) {
        printf("  [%s] No profiling events (err=%lu)\n", label, (unsigned long)err);
        return;
    }
    printf("  [%s] %u profiling events:\n", label, numEvents);
    print_profile_events(2, events, numEvents);
}

/* ---- Tensor helper ---- */
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

/* ---- 准备测试数据 ---- */
static void fill_test_data(uint16_t* act, uint16_t* wt, uint16_t* bias,
                            float act_val, float wt_val) {
    for (int i = 0; i < 1024; i++) {
        act[i] = f32_to_f16(act_val);
        wt[i] = f32_to_f16(wt_val);
    }
    uint8_t *bp = (uint8_t*)bias;
    memset(bp, 0, 128 * 2);
    for (int i = 0; i < 32; i++) bp[i * 4 + 1] = 0x3C; /* scale=1.0 */
}

static void setup_exec_tensors(Qnn_Tensor_t* eIn, Qnn_Tensor_t* eOut,
                                Qnn_Tensor_t actT, Qnn_Tensor_t wtT,
                                Qnn_Tensor_t biasT, Qnn_Tensor_t outT,
                                uint16_t* actData, uint16_t* wtData,
                                uint16_t* biasData, uint16_t* outData) {
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
}

/* ================================================================
 * 构建 op 链
 * ================================================================ */
static int build_chain_graph(Qnn_ContextHandle_t ctx, const char* name,
                             uint32_t vtcm_mb, int chain_len,
                             Qnn_GraphHandle_t* out_graph,
                             Qnn_Tensor_t* out_actT, Qnn_Tensor_t* out_wtT,
                             Qnn_Tensor_t* out_biasT, Qnn_Tensor_t* out_outT) {
    Qnn_ErrorHandle_t err;
    QnnHtpGraph_CustomConfig_t vtcmCfg;
    memset(&vtcmCfg, 0, sizeof(vtcmCfg));
    vtcmCfg.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE_IN_MB;
    vtcmCfg.vtcmSizeInMB = vtcm_mb;

    QnnGraph_Config_t graphCfg1;
    memset(&graphCfg1, 0, sizeof(graphCfg1));
    graphCfg1.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    graphCfg1.customConfig = &vtcmCfg;
    const QnnGraph_Config_t* cfgs[] = {&graphCfg1, NULL};

    err = g_qnn.graphCreate(ctx, name, cfgs, out_graph);
    if (err != QNN_SUCCESS) {
        printf("    graphCreate(%s) failed: %lu\n", name, (unsigned long)err);
        return -1;
    }

    uint32_t matDims[] = {1, 1, 32, 32};
    uint32_t biasDims[] = {1, 1, 1, 128};
    /* 每个 tensor/op 需要独立的 name buffer（QNN 持有指针） */
    char actName[64], wtName[64], biasName[64];

    snprintf(actName, sizeof(actName), "%s_act", name);
    *out_actT = make_tensor(actName, QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, 4, matDims);
    snprintf(wtName, sizeof(wtName), "%s_wt", name);
    *out_wtT = make_tensor(wtName, QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, 4, matDims);
    snprintf(biasName, sizeof(biasName), "%s_bias", name);
    *out_biasT = make_tensor(biasName, QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, 4, biasDims);
    g_qnn.tensorCreateGraphTensor(*out_graph, out_actT);
    g_qnn.tensorCreateGraphTensor(*out_graph, out_wtT);
    g_qnn.tensorCreateGraphTensor(*out_graph, out_biasT);

    /* 每个 op 和中间 tensor 需要独立的 name buffer */
    char midNames[16][64], opNames[16][64];
    Qnn_Tensor_t prev_out = *out_actT;
    for (int i = 0; i < chain_len && i < 16; i++) {
        bool is_last = (i == chain_len - 1);
        Qnn_Tensor_t cur_out;
        if (is_last) {
            snprintf(midNames[i], sizeof(midNames[i]), "%s_output", name);
            cur_out = make_tensor(midNames[i], QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, 4, matDims);
        } else {
            snprintf(midNames[i], sizeof(midNames[i]), "%s_mid%d", name, i);
            cur_out = make_tensor(midNames[i], QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 4, matDims);
        }
        g_qnn.tensorCreateGraphTensor(*out_graph, &cur_out);

        snprintf(opNames[i], sizeof(opNames[i]), "%s_op%d", name, i);
        Qnn_Tensor_t inputs[] = {prev_out, *out_wtT, *out_biasT};
        Qnn_OpConfig_t op;
        memset(&op, 0, sizeof(op));
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = opNames[i];
        op.v1.packageName = "HvxHmxMixPackage";
        op.v1.typeName = "HvxHmxMix";
        op.v1.numOfInputs = 3;
        op.v1.inputTensors = inputs;
        op.v1.numOfOutputs = 1;
        op.v1.outputTensors = &cur_out;
        err = g_qnn.graphAddNode(*out_graph, op);
        if (err != QNN_SUCCESS) {
            printf("    graphAddNode(%s) failed: %lu\n", opNames[i], (unsigned long)err);
            return -1;
        }
        prev_out = cur_out;
        if (is_last) *out_outT = cur_out;
    }
    return 0;
}

/* ================================================================
 * 构建独立 context + graph（用于 VTCM 共享实验）
 * ================================================================ */
static int build_independent_graph(Qnn_BackendHandle_t backend, const char* name,
                                   uint32_t vtcm_offset, uint32_t vtcm_size, uint32_t vtcm_total,
                                   Qnn_ContextHandle_t* out_ctx, Qnn_GraphHandle_t* out_graph,
                                   Qnn_Tensor_t* out_actT, Qnn_Tensor_t* out_wtT,
                                   Qnn_Tensor_t* out_biasT, Qnn_Tensor_t* out_outT) {
    Qnn_ErrorHandle_t err;
    const QnnContext_Config_t* ctxCfgs[] = {NULL};
    err = g_qnn.contextCreate(backend, NULL, ctxCfgs, out_ctx);
    if (err != QNN_SUCCESS) {
        printf("    contextCreate(%s) failed: %lu\n", name, (unsigned long)err);
        return -1;
    }

    QnnHtpGraph_CustomConfig_t vtcmCfg;
    memset(&vtcmCfg, 0, sizeof(vtcmCfg));
    vtcmCfg.option = QNN_HTP_GRAPH_CONFIG_OPTION_PARALLEL_GRAPH_EXECUTION_CONFIG;
    vtcmCfg.parallelGraphExecutionConfig.concurrency = 0;
    vtcmCfg.parallelGraphExecutionConfig.vtcmConfig.sizeInBytes = vtcm_size;
    vtcmCfg.parallelGraphExecutionConfig.vtcmConfig.offsetInBytes = vtcm_offset;
    vtcmCfg.parallelGraphExecutionConfig.vtcmConfig.sizeTotalInBytes = vtcm_total;

    QnnGraph_Config_t grCfg;
    memset(&grCfg, 0, sizeof(grCfg));
    grCfg.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    grCfg.customConfig = &vtcmCfg;
    const QnnGraph_Config_t* cfgs[] = {&grCfg, NULL};

    err = g_qnn.graphCreate(*out_ctx, name, cfgs, out_graph);
    if (err != QNN_SUCCESS) {
        printf("    graphCreate(%s) failed: %lu\n", name, (unsigned long)err);
        return -1;
    }

    uint32_t matDims[] = {1, 1, 32, 32};
    uint32_t biasDims[] = {1, 1, 1, 128};
    char actName[64], wtName[64], biasName[64], outName[64], opName[64];

    snprintf(actName, sizeof(actName), "%s_act", name);
    *out_actT = make_tensor(actName, QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, 4, matDims);
    snprintf(wtName, sizeof(wtName), "%s_wt", name);
    *out_wtT = make_tensor(wtName, QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, 4, matDims);
    snprintf(biasName, sizeof(biasName), "%s_bias", name);
    *out_biasT = make_tensor(biasName, QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, 4, biasDims);
    snprintf(outName, sizeof(outName), "%s_out", name);
    *out_outT = make_tensor(outName, QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, 4, matDims);

    g_qnn.tensorCreateGraphTensor(*out_graph, out_actT);
    g_qnn.tensorCreateGraphTensor(*out_graph, out_wtT);
    g_qnn.tensorCreateGraphTensor(*out_graph, out_biasT);
    g_qnn.tensorCreateGraphTensor(*out_graph, out_outT);

    snprintf(opName, sizeof(opName), "%s_op0", name);
    Qnn_Tensor_t ins[] = {*out_actT, *out_wtT, *out_biasT};
    Qnn_OpConfig_t op;
    memset(&op, 0, sizeof(op));
    op.version = QNN_OPCONFIG_VERSION_1;
    op.v1.name = opName;
    op.v1.packageName = "HvxHmxMixPackage";
    op.v1.typeName = "HvxHmxMix";
    op.v1.numOfInputs = 3;
    op.v1.inputTensors = ins;
    op.v1.numOfOutputs = 1;
    op.v1.outputTensors = out_outT;
    g_qnn.graphAddNode(*out_graph, op);

    err = g_qnn.graphFinalize(*out_graph, NULL, NULL);
    if (err != QNN_SUCCESS) {
        printf("    graphFinalize(%s) failed: %lu\n", name, (unsigned long)err);
        return -1;
    }
    return 0;
}

/* ================================================================
 * 实验 1+2
 * ================================================================ */
static int experiment_vtcm_size(Qnn_BackendHandle_t backend) {
    printf("========================================\n");
    printf("  实验 1+2: VTCM 大小 + Profiling\n");
    printf("========================================\n\n");

    struct { uint32_t vtcm_mb; int chain_len; const char* label; } configs[] = {
        {0, 1, "vtcm=MAX, chain=1"}, {0, 4, "vtcm=MAX, chain=4"}, {0, 8, "vtcm=MAX, chain=8"},
        {1, 1, "vtcm=1MB, chain=1"}, {1, 4, "vtcm=1MB, chain=4"}, {1, 8, "vtcm=1MB, chain=8"},
        {4, 1, "vtcm=4MB, chain=1"}, {4, 4, "vtcm=4MB, chain=4"}, {4, 8, "vtcm=4MB, chain=8"},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    for (int c = 0; c < num_configs; c++) {
        printf("--- %s ---\n", configs[c].label);
        Qnn_ProfileHandle_t profile = NULL;
        Qnn_ErrorHandle_t err = g_qnn.profileCreate(backend, QNN_PROFILE_LEVEL_DETAILED, &profile);
        if (err != QNN_SUCCESS) { profile = NULL; }

        Qnn_ContextHandle_t ctx = NULL;
        const QnnContext_Config_t* ctxCfgs[] = {NULL};
        err = g_qnn.contextCreate(backend, NULL, ctxCfgs, &ctx);
        if (err != QNN_SUCCESS) {
            if (profile) g_qnn.profileFree(profile);
            continue;
        }

        Qnn_GraphHandle_t graph = NULL;
        Qnn_Tensor_t actT, wtT, biasT, outT;
        char graphName[64];
        snprintf(graphName, sizeof(graphName), "chain_%u_%d", configs[c].vtcm_mb, configs[c].chain_len);
        if (build_chain_graph(ctx, graphName, configs[c].vtcm_mb, configs[c].chain_len,
                              &graph, &actT, &wtT, &biasT, &outT) != 0) {
            g_qnn.contextFree(ctx, NULL);
            if (profile) g_qnn.profileFree(profile);
            continue;
        }

        err = g_qnn.graphFinalize(graph, profile, NULL);
        if (err != QNN_SUCCESS) {
            g_qnn.contextFree(ctx, NULL);
            if (profile) g_qnn.profileFree(profile);
            continue;
        }
        if (profile) dump_profile(profile, "finalize");

        uint16_t *actData = (uint16_t*)malloc(1024*2);
        uint16_t *wtData = (uint16_t*)malloc(1024*2);
        uint16_t *biasData = (uint16_t*)calloc(128, 2);
        uint16_t *outData = (uint16_t*)calloc(1024, 2);
        fill_test_data(actData, wtData, biasData, 1.0f, 0.1f);

        Qnn_Tensor_t eIn[3], eOut[1];
        setup_exec_tensors(eIn, eOut, actT, wtT, biasT, outT, actData, wtData, biasData, outData);

        g_qnn.graphExecute(graph, eIn, 3, eOut, 1, NULL, NULL); /* warmup */

        struct timespec t0, t1;
        int N = 100;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < N; i++)
            g_qnn.graphExecute(graph, eIn, 3, eOut, 1, (i == N-1) ? profile : NULL, NULL);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double us = ((t1.tv_sec-t0.tv_sec)*1e6 + (t1.tv_nsec-t0.tv_nsec)/1e3);
        printf("  execute: %.0f us/iter (avg over %d)\n", us/N, N);
        printf("  out[0]=%.4f\n", f16_to_f32(outData[0]));
        if (profile) dump_profile(profile, "execute");

        free(actData); free(wtData); free(biasData); free(outData);
        g_qnn.contextFree(ctx, NULL);
        if (profile) g_qnn.profileFree(profile);
        printf("\n");
    }
    return 0;
}

/* ================================================================
 * 实验 3: 多图 VTCM 共享（两个独立 context）
 * ================================================================ */
static int experiment_vtcm_sharing(Qnn_BackendHandle_t backend) {
    printf("========================================\n");
    printf("  实验 3: 多图 VTCM 共享\n");
    printf("========================================\n\n");

    const uint32_t VTCM_TOTAL = 8*1024*1024, VTCM_HALF = 4*1024*1024;
    Qnn_ContextHandle_t ctxA = NULL, ctxB = NULL;
    Qnn_GraphHandle_t graphA = NULL, graphB = NULL;
    Qnn_Tensor_t aAct, aWt, aBias, aOut, bAct, bWt, bBias, bOut;

    printf("  创建两个独立的 context（模拟两个独立 AI 模型）\n\n");

    printf("  Context A: vtcm=[0, 4MB)\n");
    if (build_independent_graph(backend, "graph_a", 0, VTCM_HALF, VTCM_TOTAL,
                                &ctxA, &graphA, &aAct, &aWt, &aBias, &aOut) != 0) {
        printf("  NOTE: Parallel graph VTCM config may not be supported.\n");
        if (ctxA) g_qnn.contextFree(ctxA, NULL);
        return 0;
    }
    printf("    OK\n");

    printf("  Context B: vtcm=[4MB, 8MB)\n");
    if (build_independent_graph(backend, "graph_b", VTCM_HALF, VTCM_HALF, VTCM_TOTAL,
                                &ctxB, &graphB, &bAct, &bWt, &bBias, &bOut) != 0) {
        printf("  NOTE: Parallel graph VTCM config may not be supported.\n");
        g_qnn.contextFree(ctxA, NULL);
        if (ctxB) g_qnn.contextFree(ctxB, NULL);
        return 0;
    }
    printf("    OK\n");

    uint16_t *actData = (uint16_t*)malloc(1024*2);
    uint16_t *wtData = (uint16_t*)malloc(1024*2);
    uint16_t *biasData = (uint16_t*)calloc(128, 2);
    uint16_t *outA = (uint16_t*)calloc(1024, 2);
    uint16_t *outB = (uint16_t*)calloc(1024, 2);
    fill_test_data(actData, wtData, biasData, 1.0f, 1.0f);

    Qnn_Tensor_t eInA[3], eOutA[1], eInB[3], eOutB[1];
    setup_exec_tensors(eInA, eOutA, aAct, aWt, aBias, aOut, actData, wtData, biasData, outA);
    setup_exec_tensors(eInB, eOutB, bAct, bWt, bBias, bOut, actData, wtData, biasData, outB);

    g_qnn.graphExecute(graphA, eInA, 3, eOutA, 1, NULL, NULL);
    g_qnn.graphExecute(graphB, eInB, 3, eOutB, 1, NULL, NULL);

    struct timespec t0, t1;
    int N = 100;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < N; i++) {
        g_qnn.graphExecute(graphA, eInA, 3, eOutA, 1, NULL, NULL);
        g_qnn.graphExecute(graphB, eInB, 3, eOutB, 1, NULL, NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double us = ((t1.tv_sec-t0.tv_sec)*1e6 + (t1.tv_nsec-t0.tv_nsec)/1e3);
    printf("\n  Sequential (A+B): %.0f us/pair (avg over %d)\n", us/N, N);
    printf("  graph_a out[0]=%.1f  graph_b out[0]=%.1f\n", f16_to_f32(outA[0]), f16_to_f32(outB[0]));

    free(actData); free(wtData); free(biasData); free(outA); free(outB);
    g_qnn.contextFree(ctxA, NULL);
    g_qnn.contextFree(ctxB, NULL);
    return 0;
}

/* ================================================================ */
int main(int argc, char** argv) {
    printf("========================================\n");
    printf("  Chapter 7: QNN VTCM 实验\n");
    printf("========================================\n\n");

    const char* libPath = (argc > 1) ? argv[1] : "./libQnnHtp.so";
    printf("[Load] QNN backend: %s\n", libPath);
    if (load_backend(libPath) != 0) { printf("[FAIL] Cannot load backend\n"); return 1; }
    printf("[Load] OK\n\n");

    Qnn_ErrorHandle_t err;
    Qnn_BackendHandle_t backend = NULL;
    const QnnBackend_Config_t* beCfgs[] = {NULL};
    err = g_qnn.backendCreate(NULL, (const QnnBackend_Config_t**)beCfgs, &backend);
    if (err != QNN_SUCCESS) { printf("[FAIL] backendCreate=%lu\n", (unsigned long)err); return 1; }

    printf("[Register] Custom op packages...\n");
    err = g_qnn.backendRegisterOpPackage(backend, "./libHvxHmxMix_cpu.so", "HvxHmxMixInterfaceProvider", "CPU");
    if (err != QNN_SUCCESS) { printf("[FAIL] CPU reg=%lu\n", (unsigned long)err); return 1; }
    err = g_qnn.backendRegisterOpPackage(backend, "./libHvxHmxMix_htp.so", "HvxHmxMixInterfaceProvider", "HTP");
    if (err != QNN_SUCCESS) { printf("[FAIL] HTP reg=%lu\n", (unsigned long)err); return 1; }
    printf("[Register] OK\n\n");

    experiment_vtcm_size(backend);
    experiment_vtcm_sharing(backend);

    printf("========================================\n");
    printf("  All experiments complete.\n");
    printf("========================================\n");
    g_qnn.backendFree(backend);
    return 0;
}
