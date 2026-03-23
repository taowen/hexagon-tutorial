/*
 * QHPI HvxHmxMix Op Package Interface
 *
 * 标准 QnnOpPackage interface + QHPI entry point (qhpi_init).
 */

#include "HTP/QnnHtpCommon.h"
#include "HTP/core/qhpi.h"
#include "QnnOpPackage.h"
#include <array>
#include <string>

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

extern void register_hvx_hmx_mix_ops();

static constexpr auto sg_packageName = THIS_PKG_NAME_STR;
static constexpr auto sg_opName = "HvxHmxMix";
static std::array<const char *, 1> sg_opNames{{sg_opName}};

static Qnn_ApiVersion_t sg_sdkApiVersion = QNN_HTP_API_VERSION_INIT;
static Qnn_Version_t sg_opsetVersion = {1, 0, 0};
static QnnOpPackage_Info_t sg_packageInfo = {
    sg_packageName,
    sg_opNames.data(),
    nullptr,
    sg_opNames.size(),
    nullptr,
    0,
    "hvx_hmx_mix_qhpi",
    &sg_sdkApiVersion,
    nullptr,
    &sg_opsetVersion,
    {0}};

static QnnOpPackage_GlobalInfrastructure_t sg_globalInfra = nullptr;
static bool sg_initialized = false;
static QnnLog_Callback_t sg_logCallback = nullptr;
static QnnLog_Level_t sg_maxLogLevel = (QnnLog_Level_t)0;
static bool sg_logInitialized = false;

Qnn_ErrorHandle_t mixInit(QnnOpPackage_GlobalInfrastructure_t infra) {
    if (sg_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
    sg_globalInfra = infra;
    sg_initialized = true;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t mixGetInfo(const QnnOpPackage_Info_t **info) {
    if (!sg_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    if (!info) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    *info = &sg_packageInfo;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t mixValidateOpConfig(Qnn_OpConfig_t opConfig) {
    if (std::string(sg_packageName) != opConfig.v1.packageName)
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if (std::string(opConfig.v1.typeName) == sg_opName) {
        if (opConfig.v1.numOfInputs != 3 || opConfig.v1.numOfOutputs != 1)
            return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    } else {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t mixLogInit(QnnLog_Callback_t cb, QnnLog_Level_t level) {
    if (!cb) return QNN_LOG_ERROR_INVALID_ARGUMENT;
    sg_logCallback = cb;
    sg_maxLogLevel = level;
    sg_logInitialized = true;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t mixLogSetLevel(QnnLog_Level_t level) {
    sg_maxLogLevel = level;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t mixLogTerminate() {
    sg_logCallback = nullptr;
    sg_logInitialized = false;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t mixCreateOpImpl(
    QnnOpPackage_GraphInfrastructure_t, QnnOpPackage_Node_t, QnnOpPackage_OpImpl_t*) {
    return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
}

Qnn_ErrorHandle_t mixFreeOpImpl(QnnOpPackage_OpImpl_t) {
    return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
}

Qnn_ErrorHandle_t mixTerminate() {
    if (!sg_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    sg_globalInfra = nullptr;
    sg_initialized = false;
    return QNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

Qnn_ErrorHandle_t HvxHmxMixInterfaceProvider(QnnOpPackage_Interface_t *interface) {
    if (!interface) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    interface->interfaceVersion = {1, 4, 0};
    interface->v1_4.init = mixInit;
    interface->v1_4.terminate = mixTerminate;
    interface->v1_4.getInfo = mixGetInfo;
    interface->v1_4.validateOpConfig = mixValidateOpConfig;
    interface->v1_4.createOpImpl = mixCreateOpImpl;
    interface->v1_4.freeOpImpl = mixFreeOpImpl;
    interface->v1_4.logInitialize = mixLogInit;
    interface->v1_4.logSetLevel = mixLogSetLevel;
    interface->v1_4.logTerminate = mixLogTerminate;
    return QNN_SUCCESS;
}

const char *qhpi_init() {
    register_hvx_hmx_mix_ops();
    return THIS_PKG_NAME_STR;
}

#ifdef __cplusplus
}
#endif
