#!/bin/bash
# 部署 Genie 运行时和模型到手机
# 用法: ./deploy.sh [1b|3b]

set -e

MODEL=${1:-1b}
QNN_SDK=$(cd "$(dirname "$0")/../tools/qnn-sdk" && pwd)
DEST=/data/local/tmp/genie_${MODEL}

echo "=== 部署 Genie ${MODEL} 模型到 ${DEST} ==="

# 创建目录
adb shell "mkdir -p ${DEST}/models"

# 推送 Genie 运行时库 (QAIRT SDK v2.44.0)
echo "--- 推送运行时库 ---"
adb push ${QNN_SDK}/bin/aarch64-android/genie-t2t-run ${DEST}/
adb push ${QNN_SDK}/lib/aarch64-android/libGenie.so ${DEST}/
adb push ${QNN_SDK}/lib/aarch64-android/libQnnHtp.so ${DEST}/
adb push ${QNN_SDK}/lib/aarch64-android/libQnnHtpPrepare.so ${DEST}/
adb push ${QNN_SDK}/lib/aarch64-android/libQnnHtpV79Stub.so ${DEST}/
adb push ${QNN_SDK}/lib/aarch64-android/libQnnSystem.so ${DEST}/
adb push ${QNN_SDK}/lib/aarch64-android/libQnnHtpNetRunExtensions.so ${DEST}/
adb push ${QNN_SDK}/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so ${DEST}/

# 推送模型文件
BUNDLE_DIR=$(cd "$(dirname "$0")/genie_bundle_${MODEL}" && pwd)
echo "--- 推送模型文件 (${BUNDLE_DIR}) ---"

if [ "${MODEL}" = "1b" ]; then
    adb push ${BUNDLE_DIR}/models/ ${DEST}/models/
    adb push ${BUNDLE_DIR}/tokenizer.json ${DEST}/
    adb push ${BUNDLE_DIR}/htp-model-config-llama32-1b-gqa.json ${DEST}/
    adb push ${BUNDLE_DIR}/htp_backend_ext_config.json ${DEST}/
elif [ "${MODEL}" = "3b" ]; then
    adb push ${BUNDLE_DIR}/*.bin ${DEST}/
    adb push ${BUNDLE_DIR}/*.json ${DEST}/
else
    echo "未知模型: ${MODEL}。用法: ./deploy.sh [1b|3b]"
    exit 1
fi

echo "=== 部署完成 ==="
echo "运行: adb shell \"cd ${DEST} && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./genie-t2t-run -c <config>.json -p 'Hello'\""
