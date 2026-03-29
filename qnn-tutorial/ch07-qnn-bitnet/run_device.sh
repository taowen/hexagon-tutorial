#!/bin/bash
# Push files to device and run BitNet inference via qnn_llama_runner
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ET_DIR="$SCRIPT_DIR/../../tools/executorch"
QNN_SDK_ROOT=/home/taowen/qnn_sdk/qairt/2.44.0.260225
MODEL_DIR=~/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T/snapshots/04c3b9ad9361b824064a1f25ea60a8be9599b127
DEVICE_DIR=/data/local/tmp/llama

PROMPT="${1:-Hello}"
SEQ_LEN="${2:-128}"

echo "=== Pushing files to device ==="
adb shell "mkdir -p $DEVICE_DIR"

# Runner binary (rebuilt with -DSUPPORT_REGEX_LOOKAHEAD=ON for HF tokenizer)
adb push "$ET_DIR/examples/qualcomm/oss_scripts/llama/qnn_llama_runner" "$DEVICE_DIR/"
adb shell "chmod +x $DEVICE_DIR/qnn_llama_runner"

# .pte model (exported with --ptq 16a4w --use_tman)
adb push "$SCRIPT_DIR/artifacts/kv_llama_qnn.pte" "$DEVICE_DIR/"

# Tokenizer
adb push "$MODEL_DIR/tokenizer.json" "$DEVICE_DIR/"

# TMAN Op Package (must be named libQnnTMANOpPackage.so, hardcoded in QnnBackendCommon.cpp:77)
adb push "$SCRIPT_DIR/build/hexagon-v75/libTMANOpPackage_htp.so" "$DEVICE_DIR/libQnnTMANOpPackage.so"

# QNN runtime libraries (aarch64)
adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so" "$DEVICE_DIR/"
adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so" "$DEVICE_DIR/"
adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so" "$DEVICE_DIR/"
adb push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV75Stub.so" "$DEVICE_DIR/"

# Hexagon skel (DSP side)
adb push "$QNN_SDK_ROOT/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so" "$DEVICE_DIR/"

# executorch QNN backend
adb push "$ET_DIR/build-android/lib/libqnn_executorch_backend.so" "$DEVICE_DIR/"

echo ""
echo "=== Running inference ==="
# Notes:
#   -eval_mode 0: model was exported in kv-only mode (no prefill_forward method)
#   -kv_updater SmartMask: required because gflags DEFINE_string has swapped default/description
#   --prompt: double-dash required for CollectPrompts() to parse correctly
adb shell "cd $DEVICE_DIR && \
    LD_LIBRARY_PATH=$DEVICE_DIR \
    ADSP_LIBRARY_PATH='$DEVICE_DIR;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' \
    ./qnn_llama_runner \
    --model_path kv_llama_qnn.pte \
    --tokenizer_path tokenizer.json \
    --prompt '$PROMPT' \
    --seq_len $SEQ_LEN \
    -eval_mode 0 \
    -kv_updater SmartMask"
