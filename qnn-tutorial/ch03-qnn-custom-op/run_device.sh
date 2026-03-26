#!/usr/bin/env bash
#
# run_device.sh — 推送到真机, 通过 QNN HTP 后端执行自定义算子
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
QNN_SDK="$ROOT_DIR/tools/qnn-sdk"
DEVICE_DIR="/data/local/tmp/ch03"

# 检查编译产物
for f in \
    "$SCRIPT_DIR/build/aarch64/qnn_hvx_hmx_test" \
    "$SCRIPT_DIR/build/aarch64/libHvxHmxMix_cpu.so" \
    "$SCRIPT_DIR/build/hexagon-v75/libHvxHmxMix_htp.so"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found. Run build.sh first."
        exit 1
    fi
done

# 检查 adb
if ! adb devices | grep -q "device$"; then
    echo "ERROR: No device connected."
    exit 1
fi

echo "=== Pushing files to device ==="
adb shell "mkdir -p $DEVICE_DIR"

# 推送测试程序和自定义 op 库
adb push "$SCRIPT_DIR/build/aarch64/qnn_hvx_hmx_test" "$DEVICE_DIR/"
adb push "$SCRIPT_DIR/build/aarch64/libHvxHmxMix_cpu.so" "$DEVICE_DIR/"
adb push "$SCRIPT_DIR/build/hexagon-v75/libHvxHmxMix_htp.so" "$DEVICE_DIR/"

# 推送 QNN 运行时库
adb push "$QNN_SDK/lib/aarch64-android/libQnnHtp.so" "$DEVICE_DIR/"
adb push "$QNN_SDK/lib/aarch64-android/libQnnHtpPrepare.so" "$DEVICE_DIR/"
adb push "$QNN_SDK/lib/aarch64-android/libQnnHtpV75Stub.so" "$DEVICE_DIR/"
adb push "$QNN_SDK/lib/aarch64-android/libQnnSystem.so" "$DEVICE_DIR/"

# 推送 HTP skel (unsigned PD)
adb push "$QNN_SDK/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so" "$DEVICE_DIR/"

echo ""
echo "=== Running QNN Custom Op Test ==="
echo ""
adb shell "cd $DEVICE_DIR && \
    export LD_LIBRARY_PATH=$DEVICE_DIR:\$LD_LIBRARY_PATH && \
    export ADSP_LIBRARY_PATH=\"$DEVICE_DIR;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp\" && \
    chmod +x qnn_hvx_hmx_test && \
    ./qnn_hvx_hmx_test" 2>&1
