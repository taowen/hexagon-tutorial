#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD="$SCRIPT_DIR/build"
DEVICE_DIR="/data/local/tmp/ch10"

for f in "$BUILD/vlut16_test" "$BUILD/libbitnet_test_skel.so"; do
    [ -f "$f" ] || { echo "Missing: $f -- run build.sh first"; exit 1; }
done

adb devices | grep -q "device$" || { echo "No device connected"; exit 1; }

echo "Pushing to device..."
adb shell "mkdir -p $DEVICE_DIR"
adb push "$BUILD/vlut16_test" "$DEVICE_DIR/"
adb push "$BUILD/libbitnet_test_skel.so" "$DEVICE_DIR/"
adb shell "chmod +x $DEVICE_DIR/vlut16_test"

echo "Running VLUT16 test..."
adb shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=$DEVICE_DIR ADSP_LIBRARY_PATH=$DEVICE_DIR ./vlut16_test"

echo ""
echo "DSP logs (FARF output):"
adb logcat -d -s adsprpc -s HAP_debug | tail -50
