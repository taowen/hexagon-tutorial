#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
HEXKL="$ROOT_DIR/tools/hexkl-addon"
DEVICE_DIR="/data/local/tmp/ch09"

DEMO="$BUILD_DIR/libdemo_native_kv.so"
RUN_MAIN="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon"
RUN_MAIN_SKEL="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/hexagon_toolv87_v75/librun_main_on_hexagon_skel.so"
HEXKL_SKEL="$HEXKL/lib/hexagon_toolv19_v75/libhexkl_skel.so"

for f in "$DEMO" "$RUN_MAIN" "$RUN_MAIN_SKEL" "$HEXKL_SKEL"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found."
        exit 1
    fi
done

if ! adb devices | grep -q "device$"; then
    echo "ERROR: No device connected."
    exit 1
fi

echo "=== Pushing files to device ==="
adb shell "mkdir -p $DEVICE_DIR"
adb push "$DEMO"           "$DEVICE_DIR/"
adb push "$RUN_MAIN"       "$DEVICE_DIR/"
adb push "$RUN_MAIN_SKEL"  "$DEVICE_DIR/"
adb push "$HEXKL_SKEL"     "$DEVICE_DIR/"
adb shell "chmod +x $DEVICE_DIR/run_main_on_hexagon"
adb shell "echo '0x1f' > $DEVICE_DIR/run_main_on_hexagon.farf"

echo ""
echo "=== NativeKV vs SmartMask KV Cache Benchmark ==="
echo ""
adb logcat -c
timeout 60 adb shell "cd $DEVICE_DIR && \
    DSP_LIBRARY_PATH=$DEVICE_DIR \
    LD_LIBRARY_PATH=$DEVICE_DIR \
    ADSP_LIBRARY_PATH=$DEVICE_DIR \
    ./run_main_on_hexagon 3 libdemo_native_kv.so" 2>&1 || echo "(timeout or failed)"

sleep 1
adb logcat -d | grep "\[KV\]" | sed 's/.*\[DU\]: //' | tail -50
