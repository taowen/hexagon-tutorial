#!/usr/bin/env bash
#
# run_device.sh -- Push and run ch08 demos on the device
#
# Prerequisites:
#   1. Phone connected via USB, adb available
#   2. build.sh has been run successfully
#   3. HexKL addon installed (see install_hexkl.sh)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
HEXKL="$ROOT_DIR/tools/hexkl-addon"
DEVICE_DIR="/data/local/tmp/ch08"

# ---- Local file paths ----
DEMO_HVX_HMX="$BUILD_DIR/libdemo_hvx_hmx.so"
DEMO_MICRO="$BUILD_DIR/libdemo_micro.so"
DEMO_SDKL="$BUILD_DIR/demo_sdkl"
RUN_MAIN="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon"
RUN_MAIN_SKEL="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/hexagon_toolv87_v75/librun_main_on_hexagon_skel.so"
HEXKL_SKEL="$HEXKL/lib/hexagon_toolv19_v75/libhexkl_skel.so"
SDKL_LIB="$HEXKL/lib/armv8_android26/libsdkl.so"

# ---- Check build artifacts ----
for f in "$DEMO_HVX_HMX" "$DEMO_MICRO" "$DEMO_SDKL"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found. Run build.sh first."
        exit 1
    fi
done

for f in "$RUN_MAIN" "$RUN_MAIN_SKEL" "$HEXKL_SKEL" "$SDKL_LIB"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found. Check SDK / HexKL installation."
        exit 1
    fi
done

# ---- Check adb connection ----
if ! adb devices | grep -q "device$"; then
    echo "ERROR: No device connected. Check USB and run 'adb devices'."
    exit 1
fi

# ---- Push files to device ----
echo "=== Pushing files to device ==="
adb shell "mkdir -p $DEVICE_DIR"
adb push "$DEMO_HVX_HMX"  "$DEVICE_DIR/"
adb push "$DEMO_MICRO"    "$DEVICE_DIR/"
adb push "$DEMO_SDKL"     "$DEVICE_DIR/"
adb push "$RUN_MAIN"      "$DEVICE_DIR/"
adb push "$RUN_MAIN_SKEL" "$DEVICE_DIR/"
adb push "$HEXKL_SKEL"    "$DEVICE_DIR/"
adb push "$SDKL_LIB"      "$DEVICE_DIR/"
adb shell "chmod +x $DEVICE_DIR/run_main_on_hexagon $DEVICE_DIR/demo_sdkl"

# ---- Enable FARF logging (required for FARF output to appear in logcat) ----
adb shell "echo '0x1f' > $DEVICE_DIR/run_main_on_hexagon.farf"

# ---- Experiment 1: HVX + HMX combined (DSP-side) ----
echo ""
echo "=== Experiment 1: HVX + HMX Combined Pipeline (DSP-side) ==="
echo "    Simulates LLM inference: HVX dequant → HMX matmul → HVX bias add"
echo ""
adb logcat -c
timeout 30 adb shell "cd $DEVICE_DIR && \
    DSP_LIBRARY_PATH=$DEVICE_DIR \
    LD_LIBRARY_PATH=$DEVICE_DIR \
    ADSP_LIBRARY_PATH=$DEVICE_DIR \
    ./run_main_on_hexagon 3 libdemo_hvx_hmx.so" 2>&1 || echo "(timeout or failed)"

sleep 1
adb logcat -d | grep "\[HVX+HMX\]" | sed 's/.*\[DU\]: //' | tail -40

# ---- Experiment 2: SDKL matmul (ARM-side) ----
echo ""
echo "=== Experiment 2: SDKL matmul (ARM-side) ==="
echo ""
timeout 60 adb shell "cd $DEVICE_DIR && \
    LD_LIBRARY_PATH=$DEVICE_DIR \
    ADSP_LIBRARY_PATH=$DEVICE_DIR \
    ./demo_sdkl" 2>&1 || echo "(timeout or failed)"

# ---- Experiment 3: Micro API (DSP-side) ----
echo ""
echo "=== Experiment 3: Micro API (DSP-side, run_main_on_hexagon) ==="
echo ""
adb logcat -c
timeout 30 adb shell "cd $DEVICE_DIR && \
    DSP_LIBRARY_PATH=$DEVICE_DIR \
    LD_LIBRARY_PATH=$DEVICE_DIR \
    ADSP_LIBRARY_PATH=$DEVICE_DIR \
    ./run_main_on_hexagon 3 libdemo_micro.so" 2>&1 || echo "(timeout or failed)"

sleep 1
adb logcat -d | grep "\[MICRO\]" | sed 's/.*\[DU\]: //' | tail -30
