#!/usr/bin/env bash
#
# run_device.sh -- Push and run ch05-hmx experiments on the device
#
# Prerequisites:
#   1. Phone connected via USB, adb available
#   2. build.sh has been run successfully
#   3. HexKL addon installed (see install_hexkl.sh)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
HEXKL="$ROOT_DIR/tools/hexkl-addon"
DEVICE_DIR="/data/local/tmp/ch05"

# ---- Local file paths ----
EXP1_TILE_BASICS="$BUILD_DIR/libexp1_tile_basics.so"
EXP2_WEIGHT_LAYOUT="$BUILD_DIR/libexp2_weight_layout.so"
EXP3_STREAMING="$BUILD_DIR/libexp3_streaming.so"
EXP4_PIPELINE="$BUILD_DIR/libexp4_pipeline.so"
RUN_MAIN="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon"
RUN_MAIN_SKEL="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/hexagon_toolv87_v75/librun_main_on_hexagon_skel.so"
HEXKL_SKEL="$HEXKL/lib/hexagon_toolv19_v75/libhexkl_skel.so"

# ---- Check build artifacts ----
for f in "$EXP1_TILE_BASICS" "$EXP2_WEIGHT_LAYOUT" "$EXP3_STREAMING" "$EXP4_PIPELINE"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found. Run build.sh first."
        exit 1
    fi
done

for f in "$RUN_MAIN" "$RUN_MAIN_SKEL" "$HEXKL_SKEL"; do
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
adb push "$EXP1_TILE_BASICS"  "$DEVICE_DIR/"
adb push "$EXP2_WEIGHT_LAYOUT" "$DEVICE_DIR/"
adb push "$EXP3_STREAMING"    "$DEVICE_DIR/"
adb push "$EXP4_PIPELINE"     "$DEVICE_DIR/"
adb push "$RUN_MAIN"          "$DEVICE_DIR/"
adb push "$RUN_MAIN_SKEL"     "$DEVICE_DIR/"
adb push "$HEXKL_SKEL"        "$DEVICE_DIR/"
adb shell "chmod +x $DEVICE_DIR/run_main_on_hexagon"

# ---- Enable FARF logging (required for FARF output to appear in logcat) ----
adb shell "echo '0x1f' > $DEVICE_DIR/run_main_on_hexagon.farf"

# ---- Experiment 1: Tile Basics (DSP-side) ----
echo ""
echo "=== Experiment 1: Tile Basics (DSP-side) ==="
echo "    Micro API tile exploration"
echo ""
adb logcat -c
timeout 30 adb shell "cd $DEVICE_DIR && \
    DSP_LIBRARY_PATH=$DEVICE_DIR \
    LD_LIBRARY_PATH=$DEVICE_DIR \
    ADSP_LIBRARY_PATH=$DEVICE_DIR \
    ./run_main_on_hexagon 3 libexp1_tile_basics.so" 2>&1 || echo "(timeout or failed)"

sleep 1
adb logcat -d | grep "\[MICRO\]" | sed 's/.*\[DU\]: //' | tail -30

# ---- Experiment 2: Weight Layout (DSP-side) ----
echo ""
echo "=== Experiment 2: Weight Layout Optimization Benchmark (DSP-side) ==="
echo "    L0 baseline → L1 wt-cached → L2 preformatted → compute-only"
echo ""
adb logcat -c
timeout 180 adb shell "cd $DEVICE_DIR && \
    DSP_LIBRARY_PATH=$DEVICE_DIR \
    LD_LIBRARY_PATH=$DEVICE_DIR \
    ADSP_LIBRARY_PATH=$DEVICE_DIR \
    ./run_main_on_hexagon 3 libexp2_weight_layout.so" 2>&1 || echo "(timeout or failed)"

sleep 1
adb logcat -d | grep "\[HMX\]" | sed 's/.*\[DU\]: //' | tail -80

# ---- Experiment 3: VTCM Streaming for Large Matrices (DSP-side) ----
echo ""
echo "=== Experiment 3: VTCM Streaming for Large Matrices (DSP-side) ==="
echo "    L2 preformatted vs Stream baseline vs Stream prefmt"
echo "    Tests LLM-sized matrices (4096x4096, 4096x11008)"
echo ""
adb logcat -c
timeout 600 adb shell "cd $DEVICE_DIR && \
    DSP_LIBRARY_PATH=$DEVICE_DIR \
    LD_LIBRARY_PATH=$DEVICE_DIR \
    ADSP_LIBRARY_PATH=$DEVICE_DIR \
    ./run_main_on_hexagon 3 libexp3_streaming.so" 2>&1 || echo "(timeout or failed)"

sleep 1
adb logcat -d | grep "\[STREAM\]" | sed 's/.*\[DU\]: //' | tail -80

# ---- Experiment 4: Pipeline (DSP-side) ----
echo ""
echo "=== Experiment 4: Direct HMX ASM Pipeline ==="
echo "    Phase breakdown: act_pack | wt_load | hmx_compute | out_unpack"
echo ""
adb logcat -c
timeout 300 adb shell "cd $DEVICE_DIR && \
    DSP_LIBRARY_PATH=$DEVICE_DIR \
    LD_LIBRARY_PATH=$DEVICE_DIR \
    ADSP_LIBRARY_PATH=$DEVICE_DIR \
    ./run_main_on_hexagon 3 libexp4_pipeline.so" 2>&1 || echo "(timeout or failed)"

sleep 1
adb logcat -d | grep "\[DIRECT\]" | sed 's/.*\[DU\]: //' | tail -80
