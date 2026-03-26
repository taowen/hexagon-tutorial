#!/usr/bin/env bash
#
# ch06: 推送到真机并在 CDSP 上运行
#
# 和 ch02 一样：用 run_main_on_hexagon 加载 .so 到 CDSP 执行。
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
DEVICE_DIR="/data/local/tmp/ch06"

SO_FILE="$BUILD_DIR/libvtcm_demo.so"
RUN_MAIN="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon"
RUN_MAIN_SKEL="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/hexagon_toolv87_v75/librun_main_on_hexagon_skel.so"

if [ ! -f "$SO_FILE" ]; then
    echo "ERROR: $SO_FILE not found. Run build.sh first."
    exit 1
fi

if [ ! -f "$RUN_MAIN" ]; then
    echo "ERROR: run_main_on_hexagon not found at $RUN_MAIN"
    exit 1
fi

# ---- 检查 adb 连接 ----
if ! adb devices | grep -q "device$"; then
    echo "ERROR: No device connected."
    exit 1
fi

echo "========================================"
echo "  ch06: Run VTCM demo on device"
echo "========================================"

# ---- 推送文件 ----
echo "[1/2] Pushing to device..."
adb shell "mkdir -p $DEVICE_DIR"
adb push "$SO_FILE" "$DEVICE_DIR/"
adb push "$RUN_MAIN" "$DEVICE_DIR/"
adb push "$RUN_MAIN_SKEL" "$DEVICE_DIR/"
adb shell "chmod +x $DEVICE_DIR/run_main_on_hexagon"

# ---- 启用 FARF 日志 ----
adb shell "echo 'vtcm_demo=0x1f' > $DEVICE_DIR/run_main_on_hexagon.farf"

# ---- 运行 ----
echo ""
echo "[2/2] Running on CDSP..."
echo ""

adb logcat -c
timeout 30 adb shell "cd $DEVICE_DIR && \
    DSP_LIBRARY_PATH=$DEVICE_DIR \
    LD_LIBRARY_PATH=$DEVICE_DIR \
    ./run_main_on_hexagon 3 libvtcm_demo.so" 2>&1 \
    || echo "(timed out or failed)"

# ---- 提取 DSP 日志 ----
sleep 1
echo ""
echo "=== DSP Output ==="
adb logcat -d -v brief \
    | grep "\[DU\]" \
    | grep -v "open_mod_table\|Reset loading\|Set loading\|PERF:\|listener\|fastrpc" \
    | sed 's/.*\[DU\]: //' \
    | grep -v "^qurt_\|^thread_new\|^RX VA" \
    | tail -60
