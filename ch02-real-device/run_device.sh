#!/usr/bin/env bash
#
# run_device.sh — 推送到真机并在 CDSP 上运行
#
# 前提:
#   1. 手机通过 USB 连接, adb 可用
#   2. 已运行 build.sh 编译出 .so
#   3. run_main_on_hexagon 已在设备上 (SDK 自带)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
DEVICE_DIR="/data/local/tmp/ch02"

SO_FILE="$BUILD_DIR/libtest_hvx_hmx_device.so"
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
    echo "ERROR: No device connected. Check USB and run 'adb devices'."
    exit 1
fi

# ---- 推送文件 ----
echo "=== Pushing files to device ==="
adb shell "mkdir -p $DEVICE_DIR"
adb push "$SO_FILE" "$DEVICE_DIR/"
adb push "$RUN_MAIN" "$DEVICE_DIR/"
adb push "$RUN_MAIN_SKEL" "$DEVICE_DIR/"
adb shell "chmod +x $DEVICE_DIR/run_main_on_hexagon"

# ---- 运行 ----
echo ""
echo "=== Running on CDSP (30s timeout) ==="
echo ""

# 清空 logcat, 运行测试
adb logcat -c
timeout 30 adb shell "cd $DEVICE_DIR && \
    DSP_LIBRARY_PATH=\"$DEVICE_DIR\" \
    LD_LIBRARY_PATH=$DEVICE_DIR \
    ./run_main_on_hexagon 3 libtest_hvx_hmx_device.so" 2>&1 \
    || echo "(timed out or failed)"

# ---- 提取 DSP 日志 ----
sleep 1
echo ""
echo "=== DSP Logs ==="
# FARF 日志通过 logcat 输出, 过滤 [DU] 标记, 去掉框架噪声
adb logcat -d -v brief \
    | grep "\[DU\]" \
    | grep -v "open_mod_table\|Reset loading\|Set loading\|PERF:\|listener\|fastrpc" \
    | sed 's/.*\[DU\]: //' \
    | grep -v "^qurt_\|^thread_new\|^RX VA" \
    | tail -50
