#!/bin/bash
set -euo pipefail

#
# ch05: 推送并运行 dspqueue demo
#
# 在真机上对比 dspqueue vs FastRPC 的通信开销。
#
# 预期输出类似：
#   [Bench] dspqueue OP_SCALE:  ~160 us overhead/op
#   [Bench] FastRPC OP_SCALE:   ~380 us overhead/op
#
# 差异来自 FastRPC 的内核态切换开销。
# llama.cpp 选择 dspqueue 正是因为这个差异。
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD="$SCRIPT_DIR/build"
ARM_BIN="$BUILD/dspqueue_demo"
DSP_SKEL="$BUILD/libdspqueue_demo_skel.so"
DEVICE_DIR="/data/local/tmp/ch05_dspqueue_demo"

# ---- 检查构建产物 ----
if [ ! -f "$ARM_BIN" ] || [ ! -f "$DSP_SKEL" ]; then
    echo "ERROR: Build artifacts not found. Run build.sh first."
    exit 1
fi

echo "========================================"
echo "  ch05: Run dspqueue demo on device"
echo "========================================"
echo ""

# ---- 推送 ----
echo "[1/2] Pushing to device ..."
adb shell "mkdir -p $DEVICE_DIR"
adb push "$ARM_BIN"  "$DEVICE_DIR/"
adb push "$DSP_SKEL" "$DEVICE_DIR/"

# ---- 运行 ----
echo ""
echo "[2/2] Running benchmark ..."
echo ""

adb shell "cd $DEVICE_DIR && \
    chmod +x dspqueue_demo && \
    ADSP_LIBRARY_PATH=. ./dspqueue_demo"

echo ""
echo "========================================"
echo "  Done"
echo "========================================"
