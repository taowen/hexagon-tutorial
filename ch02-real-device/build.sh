#!/usr/bin/env bash
#
# build.sh — 编译第二章示例 (DSP 端 .so)
#
# 产出: build/libtest_hvx_hmx_device.so
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"

# ---- 工具路径 ----
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
HEXAGON_TOOLS_BIN="$HEXAGON_SDK/tools/HEXAGON_Tools/19.0.04/Tools/bin"
HEX_CC="$HEXAGON_TOOLS_BIN/hexagon-clang"

if [ ! -f "$HEX_CC" ]; then
    echo "ERROR: hexagon-clang not found at $HEX_CC"
    echo "Run install_tools.sh first."
    exit 1
fi

# ---- 编译 DSP 端 .so ----
# 关键区别 (vs 第一章):
#   - -shared -fPIC: 编译为共享库 (不是 ELF 可执行文件)
#   - -mv75: 真机 V75 架构 (模拟器用 -mv73)
#   - 头文件来自 SDK (不是 H2 Hypervisor)
#   - 不需要 -moslib=h2, 不需要 --section-start
echo "=== Compiling DSP .so ==="
$HEX_CC -mv75 -O2 \
    -mhvx -mhvx-length=128B \
    -mhmx \
    -shared -fPIC \
    -I "$HEXAGON_SDK/incs" \
    -I "$HEXAGON_SDK/incs/stddef" \
    -I "$HEXAGON_SDK/rtos/qurt/computev75/include/qurt" \
    "$SCRIPT_DIR/test_hvx_hmx_device.c" \
    -o "$BUILD_DIR/libtest_hvx_hmx_device.so"

echo "  -> $BUILD_DIR/libtest_hvx_hmx_device.so"
echo "Done. Next: bash ch02-real-device/run_device.sh"
