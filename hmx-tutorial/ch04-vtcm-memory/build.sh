#!/usr/bin/env bash
#
# ch06: Build VTCM demo (DSP .so)
#
# Same as ch02: compile to .so, loaded via run_main_on_hexagon.
# No IDL/stub/skel needed — multiple source files compiled together.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"

# ---- Tool paths (same as ch02) ----
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
HEX_CC="$HEXAGON_SDK/tools/HEXAGON_Tools/19.0.04/Tools/bin/hexagon-clang"

if [ ! -f "$HEX_CC" ]; then
    echo "ERROR: hexagon-clang not found at $HEX_CC"
    echo "Run ch01-simulator-setup/install_tools.sh first."
    exit 1
fi

echo "========================================"
echo "  ch06: Build VTCM demo"
echo "========================================"

# ---- Compile DSP .so from all src/*.c files ----
#
# Same compiler options as ch02:
#   -shared -fPIC: shared library
#   -mv75: v75 architecture (Snapdragon 8 Gen 3)
#   -mhvx -mhvx-length=128B: HVX intrinsics
#
$HEX_CC -mv75 -O2 \
    -mhvx -mhvx-length=128B \
    -shared -fPIC \
    -I "$HEXAGON_SDK/incs" \
    -I "$HEXAGON_SDK/incs/stddef" \
    -I "$HEXAGON_SDK/rtos/qurt/computev75/include/qurt" \
    -I "$SCRIPT_DIR/src" \
    "$SCRIPT_DIR"/src/*.c \
    -o "$BUILD_DIR/libvtcm_demo.so"

echo "  → $BUILD_DIR/libvtcm_demo.so"
echo ""
echo "Done. Next: bash ch06-vtcm-memory/run_device.sh"
