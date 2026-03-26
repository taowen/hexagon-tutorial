#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
HEXKL="$ROOT_DIR/tools/hexkl-addon"
HEXAGON_TOOLS="$HEXAGON_SDK/tools/HEXAGON_Tools/19.0.04/Tools"
HEX_CC="$HEXAGON_TOOLS/bin/hexagon-clang"

if [ ! -d "$HEXKL" ]; then
    echo "HexKL addon not found. Run ch08/install_hexkl.sh first."
    exit 1
fi

mkdir -p "$BUILD_DIR"

HEX_CFLAGS="-mv75 -G0 -fpic -mhvx -mhvx-length=128B -O3 \
    -mllvm -enable-xqf-gen=true \
    -Wall -Wno-unused-function -fno-zero-initialized-in-bss -fdata-sections"

HEX_INCLUDES="\
    -I$HEXAGON_SDK/rtos/qurt/computev75/include \
    -I$HEXAGON_SDK/rtos/qurt/computev75/include/qurt \
    -I$HEXAGON_SDK/rtos/qurt/computev75/include/posix \
    -I$HEXAGON_SDK/incs \
    -I$HEXAGON_SDK/incs/stddef \
    -I$HEXAGON_SDK/ipc/fastrpc/incs \
    -I$HEXKL/include"

HEX_LDFLAGS="-shared -Wl,-Bsymbolic -Wl,--no-threads \
    -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=free \
    -Wl,--wrap=realloc -Wl,--wrap=memalign"

echo "=== Compiling demo_native_kv.c ==="
$HEX_CC $HEX_CFLAGS $HEX_INCLUDES \
    -c "$SCRIPT_DIR/src/demo_native_kv.c" \
    -o "$BUILD_DIR/demo_native_kv.obj"

echo "=== Linking libdemo_native_kv.so ==="
$HEX_CC -mv75 $HEX_LDFLAGS \
    "$BUILD_DIR/demo_native_kv.obj" \
    -Wl,--start-group "$HEXKL/lib/hexagon_toolv19_v75/libhexkl_micro.a" -Wl,--end-group \
    -o "$BUILD_DIR/libdemo_native_kv.so"

echo "=== Build complete ==="
echo "  DSP: $BUILD_DIR/libdemo_native_kv.so"
