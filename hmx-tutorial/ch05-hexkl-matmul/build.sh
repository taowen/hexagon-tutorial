#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
HEXKL="$ROOT_DIR/tools/hexkl-addon"
HEXAGON_TOOLS="$HEXAGON_SDK/tools/HEXAGON_Tools/19.0.04/Tools"
HEX_CC="$HEXAGON_TOOLS/bin/hexagon-clang"
NDK="/home/taowen/android-sdk/ndk/27.2.12479018"
ARM_CC="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/clang"

if [ ! -d "$HEXKL" ]; then
    echo "Run install_hexkl.sh first"
    exit 1
fi

mkdir -p "$BUILD_DIR"

# Common Hexagon flags
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

# ---------- 1. demo_hvx_hmx (DSP side .so -- HVX + HMX combined) ----------
echo "=== Compiling demo_hvx_hmx.c ==="
$HEX_CC $HEX_CFLAGS $HEX_INCLUDES \
    -c "$SCRIPT_DIR/src/demo_hvx_hmx.c" \
    -o "$BUILD_DIR/demo_hvx_hmx.obj"

echo "=== Linking libdemo_hvx_hmx.so ==="
$HEX_CC -mv75 $HEX_LDFLAGS \
    "$BUILD_DIR/demo_hvx_hmx.obj" \
    -Wl,--start-group "$HEXKL/lib/hexagon_toolv19_v75/libhexkl_micro.a" -Wl,--end-group \
    -o "$BUILD_DIR/libdemo_hvx_hmx.so"

# ---------- 2. demo_micro (DSP side .so) ----------
echo "=== Compiling demo_micro.c ==="
$HEX_CC $HEX_CFLAGS $HEX_INCLUDES \
    -c "$SCRIPT_DIR/src/demo_micro.c" \
    -o "$BUILD_DIR/demo_micro.obj"

echo "=== Linking libdemo_micro.so ==="
$HEX_CC -mv75 $HEX_LDFLAGS \
    "$BUILD_DIR/demo_micro.obj" \
    -Wl,--start-group "$HEXKL/lib/hexagon_toolv19_v75/libhexkl_micro.a" -Wl,--end-group \
    -o "$BUILD_DIR/libdemo_micro.so"

# ---------- 3. demo_sdkl (ARM side executable) ----------
echo "=== Compiling demo_sdkl ==="
$ARM_CC --target=aarch64-linux-android31 \
    -march=armv8.2-a+dotprod+i8mm+fp16 -ffast-math -O3 \
    -fPIE -pie \
    -I"$HEXKL/include" \
    -I"$HEXAGON_SDK/incs" \
    "$SCRIPT_DIR/src/demo_sdkl.c" \
    "$HEXKL/lib/armv8_android26/libsdkl.so" \
    -L"$HEXAGON_SDK/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc \
    -llog -lm \
    -o "$BUILD_DIR/demo_sdkl"

echo "=== Build complete ==="
echo "  DSP hvx_hmx: $BUILD_DIR/libdemo_hvx_hmx.so"
echo "  DSP micro:   $BUILD_DIR/libdemo_micro.so"
echo "  ARM sdkl:    $BUILD_DIR/demo_sdkl"
