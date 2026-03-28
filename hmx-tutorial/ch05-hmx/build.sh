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

# ---------- exp1_tile_basics (DSP side .so -- hexkl_micro.a only) ----------
echo "=== Compiling exp1_tile_basics.c ==="
$HEX_CC $HEX_CFLAGS $HEX_INCLUDES \
    -c "$SCRIPT_DIR/src/exp1_tile_basics.c" \
    -o "$BUILD_DIR/exp1_tile_basics.obj"

echo "=== Linking libexp1_tile_basics.so ==="
$HEX_CC -mv75 $HEX_LDFLAGS \
    "$BUILD_DIR/exp1_tile_basics.obj" \
    -Wl,--start-group "$HEXKL/lib/hexagon_toolv19_v75/libhexkl_micro.a" -Wl,--end-group \
    -o "$BUILD_DIR/libexp1_tile_basics.so"

# ---------- exp2_weight_layout (DSP side .so -- hexkl_micro.a + hexkl_macro.a) ----------
echo "=== Compiling exp2_weight_layout.c ==="
$HEX_CC $HEX_CFLAGS $HEX_INCLUDES \
    -c "$SCRIPT_DIR/src/exp2_weight_layout.c" \
    -o "$BUILD_DIR/exp2_weight_layout.obj"

echo "=== Linking libexp2_weight_layout.so ==="
$HEX_CC -mv75 $HEX_LDFLAGS \
    "$BUILD_DIR/exp2_weight_layout.obj" \
    -Wl,--start-group "$HEXKL/lib/hexagon_toolv19_v75/libhexkl_micro.a" "$HEXKL/lib/hexagon_toolv19_v75/libhexkl_macro.a" -Wl,--end-group \
    -o "$BUILD_DIR/libexp2_weight_layout.so"

# ---------- exp3_streaming (DSP side .so -- hexkl_micro.a + hexkl_macro.a) ----------
echo "=== Compiling exp3_streaming.c ==="
$HEX_CC $HEX_CFLAGS $HEX_INCLUDES \
    -c "$SCRIPT_DIR/src/exp3_streaming.c" \
    -o "$BUILD_DIR/exp3_streaming.obj"

echo "=== Linking libexp3_streaming.so ==="
$HEX_CC -mv75 $HEX_LDFLAGS \
    "$BUILD_DIR/exp3_streaming.obj" \
    -Wl,--start-group "$HEXKL/lib/hexagon_toolv19_v75/libhexkl_micro.a" "$HEXKL/lib/hexagon_toolv19_v75/libhexkl_macro.a" -Wl,--end-group \
    -o "$BUILD_DIR/libexp3_streaming.so"

# ---------- exp4_pipeline (DSP side .so -- hexkl_micro.a + hexkl_macro.a, -mhmx) ----------
echo "=== Compiling exp4_pipeline.c ==="
$HEX_CC $HEX_CFLAGS -mhmx $HEX_INCLUDES \
    -c "$SCRIPT_DIR/src/exp4_pipeline.c" \
    -o "$BUILD_DIR/exp4_pipeline.obj"

echo "=== Linking libexp4_pipeline.so ==="
$HEX_CC -mv75 $HEX_LDFLAGS \
    "$BUILD_DIR/exp4_pipeline.obj" \
    -Wl,--start-group "$HEXKL/lib/hexagon_toolv19_v75/libhexkl_micro.a" "$HEXKL/lib/hexagon_toolv19_v75/libhexkl_macro.a" -Wl,--end-group \
    -o "$BUILD_DIR/libexp4_pipeline.so"

# ---------- exp5_standalone_asm (DSP side .so -- hexkl_micro.a + hexkl_macro.a, -mhmx) ----------
echo "=== Compiling exp5_standalone_asm.c ==="
$HEX_CC $HEX_CFLAGS -mhmx $HEX_INCLUDES \
    -c "$SCRIPT_DIR/src/exp5_standalone_asm.c" \
    -o "$BUILD_DIR/exp5_standalone_asm.obj"

echo "=== Linking libexp5_standalone_asm.so ==="
$HEX_CC -mv75 $HEX_LDFLAGS \
    "$BUILD_DIR/exp5_standalone_asm.obj" \
    -Wl,--start-group "$HEXKL/lib/hexagon_toolv19_v75/libhexkl_micro.a" "$HEXKL/lib/hexagon_toolv19_v75/libhexkl_macro.a" -Wl,--end-group \
    -o "$BUILD_DIR/libexp5_standalone_asm.so"

# ---------- exp6_init_test (DSP side .so -- hexkl_micro.a + hexkl_macro.a, -mhmx) ----------
echo "=== Compiling exp6_init_test.c ==="
$HEX_CC $HEX_CFLAGS -mhmx $HEX_INCLUDES \
    -c "$SCRIPT_DIR/src/exp6_init_test.c" \
    -o "$BUILD_DIR/exp6_init_test.obj"

echo "=== Linking libexp6_init_test.so ==="
$HEX_CC -mv75 $HEX_LDFLAGS \
    "$BUILD_DIR/exp6_init_test.obj" \
    -Wl,--start-group "$HEXKL/lib/hexagon_toolv19_v75/libhexkl_micro.a" "$HEXKL/lib/hexagon_toolv19_v75/libhexkl_macro.a" -Wl,--end-group \
    -o "$BUILD_DIR/libexp6_init_test.so"

echo "=== Build complete ==="
echo "  DSP exp1_tile_basics:    $BUILD_DIR/libexp1_tile_basics.so"
echo "  DSP exp2_weight_layout:  $BUILD_DIR/libexp2_weight_layout.so"
echo "  DSP exp3_streaming:      $BUILD_DIR/libexp3_streaming.so"
echo "  DSP exp4_pipeline:       $BUILD_DIR/libexp4_pipeline.so"
echo "  DSP exp5_standalone_asm: $BUILD_DIR/libexp5_standalone_asm.so"
echo "  DSP exp6_init_test:      $BUILD_DIR/libexp6_init_test.so"
