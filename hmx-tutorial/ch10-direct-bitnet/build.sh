#!/bin/bash
set -euo pipefail

#
# ch10: Build BitNet VLUT16 exploration test
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC="$SCRIPT_DIR/src"
BUILD="$SCRIPT_DIR/build"

SDK="$ROOT_DIR/tools/hexagon-sdk"
ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-${HOME}/android-sdk/ndk/27.2.12479018}"
HCC="$SDK/tools/HEXAGON_Tools/19.0.04/Tools/bin/hexagon-clang"
QAIC="$SDK/ipc/fastrpc/qaic/Ubuntu/qaic"
NDK_CC="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang"

echo "========================================"
echo "  ch10: Build BitNet VLUT16 Test"
echo "========================================"

mkdir -p "$BUILD"

# ========================================
# Step 1: Generate IDL stubs
# ========================================
echo "[1/4] Generating stub/skel from IDL ..."
"$QAIC" -mdll \
    -I "$SDK/incs" \
    -I "$SDK/incs/stddef" \
    "$SRC/bitnet_test.idl" \
    -o "$BUILD"

# ========================================
# Step 2: Compile DSP skel objects
# ========================================
echo "[2/4] Compiling DSP skel ..."
DSP_FLAGS=(-mv75 -O2 -fPIC -mhvx -mhvx-length=128B)
DSP_INCS=(
    -I "$SDK/incs"
    -I "$SDK/incs/stddef"
    -I "$SDK/rtos/qurt/computev75/include/qurt"
    -I "$SRC"
    -I "$BUILD"
)

"$HCC" "${DSP_FLAGS[@]}" "${DSP_INCS[@]}" \
    -c "$SRC/dsp/skel.c" -o "$BUILD/skel.obj"

"$HCC" "${DSP_FLAGS[@]}" "${DSP_INCS[@]}" \
    -c "$BUILD/bitnet_test_skel.c" -o "$BUILD/bitnet_test_skel.obj"

# ========================================
# Step 3: Link DSP shared library
# ========================================
echo "[3/4] Linking DSP skel ..."
"$HCC" -mv75 -shared -Wl,-Bsymbolic \
    "$BUILD/skel.obj" "$BUILD/bitnet_test_skel.obj" \
    -o "$BUILD/libbitnet_test_skel.so"
echo "  -> $BUILD/libbitnet_test_skel.so"

# ========================================
# Step 4: Compile ARM binary
# ========================================
echo "[4/4] Building ARM binary ..."
ARM_INCS=(
    -I "$SDK/incs"
    -I "$SDK/incs/stddef"
    -I "$SDK/ipc/fastrpc/rpcmem/inc"
    -I "$SRC"
    -I "$BUILD"
)
ARM_LIBS=(
    "$BUILD/bitnet_test_stub.c"
    "$SDK/ipc/fastrpc/rpcmem/prebuilt/android_aarch64/rpcmem.a"
    -L "$SDK/ipc/fastrpc/remote/ship/android_aarch64"
    -l cdsprpc
    -lm
)

"$NDK_CC" -O2 \
    "${ARM_INCS[@]}" \
    "$SRC/arm/main.c" \
    "${ARM_LIBS[@]}" \
    -o "$BUILD/vlut16_test"
echo "  -> $BUILD/vlut16_test"

echo ""
echo "========================================"
echo "  Build complete!"
echo "========================================"
echo ""
echo "Artifacts:"
echo "  vlut16_test:              $BUILD/vlut16_test"
echo "  libbitnet_test_skel.so:   $BUILD/libbitnet_test_skel.so"
echo ""
echo "Next: bash $(dirname "$0")/run_device.sh"
