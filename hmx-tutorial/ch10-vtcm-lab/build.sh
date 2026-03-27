#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CH08_DIR="$SCRIPT_DIR/../ch08-mnist-train"
CH08_SRC="$CH08_DIR/src"
CH08_BUILD="$CH08_DIR/build"
SRC="$SCRIPT_DIR/src"
BUILD="$SCRIPT_DIR/build"

SDK="$ROOT_DIR/tools/hexagon-sdk"
HCC="$SDK/tools/HEXAGON_Tools/19.0.04/Tools/bin/hexagon-clang"
NDK="${ANDROID_NDK_HOME:-/home/taowen/android-sdk/ndk/27.2.12479018}"
ARM_CC="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android35-clang"

echo "========================================"
echo "  ch10: Build VTCM Lab"
echo "========================================"

# Step 1: Ensure ch08 built (need IDL stubs)
if [ ! -f "$CH08_BUILD/mnist_train_skel.obj" ]; then
    echo "[1/4] Building ch08 first..."
    bash "$CH08_DIR/build.sh"
else
    echo "[1/4] ch08 IDL stubs found"
fi

mkdir -p "$BUILD"

# Step 2: Compile DSP skel
echo "[2/4] Compiling skel_lab.c..."
"$HCC" -mv75 -O2 -fPIC -mhvx -mhvx-length=128B \
    -I "$SDK/incs" -I "$SDK/incs/stddef" \
    -I "$SDK/rtos/qurt/computev75/include/qurt" \
    -I "$CH08_SRC" -I "$SRC" -I "$CH08_BUILD" \
    -c "$SRC/dsp/skel_lab.c" -o "$BUILD/skel_lab.obj"

# Step 3: Link DSP skel
echo "[3/4] Linking libmnist_train_skel.so..."
"$HCC" -mv75 -shared -Wl,-Bsymbolic \
    "$BUILD/skel_lab.obj" \
    "$CH08_BUILD/mnist_train_skel.obj" \
    -o "$BUILD/libmnist_train_skel.so"

# Step 4: Compile ARM binary
echo "[4/4] Compiling ARM binary..."
"$ARM_CC" -O2 -Wall \
    -I "$SDK/incs" -I "$SDK/incs/stddef" \
    -I "$SDK/ipc/fastrpc/rpcmem/inc" \
    -I "$CH08_SRC" -I "$SRC" -I "$CH08_BUILD" \
    "$SRC/arm/test_vtcm.c" "$CH08_BUILD/mnist_train_stub.c" \
    "$SDK/ipc/fastrpc/rpcmem/prebuilt/android_aarch64/rpcmem.a" \
    -L "$SDK/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc \
    -lm \
    -o "$BUILD/test_vtcm"

echo ""
echo "========================================"
echo "  Build complete!"
echo "========================================"
echo "  test_vtcm:               $BUILD/test_vtcm"
echo "  libmnist_train_skel.so:  $BUILD/libmnist_train_skel.so"
echo ""
echo "Next: bash run_device.sh"
