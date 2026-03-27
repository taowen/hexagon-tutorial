#!/bin/bash
set -euo pipefail

#
# ch09: Build VTCM Training
#
# Produces:
#   libmnist_train_skel.so  -- skel_vtcm.c (VTCM-resident training + OP_EVAL)
#   train_vtcm              -- ARM driver with DSP-side evaluation
#
# Reuses from ch08: IDL stubs, mnist_train_skel.obj, dspqueue infrastructure
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CH08_DIR="$SCRIPT_DIR/../ch08-mnist-train"
CH08_SRC="$CH08_DIR/src"
CH08_BUILD="$CH08_DIR/build"
SRC="$SCRIPT_DIR/src"
BUILD="$SCRIPT_DIR/build"

SDK="$ROOT_DIR/tools/hexagon-sdk"
HCC="$SDK/tools/HEXAGON_Tools/19.0.04/Tools/bin/hexagon-clang"
ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-${HOME}/android-sdk/ndk/27.2.12479018}"
NDK_CC="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang"

echo "========================================"
echo "  ch09: Build VTCM Training (dspqueue)"
echo "========================================"

# ========================================
# Step 1: Ensure ch08 is built
# ========================================
if [ ! -f "$CH08_BUILD/mnist_train_skel.obj" ] || [ ! -f "$CH08_BUILD/mnist_train_stub.c" ]; then
    echo "[1/4] ch08 not built -- building now..."
    bash "$CH08_DIR/build.sh"
else
    echo "[1/4] ch08 already built"
fi

mkdir -p "$BUILD"

# ========================================
# Step 2: Compile skel_vtcm.c (DSP)
# ========================================
echo "[2/4] Compiling skel_vtcm.c ..."
DSP_FLAGS=(-mv75 -O2 -fPIC -mhvx -mhvx-length=128B)
DSP_INCS=(
    -I "$SDK/incs"
    -I "$SDK/incs/stddef"
    -I "$SDK/rtos/qurt/computev75/include/qurt"
    -I "$CH08_SRC"
    -I "$SRC"
    -I "$CH08_BUILD"
)

"$HCC" "${DSP_FLAGS[@]}" "${DSP_INCS[@]}" \
    -c "$SRC/dsp/skel_vtcm.c" -o "$BUILD/skel_vtcm.obj"
echo "  -> $BUILD/skel_vtcm.obj"

# ========================================
# Step 3: Link libmnist_train_skel.so
# ========================================
echo "[3/4] Linking libmnist_train_skel.so ..."
"$HCC" -mv75 -shared -Wl,-Bsymbolic \
    "$BUILD/skel_vtcm.obj" \
    "$CH08_BUILD/mnist_train_skel.obj" \
    -o "$BUILD/libmnist_train_skel.so"
echo "  -> $BUILD/libmnist_train_skel.so"

# ========================================
# Step 4: Build ARM binary (train_vtcm)
# ========================================
echo "[4/4] Building train_vtcm (ARM, DSP-side eval) ..."
ARM_INCS=(
    -I "$SDK/incs"
    -I "$SDK/incs/stddef"
    -I "$SDK/ipc/fastrpc/rpcmem/inc"
    -I "$CH08_SRC"
    -I "$SRC"
    -I "$CH08_BUILD"
)
ARM_LIBS=(
    "$CH08_BUILD/mnist_train_stub.c"
    "$SDK/ipc/fastrpc/rpcmem/prebuilt/android_aarch64/rpcmem.a"
    -L "$SDK/ipc/fastrpc/remote/ship/android_aarch64"
    -l cdsprpc
    -lm
)

"$NDK_CC" -O2 \
    "${ARM_INCS[@]}" \
    "$SRC/arm/train_vtcm_dspq.c" \
    "${ARM_LIBS[@]}" \
    -o "$BUILD/train_vtcm"
echo "  -> $BUILD/train_vtcm"

echo ""
echo "========================================"
echo "  Build complete!"
echo "========================================"
echo ""
echo "Artifacts:"
echo "  libmnist_train_skel.so:  $BUILD/libmnist_train_skel.so"
echo "  train_vtcm:              $BUILD/train_vtcm"
echo ""
echo "Next: bash run_device.sh"
