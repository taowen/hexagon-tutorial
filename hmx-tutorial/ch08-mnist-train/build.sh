#!/bin/bash
set -euo pipefail

#
# ch08: Build MNIST training -- HVX f32 dspqueue fused
#
#   1. train_cpu       -- ARM CPU baseline (no SDK deps)
#   2. train_fused     -- ARM + dspqueue fused HVX training
#   DSP skel:
#     skel_fused.so    -- fused training (OP_REGISTER_NET, OP_TRAIN_BATCH, OP_SYNC)
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
ARM_CC="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/clang"

echo "========================================"
echo "  ch08: Build MNIST Training (HVX f32)"
echo "========================================"

mkdir -p "$BUILD"

# ========================================
# Step 1: CPU-only static build (no SDK deps)
# ========================================
echo "[1/4] Building train_cpu (static, CPU-only) ..."
$ARM_CC --target=aarch64-linux-android31 -O2 \
    -DCPU_ONLY_BUILD \
    -I "$SRC" \
    "$SRC/arm/train_cpu.c" \
    -lm -static \
    -o "$BUILD/train_cpu"
echo "  -> $BUILD/train_cpu"

# ========================================
# Check SDK tools -- skip DSP builds if missing
# ========================================
if [ ! -e "$QAIC" ] || [ ! -e "$HCC" ]; then
    echo ""
    echo "[WARN] Hexagon SDK tools not found. Skipping DSP build."
    echo "  Only train_cpu (CPU-only) was built."
    echo ""
    echo "========================================"
    echo "  Build complete (CPU-only)"
    echo "========================================"
    ls -lh "$BUILD"/train*
    exit 0
fi

# ========================================
# Step 2: Generate stub/skel from IDL
# ========================================
echo "[2/4] Generating stub/skel from IDL ..."
"$QAIC" -mdll \
    -I "$SDK/incs" \
    -I "$SDK/incs/stddef" \
    "$SRC/common/mnist_train.idl" \
    -o "$BUILD"

# ========================================
# Common flags for ARM DSP build
# ========================================
ARM_INCS=(
    -I "$SDK/incs"
    -I "$SDK/incs/stddef"
    -I "$SDK/ipc/fastrpc/rpcmem/inc"
    -I "$SRC"
    -I "$BUILD"
)
ARM_LIBS=(
    "$BUILD/mnist_train_stub.c"
    "$SDK/ipc/fastrpc/rpcmem/prebuilt/android_aarch64/rpcmem.a"
    -L "$SDK/ipc/fastrpc/remote/ship/android_aarch64"
    -l cdsprpc
    -lm
)

# ========================================
# Step 3: Build train_fused (ARM)
# ========================================
echo "[3/4] Building train_fused ..."
"$NDK_CC" -O2 \
    "${ARM_INCS[@]}" \
    "$SRC/arm/train_fused.c" \
    "${ARM_LIBS[@]}" \
    -o "$BUILD/train_fused"
echo "  -> $BUILD/train_fused"

# ========================================
# Step 4: Build skel_fused.so (DSP)
# ========================================
echo "[4/4] Building skel_fused.so ..."
DSP_INCS=(
    -I "$SDK/incs"
    -I "$SDK/incs/stddef"
    -I "$SDK/rtos/qurt/computev75/include/qurt"
    -I "$SRC"
    -I "$BUILD"
)
DSP_FLAGS=(-mv75 -O2 -fPIC -mhvx -mhvx-length=128B)

"$HCC" "${DSP_FLAGS[@]}" "${DSP_INCS[@]}" \
    -c "$BUILD/mnist_train_skel.c" -o "$BUILD/mnist_train_skel.obj"

"$HCC" "${DSP_FLAGS[@]}" "${DSP_INCS[@]}" \
    -c "$SRC/dsp/skel_fused.c" -o "$BUILD/skel_fused.obj"
"$HCC" -mv75 -shared -Wl,-Bsymbolic \
    "$BUILD/skel_fused.obj" "$BUILD/mnist_train_skel.obj" \
    -o "$BUILD/skel_fused.so"
echo "  -> $BUILD/skel_fused.so"

echo ""
echo "========================================"
echo "  Build complete!"
echo "========================================"
echo ""
echo "Artifacts:"
echo "  train_cpu:       $BUILD/train_cpu"
echo "  train_fused:     $BUILD/train_fused"
echo "  skel_fused.so:   $BUILD/skel_fused.so"
echo ""
echo "Next: bash $(dirname "$0")/run_device.sh"
