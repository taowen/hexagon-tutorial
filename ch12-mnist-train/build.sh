#!/bin/bash
set -euo pipefail

#
# ch12: Build MNIST training -- 4 step binaries + 3 DSP skels
#
#   1. step1_cpu     -- ARM static, CPU-only (no SDK deps)
#   2. step2_fastrpc -- ARM + FastRPC matmul offload
#   3. step3_dspqueue -- ARM + dspqueue matmul offload
#   4. step4_fused   -- ARM + dspqueue fused training
#   DSP skels (one per DSP step):
#     step2_skel.so  -- FastRPC matmul only
#     step3_skel.so  -- dspqueue OP_MATMUL only
#     step4_skel.so  -- fused training (OP_REGISTER_NET, OP_TRAIN_BATCH, OP_SYNC)
#
# Same pattern as ch05/build.sh: qaic + NDK clang + hexagon-clang
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$SCRIPT_DIR/src"
BUILD="$SCRIPT_DIR/build"

SDK="$ROOT_DIR/tools/hexagon-sdk"
ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-${HOME}/android-sdk/ndk/27.2.12479018}"
HCC="$SDK/tools/HEXAGON_Tools/19.0.04/Tools/bin/hexagon-clang"
QAIC="$SDK/ipc/fastrpc/qaic/Ubuntu/qaic"
NDK_CC="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang"
ARM_CC="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/clang"

echo "========================================"
echo "  ch12: Build MNIST Training"
echo "========================================"

mkdir -p "$BUILD"

# ========================================
# Step 1: CPU-only static build (no SDK deps)
# ========================================
echo "[1/6] Building step1_cpu (static, CPU-only) ..."
$ARM_CC --target=aarch64-linux-android31 -O2 \
    -DCPU_ONLY_BUILD \
    "$SRC/step1_cpu.c" \
    -lm -static \
    -o "$BUILD/step1_cpu"
echo "  -> $BUILD/step1_cpu"

# ========================================
# Check SDK tools -- skip DSP builds if missing
# ========================================
if [ ! -e "$QAIC" ] || [ ! -e "$HCC" ]; then
    echo ""
    echo "[WARN] Hexagon SDK tools not found. Skipping DSP build."
    echo "  Only step1_cpu (CPU-only) was built."
    echo ""
    echo "========================================"
    echo "  Build complete (CPU-only)"
    echo "========================================"
    ls -lh "$BUILD"/step*
    exit 0
fi

# ========================================
# Step 2: Generate stub/skel from IDL
# ========================================
echo "[2/6] Generating stub/skel from IDL ..."
"$QAIC" -mdll \
    -I "$SDK/incs" \
    -I "$SDK/incs/stddef" \
    "$SRC/mnist_train.idl" \
    -o "$BUILD"

# ========================================
# Common flags for ARM DSP builds (steps 2-4)
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
# Step 3: Build step2_fastrpc
# ========================================
echo "[3/6] Building step2_fastrpc ..."
"$NDK_CC" -O2 \
    "${ARM_INCS[@]}" \
    "$SRC/step2_fastrpc.c" \
    "${ARM_LIBS[@]}" \
    -o "$BUILD/step2_fastrpc"
echo "  -> $BUILD/step2_fastrpc"

# ========================================
# Step 4: Build step3_dspqueue
# ========================================
echo "[4/6] Building step3_dspqueue ..."
"$NDK_CC" -O2 \
    "${ARM_INCS[@]}" \
    "$SRC/step3_dspqueue.c" \
    "${ARM_LIBS[@]}" \
    -o "$BUILD/step3_dspqueue"
echo "  -> $BUILD/step3_dspqueue"

# ========================================
# Step 5: Build step4_fused
# ========================================
echo "[5/6] Building step4_fused ..."
"$NDK_CC" -O2 \
    "${ARM_INCS[@]}" \
    "$SRC/step4_fused.c" \
    "${ARM_LIBS[@]}" \
    -o "$BUILD/step4_fused"
echo "  -> $BUILD/step4_fused"

# ========================================
# Step 6: Build DSP skel (shared object, compiled once)
# ========================================
echo "[6/8] Building DSP skel object ..."
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

# ========================================
# Step 7: Build step2_skel.so (FastRPC matmul only)
# ========================================
echo "[7/8] Building step2_skel.so ..."
"$HCC" "${DSP_FLAGS[@]}" "${DSP_INCS[@]}" \
    -c "$SRC/step2_dsp.c" -o "$BUILD/step2_dsp.obj"
"$HCC" -mv75 -shared -Wl,-Bsymbolic \
    "$BUILD/step2_dsp.obj" "$BUILD/mnist_train_skel.obj" \
    -o "$BUILD/step2_skel.so"
echo "  -> $BUILD/step2_skel.so"

# ========================================
# Step 8a: Build step3_skel.so (dspqueue OP_MATMUL only)
# ========================================
echo "[8/8] Building step3_skel.so + step4_skel.so ..."
"$HCC" "${DSP_FLAGS[@]}" "${DSP_INCS[@]}" \
    -c "$SRC/step3_dsp.c" -o "$BUILD/step3_dsp.obj"
"$HCC" -mv75 -shared -Wl,-Bsymbolic \
    "$BUILD/step3_dsp.obj" "$BUILD/mnist_train_skel.obj" \
    -o "$BUILD/step3_skel.so"
echo "  -> $BUILD/step3_skel.so"

# ========================================
# Step 8b: Build step4_skel.so (fused training)
# ========================================
"$HCC" "${DSP_FLAGS[@]}" "${DSP_INCS[@]}" \
    -c "$SRC/step4_dsp.c" -o "$BUILD/step4_dsp.obj"
"$HCC" -mv75 -shared -Wl,-Bsymbolic \
    "$BUILD/step4_dsp.obj" "$BUILD/mnist_train_skel.obj" \
    -o "$BUILD/step4_skel.so"
echo "  -> $BUILD/step4_skel.so"

echo ""
echo "========================================"
echo "  Build complete!"
echo "========================================"
echo ""
echo "Artifacts:"
echo "  step1_cpu:      $BUILD/step1_cpu"
echo "  step2_fastrpc:  $BUILD/step2_fastrpc"
echo "  step3_dspqueue: $BUILD/step3_dspqueue"
echo "  step4_fused:    $BUILD/step4_fused"
echo "  step2_skel.so:  $BUILD/step2_skel.so"
echo "  step3_skel.so:  $BUILD/step3_skel.so"
echo "  step4_skel.so:  $BUILD/step4_skel.so"
echo ""
echo "Next: bash $(dirname "$0")/run_device.sh"
