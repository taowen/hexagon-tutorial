#!/bin/bash
set -euo pipefail

#
# ch08: Build MNIST training -- 4 step binaries + 3 DSP skels
#
#   1. train_cpu       -- ARM static, CPU-only (no SDK deps)
#   2. train_fastrpc   -- ARM + FastRPC matmul offload
#   3. train_dspqueue  -- ARM + dspqueue matmul offload
#   4. train_fused     -- ARM + dspqueue fused training
#   DSP skels (one per DSP step):
#     skel_fastrpc.so  -- FastRPC matmul only
#     skel_dspqueue.so -- dspqueue OP_MATMUL only
#     skel_fused.so    -- fused training (OP_REGISTER_NET, OP_TRAIN_BATCH, OP_SYNC)
#
# Same pattern as ch05/build.sh: qaic + NDK clang + hexagon-clang
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC="$SCRIPT_DIR/src"
BUILD="$SCRIPT_DIR/build"

SDK="$ROOT_DIR/tools/hexagon-sdk"
HEXKL="$ROOT_DIR/tools/hexkl-addon"
ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-${HOME}/android-sdk/ndk/27.2.12479018}"
HCC="$SDK/tools/HEXAGON_Tools/19.0.04/Tools/bin/hexagon-clang"
QAIC="$SDK/ipc/fastrpc/qaic/Ubuntu/qaic"
NDK_CC="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang"
ARM_CC="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/clang"

echo "========================================"
echo "  ch08: Build MNIST Training"
echo "========================================"

mkdir -p "$BUILD"

# ========================================
# Step 1: CPU-only static build (no SDK deps)
# ========================================
echo "[1/6] Building train_cpu (static, CPU-only) ..."
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
echo "[2/6] Generating stub/skel from IDL ..."
"$QAIC" -mdll \
    -I "$SDK/incs" \
    -I "$SDK/incs/stddef" \
    "$SRC/common/mnist_train.idl" \
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
# Step 3: Build train_fastrpc
# ========================================
echo "[3/6] Building train_fastrpc ..."
"$NDK_CC" -O2 \
    "${ARM_INCS[@]}" \
    "$SRC/arm/train_fastrpc.c" \
    "${ARM_LIBS[@]}" \
    -o "$BUILD/train_fastrpc"
echo "  -> $BUILD/train_fastrpc"

# ========================================
# Step 4: Build train_dspqueue
# ========================================
echo "[4/6] Building train_dspqueue ..."
"$NDK_CC" -O2 \
    "${ARM_INCS[@]}" \
    "$SRC/arm/train_dspqueue.c" \
    "${ARM_LIBS[@]}" \
    -o "$BUILD/train_dspqueue"
echo "  -> $BUILD/train_dspqueue"

# ========================================
# Step 5: Build train_fused
# ========================================
echo "[5/6] Building train_fused ..."
"$NDK_CC" -O2 \
    "${ARM_INCS[@]}" \
    "$SRC/arm/train_fused.c" \
    "${ARM_LIBS[@]}" \
    -o "$BUILD/train_fused"
echo "  -> $BUILD/train_fused"

# ========================================
# Tests: Adjoint correctness test (runs on DSP)
# ========================================
echo "  Building test_adjoint_dsp (DSP adjoint test) ..."
"$NDK_CC" -O2 \
    "${ARM_INCS[@]}" \
    "$SRC/arm/test_adjoint_dsp.c" \
    "${ARM_LIBS[@]}" \
    -o "$BUILD/test_adjoint_dsp"

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
# Step 7: Build skel_fastrpc.so (FastRPC matmul only)
# ========================================
echo "[7/8] Building skel_fastrpc.so ..."
"$HCC" "${DSP_FLAGS[@]}" "${DSP_INCS[@]}" \
    -c "$SRC/dsp/skel_fastrpc.c" -o "$BUILD/skel_fastrpc.obj"
"$HCC" -mv75 -shared -Wl,-Bsymbolic \
    "$BUILD/skel_fastrpc.obj" "$BUILD/mnist_train_skel.obj" \
    -o "$BUILD/skel_fastrpc.so"
echo "  -> $BUILD/skel_fastrpc.so"

# ========================================
# Step 8a: Build skel_dspqueue.so (dspqueue OP_MATMUL only)
# ========================================
echo "[8/8] Building skel_dspqueue.so + skel_fused.so ..."
"$HCC" "${DSP_FLAGS[@]}" "${DSP_INCS[@]}" \
    -c "$SRC/dsp/skel_dspqueue.c" -o "$BUILD/skel_dspqueue.obj"
"$HCC" -mv75 -shared -Wl,-Bsymbolic \
    "$BUILD/skel_dspqueue.obj" "$BUILD/mnist_train_skel.obj" \
    -o "$BUILD/skel_dspqueue.so"
echo "  -> $BUILD/skel_dspqueue.so"

# ========================================
# Step 8b: Build skel_fused.so (fused training)
# ========================================
"$HCC" "${DSP_FLAGS[@]}" "${DSP_INCS[@]}" \
    -c "$SRC/dsp/skel_fused.c" -o "$BUILD/skel_fused.obj"
"$HCC" -mv75 -shared -Wl,-Bsymbolic \
    "$BUILD/skel_fused.obj" "$BUILD/mnist_train_skel.obj" \
    -o "$BUILD/skel_fused.so"
echo "  -> $BUILD/skel_fused.so"

# ========================================
# Step 8c: Build skel_fused_hmx.so (fused training with HMX matmul)
# ========================================
echo "  Building skel_fused_hmx.so (HMX accelerated) ..."
"$HCC" "${DSP_FLAGS[@]}" -mhmx -DUSE_HMX \
    "${DSP_INCS[@]}" -I "$HEXKL/include" \
    -c "$SRC/dsp/skel_fused.c" -o "$BUILD/skel_fused_hmx.obj"
"$HCC" -mv75 -shared -Wl,-Bsymbolic \
    "$BUILD/skel_fused_hmx.obj" "$BUILD/mnist_train_skel.obj" \
    -Wl,--start-group "$HEXKL/lib/hexagon_toolv19_v75/libhexkl_micro.a" -Wl,--end-group \
    -o "$BUILD/skel_fused_hmx.so"
echo "  -> $BUILD/skel_fused_hmx.so"

# ========================================
# Step 8d: Build skel_fused_f16.so (fused f16 training with HMX)
# ========================================
echo "  Building skel_fused_f16.so (HMX f16) ..."
"$HCC" "${DSP_FLAGS[@]}" -mhmx -DUSE_HMX \
    "${DSP_INCS[@]}" -I "$HEXKL/include" \
    -c "$SRC/dsp/skel_fused_f16.c" -o "$BUILD/skel_fused_f16.obj"
"$HCC" -mv75 -shared -Wl,-Bsymbolic \
    "$BUILD/skel_fused_f16.obj" "$BUILD/mnist_train_skel.obj" \
    -Wl,--start-group "$HEXKL/lib/hexagon_toolv19_v75/libhexkl_micro.a" -Wl,--end-group \
    -o "$BUILD/skel_fused_f16.so"
echo "  -> $BUILD/skel_fused_f16.so"

# ========================================
# Step 9: Build train_fused_f16 (ARM)
# ========================================
echo "  Building train_fused_f16 ..."
"$NDK_CC" -O2 \
    "${ARM_INCS[@]}" \
    "$SRC/arm/train_fused_f16.c" \
    "${ARM_LIBS[@]}" \
    -o "$BUILD/train_fused_f16"
echo "  -> $BUILD/train_fused_f16"

echo ""
echo "========================================"
echo "  Build complete!"
echo "========================================"
echo ""
echo "Artifacts:"
echo "  train_cpu:           $BUILD/train_cpu"
echo "  train_fastrpc:       $BUILD/train_fastrpc"
echo "  train_dspqueue:      $BUILD/train_dspqueue"
echo "  train_fused:         $BUILD/train_fused"
echo "  train_fused_f16:     $BUILD/train_fused_f16     (HMX f16)"
echo "  skel_fastrpc.so:     $BUILD/skel_fastrpc.so"
echo "  skel_dspqueue.so:    $BUILD/skel_dspqueue.so"
echo "  skel_fused.so:       $BUILD/skel_fused.so"
echo "  skel_fused_hmx.so:   $BUILD/skel_fused_hmx.so  (HMX accelerated)"
echo "  skel_fused_f16.so:   $BUILD/skel_fused_f16.so  (HMX f16)"
echo ""
echo "Next: bash $(dirname "$0")/run_device.sh"
