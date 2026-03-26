#!/usr/bin/env bash
#
# run_device.sh -- Run MNIST training on device (4-step experiment)
#
# Runs all 4 steps in sequence:
#   step1_cpu      -- CPU baseline (no DSP)
#   step2_fastrpc  -- FastRPC matmul offload (5 kernel transitions per batch)
#   step3_dspqueue -- dspqueue matmul offload (5 userspace calls per batch)
#   step4_fused    -- dspqueue fused training (1 call per batch)
#
# Usage:
#   ./run_device.sh              # Run all steps, 5 epochs, batch=32
#   ./run_device.sh 3 128        # Run all steps, 3 epochs, batch=128
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
DEVICE_DIR="/data/local/tmp/ch12"

EPOCHS="${1:-5}"
BATCH_SIZE="${2:-32}"

# ---- Check adb connection ----
if ! adb devices | grep -q "device$"; then
    echo "ERROR: No device connected. Check USB and run 'adb devices'."
    exit 1
fi

# ---- Download MNIST data ----
DATA_DIR="$SCRIPT_DIR/data"
mkdir -p "$DATA_DIR"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    if [ ! -f "$DATA_DIR/$f" ]; then
        echo "Downloading $f..."
        curl -sL "https://storage.googleapis.com/cvdf-datasets/mnist/${f}.gz" | gunzip > "$DATA_DIR/$f"
    fi
done
echo "MNIST data ready in $DATA_DIR"

# ---- Push files to device ----
echo "=== Pushing files to device ==="
adb shell "mkdir -p $DEVICE_DIR"
adb push "$DATA_DIR"/* "$DEVICE_DIR/"

# Push step binaries
for step in step1_cpu step2_fastrpc step3_dspqueue step4_fused; do
    if [ -f "$BUILD_DIR/$step" ]; then
        adb push "$BUILD_DIR/$step" "$DEVICE_DIR/"
        adb shell "chmod +x $DEVICE_DIR/$step"
    fi
done

# Push DSP skel .so files (each step gets its own)
for skel in step2_skel.so step3_skel.so step4_skel.so; do
    if [ -f "$BUILD_DIR/$skel" ]; then
        adb push "$BUILD_DIR/$skel" "$DEVICE_DIR/"
    fi
done

echo ""

# ---- DSP environment ----
DSP_ENV="LD_LIBRARY_PATH=$DEVICE_DIR ADSP_LIBRARY_PATH=$DEVICE_DIR"

# ---- Step 1: CPU baseline ----
if [ -f "$BUILD_DIR/step1_cpu" ]; then
    echo "============================================"
    echo "  Step 1: CPU Baseline (f32 scalar matmul)"
    echo "============================================"
    adb shell "cd $DEVICE_DIR && ./step1_cpu $EPOCHS $BATCH_SIZE"
    echo ""
    echo "--------------------------------------------"
    echo ""
else
    echo "[SKIP] step1_cpu not built"
fi

# ---- Step 2: FastRPC matmul offload ----
if [ -f "$BUILD_DIR/step2_fastrpc" ] && [ -f "$BUILD_DIR/step2_skel.so" ]; then
    echo "============================================"
    echo "  Step 2: FastRPC Matmul Offload"
    echo "============================================"
    adb push "$BUILD_DIR/step2_skel.so" "$DEVICE_DIR/libmnist_train_skel.so"
    adb shell "cd $DEVICE_DIR && $DSP_ENV ./step2_fastrpc $EPOCHS $BATCH_SIZE"
    echo ""
    echo "--------------------------------------------"
    echo ""
else
    echo "[SKIP] step2_fastrpc not built (needs Hexagon SDK)"
fi

# ---- Step 3: dspqueue matmul offload ----
if [ -f "$BUILD_DIR/step3_dspqueue" ] && [ -f "$BUILD_DIR/step3_skel.so" ]; then
    echo "============================================"
    echo "  Step 3: dspqueue Matmul Offload"
    echo "============================================"
    adb push "$BUILD_DIR/step3_skel.so" "$DEVICE_DIR/libmnist_train_skel.so"
    adb shell "cd $DEVICE_DIR && $DSP_ENV ./step3_dspqueue $EPOCHS $BATCH_SIZE"
    echo ""
    echo "--------------------------------------------"
    echo ""
else
    echo "[SKIP] step3_dspqueue not built (needs Hexagon SDK)"
fi

# ---- Step 4: dspqueue fused training ----
if [ -f "$BUILD_DIR/step4_fused" ] && [ -f "$BUILD_DIR/step4_skel.so" ]; then
    echo "============================================"
    echo "  Step 4: dspqueue Fused Training"
    echo "============================================"
    adb push "$BUILD_DIR/step4_skel.so" "$DEVICE_DIR/libmnist_train_skel.so"
    adb shell "cd $DEVICE_DIR && $DSP_ENV ./step4_fused $EPOCHS $BATCH_SIZE"
    echo ""
else
    echo "[SKIP] step4_fused not built (needs Hexagon SDK)"
fi

echo "============================================"
echo "  All steps complete!"
echo "  Compare timing across steps to see the"
echo "  effect of each optimization."
echo "============================================"
