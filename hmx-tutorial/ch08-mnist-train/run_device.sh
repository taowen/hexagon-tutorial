#!/usr/bin/env bash
#
# run_device.sh -- Run MNIST training on device
#
# Runs CPU baseline and HVX fused dspqueue training:
#   train_cpu    -- CPU baseline (no DSP)
#   train_fused  -- dspqueue fused HVX f32 training (1 call per batch)
#
# Usage:
#   ./run_device.sh              # Run all, 5 epochs, batch=32
#   ./run_device.sh 3 128        # 3 epochs, batch=128
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
DEVICE_DIR="/data/local/tmp/ch08"

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

# Push binaries
for bin in train_cpu train_fused; do
    if [ -f "$BUILD_DIR/$bin" ]; then
        adb push "$BUILD_DIR/$bin" "$DEVICE_DIR/"
        adb shell "chmod +x $DEVICE_DIR/$bin"
    fi
done

# Push DSP skel
if [ -f "$BUILD_DIR/skel_fused.so" ]; then
    adb push "$BUILD_DIR/skel_fused.so" "$DEVICE_DIR/"
fi

echo ""

# ---- DSP environment ----
DSP_ENV="LD_LIBRARY_PATH=$DEVICE_DIR ADSP_LIBRARY_PATH=$DEVICE_DIR"

# ---- CPU baseline ----
if [ -f "$BUILD_DIR/train_cpu" ]; then
    echo "============================================"
    echo "  CPU Baseline (f32 scalar matmul)"
    echo "============================================"
    adb shell "cd $DEVICE_DIR && ./train_cpu $EPOCHS $BATCH_SIZE"
    echo ""
    echo "--------------------------------------------"
    echo ""
else
    echo "[SKIP] train_cpu not built"
fi

# ---- HVX fused dspqueue training ----
if [ -f "$BUILD_DIR/train_fused" ] && [ -f "$BUILD_DIR/skel_fused.so" ]; then
    echo "============================================"
    echo "  HVX Fused Training (dspqueue, 1 call/batch)"
    echo "============================================"
    adb push "$BUILD_DIR/skel_fused.so" "$DEVICE_DIR/libmnist_train_skel.so"
    adb shell "cd $DEVICE_DIR && $DSP_ENV ./train_fused $EPOCHS $BATCH_SIZE"
    echo ""
else
    echo "[SKIP] train_fused not built (needs Hexagon SDK)"
fi

echo "============================================"
echo "  All steps complete!"
echo "============================================"
