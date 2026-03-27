#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CH08_DIR="$SCRIPT_DIR/../ch08-mnist-train"
CH08_BUILD="$CH08_DIR/build"
BUILD="$SCRIPT_DIR/build"
DEVICE_DIR="/data/local/tmp/ch09"

echo "========================================"
echo "  ch09: VTCM Training (dspqueue + VTCM)"
echo "========================================"

# Check build artifacts
for f in "$BUILD/train_vtcm" "$BUILD/libmnist_train_skel.so"; do
    if [ ! -f "$f" ]; then
        echo "[ERROR] Missing: $f -- run build.sh first"
        exit 1
    fi
done

# Check adb connection
if ! adb devices | grep -q "device$"; then
    echo "ERROR: No device connected. Check USB and run 'adb devices'."
    exit 1
fi

# Download MNIST data if needed
DATA_DIR="$SCRIPT_DIR/data"
mkdir -p "$DATA_DIR"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    if [ ! -f "$DATA_DIR/$f" ]; then
        echo "Downloading $f..."
        curl -sL "https://storage.googleapis.com/cvdf-datasets/mnist/${f}.gz" | gunzip > "$DATA_DIR/$f"
    fi
done
echo "MNIST data ready in $DATA_DIR"

# Push to device
echo "[1/3] Pushing to device..."
adb shell "mkdir -p $DEVICE_DIR"
adb push "$BUILD/train_vtcm" "$DEVICE_DIR/"
adb push "$BUILD/libmnist_train_skel.so" "$DEVICE_DIR/"
adb push "$DATA_DIR"/* "$DEVICE_DIR/"
adb shell "chmod +x $DEVICE_DIR/train_vtcm"

DSP_ENV="LD_LIBRARY_PATH=$DEVICE_DIR ADSP_LIBRARY_PATH=$DEVICE_DIR"

EPOCHS="${1:-5}"

# ========================================
# Run 1: ch08 baseline for comparison
# ========================================
echo ""
echo "[2/3] Running ch08 baseline (skel_fused.so, batch=128, $EPOCHS epochs)..."
adb push "$CH08_BUILD/train_fused" "$DEVICE_DIR/"
adb push "$CH08_BUILD/skel_fused.so" "$DEVICE_DIR/libmnist_train_skel.so"
adb shell "chmod +x $DEVICE_DIR/train_fused"
adb shell "cd $DEVICE_DIR && $DSP_ENV ./train_fused $EPOCHS 128"

# ========================================
# Run 2: ch09 VTCM version with DSP-side eval
# ========================================
echo ""
echo "[3/3] Running ch09 VTCM version (batch=128, $EPOCHS epochs, DSP-side eval)..."
adb push "$BUILD/libmnist_train_skel.so" "$DEVICE_DIR/"
adb shell "cd $DEVICE_DIR && $DSP_ENV ./train_vtcm $EPOCHS 128"

echo ""
echo "========================================"
echo "Done!"
echo "========================================"
