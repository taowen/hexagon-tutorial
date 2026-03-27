#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD="$SCRIPT_DIR/build"
DEVICE_DIR="/data/local/tmp/ch10"

echo "========================================"
echo "  ch10: VTCM Lab"
echo "========================================"

for f in "$BUILD/test_vtcm" "$BUILD/libmnist_train_skel.so"; do
    if [ ! -f "$f" ]; then
        echo "[ERROR] Missing: $f -- run build.sh first"
        exit 1
    fi
done

if ! adb devices | grep -q "device$"; then
    echo "ERROR: No device connected"
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

echo "[1/2] Pushing to device..."
adb shell "mkdir -p $DEVICE_DIR"
adb push "$BUILD/test_vtcm" "$DEVICE_DIR/"
adb push "$BUILD/libmnist_train_skel.so" "$DEVICE_DIR/"
adb push "$DATA_DIR"/* "$DEVICE_DIR/"
adb shell "chmod +x $DEVICE_DIR/test_vtcm"

EPOCHS="${1:-10}"
echo "[2/2] Running VTCM lab ($EPOCHS epochs for training test)..."
adb shell "cd $DEVICE_DIR && \
    LD_LIBRARY_PATH=$DEVICE_DIR \
    ADSP_LIBRARY_PATH='$DEVICE_DIR;/vendor/lib/rfsa/dsp/sdk;/vendor/lib/rfsa/dsp/testsig;/dsp' \
    ./test_vtcm $EPOCHS"

echo ""
echo "Done!"
