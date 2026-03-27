#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TOOLS_DIR="$ROOT_DIR/tools"

echo "=== Installing HexKL addon ==="

if [ -f "$TOOLS_DIR/hexkl-addon/include/sdkl.h" ]; then
    echo "=== HexKL addon already installed at $TOOLS_DIR/hexkl-addon"
    exit 0
fi

mkdir -p "$TOOLS_DIR"

LOCAL_CACHE="/home/taowen/hexagon-mlir/downloads/Hexagon_KL.Core.1.0.0.Linux-Any.zip"
ZIP_PATH="/tmp/hexkl.zip"

if [ -f "$LOCAL_CACHE" ]; then
    echo "=== Found local copy at $LOCAL_CACHE"
    ZIP_PATH="$LOCAL_CACHE"
else
    echo "=== Downloading HexKL addon..."
    wget -q "https://softwarecenter.qualcomm.com/api/download/software/tools/Hexagon_KL/Linux/1.0.0/Hexagon_KL.Core.1.0.0.Linux-Any.zip" -O "$ZIP_PATH"
    echo "=== Download complete"
fi

# The outer zip contains per-SDK-version inner zips.
# We need hexkl-1.0.0-beta1-6.4.0.0.zip (matching our SDK 6.4.0).
INNER_ZIP="hexkl-1.0.0-beta1-6.4.0.0.zip"
TMP_EXTRACT="/tmp/hexkl_extract_$$"
mkdir -p "$TMP_EXTRACT"

echo "=== Extracting outer zip..."
unzip -q "$ZIP_PATH" "$INNER_ZIP" -d "$TMP_EXTRACT"

echo "=== Extracting inner zip ($INNER_ZIP) to $TOOLS_DIR/hexkl-addon..."
rm -rf "$TOOLS_DIR/hexkl-addon"
unzip -q "$TMP_EXTRACT/$INNER_ZIP" -d "$TMP_EXTRACT/inner"
# The inner zip extracts to hexkl_addon/ — move it to our target path
mv "$TMP_EXTRACT/inner/hexkl_addon" "$TOOLS_DIR/hexkl-addon"
rm -rf "$TMP_EXTRACT"

echo "=== Verifying installation..."
MISSING=0
for f in \
    "$TOOLS_DIR/hexkl-addon/include/sdkl.h" \
    "$TOOLS_DIR/hexkl-addon/include/hexkl_macro.h" \
    "$TOOLS_DIR/hexkl-addon/include/hexkl_micro.h" \
    "$TOOLS_DIR/hexkl-addon/lib/hexagon_toolv19_v75/libhexkl_skel.so" \
    "$TOOLS_DIR/hexkl-addon/lib/armv8_android26/libsdkl.so"; do
    if [ ! -f "$f" ]; then
        echo "=== ERROR: Missing expected file: $f"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo "=== Installation may be incomplete, some expected files are missing"
    exit 1
fi

echo "=== HexKL addon installed successfully"
echo "=== Installed files summary:"
echo "    Headers:"
ls -1 "$TOOLS_DIR/hexkl-addon/include/"/*.h 2>/dev/null | sed 's/^/      /'
echo "    Libraries (Hexagon v75):"
ls -1 "$TOOLS_DIR/hexkl-addon/lib/hexagon_toolv19_v75/"*.so 2>/dev/null | sed 's/^/      /'
echo "    Libraries (ARM Android):"
ls -1 "$TOOLS_DIR/hexkl-addon/lib/armv8_android26/"*.so 2>/dev/null | sed 's/^/      /'
