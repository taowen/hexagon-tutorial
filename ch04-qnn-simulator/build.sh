#!/usr/bin/env bash
#
# build.sh — Build chapter 4 QNN custom op (x86 simulator via libnative)
#
# Output:
#   build/x86_64/libHvxHmxMix_htp.so  — HTP op package (HVX/HMX via libnative simulation on x86)
#   build/x86_64/qnn_hvx_hmx_test     — Host test program
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Tool paths ----
QNN_SDK="/tmp/qnn_sdk/qairt/2.44.0.260225"
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
HEXAGON_TOOLS="$HEXAGON_SDK/tools/HEXAGON_Tools/19.0.04/Tools"
LIBNATIVE="$HEXAGON_TOOLS/libnative"

PACKAGE_NAME=HvxHmxMixPackage

# ---- Check tools exist ----
for f in \
    "$QNN_SDK/include/QNN/QnnInterface.h" \
    "$LIBNATIVE/lib/libnative.a" \
    "$LIBNATIVE/include/hvx_hexagon_protos.h"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found"
        exit 1
    fi
done

for cmd in clang++ gcc; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found in PATH"
        exit 1
    fi
done

echo "=== Building QNN Custom Op for Simulator (x86_64 via libnative) ==="

# ---- x86_64 HTP op package (.so with libnative) ----
# Uses clang++ and links libnative.a so the SAME HVX/HMX intrinsics
# that would run on real Hexagon DSP work on x86 via simulation.
echo ""
echo "--- HTP op package (x86_64 via libnative) ---"
mkdir -p "$SCRIPT_DIR/build/x86_64"

clang++ -std=c++17 -O2 -fPIC -shared \
    -D__HVXDBL__ -DUSE_OS_LINUX -DPREPARE_DISABLED \
    -DTHIS_PKG_NAME=$PACKAGE_NAME \
    -I"$QNN_SDK/include/QNN" \
    -I"$LIBNATIVE/include" \
    -fomit-frame-pointer -fvisibility=default \
    -Wno-missing-braces -Wno-unused-function -Wno-format \
    -Wno-invalid-offsetof -Wno-unused-variable -Wno-unused-parameter \
    '-DQNN_API=__attribute__((visibility("default")))' \
    '-D__QAIC_HEADER_EXPORT=__attribute__((visibility("default")))' \
    -o "$SCRIPT_DIR/build/x86_64/libHvxHmxMix_htp.so" \
    "$SCRIPT_DIR/src/dsp/HvxHmxInterface.cpp" \
    "$SCRIPT_DIR/src/dsp/HvxHmxOp.cpp" \
    -Wl,--whole-archive -L"$LIBNATIVE/lib" -lnative -Wl,--no-whole-archive \
    -lpthread

echo "  -> build/x86_64/libHvxHmxMix_htp.so"

# ---- Host test program ----
echo ""
echo "--- Host test app (x86_64) ---"

gcc -O2 -Wall -std=c11 \
    -I"$QNN_SDK/include/QNN" \
    -o "$SCRIPT_DIR/build/x86_64/qnn_hvx_hmx_test" \
    "$SCRIPT_DIR/src/host/qnn_hvx_hmx_test.c" \
    -ldl -lm

echo "  -> build/x86_64/qnn_hvx_hmx_test"

echo ""
echo "=== Build complete ==="
echo "Next: bash ch04-qnn-simulator/run_sim.sh"
