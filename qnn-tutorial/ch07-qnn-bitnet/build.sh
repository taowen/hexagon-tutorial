#!/usr/bin/env bash
#
# build.sh — Build TMAN BitNet op package (ch07)
#
# Output:
#   build/hexagon-v75/libTMANOpPackage_htp.so  — DSP kernel (HTP)
#   build/aarch64/libTMANOpPackage_cpu.so      — ARM CPU fallback
#   build/aarch64/bitnet_test                  — ARM test app
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- Tool paths ----
QNN_SDK="$ROOT_DIR/tools/qnn-sdk"
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
HEXAGON_TOOLS="$HEXAGON_SDK/tools/HEXAGON_Tools/19.0.04/Tools"
NDK="/home/taowen/android-sdk/ndk/27.2.12479018"
TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/linux-x86_64"

HEX_CXX="$HEXAGON_TOOLS/bin/hexagon-clang++"
ARM_CC="$TOOLCHAIN/bin/aarch64-linux-android31-clang"
ARM_CXX="$TOOLCHAIN/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=$TOOLCHAIN/sysroot -stdlib=libc++ -static-libstdc++"
LIBNATIVE="$HEXAGON_TOOLS/libnative"

PACKAGE_NAME=TMANOpPackage

# Check tools
for f in "$HEX_CXX" "$ARM_CC" "$QNN_SDK/include/QNN/QnnInterface.h"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found"
        exit 1
    fi
done

echo "=== Building ch07: TMAN BitNet Op Package ==="

# ---- DSP (Hexagon V75) ----
echo ""
echo "--- DSP package (hexagon-v75) ---"
mkdir -p "$SCRIPT_DIR/build/hexagon-v75"

$HEX_CXX -std=c++17 -O2 -fPIC -shared \
    -mhvx -mhvx-length=128B -mv75 \
    -DUSE_OS_QURT -DPREPARE_DISABLED \
    -DTHIS_PKG_NAME=$PACKAGE_NAME \
    '-DTHIS_PKG_NAME_STR="TMANOpPackage"' \
    -I"$QNN_SDK/include/QNN" \
    -I"$HEXAGON_SDK/rtos/qurt/computev75/include/qurt" \
    -I"$HEXAGON_SDK/rtos/qurt/computev75/include/posix" \
    -I"$HEXAGON_SDK/incs" \
    -I"$HEXAGON_SDK/incs/stddef" \
    -I"$SCRIPT_DIR/src/dsp/include" \
    -Wall -Wno-missing-braces -Wno-unused-function -Wno-format \
    -Wno-unused-command-line-argument -Wno-unused-variable -Wno-unused-parameter \
    -fvisibility=default -stdlib=libc++ \
    '-DQNN_API=__attribute__((visibility("default")))' \
    '-D__QAIC_HEADER_EXPORT=__attribute__((visibility("default")))' \
    -o "$SCRIPT_DIR/build/hexagon-v75/libTMANOpPackage_htp.so" \
    "$SCRIPT_DIR/src/dsp/TMANOpPackageInterface.cpp" \
    "$SCRIPT_DIR/src/dsp/ops/TMANPrecompute.cpp" \
    "$SCRIPT_DIR/src/dsp/ops/TMANLinear.cpp" \
    "$SCRIPT_DIR/src/dsp/ops/TMANFinalize.cpp"

echo "  -> build/hexagon-v75/libTMANOpPackage_htp.so"

# ---- ARM CPU fallback .so ----
echo ""
echo "--- ARM package (aarch64-android) ---"
mkdir -p "$SCRIPT_DIR/build/aarch64"

$ARM_CXX -std=c++17 -O2 -fPIC -shared \
    -D__HVXDBL__ -DUSE_OS_LINUX -DANDROID -DPREPARE_DISABLED \
    -DTHIS_PKG_NAME=$PACKAGE_NAME \
    '-DTHIS_PKG_NAME_STR="TMANOpPackage"' \
    -I"$QNN_SDK/include/QNN" \
    -I"$LIBNATIVE/include" \
    -I"$SCRIPT_DIR/src/dsp/include" \
    -fomit-frame-pointer -fvisibility=default \
    -Wno-missing-braces -Wno-unused-function -Wno-format \
    -Wno-invalid-offsetof -Wno-unused-variable -Wno-unused-parameter \
    '-DQNN_API=__attribute__((visibility("default")))' \
    '-D__QAIC_HEADER_EXPORT=__attribute__((visibility("default")))' \
    -o "$SCRIPT_DIR/build/aarch64/libTMANOpPackage_cpu.so" \
    "$SCRIPT_DIR/src/dsp/TMANOpPackageInterface.cpp" \
    "$SCRIPT_DIR/src/dsp/ops/TMANPrecompute.cpp" \
    "$SCRIPT_DIR/src/dsp/ops/TMANLinear.cpp" \
    "$SCRIPT_DIR/src/dsp/ops/TMANFinalize.cpp" \
    "$SCRIPT_DIR/src/dsp/fp_extend.cpp" \
    "$SCRIPT_DIR/src/dsp/fp_trunc.cpp" \
    -L"$QNN_SDK/lib/aarch64-android" -lQnnHtp -lQnnHtpPrepare

echo "  -> build/aarch64/libTMANOpPackage_cpu.so"

echo ""
echo "=== Build complete ==="
