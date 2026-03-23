#!/usr/bin/env bash
#
# build.sh — 编译第三章 QNN 自定义算子
#
# 产出:
#   build/hexagon-v75/libHvxHmxMix_htp.so  — DSP 端 (HTP 内核)
#   build/aarch64/libHvxHmxMix_cpu.so      — ARM 端 (CPU fallback)
#   build/aarch64/qnn_hvx_hmx_test         — ARM 端测试程序
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- 工具路径 ----
QNN_SDK="/tmp/qnn_sdk/qairt/2.44.0.260225"
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
HEXAGON_TOOLS="$HEXAGON_SDK/tools/HEXAGON_Tools/19.0.04/Tools"
NDK="/home/taowen/android-sdk/ndk/27.2.12479018"
TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/linux-x86_64"

HEX_CXX="$HEXAGON_TOOLS/bin/hexagon-clang++"
ARM_CC="$TOOLCHAIN/bin/aarch64-linux-android31-clang"
ARM_CXX="$TOOLCHAIN/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=$TOOLCHAIN/sysroot -stdlib=libc++ -static-libstdc++"
LIBNATIVE="$HEXAGON_TOOLS/libnative"

PACKAGE_NAME=HvxHmxMixPackage

# 检查工具
for f in "$HEX_CXX" "$ARM_CC" "$QNN_SDK/include/QNN/QnnInterface.h"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found"
        exit 1
    fi
done

echo "=== Building QNN Custom Op: HVX + HMX Mix ==="

# ---- DSP 端 .so (Hexagon V75) ----
echo ""
echo "--- DSP package (hexagon-v75) ---"
mkdir -p "$SCRIPT_DIR/build/hexagon-v75"

$HEX_CXX -std=c++17 -O2 -fPIC -shared \
    -mhvx -mhvx-length=128B -mhmx -mv75 \
    -DUSE_OS_QURT -DPREPARE_DISABLED \
    -DTHIS_PKG_NAME=$PACKAGE_NAME \
    -I"$QNN_SDK/include/QNN" \
    -I"$HEXAGON_SDK/rtos/qurt/computev75/include/qurt" \
    -I"$HEXAGON_SDK/rtos/qurt/computev75/include/posix" \
    -I"$HEXAGON_SDK/incs" \
    -I"$HEXAGON_SDK/incs/stddef" \
    -Wall -Wno-missing-braces -Wno-unused-function -Wno-format \
    -Wno-unused-command-line-argument -fvisibility=default -stdlib=libc++ \
    '-DQNN_API=__attribute__((visibility("default")))' \
    '-D__QAIC_HEADER_EXPORT=__attribute__((visibility("default")))' \
    -o "$SCRIPT_DIR/build/hexagon-v75/libHvxHmxMix_htp.so" \
    "$SCRIPT_DIR/src/dsp/HvxHmxInterface.cpp" \
    "$SCRIPT_DIR/src/dsp/HvxHmxOp.cpp"

echo "  -> build/hexagon-v75/libHvxHmxMix_htp.so"

# ---- ARM 端 .so (CPU fallback + interface) ----
echo ""
echo "--- ARM package (aarch64-android) ---"
mkdir -p "$SCRIPT_DIR/build/aarch64"

$ARM_CXX -std=c++17 -O2 -fPIC -shared \
    -D__HVXDBL__ -DUSE_OS_LINUX -DANDROID -DPREPARE_DISABLED \
    -DTHIS_PKG_NAME=$PACKAGE_NAME \
    -I"$QNN_SDK/include/QNN" \
    -I"$LIBNATIVE/include" \
    -fomit-frame-pointer -fvisibility=default \
    -Wno-missing-braces -Wno-unused-function -Wno-format \
    -Wno-invalid-offsetof -Wno-unused-variable -Wno-unused-parameter \
    '-DQNN_API=__attribute__((visibility("default")))' \
    '-D__QAIC_HEADER_EXPORT=__attribute__((visibility("default")))' \
    -o "$SCRIPT_DIR/build/aarch64/libHvxHmxMix_cpu.so" \
    "$SCRIPT_DIR/src/dsp/HvxHmxInterface.cpp" \
    "$SCRIPT_DIR/src/dsp/HvxHmxOp.cpp" \
    -L"$QNN_SDK/lib/aarch64-android" -lQnnHtp -lQnnHtpPrepare

echo "  -> build/aarch64/libHvxHmxMix_cpu.so"

# ---- ARM 端测试程序 ----
echo ""
echo "--- Host test app ---"

$ARM_CC -O2 -Wall -std=c11 \
    -I"$QNN_SDK/include/QNN" \
    -o "$SCRIPT_DIR/build/aarch64/qnn_hvx_hmx_test" \
    "$SCRIPT_DIR/src/host/qnn_hvx_hmx_test.c" \
    -ldl -lm

echo "  -> build/aarch64/qnn_hvx_hmx_test"

echo ""
echo "=== Build complete ==="
echo "Next: bash ch03-qnn-custom-op/run_device.sh"
