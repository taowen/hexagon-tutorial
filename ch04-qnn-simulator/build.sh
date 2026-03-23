#!/usr/bin/env bash
#
# build.sh — 编译第四章 QNN 自定义算子 (x86 模拟器版本)
#
# 产出:
#   build/hexagon-v75/libHvxHmxMix_htp.so  — DSP 端 (HTP 内核, hexagon binary)
#   build/x86_64/libHvxHmxMix_cpu.so       — x86 端 (CPU fallback)
#   build/x86_64/qnn_hvx_hmx_test          — x86 端测试程序
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- 工具路径 ----
QNN_SDK="/tmp/qnn_sdk/qairt/2.44.0.260225"
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
HEXAGON_TOOLS="$HEXAGON_SDK/tools/HEXAGON_Tools/19.0.04/Tools"

HEX_CXX="$HEXAGON_TOOLS/bin/hexagon-clang++"
LIBNATIVE="$HEXAGON_TOOLS/libnative"

PACKAGE_NAME=HvxHmxMixPackage

# 检查工具
for f in "$HEX_CXX" "$QNN_SDK/include/QNN/QnnInterface.h"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found"
        exit 1
    fi
done

echo "=== Building QNN Custom Op for Simulator (x86_64) ==="

# ---- DSP 端 .so (Hexagon V75) ----
# 和 ch03 完全一样: hexagon-clang++ 编译, 在 QEMU 中运行
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

# ---- x86_64 端 .so (CPU fallback + interface) ----
# 关键区别: 用系统 clang++ 编译, 链接 x86_64 QNN 库
echo ""
echo "--- CPU fallback package (x86_64-linux) ---"
mkdir -p "$SCRIPT_DIR/build/x86_64"

g++ -std=c++17 -O2 -fPIC -shared \
    -D__HVXDBL__ -DUSE_OS_LINUX -DPREPARE_DISABLED \
    -DTHIS_PKG_NAME=$PACKAGE_NAME \
    -I"$QNN_SDK/include/QNN" \
    -I"$LIBNATIVE/include" \
    -fomit-frame-pointer -fvisibility=default \
    -Wno-missing-braces -Wno-unused-function -Wno-format \
    -Wno-invalid-offsetof -Wno-unused-variable -Wno-unused-parameter \
    '-DQNN_API=__attribute__((visibility("default")))' \
    '-D__QAIC_HEADER_EXPORT=__attribute__((visibility("default")))' \
    -o "$SCRIPT_DIR/build/x86_64/libHvxHmxMix_cpu.so" \
    "$SCRIPT_DIR/src/dsp/HvxHmxInterface.cpp" \
    "$SCRIPT_DIR/src/dsp/HvxHmxOp.cpp" \
    -L"$QNN_SDK/lib/x86_64-linux-clang" -lQnnHtp -lHtpPrepare

echo "  -> build/x86_64/libHvxHmxMix_cpu.so"

# ---- x86_64 端测试程序 ----
echo ""
echo "--- Host test app (x86_64-linux) ---"

gcc -O2 -Wall -std=c11 \
    -I"$QNN_SDK/include/QNN" \
    -o "$SCRIPT_DIR/build/x86_64/qnn_hvx_hmx_test" \
    "$SCRIPT_DIR/src/host/qnn_hvx_hmx_test.c" \
    -ldl -lm

echo "  -> build/x86_64/qnn_hvx_hmx_test"

echo ""
echo "=== Build complete ==="
echo "Next: bash ch04-qnn-simulator/run_sim.sh"
