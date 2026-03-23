#!/usr/bin/env bash
#
# run_sim.sh — 在 x86 上通过 QEMU 模拟器运行 QNN 自定义算子
#
# 无需真机! QNN 的 x86_64 libQnnHtp.so 内部使用 libQnnHtpQemu.so
# 来仿真 Hexagon DSP, 运行 hexagon-v75 编译的 .so
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
QNN_SDK="/tmp/qnn_sdk/qairt/2.44.0.260225"

BUILD_DIR="$SCRIPT_DIR/build/x86_64"
HEX_BUILD_DIR="$SCRIPT_DIR/build/hexagon-v75"

# 检查编译产物
for f in \
    "$BUILD_DIR/qnn_hvx_hmx_test" \
    "$BUILD_DIR/libHvxHmxMix_cpu.so" \
    "$HEX_BUILD_DIR/libHvxHmxMix_htp.so"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found. Run build.sh first."
        exit 1
    fi
done

# 复制 HTP .so 到 x86_64 build 目录, 让测试程序能找到
cp "$HEX_BUILD_DIR/libHvxHmxMix_htp.so" "$BUILD_DIR/"

# 创建 QNN 运行时库的符号链接, 让 dlopen("./libQnnHtp.so") 能找到
for lib in libQnnHtp.so libQnnHtpQemu.so libHtpPrepare.so libQnnSystem.so; do
    ln -sf "$QNN_SDK/lib/x86_64-linux-clang/$lib" "$BUILD_DIR/$lib" 2>/dev/null || true
done
# Skel 库也需要在同一目录
ln -sf "$QNN_SDK/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so" "$BUILD_DIR/libQnnHtpV75Skel.so" 2>/dev/null || true

echo "=== Running QNN Custom Op on Simulator (x86_64 + QEMU) ==="
echo ""

# 设置 LD_LIBRARY_PATH:
#   1. build 目录 (libHvxHmxMix_cpu.so, libHvxHmxMix_htp.so)
#   2. QNN x86_64 运行时 (libQnnHtp.so, libQnnHtpQemu.so, libHtpPrepare.so)
#   3. hexagon-v75 skel (libQnnHtpV75Skel.so)
export LD_LIBRARY_PATH="$BUILD_DIR:$QNN_SDK/lib/x86_64-linux-clang:$QNN_SDK/lib/hexagon-v75/unsigned${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$BUILD_DIR"
./qnn_hvx_hmx_test
