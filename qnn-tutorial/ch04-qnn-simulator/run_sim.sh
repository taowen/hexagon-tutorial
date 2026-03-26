#!/usr/bin/env bash
#
# run_sim.sh — 在 x86 上通过 libnative 模拟运行 QNN 自定义算子
#
# 无需真机! libnative 库在 x86 上模拟 HVX/HMX 指令执行.
# 自定义算子的 HVX ReLU + HMX matmul 代码在 x86 上通过 libnative 真实执行,
# 而非 CPU fallback.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
QNN_SDK="/tmp/qnn_sdk/qairt/2.44.0.260225"

BUILD_DIR="$SCRIPT_DIR/build/x86_64"

# 检查编译产物
for f in \
    "$BUILD_DIR/qnn_hvx_hmx_test" \
    "$BUILD_DIR/libHvxHmxMix_htp.so"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found. Run build.sh first."
        exit 1
    fi
done

# 创建 QNN 运行时库的符号链接
for lib in libQnnHtp.so libHtpPrepare.so libQnnSystem.so; do
    ln -sf "$QNN_SDK/lib/x86_64-linux-clang/$lib" "$BUILD_DIR/$lib" 2>/dev/null || true
done

echo "=== Running QNN Custom Op on x86 (libnative HVX/HMX simulation) ==="
echo ""

export LD_LIBRARY_PATH="$BUILD_DIR:$QNN_SDK/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$BUILD_DIR"
./qnn_hvx_hmx_test
