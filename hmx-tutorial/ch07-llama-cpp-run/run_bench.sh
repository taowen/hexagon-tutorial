#!/bin/bash
set -euo pipefail

#
# ch10: 在 NPU 上跑 llama-bench，测量 prefill 和 decode 速度
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$SCRIPT_DIR/pkg/llama.cpp"
DEVICE_BASE="/data/local/tmp/llama.cpp"
DEVICE_GGUF="/data/local/tmp/gguf"

MODEL="${MODEL:-Qwen3-0.6B-Q4_0.gguf}"

if [ ! -f "$PKG_DIR/bin/llama-bench" ]; then
    echo "ERROR: Build artifacts not found. Run build.sh first."
    exit 1
fi

echo "========================================"
echo "  ch10: llama-bench on NPU"
echo "========================================"
echo "  Model: $MODEL"
echo ""

# 确保最新二进制已推送
adb push "$PKG_DIR" "$DEVICE_BASE" >/dev/null

echo "Running benchmark: pp128 (prefill 128 tokens) + tg64 (generate 64 tokens) ..."
echo ""

adb shell " \
    cd $DEVICE_BASE && \
    LD_LIBRARY_PATH=$DEVICE_BASE/lib \
    ADSP_LIBRARY_PATH=$DEVICE_BASE/lib \
    ./bin/llama-bench \
        -m $DEVICE_GGUF/$MODEL \
        -ngl 99 \
        --device HTP0 \
        -t 6 \
        -fa 1 \
        -p 128 -n 64 \
        2>&1
"

echo ""
echo "========================================"
echo "  Done"
echo "========================================"
