#!/bin/bash
set -euo pipefail

#
# ch10: 推送 llama.cpp 到手机并用 NPU 跑推理
#
# 用法：
#   bash run_device.sh                              # 用默认模型 Qwen3-0.6B-Q4_0.gguf
#   MODEL=Llama-3.2-1B-Instruct-Q4_0.gguf bash run_device.sh
#   PROMPT="Hello" bash run_device.sh               # 自定义 prompt
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$SCRIPT_DIR/pkg/llama.cpp"
DEVICE_BASE="/data/local/tmp/llama.cpp"
DEVICE_GGUF="/data/local/tmp/gguf"

# 默认模型
MODEL="${MODEL:-Qwen3-0.6B-Q4_0.gguf}"
PROMPT="${PROMPT:-你好，请用一句话介绍你自己。}"

# ---- 检查构建产物 ----
if [ ! -f "$PKG_DIR/bin/llama-cli" ]; then
    echo "ERROR: Build artifacts not found. Run build.sh first."
    exit 1
fi

echo "========================================"
echo "  ch10: Run llama.cpp on device (NPU)"
echo "========================================"
echo "  Model:  $MODEL"
echo "  Prompt: $PROMPT"
echo ""

# ========================================
# Step 1: 推送 llama.cpp 到手机
# ========================================
echo "[1/3] Pushing llama.cpp to device ..."
adb push "$PKG_DIR" "$DEVICE_BASE"

# ========================================
# Step 2: 检查模型文件
# ========================================
echo ""
echo "[2/3] Checking model on device ..."
if ! adb shell "ls $DEVICE_GGUF/$MODEL" >/dev/null 2>&1; then
    echo ""
    echo "ERROR: Model not found on device: $DEVICE_GGUF/$MODEL"
    echo ""
    echo "请先下载模型并推送到手机："
    echo ""
    echo "  # 方法1：从 Hugging Face 下载（以 Qwen3-0.6B 为例）"
    echo "  wget https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_0.gguf"
    echo "  adb shell mkdir -p $DEVICE_GGUF"
    echo "  adb push Qwen3-0.6B-Q4_0.gguf $DEVICE_GGUF/"
    echo ""
    echo "  # 方法2：用 Llama-3.2-1B"
    echo "  wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf"
    echo "  adb push Llama-3.2-1B-Instruct-Q4_0.gguf $DEVICE_GGUF/"
    echo "  MODEL=Llama-3.2-1B-Instruct-Q4_0.gguf bash run_device.sh"
    echo ""
    exit 1
fi
echo "  Found: $DEVICE_GGUF/$MODEL"

# ========================================
# Step 3: 在 NPU 上运行推理
# ========================================
echo ""
echo "[3/3] Running inference on NPU (HTP0) ..."
echo ""

adb shell " \
    cd $DEVICE_BASE && \
    LD_LIBRARY_PATH=$DEVICE_BASE/lib \
    ADSP_LIBRARY_PATH=$DEVICE_BASE/lib \
    ./bin/llama-cli \
        --no-mmap \
        -m $DEVICE_GGUF/$MODEL \
        -ngl 99 \
        --device HTP0 \
        -t 6 \
        --ctx-size 2048 \
        -fa on \
        -no-cnv \
        -p '$PROMPT' \
        2>&1
"

echo ""
echo "========================================"
echo "  Done"
echo "========================================"
