#!/bin/bash
set -euo pipefail

#
# ch10: 编译 llama.cpp（ARM 端 + Hexagon DSP 端）
#
# 编译产物：
#   pkg/llama.cpp/bin/llama-cli          ARM 端可执行文件
#   pkg/llama.cpp/bin/llama-bench        ARM 端 benchmark
#   pkg/llama.cpp/lib/libggml-hexagon.so ARM 端 Hexagon 后端
#   pkg/llama.cpp/lib/libggml-htp-v75.so DSP 端算子库（v75 = 骁龙 8 Gen 3）
#   ...以及 v68/v69/v73/v79/v81 版本
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- SDK 路径 ----
HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:-$ROOT_DIR/tools/hexagon-sdk}"
ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-${HOME}/android-sdk/ndk/27.2.12479018}"

# llama.cpp 源码路径（默认在 tutorial 同级目录）
LLAMA_CPP="${LLAMA_CPP:-${HOME}/llama.cpp}"

# ---- 检查依赖 ----
for d in "$HEXAGON_SDK_ROOT" "$ANDROID_NDK_HOME" "$LLAMA_CPP"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: Directory not found: $d"
        echo ""
        echo "请设置以下环境变量："
        echo "  HEXAGON_SDK_ROOT  Hexagon SDK 路径（默认 $ROOT_DIR/tools/hexagon-sdk）"
        echo "  ANDROID_NDK_HOME  Android NDK 路径"
        echo "  LLAMA_CPP         llama.cpp 源码路径（默认 ~/llama.cpp）"
        exit 1
    fi
done

echo "========================================"
echo "  ch10: Build llama.cpp with Hexagon"
echo "========================================"
echo "  LLAMA_CPP:         $LLAMA_CPP"
echo "  HEXAGON_SDK_ROOT:  $HEXAGON_SDK_ROOT"
echo "  ANDROID_NDK_HOME:  $ANDROID_NDK_HOME"
echo ""

BUILD_DIR="$SCRIPT_DIR/build"
PKG_DIR="$SCRIPT_DIR/pkg/llama.cpp"

# ========================================
# Step 1: CMake configure
# ========================================
#
# 关键参数：
#   GGML_HEXAGON=ON        启用 Hexagon 后端
#   GGML_OPENCL=OFF        本章只关注 NPU，关闭 GPU
#   GGML_OPENMP=OFF        Android 上不用 OpenMP
#   CMAKE_TOOLCHAIN_FILE   使用 NDK 的交叉编译工具链
#
echo "[1/3] CMake configure ..."
cmake -S "$LLAMA_CPP" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-31 \
    -DCMAKE_C_FLAGS="-march=armv8.7a+fp16 -fvectorize -ffp-model=fast -fno-finite-math-only -flto" \
    -DCMAKE_CXX_FLAGS="-march=armv8.7a+fp16 -fvectorize -ffp-model=fast -fno-finite-math-only -flto" \
    -DGGML_HEXAGON=ON \
    -DGGML_OPENCL=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" \
    -DPREBUILT_LIB_DIR="android_aarch64" \
    -DLLAMA_OPENSSL=OFF

# ========================================
# Step 2: Build
# ========================================
echo ""
echo "[2/3] Building (ARM + DSP) ..."
cmake --build "$BUILD_DIR" -j$(nproc)

# ========================================
# Step 3: Install to pkg/
# ========================================
echo ""
echo "[3/3] Installing to $PKG_DIR ..."
cmake --install "$BUILD_DIR" --prefix "$PKG_DIR"

echo ""
echo "========================================"
echo "  Build complete!"
echo "========================================"
echo ""
echo "Artifacts:"
echo "  $PKG_DIR/bin/llama-cli"
echo "  $PKG_DIR/bin/llama-bench"
echo "  $PKG_DIR/lib/libggml-hexagon.so"
ls "$PKG_DIR/lib"/libggml-htp-*.so 2>/dev/null | while read f; do
    echo "  $f"
done
echo ""
echo "Next: bash $(dirname "$0")/run_device.sh"
