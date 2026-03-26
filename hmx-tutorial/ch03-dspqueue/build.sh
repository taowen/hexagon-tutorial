#!/bin/bash
set -euo pipefail

#
# ch05: 构建 dspqueue demo（ARM 端 + DSP skel 端）
#
# 和 llama.cpp 的构建类似，分两步：
#   1. ARM 端：用 Android NDK 的 aarch64-linux-android-clang 编译
#   2. DSP 端：用 Hexagon SDK 的 hexagon-clang 编译
#
# llama.cpp 用 CMake 自动处理这两步，我们手动执行以展示完整流程。
#

# ---- 路径（和 ch01-ch04 保持一致） ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC="$SCRIPT_DIR/src"
BUILD="$SCRIPT_DIR/build"

# SDK 路径：install_tools.sh 安装到 tools/hexagon-sdk
SDK="$ROOT_DIR/tools/hexagon-sdk"
ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-${HOME}/android-sdk/ndk/27.2.12479018}"
HCC="$SDK/tools/HEXAGON_Tools/19.0.04/Tools/bin/hexagon-clang"
QAIC="$SDK/ipc/fastrpc/qaic/Ubuntu/qaic"
NDK_CC="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang"

# ---- 检查依赖 ----
for f in "$SDK" "$ANDROID_NDK_HOME" "$HCC" "$QAIC" "$NDK_CC"; do
    if [ ! -e "$f" ]; then
        echo "ERROR: Not found: $f"
        exit 1
    fi
done

echo "========================================"
echo "  ch05: Build dspqueue demo"
echo "========================================"
echo "  SRC:  $SRC"
echo "  SDK:  $SDK"
echo ""

mkdir -p "$BUILD"

# ========================================
# Step 1: 用 qaic 从 IDL 生成 stub/skel
# ========================================
#
# qaic 是 FastRPC 的 IDL 编译器，输入 .idl 文件，输出：
#   - dspqueue_demo.h          (ARM/DSP 共用的头文件)
#   - dspqueue_demo_stub.c     (ARM 端 stub — 封装 FastRPC 调用)
#   - dspqueue_demo_skel.c     (DSP 端 skel — 解包 FastRPC 参数)
#
# llama.cpp 的 htp_iface.idl 也是用 qaic 生成 htp_iface_stub.c / htp_iface_skel.c
#
echo "[1/3] Generating stub/skel from IDL ..."
"$QAIC" -mdll \
    -I "$SDK/incs" \
    -I "$SDK/incs/stddef" \
    "$SRC/dspqueue_demo.idl" \
    -o "$BUILD"

# ========================================
# Step 2: 编译 ARM 端
# ========================================
#
# ARM 端链接：
#   - libcdsprpc.so   (FastRPC + dspqueue 运行时)
#   - librpcmem.a     (共享内存分配)
#
# llama.cpp 的 ARM 端还链接 libggml-hexagon.so（ggml backend）
#
echo "[2/3] Building ARM side ..."
"$NDK_CC" -O2 \
    -I "$SDK/incs" \
    -I "$SDK/incs/stddef" \
    -I "$SDK/ipc/fastrpc/rpcmem/inc" \
    -I "$SRC" \
    -I "$BUILD" \
    "$SRC/demo_cpu.c" \
    "$BUILD/dspqueue_demo_stub.c" \
    "$SDK/ipc/fastrpc/rpcmem/prebuilt/android_aarch64/rpcmem.a" \
    -L "$SDK/ipc/fastrpc/remote/ship/android_aarch64" \
    -l cdsprpc \
    -o "$BUILD/dspqueue_demo"

echo "  → $BUILD/dspqueue_demo"

# ========================================
# Step 3: 编译 DSP 端
# ========================================
#
# DSP 端用 hexagon-clang 编译为 .so，运行在 Hexagon DSP 上。
# -mv75 表示 Hexagon v75 架构（骁龙 8 Gen 3）。
#
# llama.cpp 编译 6 个版本（v68 ~ v81），我们只编译 v75。
#
echo "[3/3] Building DSP skel side ..."
"$HCC" -mv75 -O2 -shared -fPIC \
    -I "$SDK/incs" \
    -I "$SDK/incs/stddef" \
    -I "$SDK/rtos/qurt/computev75/include/qurt" \
    -I "$SRC" \
    -I "$BUILD" \
    "$BUILD/dspqueue_demo_skel.c" \
    "$SRC/demo_dsp.c" \
    -o "$BUILD/libdspqueue_demo_skel.so"

echo "  → $BUILD/libdspqueue_demo_skel.so"

echo ""
echo "========================================"
echo "  Build complete!"
echo "========================================"
echo ""
echo "Artifacts:"
echo "  ARM binary:  $BUILD/dspqueue_demo"
echo "  DSP skel:    $BUILD/libdspqueue_demo_skel.so"
echo ""
echo "Next: bash $(dirname "$0")/run_device.sh"
