#!/usr/bin/env bash
#
# install_tools.sh — 安装 Hexagon SDK、Toolchain 和 H2 Hypervisor
#
# 用法:
#   bash ch01-simulator-setup/install_tools.sh
#
# 产出:
#   tools/hexagon-sdk   -> Hexagon SDK 6.4.0.2
#   tools/h2-install    -> H2 Hypervisor (编译好的 booter + libh2)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TOOLS_DIR="$ROOT_DIR/tools"
DOWNLOADS_DIR="$ROOT_DIR/downloads"

mkdir -p "$TOOLS_DIR" "$DOWNLOADS_DIR"

# ============================================================
# 1. 下载 Hexagon SDK 6.4.0.2
# ============================================================
SDK_VERSION="6.4.0.2"
SDK_ZIP="Hexagon_SDK_lnx.zip"
SDK_URL="https://softwarecenter.qualcomm.com/api/download/software/sdks/Hexagon_SDK/Linux/Debian/${SDK_VERSION}/${SDK_ZIP}"

if [ ! -f "$DOWNLOADS_DIR/$SDK_ZIP" ]; then
    echo "[1/4] Downloading Hexagon SDK ${SDK_VERSION}..."
    wget -q --show-progress "$SDK_URL" -O "$DOWNLOADS_DIR/$SDK_ZIP"
else
    echo "[1/4] Hexagon SDK already downloaded."
fi

# ============================================================
# 2. 解压 Hexagon SDK
# ============================================================
if [ ! -d "$TOOLS_DIR/hexagon-sdk" ]; then
    echo "[2/4] Extracting Hexagon SDK..."
    TMP_SDK="$TOOLS_DIR/_sdk_extract"
    mkdir -p "$TMP_SDK"
    unzip -q "$DOWNLOADS_DIR/$SDK_ZIP" -d "$TMP_SDK"
    mv "$TMP_SDK/Hexagon_SDK/$SDK_VERSION" "$TOOLS_DIR/hexagon-sdk"
    rm -rf "$TMP_SDK"
else
    echo "[2/4] Hexagon SDK already extracted."
fi

HEXAGON_TOOLS_BIN="$TOOLS_DIR/hexagon-sdk/tools/HEXAGON_Tools/19.0.04/Tools/bin"
if [ ! -f "$HEXAGON_TOOLS_BIN/hexagon-clang" ]; then
    echo "ERROR: hexagon-clang not found at $HEXAGON_TOOLS_BIN"
    exit 1
fi
echo "     hexagon-clang: $HEXAGON_TOOLS_BIN/hexagon-clang"
echo "     hexagon-sim:   $HEXAGON_TOOLS_BIN/hexagon-sim"

# ============================================================
# 3. 克隆并编译 H2 Hypervisor
# ============================================================
H2_SRC="$TOOLS_DIR/hexagon-hypervisor"
if [ ! -d "$H2_SRC" ]; then
    echo "[3/4] Cloning hexagon-hypervisor..."
    git clone https://github.com/qualcomm/hexagon-hypervisor "$H2_SRC"
else
    echo "[3/4] hexagon-hypervisor already cloned."
fi

if [ ! -f "$H2_SRC/install/bin/booter" ]; then
    echo "[4/4] Building H2 Hypervisor..."
    export PATH="$HEXAGON_TOOLS_BIN:$PATH"
    (cd "$H2_SRC" && make ARCHV=75 TARGET=ref USE_PKW=0)
else
    echo "[4/4] H2 Hypervisor already built."
fi

# 创建快捷软链接
ln -sfn "$H2_SRC/install" "$TOOLS_DIR/h2-install"

echo ""
echo "=== Installation complete ==="
echo "  SDK:     $TOOLS_DIR/hexagon-sdk"
echo "  H2:      $TOOLS_DIR/h2-install"
echo "  Booter:  $TOOLS_DIR/h2-install/bin/booter"
