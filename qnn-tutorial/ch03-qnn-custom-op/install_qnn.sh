#!/usr/bin/env bash
#
# install_qnn.sh — 安装 QNN SDK (Qualcomm AI Runtime)
#
# 用法:
#   bash ch03-qnn-custom-op/install_qnn.sh
#
# 产出:
#   tools/qnn-sdk -> qairt/2.44.0.260225 (QNN SDK 根目录)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TOOLS_DIR="$ROOT_DIR/tools"
DOWNLOADS_DIR="$ROOT_DIR/downloads"

mkdir -p "$TOOLS_DIR" "$DOWNLOADS_DIR"

# ============================================================
# 1. 下载 QNN SDK 2.44.0.260225
# ============================================================
QNN_VERSION="2.44.0.260225"
QNN_ZIP="v${QNN_VERSION}.zip"
QNN_URL="https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${QNN_VERSION}/${QNN_ZIP}"

if [ ! -f "$DOWNLOADS_DIR/$QNN_ZIP" ]; then
    echo "[1/2] Downloading QNN SDK ${QNN_VERSION}..."
    echo "  URL: $QNN_URL"
    wget -q --show-progress "$QNN_URL" -O "$DOWNLOADS_DIR/$QNN_ZIP"
else
    echo "[1/2] QNN SDK already downloaded."
fi

# ============================================================
# 2. 解压 QNN SDK
# ============================================================
if [ ! -d "$TOOLS_DIR/qnn-sdk" ]; then
    echo "[2/2] Extracting QNN SDK..."
    TMP_QNN="$TOOLS_DIR/_qnn_extract"
    mkdir -p "$TMP_QNN"
    unzip -q "$DOWNLOADS_DIR/$QNN_ZIP" -d "$TMP_QNN"
    mv "$TMP_QNN/qairt/$QNN_VERSION" "$TOOLS_DIR/qnn-sdk"
    rm -rf "$TMP_QNN"
else
    echo "[2/2] QNN SDK already extracted."
fi

# 验证关键文件
for f in \
    "$TOOLS_DIR/qnn-sdk/include/QNN/QnnInterface.h" \
    "$TOOLS_DIR/qnn-sdk/lib/aarch64-android/libQnnHtp.so" \
    "$TOOLS_DIR/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Expected file not found: $f"
        exit 1
    fi
done

echo ""
echo "=== QNN SDK installation complete ==="
echo "  Root:    $TOOLS_DIR/qnn-sdk"
echo "  Headers: $TOOLS_DIR/qnn-sdk/include/QNN/"
echo "  Libs:    $TOOLS_DIR/qnn-sdk/lib/"
