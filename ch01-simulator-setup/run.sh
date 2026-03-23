#!/usr/bin/env bash
#
# run.sh — 编译并运行第一章示例
#
# 用法:
#   bash ch01-simulator-setup/run.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- 工具路径 ----
TOOLS_BIN="$ROOT_DIR/tools/hexagon-sdk/tools/HEXAGON_Tools/19.0.04/Tools/bin"
H2_INSTALL="$ROOT_DIR/tools/h2-install"
H2_KERNEL="$ROOT_DIR/tools/hexagon-hypervisor/kernel/include"

# 如果 h2-install 是符号链接, 解析出 hypervisor 根目录来找 kernel/include
if [ -L "$H2_INSTALL" ]; then
    H2_ROOT="$(dirname "$(readlink -f "$H2_INSTALL")")"
    H2_KERNEL="$H2_ROOT/kernel/include"
fi

CLANG="$TOOLS_BIN/hexagon-clang"
SIM="$TOOLS_BIN/hexagon-sim"
BOOTER="$H2_INSTALL/bin/booter"

# 检查工具是否存在
for f in "$CLANG" "$SIM" "$BOOTER"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found. Run install_tools.sh first."
        exit 1
    fi
done

# ---- 编译 ----
SRC="$SCRIPT_DIR/test_hvx_hmx.c"
OUT="$SCRIPT_DIR/test_hvx_hmx"

echo "=== Compiling ==="
echo "  $CLANG -O2 -mv73 -mhvx -mhvx-length=128B -mhmx ..."
$CLANG -O2 -mv73 \
    -mhvx -mhvx-length=128B \
    -mhmx \
    -DARCHV=73 \
    -I "$H2_INSTALL/include" \
    -I "$H2_KERNEL" \
    -moslib=h2 \
    -Wl,-L,"$H2_INSTALL/lib" \
    -Wl,--section-start=.start=0x02000000 \
    -o "$OUT" "$SRC"
echo "  -> $OUT"

# ---- 运行 ----
echo ""
echo "=== Running on hexagon-sim ==="
echo "  $SIM --mv73 --mhmx 1 -- booter ... test_hvx_hmx"
echo ""
$SIM --mv73 --mhmx 1 --simulated_returnval \
    -- "$BOOTER" \
    --ext_power 1 \
    --use_ext 1 \
    --fence_hi 0xfe000000 \
    "$OUT"
