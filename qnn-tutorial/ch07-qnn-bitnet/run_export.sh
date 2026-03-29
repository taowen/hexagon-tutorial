#!/bin/bash
# Export BitNet b1.58-2B-4T to .pte via executorch + QNN backend
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ET_DIR="$SCRIPT_DIR/../../tools/executorch"
QNN_SDK_ROOT=/home/taowen/qnn_sdk/qairt/2.44.0.260225
MODEL_DIR=~/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T/snapshots/04c3b9ad9361b824064a1f25ea60a8be9599b127

cd "$ET_DIR"
source .venv/bin/activate
export PYTHONPATH=$(pwd)/src:$(pwd)
export QNN_SDK_ROOT
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH

# flatc is needed by the export pipeline (built as part of executorch x86 build)
export PATH=$(pwd)/build-x86/third-party/flatbuffers:$PATH

# x86 TMAN op package for graph compilation on host.
# Format: <path_to_so>:<interface_provider>  (NO :HTP suffix on x86!)
TMAN_X86_SO="$ET_DIR/backends/qualcomm/runtime/op_packages/TMANOpPackage/build/x86_64-linux-clang/libQnnTMANOpPackage.so"
if [ ! -f "$TMAN_X86_SO" ]; then
    echo "ERROR: x86 TMAN op package not found at $TMAN_X86_SO"
    echo "Build it first: cd $ET_DIR/backends/qualcomm/runtime/op_packages/TMANOpPackage && make x86_64-linux-clang"
    exit 1
fi
export QNN_OP_PACKAGE_PATHS="${TMAN_X86_SO}:TMANOpPackageInterfaceProvider"

mkdir -p "$SCRIPT_DIR/artifacts"

python3 examples/qualcomm/oss_scripts/bitnet/bitnet.py \
    -m SM8650 \
    -b build-android \
    --llama_model bitnet \
    --model_dir "$MODEL_DIR" \
    --tokenizer_model "$MODEL_DIR/tokenizer.json" \
    --prompt "Hello" \
    --use_tman \
    --ptq 16a4w \
    --compile_only \
    --artifact "$SCRIPT_DIR/artifacts"

echo "Export done. Artifacts in $SCRIPT_DIR/artifacts/"
ls -lh "$SCRIPT_DIR/artifacts/"
