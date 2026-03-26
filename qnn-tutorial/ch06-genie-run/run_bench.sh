#!/bin/bash
# Genie 推理 benchmark
# 用法: ./run_bench.sh [1b|3b]

set -e

MODEL=${1:-1b}
DEST=/data/local/tmp/genie_${MODEL}

if [ "${MODEL}" = "1b" ]; then
    CONFIG=htp-model-config-llama32-1b-gqa.json
elif [ "${MODEL}" = "3b" ]; then
    CONFIG=genie_config.json
else
    echo "未知模型: ${MODEL}"
    exit 1
fi

echo "=== Genie ${MODEL} Benchmark ==="

# 短 prompt
echo ""
echo "--- 短 prompt ---"
adb shell "cd ${DEST} && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  ./genie-t2t-run -c ${CONFIG} --profile perf_short.txt \
  -p 'What is the capital of France?' 2>&1"
echo ""
echo "Profile:"
adb shell "cat ${DEST}/perf_short.txt"

# 长 prompt (~128 tokens)
echo ""
echo "--- 长 prompt (~128 tokens) ---"
PROMPT='<|begin_of_text|><|start_header_id|>user<|end_header_id|>

The following is a detailed question about computer science. In the field of machine learning, neural networks have become increasingly important. Transformers, introduced in the paper Attention Is All You Need by Vaswani et al, revolutionized NLP by replacing recurrent architectures with self-attention mechanisms. The key innovation was processing all positions simultaneously, enabling better parallelization. Since then, transformer models like BERT, GPT, and successors achieved remarkable results. What are the main challenges when deploying large transformer models on mobile devices?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
'

# Push prompt to file (shell escaping is tricky for long prompts)
echo "${PROMPT}" > /tmp/genie_long_prompt.txt
adb push /tmp/genie_long_prompt.txt ${DEST}/long_prompt.txt

adb shell "cd ${DEST} && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  ./genie-t2t-run -c ${CONFIG} --prompt_file long_prompt.txt --profile perf_long.txt 2>&1"
echo ""
echo "Profile:"
adb shell "cat ${DEST}/perf_long.txt"
