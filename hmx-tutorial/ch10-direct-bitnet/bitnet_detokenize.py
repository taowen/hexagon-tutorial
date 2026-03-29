#!/usr/bin/env python3
"""
Detokenize output tokens from BitNet inference on Hexagon DSP.

Reads a flat int32 binary file of token IDs (produced by the ARM
driver) and decodes them back to text using the BitNet tokenizer.

Usage:
    # First pull from device:
    adb pull /data/local/tmp/bitnet_model/output_tokens.bin .

    python detokenize.py output_tokens.bin
    python detokenize.py --input-tokens input_tokens.bin output_tokens.bin
"""

import argparse
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Detokenize BitNet inference output")
    parser.add_argument("output_file", nargs="?", default="output_tokens.bin",
                        help="Output tokens file (default: output_tokens.bin)")
    parser.add_argument("--input-tokens", "-i", default=None,
                        help="Also decode the input tokens file for context")
    parser.add_argument("--model", default="microsoft/bitnet-b1.58-2B-4T",
                        help="HuggingFace model name for tokenizer")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: transformers package not installed.", file=sys.stderr)
        print("Install with: pip install transformers", file=sys.stderr)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Decode input tokens if provided
    if args.input_tokens:
        input_tokens = np.fromfile(args.input_tokens, dtype=np.int32).tolist()
        input_text = tokenizer.decode(input_tokens)
        print(f"Input tokens ({len(input_tokens)}): {input_tokens}")
        print(f"Input text: {input_text}")
        print()

    # Decode output tokens
    output_tokens = np.fromfile(args.output_file, dtype=np.int32).tolist()
    output_text = tokenizer.decode(output_tokens)

    print(f"Output tokens ({len(output_tokens)}): {output_tokens}")
    print(f"Output text: {output_text}")

    # Show individual tokens
    print()
    print("Token breakdown:")
    for i, tok in enumerate(output_tokens):
        piece = tokenizer.decode([tok])
        print(f"  [{i}] {tok} -> {repr(piece)}")

    # If both input and output are available, show the full sequence
    if args.input_tokens:
        input_tokens = np.fromfile(args.input_tokens, dtype=np.int32).tolist()
        full_tokens = input_tokens + output_tokens
        full_text = tokenizer.decode(full_tokens)
        print()
        print(f"Full sequence ({len(full_tokens)} tokens):")
        print(full_text)

if __name__ == "__main__":
    main()
