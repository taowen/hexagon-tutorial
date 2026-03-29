#!/usr/bin/env python3
"""
Tokenize text input for BitNet inference on Hexagon DSP.

Encodes the input text using the BitNet tokenizer and writes the
resulting token IDs as a flat int32 binary file that the ARM driver
can load directly.

Usage:
    python tokenize.py "The meaning of life is"
    python tokenize.py --output weights/full_model/input_tokens.bin "Hello world"

The output file is pushed to the device with adb:
    adb push input_tokens.bin /data/local/tmp/bitnet_model/input_tokens.bin
"""

import argparse
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Tokenize text for BitNet inference")
    parser.add_argument("text", nargs="?", default="The meaning of life is",
                        help="Text to tokenize (default: 'The meaning of life is')")
    parser.add_argument("--output", "-o", default="input_tokens.bin",
                        help="Output file path (default: input_tokens.bin)")
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
    tokens = tokenizer.encode(args.text)

    print(f"Input text: {args.text}")
    print(f"Tokens ({len(tokens)}): {tokens}")

    # Decode individual tokens for inspection
    for i, tok in enumerate(tokens):
        piece = tokenizer.decode([tok])
        print(f"  [{i}] {tok} -> {repr(piece)}")

    # Save as int32 array
    arr = np.array(tokens, dtype=np.int32)
    arr.tofile(args.output)
    print(f"\nSaved {len(tokens)} tokens to {args.output}")
    print(f"File size: {arr.nbytes} bytes")
    print(f"\nTo push to device:")
    print(f"  adb push {args.output} /data/local/tmp/bitnet_model/input_tokens.bin")

if __name__ == "__main__":
    main()
