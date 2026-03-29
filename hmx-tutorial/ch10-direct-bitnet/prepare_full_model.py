#!/usr/bin/env python3
"""Export full BitNet-2B model for on-device inference.

Model: microsoft/bitnet-b1.58-2B-4T
  - 30 decoder layers
  - hidden_size=2560, intermediate_size=6912
  - num_heads=20, num_kv_heads=5, head_dim=128
  - Tied embedding/lm_head

Output: weights/full_model/ directory with per-layer binaries,
        embedding, RoPE tables, and metadata.

Usage:
  python3 prepare_full_model.py

Dependencies:
  pip install safetensors torch numpy
"""

import argparse
import glob
import json
import os
import struct
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 2560
INTERMEDIATE_SIZE = 6912
NUM_HEADS = 20
NUM_KV_HEADS = 5
HEAD_DIM = 128
NUM_LAYERS = 30
ROPE_THETA = 500000.0
RMS_NORM_EPS = 1e-5
VOCAB_SIZE = 128256
MAX_SEQ_LEN = 4096

LAYER_SHAPES = {
    "q_proj":    (NUM_HEADS * HEAD_DIM, HIDDEN_SIZE),       # (2560, 2560)
    "k_proj":    (NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE),    # (640, 2560)
    "v_proj":    (NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE),    # (640, 2560)
    "o_proj":    (HIDDEN_SIZE, NUM_HEADS * HEAD_DIM),        # (2560, 2560)
    "gate_proj": (INTERMEDIATE_SIZE, HIDDEN_SIZE),           # (6912, 2560)
    "up_proj":   (INTERMEDIATE_SIZE, HIDDEN_SIZE),           # (6912, 2560)
    "down_proj": (HIDDEN_SIZE, INTERMEDIATE_SIZE),           # (2560, 6912)
}

# HF key suffixes for ternary projections
PROJ_KEY_MAP = {
    "q_proj":    "self_attn.q_proj.weight",
    "k_proj":    "self_attn.k_proj.weight",
    "v_proj":    "self_attn.v_proj.weight",
    "o_proj":    "self_attn.o_proj.weight",
    "gate_proj": "mlp.gate_proj.weight",
    "up_proj":   "mlp.up_proj.weight",
    "down_proj": "mlp.down_proj.weight",
}

# HF key suffixes for weight scales
SCALE_KEY_MAP = {
    "q_proj":    "self_attn.q_proj.weight_scale",
    "k_proj":    "self_attn.k_proj.weight_scale",
    "v_proj":    "self_attn.v_proj.weight_scale",
    "o_proj":    "self_attn.o_proj.weight_scale",
    "gate_proj": "mlp.gate_proj.weight_scale",
    "up_proj":   "mlp.up_proj.weight_scale",
    "down_proj": "mlp.down_proj.weight_scale",
}

# HF key suffixes for RMSNorm weights
NORM_KEY_MAP = {
    "input_layernorm":          "input_layernorm.weight",
    "post_attention_layernorm": "post_attention_layernorm.weight",
    "attn_sub_norm":            "self_attn.attn_sub_norm.weight",
    "ffn_sub_norm":             "mlp.ffn_sub_norm.weight",
}

NORM_DIMS = {
    "input_layernorm":          HIDDEN_SIZE,
    "post_attention_layernorm": HIDDEN_SIZE,
    "attn_sub_norm":            HIDDEN_SIZE,
    "ffn_sub_norm":             INTERMEDIATE_SIZE,
}

# Layer header: 32 uint32 fields = 128 bytes exactly
LAYER_HEADER_SIZE = 128
LAYER_HEADER_MAGIC = 0x4C595200  # "LYR\0"
ALIGN = 128  # HVX alignment


# ---------------------------------------------------------------------------
# Weight unpacking / packing (copied from prepare_weights.py)
# ---------------------------------------------------------------------------
def unpack_hf_ternary(packed: np.ndarray, M: int, K: int) -> np.ndarray:
    """Unpack HuggingFace packed 2-bit ternary weights.

    The HF format packs 4 ternary values per uint8 along the output (M) dimension:
      packed shape: [M // 4, K] uint8
      For position i in [0..3]: val = ((byte >> (2*i)) & 3) - 1
      giving values in {-1, 0, 1}.

    Returns: [M, K] int8 with values in {-1, 0, 1}
    """
    assert packed.shape == (M // 4, K), (
        f"Expected packed shape ({M // 4}, {K}), got {packed.shape}"
    )
    packed_rows = M // 4
    w = np.zeros((M, K), dtype=np.int8)
    for i in range(4):
        w[i * packed_rows : (i + 1) * packed_rows, :] = (
            (packed.astype(np.int16) >> (2 * i)) & 3
        ).astype(np.int8) - 1
    return w


def pack_weights_for_dsp(W_ternary: np.ndarray, M: int, K: int) -> np.ndarray:
    """Pack ternary weights into format for bitnet_gemv_opt.

    W_ternary: numpy array of shape [M, K] with values in {-1, 0, 1}
    Returns: numpy array of shape [2 * Q * M] uint8, where Q = K // 4

    Layout: packed[b * Q * M + q * M + m] = 4-bit nibble
      where b = bit_plane (0 or 1), q = K_group, m = output_dim.

    Encoding: enc = w + 2, so {-1,0,1} -> {1,2,3}
    For each nibble: bit b of enc for 4 consecutive K positions.
    """
    Q = K // 4
    enc = (W_ternary.astype(np.int16) + 2).astype(np.uint8)  # {1, 2, 3}

    packed = np.zeros(2 * Q * M, dtype=np.uint8)
    for b in range(2):  # bit planes
        bits = (enc >> b) & 1  # [M, K] of 0/1
        for q in range(Q):
            # 4 consecutive K values -> 1 nibble per output dim
            nibble = (
                bits[:, q * 4 + 0]
                | (bits[:, q * 4 + 1] << 1)
                | (bits[:, q * 4 + 2] << 2)
                | (bits[:, q * 4 + 3] << 3)
            )
            packed[b * Q * M + q * M : b * Q * M + q * M + M] = nibble.astype(
                np.uint8
            )
    return packed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def align_up(offset: int, alignment: int = ALIGN) -> int:
    return (offset + alignment - 1) & ~(alignment - 1)


def load_safetensors():
    """Load model.safetensors from HF cache."""
    from safetensors import safe_open

    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    sf_files = glob.glob(cache_dir + "/**/model.safetensors", recursive=True)
    # Filter to only include files from the bitnet model directory
    bitnet_files = [f for f in sf_files if "bitnet" in f.lower()]
    if bitnet_files:
        sf_files = bitnet_files

    assert sf_files, (
        "Model not found in HF cache. Run: "
        "huggingface-cli download microsoft/bitnet-b1.58-2B-4T"
    )
    # Use the most recently modified file if multiple matches
    sf_files.sort(key=os.path.getmtime, reverse=True)
    path = sf_files[0]
    print(f"Loading safetensors from: {path}")
    return safe_open(path, framework="pt")


def build_rope_tables():
    """Build cos/sin tables for RoPE.

    Returns: (cos_table, sin_table) each of shape [MAX_SEQ_LEN, HEAD_DIM // 2].
    """
    half_dim = HEAD_DIM // 2  # 64
    freqs = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    positions = np.arange(MAX_SEQ_LEN, dtype=np.float32)
    angles = np.outer(positions, freqs)  # [4096, 64]
    cos_table = np.cos(angles).astype(np.float32)
    sin_table = np.sin(angles).astype(np.float32)
    return cos_table, sin_table


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------
def export_embedding(sf, out_dir: str) -> int:
    """Export embedding table as float16.

    Returns: size in bytes.
    """
    import torch

    print("\n" + "=" * 60)
    print("Exporting embedding (model.embed_tokens.weight)")
    print("=" * 60)

    tensor = sf.get_tensor("model.embed_tokens.weight")
    print(f"  Shape: {tuple(tensor.shape)}, dtype: {tensor.dtype}")
    assert tuple(tensor.shape) == (VOCAB_SIZE, HIDDEN_SIZE), (
        f"Unexpected shape: {tuple(tensor.shape)}"
    )

    # Convert bf16 -> f16
    f16_tensor = tensor.to(torch.float16)
    f16_np = f16_tensor.numpy()

    fpath = os.path.join(out_dir, "embedding.bin")
    f16_np.tofile(fpath)
    fsize = os.path.getsize(fpath)
    print(f"  Saved: embedding.bin ({fsize:,} bytes, {fsize / 1024 / 1024:.1f} MB)")
    return fsize


def export_final_norm(sf, out_dir: str) -> int:
    """Export model.norm.weight as float32.

    Returns: size in bytes.
    """
    import torch

    print("\n" + "=" * 60)
    print("Exporting final norm (model.norm.weight)")
    print("=" * 60)

    tensor = sf.get_tensor("model.norm.weight")
    print(f"  Shape: {tuple(tensor.shape)}, dtype: {tensor.dtype}")
    assert tuple(tensor.shape) == (HIDDEN_SIZE,), (
        f"Unexpected shape: {tuple(tensor.shape)}"
    )

    f32_np = tensor.to(torch.float32).numpy()
    fpath = os.path.join(out_dir, "final_norm.bin")
    f32_np.tofile(fpath)
    fsize = os.path.getsize(fpath)
    print(f"  Saved: final_norm.bin ({fsize:,} bytes)")
    return fsize


def export_rope(out_dir: str) -> tuple:
    """Export precomputed RoPE cos/sin tables.

    Returns: (cos_bytes, sin_bytes).
    """
    print("\n" + "=" * 60)
    print("Exporting RoPE tables")
    print("=" * 60)

    cos_table, sin_table = build_rope_tables()
    half_dim = HEAD_DIM // 2

    # Sanity check
    assert np.allclose(cos_table[0], 1.0), "cos at pos=0 should be all 1.0"
    assert np.allclose(sin_table[0], 0.0, atol=1e-7), "sin at pos=0 should be all 0.0"
    print(f"  Shape: ({MAX_SEQ_LEN}, {half_dim})")
    print(f"  Verified: pos=0 -> cos=1.0, sin=0.0")

    cos_path = os.path.join(out_dir, "rope_cos.bin")
    sin_path = os.path.join(out_dir, "rope_sin.bin")
    cos_table.tofile(cos_path)
    sin_table.tofile(sin_path)

    cos_size = os.path.getsize(cos_path)
    sin_size = os.path.getsize(sin_path)
    print(f"  Saved: rope_cos.bin ({cos_size:,} bytes)")
    print(f"  Saved: rope_sin.bin ({sin_size:,} bytes)")
    return cos_size, sin_size


def export_layer(sf, layer_idx: int, out_dir: str) -> int:
    """Export a single decoder layer as layer_XX.bin.

    Format: LayerHeader (128 bytes) + weight data (128-byte aligned).

    Returns: file size in bytes.
    """
    import torch

    prefix = f"model.layers.{layer_idx}."

    # Collect all data blobs: (name, data_bytes)
    blobs = []

    # --- Ternary projection weights (packed for DSP) ---
    for proj_name, (M, K) in LAYER_SHAPES.items():
        hf_key = prefix + PROJ_KEY_MAP[proj_name]
        packed_hf = sf.get_tensor(hf_key).numpy()

        W = unpack_hf_ternary(packed_hf, M, K)
        packed_dsp = pack_weights_for_dsp(W, M, K)

        # Append weight scale as a float32 right after the packed weights
        scale_key = prefix + SCALE_KEY_MAP[proj_name]
        scale_tensor = sf.get_tensor(scale_key)
        scale_f32 = scale_tensor.to(torch.float32).numpy().flatten()

        # Combine: packed weights + 4 bytes scale
        combined = np.concatenate([packed_dsp, scale_f32.view(np.uint8)])
        blobs.append((proj_name, combined.tobytes()))

    # --- RMSNorm weights (float32) ---
    for norm_name, norm_suffix in NORM_KEY_MAP.items():
        hf_key = prefix + norm_suffix
        tensor = sf.get_tensor(hf_key)
        f32_np = tensor.to(torch.float32).numpy()
        assert f32_np.shape == (NORM_DIMS[norm_name],)
        blobs.append((norm_name, f32_np.tobytes()))

    # --- Compute offsets ---
    # Header is exactly 128 bytes (32 uint32 fields)
    current_offset = LAYER_HEADER_SIZE
    blob_offsets = {}
    blob_sizes = {}

    for name, data in blobs:
        current_offset = align_up(current_offset)
        blob_offsets[name] = current_offset
        blob_sizes[name] = len(data)
        current_offset += len(data)

    total_size = align_up(current_offset)

    # --- Build header (32 uint32 = 128 bytes) ---
    header_fields = [
        LAYER_HEADER_MAGIC,                           # 0: magic
        layer_idx,                                     # 1: layer_idx
        HIDDEN_SIZE,                                   # 2: hidden_size
        INTERMEDIATE_SIZE,                             # 3: intermediate_size
        NUM_HEADS,                                     # 4: num_heads
        NUM_KV_HEADS,                                  # 5: num_kv_heads
        HEAD_DIM,                                      # 6: head_dim
        blob_offsets["q_proj"],                        # 7: q_proj_offset
        blob_sizes["q_proj"],                          # 8: q_proj_size
        blob_offsets["k_proj"],                        # 9: k_proj_offset
        blob_sizes["k_proj"],                          # 10: k_proj_size
        blob_offsets["v_proj"],                        # 11: v_proj_offset
        blob_sizes["v_proj"],                          # 12: v_proj_size
        blob_offsets["o_proj"],                        # 13: o_proj_offset
        blob_sizes["o_proj"],                          # 14: o_proj_size
        blob_offsets["gate_proj"],                     # 15: gate_proj_offset
        blob_sizes["gate_proj"],                       # 16: gate_proj_size
        blob_offsets["up_proj"],                       # 17: up_proj_offset
        blob_sizes["up_proj"],                         # 18: up_proj_size
        blob_offsets["down_proj"],                     # 19: down_proj_offset
        blob_sizes["down_proj"],                       # 20: down_proj_size
        blob_offsets["input_layernorm"],               # 21: input_ln_offset
        blob_sizes["input_layernorm"],                 # 22: input_ln_size
        blob_offsets["post_attention_layernorm"],      # 23: post_attn_ln_offset
        blob_sizes["post_attention_layernorm"],        # 24: post_attn_ln_size
        blob_offsets["attn_sub_norm"],                 # 25: attn_sub_norm_offset
        blob_sizes["attn_sub_norm"],                   # 26: attn_sub_norm_size
        blob_offsets["ffn_sub_norm"],                  # 27: ffn_sub_norm_offset
        blob_sizes["ffn_sub_norm"],                    # 28: ffn_sub_norm_size
        0,                                             # 29: padding
        0,                                             # 30: padding
        0,                                             # 31: padding
    ]
    assert len(header_fields) == 32
    header = struct.pack("<" + "I" * 32, *header_fields)
    assert len(header) == LAYER_HEADER_SIZE

    # --- Write file ---
    fname = f"layer_{layer_idx:02d}.bin"
    fpath = os.path.join(out_dir, fname)

    with open(fpath, "wb") as f:
        # Header
        f.write(header)

        # Data blobs with alignment padding
        for name, data in blobs:
            offset = blob_offsets[name]
            current_pos = f.tell()
            if current_pos < offset:
                f.write(b"\x00" * (offset - current_pos))
            f.write(data)

        # Pad to total_size
        current_pos = f.tell()
        if current_pos < total_size:
            f.write(b"\x00" * (total_size - current_pos))

    actual_size = os.path.getsize(fpath)
    assert actual_size == total_size, (
        f"Layer {layer_idx}: file size mismatch {actual_size} != {total_size}"
    )
    return actual_size


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Export full BitNet-2B model for on-device inference"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: weights/full_model/)"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(script_dir, "weights", "full_model")

    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print(f"Model: microsoft/bitnet-b1.58-2B-4T")
    print(f"Layers: {NUM_LAYERS}, hidden_size: {HIDDEN_SIZE}")
    print()

    t_start = time.time()

    # -----------------------------------------------------------------------
    # Load safetensors
    # -----------------------------------------------------------------------
    sf = load_safetensors()

    # -----------------------------------------------------------------------
    # Export embedding
    # -----------------------------------------------------------------------
    embedding_bytes = export_embedding(sf, out_dir)

    # -----------------------------------------------------------------------
    # Export final norm
    # -----------------------------------------------------------------------
    final_norm_bytes = export_final_norm(sf, out_dir)

    # -----------------------------------------------------------------------
    # Export RoPE tables
    # -----------------------------------------------------------------------
    rope_cos_bytes, rope_sin_bytes = export_rope(out_dir)

    # -----------------------------------------------------------------------
    # Export all 30 layers
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Exporting {NUM_LAYERS} decoder layers")
    print("=" * 60)

    layer_files = []
    layer_sizes = []

    for i in range(NUM_LAYERS):
        t_layer = time.time()
        print(f"\n  Layer {i:2d}/{NUM_LAYERS - 1} ... ", end="", flush=True)
        layer_size = export_layer(sf, i, out_dir)
        layer_files.append(f"layer_{i:02d}.bin")
        layer_sizes.append(layer_size)
        elapsed = time.time() - t_layer
        print(f"done ({layer_size:,} bytes, {layer_size / 1024 / 1024:.1f} MB, {elapsed:.1f}s)")

    # -----------------------------------------------------------------------
    # Write metadata.json
    # -----------------------------------------------------------------------
    total_model_bytes = (
        embedding_bytes
        + final_norm_bytes
        + rope_cos_bytes
        + rope_sin_bytes
        + sum(layer_sizes)
    )

    metadata = {
        "model": "microsoft/bitnet-b1.58-2B-4T",
        "vocab_size": VOCAB_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_heads": NUM_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "num_layers": NUM_LAYERS,
        "max_seq_len": MAX_SEQ_LEN,
        "rope_theta": ROPE_THETA,
        "rms_norm_eps": RMS_NORM_EPS,
        "embedding_dtype": "float16",
        "embedding_shape": [VOCAB_SIZE, HIDDEN_SIZE],
        "embedding_bytes": embedding_bytes,
        "final_norm_bytes": final_norm_bytes,
        "rope_cos_bytes": rope_cos_bytes,
        "rope_sin_bytes": rope_sin_bytes,
        "layer_files": layer_files,
        "layer_sizes": layer_sizes,
        "total_model_bytes": total_model_bytes,
    }

    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    t_total = time.time() - t_start

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  embedding.bin         {embedding_bytes:>14,} bytes  ({embedding_bytes / 1024 / 1024:.1f} MB)")
    print(f"  final_norm.bin        {final_norm_bytes:>14,} bytes")
    print(f"  rope_cos.bin          {rope_cos_bytes:>14,} bytes")
    print(f"  rope_sin.bin          {rope_sin_bytes:>14,} bytes")
    print(f"  30 layer files        {sum(layer_sizes):>14,} bytes  ({sum(layer_sizes) / 1024 / 1024:.1f} MB)")
    for i, (lf, ls) in enumerate(zip(layer_files, layer_sizes)):
        print(f"    {lf}            {ls:>14,} bytes")
    print(f"  metadata.json")
    print(f"  ---")
    print(f"  TOTAL                 {total_model_bytes:>14,} bytes  ({total_model_bytes / 1024 / 1024:.1f} MB)")
    print(f"\n  Time: {t_total:.1f}s")
    print(f"\n  Output: {out_dir}")
    print(f"\nTo push to device:")
    print(f"  adb shell mkdir -p /data/local/tmp/bitnet_full_model")
    print(f"  adb push {out_dir}/ /data/local/tmp/bitnet_full_model/")
    print("Done.")


if __name__ == "__main__":
    main()
