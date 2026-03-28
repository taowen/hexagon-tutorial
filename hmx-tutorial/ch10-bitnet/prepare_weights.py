#!/usr/bin/env python3
"""
prepare_weights.py -- Download BitNet model, extract one decoder layer's weights,
pack them into DSP format, and generate reference output for verification.

Model: microsoft/bitnet-b1.58-2B-4T
  hidden_size=2560, intermediate_size=6912
  num_heads=20, num_kv_heads=5, head_dim=128
  30 decoder layers, RoPE theta=500000.0
  Activation: ReLU-squared (not SiLU)
  Has attn_sub_norm and ffn_sub_norm (extra RMSNorms)

Usage:
  python3 prepare_weights.py [--layer N]

Dependencies:
  pip install safetensors torch numpy huggingface_hub
"""

import argparse
import json
import math
import os
import struct
import sys

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

# Derived
GQA_RATIO = NUM_HEADS // NUM_KV_HEADS  # 4
Q_DIM = NUM_HEADS * HEAD_DIM            # 2560
K_DIM = NUM_KV_HEADS * HEAD_DIM         # 640
V_DIM = NUM_KV_HEADS * HEAD_DIM         # 640
O_DIM = HIDDEN_SIZE                      # 2560

# Linear layer shapes: (out_features, in_features) = (M, K) in GEMV terms
LAYER_SHAPES = {
    "q_proj":    (Q_DIM, HIDDEN_SIZE),      # (2560, 2560)
    "k_proj":    (K_DIM, HIDDEN_SIZE),      # (640, 2560)
    "v_proj":    (V_DIM, HIDDEN_SIZE),      # (640, 2560)
    "o_proj":    (O_DIM, NUM_HEADS * HEAD_DIM),  # (2560, 2560)
    "gate_proj": (INTERMEDIATE_SIZE, HIDDEN_SIZE),  # (6912, 2560)
    "up_proj":   (INTERMEDIATE_SIZE, HIDDEN_SIZE),  # (6912, 2560)
    "down_proj": (HIDDEN_SIZE, INTERMEDIATE_SIZE),  # (2560, 6912)
}


# ---------------------------------------------------------------------------
# HuggingFace packed format: unpack 2-bit ternary
# ---------------------------------------------------------------------------
def unpack_hf_ternary(packed: np.ndarray, M: int, K: int) -> np.ndarray:
    """Unpack HuggingFace packed 2-bit ternary weights.

    The HF format packs 4 ternary values per uint8 along the output (M) dimension:
      packed shape: [M // 4, K] uint8
      For position i in [0..3]: val = ((byte >> (2*i)) & 3) - 1
      giving values in {-1, 0, 1}.

    The 4 values from each byte go to BLOCK positions (matching executorch's
    unpack_weights): bits 0-1 -> rows [0:M/4], bits 2-3 -> rows [M/4:M/2], etc.

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


# ---------------------------------------------------------------------------
# DSP weight packing (matches bitnet_pack_weights in bitnet_gemv.h)
# ---------------------------------------------------------------------------
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
# Reference computation helpers (pure numpy, matching DSP ops)
# ---------------------------------------------------------------------------
def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = RMS_NORM_EPS) -> np.ndarray:
    """RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight"""
    rms = np.sqrt(np.mean(x * x) + eps)
    return x / rms * weight


def relu_squared(x: np.ndarray) -> np.ndarray:
    """ReLU squared: max(0, x)^2"""
    r = np.maximum(0.0, x)
    return r * r


def rope_apply(x: np.ndarray, cos_vals: np.ndarray, sin_vals: np.ndarray,
               head_dim: int) -> np.ndarray:
    """Apply RoPE with two-half split convention (matching hvx_rope_f32).

    x: [head_dim]
    cos_vals, sin_vals: [head_dim // 2]
    Returns: [head_dim]
    """
    half = head_dim // 2
    x_r = x[:half]
    x_i = x[half:]
    out = np.zeros_like(x)
    out[:half] = x_r * cos_vals - x_i * sin_vals
    out[half:] = x_r * sin_vals + x_i * cos_vals
    return out


def ternary_matmul(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Matrix-vector multiply with ternary weights: y = W @ x.

    W: [M, K] int8 ternary
    x: [K] float32
    Returns: [M] float32
    """
    return W.astype(np.float32) @ x


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


# ---------------------------------------------------------------------------
# Reference: run one decoder layer
# ---------------------------------------------------------------------------
def run_decoder_layer(
    x: np.ndarray,           # [hidden_size] float32
    weights: dict,           # all layer weights (ternary int8 and f32 norms)
    pos: int,                # sequence position
    cos_table: np.ndarray,   # [max_pos, head_dim/2]
    sin_table: np.ndarray,   # [max_pos, head_dim/2]
) -> np.ndarray:
    """Run one BitNet decoder layer (seq_len=1 decode mode).

    Architecture:
      1. input_layernorm(x)
      2. Q/K/V projections (ternary matmuls)
      3. RoPE on Q and K
      4. Attention (seq_len=1: softmax of single score = 1.0, so attn_out = V)
      5. attn_sub_norm
      6. O projection
      7. residual = x + attn_output
      8. post_attention_layernorm(residual)
      9. gate/up projections
      10. ReLU-squared * up (gating)
      11. ffn_sub_norm
      12. down projection
      13. output = residual + ffn_output
    """
    hidden = HIDDEN_SIZE
    scale = 1.0 / math.sqrt(HEAD_DIM)

    # --- Attention block ---
    # Step 1: Input LN
    normed = rmsnorm(x, weights["input_layernorm"])

    # Step 2: Q/K/V projections
    q_all = ternary_matmul(weights["q_proj"], normed)  # [2560]
    k_all = ternary_matmul(weights["k_proj"], normed)  # [640]
    v_all = ternary_matmul(weights["v_proj"], normed)  # [640]

    # Step 3: RoPE on Q and K (per head)
    cos_vals = cos_table[pos]  # [head_dim/2]
    sin_vals = sin_table[pos]  # [head_dim/2]

    q_rope = np.zeros_like(q_all)
    for h in range(NUM_HEADS):
        q_rope[h * HEAD_DIM : (h + 1) * HEAD_DIM] = rope_apply(
            q_all[h * HEAD_DIM : (h + 1) * HEAD_DIM], cos_vals, sin_vals, HEAD_DIM
        )
    k_rope = np.zeros_like(k_all)
    for h in range(NUM_KV_HEADS):
        k_rope[h * HEAD_DIM : (h + 1) * HEAD_DIM] = rope_apply(
            k_all[h * HEAD_DIM : (h + 1) * HEAD_DIM], cos_vals, sin_vals, HEAD_DIM
        )

    # Step 4: Attention (seq_len=1 decode: softmax of single score = 1.0)
    # For each query head, the attention output is simply the corresponding V head.
    # With GQA, multiple Q heads share one KV head.
    attn_out = np.zeros(NUM_HEADS * HEAD_DIM, dtype=np.float32)
    for h in range(NUM_HEADS):
        kv_h = h // GQA_RATIO
        # With seq_len=1, attention is trivial: output = V
        attn_out[h * HEAD_DIM : (h + 1) * HEAD_DIM] = (
            v_all[kv_h * HEAD_DIM : (kv_h + 1) * HEAD_DIM]
        )

    # Step 5: attn_sub_norm
    attn_normed = rmsnorm(attn_out, weights["attn_sub_norm"])

    # Step 6: O projection
    attn_proj = ternary_matmul(weights["o_proj"], attn_normed)

    # Step 7: Residual
    residual = x + attn_proj

    # --- FFN block ---
    # Step 8: Post-attention LN
    ffn_normed = rmsnorm(residual, weights["post_attention_layernorm"])

    # Step 9: Gate and Up projections
    gate = ternary_matmul(weights["gate_proj"], ffn_normed)  # [6912]
    up = ternary_matmul(weights["up_proj"], ffn_normed)      # [6912]

    # Step 10: ReLU-squared gating
    ffn_hidden = relu_squared(gate) * up

    # Step 11: ffn_sub_norm
    ffn_sub_normed = rmsnorm(ffn_hidden, weights["ffn_sub_norm"])

    # Step 12: Down projection
    ffn_out = ternary_matmul(weights["down_proj"], ffn_sub_normed)

    # Step 13: Final residual
    output = residual + ffn_out

    return output


# ---------------------------------------------------------------------------
# RoPE table generation
# ---------------------------------------------------------------------------
def build_rope_tables(max_pos: int, head_dim: int, theta: float = ROPE_THETA):
    """Build cos/sin tables for RoPE.

    Returns: (cos_table, sin_table) each of shape [max_pos, head_dim // 2].
    Convention matches hvx_rope_f32: freq_i = 1 / (theta^(2i / head_dim)).
    """
    half = head_dim // 2
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    # freqs shape: [half]

    positions = np.arange(max_pos, dtype=np.float64)  # [max_pos]
    angles = np.outer(positions, freqs)  # [max_pos, half]

    cos_table = np.cos(angles).astype(np.float32)
    sin_table = np.sin(angles).astype(np.float32)
    return cos_table, sin_table


# ---------------------------------------------------------------------------
# Download model files
# ---------------------------------------------------------------------------
def download_model_files(layer: int):
    """Download the model safetensors and extract weights for the given layer.

    Handles both single-file (model.safetensors) and sharded models.
    Uses safetensors metadata to selectively load only needed tensors.

    Returns a dict of weight_name -> numpy array for the requested layer.
    """
    from huggingface_hub import hf_hub_download, HfApi
    from safetensors import safe_open

    repo_id = "microsoft/bitnet-b1.58-2B-4T"
    layer_prefix = f"model.layers.{layer}."

    # Check if the model is sharded or a single file
    api = HfApi()
    repo_files = api.list_repo_files(repo_id)
    has_index = "model.safetensors.index.json" in repo_files
    has_single = "model.safetensors" in repo_files

    if has_index:
        # Sharded model: find which shard(s) to download
        index_path = hf_hub_download(repo_id, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        needed_files = set()
        needed_keys = []
        for key, shard_file in weight_map.items():
            if key.startswith(layer_prefix):
                needed_files.add(shard_file)
                needed_keys.append(key)
        shard_files = sorted(needed_files)
    elif has_single:
        # Single file model
        shard_files = ["model.safetensors"]
        needed_keys = None  # will discover from file
    else:
        print("ERROR: No safetensors files found in the repo.")
        sys.exit(1)

    print(f"Model file(s) to load: {shard_files}")

    all_weights = {}
    for shard_file in shard_files:
        print(f"\nDownloading {shard_file}...")
        local_path = hf_hub_download(repo_id, shard_file)
        print(f"  Loading tensors from {local_path}...")

        # Use safe_open with torch framework to handle bfloat16, then convert
        with safe_open(local_path, framework="pt") as f:
            keys_in_file = f.keys()
            layer_keys = [k for k in keys_in_file if k.startswith(layer_prefix)]

            if not layer_keys:
                print(f"  WARNING: No layer {layer} keys found in {shard_file}")
                continue

            print(f"  Found {len(layer_keys)} keys for layer {layer}:")
            for key in sorted(layer_keys):
                tensor = f.get_tensor(key)
                print(f"    {key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
                # Convert to numpy: bfloat16 -> float32 -> numpy
                if tensor.dtype == __import__('torch').bfloat16:
                    all_weights[key] = tensor.float().numpy()
                else:
                    all_weights[key] = tensor.numpy()

    if not all_weights:
        print(f"ERROR: No weights found for layer {layer}.")
        sys.exit(1)

    return all_weights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prepare BitNet weights for DSP execution"
    )
    parser.add_argument("--layer", type=int, default=0,
                        help="Decoder layer to export (default: 0)")
    parser.add_argument("--rope-positions", type=int, default=11,
                        help="Number of RoPE positions to precompute (default: 11)")
    args = parser.parse_args()

    layer = args.layer
    max_pos = args.rope_positions

    if layer < 0 or layer >= NUM_LAYERS:
        print(f"ERROR: Layer must be in [0, {NUM_LAYERS - 1}]")
        sys.exit(1)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print(f"Exporting layer {layer}...\n")

    # -----------------------------------------------------------------------
    # Step 1: Download and load weights
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Downloading model weights from HuggingFace")
    print("=" * 60)
    raw_weights = download_model_files(layer)

    prefix = f"model.layers.{layer}."

    # -----------------------------------------------------------------------
    # Step 2: Unpack ternary weights and repack for DSP
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Unpacking ternary weights and repacking for DSP")
    print("=" * 60)

    # Map from our short names to HF weight key suffixes
    proj_key_map = {
        "q_proj":    "self_attn.q_proj.weight",
        "k_proj":    "self_attn.k_proj.weight",
        "v_proj":    "self_attn.v_proj.weight",
        "o_proj":    "self_attn.o_proj.weight",
        "gate_proj": "mlp.gate_proj.weight",
        "up_proj":   "mlp.up_proj.weight",
        "down_proj": "mlp.down_proj.weight",
    }

    ternary_weights = {}  # name -> [M, K] int8 arrays
    packed_weights = {}   # name -> packed uint8 arrays
    metadata = {"layer": layer, "files": {}}

    for name, (M, K) in LAYER_SHAPES.items():
        hf_key = prefix + proj_key_map[name]
        packed_hf = raw_weights[hf_key]

        print(f"\n{name}: HF shape={packed_hf.shape}, expected M={M}, K={K}")

        # Unpack from HF 2-bit format
        W = unpack_hf_ternary(packed_hf, M, K)
        ternary_weights[name] = W

        # Verify ternary values
        unique_vals = np.unique(W)
        assert all(v in [-1, 0, 1] for v in unique_vals), (
            f"{name}: unexpected values {unique_vals}"
        )
        counts = {v: int(np.sum(W == v)) for v in [-1, 0, 1]}
        total = M * K
        print(f"  Ternary distribution: -1={counts[-1]} ({100*counts[-1]/total:.1f}%), "
              f"0={counts[0]} ({100*counts[0]/total:.1f}%), "
              f"+1={counts[1]} ({100*counts[1]/total:.1f}%)")

        # Pack for DSP
        Q = K // 4
        packed = pack_weights_for_dsp(W, M, K)
        packed_weights[name] = packed

        # Save packed weights
        fname = f"layer{layer}_{name}.bin"
        fpath = os.path.join(out_dir, fname)
        packed.tofile(fpath)
        fsize = os.path.getsize(fpath)
        print(f"  Packed: {2 * Q * M} bytes -> {fname} ({fsize} bytes)")
        metadata["files"][fname] = {
            "type": "packed_weights",
            "M": M, "K": K, "Q": Q,
            "size_bytes": fsize,
            "format": "packed_w[b][q][m], b=2 bit planes, q=K/4 groups, m=output_dim",
        }

    # -----------------------------------------------------------------------
    # Step 3: Save RMSNorm weights as f32
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Saving RMSNorm weights")
    print("=" * 60)

    norm_key_map = {
        "input_layernorm":         ("input_layernorm.weight",         HIDDEN_SIZE),
        "post_attention_layernorm": ("post_attention_layernorm.weight", HIDDEN_SIZE),
        "attn_sub_norm":           ("self_attn.attn_sub_norm.weight", HIDDEN_SIZE),
        "ffn_sub_norm":            ("mlp.ffn_sub_norm.weight",        INTERMEDIATE_SIZE),
    }

    norm_fname_map = {
        "input_layernorm":         f"layer{layer}_input_ln.bin",
        "post_attention_layernorm": f"layer{layer}_post_attn_ln.bin",
        "attn_sub_norm":           f"layer{layer}_attn_sub_norm.bin",
        "ffn_sub_norm":            f"layer{layer}_ffn_sub_norm.bin",
    }

    norm_weights = {}
    for name, (suffix, dim) in norm_key_map.items():
        hf_key = prefix + suffix
        w = raw_weights[hf_key].astype(np.float32)
        assert w.shape == (dim,), f"{name}: expected shape ({dim},), got {w.shape}"
        norm_weights[name] = w

        fname = norm_fname_map[name]
        fpath = os.path.join(out_dir, fname)
        w.tofile(fpath)
        fsize = os.path.getsize(fpath)
        print(f"  {name}: shape={w.shape} -> {fname} ({fsize} bytes)")
        metadata["files"][fname] = {
            "type": "rmsnorm_weight",
            "dim": dim,
            "dtype": "float32",
            "size_bytes": fsize,
        }

    # -----------------------------------------------------------------------
    # Step 4: Compute reference output
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Computing reference output")
    print("=" * 60)

    # Build RoPE tables
    cos_table, sin_table = build_rope_tables(max_pos, HEAD_DIM, ROPE_THETA)

    # Generate deterministic input
    rng = np.random.RandomState(42)
    test_input = rng.randn(HIDDEN_SIZE).astype(np.float32)

    # Gather all weights needed for the decoder layer
    layer_weights = {}
    layer_weights.update(ternary_weights)      # q_proj, k_proj, etc.
    layer_weights.update(norm_weights)          # input_layernorm, etc.

    print(f"  Test input: shape={test_input.shape}, "
          f"mean={test_input.mean():.4f}, std={test_input.std():.4f}")

    # Run reference computation
    pos = 0  # position 0 for test
    test_output = run_decoder_layer(test_input, layer_weights, pos,
                                    cos_table, sin_table)
    print(f"  Test output: shape={test_output.shape}, "
          f"mean={test_output.mean():.4f}, std={test_output.std():.4f}")
    print(f"  Output first 8: {test_output[:8]}")
    print(f"  Output last 8:  {test_output[-8:]}")

    # Also compute intermediate results for debugging
    normed = rmsnorm(test_input, norm_weights["input_layernorm"])
    q_ref = ternary_matmul(ternary_weights["q_proj"], normed)
    print(f"  Q proj first 8: {q_ref[:8]}")

    # Save test input
    fname = "test_input.bin"
    fpath = os.path.join(out_dir, fname)
    test_input.tofile(fpath)
    fsize = os.path.getsize(fpath)
    print(f"\n  Saved {fname}: {fsize} bytes")
    metadata["files"][fname] = {
        "type": "test_input",
        "dim": HIDDEN_SIZE,
        "dtype": "float32",
        "size_bytes": fsize,
        "seed": 42,
    }

    # Save test output
    fname = "test_output.bin"
    fpath = os.path.join(out_dir, fname)
    test_output.tofile(fpath)
    fsize = os.path.getsize(fpath)
    print(f"  Saved {fname}: {fsize} bytes")
    metadata["files"][fname] = {
        "type": "test_output",
        "dim": HIDDEN_SIZE,
        "dtype": "float32",
        "size_bytes": fsize,
        "position": pos,
        "note": "reference output from PyTorch/numpy for one decoder layer",
    }

    # -----------------------------------------------------------------------
    # Step 5: Save RoPE tables
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Saving RoPE tables")
    print("=" * 60)

    half_dim = HEAD_DIM // 2

    fname = "rope_cos.bin"
    fpath = os.path.join(out_dir, fname)
    cos_table.tofile(fpath)
    fsize = os.path.getsize(fpath)
    print(f"  {fname}: shape=({max_pos}, {half_dim}) -> {fsize} bytes")
    metadata["files"][fname] = {
        "type": "rope_table",
        "component": "cos",
        "shape": [max_pos, half_dim],
        "dtype": "float32",
        "size_bytes": fsize,
        "theta": ROPE_THETA,
        "note": "cos_table[pos * half_dim + i], two-half split convention",
    }

    fname = "rope_sin.bin"
    fpath = os.path.join(out_dir, fname)
    sin_table.tofile(fpath)
    fsize = os.path.getsize(fpath)
    print(f"  {fname}: shape=({max_pos}, {half_dim}) -> {fsize} bytes")
    metadata["files"][fname] = {
        "type": "rope_table",
        "component": "sin",
        "shape": [max_pos, half_dim],
        "dtype": "float32",
        "size_bytes": fsize,
        "theta": ROPE_THETA,
    }

    # Sanity check: at pos=0, cos should be all 1.0, sin should be all 0.0
    assert np.allclose(cos_table[0], 1.0), "cos at pos=0 should be all 1.0"
    assert np.allclose(sin_table[0], 0.0, atol=1e-7), "sin at pos=0 should be all 0.0"
    print(f"  Verified: pos=0 -> cos=1.0, sin=0.0")

    # -----------------------------------------------------------------------
    # Save metadata
    # -----------------------------------------------------------------------
    metadata["model"] = "microsoft/bitnet-b1.58-2B-4T"
    metadata["config"] = {
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_heads": NUM_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "num_layers": NUM_LAYERS,
        "rope_theta": ROPE_THETA,
        "rms_norm_eps": RMS_NORM_EPS,
        "activation": "relu_squared",
        "gqa_ratio": GQA_RATIO,
    }
    metadata["rope_positions"] = max_pos
    metadata["test_position"] = pos

    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Saved metadata.json")

    # -----------------------------------------------------------------------
    # Step 6: Assemble decoder_layer.bin (WeightLayout header + all data)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 6: Assembling decoder_layer.bin")
    print("=" * 60)

    # The WeightLayout struct (must match protocol.h exactly):
    #   uint32_t total_size
    #   uint32_t input_offset
    #   uint32_t ref_output_offset
    #   uint32_t pos
    #   uint32_t q_proj_offset, k_proj_offset, v_proj_offset, o_proj_offset
    #   uint32_t gate_proj_offset, up_proj_offset, down_proj_offset
    #   uint32_t input_ln_offset, post_attn_ln_offset, attn_sub_norm_offset, ffn_sub_norm_offset
    #   uint32_t rope_cos_offset, rope_sin_offset, rope_max_pos
    #   uint32_t k_cache_offset, v_cache_offset, kv_seq_len, kv_max_seq_len
    # Total: 22 uint32 fields = 88 bytes

    ALIGN = 128  # HVX alignment

    def align_up(offset):
        return (offset + ALIGN - 1) & ~(ALIGN - 1)

    # Reserve space for header (round up to 128 bytes)
    header_size = 22 * 4  # 88 bytes
    header_padded = align_up(header_size)

    # Build the data section: concatenate all payloads with 128-byte alignment
    data_blobs = []  # list of (name, offset, bytes)
    current_offset = header_padded

    def add_blob(name, arr):
        nonlocal current_offset
        raw = arr.tobytes()
        offset = align_up(current_offset)
        data_blobs.append((name, offset, raw))
        current_offset = offset + len(raw)
        return offset

    offsets = {}

    # Test input/output
    offsets["input"] = add_blob("test_input", test_input)
    offsets["ref_output"] = add_blob("test_output", test_output)

    # Packed projection weights
    for name in ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]:
        offsets[name] = add_blob(name, packed_weights[name])

    # RMSNorm weights
    offsets["input_ln"] = add_blob("input_ln", norm_weights["input_layernorm"])
    offsets["post_attn_ln"] = add_blob("post_attn_ln", norm_weights["post_attention_layernorm"])
    offsets["attn_sub_norm"] = add_blob("attn_sub_norm", norm_weights["attn_sub_norm"])
    offsets["ffn_sub_norm"] = add_blob("ffn_sub_norm", norm_weights["ffn_sub_norm"])

    # RoPE tables (flattened: [max_pos * half_dim])
    offsets["rope_cos"] = add_blob("rope_cos", cos_table.ravel())
    offsets["rope_sin"] = add_blob("rope_sin", sin_table.ravel())

    # KV cache (zero-filled, for seq_len=1 decode at pos=0)
    kv_max_seq = 4  # small for testing
    kv_cache_size = NUM_KV_HEADS * kv_max_seq * HEAD_DIM
    k_cache = np.zeros(kv_cache_size, dtype=np.float32)
    v_cache = np.zeros(kv_cache_size, dtype=np.float32)
    offsets["k_cache"] = add_blob("k_cache", k_cache)
    offsets["v_cache"] = add_blob("v_cache", v_cache)

    total_size = align_up(current_offset)

    # Build the header
    header = struct.pack(
        "<" + "I" * 22,
        total_size,                    # total_size
        offsets["input"],              # input_offset
        offsets["ref_output"],         # ref_output_offset
        pos,                           # pos
        offsets["q_proj"],             # q_proj_offset
        offsets["k_proj"],             # k_proj_offset
        offsets["v_proj"],             # v_proj_offset
        offsets["o_proj"],             # o_proj_offset
        offsets["gate_proj"],          # gate_proj_offset
        offsets["up_proj"],            # up_proj_offset
        offsets["down_proj"],          # down_proj_offset
        offsets["input_ln"],           # input_ln_offset
        offsets["post_attn_ln"],       # post_attn_ln_offset
        offsets["attn_sub_norm"],      # attn_sub_norm_offset
        offsets["ffn_sub_norm"],       # ffn_sub_norm_offset
        offsets["rope_cos"],           # rope_cos_offset
        offsets["rope_sin"],           # rope_sin_offset
        max_pos,                       # rope_max_pos
        offsets["k_cache"],            # k_cache_offset
        offsets["v_cache"],            # v_cache_offset
        0,                             # kv_seq_len (empty at start)
        kv_max_seq,                    # kv_max_seq_len
    )

    # Write the binary file
    bin_path = os.path.join(out_dir, "decoder_layer.bin")
    with open(bin_path, "wb") as f:
        # Header (padded to 128 bytes)
        f.write(header)
        f.write(b'\x00' * (header_padded - len(header)))

        # Data blobs
        for name, offset, raw in data_blobs:
            # Pad to reach the expected offset
            current_pos = f.tell()
            if current_pos < offset:
                f.write(b'\x00' * (offset - current_pos))
            f.write(raw)

        # Pad to total_size
        current_pos = f.tell()
        if current_pos < total_size:
            f.write(b'\x00' * (total_size - current_pos))

    actual_size = os.path.getsize(bin_path)
    print(f"  decoder_layer.bin: {actual_size:,} bytes ({actual_size / 1024 / 1024:.1f} MB)")
    print(f"  Header: {header_padded} bytes, data start: {header_padded}")
    for name, offset, raw in data_blobs:
        print(f"    {name:20s}: offset={offset:>10,}, size={len(raw):>12,}")

    assert actual_size == total_size, (
        f"File size mismatch: {actual_size} != {total_size}"
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    total_bytes = 0
    for fname, info in sorted(metadata["files"].items()):
        sz = info["size_bytes"]
        total_bytes += sz
        print(f"  {fname:40s}  {sz:>12,} bytes")

    print(f"  {'TOTAL (individual)':40s}  {total_bytes:>12,} bytes ({total_bytes / 1024 / 1024:.1f} MB)")
    print(f"  {'decoder_layer.bin':40s}  {actual_size:>12,} bytes ({actual_size / 1024 / 1024:.1f} MB)")
    print(f"\nAll files saved to: {out_dir}")
    print(f"\nTo push to device:")
    print(f"  adb shell mkdir -p /data/local/tmp/bitnet_weights")
    print(f"  adb push {bin_path} /data/local/tmp/bitnet_weights/")
    print("Done.")


if __name__ == "__main__":
    main()
