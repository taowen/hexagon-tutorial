# Chapter 9: Local-Memory Training — Eliminating rpcmem Write Penalty

## Goal

ch08's HVX training operates on DDR shared memory (rpcmem). The ARM side writes batch data to this shared memory, which is ~23x slower than regular malloc (see ch08 findings). Can we reduce this overhead by mirroring all data to DSP-local memory?

## Background

In ch08, network buffers (weights, activations, gradients) live in rpcmem shared memory so both ARM and DSP can access them. The DSP reads/writes these via L2 cache. Two problems:

1. **rpcmem write penalty**: ARM writing to shared memory is 23x slower than malloc
2. **L2 cache pressure**: Training's ~1.9 MB working set competes for L2 cache space

The idea: allocate a mirror of all buffers in DSP-local memory. Copy weights once at init, copy each batch's input from DDR, compute entirely on local pointers, and only copy weights back for ARM evaluation.

### VTCM vs DSP Heap

Hexagon has 8 MB of VTCM (Vector Tightly-Coupled Memory) with 1-cycle access latency. Our 1.9 MB working set fits easily.

**However, in practice VTCM has reliability issues in the dspqueue callback context.** We observed non-deterministic training divergence when using VTCM — the corruption point varied between runs (epoch 2-5), suggesting VTCM is being shared or reclaimed by other system services on the live Android device.

The DSP heap (`memalign`) provides a reliable alternative. It goes through L2 cache like DDR, but avoids the rpcmem write penalty since it's DSP-private memory.

## Architecture

### ARM Side: Unchanged

The ch08 `train_fused` binary is reused. We just swap the DSP skel `.so`.

### Data Flow

```
OP_REGISTER_NET:
  1. Allocate local memory (DSP heap or VTCM)
  2. Bump-allocate all buffers (128B aligned)
  3. memcpy weights DDR → local (one-time)

OP_TRAIN_BATCH:
  1. memcpy input DDR → local (~400KB per batch)
  2. Forward/backward/SGD on local pointers
  3. Weights updated in-place (no DDR write-back)
  4. Return loss + accuracy

OP_SYNC:
  1. memcpy weights local → DDR (for ARM eval)
  2. dspqueue flush
```

### Memory Budget (batch=128, f32)

| Buffer | Shape | Size |
|--------|-------|------|
| W1, dW1 | 128×800 ×2 | 800 KB |
| W2, dW2 | 32×128 ×2 | 32 KB |
| B1, B2 | 128 + 32 | 0.6 KB |
| hidden + hidden_pre | 128×128 ×2 | 128 KB |
| logits + probs | 128×32 ×2 | 32 KB |
| dlogits + dhidden | 128×32 + 128×128 | 80 KB |
| input (per batch) | 128×800 | 400 KB |
| **Total** | | **~1.5 MB** |

## Results

| Configuration | Epoch (ms) | DSP time (us) | Overhead |
|--------------|-----------|---------------|----------|
| ch08 DDR baseline | ~1840 | 7,420,842 | — |
| ch09 DSP heap mirror | ~2090 | 7,607,062 | +14% |

The DSP-local mirror adds ~14% overhead. The overhead comes from:
- **Input copy**: 400 KB DDR→local per batch × 468 batches = ~183 MB/epoch
- **Weight sync**: 850 KB local→DDR per epoch (negligible)

The DSP-side processing time increased by only 2.5% (7.61M vs 7.42M us). The remaining overhead is in the memcpy operations themselves.

### Why No Speedup?

The working set (~1.5 MB) fits comfortably in the Hexagon L2 cache (~1 MB + HW prefetcher). Once the first batch warms up the cache, subsequent accesses hit L2 regardless of whether data is in shared DDR or DSP-private DDR. The mirror adds copy overhead without reducing L2 hit rate.

**When would local memory help?**
- Working sets larger than L2 (several MB), where VTCM's 1-cycle latency avoids cache misses
- Streaming workloads where data is read once (no L2 benefit)
- HMX operations that require VTCM-resident data

### VTCM Findings

With `#define USE_VTCM` in skel_vtcm.c:
- **3.2x slower** than DDR baseline (~4800 ms/epoch)
- **Non-deterministic corruption**: training diverges at varying epochs
- Same code with DSP heap works perfectly — issue is VTCM-specific
- Likely cause: VTCM shared with other system services (GPU compute, camera ISP, NPU) on a running Android device

## Files

```
ch09-vtcm-train/
├── build.sh
├── run_device.sh
├── README.md
└── src/dsp/
    ├── skel_vtcm.c           # Local-memory training skel
    └── hvx_matmul_vtcm.h     # Matmul with pointer-based scratch
```

## Build & Run

```bash
bash build.sh
bash run_device.sh          # 5 epochs, batch=128
bash run_device.sh 3 64     # custom
```

## Key Takeaway

For MNIST-scale training on Hexagon v75, the L2 cache is already effective. The working set fits in L2, so mirroring to local memory adds copy overhead without reducing access latency. VTCM's 1-cycle benefit would matter more for larger models where L2 misses dominate.
