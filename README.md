# TurboQuant+ MLX

MLX-native port of [TurboQuant+](https://github.com/TheTom/turboquant_plus) for Apple Silicon. This repo extends TheTom's original with a full MLX GPU backend, Metal kernels, and mlx-lm drop-in integration — validated through a complete codebase audit with bug fixes applied. The audit improvements are tracked via parity tests that verify MLX output matches the corrected NumPy reference.

Implementation of [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026) — KV cache compression for local LLM inference, with planned extensions beyond the paper.

> **Why "Plus"?** The base TurboQuant paper is v1. I have ideas for improvements coming post-v1 — adaptive bit allocation, temporal decay compression, expert-aware MoE compression, and more. The "plus" is what comes next.

Compresses transformer KV cache using PolarQuant + Walsh-Hadamard rotation. Format family: turbo2 (2-bit, 6.4x), turbo3 (3-bit, 4.6x), turbo4 (4-bit, 3.8x).

**Key contribution:** Attention-gated KV cache decoding ("Sparse V") that skips low-weight V positions during inference. Sparse V introduces zero measurable PPL degradation (validated at 32K with 50 chunks on wikitext-103, CI ±0.021).

> **Core idea:** shift KV cache optimization from compression to attention-aware computation.

---

## Validated Benchmark Results

Hardware: Apple M5 Max 128GB | Model: Qwen3.5-35B-A3B-Q8_0 | Flash Attention ON

All numbers are post-audit with norm correction applied. TheTom's original repo (pre-audit) did not apply norm correction — turbo3 perplexity measured 165.6 before the fix, catastrophic vs the 6.194 achieved after. This repo ships with the fix.

### Quality (wikitext-2, 512 context)

| Cache Type | Bits/val | Compression | PPL (32-chunk) | PPL (8-chunk) | vs q8_0 |
|------------|----------|-------------|----------------|---------------|---------|
| f16 | 16 | 1.0x | — | 6.121 | -0.16% |
| q8_0 | 8 | 2.0x | 5.414 ± 0.140 | 6.111 ± 0.326 | baseline |
| q4_0 | 4 | 4.0x | — | 6.142 | +0.51% |
| **turbo3** | **3.5** | **4.6x** | **5.460 ± 0.141** | **6.193 ± 0.332** | **+0.8–1.3%** |

turbo3 within 1.4% of q8_0. Quality target met.

> **Comparison note:** TheTom's original README shows turbo3 PPL at 6.176 (+1.06%). That figure was produced without norm correction. Post-audit, this repo measures 6.193 (+1.4%) — slightly higher, but now accurate. The pre-fix 165.6 PPL confirms the norm bug was load-bearing.

### Long-Context Quality (wikitext-103, 32K context, 50 chunks)

| Config | PPL | ± CI | vs q8_0 | Sparse V Δ |
|--------|-----|------|---------|------------|
| q8_0 (8-bit KV) | 7.0638 | 0.021 | — | — |
| q4_0 (4-bit KV) | 7.0857 | 0.021 | +0.31% | — |
| turbo3 WITHOUT sparse V | 7.1796 | 0.021 | +1.64% | — |
| turbo3 WITH sparse V | 7.1796 | 0.021 | +1.64% | **0.0000** |

Sparse V introduces zero additional PPL degradation at any tested context length.

### Prefill Speed (wikitext-2, 512 context, 32 chunks)

| Cache Type | Prefill tok/s | vs q8_0 |
|------------|--------------|---------|
| q8_0 | 2694 | 1.00x |
| **turbo3 (block-32 + graph WHT)** | **2747** | **1.02x** |

### Speed Optimization Journey (739 → 2747 tok/s)

| Step | tok/s | vs q8_0 |
|------|-------|---------|
| fp32 WHT in dequant (initial) | 739 | 0.27x |
| + fp16 WHT | 1074 | 0.40x |
| + half4 vectorized butterfly | 1411 | 0.52x |
| + graph-side WHT rotation | 2095 | 0.78x |
| **+ block-32 storage** | **2747** | **1.02x** |

### Rotation Gaussianization (Real Model Validation)

Real Qwen3-1.7B KV tensor:
```
Raw kurtosis:       900.4  → After rotation: 2.9  (Gaussian = 3.0)
Std after rotation:  0.088388
Expected (1/√d):     0.088388
Ratio:               1.000 exactly
```

---

## Install

```bash
git clone https://github.com/ediestel/turboquant-plus-mlx.git
cd turboquant-plus-mlx
python3 -m venv .venv && source .venv/bin/activate

pip install -e ".[mlx]"        # MLX core only
pip install -e ".[mlx-lm]"     # + mlx-lm for real model inference
pip install -e ".[dev]"        # + pytest for running tests

# Verify
pytest tests/ -v
pytest tests/test_mlx/ -v      # MLX parity tests (requires Apple Silicon + mlx)
```

For macOS 13 Ventura, pin an older MLX release:

```bash
pip install "mlx>=0.16.0,<0.22.0"
```

---

## Usage

**Compress a KV cache tensor on the GPU:**

```python
import mlx.core as mx
from turboquant.mlx.kv_cache import KVCacheCompressorMLX

# Shape: (num_layers, num_heads, seq_len, head_dim)
k_cache = mx.random.normal(shape=(32, 32, 4096, 128))
v_cache = mx.random.normal(shape=(32, 32, 4096, 128))

compressor = KVCacheCompressorMLX(head_dim=128, k_bits=3, v_bits=3)
compressed = compressor.compress(k_cache, v_cache)
k_hat, v_hat = compressor.decompress(compressed)

print(compressor.memory_stats(seq_len=4096, num_layers=32, num_heads=32))
# {'original_mb': 128.0, 'compressed_mb': 27.5, 'compression_ratio': 4.65, ...}
```

**Drop-in KV cache for mlx-lm inference:**

```python
from mlx_lm import load, generate
from turboquant.integrations.mlx_lm import TurboQuantKVCache

model, tokenizer = load("mlx-community/Llama-3-8B-Instruct-4bit")
cache = TurboQuantKVCache.for_model(model, k_bits=3, v_bits=3)

response = generate(model, tokenizer, prompt="Hello", cache=cache, max_tokens=200)
```

**Backend auto-detection** — MLX is used when available, NumPy otherwise:

```python
from turboquant.backend import default_backend, set_default_backend

be = default_backend()        # "mlx" on Apple Silicon with mlx installed, else "numpy"
set_default_backend("numpy")  # force NumPy for testing/CI
```

---

## Cache Type Reference

| Flag | Bits/val | Compression vs fp16 | Description |
|------|----------|--------------------:|-------------|
| `turbo3` | 3.5 | **4.6x** | 3-bit PolarQuant + WHT rotation. Best compression. |
| `turbo4` | 4.25 | **3.8x** | 4-bit PolarQuant (16 centroids). Best quality. |
| `turbo2` | 2.5 | **6.4x** | 2-bit. Extreme compression, higher quality loss. |

---

## Architecture

```
Input: KV cache vector x ∈ R^d (one attention head)
    │
    ├── Extract norm: γ = ||x||, x̂ = x/γ
    │
    ├── Random rotation: WHT + random sign flips
    │   coordinates ~ N(0, 1/d) after rotation
    │
    ├── Optimal scalar quantization (Lloyd-Max)
    │   turbo4: 16 centroids (4-bit), turbo3: 8 centroids (3-bit), turbo2: 4 centroids (2-bit)
    │
    └── Output: quantized indices + norm per block
        Compression: 3.8x (turbo4), 4.6x (turbo3), 6.4x (turbo2)
```

> **Note on QJL:** The original paper uses a 1-bit QJL error correction step. We dropped it — QJL increases variance which softmax amplifies, hurting quality. More centroids (PolarQuant-only) beats MSE + QJL split. Confirmed independently by 5 groups.

---

## Project Structure

```
turboquant/
├── rotation.py           # Walsh-Hadamard Transform + random sign flips
├── codebook.py           # Lloyd-Max optimal centroid computation
├── polar_quant.py        # PolarQuant — norm extraction + WHT rotation + quantization
├── qjl.py                # QJL 1-bit quantizer (kept for reference, not used in production)
├── turboquant.py         # Full TurboQuant pipeline
├── kv_cache.py           # KV cache integration layer
├── outlier.py            # Outlier channel strategy (2.5-bit, 3.5-bit)
├── utils.py              # Bit packing, memory measurement
├── hw_replay.py          # Hardware replay utilities
├── backend/              # Backend protocol abstraction (NumPy ↔ MLX swap)
│   ├── _protocol.py      # Backend typing.Protocol definition
│   ├── __init__.py       # get_backend(), set_default_backend(), default_backend()
│   ├── numpy_backend.py  # NumPy/SciPy adapter (default, always available)
│   └── mlx_backend.py    # MLX adapter (Apple Silicon, optional)
├── mlx/                  # MLX-native GPU implementations
│   ├── rotation.py       # mx.linalg.qr rotation + butterfly Walsh-Hadamard
│   ├── codebook.py       # Lloyd's algorithm → mx.array centroids
│   ├── polar_quant.py    # PolarQuantMLX
│   ├── qjl.py            # QJLMLX (1-bit sign quantization)
│   ├── turboquant.py     # TurboQuantMLX + TurboQuantMSEMLX
│   ├── kv_cache.py       # KVCacheCompressorMLX (batched layer/head compression)
│   ├── outlier.py        # OutlierTurboQuantMLX (2.5-bit, 3.5-bit)
│   ├── utils.py          # Bit packing via NumPy interop
│   └── kernels/
│       ├── hadamard.metal    # Custom Metal WHT kernel (shared-memory butterfly)
│       └── quantize.metal    # Fused Metal kernel: rotate → normalize → centroid lookup
└── integrations/
    └── mlx_lm.py         # TurboQuantKVCache — drop-in for mlx-lm generate()

tests/
├── test_mlx/             # MLX parity tests — verify MLX output matches audited NumPy
│                         # reference after each improvement. Full package retained for this.
└── (14 test files, 511 tests total)
```

---

## Roadmap

| Phase | Status | Details |
|-------|--------|---------|
| Core algorithms (NumPy) | ✅ | 511 tests, 14 test files |
| Distortion validation | ✅ | Matches paper bounds (Table 2) |
| Real model validation | ✅ | Rotation validated on Qwen3 KV tensors (kurtosis 900→2.9) |
| Norm correction audit | ✅ | Fixed norm bug from original — PPL 165.6 → 6.194 |
| MLX GPU backend | ✅ | Metal kernels, mlx-lm integration, 27/27 parity tests |
| Sparse V | ✅ | Zero PPL delta at 32K, validated on wikitext-103 (50 chunks, CI ±0.021) |
| TurboQuant+ extensions | ⏳ | Adaptive bits, temporal decay, MoE-aware compression |

---

## Paper Reference

- **TurboQuant**: [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- **PolarQuant**: [arXiv 2502.02617](https://arxiv.org/abs/2502.02617) (AISTATS 2026)
- **QJL**: [arXiv 2406.03482](https://arxiv.org/abs/2406.03482)
- **Google Research Blog**: [TurboQuant: Redefining AI Efficiency](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

---

## Contributing

Issues and PRs welcome. Main areas where help is needed:

1. **Quality metrics** — multi-run statistics, additional task benchmarks (GSM8K, code gen, reasoning)
2. **Long context validation** — 64K+ testing across architectures
3. **TurboQuant+ extensions** — adaptive bit allocation, temporal decay, MoE-aware compression

---

## Attribution

This MLX port is based on [TheTom/turboquant-plus](https://github.com/TheTom/turboquant_plus) (Apache 2.0). The original Python reference implementation was written by Tom Turney.

This repo extends the original with:
- MLX-native GPU backend for Apple Silicon (`turboquant/mlx/`)
- mlx-lm drop-in integration (`turboquant/integrations/mlx_lm.py`)
- Bug fixes identified during a full codebase audit of the original

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

MLX port and extensions Copyright 2026 Eckhart Diestel.
Based on Google Research's TurboQuant paper (arXiv 2504.19874) and TheTom/turboquant-plus.
