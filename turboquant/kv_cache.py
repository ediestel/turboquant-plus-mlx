"""KV Cache integration layer for TurboQuant.

Compresses transformer KV cache tensors using TurboQuant (for K cache, inner product
preservation) and PolarQuant MSE-only (for V cache, MSE preservation).

KV cache shape: (num_layers, num_heads, seq_len, head_dim)
Quantization is along head_dim — each (head_dim,) vector is quantized independently.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from turboquant.turboquant import TurboQuant, TurboQuantMSE, CompressedVector
from turboquant.adaptive_bits import AllocationPlan
from turboquant.temporal_decay import TemporalDecayScheduler
from turboquant.moe_compression import MoEBitPlan


@dataclass
class CompressedKVCache:
    """Container for a compressed KV cache."""
    # Per-layer, per-head compressed K vectors
    k_compressed: list[list[CompressedVector]] = field(default_factory=list)
    # Per-layer, per-head compressed V (indices + norms)
    v_indices: list[list[np.ndarray]] = field(default_factory=list)
    v_norms: list[list[np.ndarray]] = field(default_factory=list)

    num_layers: int = 0
    num_heads: int = 0
    seq_len: int = 0
    head_dim: int = 0
    k_bit_width: int = 0
    v_bit_width: int = 0

    # Set by AdaptiveKVCacheCompressor: effective integer bit widths per (layer, head).
    # None when the base compressor was used (all heads at k_bit_width / v_bit_width).
    k_bits_matrix: Optional[np.ndarray] = None  # shape (num_layers, num_heads), dtype int
    v_bits_matrix: Optional[np.ndarray] = None  # shape (num_layers, num_heads), dtype int


class KVCacheCompressor:
    """Compress and decompress transformer KV cache tensors.

    Uses:
    - TurboQuant (Algorithm 2) for K cache — inner product preservation matters
      for attention score computation (Q @ K^T). Stores indices + 2 norms per vector
      (vector_norms for rescaling, residual_norms for QJL stage).
    - TurboQuantMSE (Algorithm 1) for V cache — MSE preservation matters
      for value reconstruction (attn_weights @ V). Stores indices + 1 norm per vector
      (norms still needed for rescaling despite no QJL stage).

    Usage:
        compressor = KVCacheCompressor(head_dim=128, k_bits=3, v_bits=3)

        # Compress
        compressed = compressor.compress(k_cache, v_cache)

        # Decompress
        k_hat, v_hat = compressor.decompress(compressed)

        # Or compress streaming (one token at a time)
        compressor.compress_token(k_vec, v_vec, layer=0, head=0)
    """

    def __init__(
        self,
        head_dim: int,
        k_bits: int = 3,
        v_bits: int = 3,
        seed: int = 42,
        norm_correction: bool = True,
        allocation_plan: Optional[AllocationPlan] = None,
        decay_scheduler: Optional[TemporalDecayScheduler] = None,
        moe_plans: Optional[dict] = None,
    ):
        """
        Args:
            head_dim: Dimension of each attention head vector.
            k_bits: Default bit-width for K cache (TurboQuant, inner product).
            v_bits: Default bit-width for V cache (PolarQuant MSE-only).
            seed: Random seed.
            norm_correction: Whether to apply norm correction on dequantization.
            allocation_plan: Optional per-(layer, head) bit allocation. When set,
                ``adaptive_bits_enabled`` is True in memory_stats().
            decay_scheduler: Optional temporal decay scheduler. When set,
                ``temporal_decay_enabled`` is True in memory_stats().
            moe_plans: Optional dict mapping layer_idx -> MoEBitPlan. When set
                and non-empty, ``moe_routing_enabled`` is True in memory_stats().
        """
        self.head_dim = head_dim
        self.k_bits = k_bits
        self.v_bits = v_bits
        self._seed = seed
        self._norm_correction = norm_correction

        self._allocation_plan = allocation_plan
        self._decay_scheduler = decay_scheduler
        self._moe_plans: dict = moe_plans if moe_plans is not None else {}

        # Default quantizers (backward-compatible, used when no allocation_plan)
        self.k_quantizer = TurboQuant(
            head_dim, bit_width=k_bits, seed=seed, norm_correction=norm_correction,
        )
        self.v_quantizer = TurboQuantMSE(
            head_dim, bit_width=v_bits, seed=seed + 500, norm_correction=norm_correction,
        )

        # Quantizer caches: int bit-width → quantizer instance
        self._k_quantizer_cache: dict[int, TurboQuant] = {k_bits: self.k_quantizer}
        self._v_quantizer_cache: dict[int, TurboQuantMSE] = {v_bits: self.v_quantizer}

    def compress(self, k_cache: np.ndarray, v_cache: np.ndarray) -> CompressedKVCache:
        """Compress full KV cache tensors.

        Args:
            k_cache: Key cache, shape (num_layers, num_heads, seq_len, head_dim).
            v_cache: Value cache, same shape.

        Returns:
            CompressedKVCache with compressed K and V.
        """
        num_layers, num_heads, seq_len, head_dim = k_cache.shape
        assert head_dim == self.head_dim
        assert v_cache.shape == k_cache.shape

        result = CompressedKVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            k_bit_width=self.k_bits,
            v_bit_width=self.v_bits,
        )

        for layer in range(num_layers):
            k_layer = []
            v_layer_idx = []
            v_layer_norms = []
            for head in range(num_heads):
                # K: batch quantize all seq positions for this layer/head
                k_vecs = k_cache[layer, head]  # (seq_len, head_dim)
                k_compressed = self.k_quantizer.quantize(k_vecs)
                k_layer.append(k_compressed)

                # V: MSE quantize (returns indices + norms)
                v_vecs = v_cache[layer, head]  # (seq_len, head_dim)
                v_indices, v_norms = self.v_quantizer.quantize(v_vecs)
                v_layer_idx.append(v_indices)
                v_layer_norms.append(v_norms)

            result.k_compressed.append(k_layer)
            result.v_indices.append(v_layer_idx)
            result.v_norms.append(v_layer_norms)

        return result

    def decompress(self, compressed: CompressedKVCache) -> tuple[np.ndarray, np.ndarray]:
        """Decompress back to full KV cache tensors.

        Returns:
            (k_cache, v_cache) both shape (num_layers, num_heads, seq_len, head_dim).
        """
        k_cache = np.zeros((
            compressed.num_layers, compressed.num_heads,
            compressed.seq_len, compressed.head_dim
        ))
        v_cache = np.zeros_like(k_cache)

        for layer in range(compressed.num_layers):
            for head in range(compressed.num_heads):
                k_cache[layer, head] = self.k_quantizer.dequantize(
                    compressed.k_compressed[layer][head]
                )
                v_cache[layer, head] = self.v_quantizer.dequantize(
                    compressed.v_indices[layer][head],
                    compressed.v_norms[layer][head],
                )

        return k_cache, v_cache

    def _get_k_quantizer(self, bits: int) -> TurboQuant:
        """Return a cached TurboQuant instance for the given integer bit-width.

        The cache key is the integer bit-width after rounding and clipping.
        Same bit-width always returns the same instance.
        """
        bits_i = int(np.clip(round(bits), 1, 8))
        if bits_i not in self._k_quantizer_cache:
            self._k_quantizer_cache[bits_i] = TurboQuant(
                self.head_dim, bit_width=bits_i,
                seed=self._seed, norm_correction=self._norm_correction,
            )
        return self._k_quantizer_cache[bits_i]

    def _get_v_quantizer(self, bits: int) -> TurboQuantMSE:
        """Return a cached TurboQuantMSE instance for the given integer bit-width."""
        bits_i = int(np.clip(round(bits), 1, 8))
        if bits_i not in self._v_quantizer_cache:
            self._v_quantizer_cache[bits_i] = TurboQuantMSE(
                self.head_dim, bit_width=bits_i,
                seed=self._seed + 500, norm_correction=self._norm_correction,
            )
        return self._v_quantizer_cache[bits_i]

    def effective_bits_matrix(
        self, num_layers: int, num_heads: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return target bit-width matrices shape (num_layers, num_heads).

        If an allocation_plan is set, returns its arrays (clipped to the
        requested shape if necessary). Otherwise returns constant arrays
        filled with self.k_bits / self.v_bits.

        Returns:
            (k_bits_matrix, v_bits_matrix) both float64, shape (num_layers, num_heads).
        """
        if self._allocation_plan is not None:
            return (
                self._allocation_plan.k_bits[:num_layers, :num_heads].copy(),
                self._allocation_plan.v_bits[:num_layers, :num_heads].copy(),
            )
        k_mat = np.full((num_layers, num_heads), float(self.k_bits), dtype=np.float64)
        v_mat = np.full((num_layers, num_heads), float(self.v_bits), dtype=np.float64)
        return k_mat, v_mat

    def memory_stats(self, seq_len: int, num_layers: int, num_heads: int) -> dict:
        """Compute memory usage statistics.

        Returns dict with original_mb, compressed_mb, ratio.
        """
        n_vectors = num_layers * num_heads * seq_len
        original_bytes = n_vectors * self.head_dim * 2  # fp16

        # K: b bits per coord + 64-bit norms (vector_norms + residual_norms, both float32)
        k_bits_total = n_vectors * (self.head_dim * self.k_bits + 64)
        # V: b bits per coord + 32-bit norm (norms stored for rescaling despite no QJL stage)
        v_bits_total = n_vectors * (self.head_dim * self.v_bits + 32)

        compressed_bytes = (k_bits_total + v_bits_total) / 8

        return {
            "original_mb": original_bytes / 1024 / 1024,
            "compressed_mb": compressed_bytes / 1024 / 1024,
            "compression_ratio": original_bytes / compressed_bytes,
            "k_bits_per_value": self.k_bits,
            "v_bits_per_value": self.v_bits,
            "adaptive_bits_enabled": self._allocation_plan is not None,
            "temporal_decay_enabled": self._decay_scheduler is not None,
            "moe_routing_enabled": bool(self._moe_plans),
        }
