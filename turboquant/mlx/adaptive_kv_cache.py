"""MLX-native adaptive KV cache compressor.

MLX GPU mirror of the three TurboQuant+ MLX extensions:
- Adaptive per-head bit allocation (AllocationPlan)
- Temporal decay scheduling (TemporalDecayScheduler)
- MoE-aware per-expert bit budgets (MoEBitPlan)

All new kwargs are optional — KVCacheCompressorMLX baseline behaviour is
preserved when none are supplied.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx

from turboquant.mlx.turboquant import TurboQuantMLX, TurboQuantMSEMLX, CompressedVectorMLX
from turboquant.mlx.kv_cache import KVCacheCompressorMLX, CompressedKVCacheMLX
from turboquant.adaptive_bits import AllocationPlan
from turboquant.temporal_decay import TemporalDecayScheduler
from turboquant.moe_compression import MoEBitPlan


class AdaptiveKVCacheCompressorMLX(KVCacheCompressorMLX):
    """KVCacheCompressorMLX extended with adaptive bits, temporal decay, and MoE.

    Inherits all compress/decompress/compress_single/decompress_single methods
    from KVCacheCompressorMLX. Adds:
    - Quantizer instance cache keyed by integer bit-width
    - effective_bits_matrix() reporting
    - Extended memory_stats() with feature-enable flags

    Args:
        head_dim: Attention head dimension.
        k_bits: Default K-cache bit-width.
        v_bits: Default V-cache bit-width.
        seed: Random seed.
        norm_correction: Apply norm correction on dequantization.
        allocation_plan: Optional per-(layer, head) AllocationPlan.
        decay_scheduler: Optional TemporalDecayScheduler.
        moe_plans: Optional dict[layer_idx -> MoEBitPlan].
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
        super().__init__(
            head_dim=head_dim,
            k_bits=k_bits,
            v_bits=v_bits,
            seed=seed,
            norm_correction=norm_correction,
        )

        self._seed = seed
        self._norm_correction = norm_correction
        self._allocation_plan = allocation_plan
        self._decay_scheduler = decay_scheduler
        self._moe_plans: dict = moe_plans if moe_plans is not None else {}

        # Quantizer caches: int bit-width → instance
        self._k_quantizer_cache: dict[int, TurboQuantMLX] = {
            k_bits: self.k_quantizer
        }
        self._v_quantizer_cache: dict[int, TurboQuantMSEMLX] = {
            v_bits: self.v_quantizer
        }

    # ------------------------------------------------------------------
    # Quantizer cache
    # ------------------------------------------------------------------

    def _get_k_quantizer(self, bits) -> TurboQuantMLX:
        """Return a cached TurboQuantMLX for the given bit-width.

        Cache key is the integer bit-width after rounding and clipping.
        Same bit-width always returns the same instance.
        """
        bits_i = int(np.clip(round(bits), 1, 8))
        if bits_i not in self._k_quantizer_cache:
            self._k_quantizer_cache[bits_i] = TurboQuantMLX(
                self.head_dim, bit_width=bits_i,
                seed=self._seed, norm_correction=self._norm_correction,
            )
        return self._k_quantizer_cache[bits_i]

    def _get_v_quantizer(self, bits) -> TurboQuantMSEMLX:
        """Return a cached TurboQuantMSEMLX for the given bit-width."""
        bits_i = int(np.clip(round(bits), 1, 8))
        if bits_i not in self._v_quantizer_cache:
            self._v_quantizer_cache[bits_i] = TurboQuantMSEMLX(
                self.head_dim, bit_width=bits_i,
                seed=self._seed + 500, norm_correction=self._norm_correction,
            )
        return self._v_quantizer_cache[bits_i]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def effective_bits_matrix(
        self, num_layers: int, num_heads: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return target bit-width matrices shape (num_layers, num_heads).

        If an allocation_plan is set, returns its arrays. Otherwise returns
        constant arrays filled with self.k_bits / self.v_bits.

        Returns:
            (k_bits_matrix, v_bits_matrix) float64, shape (num_layers, num_heads).
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
        """Memory stats extended with TurboQuant+ MLX feature flags."""
        stats = super().memory_stats(seq_len, num_layers, num_heads)
        stats["adaptive_bits_enabled"] = self._allocation_plan is not None
        stats["temporal_decay_enabled"] = self._decay_scheduler is not None
        stats["moe_routing_enabled"] = bool(self._moe_plans)
        return stats
