"""Adaptive per-layer/per-head bit allocation for KV cache compression.

Provides sensitivity-aware bit allocation: heads with higher sensitivity scores
receive more bits, low-sensitivity heads receive fewer, subject to a global
base_bits budget and min/max clamps.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class SensitivityProfile:
    """Per-layer/per-head sensitivity scores.

    Args:
        scores: Shape (n_layers, n_heads). Higher values = more sensitive = more bits.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads per layer.
    """
    scores: np.ndarray
    n_layers: int
    n_heads: int

    def __post_init__(self):
        self.scores = np.asarray(self.scores, dtype=np.float64)
        if self.scores.shape != (self.n_layers, self.n_heads):
            raise ValueError(
                f"scores.shape {self.scores.shape} != (n_layers={self.n_layers}, "
                f"n_heads={self.n_heads})"
            )


@dataclass
class AllocationPlan:
    """Per-layer/per-head target bit allocations.

    Note: bits are *target* floats. The actual quantizer used is selected by
    rounding to the nearest supported integer bit-width. Callers should not
    assume ``k_bits[l, h]`` maps directly to a quantizer without rounding.
    """
    k_bits: np.ndarray   # shape (n_layers, n_heads), float64
    v_bits: np.ndarray   # shape (n_layers, n_heads), float64

    @property
    def n_layers(self) -> int:
        return self.k_bits.shape[0]

    @property
    def n_heads(self) -> int:
        return self.k_bits.shape[1]


class AdaptiveBitAllocator:
    """Allocate bit-widths per (layer, head) based on sensitivity scores.

    Uses z-score normalization of sensitivity scores to shift bits around
    ``base_bits``. When all scores are equal (zero variance), returns a
    uniform allocation at ``base_bits``.

    Args:
        base_bits: Mean bit-width target.
        sensitivity_scale: Scale factor applied to z-scored sensitivity.
            0.0 → uniform allocation regardless of scores.
        min_bits: Minimum allowed bit-width.
        max_bits: Maximum allowed bit-width.
    """

    def __init__(
        self,
        base_bits: float = 3.5,
        sensitivity_scale: float = 0.5,
        min_bits: float = 1.0,
        max_bits: float = 8.0,
    ):
        self.base_bits = float(base_bits)
        self.sensitivity_scale = float(sensitivity_scale)
        self.min_bits = float(min_bits)
        self.max_bits = float(max_bits)

    def allocate(self, profile: SensitivityProfile) -> AllocationPlan:
        """Allocate bits from a sensitivity profile.

        Args:
            profile: Sensitivity scores shape (n_layers, n_heads).

        Returns:
            AllocationPlan with k_bits and v_bits both derived from the profile.
        """
        scores = profile.scores.astype(np.float64)
        std = float(np.std(scores))

        if std < 1e-12 or self.sensitivity_scale == 0.0:
            bits = np.full_like(scores, self.base_bits)
        else:
            mean = float(np.mean(scores))
            z = (scores - mean) / std
            bits = np.clip(
                self.base_bits + self.sensitivity_scale * z,
                self.min_bits,
                self.max_bits,
            )

        return AllocationPlan(k_bits=bits.copy(), v_bits=bits.copy())

    def allocate_uniform(self, n_layers: int, n_heads: int) -> AllocationPlan:
        """Return a uniform plan with all heads at ``base_bits``.

        Args:
            n_layers: Number of transformer layers.
            n_heads: Number of attention heads.
        """
        bits = np.full((n_layers, n_heads), self.base_bits, dtype=np.float64)
        return AllocationPlan(k_bits=bits.copy(), v_bits=bits.copy())
