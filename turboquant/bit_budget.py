"""BitBudgetPolicy protocol and concrete implementations for KV cache bit allocation.

Provides a clean, composable interface for specifying per-(layer, head) bit budgets:
- UniformPolicy: same bit width everywhere
- LayerHeadPolicy: backed by an AllocationPlan matrix
- SensitivityCalibratedPolicy: auto-calibrates from attention entropy measurements
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from turboquant.adaptive_bits import AllocationPlan, AdaptiveBitAllocator, SensitivityProfile


@dataclass(frozen=True)
class SlotKey:
    """Identifies a (layer, head) slot in the KV cache.

    Frozen and hashable — safe to use as a dict key.
    """
    layer_idx: int
    head_idx: int


@runtime_checkable
class BitBudgetPolicy(Protocol):
    """Protocol for per-slot bit allocation policies.

    Any object implementing k_bits_for() and v_bits_for() satisfies this protocol.
    """

    def k_bits_for(self, slot: SlotKey) -> float:
        """Return target bit-width for the K cache at this slot."""
        ...

    def v_bits_for(self, slot: SlotKey) -> float:
        """Return target bit-width for the V cache at this slot."""
        ...


class UniformPolicy:
    """Returns the same bit-width for every (layer, head) slot.

    Args:
        base_bits: Target bit-width for all slots.
    """

    def __init__(self, base_bits: float = 3.5):
        self.base_bits = float(base_bits)

    def k_bits_for(self, slot: SlotKey) -> float:
        return self.base_bits

    def v_bits_for(self, slot: SlotKey) -> float:
        return self.base_bits

    def to_allocation_plan(self, n_layers: int, n_heads: int) -> AllocationPlan:
        """Return a uniform AllocationPlan of shape (n_layers, n_heads)."""
        allocator = AdaptiveBitAllocator(
            base_bits=self.base_bits, sensitivity_scale=0.0
        )
        return allocator.allocate_uniform(n_layers, n_heads)


class LayerHeadPolicy:
    """Per-(layer, head) bit allocation backed by an AllocationPlan matrix.

    Args:
        plan: AllocationPlan with k_bits and v_bits matrices of shape (n_layers, n_heads).
    """

    def __init__(self, plan: AllocationPlan):
        self._plan = plan

    def k_bits_for(self, slot: SlotKey) -> float:
        return float(self._plan.k_bits[slot.layer_idx, slot.head_idx])

    def v_bits_for(self, slot: SlotKey) -> float:
        return float(self._plan.v_bits[slot.layer_idx, slot.head_idx])

    def to_allocation_plan(self, n_layers: int = None, n_heads: int = None) -> AllocationPlan:
        """Return the inner AllocationPlan (shape args ignored — already fixed at init)."""
        return self._plan

    @classmethod
    def from_uniform(
        cls, n_layers: int, n_heads: int, base_bits: float = 3.5
    ) -> "LayerHeadPolicy":
        """Construct a uniform LayerHeadPolicy."""
        allocator = AdaptiveBitAllocator(base_bits=base_bits, sensitivity_scale=0.0)
        return cls(allocator.allocate_uniform(n_layers, n_heads))


class SensitivityCalibratedPolicy:
    """Auto-calibrates per-head bit allocation from attention entropy measurements.

    During a calibration pass, call record_attention_weights() for each observed
    attention head. Peaked (low-entropy) heads receive more bits; diffuse
    (high-entropy) heads receive fewer. Call calibrate() to materialise the plan.
    k_bits_for() / v_bits_for() lazily call calibrate() if needed.

    Before any samples are recorded, returns uniform base_bits (uncalibrated fallback).

    Args:
        n_layers: Number of transformer layers (fixed at construction).
        n_heads: Number of attention heads per layer (fixed at construction).
        base_bits: Mean bit-width target.
        sensitivity_scale: Controls spread around base_bits. 0 → uniform.
        min_bits: Minimum allowed bit-width.
        max_bits: Maximum allowed bit-width.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        base_bits: float = 3.5,
        sensitivity_scale: float = 0.5,
        min_bits: float = 1.0,
        max_bits: float = 8.0,
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self._allocator = AdaptiveBitAllocator(
            base_bits=base_bits,
            sensitivity_scale=sensitivity_scale,
            min_bits=min_bits,
            max_bits=max_bits,
        )
        self._entropy_sums = np.zeros((n_layers, n_heads), dtype=np.float64)
        self._sample_counts = np.zeros((n_layers, n_heads), dtype=np.int64)
        self._plan: Optional[AllocationPlan] = None

    # ------------------------------------------------------------------
    # Data accumulation
    # ------------------------------------------------------------------

    def record_attention_weights(
        self,
        attn_weights: np.ndarray,
        layer_idx: int,
        head_idx: int,
    ) -> None:
        """Accumulate entropy from one attention weight vector.

        Args:
            attn_weights: Attention probabilities, shape (seq_len,) or any shape.
                          Will be flattened, normalised, and converted to float64.
                          Samples with zero-sum (all-zero weights) are ignored.
            layer_idx: Transformer layer index (0-based).
            head_idx: Attention head index (0-based).
        """
        w = np.asarray(attn_weights, dtype=np.float64).ravel()
        total = float(np.sum(w))
        if total <= 0.0:
            return  # ignore invalid / all-zero samples
        w = np.clip(w / total, 1e-12, 1.0)  # renormalize then clip
        entropy = float(-np.sum(w * np.log2(w)))
        self._entropy_sums[layer_idx, head_idx] += entropy
        self._sample_counts[layer_idx, head_idx] += 1
        self._plan = None  # invalidate cached plan

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self) -> AllocationPlan:
        """Compute AllocationPlan from accumulated entropy statistics.

        Low entropy (peaked attention) → high sensitivity → more bits.
        High entropy (diffuse attention) → low sensitivity → fewer bits.
        Unobserved heads receive the overall mean entropy (not zero).
        If no samples at all, returns a uniform plan at base_bits.

        Returns:
            AllocationPlan, also cached in _plan.
        """
        if not self.is_calibrated:
            self._plan = self._allocator.allocate_uniform(self.n_layers, self.n_heads)
            return self._plan

        counts = np.maximum(self._sample_counts, 1)
        mean_entropy = self._entropy_sums / counts

        # Fill unobserved heads with overall mean
        observed = self._sample_counts > 0
        if observed.any():
            overall_mean = float(np.mean(mean_entropy[observed]))
            mean_entropy[~observed] = overall_mean

        # Low entropy = peaked attention = higher sensitivity → invert
        sensitivity = -mean_entropy
        sensitivity -= float(np.min(sensitivity))  # shift so minimum = 0

        # All heads have equal entropy → uniform plan
        if float(np.max(sensitivity)) < 1e-12:
            self._plan = self._allocator.allocate_uniform(self.n_layers, self.n_heads)
            return self._plan

        profile = SensitivityProfile(
            scores=sensitivity,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
        )
        self._plan = self._allocator.allocate(profile)
        return self._plan

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        """True if at least one attention sample has been recorded."""
        return bool(np.any(self._sample_counts > 0))

    def k_bits_for(self, slot: SlotKey) -> float:
        plan = self._plan if self._plan is not None else self.calibrate()
        return float(plan.k_bits[slot.layer_idx, slot.head_idx])

    def v_bits_for(self, slot: SlotKey) -> float:
        plan = self._plan if self._plan is not None else self.calibrate()
        return float(plan.v_bits[slot.layer_idx, slot.head_idx])

    def to_allocation_plan(
        self, n_layers: int = None, n_heads: int = None
    ) -> AllocationPlan:
        """Return calibrated plan (shape args ignored — fixed at construction)."""
        return self._plan if self._plan is not None else self.calibrate()

    def reset(self) -> None:
        """Clear all accumulated entropy data and invalidate the cached plan."""
        self._entropy_sums[:] = 0.0
        self._sample_counts[:] = 0
        self._plan = None
