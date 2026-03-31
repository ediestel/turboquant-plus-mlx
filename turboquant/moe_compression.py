"""MoE-aware compression: per-expert bit allocation for Mixture-of-Experts models.

In MoE models (Qwen3.5-35B-A3B, Mixtral, etc.), different experts see different
input distributions and fire at different rates. This module assigns higher bit
budgets to frequently-activated, high-importance experts and lower bits (or
eviction) to rarely-used ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ExpertRoutingStats:
    """Activation statistics for all experts in one MoE layer.

    Args:
        n_experts: Total number of experts in this layer.
        n_experts_used: Number of experts activated per token (top-k).
        activation_counts: How many times each expert was activated.
            Shape (n_experts,), dtype int64, all values >= 0.
        layer_idx: Optional layer index for bookkeeping.
    """
    n_experts: int
    n_experts_used: int
    activation_counts: np.ndarray
    layer_idx: Optional[int] = None

    def __post_init__(self):
        self.activation_counts = np.asarray(self.activation_counts, dtype=np.int64)
        if len(self.activation_counts) != self.n_experts:
            raise ValueError(
                f"activation_counts length {len(self.activation_counts)} != "
                f"n_experts={self.n_experts}"
            )
        if self.n_experts_used > self.n_experts:
            raise ValueError(
                f"n_experts_used={self.n_experts_used} cannot exceed "
                f"n_experts={self.n_experts}"
            )
        if np.any(self.activation_counts < 0):
            raise ValueError("activation_counts must be non-negative")

    @property
    def utilisation(self) -> np.ndarray:
        """Per-expert utilisation, normalised so the sum equals ``n_experts_used``.

        If total counts are zero (no activations recorded), returns a uniform
        distribution summing to ``n_experts_used``.
        """
        total = float(np.sum(self.activation_counts))
        if total < 1e-12:
            return np.full(self.n_experts, self.n_experts_used / self.n_experts)
        return self.activation_counts.astype(np.float64) / total * self.n_experts_used

    @property
    def inactive_experts(self) -> np.ndarray:
        """Indices of experts with zero activation count."""
        return np.where(self.activation_counts == 0)[0]

    @classmethod
    def uniform(cls, n_experts: int, n_experts_used: int) -> "ExpertRoutingStats":
        """Create stats with equal activation counts for all experts."""
        counts = np.ones(n_experts, dtype=np.int64)
        return cls(n_experts=n_experts, n_experts_used=n_experts_used,
                   activation_counts=counts)

    @classmethod
    def from_routing_indices(
        cls,
        routing_indices: np.ndarray,
        n_experts: int,
        n_experts_used: int,
        layer_idx: Optional[int] = None,
    ) -> "ExpertRoutingStats":
        """Build stats by counting expert activations from routing indices.

        Out-of-range indices (< 0 or >= n_experts) are **ignored** — they are
        not counted toward any expert. This preserves count integrity rather
        than silently biasing edge experts via clipping.

        Args:
            routing_indices: Array of expert indices, any shape. Will be flattened.
            n_experts: Total number of experts.
            n_experts_used: Number of experts activated per token (top-k).
            layer_idx: Optional layer index.
        """
        flat = np.asarray(routing_indices).ravel()
        valid = flat[(flat >= 0) & (flat < n_experts)]
        counts = np.bincount(valid.astype(np.int64), minlength=n_experts).astype(np.int64)
        return cls(
            n_experts=n_experts,
            n_experts_used=n_experts_used,
            activation_counts=counts,
            layer_idx=layer_idx,
        )


@dataclass
class MoEBitPlan:
    """Per-expert bit allocation plan for one MoE layer.

    Attributes:
        expert_bits: Target bit-width per expert, shape (n_experts,).
            Evicted experts have expert_bits[i] == 0.0.
        evict_mask: True where expert should be evicted, shape (n_experts,).
    """
    expert_bits: np.ndarray
    evict_mask: np.ndarray

    def bits_for(self, expert_id: int) -> float:
        """Return target bit-width for the given expert."""
        return float(self.expert_bits[expert_id])

    def should_evict(self, expert_id: int) -> bool:
        """Return True if the given expert should be evicted."""
        return bool(self.evict_mask[expert_id])

    def summary(self) -> str:
        n = len(self.expert_bits)
        evicted = int(np.sum(self.evict_mask))
        retained = self.expert_bits[~self.evict_mask]
        mean_b = float(np.mean(retained)) if len(retained) > 0 else 0.0
        return (
            f"MoEBitPlan(experts={n}, evicted={evicted}, "
            f"mean_bits_retained={mean_b:.2f})"
        )


class MoECompressionRouter:
    """Compute per-expert bit allocations from routing statistics.

    Bit allocation uses z-score normalised utilisation. When variance is zero
    or ``sensitivity_scale=0``, returns uniform bits.

    Args:
        base_bits: Mean target bit-width.
        sensitivity_scale: Scaling factor for z-scored utilisation. 0 → uniform.
        min_bits: Minimum bit-width for retained experts.
        max_bits: Maximum bit-width.
        evict_inactive: If True, experts with zero activation count are evicted
            unconditionally (expert_bits set to 0.0).
        eviction_utilisation_threshold: If set, experts with
            ``utilisation < threshold * mean(utilisation)`` are additionally evicted.
    """

    def __init__(
        self,
        base_bits: float = 3.5,
        sensitivity_scale: float = 0.5,
        min_bits: float = 2.0,
        max_bits: float = 6.0,
        evict_inactive: bool = False,
        eviction_utilisation_threshold: Optional[float] = None,
    ):
        self.base_bits = float(base_bits)
        self.sensitivity_scale = float(sensitivity_scale)
        self.min_bits = float(min_bits)
        self.max_bits = float(max_bits)
        self.evict_inactive = evict_inactive
        self.eviction_utilisation_threshold = eviction_utilisation_threshold

    def plan(self, stats: ExpertRoutingStats) -> MoEBitPlan:
        """Compute a bit plan from routing statistics.

        Args:
            stats: Expert activation statistics.

        Returns:
            MoEBitPlan with per-expert bits and eviction flags.
        """
        util = stats.utilisation  # sums to n_experts_used
        n = stats.n_experts

        std = float(np.std(util))
        if std < 1e-12 or self.sensitivity_scale == 0.0:
            bits = np.full(n, self.base_bits, dtype=np.float64)
        else:
            mean = float(np.mean(util))
            z = (util - mean) / std
            bits = np.clip(
                self.base_bits + self.sensitivity_scale * z,
                self.min_bits,
                self.max_bits,
            )

        evict_mask = np.zeros(n, dtype=bool)

        # Rule 1: evict zero-count experts unconditionally
        if self.evict_inactive:
            evict_mask |= stats.activation_counts == 0

        # Rule 2: evict low-utilisation experts below threshold
        if self.eviction_utilisation_threshold is not None:
            mean_util = float(np.mean(util))
            if mean_util > 1e-12:
                evict_mask |= util < self.eviction_utilisation_threshold * mean_util

        bits[evict_mask] = 0.0
        return MoEBitPlan(expert_bits=bits, evict_mask=evict_mask)

    def plan_uniform(self, n_experts: int, n_experts_used: int) -> MoEBitPlan:
        """Return a uniform plan: all experts retained at ``base_bits``.

        Args:
            n_experts: Total number of experts.
            n_experts_used: Number activated per token (used for stats construction).
        """
        bits = np.full(n_experts, self.base_bits, dtype=np.float64)
        evict_mask = np.zeros(n_experts, dtype=bool)
        return MoEBitPlan(expert_bits=bits, evict_mask=evict_mask)

    def update_stats(
        self,
        existing: ExpertRoutingStats,
        new_routing_indices: np.ndarray,
        decay_factor: float = 0.9,
    ) -> ExpertRoutingStats:
        """Blend existing stats with new routing observations via EMA.

        Formula:
            updated = decay_factor * existing.counts + (1 - decay_factor) * new_counts

        where ``new_counts`` are derived from ``new_routing_indices`` (invalid
        indices ignored). Result is normalised to integer counts.

        Special cases:
            decay_factor=1.0 → return existing counts unchanged.
            decay_factor=0.0 → return only new counts.

        Args:
            existing: Prior routing statistics.
            new_routing_indices: Routing decisions from new tokens.
            decay_factor: EMA weight on existing counts. In [0, 1].

        Returns:
            New ExpertRoutingStats with blended counts and preserved layer_idx.
        """
        new_stats = ExpertRoutingStats.from_routing_indices(
            routing_indices=new_routing_indices,
            n_experts=existing.n_experts,
            n_experts_used=existing.n_experts_used,
            layer_idx=existing.layer_idx,
        )

        if decay_factor >= 1.0:
            return ExpertRoutingStats(
                n_experts=existing.n_experts,
                n_experts_used=existing.n_experts_used,
                activation_counts=existing.activation_counts.copy(),
                layer_idx=existing.layer_idx,
            )
        if decay_factor <= 0.0:
            return ExpertRoutingStats(
                n_experts=existing.n_experts,
                n_experts_used=existing.n_experts_used,
                activation_counts=new_stats.activation_counts.copy(),
                layer_idx=existing.layer_idx,
            )

        blended = (
            decay_factor * existing.activation_counts.astype(np.float64)
            + (1.0 - decay_factor) * new_stats.activation_counts.astype(np.float64)
        )
        counts = np.round(blended).astype(np.int64)
        counts = np.maximum(counts, 0)

        return ExpertRoutingStats(
            n_experts=existing.n_experts,
            n_experts_used=existing.n_experts_used,
            activation_counts=counts,
            layer_idx=existing.layer_idx,
        )
