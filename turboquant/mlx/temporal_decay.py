"""MLX-aware temporal decay for KV cache compression.

The scheduling logic (DecayConfig, DecaySchedule, TemporalDecayScheduler) is
backend-agnostic pure Python/NumPy and is re-exported directly from the
reference module. This module adds MLX-specific utilities for applying a
DecaySchedule to mx.array KV data.
"""

from __future__ import annotations

import numpy as np

# Re-export the backend-agnostic scheduling types — no duplication needed.
from turboquant.temporal_decay import (
    DecayConfig,
    DecayMode,
    DecaySchedule,
    TemporalDecayScheduler,
)

try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


def apply_eviction_mlx(
    k_cache: "mx.array",
    v_cache: "mx.array",
    schedule: DecaySchedule,
) -> tuple["mx.array", "mx.array"]:
    """Apply an eviction schedule to mx.array KV cache tensors.

    Drops token positions flagged for eviction along the seq_len axis (axis 2).
    Positions NOT in evict_mask are retained in original order.

    Args:
        k_cache: mx.array shape (num_layers, num_heads, seq_len, head_dim).
        v_cache: mx.array same shape.
        schedule: DecaySchedule for seq_len tokens. evict_mask is boolean,
            True = evict. len(schedule.positions) must equal seq_len.

    Returns:
        (k_retained, v_retained) with shape
        (num_layers, num_heads, retained_count, head_dim).
        If no tokens are evicted, returns the originals unchanged.
    """
    if not _MLX_AVAILABLE:
        raise ImportError("mlx is required for apply_eviction_mlx")

    if schedule.evicted_count == 0:
        return k_cache, v_cache

    retain_indices = np.where(~schedule.evict_mask)[0]
    idx = mx.array(retain_indices, dtype=mx.int32)

    # Index along seq_len axis (axis 2)
    k_retained = k_cache[:, :, idx, :]
    v_retained = v_cache[:, :, idx, :]
    mx.eval(k_retained, v_retained)
    return k_retained, v_retained


def decay_scores_mlx(seq_len: int, config: DecayConfig) -> "mx.array":
    """Compute decay scores as an mx.array.

    Convenience wrapper: computes NumPy scores and returns as mx.array.

    Args:
        seq_len: Sequence length.
        config: DecayConfig.

    Returns:
        mx.array float32, shape (seq_len,). Newest token = 1.0.
    """
    if not _MLX_AVAILABLE:
        raise ImportError("mlx is required for decay_scores_mlx")

    scheduler = TemporalDecayScheduler(config)
    scores_np = scheduler.decay_scores(seq_len)
    return mx.array(scores_np)


__all__ = [
    "DecayConfig",
    "DecayMode",
    "DecaySchedule",
    "TemporalDecayScheduler",
    "apply_eviction_mlx",
    "decay_scores_mlx",
]
