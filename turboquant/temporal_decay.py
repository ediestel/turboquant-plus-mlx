"""Temporal decay scheduling for KV cache compression.

Old KV cache tokens are progressively assigned lower bit-widths (BIT_REDUCTION),
evicted (EVICTION), or both (HYBRID) as they age. This reduces memory pressure at
long contexts where the majority of cached tokens are old.

Decay formula:
    positions = np.arange(seq_len)           # 0 = oldest, seq_len-1 = newest
    max_pos   = max(seq_len - 1, 1)
    age       = (max_pos - positions) / max_pos  # newest → 0.0, oldest → 1.0
    score     = exp(-decay_lambda * age)         # newest → 1.0, oldest → exp(-lambda)

Quality validation:  benchmarks/temporal_decay_prototype.py confirms cosine sim
>0.80 for 3→2 bit requantization.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class DecayMode(enum.Enum):
    BIT_REDUCTION = "bit_reduction"
    EVICTION = "eviction"
    HYBRID = "hybrid"


@dataclass
class DecayConfig:
    """Configuration for temporal decay scheduling.

    Args:
        decay_lambda: Exponential decay rate. Must be > 0.
        base_bits: Bit-width assigned to the newest token (decay score = 1.0).
        min_bits: Bit-width floor. Must satisfy 0 < min_bits < base_bits.
        eviction_threshold: Tokens with score < threshold are evicted (EVICTION/HYBRID).
            Must be in (0, 1) exclusive.
        reduction_threshold: Tokens with score < threshold get min_bits (BIT_REDUCTION).
            Must be in (0, 1) exclusive.
        window_size: If set, tokens older than the most recent ``window_size`` positions
            are always evicted regardless of score. Must be > 0 when set.
            window_size >= seq_len → no window eviction for that sequence.
        mode: Decay operating mode.
    """
    decay_lambda: float = 1.0
    base_bits: float = 3.5
    min_bits: float = 2.0
    eviction_threshold: float = 0.1
    reduction_threshold: float = 0.3
    window_size: Optional[int] = None
    mode: DecayMode = DecayMode.BIT_REDUCTION

    def __post_init__(self):
        if self.decay_lambda <= 0:
            raise ValueError(f"decay_lambda must be positive, got {self.decay_lambda}")
        if not (self.min_bits > 0 and self.min_bits < self.base_bits):
            raise ValueError(
                f"min_bits must satisfy 0 < min_bits < base_bits, "
                f"got min_bits={self.min_bits}, base_bits={self.base_bits}"
            )
        if not (0.0 < self.eviction_threshold < 1.0):
            raise ValueError(
                f"eviction_threshold must be in (0, 1), got {self.eviction_threshold}"
            )
        if not (0.0 < self.reduction_threshold < 1.0):
            raise ValueError(
                f"reduction_threshold must be in (0, 1), got {self.reduction_threshold}"
            )
        if self.window_size is not None and self.window_size <= 0:
            raise ValueError(
                f"window_size must be > 0 when set, got {self.window_size}"
            )


@dataclass
class DecaySchedule:
    """Output of the temporal decay scheduler for a given sequence length.

    Attributes:
        positions: Token positions, shape (seq_len,). 0 = oldest.
        decay_scores: Decay score per position in [0, 1]. shape (seq_len,).
        bits_per_token: Target bit-width per position. Evicted positions carry
            min_bits nominally but are masked out and not used. shape (seq_len,).
        evict_mask: True where token should be evicted. shape (seq_len,).
    """
    positions: np.ndarray
    decay_scores: np.ndarray
    bits_per_token: np.ndarray
    evict_mask: np.ndarray

    @property
    def retained_count(self) -> int:
        return int(np.sum(~self.evict_mask))

    @property
    def evicted_count(self) -> int:
        return int(np.sum(self.evict_mask))

    @property
    def mean_bits_retained(self) -> float:
        """Mean target bits over retained (non-evicted) tokens only.

        Returns 0.0 if all tokens are evicted.
        """
        retained = self.bits_per_token[~self.evict_mask]
        if len(retained) == 0:
            return 0.0
        return float(np.mean(retained))

    def summary(self) -> str:
        seq_len = len(self.positions)
        return (
            f"DecaySchedule(seq_len={seq_len}, "
            f"retained={self.retained_count}, "
            f"evicted={self.evicted_count}, "
            f"mean_bits_retained={self.mean_bits_retained:.2f})"
        )


class TemporalDecayScheduler:
    """Schedule per-token bit allocations and eviction decisions.

    Args:
        config: Decay configuration. Defaults to BIT_REDUCTION with lambda=1.0.
    """

    _DEFAULT_CONFIG = None  # lazy singleton

    def __init__(self, config: Optional[DecayConfig] = None):
        if config is None:
            if TemporalDecayScheduler._DEFAULT_CONFIG is None:
                TemporalDecayScheduler._DEFAULT_CONFIG = DecayConfig()
            config = TemporalDecayScheduler._DEFAULT_CONFIG
        self.config = config

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def decay_scores(
        self, seq_len: int, positions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute exponential decay scores.

        Newest token (position seq_len-1) → 1.0.
        Oldest token (position 0) → exp(-decay_lambda).

        Args:
            seq_len: Total sequence length.
            positions: Optional subset of positions to score. If None, scores all.

        Returns:
            Float32 array of shape (len(positions),) or (seq_len,).
        """
        if positions is None:
            pos = np.arange(seq_len, dtype=np.float64)
        else:
            pos = np.asarray(positions, dtype=np.float64)

        max_pos = seq_len - 1
        if max_pos <= 0:
            age = np.zeros_like(pos)  # single token is newest → age 0 → score 1.0
        else:
            age = (max_pos - pos) / max_pos
        scores = np.exp(-self.config.decay_lambda * age)
        return scores.astype(np.float32)

    def bits_from_decay(self, scores: np.ndarray) -> np.ndarray:
        """Map decay scores to target bit-widths via linear interpolation.

        score=0.0 → min_bits, score=1.0 → base_bits.

        Args:
            scores: Float array of decay scores in [0, 1].

        Returns:
            Float32 array of target bit-widths.
        """
        scores = np.asarray(scores, dtype=np.float64)
        cfg = self.config
        bits = cfg.min_bits + (cfg.base_bits - cfg.min_bits) * scores
        return bits.astype(np.float32)

    # ------------------------------------------------------------------
    # Schedule construction
    # ------------------------------------------------------------------

    def schedule(self, seq_len: int) -> DecaySchedule:
        """Build a full decay schedule for a sequence.

        Args:
            seq_len: Number of tokens in the sequence.

        Returns:
            DecaySchedule for all positions 0..seq_len-1.
        """
        positions = np.arange(seq_len, dtype=np.int32)
        scores = self.decay_scores(seq_len)
        bits = self.bits_from_decay(scores)
        evict_mask = np.zeros(seq_len, dtype=bool)

        cfg = self.config

        # Window eviction (applied first — highest precedence)
        if cfg.window_size is not None and cfg.window_size < seq_len:
            cutoff = seq_len - cfg.window_size
            evict_mask[:cutoff] = True

        if cfg.mode in (DecayMode.EVICTION, DecayMode.HYBRID):
            evict_mask |= scores < cfg.eviction_threshold

        if cfg.mode in (DecayMode.BIT_REDUCTION, DecayMode.HYBRID):
            low = scores < cfg.reduction_threshold
            bits[low & ~evict_mask] = cfg.min_bits

        if cfg.mode == DecayMode.BIT_REDUCTION:
            # No eviction in pure bit-reduction mode
            evict_mask[:] = False

        # Evicted positions carry min_bits nominally
        bits[evict_mask] = cfg.min_bits

        return DecaySchedule(
            positions=positions,
            decay_scores=scores,
            bits_per_token=bits,
            evict_mask=evict_mask,
        )

    def update_schedule(self, prev: DecaySchedule, new_seq_len: int) -> DecaySchedule:
        """Extend a schedule to a longer sequence using append semantics.

        Existing positions keep their prior ``evict_mask`` and ``bits_per_token``
        unchanged. New tokens (indices ``prev_len..new_seq_len-1``) receive
        decay scores near 1.0 and are not evicted. Previously evicted tokens
        do NOT become retained.

        Args:
            prev: Previous schedule of length ``prev_len``.
            new_seq_len: Target length. Must be >= len(prev.positions).

        Returns:
            New DecaySchedule of length ``new_seq_len``.
        """
        prev_len = len(prev.positions)
        if new_seq_len <= prev_len:
            return prev

        n_new = new_seq_len - prev_len

        # Scores for the full new sequence (oldest→newest in updated context)
        new_scores = self.decay_scores(new_seq_len)

        # Positions
        positions = np.arange(new_seq_len, dtype=np.int32)

        # bits: preserve old, compute fresh for new tokens
        new_bits_for_appended = self.bits_from_decay(new_scores[prev_len:])
        bits = np.concatenate([prev.bits_per_token, new_bits_for_appended])

        # evict_mask: preserve old decisions, new tokens not evicted
        evict_mask = np.concatenate([prev.evict_mask, np.zeros(n_new, dtype=bool)])

        return DecaySchedule(
            positions=positions,
            decay_scores=new_scores,
            bits_per_token=bits,
            evict_mask=evict_mask,
        )
