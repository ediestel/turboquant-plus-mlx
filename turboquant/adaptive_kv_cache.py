"""AdaptiveKVCacheCompressor: composing orchestrator for KV cache compression.

Unlike KVCacheCompressor (which stores feature objects as metadata but applies only
the default quantizers), AdaptiveKVCacheCompressor actually wires the three TurboQuant+
extensions into compress():

  - BitBudgetPolicy → per-(layer, head) quantizer selection
  - TieredDecayPolicy / TemporalDecayScheduler → head-level bit-width cap from token ages
  - MoECompressionRouter → live expert routing stats → per-layer bit plan

Key design invariants:
  - _policy is the authoritative source; _allocation_plan is its cached materialization.
  - _normalize_bits() is the single canonical rounding rule (prevents compress/decompress
    mismatch).
  - CompressedKVCache.k_bits_matrix / v_bits_matrix record effective integer bit widths
    so decompress() can reconstruct using the exact quantizers used during compress().
  - MoE and decay operate as caps (min), not unconditional overrides, so prior policy
    information is preserved.
  - Head→expert round-robin is a placeholder mapping; see NOTE in compress().
  - Decay is applied at head granularity (mean retained-token bits); per-token
    requantization belongs to future fused-kernel work.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from turboquant.kv_cache import KVCacheCompressor, CompressedKVCache
from turboquant.adaptive_bits import AllocationPlan
from turboquant.temporal_decay import TemporalDecayScheduler
from turboquant.moe_compression import MoECompressionRouter, ExpertRoutingStats
from turboquant.bit_budget import BitBudgetPolicy, SlotKey


# ---------------------------------------------------------------------------
# Shared bit-normalisation helper
# ---------------------------------------------------------------------------

def _normalize_bits(bits: float) -> int:
    """Canonical rounding from float target to integer quantizer bit-width.

    Clips to [1, 8]. Used in compress() and _get_k/v_quantizer() to ensure
    the same integer is used when writing into bit matrices and when selecting
    quantizers — preventing compress/decompress mismatches.
    """
    return int(np.clip(round(bits), 1, 8))


# ---------------------------------------------------------------------------
# Policy → AllocationPlan conversion
# ---------------------------------------------------------------------------

def _policy_to_plan(
    policy: BitBudgetPolicy, n_layers: int, n_heads: int
) -> AllocationPlan:
    """Convert any BitBudgetPolicy to an AllocationPlan.

    Fast-path: if policy exposes to_allocation_plan(), call it directly to
    avoid O(n_layers * n_heads) Python-level slot iteration.
    Fallback: iterate all slots.
    """
    if hasattr(policy, "to_allocation_plan"):
        return policy.to_allocation_plan(n_layers, n_heads)

    k_bits = np.zeros((n_layers, n_heads), dtype=np.float64)
    v_bits = np.zeros((n_layers, n_heads), dtype=np.float64)
    for layer in range(n_layers):
        for head in range(n_heads):
            slot = SlotKey(layer, head)
            k_bits[layer, head] = policy.k_bits_for(slot)
            v_bits[layer, head] = policy.v_bits_for(slot)
    return AllocationPlan(k_bits=k_bits, v_bits=v_bits)


# ---------------------------------------------------------------------------
# Tiered decay policy
# ---------------------------------------------------------------------------

class TieredDecayPolicy:
    """Step-wise token-age decay: maps decay score → bit width by threshold lookup.

    Args:
        decay_tiers: List of (score_threshold, bits) pairs. Sorted descending by
            threshold at construction. Tokens with score < threshold[i] get bits[i].
            Tokens above all thresholds get base_bits.
        base_bits: Bit-width for tokens with high decay scores (young tokens).

    NOTE: AdaptiveKVCacheCompressor applies this at head granularity — the mean
    retained-token score determines a single effective bit-width for the whole head.
    True per-token requantization requires future fused-kernel work.
    """

    def __init__(
        self, decay_tiers: list[tuple[float, float]], base_bits: float
    ):
        self.tiers = sorted(decay_tiers, key=lambda x: x[0], reverse=True)
        self.base_bits = float(base_bits)

    def bits_for_score(self, score: float) -> float:
        """Look up bit-width for a given decay score.

        Iterates thresholds in ascending order: returns the bits corresponding
        to the lowest threshold that the score still falls below.
        E.g., tiers=[(0.5, 3.0), (0.2, 2.0)]:
          score 0.4 → 3.0 (below 0.5 but not below 0.2)
          score 0.1 → 2.0 (below 0.2, the lowest tier)
        """
        for threshold, bits in reversed(self.tiers):  # ascending threshold order
            if score < threshold:
                return float(bits)
        return self.base_bits

    def mean_bits_for_schedule(self, decay_sched) -> float:
        """Mean effective bit-width across retained tokens using tier lookup.

        Returns base_bits if all tokens are evicted.
        """
        retained_mask = ~decay_sched.evict_mask
        if not retained_mask.any():
            return self.base_bits
        scores = decay_sched.decay_scores[retained_mask]
        per_token = np.array([self.bits_for_score(float(s)) for s in scores])
        return float(np.mean(per_token))


# ---------------------------------------------------------------------------
# AdaptiveKVCacheCompressor
# ---------------------------------------------------------------------------

class AdaptiveKVCacheCompressor(KVCacheCompressor):
    """KVCacheCompressor that actually applies adaptive bits, decay, and MoE in compress().

    The base KVCacheCompressor stores these feature objects but never calls them.
    This subclass overrides compress() and decompress() to use them.

    Args:
        head_dim: Attention head dimension.
        n_layers: Number of transformer layers. Required when policy is provided.
        n_heads: Number of attention heads. Required when policy is provided.
        k_bits: Default K-cache bit-width (used when no policy/allocation_plan active).
        v_bits: Default V-cache bit-width.
        policy: BitBudgetPolicy (authoritative). Materialized into _allocation_plan at init.
        decay_tiers: Step-wise decay tiers [(score_threshold, bits)]. Stored as
            TieredDecayPolicy; does NOT replace decay_scheduler if both provided.
        moe_adapter: MoECompressionRouter for live routing stat updates via
            update_moe_routing(). Routing stats accumulated in _moe_stats.
        moe_n_experts: Number of experts per MoE layer (for update_moe_routing bootstrap).
        moe_n_experts_used: Experts activated per token (top-k).
        seed: Random seed.
        norm_correction: Apply norm correction on dequantization.
        allocation_plan: Optional AllocationPlan (used only if policy is None).
        decay_scheduler: Optional TemporalDecayScheduler (used if decay_tiers is None).
        moe_plans: Optional pre-built MoE plans dict[layer_idx -> MoEBitPlan].
    """

    def __init__(
        self,
        head_dim: int,
        n_layers: int = None,
        n_heads: int = None,
        k_bits: int = 3,
        v_bits: int = 3,
        policy: Optional[BitBudgetPolicy] = None,
        decay_tiers: Optional[list] = None,
        moe_adapter: Optional[MoECompressionRouter] = None,
        moe_n_experts: int = 8,
        moe_n_experts_used: int = 2,
        seed: int = 42,
        norm_correction: bool = True,
        allocation_plan: Optional[AllocationPlan] = None,
        decay_scheduler: Optional[TemporalDecayScheduler] = None,
        moe_plans: Optional[dict] = None,
    ):
        if policy is not None:
            if n_layers is None or n_heads is None:
                raise ValueError(
                    "n_layers and n_heads are required when policy is provided"
                )
            allocation_plan = _policy_to_plan(policy, n_layers, n_heads)

        super().__init__(
            head_dim=head_dim,
            k_bits=k_bits,
            v_bits=v_bits,
            seed=seed,
            norm_correction=norm_correction,
            allocation_plan=allocation_plan,
            decay_scheduler=decay_scheduler,
            moe_plans=moe_plans,
        )

        self._policy = policy
        self._n_layers = n_layers
        self._n_heads = n_heads
        self._tiered_decay: Optional[TieredDecayPolicy] = (
            TieredDecayPolicy(decay_tiers, float(v_bits)) if decay_tiers is not None else None
        )
        self._moe_adapter = moe_adapter
        self._moe_n_experts = moe_n_experts
        self._moe_n_experts_used = moe_n_experts_used
        self._moe_stats: dict[int, ExpertRoutingStats] = {}

    # ------------------------------------------------------------------
    # Override compress() to actually apply all features
    # ------------------------------------------------------------------

    def compress(
        self, k_cache: np.ndarray, v_cache: np.ndarray
    ) -> CompressedKVCache:
        """Compress with per-slot bit selection applied.

        Decision order for each (layer, head):
          1. Base bits from policy/allocation_plan (or k_bits/v_bits defaults).
          2. Temporal decay cap: min(base, mean_retained_bits_from_schedule).
          3. MoE cap: min(current, moe_bits) — never an unconditional override.
          4. Integerize via _normalize_bits(), select quantizer, compress.

        Stores effective integer bit-widths in result.k_bits_matrix / v_bits_matrix.

        NOTE (temporal decay): decay schedule is collapsed to a single per-head
        mean bit-width. True per-token requantization requires future fused-kernel work.

        NOTE (MoE): head→expert round-robin is a placeholder mapping until
        per-token or per-head expert routing metadata is available.
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

        k_bits_mat = np.zeros((num_layers, num_heads), dtype=np.int32)
        v_bits_mat = np.zeros((num_layers, num_heads), dtype=np.int32)

        for layer in range(num_layers):
            # Compute decay schedule once per layer (same seq_len for all heads)
            decay_sched = None
            if self._decay_scheduler is not None:
                decay_sched = self._decay_scheduler.schedule(seq_len)

            moe_plan = self._moe_plans.get(layer) if self._moe_plans else None

            k_layer = []
            v_layer_idx = []
            v_layer_norms = []

            for head in range(num_heads):
                # Step 1: base bits from allocation plan or defaults
                if self._allocation_plan is not None:
                    k_b = float(self._allocation_plan.k_bits[layer, head])
                    v_b = float(self._allocation_plan.v_bits[layer, head])
                else:
                    k_b = float(self.k_bits)
                    v_b = float(self.v_bits)

                # Step 2: temporal decay cap (head-level approximation)
                if self._tiered_decay is not None and decay_sched is not None:
                    mean_b = self._tiered_decay.mean_bits_for_schedule(decay_sched)
                    k_b = min(k_b, mean_b)
                    v_b = min(v_b, mean_b)
                elif decay_sched is not None:
                    retained_mask = ~decay_sched.evict_mask
                    if retained_mask.any():
                        mean_b = float(
                            np.mean(decay_sched.bits_per_token[retained_mask])
                        )
                        k_b = min(k_b, mean_b)
                        v_b = min(v_b, mean_b)

                # Step 3: MoE cap — min(), not unconditional override
                # NOTE: head→expert round-robin is a placeholder until per-token
                # routing metadata is available.
                if moe_plan is not None:
                    n_exp = len(moe_plan.expert_bits)
                    expert_id = head % n_exp
                    if not moe_plan.should_evict(expert_id):
                        moe_bits = moe_plan.bits_for(expert_id)
                        if moe_bits > 0:
                            k_b = min(k_b, moe_bits)
                            v_b = min(v_b, moe_bits)

                # Step 4: integerize and compress
                k_i = _normalize_bits(k_b)
                v_i = _normalize_bits(v_b)

                k_vecs = k_cache[layer, head]  # (seq_len, head_dim)
                v_vecs = v_cache[layer, head]

                k_layer.append(self._get_k_quantizer(k_i).quantize(k_vecs))
                v_idx, v_nrm = self._get_v_quantizer(v_i).quantize(v_vecs)
                v_layer_idx.append(v_idx)
                v_layer_norms.append(v_nrm)

                k_bits_mat[layer, head] = k_i
                v_bits_mat[layer, head] = v_i

            result.k_compressed.append(k_layer)
            result.v_indices.append(v_layer_idx)
            result.v_norms.append(v_layer_norms)

        result.k_bits_matrix = k_bits_mat
        result.v_bits_matrix = v_bits_mat
        return result

    # ------------------------------------------------------------------
    # Override decompress() to use stored bit matrices
    # ------------------------------------------------------------------

    def decompress(
        self, compressed: CompressedKVCache
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decompress using per-slot quantizers recorded during compress().

        If compressed.k_bits_matrix is set (produced by this compressor), each
        (layer, head) slot uses the exact quantizer that was used to compress it.
        If not set (legacy CompressedKVCache from base compressor), falls back to
        self.k_quantizer / self.v_quantizer.
        """
        if compressed.k_bits_matrix is None:
            # Legacy path: base compressor produced this — use default quantizers
            return super().decompress(compressed)

        k_cache = np.zeros((
            compressed.num_layers, compressed.num_heads,
            compressed.seq_len, compressed.head_dim,
        ))
        v_cache = np.zeros_like(k_cache)

        for layer in range(compressed.num_layers):
            for head in range(compressed.num_heads):
                k_i = int(compressed.k_bits_matrix[layer, head])
                v_i = int(compressed.v_bits_matrix[layer, head])

                k_cache[layer, head] = self._get_k_quantizer(k_i).dequantize(
                    compressed.k_compressed[layer][head]
                )
                v_cache[layer, head] = self._get_v_quantizer(v_i).dequantize(
                    compressed.v_indices[layer][head],
                    compressed.v_norms[layer][head],
                )

        return k_cache, v_cache

    # ------------------------------------------------------------------
    # Policy refresh
    # ------------------------------------------------------------------

    def refresh_policy(self) -> None:
        """Recompute _allocation_plan from the stored policy.

        Call after SensitivityCalibratedPolicy.calibrate() to propagate updated
        entropy-based allocations into this compressor. Has no effect if no
        policy was provided at construction.

        Raises:
            ValueError: If policy was provided but n_layers/n_heads are missing.
        """
        if self._policy is None:
            return
        if self._n_layers is None or self._n_heads is None:
            raise ValueError(
                "n_layers and n_heads are required for refresh_policy()"
            )
        self._allocation_plan = _policy_to_plan(
            self._policy, self._n_layers, self._n_heads
        )

    # ------------------------------------------------------------------
    # Live MoE routing update
    # ------------------------------------------------------------------

    def update_moe_routing(
        self,
        layer_idx: int,
        routing_indices: np.ndarray,
        decay_factor: float = 0.9,
    ) -> None:
        """Update MoE routing stats for a layer and rebuild its plan.

        Accumulates routing observations via EMA if prior stats exist, or
        bootstraps from scratch on first call for a given layer.

        Args:
            layer_idx: Which transformer layer to update.
            routing_indices: Expert indices from recent tokens (any shape, flattened).
            decay_factor: EMA weight on existing counts. In [0, 1].
        """
        if self._moe_adapter is None:
            return

        n_exp = self._moe_n_experts
        n_used = self._moe_n_experts_used

        new_partial = ExpertRoutingStats.from_routing_indices(
            routing_indices=routing_indices,
            n_experts=n_exp,
            n_experts_used=n_used,
            layer_idx=layer_idx,
        )

        existing = self._moe_stats.get(layer_idx)
        if existing is None:
            self._moe_stats[layer_idx] = new_partial
        else:
            self._moe_stats[layer_idx] = self._moe_adapter.update_stats(
                existing, routing_indices, decay_factor=decay_factor
            )

        self._moe_plans[layer_idx] = self._moe_adapter.plan(
            self._moe_stats[layer_idx]
        )

    # ------------------------------------------------------------------
    # Extended memory stats
    # ------------------------------------------------------------------

    def memory_stats(
        self, seq_len: int, num_layers: int, num_heads: int
    ) -> dict:
        """Memory stats with separate configured/applied flags.

        Adds _configured and _applied variants alongside legacy _enabled keys
        (retained for backward compatibility).
        """
        stats = super().memory_stats(seq_len, num_layers, num_heads)

        adaptive = self._policy is not None or self._allocation_plan is not None
        decay = (
            self._decay_scheduler is not None or self._tiered_decay is not None
        )
        moe = bool(self._moe_plans) or self._moe_adapter is not None

        stats["adaptive_bits_configured"] = adaptive
        stats["temporal_decay_configured"] = decay
        stats["moe_routing_configured"] = moe
        stats["adaptive_bits_applied"] = adaptive
        stats["temporal_decay_applied"] = decay
        stats["moe_routing_applied"] = moe
        # Legacy keys for backward compat
        stats["adaptive_bits_enabled"] = adaptive
        stats["temporal_decay_enabled"] = decay
        stats["moe_routing_enabled"] = moe

        return stats
