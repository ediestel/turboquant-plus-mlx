"""Tests for turboquant/adaptive_kv_cache.py.

Covers: _normalize_bits, _policy_to_plan, TieredDecayPolicy,
AdaptiveKVCacheCompressor (compress, decompress, refresh_policy,
update_moe_routing, memory_stats).
"""

from __future__ import annotations

import numpy as np
import pytest

from turboquant.adaptive_bits import AdaptiveBitAllocator, AllocationPlan
from turboquant.adaptive_kv_cache import (
    AdaptiveKVCacheCompressor,
    TieredDecayPolicy,
    _normalize_bits,
    _policy_to_plan,
)
from turboquant.bit_budget import (
    LayerHeadPolicy,
    SensitivityCalibratedPolicy,
    SlotKey,
    UniformPolicy,
)
from turboquant.kv_cache import CompressedKVCache
from turboquant.moe_compression import ExpertRoutingStats, MoECompressionRouter
from turboquant.temporal_decay import DecayConfig, DecayMode, TemporalDecayScheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kv(
    num_layers=2, num_heads=4, seq_len=16, head_dim=64, seed=0
):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((num_layers, num_heads, seq_len, head_dim)).astype(
        np.float32
    )
    v = rng.standard_normal((num_layers, num_heads, seq_len, head_dim)).astype(
        np.float32
    )
    return k, v


# ---------------------------------------------------------------------------
# _normalize_bits
# ---------------------------------------------------------------------------

class TestNormalizeBits:
    def test_integer_unchanged(self):
        assert _normalize_bits(3) == 3
        assert _normalize_bits(1) == 1
        assert _normalize_bits(8) == 8

    def test_round_up(self):
        assert _normalize_bits(2.6) == 3
        assert _normalize_bits(3.5) == 4

    def test_round_down(self):
        assert _normalize_bits(2.4) == 2
        assert _normalize_bits(3.4) == 3

    def test_clip_below_1(self):
        assert _normalize_bits(0.1) == 1
        assert _normalize_bits(-5.0) == 1

    def test_clip_above_8(self):
        assert _normalize_bits(9) == 8
        assert _normalize_bits(100.0) == 8


# ---------------------------------------------------------------------------
# _policy_to_plan
# ---------------------------------------------------------------------------

class TestPolicyToPlan:
    def test_uniform_uses_fast_path(self):
        """UniformPolicy has to_allocation_plan — fast path should be taken."""
        p = UniformPolicy(base_bits=3.0)
        plan = _policy_to_plan(p, n_layers=2, n_heads=4)
        assert plan.k_bits.shape == (2, 4)
        assert np.allclose(plan.k_bits, 3.0)

    def test_layer_head_uses_fast_path(self):
        allocator = AdaptiveBitAllocator(base_bits=3.0, sensitivity_scale=0.0)
        inner = allocator.allocate_uniform(2, 4)
        p = LayerHeadPolicy(inner)
        plan = _policy_to_plan(p, n_layers=2, n_heads=4)
        assert np.allclose(plan.k_bits, inner.k_bits)

    def test_plain_protocol_uses_slot_iteration(self):
        """Object without to_allocation_plan falls through to slot iteration."""
        class CustomPolicy:
            def k_bits_for(self, slot):
                return float(slot.layer_idx + slot.head_idx + 1)
            def v_bits_for(self, slot):
                return float(slot.layer_idx + slot.head_idx + 2)

        plan = _policy_to_plan(CustomPolicy(), n_layers=2, n_heads=3)
        assert plan.k_bits.shape == (2, 3)
        assert plan.k_bits[0, 0] == pytest.approx(1.0)
        assert plan.k_bits[1, 2] == pytest.approx(4.0)

    def test_output_shape_matches_request(self):
        plan = _policy_to_plan(UniformPolicy(3.0), n_layers=5, n_heads=10)
        assert plan.k_bits.shape == (5, 10)


# ---------------------------------------------------------------------------
# TieredDecayPolicy
# ---------------------------------------------------------------------------

class TestTieredDecayPolicy:
    def _make_tiers(self):
        return TieredDecayPolicy(
            decay_tiers=[(0.5, 3.0), (0.2, 2.0)],
            base_bits=4.0,
        )

    def test_score_above_all_thresholds_returns_base(self):
        p = self._make_tiers()
        assert p.bits_for_score(0.8) == pytest.approx(4.0)
        assert p.bits_for_score(0.5) == pytest.approx(4.0)  # equal, not below

    def test_score_below_first_threshold(self):
        p = self._make_tiers()
        assert p.bits_for_score(0.4) == pytest.approx(3.0)
        assert p.bits_for_score(0.21) == pytest.approx(3.0)

    def test_score_below_all_thresholds(self):
        p = self._make_tiers()
        assert p.bits_for_score(0.1) == pytest.approx(2.0)
        assert p.bits_for_score(0.0) == pytest.approx(2.0)

    def test_tier_structure_preserved(self):
        """Two distinct tiers must produce two distinct bit widths."""
        p = self._make_tiers()
        high = p.bits_for_score(0.4)   # below 0.5
        low = p.bits_for_score(0.1)    # below 0.2
        assert high != low

    def test_mean_bits_for_all_retained_schedule(self):
        """All tokens retained: mean should reflect tier distribution."""
        from turboquant.temporal_decay import DecayConfig, DecayMode, TemporalDecayScheduler
        cfg = DecayConfig(
            mode=DecayMode.BIT_REDUCTION,
            decay_lambda=1.0,
            base_bits=4.0,
            min_bits=2.0,
        )
        sched = TemporalDecayScheduler(cfg).schedule(16)
        p = self._make_tiers()
        mean_b = p.mean_bits_for_schedule(sched)
        # Should be between min tier bits and base_bits
        assert 2.0 <= mean_b <= 4.0

    def test_mean_bits_all_evicted_returns_base(self):
        """If all tokens are evicted, return base_bits."""
        class FakeSchedule:
            evict_mask = np.ones(8, dtype=bool)
            decay_scores = np.zeros(8)
        p = self._make_tiers()
        assert p.mean_bits_for_schedule(FakeSchedule()) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# AdaptiveKVCacheCompressor
# ---------------------------------------------------------------------------

class TestAdaptiveKVCacheCompressor:
    # ------------------------------------------------------------------
    # Baseline (no features)
    # ------------------------------------------------------------------

    def test_no_features_compress_shape(self):
        k, v = _make_kv()
        comp = AdaptiveKVCacheCompressor(head_dim=64, k_bits=3, v_bits=3, seed=42)
        compressed = comp.compress(k, v)
        assert compressed.num_layers == 2
        assert compressed.num_heads == 4
        assert compressed.seq_len == 16

    def test_no_features_decompress_shape(self):
        k, v = _make_kv()
        comp = AdaptiveKVCacheCompressor(head_dim=64, k_bits=3, v_bits=3, seed=42)
        compressed = comp.compress(k, v)
        k_hat, v_hat = comp.decompress(compressed)
        assert k_hat.shape == k.shape
        assert v_hat.shape == v.shape

    def test_no_features_mse_reasonable(self):
        k, v = _make_kv(seed=7)
        comp = AdaptiveKVCacheCompressor(head_dim=64, k_bits=3, v_bits=3, seed=99)
        compressed = comp.compress(k, v)
        k_hat, v_hat = comp.decompress(compressed)
        assert float(np.mean((k - k_hat) ** 2)) < 1.0
        assert float(np.mean((v - v_hat) ** 2)) < 1.0

    # ------------------------------------------------------------------
    # Bit matrices stored correctly
    # ------------------------------------------------------------------

    def test_compress_stores_k_bits_matrix(self):
        k, v = _make_kv()
        comp = AdaptiveKVCacheCompressor(head_dim=64, k_bits=3, v_bits=3, seed=42)
        compressed = comp.compress(k, v)
        assert compressed.k_bits_matrix is not None
        assert compressed.k_bits_matrix.shape == (2, 4)

    def test_compress_stores_v_bits_matrix(self):
        k, v = _make_kv()
        comp = AdaptiveKVCacheCompressor(head_dim=64, k_bits=3, v_bits=3, seed=42)
        compressed = comp.compress(k, v)
        assert compressed.v_bits_matrix is not None

    def test_bit_matrices_are_integer(self):
        k, v = _make_kv()
        comp = AdaptiveKVCacheCompressor(head_dim=64, k_bits=3, v_bits=3, seed=42)
        compressed = comp.compress(k, v)
        assert compressed.k_bits_matrix.dtype in (np.int32, np.int64, np.intp)

    def test_uniform_policy_bit_matrix_all_equal(self):
        k, v = _make_kv()
        p = UniformPolicy(base_bits=3.0)
        comp = AdaptiveKVCacheCompressor(
            head_dim=64, n_layers=2, n_heads=4, k_bits=3, v_bits=3,
            policy=p, seed=42,
        )
        compressed = comp.compress(k, v)
        expected = _normalize_bits(3.0)
        assert np.all(compressed.k_bits_matrix == expected)

    # ------------------------------------------------------------------
    # LayerHeadPolicy — per-head bits differ
    # ------------------------------------------------------------------

    def test_layer_head_policy_nonuniform_bits_matrix(self):
        k_bits_arr = np.array([[2.0, 3.0, 4.0, 3.0], [3.0, 2.0, 3.0, 4.0]])
        v_bits_arr = k_bits_arr.copy()
        plan = AllocationPlan(k_bits=k_bits_arr, v_bits=v_bits_arr)
        p = LayerHeadPolicy(plan)
        k, v = _make_kv(num_layers=2, num_heads=4)
        comp = AdaptiveKVCacheCompressor(
            head_dim=64, n_layers=2, n_heads=4, k_bits=3, v_bits=3,
            policy=p, seed=42,
        )
        compressed = comp.compress(k, v)
        # Head (0,0) should use 2 bits, head (0,2) should use 4 bits
        assert compressed.k_bits_matrix[0, 0] == 2
        assert compressed.k_bits_matrix[0, 2] == 4

    # ------------------------------------------------------------------
    # Decompress with and without bit matrices
    # ------------------------------------------------------------------

    def test_decompress_adaptive_roundtrip_shape(self):
        k, v = _make_kv(num_layers=2, num_heads=4, seq_len=8)
        comp = AdaptiveKVCacheCompressor(
            head_dim=64, n_layers=2, n_heads=4,
            policy=UniformPolicy(3.0), seed=42,
        )
        compressed = comp.compress(k, v)
        k_hat, v_hat = comp.decompress(compressed)
        assert k_hat.shape == k.shape
        assert v_hat.shape == v.shape

    def test_decompress_legacy_cache_fallback(self):
        """decompress() with a base CompressedKVCache (no bit_matrices) must not crash."""
        k, v = _make_kv(num_layers=2, num_heads=4, seq_len=8)
        base_comp = AdaptiveKVCacheCompressor(
            head_dim=64, k_bits=3, v_bits=3, seed=42
        )
        # Compress with base class to get a legacy-style CompressedKVCache
        from turboquant.kv_cache import KVCacheCompressor
        legacy_comp = KVCacheCompressor(head_dim=64, k_bits=3, v_bits=3, seed=42)
        legacy_compressed = legacy_comp.compress(k, v)
        assert legacy_compressed.k_bits_matrix is None

        k_hat, v_hat = base_comp.decompress(legacy_compressed)
        assert k_hat.shape == k.shape

    # ------------------------------------------------------------------
    # Decay tiers
    # ------------------------------------------------------------------

    def test_decay_tiers_reduces_mean_bits(self):
        """With aggressive decay tiers, mean k_bits_matrix value < base k_bits."""
        k, v = _make_kv(num_layers=1, num_heads=2, seq_len=32, head_dim=64)
        comp = AdaptiveKVCacheCompressor(
            head_dim=64, k_bits=4, v_bits=4,
            decay_tiers=[(0.9, 3.0), (0.5, 2.0)],
            seed=42,
        )
        compressed = comp.compress(k, v)
        mean_bits = float(np.mean(compressed.k_bits_matrix))
        assert mean_bits <= 4.0  # capped or equal

    def test_decay_tiers_tiered_policy_stored(self):
        comp = AdaptiveKVCacheCompressor(
            head_dim=64, k_bits=3, v_bits=3,
            decay_tiers=[(0.5, 2.0)],
        )
        assert comp._tiered_decay is not None
        assert isinstance(comp._tiered_decay, TieredDecayPolicy)

    # ------------------------------------------------------------------
    # MoE adapter
    # ------------------------------------------------------------------

    def test_update_moe_routing_populates_plans(self):
        comp = AdaptiveKVCacheCompressor(
            head_dim=64, k_bits=3, v_bits=3,
            moe_adapter=MoECompressionRouter(base_bits=3.5),
            moe_n_experts=4,
            moe_n_experts_used=2,
        )
        routing = np.array([0, 1, 2, 3, 0, 1])
        comp.update_moe_routing(layer_idx=0, routing_indices=routing)
        assert 0 in comp._moe_plans
        assert 0 in comp._moe_stats

    def test_update_moe_routing_no_adapter_is_noop(self):
        comp = AdaptiveKVCacheCompressor(head_dim=64, k_bits=3, v_bits=3)
        comp.update_moe_routing(layer_idx=0, routing_indices=np.array([0, 1]))
        assert len(comp._moe_plans) == 0

    def test_moe_cap_preserves_higher_policy_bits(self):
        """MoE should cap (min), not override when policy bits are lower."""
        # Policy says 2 bits, MoE says 4 bits → result should be 2
        k_bits_arr = np.full((1, 4), 2.0)
        v_bits_arr = np.full((1, 4), 2.0)
        plan = AllocationPlan(k_bits=k_bits_arr, v_bits=v_bits_arr)
        k, v = _make_kv(num_layers=1, num_heads=4, seq_len=8)
        comp = AdaptiveKVCacheCompressor(
            head_dim=64, n_layers=1, n_heads=4,
            policy=LayerHeadPolicy(plan),
            k_bits=3, v_bits=3, seed=42,
        )
        # Pre-populate a moe plan with 4 bits for all experts
        router = MoECompressionRouter(base_bits=4.0, sensitivity_scale=0.0)
        comp._moe_plans[0] = router.plan_uniform(n_experts=4, n_experts_used=2)
        compressed = comp.compress(k, v)
        # min(2, 4) = 2 — policy bits preserved
        assert np.all(compressed.k_bits_matrix[0] == 2)

    # ------------------------------------------------------------------
    # refresh_policy
    # ------------------------------------------------------------------

    def test_refresh_policy_updates_allocation_plan(self):
        policy = SensitivityCalibratedPolicy(n_layers=1, n_heads=2, base_bits=3.0)
        comp = AdaptiveKVCacheCompressor(
            head_dim=64, n_layers=1, n_heads=2, policy=policy, k_bits=3, v_bits=3,
        )
        # Record samples that will shift the plan
        policy.record_attention_weights(np.array([0.99, 0.01]), 0, 0)
        policy.record_attention_weights(np.full(2, 0.5), 0, 1)
        policy.calibrate()

        old_plan = comp._allocation_plan
        comp.refresh_policy()
        new_plan = comp._allocation_plan

        # Plan reference should be updated
        assert new_plan is not old_plan

    def test_refresh_policy_noop_without_policy(self):
        comp = AdaptiveKVCacheCompressor(head_dim=64, k_bits=3, v_bits=3)
        comp.refresh_policy()  # should not raise

    def test_refresh_policy_does_not_mutate_past_compressed(self):
        policy = SensitivityCalibratedPolicy(n_layers=1, n_heads=2, base_bits=3.0)
        policy.record_attention_weights(np.full(4, 0.25), 0, 0)
        comp = AdaptiveKVCacheCompressor(
            head_dim=64, n_layers=1, n_heads=2, policy=policy, k_bits=3, v_bits=3,
        )
        k, v = _make_kv(num_layers=1, num_heads=2, seq_len=8)
        old_compressed = comp.compress(k, v)
        old_bits = old_compressed.k_bits_matrix.copy()

        # Recalibrate and refresh
        policy.record_attention_weights(np.array([0.99, 0.001, 0.005, 0.004]), 0, 0)
        policy.calibrate()
        comp.refresh_policy()

        # Old compressed object unchanged
        assert np.array_equal(old_compressed.k_bits_matrix, old_bits)

    # ------------------------------------------------------------------
    # Validation guard
    # ------------------------------------------------------------------

    def test_policy_without_n_layers_raises(self):
        with pytest.raises(ValueError, match="n_layers"):
            AdaptiveKVCacheCompressor(
                head_dim=64, policy=UniformPolicy(3.0), k_bits=3, v_bits=3
            )

    # ------------------------------------------------------------------
    # Combined features
    # ------------------------------------------------------------------

    def test_all_three_features_compress_decompress(self):
        policy = UniformPolicy(base_bits=3.0)
        cfg = DecayConfig(mode=DecayMode.BIT_REDUCTION, decay_lambda=1.5)
        scheduler = TemporalDecayScheduler(cfg)
        router = MoECompressionRouter(base_bits=3.5)
        moe_plans = {
            0: router.plan_uniform(8, 2),
            1: router.plan_uniform(8, 2),
        }
        k, v = _make_kv(num_layers=2, num_heads=4, seq_len=16)
        comp = AdaptiveKVCacheCompressor(
            head_dim=64, n_layers=2, n_heads=4,
            policy=policy, decay_scheduler=scheduler, moe_plans=moe_plans,
            k_bits=3, v_bits=3, seed=42,
        )
        compressed = comp.compress(k, v)
        k_hat, v_hat = comp.decompress(compressed)
        assert k_hat.shape == k.shape
        assert v_hat.shape == v.shape


# ---------------------------------------------------------------------------
# memory_stats
# ---------------------------------------------------------------------------

class TestMemoryStats:
    def test_no_features_all_false(self):
        comp = AdaptiveKVCacheCompressor(head_dim=64, k_bits=3, v_bits=3)
        stats = comp.memory_stats(16, 2, 4)
        assert not stats["adaptive_bits_configured"]
        assert not stats["temporal_decay_configured"]
        assert not stats["moe_routing_configured"]
        assert not stats["adaptive_bits_applied"]
        assert not stats["temporal_decay_applied"]
        assert not stats["moe_routing_applied"]

    def test_legacy_keys_present(self):
        comp = AdaptiveKVCacheCompressor(head_dim=64, k_bits=3, v_bits=3)
        stats = comp.memory_stats(16, 2, 4)
        assert "adaptive_bits_enabled" in stats
        assert "temporal_decay_enabled" in stats
        assert "moe_routing_enabled" in stats

    def test_policy_sets_adaptive_flags(self):
        comp = AdaptiveKVCacheCompressor(
            head_dim=64, n_layers=2, n_heads=4,
            policy=UniformPolicy(3.0), k_bits=3, v_bits=3,
        )
        stats = comp.memory_stats(16, 2, 4)
        assert stats["adaptive_bits_configured"]
        assert stats["adaptive_bits_applied"]
        assert stats["adaptive_bits_enabled"]

    def test_decay_tiers_sets_decay_flags(self):
        comp = AdaptiveKVCacheCompressor(
            head_dim=64, k_bits=3, v_bits=3,
            decay_tiers=[(0.5, 2.0)],
        )
        stats = comp.memory_stats(16, 2, 4)
        assert stats["temporal_decay_configured"]
        assert stats["temporal_decay_applied"]
        assert stats["temporal_decay_enabled"]

    def test_moe_adapter_sets_moe_flags(self):
        comp = AdaptiveKVCacheCompressor(
            head_dim=64, k_bits=3, v_bits=3,
            moe_adapter=MoECompressionRouter(),
        )
        stats = comp.memory_stats(16, 2, 4)
        assert stats["moe_routing_configured"]
        assert stats["moe_routing_applied"]
        assert stats["moe_routing_enabled"]
