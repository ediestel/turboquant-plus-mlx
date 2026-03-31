"""Integration tests for KVCacheCompressor with adaptive bits, temporal decay,
and MoE-aware compression.

All new parameters are optional — the baseline (no new params) must behave
identically to the original implementation.
"""

import pytest
import numpy as np

from turboquant.kv_cache import KVCacheCompressor, CompressedKVCache
from turboquant.adaptive_bits import AdaptiveBitAllocator, SensitivityProfile
from turboquant.temporal_decay import DecayConfig, DecayMode, TemporalDecayScheduler
from turboquant.moe_compression import (
    ExpertRoutingStats, MoECompressionRouter, MoEBitPlan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kv(num_layers=2, num_heads=4, seq_len=16, head_dim=64, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((num_layers, num_heads, seq_len, head_dim)).astype(np.float32)
    v = rng.standard_normal((num_layers, num_heads, seq_len, head_dim)).astype(np.float32)
    return k, v


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """KVCacheCompressor with no new params must behave as before."""

    def test_compress_decompress_baseline(self):
        k, v = _make_kv()
        comp = KVCacheCompressor(head_dim=64, k_bits=3, v_bits=3, seed=42)
        compressed = comp.compress(k, v)
        k_hat, v_hat = comp.decompress(compressed)
        assert k_hat.shape == k.shape
        assert v_hat.shape == v.shape

    def test_memory_stats_baseline(self):
        comp = KVCacheCompressor(head_dim=64, k_bits=3, v_bits=3)
        stats = comp.memory_stats(seq_len=128, num_layers=4, num_heads=8)
        assert stats["compression_ratio"] > 1.0
        assert stats["k_bits_per_value"] == pytest.approx(3.0)
        assert not stats["adaptive_bits_enabled"]
        assert not stats["temporal_decay_enabled"]
        assert not stats["moe_routing_enabled"]

    def test_distortion_reasonable_baseline(self):
        k, v = _make_kv(num_layers=1, num_heads=2, seq_len=32, head_dim=64, seed=1)
        comp = KVCacheCompressor(head_dim=64, k_bits=3, v_bits=3, seed=99)
        compressed = comp.compress(k, v)
        k_hat, v_hat = comp.decompress(compressed)
        k_mse = float(np.mean((k - k_hat) ** 2))
        v_mse = float(np.mean((v - v_hat) ** 2))
        assert k_mse < 1.0
        assert v_mse < 1.0


# ---------------------------------------------------------------------------
# Adaptive bits integration
# ---------------------------------------------------------------------------

class TestAdaptiveBitsIntegration:
    def test_uniform_plan_matches_baseline(self):
        """Uniform adaptive plan should compress at the same base bit rate."""
        k, v = _make_kv(num_layers=2, num_heads=4, seq_len=16, head_dim=64)
        alloc = AdaptiveBitAllocator(base_bits=3.0, sensitivity_scale=0.0)
        plan = alloc.allocate_uniform(n_layers=2, n_heads=4)

        comp_adap = KVCacheCompressor(
            head_dim=64, k_bits=3, v_bits=3, seed=7, allocation_plan=plan
        )
        # Both should produce the same effective bit width
        stats_adap = comp_adap.memory_stats(16, 2, 4)
        assert stats_adap["adaptive_bits_enabled"]
        assert stats_adap["k_bits_per_value"] == pytest.approx(3.0, abs=0.01)

    def test_effective_bits_matrix_shape(self):
        alloc = AdaptiveBitAllocator(base_bits=3.5, sensitivity_scale=0.5)
        scores = np.array([[1.0, 3.0, 0.5, 2.0], [2.0, 0.5, 1.5, 1.0]])
        profile = SensitivityProfile(scores=scores, n_layers=2, n_heads=4)
        plan = alloc.allocate(profile)

        comp = KVCacheCompressor(head_dim=64, k_bits=3, v_bits=3, allocation_plan=plan)
        k_mat, v_mat = comp.effective_bits_matrix(num_layers=2, num_heads=4)
        assert k_mat.shape == (2, 4)
        assert v_mat.shape == (2, 4)

    def test_per_head_bits_differ_with_nonuniform_sensitivity(self):
        alloc = AdaptiveBitAllocator(base_bits=3.5, sensitivity_scale=1.0)
        scores = np.array([[1.0, 10.0, 0.1, 1.0]])
        profile = SensitivityProfile(scores=scores, n_layers=1, n_heads=4)
        plan = alloc.allocate(profile)

        comp = KVCacheCompressor(head_dim=64, k_bits=3, v_bits=3, allocation_plan=plan)
        k_mat, _ = comp.effective_bits_matrix(num_layers=1, num_heads=4)
        # Head 1 (highest sensitivity) should have more bits than head 2 (lowest)
        assert k_mat[0, 1] > k_mat[0, 2]

    def test_adaptive_compress_produces_valid_output(self):
        k, v = _make_kv(num_layers=2, num_heads=4, seq_len=8, head_dim=64)
        alloc = AdaptiveBitAllocator(base_bits=3.0, sensitivity_scale=0.5)
        plan = alloc.allocate_uniform(n_layers=2, n_heads=4)
        comp = KVCacheCompressor(
            head_dim=64, k_bits=3, v_bits=3, seed=42, allocation_plan=plan
        )
        compressed = comp.compress(k, v)
        k_hat, v_hat = comp.decompress(compressed)
        assert k_hat.shape == k.shape
        assert v_hat.shape == v.shape

    def test_uniform_sensitivity_gives_uniform_plan(self):
        alloc = AdaptiveBitAllocator(base_bits=3.5, sensitivity_scale=1.0)
        scores = np.ones((2, 4))
        profile = SensitivityProfile(scores=scores, n_layers=2, n_heads=4)
        plan = alloc.allocate(profile)
        assert np.allclose(plan.k_bits, 3.5)


# ---------------------------------------------------------------------------
# Temporal decay integration
# ---------------------------------------------------------------------------

class TestTemporalDecayIntegration:
    def test_eviction_mode_reduces_retained_tokens(self):
        cfg = DecayConfig(
            mode=DecayMode.EVICTION,
            decay_lambda=4.0,
            eviction_threshold=0.2,
        )
        scheduler = TemporalDecayScheduler(cfg)
        # Confirm that some tokens would be evicted
        sched = scheduler.schedule(32)
        assert sched.evicted_count > 0

    def test_temporal_decay_flag_in_memory_stats(self):
        cfg = DecayConfig(mode=DecayMode.EVICTION)
        scheduler = TemporalDecayScheduler(cfg)
        comp = KVCacheCompressor(
            head_dim=64, k_bits=3, v_bits=3, decay_scheduler=scheduler
        )
        stats = comp.memory_stats(32, 2, 4)
        assert stats["temporal_decay_enabled"]

    def test_bit_reduction_mode_compress_runs(self):
        k, v = _make_kv(num_layers=1, num_heads=2, seq_len=16, head_dim=64)
        cfg = DecayConfig(mode=DecayMode.BIT_REDUCTION, decay_lambda=1.0)
        scheduler = TemporalDecayScheduler(cfg)
        comp = KVCacheCompressor(
            head_dim=64, k_bits=3, v_bits=3, seed=42, decay_scheduler=scheduler
        )
        # Should not raise
        compressed = comp.compress(k, v)
        assert compressed is not None

    def test_no_decay_same_as_baseline(self):
        """Without temporal decay, output shape is identical to baseline."""
        k, v = _make_kv(num_layers=1, num_heads=2, seq_len=8, head_dim=64)
        comp_no_decay = KVCacheCompressor(head_dim=64, k_bits=3, v_bits=3, seed=42)
        compressed = comp_no_decay.compress(k, v)
        k_hat, v_hat = comp_no_decay.decompress(compressed)
        assert k_hat.shape == k.shape


# ---------------------------------------------------------------------------
# MoE routing integration
# ---------------------------------------------------------------------------

class TestMoERoutingIntegration:
    def _build_moe_plans(self, n_experts=8, n_used=2, n_layers=2):
        router = MoECompressionRouter(base_bits=3.5, sensitivity_scale=0.5)
        counts = np.array([50, 40, 30, 20, 10, 5, 2, 1], dtype=np.int64)
        plans = {}
        for layer in range(n_layers):
            stats = ExpertRoutingStats(
                n_experts=n_experts,
                n_experts_used=n_used,
                activation_counts=counts,
                layer_idx=layer,
            )
            plans[layer] = router.plan(stats)
        return plans

    def test_moe_routing_flag_in_memory_stats(self):
        moe_plans = self._build_moe_plans(n_layers=2)
        comp = KVCacheCompressor(
            head_dim=64, k_bits=3, v_bits=3, moe_plans=moe_plans
        )
        stats = comp.memory_stats(16, 2, 4)
        assert stats["moe_routing_enabled"]

    def test_moe_compress_runs_without_error(self):
        k, v = _make_kv(num_layers=2, num_heads=4, seq_len=8, head_dim=64)
        moe_plans = self._build_moe_plans(n_layers=2)
        comp = KVCacheCompressor(
            head_dim=64, k_bits=3, v_bits=3, seed=42, moe_plans=moe_plans
        )
        compressed = comp.compress(k, v)
        k_hat, v_hat = comp.decompress(compressed)
        assert k_hat.shape == k.shape
        assert v_hat.shape == v.shape

    def test_empty_moe_plans_is_backward_compatible(self):
        k, v = _make_kv()
        comp = KVCacheCompressor(head_dim=64, k_bits=3, v_bits=3, moe_plans={})
        stats = comp.memory_stats(16, 2, 4)
        assert not stats["moe_routing_enabled"]


# ---------------------------------------------------------------------------
# Combined features
# ---------------------------------------------------------------------------

class TestCombinedFeatures:
    def test_adaptive_plus_decay_runs(self):
        k, v = _make_kv(num_layers=2, num_heads=4, seq_len=16, head_dim=64)
        alloc = AdaptiveBitAllocator(base_bits=3.0, sensitivity_scale=0.3)
        plan = alloc.allocate_uniform(n_layers=2, n_heads=4)
        cfg = DecayConfig(mode=DecayMode.BIT_REDUCTION, decay_lambda=1.5)
        scheduler = TemporalDecayScheduler(cfg)
        comp = KVCacheCompressor(
            head_dim=64, k_bits=3, v_bits=3, seed=42,
            allocation_plan=plan, decay_scheduler=scheduler
        )
        compressed = comp.compress(k, v)
        assert compressed is not None

    def test_all_three_features_flags(self):
        alloc = AdaptiveBitAllocator(base_bits=3.0)
        plan = alloc.allocate_uniform(n_layers=2, n_heads=4)
        scheduler = TemporalDecayScheduler()
        moe_plans = {
            0: MoECompressionRouter().plan_uniform(8, 2),
            1: MoECompressionRouter().plan_uniform(8, 2),
        }
        comp = KVCacheCompressor(
            head_dim=64, k_bits=3, v_bits=3,
            allocation_plan=plan,
            decay_scheduler=scheduler,
            moe_plans=moe_plans,
        )
        stats = comp.memory_stats(32, 2, 4)
        assert stats["adaptive_bits_enabled"]
        assert stats["temporal_decay_enabled"]
        assert stats["moe_routing_enabled"]

    def test_quantizer_cache_reuses_instances(self):
        """Same bit width should return the same quantizer object (cache hit)."""
        comp = KVCacheCompressor(head_dim=64, k_bits=3, v_bits=3)
        q1 = comp._get_k_quantizer(3)
        q2 = comp._get_k_quantizer(3)
        assert q1 is q2

    def test_different_bit_widths_give_different_quantizers(self):
        comp = KVCacheCompressor(head_dim=64, k_bits=3, v_bits=3)
        q3 = comp._get_k_quantizer(3)
        q4 = comp._get_k_quantizer(4)
        assert q3 is not q4

    def test_float_bits_round_to_same_quantizer(self):
        """3.0 and 3 should hit the same cached quantizer."""
        comp = KVCacheCompressor(head_dim=64, k_bits=3, v_bits=3)
        q_int = comp._get_k_quantizer(3)
        q_float = comp._get_k_quantizer(3.0)
        assert q_int is q_float
