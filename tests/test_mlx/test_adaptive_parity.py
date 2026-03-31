"""Parity tests: AdaptiveKVCacheCompressorMLX matches KVCacheCompressor (NumPy).

Verifies that the MLX adaptive compressor produces results within floating-point
tolerance of the NumPy reference for all three feature flags.

Skipped automatically when MLX is not available (non-Apple-Silicon CI).
"""

import pytest
import numpy as np

try:
    import mlx.core as mx
    from turboquant.mlx.adaptive_kv_cache import AdaptiveKVCacheCompressorMLX
    from turboquant.mlx.temporal_decay import (
        apply_eviction_mlx, decay_scores_mlx, TemporalDecayScheduler,
    )
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")

from turboquant.kv_cache import KVCacheCompressor
from turboquant.adaptive_bits import AdaptiveBitAllocator, SensitivityProfile
from turboquant.temporal_decay import DecayConfig, DecayMode
from turboquant.moe_compression import ExpertRoutingStats, MoECompressionRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kv_np(num_layers=2, num_heads=4, seq_len=16, head_dim=64, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((num_layers, num_heads, seq_len, head_dim)).astype(np.float32)
    v = rng.standard_normal((num_layers, num_heads, seq_len, head_dim)).astype(np.float32)
    return k, v


# ---------------------------------------------------------------------------
# Feature flags parity
# ---------------------------------------------------------------------------

class TestMemoryStatsFlagParity:
    """MLX adaptive compressor reports same feature flags as NumPy version."""

    def test_baseline_flags_match(self):
        np_comp = KVCacheCompressor(head_dim=64, k_bits=3, v_bits=3)
        mlx_comp = AdaptiveKVCacheCompressorMLX(head_dim=64, k_bits=3, v_bits=3)

        np_stats = np_comp.memory_stats(16, 2, 4)
        mlx_stats = mlx_comp.memory_stats(16, 2, 4)

        assert mlx_stats["adaptive_bits_enabled"] == np_stats["adaptive_bits_enabled"]
        assert mlx_stats["temporal_decay_enabled"] == np_stats["temporal_decay_enabled"]
        assert mlx_stats["moe_routing_enabled"] == np_stats["moe_routing_enabled"]

    def test_adaptive_flag_set(self):
        alloc = AdaptiveBitAllocator(base_bits=3.0)
        plan = alloc.allocate_uniform(n_layers=2, n_heads=4)
        comp = AdaptiveKVCacheCompressorMLX(head_dim=64, k_bits=3, v_bits=3,
                                            allocation_plan=plan)
        assert comp.memory_stats(16, 2, 4)["adaptive_bits_enabled"]

    def test_decay_flag_set(self):
        scheduler = TemporalDecayScheduler(DecayConfig(mode=DecayMode.EVICTION))
        comp = AdaptiveKVCacheCompressorMLX(head_dim=64, k_bits=3, v_bits=3,
                                            decay_scheduler=scheduler)
        assert comp.memory_stats(16, 2, 4)["temporal_decay_enabled"]

    def test_moe_flag_set(self):
        router = MoECompressionRouter()
        plans = {0: router.plan_uniform(8, 2), 1: router.plan_uniform(8, 2)}
        comp = AdaptiveKVCacheCompressorMLX(head_dim=64, k_bits=3, v_bits=3,
                                            moe_plans=plans)
        assert comp.memory_stats(16, 2, 4)["moe_routing_enabled"]

    def test_empty_moe_plans_flag_false(self):
        comp = AdaptiveKVCacheCompressorMLX(head_dim=64, k_bits=3, v_bits=3,
                                            moe_plans={})
        assert not comp.memory_stats(16, 2, 4)["moe_routing_enabled"]

    def test_all_three_flags(self):
        alloc = AdaptiveBitAllocator(base_bits=3.0)
        plan = alloc.allocate_uniform(n_layers=2, n_heads=4)
        scheduler = TemporalDecayScheduler()
        router = MoECompressionRouter()
        plans = {0: router.plan_uniform(8, 2), 1: router.plan_uniform(8, 2)}

        comp = AdaptiveKVCacheCompressorMLX(
            head_dim=64, k_bits=3, v_bits=3,
            allocation_plan=plan,
            decay_scheduler=scheduler,
            moe_plans=plans,
        )
        stats = comp.memory_stats(16, 2, 4)
        assert stats["adaptive_bits_enabled"]
        assert stats["temporal_decay_enabled"]
        assert stats["moe_routing_enabled"]


# ---------------------------------------------------------------------------
# Compression ratio parity
# ---------------------------------------------------------------------------

class TestCompressionRatioParity:
    def test_baseline_ratio_matches_numpy(self):
        np_comp = KVCacheCompressor(head_dim=64, k_bits=3, v_bits=3)
        mlx_comp = AdaptiveKVCacheCompressorMLX(head_dim=64, k_bits=3, v_bits=3)

        np_stats = np_comp.memory_stats(128, 4, 8)
        mlx_stats = mlx_comp.memory_stats(128, 4, 8)

        assert mlx_stats["compression_ratio"] == pytest.approx(
            np_stats["compression_ratio"], rel=1e-4
        )


# ---------------------------------------------------------------------------
# effective_bits_matrix parity
# ---------------------------------------------------------------------------

class TestEffectiveBitsMatrixParity:
    def test_no_plan_returns_constant(self):
        comp = AdaptiveKVCacheCompressorMLX(head_dim=64, k_bits=3, v_bits=3)
        k_mat, v_mat = comp.effective_bits_matrix(2, 4)
        assert k_mat.shape == (2, 4)
        assert np.all(k_mat == 3.0)

    def test_with_plan_matches_numpy(self):
        alloc = AdaptiveBitAllocator(base_bits=3.5, sensitivity_scale=0.5)
        scores = np.array([[1.0, 3.0, 0.5, 2.0], [2.0, 0.5, 1.5, 1.0]])
        profile = SensitivityProfile(scores=scores, n_layers=2, n_heads=4)
        plan = alloc.allocate(profile)

        np_comp = KVCacheCompressor(head_dim=64, k_bits=3, v_bits=3,
                                    allocation_plan=plan)
        mlx_comp = AdaptiveKVCacheCompressorMLX(head_dim=64, k_bits=3, v_bits=3,
                                                allocation_plan=plan)

        np_k, _ = np_comp.effective_bits_matrix(2, 4)
        mlx_k, _ = mlx_comp.effective_bits_matrix(2, 4)
        assert np.allclose(np_k, mlx_k)


# ---------------------------------------------------------------------------
# Quantizer cache parity
# ---------------------------------------------------------------------------

class TestQuantizerCacheParity:
    def test_same_bits_same_instance(self):
        comp = AdaptiveKVCacheCompressorMLX(head_dim=64, k_bits=3, v_bits=3)
        q1 = comp._get_k_quantizer(3)
        q2 = comp._get_k_quantizer(3)
        assert q1 is q2

    def test_float_and_int_same_instance(self):
        comp = AdaptiveKVCacheCompressorMLX(head_dim=64, k_bits=3, v_bits=3)
        q_int = comp._get_k_quantizer(3)
        q_flt = comp._get_k_quantizer(3.0)
        assert q_int is q_flt

    def test_different_bits_different_instances(self):
        comp = AdaptiveKVCacheCompressorMLX(head_dim=64, k_bits=3, v_bits=3)
        q3 = comp._get_k_quantizer(3)
        q4 = comp._get_k_quantizer(4)
        assert q3 is not q4


# ---------------------------------------------------------------------------
# Compress / decompress roundtrip (shape parity)
# ---------------------------------------------------------------------------

class TestCompressDecompressParity:
    def test_output_shape_matches_numpy(self):
        k_np, v_np = _make_kv_np(num_layers=2, num_heads=4, seq_len=8, head_dim=64)
        k_mx = mx.array(k_np)
        v_mx = mx.array(v_np)

        np_comp = KVCacheCompressor(head_dim=64, k_bits=3, v_bits=3, seed=42)
        mlx_comp = AdaptiveKVCacheCompressorMLX(head_dim=64, k_bits=3, v_bits=3, seed=42)

        np_compressed = np_comp.compress(k_np, v_np)
        mlx_compressed = mlx_comp.compress(k_mx, v_mx)

        k_hat_np, v_hat_np = np_comp.decompress(np_compressed)
        k_hat_mlx, v_hat_mlx = mlx_comp.decompress(mlx_compressed)

        assert k_hat_np.shape == np.array(k_hat_mlx).shape
        assert v_hat_np.shape == np.array(v_hat_mlx).shape

    def test_mse_within_tolerance(self):
        """MLX and NumPy reconstructions should be close (same seed, same quantizers)."""
        k_np, v_np = _make_kv_np(num_layers=1, num_heads=2, seq_len=16, head_dim=64)
        k_mx = mx.array(k_np)
        v_mx = mx.array(v_np)

        np_comp = KVCacheCompressor(head_dim=64, k_bits=3, v_bits=3, seed=99)
        mlx_comp = AdaptiveKVCacheCompressorMLX(head_dim=64, k_bits=3, v_bits=3, seed=99)

        k_hat_np, _ = np_comp.decompress(np_comp.compress(k_np, v_np))
        k_hat_mlx, _ = mlx_comp.decompress(mlx_comp.compress(k_mx, v_mx))

        mse = float(np.mean((k_hat_np - np.array(k_hat_mlx)) ** 2))
        assert mse < 0.05  # reconstruction difference due to fp32 rounding only


# ---------------------------------------------------------------------------
# Temporal decay MLX utilities
# ---------------------------------------------------------------------------

class TestTemporalDecayMLX:
    def test_decay_scores_mlx_matches_numpy(self):
        from turboquant.temporal_decay import DecayConfig, TemporalDecayScheduler
        cfg = DecayConfig(decay_lambda=2.0)
        scheduler = TemporalDecayScheduler(cfg)

        np_scores = scheduler.decay_scores(50)
        mlx_scores = np.array(decay_scores_mlx(50, cfg))

        assert np.allclose(np_scores, mlx_scores, atol=1e-5)

    def test_apply_eviction_mlx_shape(self):
        from turboquant.temporal_decay import DecayConfig, DecayMode, TemporalDecayScheduler
        cfg = DecayConfig(mode=DecayMode.EVICTION, decay_lambda=4.0,
                          eviction_threshold=0.3)
        scheduler = TemporalDecayScheduler(cfg)
        sched = scheduler.schedule(32)

        k_np, v_np = _make_kv_np(num_layers=2, num_heads=4, seq_len=32, head_dim=64)
        k_mx = mx.array(k_np)
        v_mx = mx.array(v_np)

        k_ret, v_ret = apply_eviction_mlx(k_mx, v_mx, sched)
        assert k_ret.shape[2] == sched.retained_count
        assert v_ret.shape[2] == sched.retained_count

    def test_apply_eviction_no_eviction_returns_original(self):
        from turboquant.temporal_decay import DecayConfig, DecayMode, TemporalDecayScheduler
        cfg = DecayConfig(mode=DecayMode.BIT_REDUCTION)
        scheduler = TemporalDecayScheduler(cfg)
        sched = scheduler.schedule(16)

        k_np, v_np = _make_kv_np(num_layers=1, num_heads=2, seq_len=16, head_dim=64)
        k_mx = mx.array(k_np)
        v_mx = mx.array(v_np)

        k_ret, v_ret = apply_eviction_mlx(k_mx, v_mx, sched)
        # BIT_REDUCTION never evicts — should be identical objects
        assert k_ret is k_mx
        assert v_ret is v_mx
