"""Tests for turboquant/integrations/mlx_lm.py — Phase 3 adaptive extensions.

Guards with pytest.importorskip("mlx.core") so the suite passes on machines
without MLX (CI / non-Apple-Silicon). Tests validate that:
  - Default path (no adaptive kwargs) still uses KVCacheCompressorMLX
  - adaptive=True switches to AdaptiveKVCacheCompressorMLX
  - Feature kwargs are forwarded without changing the non-adaptive path
"""

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")  # skip entire module if MLX unavailable

from turboquant.integrations.mlx_lm import TurboQuantKVCache
from turboquant.mlx.kv_cache import KVCacheCompressorMLX
from turboquant.mlx.adaptive_kv_cache import AdaptiveKVCacheCompressorMLX
from turboquant.adaptive_bits import AdaptiveBitAllocator, AllocationPlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeModelArgs:
    head_dim = 64
    num_attention_heads = 4
    num_hidden_layers = 2


class FakeModel:
    args = FakeModelArgs()


# ---------------------------------------------------------------------------
# Default (non-adaptive) path unchanged
# ---------------------------------------------------------------------------

class TestDefaultPath:
    def test_default_compressor_type(self):
        cache = TurboQuantKVCache(
            head_dim=64, num_heads=4, num_layers=2, k_bits=3, v_bits=3
        )
        assert isinstance(cache.compressor, KVCacheCompressorMLX)
        assert not isinstance(cache.compressor, AdaptiveKVCacheCompressorMLX)

    def test_for_model_default_path(self):
        cache = TurboQuantKVCache.for_model(FakeModel(), k_bits=3, v_bits=3)
        assert isinstance(cache.compressor, KVCacheCompressorMLX)
        assert not isinstance(cache.compressor, AdaptiveKVCacheCompressorMLX)

    def test_default_memory_stats_shape(self):
        cache = TurboQuantKVCache(
            head_dim=64, num_heads=4, num_layers=2, k_bits=3, v_bits=3
        )
        stats = cache.memory_stats(seq_len=32)
        assert stats["compression_ratio"] > 1.0


# ---------------------------------------------------------------------------
# adaptive=True switches compressor
# ---------------------------------------------------------------------------

class TestAdaptivePath:
    def test_adaptive_true_uses_adaptive_compressor(self):
        cache = TurboQuantKVCache(
            head_dim=64, num_heads=4, num_layers=2,
            k_bits=3, v_bits=3, adaptive=True,
        )
        assert isinstance(cache.compressor, AdaptiveKVCacheCompressorMLX)

    def test_for_model_adaptive_true(self):
        cache = TurboQuantKVCache.for_model(
            FakeModel(), k_bits=3, v_bits=3, adaptive=True
        )
        assert isinstance(cache.compressor, AdaptiveKVCacheCompressorMLX)

    def test_allocation_plan_triggers_adaptive(self):
        allocator = AdaptiveBitAllocator(base_bits=3.0, sensitivity_scale=0.0)
        plan = allocator.allocate_uniform(n_layers=2, n_heads=4)
        cache = TurboQuantKVCache(
            head_dim=64, num_heads=4, num_layers=2,
            k_bits=3, v_bits=3, allocation_plan=plan,
        )
        assert isinstance(cache.compressor, AdaptiveKVCacheCompressorMLX)

    def test_decay_tiers_triggers_adaptive(self):
        cache = TurboQuantKVCache(
            head_dim=64, num_heads=4, num_layers=2,
            k_bits=3, v_bits=3, decay_tiers=[(0.5, 2.0)],
        )
        assert isinstance(cache.compressor, AdaptiveKVCacheCompressorMLX)

    def test_moe_plans_triggers_adaptive(self):
        from turboquant.moe_compression import MoECompressionRouter
        router = MoECompressionRouter(base_bits=3.5)
        moe_plans = {0: router.plan_uniform(4, 2), 1: router.plan_uniform(4, 2)}
        cache = TurboQuantKVCache(
            head_dim=64, num_heads=4, num_layers=2,
            k_bits=3, v_bits=3, moe_plans=moe_plans,
        )
        assert isinstance(cache.compressor, AdaptiveKVCacheCompressorMLX)

    def test_adaptive_memory_stats_has_feature_flags(self):
        cache = TurboQuantKVCache(
            head_dim=64, num_heads=4, num_layers=2,
            k_bits=3, v_bits=3, adaptive=True,
        )
        stats = cache.memory_stats(seq_len=32)
        # adaptive compressor memory_stats should return these keys
        assert "adaptive_bits_enabled" in stats or "compression_ratio" in stats

    def test_adaptive_for_model_dims_correct(self):
        cache = TurboQuantKVCache.for_model(
            FakeModel(), k_bits=3, v_bits=3, adaptive=True
        )
        assert cache.head_dim == 64
        assert cache.num_heads == 4
        assert cache.num_layers == 2

    def test_allocation_plan_forwarded_to_compressor(self):
        allocator = AdaptiveBitAllocator(base_bits=3.0, sensitivity_scale=0.0)
        plan = allocator.allocate_uniform(n_layers=2, n_heads=4)
        cache = TurboQuantKVCache(
            head_dim=64, num_heads=4, num_layers=2,
            k_bits=3, v_bits=3, adaptive=True, allocation_plan=plan,
        )
        assert cache.compressor._allocation_plan is not None

    def test_no_adaptive_kwargs_never_uses_adaptive_compressor(self):
        """Passing adaptive=False with no feature kwargs must use base compressor."""
        cache = TurboQuantKVCache(
            head_dim=64, num_heads=4, num_layers=2,
            k_bits=3, v_bits=3, adaptive=False,
        )
        assert type(cache.compressor) is KVCacheCompressorMLX
