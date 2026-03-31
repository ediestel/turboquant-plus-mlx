"""Tests for turboquant.moe_compression."""

import pytest
import numpy as np

from turboquant.moe_compression import (
    ExpertRoutingStats,
    MoEBitPlan,
    MoECompressionRouter,
)


# ---------------------------------------------------------------------------
# ExpertRoutingStats
# ---------------------------------------------------------------------------

class TestExpertRoutingStats:
    def test_wrong_counts_length_raises(self):
        counts = np.ones(4, dtype=np.int64)
        with pytest.raises(ValueError, match="length"):
            ExpertRoutingStats(n_experts=8, n_experts_used=2, activation_counts=counts)

    def test_n_experts_used_exceeds_n_experts_raises(self):
        counts = np.ones(4, dtype=np.int64)
        with pytest.raises(ValueError, match="exceed"):
            ExpertRoutingStats(n_experts=4, n_experts_used=5, activation_counts=counts)

    def test_negative_counts_raise(self):
        counts = np.array([-1, 1, 1, 1], dtype=np.int64)
        with pytest.raises(ValueError, match="non-negative"):
            ExpertRoutingStats(n_experts=4, n_experts_used=2, activation_counts=counts)

    def test_utilisation_sums_to_n_experts_used(self):
        counts = np.array([10, 20, 30, 40], dtype=np.int64)
        stats = ExpertRoutingStats(n_experts=4, n_experts_used=2, activation_counts=counts)
        assert np.sum(stats.utilisation) == pytest.approx(2.0, rel=1e-5)

    def test_utilisation_uniform(self):
        stats = ExpertRoutingStats.uniform(n_experts=8, n_experts_used=2)
        util = stats.utilisation
        assert np.allclose(util, util[0])
        assert np.sum(util) == pytest.approx(2.0, rel=1e-5)

    def test_inactive_experts(self):
        counts = np.array([10, 0, 5, 0], dtype=np.int64)
        stats = ExpertRoutingStats(n_experts=4, n_experts_used=2, activation_counts=counts)
        inactive = stats.inactive_experts
        assert set(inactive.tolist()) == {1, 3}

    def test_zero_total_counts_gives_uniform_utilisation(self):
        counts = np.zeros(4, dtype=np.int64)
        stats = ExpertRoutingStats(n_experts=4, n_experts_used=2, activation_counts=counts)
        assert np.sum(stats.utilisation) == pytest.approx(2.0, rel=1e-5)


class TestFromRoutingIndices:
    def test_basic_counting(self):
        routing = np.array([[0, 1], [1, 2], [2, 3], [0, 3]])
        stats = ExpertRoutingStats.from_routing_indices(
            routing_indices=routing, n_experts=4, n_experts_used=2
        )
        assert stats.activation_counts[0] == 2
        assert stats.activation_counts[1] == 2
        assert stats.activation_counts[2] == 2
        assert stats.activation_counts[3] == 2

    def test_flat_input(self):
        routing = np.array([0, 0, 1, 2])
        stats = ExpertRoutingStats.from_routing_indices(
            routing_indices=routing, n_experts=4, n_experts_used=1
        )
        assert stats.activation_counts[0] == 2
        assert stats.activation_counts[1] == 1
        assert stats.activation_counts[2] == 1
        assert stats.activation_counts[3] == 0

    def test_out_of_range_indices_ignored(self):
        routing = np.array([[0, 99], [1, -1]])
        stats = ExpertRoutingStats.from_routing_indices(
            routing_indices=routing, n_experts=4, n_experts_used=2
        )
        assert stats.activation_counts[0] == 1
        assert stats.activation_counts[1] == 1
        assert np.sum(stats.activation_counts) == 2

    def test_layer_idx_stored(self):
        routing = np.array([[0, 1]])
        stats = ExpertRoutingStats.from_routing_indices(
            routing_indices=routing, n_experts=4, n_experts_used=2, layer_idx=3
        )
        assert stats.layer_idx == 3

    def test_all_invalid_indices_gives_zero_counts(self):
        routing = np.array([-1, 99, 100])
        stats = ExpertRoutingStats.from_routing_indices(
            routing_indices=routing, n_experts=4, n_experts_used=2
        )
        assert np.all(stats.activation_counts == 0)


# ---------------------------------------------------------------------------
# MoECompressionRouter
# ---------------------------------------------------------------------------

class TestMoECompressionRouter:
    def _skewed_stats(self, n_experts=8, n_used=2):
        counts = np.ones(n_experts, dtype=np.int64)
        counts[0] = 100   # very high utilisation
        counts[1] = 0     # inactive
        return ExpertRoutingStats(
            n_experts=n_experts, n_experts_used=n_used, activation_counts=counts
        )

    def test_high_utilisation_gets_more_bits(self):
        router = MoECompressionRouter(
            base_bits=3.5, sensitivity_scale=0.5, evict_inactive=False
        )
        stats = self._skewed_stats()
        plan = router.plan(stats)
        assert plan.bits_for(0) >= plan.bits_for(2)

    def test_inactive_expert_evicted_when_flag_set(self):
        router = MoECompressionRouter(evict_inactive=True)
        stats = self._skewed_stats()
        plan = router.plan(stats)
        assert plan.should_evict(1)
        assert plan.expert_bits[1] == 0.0

    def test_inactive_expert_not_evicted_when_flag_clear(self):
        router = MoECompressionRouter(evict_inactive=False)
        stats = self._skewed_stats()
        plan = router.plan(stats)
        assert not plan.should_evict(1)
        assert plan.expert_bits[1] > 0.0

    def test_bits_clamped_to_min_max(self):
        router = MoECompressionRouter(
            base_bits=3.5, sensitivity_scale=1.0,
            min_bits=2.0, max_bits=6.0, evict_inactive=False
        )
        counts = np.array([1000, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)
        stats = ExpertRoutingStats(n_experts=8, n_experts_used=2, activation_counts=counts)
        plan = router.plan(stats)
        retained = plan.expert_bits[~plan.evict_mask]
        assert np.all(retained >= 2.0)
        assert np.all(retained <= 6.0)

    def test_uniform_plan_all_equal(self):
        router = MoECompressionRouter(base_bits=3.5)
        plan = router.plan_uniform(n_experts=8, n_experts_used=2)
        retained = plan.expert_bits[~plan.evict_mask]
        assert np.allclose(retained, retained[0])

    def test_zero_scale_gives_uniform_bits(self):
        router = MoECompressionRouter(
            base_bits=3.5, sensitivity_scale=0.0, evict_inactive=False
        )
        stats = self._skewed_stats()
        plan = router.plan(stats)
        retained = plan.expert_bits[~plan.evict_mask]
        assert np.allclose(retained, retained[0], atol=1e-4)

    def test_eviction_utilisation_threshold(self):
        # Experts below 10% of mean utilisation are evicted
        router = MoECompressionRouter(
            evict_inactive=False,
            eviction_utilisation_threshold=0.1,
        )
        counts = np.array([100, 1, 100, 100, 100, 100, 100, 100], dtype=np.int64)
        stats = ExpertRoutingStats(n_experts=8, n_experts_used=2, activation_counts=counts)
        plan = router.plan(stats)
        # Expert 1 has very low utilisation relative to mean -> should be evicted
        assert plan.should_evict(1)

    @pytest.mark.parametrize("n_experts", [4, 8, 64, 256])
    def test_plan_shape(self, n_experts):
        router = MoECompressionRouter()
        plan = router.plan_uniform(n_experts=n_experts, n_experts_used=2)
        assert len(plan.expert_bits) == n_experts
        assert len(plan.evict_mask) == n_experts

    def test_summary_string(self):
        router = MoECompressionRouter()
        plan = router.plan_uniform(8, 2)
        s = plan.summary()
        assert "experts=8" in s

    def test_uniform_counts_gives_uniform_bits(self):
        router = MoECompressionRouter(base_bits=3.5, sensitivity_scale=0.5, evict_inactive=False)
        counts = np.ones(8, dtype=np.int64) * 10
        stats = ExpertRoutingStats(n_experts=8, n_experts_used=2, activation_counts=counts)
        plan = router.plan(stats)
        retained = plan.expert_bits[~plan.evict_mask]
        assert np.allclose(retained, 3.5, atol=1e-4)


class TestUpdateStats:
    def test_ema_blends_counts(self):
        router = MoECompressionRouter()
        counts = np.array([10, 10, 10, 10], dtype=np.int64)
        existing = ExpertRoutingStats(
            n_experts=4, n_experts_used=2, activation_counts=counts
        )
        # New tokens only use expert 0
        new_routing = np.array([[0, 0], [0, 0]])
        updated = router.update_stats(existing, new_routing, decay_factor=0.5)
        # Expert 0 should have higher counts than others
        assert updated.activation_counts[0] > updated.activation_counts[1]

    def test_decay_factor_one_means_no_change(self):
        router = MoECompressionRouter()
        counts = np.array([5, 10, 15, 20], dtype=np.int64)
        existing = ExpertRoutingStats(
            n_experts=4, n_experts_used=2, activation_counts=counts
        )
        new_routing = np.array([[0, 1]])
        updated = router.update_stats(existing, new_routing, decay_factor=1.0)
        assert np.allclose(updated.activation_counts, counts)

    def test_decay_factor_zero_gives_only_new(self):
        router = MoECompressionRouter()
        counts = np.array([100, 100, 100, 100], dtype=np.int64)
        existing = ExpertRoutingStats(
            n_experts=4, n_experts_used=2, activation_counts=counts
        )
        new_routing = np.array([[0, 0], [0, 0]])
        updated = router.update_stats(existing, new_routing, decay_factor=0.0)
        assert updated.activation_counts[0] > 0
        assert updated.activation_counts[1] == 0

    def test_layer_idx_preserved(self):
        router = MoECompressionRouter()
        counts = np.ones(4, dtype=np.int64)
        existing = ExpertRoutingStats(
            n_experts=4, n_experts_used=2, activation_counts=counts, layer_idx=7
        )
        new_routing = np.array([[0, 1]])
        updated = router.update_stats(existing, new_routing)
        assert updated.layer_idx == 7
