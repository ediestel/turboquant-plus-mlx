"""Tests for turboquant/bit_budget.py.

Covers: SlotKey, BitBudgetPolicy protocol, UniformPolicy, LayerHeadPolicy,
and SensitivityCalibratedPolicy (including entropy accumulation, calibration,
edge cases, and reset).
"""

import numpy as np
import pytest

from turboquant.adaptive_bits import AllocationPlan, AdaptiveBitAllocator
from turboquant.bit_budget import (
    BitBudgetPolicy,
    LayerHeadPolicy,
    SensitivityCalibratedPolicy,
    SlotKey,
    UniformPolicy,
)


# ---------------------------------------------------------------------------
# SlotKey
# ---------------------------------------------------------------------------

class TestSlotKey:
    def test_create_and_access(self):
        s = SlotKey(layer_idx=2, head_idx=5)
        assert s.layer_idx == 2
        assert s.head_idx == 5

    def test_frozen_raises_on_assignment(self):
        s = SlotKey(0, 0)
        with pytest.raises((AttributeError, TypeError)):
            s.layer_idx = 99  # type: ignore[misc]

    def test_hashable(self):
        s = SlotKey(1, 2)
        d = {s: "value"}
        assert d[SlotKey(1, 2)] == "value"

    def test_equality(self):
        assert SlotKey(0, 1) == SlotKey(0, 1)
        assert SlotKey(0, 1) != SlotKey(0, 2)

    def test_usable_as_dict_key(self):
        mapping = {SlotKey(l, h): l * 4 + h for l in range(3) for h in range(4)}
        assert mapping[SlotKey(2, 3)] == 11


# ---------------------------------------------------------------------------
# BitBudgetPolicy protocol
# ---------------------------------------------------------------------------

class TestBitBudgetProtocol:
    def test_uniform_satisfies_protocol(self):
        assert isinstance(UniformPolicy(), BitBudgetPolicy)

    def test_layer_head_satisfies_protocol(self):
        allocator = AdaptiveBitAllocator(base_bits=3.0, sensitivity_scale=0.0)
        plan = allocator.allocate_uniform(2, 4)
        assert isinstance(LayerHeadPolicy(plan), BitBudgetPolicy)

    def test_calibrated_satisfies_protocol(self):
        policy = SensitivityCalibratedPolicy(n_layers=2, n_heads=4)
        assert isinstance(policy, BitBudgetPolicy)

    def test_plain_object_does_not_satisfy(self):
        class Bare:
            pass
        assert not isinstance(Bare(), BitBudgetPolicy)

    def test_partial_object_does_not_satisfy(self):
        class OnlyK:
            def k_bits_for(self, slot): return 3.0
        assert not isinstance(OnlyK(), BitBudgetPolicy)


# ---------------------------------------------------------------------------
# UniformPolicy
# ---------------------------------------------------------------------------

class TestUniformPolicy:
    def test_k_bits_uniform(self):
        p = UniformPolicy(base_bits=4.0)
        for l in range(3):
            for h in range(8):
                assert p.k_bits_for(SlotKey(l, h)) == pytest.approx(4.0)

    def test_v_bits_uniform(self):
        p = UniformPolicy(base_bits=2.5)
        assert p.v_bits_for(SlotKey(0, 0)) == pytest.approx(2.5)
        assert p.v_bits_for(SlotKey(7, 15)) == pytest.approx(2.5)

    def test_to_allocation_plan_shape(self):
        p = UniformPolicy(base_bits=3.5)
        plan = p.to_allocation_plan(n_layers=4, n_heads=8)
        assert plan.k_bits.shape == (4, 8)
        assert plan.v_bits.shape == (4, 8)

    def test_to_allocation_plan_values(self):
        p = UniformPolicy(base_bits=3.0)
        plan = p.to_allocation_plan(n_layers=2, n_heads=4)
        assert np.allclose(plan.k_bits, 3.0)
        assert np.allclose(plan.v_bits, 3.0)


# ---------------------------------------------------------------------------
# LayerHeadPolicy
# ---------------------------------------------------------------------------

class TestLayerHeadPolicy:
    def _make_plan(self):
        k = np.array([[2.0, 3.0, 4.0], [3.5, 2.5, 1.5]])
        v = np.array([[2.5, 3.5, 4.5], [3.0, 2.0, 1.0]])
        return AllocationPlan(k_bits=k, v_bits=v)

    def test_k_bits_for_correct(self):
        plan = self._make_plan()
        p = LayerHeadPolicy(plan)
        assert p.k_bits_for(SlotKey(0, 1)) == pytest.approx(3.0)
        assert p.k_bits_for(SlotKey(1, 2)) == pytest.approx(1.5)

    def test_v_bits_for_correct(self):
        plan = self._make_plan()
        p = LayerHeadPolicy(plan)
        assert p.v_bits_for(SlotKey(0, 0)) == pytest.approx(2.5)
        assert p.v_bits_for(SlotKey(1, 1)) == pytest.approx(2.0)

    def test_to_allocation_plan_round_trips(self):
        plan = self._make_plan()
        p = LayerHeadPolicy(plan)
        returned = p.to_allocation_plan()
        assert np.allclose(returned.k_bits, plan.k_bits)
        assert np.allclose(returned.v_bits, plan.v_bits)

    def test_from_uniform(self):
        p = LayerHeadPolicy.from_uniform(n_layers=3, n_heads=5, base_bits=3.0)
        for l in range(3):
            for h in range(5):
                assert p.k_bits_for(SlotKey(l, h)) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# SensitivityCalibratedPolicy
# ---------------------------------------------------------------------------

class TestSensitivityCalibratedPolicy:
    def test_not_calibrated_before_recording(self):
        p = SensitivityCalibratedPolicy(n_layers=2, n_heads=4)
        assert not p.is_calibrated

    def test_bits_for_before_calibration_returns_base(self):
        p = SensitivityCalibratedPolicy(n_layers=2, n_heads=4, base_bits=3.5)
        assert p.k_bits_for(SlotKey(0, 0)) == pytest.approx(3.5)
        assert p.v_bits_for(SlotKey(1, 3)) == pytest.approx(3.5)

    def test_zero_sum_sample_is_ignored(self):
        p = SensitivityCalibratedPolicy(n_layers=1, n_heads=2)
        p.record_attention_weights(np.zeros(8), 0, 0)
        assert not p.is_calibrated  # ignored

    def test_zero_sum_produces_no_nan(self):
        p = SensitivityCalibratedPolicy(n_layers=1, n_heads=2, base_bits=3.0)
        p.record_attention_weights(np.zeros(8), 0, 0)
        bits = p.k_bits_for(SlotKey(0, 0))
        assert np.isfinite(bits)

    def test_is_calibrated_after_recording(self):
        p = SensitivityCalibratedPolicy(n_layers=2, n_heads=4)
        p.record_attention_weights(np.array([0.9, 0.05, 0.03, 0.02]), 0, 0)
        assert p.is_calibrated

    def test_peaked_head_gets_more_bits_than_uniform(self):
        """Low-entropy (peaked) head is more sensitive → more bits."""
        p = SensitivityCalibratedPolicy(
            n_layers=1, n_heads=2, base_bits=3.5, sensitivity_scale=1.0
        )
        peaked = np.array([0.97, 0.01, 0.01, 0.01])
        uniform = np.full(4, 0.25)
        p.record_attention_weights(peaked, 0, 0)
        p.record_attention_weights(uniform, 0, 1)
        p.calibrate()
        bits_peaked = p.k_bits_for(SlotKey(0, 0))
        bits_uniform = p.k_bits_for(SlotKey(0, 1))
        assert bits_peaked > bits_uniform

    def test_uniform_entropy_across_all_heads_gives_uniform_plan(self):
        p = SensitivityCalibratedPolicy(n_layers=2, n_heads=4, base_bits=3.5)
        weights = np.full(8, 1.0 / 8)
        for l in range(2):
            for h in range(4):
                p.record_attention_weights(weights, l, h)
        plan = p.calibrate()
        assert np.allclose(plan.k_bits, plan.k_bits[0, 0], atol=1e-6)

    def test_multiple_records_per_slot_averaged(self):
        p = SensitivityCalibratedPolicy(n_layers=1, n_heads=1)
        w1 = np.array([0.9, 0.05, 0.03, 0.02])
        w2 = np.array([0.6, 0.2, 0.1, 0.1])
        p.record_attention_weights(w1, 0, 0)
        p.record_attention_weights(w2, 0, 0)
        # Two counts accumulated; just check no crash and a valid plan
        plan = p.calibrate()
        assert np.isfinite(plan.k_bits[0, 0])

    def test_unobserved_head_gets_overall_mean_not_zero(self):
        """Head (0,1) not observed; should not inherit zero-entropy sensitivity."""
        p = SensitivityCalibratedPolicy(n_layers=1, n_heads=2, base_bits=3.5)
        p.record_attention_weights(np.array([0.9, 0.05, 0.03, 0.02]), 0, 0)
        plan = p.calibrate()
        # Head (0,1) should have some finite bits near base_bits, not extreme
        assert np.isfinite(plan.k_bits[0, 1])
        assert 1.0 <= plan.k_bits[0, 1] <= 8.0

    def test_reset_clears_calibration(self):
        p = SensitivityCalibratedPolicy(n_layers=1, n_heads=2)
        p.record_attention_weights(np.array([0.9, 0.1]), 0, 0)
        assert p.is_calibrated
        p.reset()
        assert not p.is_calibrated

    def test_reset_returns_base_bits_after_reset(self):
        p = SensitivityCalibratedPolicy(n_layers=1, n_heads=2, base_bits=3.0)
        p.record_attention_weights(np.array([0.9, 0.1]), 0, 0)
        p.calibrate()
        p.reset()
        assert p.k_bits_for(SlotKey(0, 0)) == pytest.approx(3.0)

    def test_to_allocation_plan_returns_allocation_plan(self):
        p = SensitivityCalibratedPolicy(n_layers=2, n_heads=4)
        p.record_attention_weights(np.full(8, 0.125), 0, 0)
        plan = p.to_allocation_plan()
        assert isinstance(plan, AllocationPlan)
        assert plan.k_bits.shape == (2, 4)

    def test_one_hot_lower_entropy_than_uniform(self):
        """One-hot attention has entropy 0; uniform has maximum entropy."""
        p = SensitivityCalibratedPolicy(
            n_layers=1, n_heads=2, base_bits=3.5, sensitivity_scale=1.0
        )
        one_hot = np.array([1.0, 0.0, 0.0, 0.0])
        uniform = np.full(4, 0.25)
        p.record_attention_weights(one_hot, 0, 0)
        p.record_attention_weights(uniform, 0, 1)
        plan = p.calibrate()
        # One-hot = low entropy = more sensitive = more bits
        assert plan.k_bits[0, 0] >= plan.k_bits[0, 1]
