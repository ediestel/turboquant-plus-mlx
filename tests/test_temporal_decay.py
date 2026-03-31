"""Tests for turboquant.temporal_decay."""

import pytest
import numpy as np

from turboquant.temporal_decay import (
    DecayConfig, DecayMode, DecaySchedule, TemporalDecayScheduler,
)


class TestDecayConfigValidation:
    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError, match="positive"):
            DecayConfig(decay_lambda=-1.0)

    def test_zero_lambda_raises(self):
        with pytest.raises(ValueError, match="positive"):
            DecayConfig(decay_lambda=0.0)

    def test_min_bits_gt_base_bits_raises(self):
        with pytest.raises(ValueError, match="min_bits"):
            DecayConfig(base_bits=2.0, min_bits=3.0)

    def test_min_bits_equal_base_bits_raises(self):
        with pytest.raises(ValueError, match="min_bits"):
            DecayConfig(base_bits=3.0, min_bits=3.0)

    def test_invalid_eviction_threshold_raises(self):
        with pytest.raises(ValueError):
            DecayConfig(eviction_threshold=0.0)
        with pytest.raises(ValueError):
            DecayConfig(eviction_threshold=1.0)

    def test_invalid_window_size_raises(self):
        with pytest.raises(ValueError):
            DecayConfig(window_size=0)
        with pytest.raises(ValueError):
            DecayConfig(window_size=-5)

    def test_valid_config(self):
        cfg = DecayConfig(decay_lambda=2.0, base_bits=3.5, min_bits=2.0)
        assert cfg.base_bits == 3.5


class TestDecayScores:
    @pytest.mark.parametrize("seq_len", [1, 2, 10, 100, 1000])
    def test_newest_token_has_decay_one(self, seq_len):
        scheduler = TemporalDecayScheduler()
        scores = scheduler.decay_scores(seq_len)
        assert scores[-1] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("lam", [0.5, 1.0, 2.0, 5.0])
    def test_oldest_token_decay_value(self, lam):
        cfg = DecayConfig(decay_lambda=lam)
        scheduler = TemporalDecayScheduler(cfg)
        seq_len = 50
        scores = scheduler.decay_scores(seq_len)
        expected = float(np.exp(-lam))
        assert scores[0] == pytest.approx(expected, rel=1e-4)

    def test_scores_monotonically_increasing(self):
        scheduler = TemporalDecayScheduler()
        scores = scheduler.decay_scores(100)
        assert np.all(np.diff(scores) >= -1e-7)

    def test_scores_in_zero_one(self):
        scheduler = TemporalDecayScheduler()
        scores = scheduler.decay_scores(200)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1.0 + 1e-6)

    def test_seq_len_one(self):
        scheduler = TemporalDecayScheduler()
        scores = scheduler.decay_scores(1)
        assert scores[0] == pytest.approx(1.0)

    def test_selected_positions(self):
        scheduler = TemporalDecayScheduler()
        positions = np.array([0, 9])
        scores = scheduler.decay_scores(10, positions=positions)
        assert len(scores) == 2
        assert scores[1] > scores[0]


class TestBitsFromDecay:
    def test_decay_zero_gives_min_bits(self):
        cfg = DecayConfig(base_bits=3.5, min_bits=2.0)
        scheduler = TemporalDecayScheduler(cfg)
        bits = scheduler.bits_from_decay(np.array([0.0], dtype=np.float32))
        assert bits[0] == pytest.approx(2.0)

    def test_decay_one_gives_base_bits(self):
        cfg = DecayConfig(base_bits=3.5, min_bits=2.0)
        scheduler = TemporalDecayScheduler(cfg)
        bits = scheduler.bits_from_decay(np.array([1.0], dtype=np.float32))
        assert bits[0] == pytest.approx(3.5)

    def test_interpolation_midpoint(self):
        cfg = DecayConfig(base_bits=4.0, min_bits=2.0)
        scheduler = TemporalDecayScheduler(cfg)
        bits = scheduler.bits_from_decay(np.array([0.5], dtype=np.float32))
        assert bits[0] == pytest.approx(3.0)


class TestBitReductionMode:
    def test_bits_monotonically_nondecreasing(self):
        cfg = DecayConfig(mode=DecayMode.BIT_REDUCTION, decay_lambda=2.0)
        scheduler = TemporalDecayScheduler(cfg)
        sched = scheduler.schedule(100)
        assert np.all(np.diff(sched.bits_per_token) >= -1e-6)

    def test_no_eviction_in_bit_reduction_mode(self):
        cfg = DecayConfig(mode=DecayMode.BIT_REDUCTION)
        scheduler = TemporalDecayScheduler(cfg)
        sched = scheduler.schedule(50)
        assert sched.evicted_count == 0

    def test_oldest_tokens_get_min_bits(self):
        cfg = DecayConfig(
            mode=DecayMode.BIT_REDUCTION,
            decay_lambda=5.0,
            base_bits=4.0,
            min_bits=2.0,
            reduction_threshold=0.5,
        )
        scheduler = TemporalDecayScheduler(cfg)
        sched = scheduler.schedule(100)
        # Position 0 should have a very small decay -> min_bits
        assert sched.bits_per_token[0] == pytest.approx(2.0)

    def test_newest_token_gets_base_bits(self):
        cfg = DecayConfig(
            mode=DecayMode.BIT_REDUCTION,
            decay_lambda=2.0,
            base_bits=4.0,
            min_bits=2.0,
        )
        scheduler = TemporalDecayScheduler(cfg)
        sched = scheduler.schedule(100)
        assert sched.bits_per_token[-1] == pytest.approx(4.0, abs=0.1)


class TestEvictionMode:
    def test_tokens_below_threshold_evicted(self):
        cfg = DecayConfig(
            mode=DecayMode.EVICTION,
            decay_lambda=3.0,
            eviction_threshold=0.1,
        )
        scheduler = TemporalDecayScheduler(cfg)
        sched = scheduler.schedule(100)
        scores = sched.decay_scores
        # Every position with score < 0.1 must be evicted
        for i, (score, evict) in enumerate(zip(scores, sched.evict_mask)):
            if score < 0.1:
                assert evict, f"position {i} score={score:.4f} should be evicted"

    def test_newest_token_not_evicted(self):
        cfg = DecayConfig(mode=DecayMode.EVICTION, eviction_threshold=0.05)
        scheduler = TemporalDecayScheduler(cfg)
        sched = scheduler.schedule(100)
        assert not sched.evict_mask[-1]

    def test_retained_plus_evicted_equals_seq_len(self):
        cfg = DecayConfig(mode=DecayMode.EVICTION, decay_lambda=2.0)
        scheduler = TemporalDecayScheduler(cfg)
        for seq_len in [1, 10, 50, 200]:
            sched = scheduler.schedule(seq_len)
            assert sched.retained_count + sched.evicted_count == seq_len


class TestHybridMode:
    def test_hybrid_applies_both_thresholds(self):
        cfg = DecayConfig(
            mode=DecayMode.HYBRID,
            decay_lambda=4.0,
            base_bits=4.0,
            min_bits=2.0,
            eviction_threshold=0.05,
            reduction_threshold=0.3,
        )
        scheduler = TemporalDecayScheduler(cfg)
        sched = scheduler.schedule(100)
        # Some tokens should be evicted
        assert sched.evicted_count > 0
        # The oldest retained token should have min_bits
        retained_bits = sched.bits_per_token[~sched.evict_mask]
        assert np.min(retained_bits) == pytest.approx(2.0)

    def test_retained_count_plus_evicted_count_correct(self):
        cfg = DecayConfig(mode=DecayMode.HYBRID)
        scheduler = TemporalDecayScheduler(cfg)
        sched = scheduler.schedule(128)
        assert sched.retained_count + sched.evicted_count == 128


class TestWindowSizeEviction:
    def test_window_size_evicts_older_than_window(self):
        cfg = DecayConfig(
            mode=DecayMode.EVICTION,
            eviction_threshold=0.001,  # very low — only window drives eviction
            window_size=20,
        )
        scheduler = TemporalDecayScheduler(cfg)
        sched = scheduler.schedule(100)
        # Tokens at positions 0..79 should be evicted (older than window of 20)
        assert np.all(sched.evict_mask[:80])

    def test_window_size_retains_recent_tokens(self):
        cfg = DecayConfig(
            mode=DecayMode.EVICTION,
            eviction_threshold=0.001,
            window_size=10,
        )
        scheduler = TemporalDecayScheduler(cfg)
        sched = scheduler.schedule(50)
        # Last 10 tokens should NOT be evicted by window alone
        # (decay scores are near 1.0, above threshold)
        assert not np.any(sched.evict_mask[40:])

    def test_window_size_gte_seq_len_no_window_eviction(self):
        cfg = DecayConfig(
            mode=DecayMode.EVICTION,
            eviction_threshold=0.001,
            window_size=100,
        )
        scheduler = TemporalDecayScheduler(cfg)
        sched = scheduler.schedule(50)
        # window_size >= seq_len means no window eviction
        # only score-threshold eviction applies (very low threshold → none)
        assert sched.evicted_count == 0


class TestUpdateSchedule:
    def test_update_schedule_returns_new_seq_len(self):
        scheduler = TemporalDecayScheduler()
        prev = scheduler.schedule(50)
        updated = scheduler.update_schedule(prev, 60)
        assert len(updated.positions) == 60

    def test_update_schedule_newest_still_one(self):
        scheduler = TemporalDecayScheduler()
        prev = scheduler.schedule(50)
        updated = scheduler.update_schedule(prev, 75)
        assert updated.decay_scores[-1] == pytest.approx(1.0, abs=1e-6)

    def test_old_evicted_tokens_stay_evicted(self):
        cfg = DecayConfig(mode=DecayMode.EVICTION, decay_lambda=5.0, eviction_threshold=0.2)
        scheduler = TemporalDecayScheduler(cfg)
        prev = scheduler.schedule(50)
        evicted_before = prev.evict_mask.copy()
        updated = scheduler.update_schedule(prev, 60)
        # Previously evicted tokens must remain evicted
        assert np.all(updated.evict_mask[:50] >= evicted_before)

    def test_new_tokens_not_evicted(self):
        cfg = DecayConfig(mode=DecayMode.EVICTION, decay_lambda=5.0)
        scheduler = TemporalDecayScheduler(cfg)
        prev = scheduler.schedule(50)
        updated = scheduler.update_schedule(prev, 60)
        assert not np.any(updated.evict_mask[50:])

    def test_update_same_length_returns_prev(self):
        scheduler = TemporalDecayScheduler()
        prev = scheduler.schedule(50)
        updated = scheduler.update_schedule(prev, 50)
        assert len(updated.positions) == 50


class TestDecayScheduleProperties:
    def test_mean_bits_retained_empty(self):
        # Construct a schedule where everything is evicted
        positions = np.array([0, 1], dtype=np.int32)
        decay_scores = np.array([0.01, 0.02], dtype=np.float32)
        bits = np.array([2.0, 2.0], dtype=np.float32)
        evict = np.array([True, True])
        sched = DecaySchedule(
            positions=positions,
            decay_scores=decay_scores,
            bits_per_token=bits,
            evict_mask=evict,
        )
        assert sched.mean_bits_retained == 0.0
        assert sched.retained_count == 0
        assert sched.evicted_count == 2

    def test_summary_string(self):
        scheduler = TemporalDecayScheduler()
        sched = scheduler.schedule(50)
        s = sched.summary()
        assert "seq_len=50" in s
        assert "retained=" in s

    def test_seq_len_one_schedule(self):
        scheduler = TemporalDecayScheduler()
        sched = scheduler.schedule(1)
        assert len(sched.positions) == 1
        assert sched.decay_scores[0] == pytest.approx(1.0)
        assert sched.evicted_count == 0
