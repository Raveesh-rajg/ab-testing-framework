import pandas as pd
import pytest

from expkit.simulate import (
    simulate_conversion_experiment,
    simulate_simpsons_paradox,
)
from expkit.validity import srm_check, segment_analysis, lift_over_time


class TestSRM:
    def test_balanced_passes(self):
        assert srm_check(100_000, 100_200).passed

    def test_broken_split_fails(self):
        assert not srm_check(100_000, 104_000).passed

    def test_uneven_design_split(self):
        # 90/10 rollout, observed close to it -> pass
        assert srm_check(90_000, 10_050, expected_treatment_share=0.10).passed


class TestSegmentAnalysis:
    def test_detects_simpsons_paradox(self):
        df = simulate_simpsons_paradox(seed=5)
        result = segment_analysis(df, segment_col="segment")
        assert result.simpsons_flag
        # pooled negative, both segments positive
        assert result.overall.relative_lift < 0
        assert (result.by_segment["relative_lift"] > 0).all()

    def test_no_flag_on_consistent_data(self):
        df = simulate_conversion_experiment(
            n_per_arm=50_000, baseline=0.05, true_lift=0.10, seed=6
        )
        df["segment"] = ["a", "b"] * (len(df) // 2)
        result = segment_analysis(df, segment_col="segment")
        assert not result.simpsons_flag


class TestLiftOverTime:
    def test_flags_novelty(self):
        df = simulate_conversion_experiment(
            n_per_arm=200_000, baseline=0.05, true_lift=0.25,
            n_days=14, novelty_decay=0.35, seed=8,
        )
        result = lift_over_time(df)
        assert result.novelty_flag
        assert result.second_half_lift < result.first_half_lift

    def test_stable_effect_not_flagged(self):
        df = simulate_conversion_experiment(
            n_per_arm=200_000, baseline=0.05, true_lift=0.10,
            n_days=14, novelty_decay=0.0, seed=9,
        )
        assert not lift_over_time(df).novelty_flag
