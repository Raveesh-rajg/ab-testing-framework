import numpy as np
import pytest

from expkit.power import power_analysis, minimum_detectable_effect, simulated_power


class TestPowerAnalysis:
    def test_known_benchmark(self):
        """5% baseline, +10% relative (0.5pp), alpha 0.05, power 0.8.
        Standard calculators put this at ~31k per arm."""
        r = power_analysis(baseline=0.05, mde_relative=0.10)
        assert 29_000 < r.n_per_arm < 33_000

    def test_smaller_mde_needs_more_n(self):
        big = power_analysis(0.05, 0.10).n_per_arm
        small = power_analysis(0.05, 0.02).n_per_arm
        assert small > big * 10  # n scales roughly with 1/effect^2

    def test_higher_power_needs_more_n(self):
        p80 = power_analysis(0.05, 0.10, power=0.80).n_per_arm
        p95 = power_analysis(0.05, 0.10, power=0.95).n_per_arm
        assert p95 > p80

    def test_continuous_requires_std(self):
        with pytest.raises(ValueError):
            power_analysis(60.0, 0.05, metric_type="continuous")

    def test_continuous_benchmark(self):
        # d = 3/40 = 0.075 -> n ~ 2 * (1.96+0.84)^2 / d^2 ~ 2790
        r = power_analysis(60.0, 0.05, metric_type="continuous", std=40.0)
        assert 2_600 < r.n_per_arm < 3_000

    def test_invalid_rate_raises(self):
        with pytest.raises(ValueError):
            power_analysis(baseline=0.9, mde_relative=0.5)  # treatment > 1


class TestMDEInversion:
    def test_roundtrip(self):
        """MDE(n(mde)) should recover mde."""
        mde = 0.10
        n = power_analysis(0.05, mde).n_per_arm
        recovered = minimum_detectable_effect(0.05, n)
        assert recovered == pytest.approx(mde, rel=0.02)

    def test_more_n_means_smaller_mde(self):
        assert minimum_detectable_effect(0.05, 100_000) < minimum_detectable_effect(
            0.05, 10_000
        )


class TestSimulatedPower:
    def test_agrees_with_analytic(self):
        """Simulation at the analytic n should land near the target power."""
        r = power_analysis(0.05, 0.10, power=0.80)
        sim = simulated_power(0.05, 0.10, r.n_per_arm, n_sims=1500)
        assert sim == pytest.approx(0.80, abs=0.05)

    def test_null_effect_gives_alpha(self):
        """Under no effect, rejection rate should be ~alpha (validity)."""
        sim = simulated_power(0.05, 0.0, 20_000, n_sims=1500)
        assert sim == pytest.approx(0.05, abs=0.02)
