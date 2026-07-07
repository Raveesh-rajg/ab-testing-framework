import numpy as np
import pytest
from scipy import stats

from expkit.frequentist import proportion_ztest, welch_ttest


class TestProportionZTest:
    def test_null_case_not_significant(self):
        r = proportion_ztest(500, 10_000, 505, 10_000)
        assert not r.significant
        assert r.p_value > 0.5

    def test_clear_effect_significant(self):
        r = proportion_ztest(500, 10_000, 650, 10_000)
        assert r.significant
        assert r.relative_lift == pytest.approx(0.30, abs=0.01)

    def test_matches_statsmodels(self):
        from statsmodels.stats.proportion import proportions_ztest

        z_sm, p_sm = proportions_ztest([650, 500], [10_000, 10_000])
        r = proportion_ztest(500, 10_000, 650, 10_000)
        assert r.statistic == pytest.approx(z_sm, abs=1e-9)
        assert r.p_value == pytest.approx(p_sm, abs=1e-9)

    def test_ci_covers_true_effect(self):
        """Coverage: ~95% of simulated experiments should have the true
        absolute difference inside the 95% CI."""
        rng = np.random.default_rng(0)
        p_c, p_t, n = 0.10, 0.11, 20_000
        covered = 0
        sims = 400
        for _ in range(sims):
            c = rng.binomial(n, p_c)
            t = rng.binomial(n, p_t)
            r = proportion_ztest(c, n, t, n)
            if r.ci_low_abs <= (p_t - p_c) <= r.ci_high_abs:
                covered += 1
        assert covered / sims == pytest.approx(0.95, abs=0.03)

    def test_conversions_exceed_n_raises(self):
        with pytest.raises(ValueError):
            proportion_ztest(11, 10, 5, 10)


class TestWelchTTest:
    def test_matches_scipy(self):
        rng = np.random.default_rng(1)
        c = rng.normal(10, 2, 500)
        t = rng.normal(10.5, 3, 480)
        r = welch_ttest(c, t)
        t_sp, p_sp = stats.ttest_ind(t, c, equal_var=False)
        assert r.statistic == pytest.approx(t_sp)
        assert r.p_value == pytest.approx(p_sp)

    def test_null_case(self):
        rng = np.random.default_rng(2)
        c = rng.normal(10, 2, 1000)
        t = rng.normal(10, 2, 1000)
        r = welch_ttest(c, t)
        assert not r.significant

    def test_relative_lift_sign(self):
        rng = np.random.default_rng(3)
        c = rng.normal(10, 1, 2000)
        t = rng.normal(11, 1, 2000)
        r = welch_ttest(c, t)
        assert r.relative_lift == pytest.approx(0.10, abs=0.02)
        assert r.ci_low_rel < r.relative_lift < r.ci_high_rel
