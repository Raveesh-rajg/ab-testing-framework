import pytest

from expkit.bayesian import BetaBinomialTest


class TestBetaBinomial:
    def test_symmetric_data_gives_50_50(self):
        r = BetaBinomialTest().analyze(500, 10_000, 500, 10_000)
        assert r.prob_treatment_better == pytest.approx(0.5, abs=0.02)

    def test_clear_winner(self):
        r = BetaBinomialTest().analyze(500, 10_000, 650, 10_000)
        assert r.prob_treatment_better > 0.99
        assert r.expected_loss_ship < 0.001
        assert r.lift_posterior_mean == pytest.approx(0.30, abs=0.03)

    def test_expected_losses_are_complementary_in_sign(self):
        """When treatment is clearly better, loss(ship) ~ 0 and
        loss(no_ship) ~ the lift itself."""
        r = BetaBinomialTest().analyze(500, 10_000, 650, 10_000)
        assert r.expected_loss_no_ship == pytest.approx(
            r.lift_posterior_mean, rel=0.05
        )

    def test_credible_interval_contains_posterior_mean(self):
        r = BetaBinomialTest().analyze(480, 10_000, 530, 10_000)
        assert r.lift_ci_low < r.lift_posterior_mean < r.lift_ci_high

    def test_prior_dominated_by_data(self):
        """With experiment-scale n, a strong prior should barely move
        the answer."""
        weak = BetaBinomialTest(1, 1).analyze(500, 10_000, 560, 10_000)
        strong = BetaBinomialTest(50, 950).analyze(500, 10_000, 560, 10_000)
        assert weak.prob_treatment_better == pytest.approx(
            strong.prob_treatment_better, abs=0.05
        )

    def test_invalid_prior_raises(self):
        with pytest.raises(ValueError):
            BetaBinomialTest(0, 1)

    def test_agrees_with_frequentist_directionally(self):
        """Sanity: for a borderline result both frameworks should be
        borderline; p ~ 0.05 corresponds loosely to P(B>A) ~ 0.975 for
        one-sided reading of a two-sided test."""
        from expkit.frequentist import proportion_ztest

        freq = proportion_ztest(500, 10_000, 560, 10_000)
        bayes = BetaBinomialTest().analyze(500, 10_000, 560, 10_000)
        # p ~ 0.06 here; P(B>A) should be ~ 1 - p/2
        assert bayes.prob_treatment_better == pytest.approx(
            1 - freq.p_value / 2, abs=0.02
        )
