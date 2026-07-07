"""Bayesian analysis via conjugate models.

Why conjugate (Beta-Binomial) instead of MCMC (PyMC/Stan):

* For binary conversion metrics the Beta posterior is exact — MCMC would
  approximate an answer we can compute in closed form.
* No sampler diagnostics (divergences, R-hat) to babysit, so results are
  reproducible and instant, which matters for a dashboard.
* The honest limitation: conjugacy only covers simple likelihoods. For
  hierarchical models or revenue distributions with heavy tails, MCMC is
  the right tool — that trade-off is documented, not hidden.

Decision framework (Expected Loss) follows the approach popularized by
VWO / Chris Stucchio: ship when the expected loss of shipping drops below
a "threshold of caring" epsilon, rather than thresholding P(B > A) alone.
P(B>A)=95% with a huge potential downside is worse than P(B>A)=90% with a
negligible one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class BayesianResult:
    metric_name: str
    prob_treatment_better: float
    expected_loss_ship: float      # expected relative loss if we ship B and B is worse
    expected_loss_no_ship: float   # expected relative loss if we keep A and B was better
    lift_posterior_mean: float
    lift_ci_low: float             # 95% credible interval on relative lift
    lift_ci_high: float
    control_posterior: tuple[float, float]    # Beta(a, b)
    treatment_posterior: tuple[float, float]

    def summary(self, epsilon: float = 0.001) -> str:
        if self.expected_loss_ship < epsilon:
            decision = f"SHIP (expected loss {self.expected_loss_ship:.4%} < eps {epsilon:.2%})"
        elif self.expected_loss_no_ship < epsilon:
            decision = f"DO NOT SHIP (expected loss of keeping control {self.expected_loss_no_ship:.4%} < eps)"
        else:
            decision = "KEEP RUNNING (neither decision's expected loss is below epsilon)"
        return (
            f"[{self.metric_name}] P(treatment > control) = "
            f"{self.prob_treatment_better:.1%}, posterior lift "
            f"{self.lift_posterior_mean:+.2%} (95% CrI [{self.lift_ci_low:+.2%}, "
            f"{self.lift_ci_high:+.2%}]) -> {decision}"
        )


class BetaBinomialTest:
    """Beta-Binomial model for conversion-style metrics.

    Prior: Beta(prior_alpha, prior_beta). Default Beta(1, 1) (uniform) —
    with experiment-scale n, any weakly-informative prior is dominated by
    the data within a day or two of traffic. An informative prior from
    historical conversion rates can be passed for low-traffic experiments.
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior parameters must be positive")
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def analyze(
        self,
        control_conversions: int,
        control_n: int,
        treatment_conversions: int,
        treatment_n: int,
        metric_name: str = "conversion",
        n_samples: int = 200_000,
        seed: int = 42,
    ) -> BayesianResult:
        """Posterior inference by Monte Carlo over the two Beta posteriors.

        200k draws puts the MC error on P(B>A) around 0.1% — cheap and
        far below any decision boundary anyone uses.
        """
        rng = np.random.default_rng(seed)

        a_c = self.prior_alpha + control_conversions
        b_c = self.prior_beta + control_n - control_conversions
        a_t = self.prior_alpha + treatment_conversions
        b_t = self.prior_beta + treatment_n - treatment_conversions

        samples_c = rng.beta(a_c, b_c, n_samples)
        samples_t = rng.beta(a_t, b_t, n_samples)

        prob_better = float(np.mean(samples_t > samples_c))

        # Expected loss, in relative terms:
        # if we SHIP treatment, we lose (c - t)/c whenever c > t.
        loss_ship = float(np.mean(np.maximum(samples_c - samples_t, 0) / samples_c))
        # if we KEEP control, we forgo (t - c)/c whenever t > c.
        loss_no_ship = float(np.mean(np.maximum(samples_t - samples_c, 0) / samples_c))

        lift = (samples_t - samples_c) / samples_c
        ci_low, ci_high = np.percentile(lift, [2.5, 97.5])

        return BayesianResult(
            metric_name=metric_name,
            prob_treatment_better=prob_better,
            expected_loss_ship=loss_ship,
            expected_loss_no_ship=loss_no_ship,
            lift_posterior_mean=float(lift.mean()),
            lift_ci_low=float(ci_low),
            lift_ci_high=float(ci_high),
            control_posterior=(a_c, b_c),
            treatment_posterior=(a_t, b_t),
        )

    @staticmethod
    def posterior_pdf_grid(
        alpha: float, beta: float, points: int = 500
    ) -> tuple[np.ndarray, np.ndarray]:
        """(x, pdf) grid for plotting a Beta posterior."""
        x = np.linspace(0, 1, points)
        return x, stats.beta.pdf(x, alpha, beta)
