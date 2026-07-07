"""Sample-size and power analysis.

Two approaches, deliberately both:

1. Analytic (closed-form) — fast, standard, what statsmodels gives you.
2. Simulation-based — slower but works for any metric/test combination,
   and doubles as a check on the analytic answer. If the two disagree
   materially, the analytic assumptions (normality, equal variance) are
   being violated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize


@dataclass
class PowerResult:
    n_per_arm: int
    total_n: int
    baseline: float
    treatment_value: float
    relative_mde: float
    alpha: float
    power: float
    metric_type: str

    def __str__(self) -> str:
        return (
            f"n = {self.n_per_arm:,} per arm ({self.total_n:,} total) to detect "
            f"a {self.relative_mde:+.1%} relative change from baseline "
            f"{self.baseline:g} at alpha={self.alpha}, power={self.power}"
        )


def power_analysis(
    baseline: float,
    mde_relative: float,
    metric_type: str = "proportion",
    alpha: float = 0.05,
    power: float = 0.80,
    std: float | None = None,
) -> PowerResult:
    """Required sample size per arm for a two-sided test.

    Args:
        baseline: Control conversion rate (proportion) or mean (continuous).
        mde_relative: Relative lift to detect, e.g. 0.02 for +2%.
        metric_type: "proportion" or "continuous".
        alpha: Two-sided significance level.
        power: Target power (1 - beta).
        std: Standard deviation of the metric (continuous only).

    Returns:
        PowerResult with n per arm.

    Note on MDE choice: the MDE should be the smallest effect worth acting
    on (a business judgment), not the effect you hope to see. Powering for
    an optimistic large effect is the most common way experiments end up
    underpowered.
    """
    treatment_value = baseline * (1 + mde_relative)

    if metric_type == "proportion":
        if not (0 < baseline < 1 and 0 < treatment_value < 1):
            raise ValueError("proportion rates must be in (0, 1)")
        # Cohen's h handles the variance depending on the rate itself.
        effect_size = proportion_effectsize(treatment_value, baseline)
        n = NormalIndPower().solve_power(
            effect_size=effect_size, alpha=alpha, power=power, ratio=1.0
        )
    elif metric_type == "continuous":
        if std is None:
            raise ValueError("std is required for continuous metrics")
        effect_size = (treatment_value - baseline) / std  # Cohen's d
        n = TTestIndPower().solve_power(
            effect_size=effect_size, alpha=alpha, power=power, ratio=1.0
        )
    else:
        raise ValueError(f"unknown metric_type: {metric_type}")

    n_per_arm = int(np.ceil(n))
    return PowerResult(
        n_per_arm=n_per_arm,
        total_n=2 * n_per_arm,
        baseline=baseline,
        treatment_value=treatment_value,
        relative_mde=mde_relative,
        alpha=alpha,
        power=power,
        metric_type=metric_type,
    )


def minimum_detectable_effect(
    baseline: float,
    n_per_arm: int,
    metric_type: str = "proportion",
    alpha: float = 0.05,
    power: float = 0.80,
    std: float | None = None,
) -> float:
    """Invert the power calculation: given the sample you can actually get,
    what relative lift can you reliably detect?

    Useful when traffic is fixed and the question is whether the experiment
    is worth running at all.
    """
    lo, hi = 1e-4, 5.0  # relative lift search range: 0.01% to 500%
    for _ in range(60):  # bisection
        mid = (lo + hi) / 2
        result = power_analysis(
            baseline, mid, metric_type=metric_type, alpha=alpha, power=power, std=std
        )
        if result.n_per_arm > n_per_arm:
            lo = mid
        else:
            hi = mid
    return hi


def simulated_power(
    baseline: float,
    mde_relative: float,
    n_per_arm: int,
    metric_type: str = "proportion",
    alpha: float = 0.05,
    std: float | None = None,
    n_sims: int = 2000,
    seed: int = 42,
) -> float:
    """Estimate power by Monte Carlo: simulate the experiment n_sims times
    under the alternative hypothesis and count how often the test rejects.

    Cross-checks the analytic formula and generalizes to metrics where no
    closed form exists (e.g. trimmed means, ratio metrics).
    """
    rng = np.random.default_rng(seed)
    treatment_value = baseline * (1 + mde_relative)
    rejections = 0

    for _ in range(n_sims):
        if metric_type == "proportion":
            c = rng.binomial(n_per_arm, baseline)
            t = rng.binomial(n_per_arm, treatment_value)
            p_pool = (c + t) / (2 * n_per_arm)
            se = np.sqrt(p_pool * (1 - p_pool) * 2 / n_per_arm)
            if se == 0:
                continue
            z = (t / n_per_arm - c / n_per_arm) / se
            p_value = 2 * stats.norm.sf(abs(z))
        elif metric_type == "continuous":
            if std is None:
                raise ValueError("std is required for continuous metrics")
            c = rng.normal(baseline, std, n_per_arm)
            t = rng.normal(treatment_value, std, n_per_arm)
            _, p_value = stats.ttest_ind(t, c, equal_var=False)
        else:
            raise ValueError(f"unknown metric_type: {metric_type}")

        if p_value < alpha:
            rejections += 1

    return rejections / n_sims
