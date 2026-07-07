"""Frequentist analysis: two-proportion z-test, Welch's t-test, and
confidence intervals for both absolute and relative lift.

Design choices worth defending in an interview:

* Welch's t-test (unequal variances) is the default for continuous
  metrics. Student's t assumes equal variances, which experiment data
  rarely satisfies; Welch costs almost nothing when variances happen to
  be equal and is much safer when they aren't.
* The relative-lift CI uses the delta method (Fieller-style approximation)
  because "revenue up 3.2% [1.1%, 5.3%]" is how results are actually
  communicated — an absolute CI alone invites misreading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class FrequentistResult:
    metric_name: str
    test: str
    control_n: int
    treatment_n: int
    control_value: float
    treatment_value: float
    absolute_lift: float
    relative_lift: float
    ci_low_abs: float
    ci_high_abs: float
    ci_low_rel: float
    ci_high_rel: float
    statistic: float
    p_value: float
    alpha: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def summary(self) -> str:
        sig = "significant" if self.significant else "NOT significant"
        return (
            f"[{self.metric_name}] {self.test}: control={self.control_value:.4g}, "
            f"treatment={self.treatment_value:.4g}, lift={self.relative_lift:+.2%} "
            f"(95% CI [{self.ci_low_rel:+.2%}, {self.ci_high_rel:+.2%}]), "
            f"p={self.p_value:.4g} -> {sig} at alpha={self.alpha}"
        )


def proportion_ztest(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    alpha: float = 0.05,
    metric_name: str = "conversion",
) -> FrequentistResult:
    """Two-sided two-proportion z-test with pooled SE for the test statistic
    and unpooled SE for the confidence interval (the standard pairing:
    pooled variance is correct under H0, unpooled under H1)."""
    if control_conversions > control_n or treatment_conversions > treatment_n:
        raise ValueError("conversions cannot exceed sample size")

    p_c = control_conversions / control_n
    p_t = treatment_conversions / treatment_n
    diff = p_t - p_c

    # Test statistic: pooled SE (variance under the null that p_c == p_t).
    p_pool = (control_conversions + treatment_conversions) / (control_n + treatment_n)
    se_pooled = np.sqrt(p_pool * (1 - p_pool) * (1 / control_n + 1 / treatment_n))
    z = diff / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * stats.norm.sf(abs(z))

    # CI: unpooled SE.
    se_unpooled = np.sqrt(
        p_c * (1 - p_c) / control_n + p_t * (1 - p_t) / treatment_n
    )
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_low_abs = diff - z_crit * se_unpooled
    ci_high_abs = diff + z_crit * se_unpooled

    rel, rel_lo, rel_hi = _relative_lift_ci(
        p_c, se_of_mean_c=np.sqrt(p_c * (1 - p_c) / control_n),
        mean_t=p_t, se_of_mean_t=np.sqrt(p_t * (1 - p_t) / treatment_n),
        z_crit=z_crit,
    )

    return FrequentistResult(
        metric_name=metric_name,
        test="two-proportion z-test",
        control_n=control_n,
        treatment_n=treatment_n,
        control_value=p_c,
        treatment_value=p_t,
        absolute_lift=diff,
        relative_lift=rel,
        ci_low_abs=ci_low_abs,
        ci_high_abs=ci_high_abs,
        ci_low_rel=rel_lo,
        ci_high_rel=rel_hi,
        statistic=z,
        p_value=float(p_value),
        alpha=alpha,
    )


def welch_ttest(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    metric_name: str = "metric",
) -> FrequentistResult:
    """Welch's two-sided t-test for continuous metrics."""
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    mean_c, mean_t = control.mean(), treatment.mean()
    var_c, var_t = control.var(ddof=1), treatment.var(ddof=1)
    n_c, n_t = len(control), len(treatment)
    diff = mean_t - mean_c

    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

    # Welch-Satterthwaite degrees of freedom for the CI.
    se = np.sqrt(var_c / n_c + var_t / n_t)
    df = (var_c / n_c + var_t / n_t) ** 2 / (
        (var_c / n_c) ** 2 / (n_c - 1) + (var_t / n_t) ** 2 / (n_t - 1)
    )
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_low_abs = diff - t_crit * se
    ci_high_abs = diff + t_crit * se

    rel, rel_lo, rel_hi = _relative_lift_ci(
        mean_c, se_of_mean_c=np.sqrt(var_c / n_c),
        mean_t=mean_t, se_of_mean_t=np.sqrt(var_t / n_t),
        z_crit=t_crit,
    )

    return FrequentistResult(
        metric_name=metric_name,
        test="Welch's t-test",
        control_n=n_c,
        treatment_n=n_t,
        control_value=mean_c,
        treatment_value=mean_t,
        absolute_lift=diff,
        relative_lift=rel,
        ci_low_abs=ci_low_abs,
        ci_high_abs=ci_high_abs,
        ci_low_rel=rel_lo,
        ci_high_rel=rel_hi,
        statistic=float(t_stat),
        p_value=float(p_value),
        alpha=alpha,
    )


def _relative_lift_ci(
    mean_c: float,
    se_of_mean_c: float,
    mean_t: float,
    se_of_mean_t: float,
    z_crit: float,
) -> tuple[float, float, float]:
    """Delta-method CI for relative lift (mean_t - mean_c) / mean_c.

    Var(t/c) ~ (1/c^2) Var(t) + (t^2/c^4) Var(c) for independent arms.
    Adequate when mean_c is well away from zero relative to its SE; for
    metrics where that fails, bootstrap instead.
    """
    if mean_c == 0:
        return float("nan"), float("nan"), float("nan")
    rel = (mean_t - mean_c) / mean_c
    var_ratio = (se_of_mean_t**2) / mean_c**2 + (
        mean_t**2 * se_of_mean_c**2
    ) / mean_c**4
    se_rel = np.sqrt(var_ratio)
    return rel, rel - z_crit * se_rel, rel + z_crit * se_rel
