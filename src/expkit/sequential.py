"""Sequential testing: the peeking problem and an always-valid answer.

The problem: computing a classic p-value every day and stopping the first
time it dips under 0.05 inflates the false-positive rate far above 5% —
with daily peeks over a few weeks, empirically 2-6x. The fixed-horizon
p-value is only valid at a sample size chosen in advance.

The fix implemented here: the mixture Sequential Probability Ratio Test
(mSPRT, Johari, Pekelis & Walsh 2017 — the method behind Optimizely's
"Stats Engine"). It produces an *always-valid* p-value: a process p_n
that is monotone non-increasing and satisfies P(exists n: p_n <= alpha)
<= alpha under H0. You may peek every hour and stop whenever you like.

The cost is power at a fixed n — always-valid inference needs more data
to call the same effect. That trade (validity under continuous monitoring
vs. sample efficiency) is the core design decision, and the peeking
simulation in this module quantifies both sides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


def msprt_pvalue(
    theta_hat: float,
    variance_of_estimate: float,
    tau: float,
) -> float:
    """One-step always-valid p-value from the normal-mixture SPRT.

    The treatment-effect estimate theta_hat is modeled as
    N(theta, variance_of_estimate); H0: theta = 0; H1 mixes theta over
    N(0, tau^2). The likelihood ratio is then

        Lambda = sqrt(V / (V + tau^2)) *
                 exp( theta_hat^2 * tau^2 / (2 * V * (V + tau^2)) )

    and p = min(1, 1/Lambda). To use sequentially, call at each look and
    keep the running minimum (see AlwaysValidTracker).

    Args:
        theta_hat: Estimated difference in means (treatment - control).
        variance_of_estimate: Var(theta_hat), e.g. s_c^2/n_c + s_t^2/n_t.
        tau: Mixture scale — roughly the size of effects you expect to
            detect. Power is maximized when tau matches the true effect;
            it degrades gently for misspecified tau (validity never does).
    """
    if variance_of_estimate <= 0:
        raise ValueError("variance_of_estimate must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")

    v = variance_of_estimate
    t2 = tau**2
    log_lambda = 0.5 * np.log(v / (v + t2)) + (
        theta_hat**2 * t2 / (2 * v * (v + t2))
    )
    return float(min(1.0, np.exp(-log_lambda)))


@dataclass
class AlwaysValidTracker:
    """Tracks the running always-valid p-value across looks."""

    tau: float
    alpha: float = 0.05
    p_value: float = 1.0
    history: list[float] = field(default_factory=list)

    def update(self, theta_hat: float, variance_of_estimate: float) -> float:
        p_now = msprt_pvalue(theta_hat, variance_of_estimate, self.tau)
        self.p_value = min(self.p_value, p_now)
        self.history.append(self.p_value)
        return self.p_value

    @property
    def can_stop(self) -> bool:
        return self.p_value <= self.alpha


@dataclass
class PeekingResult:
    n_experiments: int
    n_looks: int
    naive_false_positive_rate: float
    msprt_false_positive_rate: float
    nominal_alpha: float

    def summary(self) -> str:
        return (
            f"A/A simulation, {self.n_experiments} experiments, "
            f"{self.n_looks} looks each (nominal alpha={self.nominal_alpha}):\n"
            f"  naive repeated z-test stopped early (false positive): "
            f"{self.naive_false_positive_rate:.1%}\n"
            f"  mSPRT always-valid p-value:                           "
            f"{self.msprt_false_positive_rate:.1%}"
        )


class PeekingSimulation:
    """A/A simulation quantifying alpha inflation from peeking.

    Simulates experiments where control and treatment share the same true
    rate (any 'significant' result is a false positive), analyzed at
    n_looks evenly spaced interim points with (a) a naive z-test and
    (b) the mSPRT always-valid p-value.
    """

    def __init__(
        self,
        baseline: float = 0.05,
        n_per_arm: int = 20_000,
        n_looks: int = 20,
        alpha: float = 0.05,
        tau: float | None = None,
        seed: int = 42,
    ):
        self.baseline = baseline
        self.n_per_arm = n_per_arm
        self.n_looks = n_looks
        self.alpha = alpha
        # Default tau: a plausible effect scale, ~10% relative lift.
        self.tau = tau if tau is not None else 0.1 * baseline
        self.rng = np.random.default_rng(seed)

    def run(self, n_experiments: int = 1000) -> PeekingResult:
        look_points = np.linspace(
            self.n_per_arm / self.n_looks, self.n_per_arm, self.n_looks
        ).astype(int)

        naive_fp = 0
        msprt_fp = 0

        for _ in range(n_experiments):
            control = self.rng.random(self.n_per_arm) < self.baseline
            treatment = self.rng.random(self.n_per_arm) < self.baseline

            csum_c = np.cumsum(control)
            csum_t = np.cumsum(treatment)

            naive_hit = False
            tracker = AlwaysValidTracker(tau=self.tau, alpha=self.alpha)

            for n in look_points:
                p_c = csum_c[n - 1] / n
                p_t = csum_t[n - 1] / n
                pool = (csum_c[n - 1] + csum_t[n - 1]) / (2 * n)
                var_hat = pool * (1 - pool) * 2 / n
                if var_hat == 0:
                    continue

                # naive fixed-horizon z-test at this look
                z = (p_t - p_c) / np.sqrt(var_hat)
                if 2 * stats.norm.sf(abs(z)) < self.alpha:
                    naive_hit = True

                tracker.update(p_t - p_c, var_hat)

            naive_fp += int(naive_hit)
            msprt_fp += int(tracker.can_stop)

        return PeekingResult(
            n_experiments=n_experiments,
            n_looks=self.n_looks,
            naive_false_positive_rate=naive_fp / n_experiments,
            msprt_false_positive_rate=msprt_fp / n_experiments,
            nominal_alpha=self.alpha,
        )
