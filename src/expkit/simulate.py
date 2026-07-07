"""Synthetic experiment data generators.

Used by the worked example and the test suite. Each generator returns
unit-level data (one row per user) so the analysis code is exercised the
same way it would be against a warehouse export.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_conversion_experiment(
    n_per_arm: int,
    baseline: float,
    true_lift: float,
    n_days: int = 14,
    novelty_decay: float = 0.0,
    seed: int = 42,
    start_date: str = "2026-06-01",
) -> pd.DataFrame:
    """Unit-level conversion data with an optional novelty effect.

    Args:
        novelty_decay: If > 0, the treatment lift decays exponentially over
            the run: lift(day d) = true_lift * exp(-novelty_decay * d).
            novelty_decay=0.35 roughly halves the lift every 2 days.

    Returns:
        DataFrame[user_id, date, arm, converted]
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    per_day = n_per_arm // n_days

    frames = []
    uid = 0
    for d, date in enumerate(dates):
        lift_today = true_lift * np.exp(-novelty_decay * d)
        p_t = baseline * (1 + lift_today)
        for arm, p in (("control", baseline), ("treatment", p_t)):
            converted = rng.random(per_day) < p
            frames.append(
                pd.DataFrame(
                    {
                        "user_id": np.arange(uid, uid + per_day),
                        "date": date,
                        "arm": arm,
                        "converted": converted.astype(int),
                    }
                )
            )
            uid += per_day
    return pd.concat(frames, ignore_index=True)


def simulate_simpsons_paradox(
    seed: int = 42,
    n_total: int = 200_000,
) -> pd.DataFrame:
    """A Simpson's paradox scenario with a realistic mechanism.

    Setup: two device segments. The treatment genuinely improves
    conversion in BOTH segments (+8% relative on desktop, +8% on mobile).
    But the treatment page loads slower on mobile, so a larger share of
    mobile users make it into the treatment logs (a mix shift — in
    practice this often appears via triggering or bot-filter asymmetries).
    Mobile converts far below desktop, so pooling drags the treatment
    average down and the pooled lift comes out NEGATIVE.

    The pooled read is wrong; the segment read is right. This generator
    exists so the framework's segment_analysis can catch exactly this.

    Returns:
        DataFrame[user_id, date, arm, segment, converted]
    """
    rng = np.random.default_rng(seed)

    # (segment, arm) -> (share of that arm's traffic, conversion rate)
    spec = {
        ("desktop", "control"): (0.60, 0.100),
        ("desktop", "treatment"): (0.35, 0.108),  # +8% relative
        ("mobile", "control"): (0.40, 0.020),
        ("mobile", "treatment"): (0.65, 0.0216),  # +8% relative
    }
    n_arm = n_total // 2

    frames = []
    uid = 0
    for (segment, arm), (share, rate) in spec.items():
        n = int(n_arm * share)
        converted = rng.random(n) < rate
        frames.append(
            pd.DataFrame(
                {
                    "user_id": np.arange(uid, uid + n),
                    "date": pd.Timestamp("2026-06-01"),
                    "arm": arm,
                    "segment": segment,
                    "converted": converted.astype(int),
                }
            )
        )
        uid += n
    return pd.concat(frames, ignore_index=True)


def simulate_revenue_experiment(
    n_per_arm: int,
    conversion_rate: float = 0.05,
    mean_order_value: float = 60.0,
    aov_lift: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Continuous-metric experiment: revenue per user.

    Revenue is zero-inflated (most users buy nothing) and log-normal for
    buyers — the shape that makes naive t-tests underpowered in practice
    and motivates large n for revenue metrics.

    Returns:
        DataFrame[user_id, arm, revenue]
    """
    rng = np.random.default_rng(seed)
    sigma = 0.8  # log-normal shape for order values

    frames = []
    uid = 0
    for arm, aov in (
        ("control", mean_order_value),
        ("treatment", mean_order_value * (1 + aov_lift)),
    ):
        buys = rng.random(n_per_arm) < conversion_rate
        # scale log-normal so its mean equals the target AOV
        mu = np.log(aov) - sigma**2 / 2
        revenue = np.where(buys, rng.lognormal(mu, sigma, n_per_arm), 0.0)
        frames.append(
            pd.DataFrame(
                {
                    "user_id": np.arange(uid, uid + n_per_arm),
                    "arm": arm,
                    "revenue": revenue,
                }
            )
        )
        uid += n_per_arm
    return pd.concat(frames, ignore_index=True)
