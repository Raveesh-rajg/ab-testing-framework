"""Validity checks that run before anyone looks at lift numbers.

An experiment with a broken randomizer or a confounded traffic mix will
happily produce a beautiful, significant, wrong answer. These checks are
the experimentation equivalent of dbt tests: cheap, automatic, and they
gate the readout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from expkit.frequentist import proportion_ztest, FrequentistResult


@dataclass
class SRMResult:
    control_n: int
    treatment_n: int
    expected_ratio: float
    observed_ratio: float
    chi2: float
    p_value: float
    passed: bool

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL — investigate before reading results"
        return (
            f"SRM check: expected split {self.expected_ratio:.2f}, observed "
            f"{self.observed_ratio:.4f} ({self.control_n:,} vs {self.treatment_n:,}), "
            f"chi2 p={self.p_value:.4g} -> {status}"
        )


def srm_check(
    control_n: int,
    treatment_n: int,
    expected_treatment_share: float = 0.5,
    threshold: float = 0.001,
) -> SRMResult:
    """Sample Ratio Mismatch check via chi-square goodness of fit.

    The threshold is deliberately strict (0.001, not 0.05): with millions
    of units even tiny assignment bugs produce astronomically small
    p-values, and a false SRM alarm is cheap while a missed one poisons
    the whole readout. An SRM failure means data loss, redirect bugs, or
    bot filtering hitting one arm harder — the experiment result cannot be
    trusted until the cause is found.
    """
    total = control_n + treatment_n
    expected = np.array(
        [total * (1 - expected_treatment_share), total * expected_treatment_share]
    )
    observed = np.array([control_n, treatment_n])
    chi2, p_value = stats.chisquare(observed, expected)
    return SRMResult(
        control_n=control_n,
        treatment_n=treatment_n,
        expected_ratio=expected_treatment_share,
        observed_ratio=treatment_n / total,
        chi2=float(chi2),
        p_value=float(p_value),
        passed=p_value >= threshold,
    )


@dataclass
class SegmentAnalysis:
    overall: FrequentistResult
    by_segment: pd.DataFrame
    simpsons_flag: bool

    def summary(self) -> str:
        lines = [self.overall.summary(), "", "By segment:"]
        for _, row in self.by_segment.iterrows():
            lines.append(
                f"  {row['segment']}: control {row['control_rate']:.4f} "
                f"(n={row['control_n']:,}), treatment {row['treatment_rate']:.4f} "
                f"(n={row['treatment_n']:,}), lift {row['relative_lift']:+.2%}, "
                f"p={row['p_value']:.3g}"
            )
        if self.simpsons_flag:
            lines.append(
                "\n  WARNING: Simpson's paradox pattern — the pooled direction "
                "disagrees with the (consistent) within-segment direction. The "
                "pooled number is confounded by segment mix; read segment-level "
                "results, and check why the mix differs between arms."
            )
        return "\n".join(lines)


def segment_analysis(
    df: pd.DataFrame,
    segment_col: str,
    arm_col: str = "arm",
    converted_col: str = "converted",
    alpha: float = 0.05,
) -> SegmentAnalysis:
    """Pooled vs. per-segment readout with a Simpson's paradox detector.

    Expects one row per unit with arm in {'control','treatment'} and a
    binary conversion column.

    The flag fires when every segment (with material sample) moves in one
    direction while the pooled estimate moves the other way — the textbook
    signature of a mix shift confounding the pooled result.
    """
    pooled = _rates(df, arm_col, converted_col)
    overall = proportion_ztest(
        pooled["control_conv"], pooled["control_n"],
        pooled["treatment_conv"], pooled["treatment_n"],
        alpha=alpha, metric_name="pooled",
    )

    rows = []
    for seg, seg_df in df.groupby(segment_col):
        r = _rates(seg_df, arm_col, converted_col)
        if min(r["control_n"], r["treatment_n"]) < 50:
            continue  # too small to say anything
        res = proportion_ztest(
            r["control_conv"], r["control_n"],
            r["treatment_conv"], r["treatment_n"],
            alpha=alpha, metric_name=str(seg),
        )
        rows.append(
            {
                "segment": seg,
                "control_n": r["control_n"],
                "treatment_n": r["treatment_n"],
                "control_rate": res.control_value,
                "treatment_rate": res.treatment_value,
                "relative_lift": res.relative_lift,
                "p_value": res.p_value,
            }
        )
    by_segment = pd.DataFrame(rows)

    simpsons = False
    if len(by_segment) >= 2:
        seg_signs = np.sign(by_segment["relative_lift"])
        pooled_sign = np.sign(overall.relative_lift)
        if (seg_signs == seg_signs.iloc[0]).all() and seg_signs.iloc[0] != pooled_sign:
            simpsons = True

    return SegmentAnalysis(overall=overall, by_segment=by_segment, simpsons_flag=simpsons)


@dataclass
class LiftOverTime:
    daily: pd.DataFrame
    novelty_flag: bool
    first_half_lift: float
    second_half_lift: float

    def summary(self) -> str:
        base = (
            f"Lift over time: first half {self.first_half_lift:+.2%}, "
            f"second half {self.second_half_lift:+.2%}."
        )
        if self.novelty_flag:
            base += (
                " WARNING: novelty-effect pattern — the lift decays materially "
                "over the run. The long-run effect is closer to the late-period "
                "estimate; do not project the full-period average forward."
            )
        return base


def lift_over_time(
    df: pd.DataFrame,
    date_col: str = "date",
    arm_col: str = "arm",
    converted_col: str = "converted",
    decay_threshold: float = 0.5,
) -> LiftOverTime:
    """Daily relative lift and a novelty-effect heuristic.

    Flags novelty when the second-half lift is less than
    (1 - decay_threshold) x the first-half lift and both halves have a
    positive first-half lift to decay from. A heuristic, not a test —
    it exists to force a human look at the trend chart.
    """
    daily = (
        df.groupby([date_col, arm_col])[converted_col]
        .agg(["mean", "count"])
        .unstack(arm_col)
    )
    lift = (
        daily[("mean", "treatment")] - daily[("mean", "control")]
    ) / daily[("mean", "control")]
    out = pd.DataFrame(
        {
            "control_rate": daily[("mean", "control")],
            "treatment_rate": daily[("mean", "treatment")],
            "relative_lift": lift,
        }
    ).reset_index()

    half = len(out) // 2
    first = _period_lift(df, out[date_col].iloc[:half], date_col, arm_col, converted_col)
    second = _period_lift(df, out[date_col].iloc[half:], date_col, arm_col, converted_col)

    novelty = first > 0 and second < first * (1 - decay_threshold)

    return LiftOverTime(
        daily=out,
        novelty_flag=bool(novelty),
        first_half_lift=first,
        second_half_lift=second,
    )


def _period_lift(df, dates, date_col, arm_col, converted_col) -> float:
    sub = df[df[date_col].isin(dates)]
    rates = sub.groupby(arm_col)[converted_col].mean()
    return float((rates["treatment"] - rates["control"]) / rates["control"])


def _rates(df: pd.DataFrame, arm_col: str, converted_col: str) -> dict:
    g = df.groupby(arm_col)[converted_col].agg(["sum", "count"])
    return {
        "control_conv": int(g.loc["control", "sum"]),
        "control_n": int(g.loc["control", "count"]),
        "treatment_conv": int(g.loc["treatment", "sum"]),
        "treatment_n": int(g.loc["treatment", "count"]),
    }
