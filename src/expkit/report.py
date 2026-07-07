"""Assemble a full experiment readout.

The report enforces an order of operations:
    1. validity gates (SRM) — if these fail, nothing else is shown
    2. primary metric, frequentist + Bayesian side by side
    3. guardrails
    4. trend/segment diagnostics

The point of showing frequentist and Bayesian together: they answer
different questions ("is this surprising under no-effect?" vs. "what
should we do?") and agreement between them is a cheap robustness check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from expkit.bayesian import BetaBinomialTest, BayesianResult
from expkit.design import ExperimentDesign
from expkit.frequentist import proportion_ztest, FrequentistResult
from expkit.validity import (
    srm_check,
    SRMResult,
    segment_analysis,
    SegmentAnalysis,
    lift_over_time,
    LiftOverTime,
)


@dataclass
class ExperimentReport:
    design: ExperimentDesign
    srm: Optional[SRMResult] = None
    frequentist: Optional[FrequentistResult] = None
    bayesian: Optional[BayesianResult] = None
    trend: Optional[LiftOverTime] = None
    segments: Optional[SegmentAnalysis] = None
    guardrail_results: list[FrequentistResult] = field(default_factory=list)

    @classmethod
    def from_unit_data(
        cls,
        design: ExperimentDesign,
        df: pd.DataFrame,
        arm_col: str = "arm",
        converted_col: str = "converted",
        date_col: Optional[str] = "date",
        segment_col: Optional[str] = None,
    ) -> "ExperimentReport":
        """Build a report from one-row-per-unit data."""
        report = cls(design=design)

        counts = df[arm_col].value_counts()
        report.srm = srm_check(
            int(counts["control"]),
            int(counts["treatment"]),
            expected_treatment_share=design.traffic_split,
        )

        g = df.groupby(arm_col)[converted_col].agg(["sum", "count"])
        c_conv, c_n = int(g.loc["control", "sum"]), int(g.loc["control", "count"])
        t_conv, t_n = int(g.loc["treatment", "sum"]), int(g.loc["treatment", "count"])

        report.frequentist = proportion_ztest(
            c_conv, c_n, t_conv, t_n,
            alpha=design.alpha,
            metric_name=design.primary_metric.name,
        )
        report.bayesian = BetaBinomialTest().analyze(
            c_conv, c_n, t_conv, t_n, metric_name=design.primary_metric.name
        )

        if date_col is not None and df[date_col].nunique() > 3:
            report.trend = lift_over_time(
                df, date_col=date_col, arm_col=arm_col, converted_col=converted_col
            )
        if segment_col is not None:
            report.segments = segment_analysis(
                df, segment_col=segment_col, arm_col=arm_col,
                converted_col=converted_col, alpha=design.alpha,
            )
        return report

    def render(self, epsilon: float = 0.001) -> str:
        """Plain-text readout, in gated order."""
        lines = [
            f"EXPERIMENT READOUT — {self.design.name}",
            f"Hypothesis: {self.design.hypothesis}",
            "=" * 72,
        ]

        if self.srm is not None:
            lines += ["", "1. Validity", "-" * 30, self.srm.summary()]
            if not self.srm.passed:
                lines += [
                    "",
                    "READOUT BLOCKED: sample ratio mismatch. Fix assignment/",
                    "logging before interpreting any metric below.",
                ]
                return "\n".join(lines)

        if self.frequentist is not None:
            lines += ["", "2. Primary metric", "-" * 30, self.frequentist.summary()]
        if self.bayesian is not None:
            lines += [self.bayesian.summary(epsilon=epsilon)]

        if self.guardrail_results:
            lines += ["", "3. Guardrails", "-" * 30]
            lines += [g.summary() for g in self.guardrail_results]

        diagnostics = []
        if self.trend is not None:
            diagnostics.append(self.trend.summary())
        if self.segments is not None:
            diagnostics.append(self.segments.summary())
        if diagnostics:
            lines += ["", "4. Diagnostics", "-" * 30]
            lines += diagnostics

        return "\n".join(lines)
