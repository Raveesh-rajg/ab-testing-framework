"""Experiment specification.

An experiment is declared before any data is collected. Forcing the
hypothesis, metrics, guardrails, and stopping rule into a written spec is
the single cheapest defense against p-hacking: the analysis plan exists
before the results do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MetricType(str, Enum):
    """How a metric is measured, which determines the test used."""

    PROPORTION = "proportion"  # binary per unit (converted / not) -> z-test
    CONTINUOUS = "continuous"  # real-valued per unit (revenue)    -> Welch t


class MetricRole(str, Enum):
    PRIMARY = "primary"      # the metric the ship decision is based on
    SECONDARY = "secondary"  # informative, not decision-driving
    GUARDRAIL = "guardrail"  # must not regress (e.g. latency, unsubscribes)


@dataclass
class Metric:
    """A single experiment metric.

    Attributes:
        name: Human-readable identifier, e.g. "checkout_conversion".
        metric_type: PROPORTION or CONTINUOUS.
        role: PRIMARY, SECONDARY, or GUARDRAIL.
        baseline: Expected control value (rate for proportions, mean for
            continuous). Used for power analysis.
        mde_relative: Minimum detectable effect as a relative lift the team
            actually cares about (0.02 = +2%). Effects smaller than this are
            not worth shipping even if statistically significant.
        guardrail_threshold: For guardrails, the maximum tolerated relative
            regression (e.g. 0.01 = at most 1% worse).
    """

    name: str
    metric_type: MetricType
    role: MetricRole = MetricRole.PRIMARY
    baseline: Optional[float] = None
    mde_relative: Optional[float] = None
    guardrail_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        if self.role == MetricRole.GUARDRAIL and self.guardrail_threshold is None:
            raise ValueError(
                f"Guardrail metric '{self.name}' needs guardrail_threshold."
            )


@dataclass
class ExperimentDesign:
    """The pre-registration document for an experiment.

    Attributes:
        name: Experiment identifier.
        hypothesis: One-sentence falsifiable statement:
            "Changing X will move metric M by at least MDE because ...".
        unit: Randomization unit ("user", "session", ...). Analysis must
            happen at the same unit, otherwise variance is understated.
        metrics: All metrics. Exactly one PRIMARY is enforced — more than
            one primary metric is a multiple-comparisons problem disguised
            as thoroughness.
        alpha: Two-sided false-positive rate (default 0.05).
        power: Target power (default 0.80).
        traffic_split: Fraction of traffic to treatment (0.5 = 50/50).
        expected_daily_units: Traffic estimate used to convert the required
            sample size into a runtime in days.
        min_runtime_days: Floor on runtime regardless of sample size, to
            cover at least one weekly seasonality cycle (default 7).
    """

    name: str
    hypothesis: str
    unit: str
    metrics: list[Metric] = field(default_factory=list)
    alpha: float = 0.05
    power: float = 0.80
    traffic_split: float = 0.5
    expected_daily_units: Optional[int] = None
    min_runtime_days: int = 7

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not 0 < self.power < 1:
            raise ValueError("power must be in (0, 1)")
        if not 0 < self.traffic_split < 1:
            raise ValueError("traffic_split must be in (0, 1)")
        primaries = [m for m in self.metrics if m.role == MetricRole.PRIMARY]
        if len(primaries) != 1:
            raise ValueError(
                f"Experiment must declare exactly one primary metric, "
                f"got {len(primaries)}."
            )

    @property
    def primary_metric(self) -> Metric:
        return next(m for m in self.metrics if m.role == MetricRole.PRIMARY)

    @property
    def guardrails(self) -> list[Metric]:
        return [m for m in self.metrics if m.role == MetricRole.GUARDRAIL]

    def runtime_days(self, required_n_per_arm: int) -> int:
        """Days needed to reach the required sample size, respecting the
        weekly-seasonality floor."""
        if self.expected_daily_units is None:
            raise ValueError("expected_daily_units not set")
        units_to_treatment = self.expected_daily_units * self.traffic_split
        units_to_control = self.expected_daily_units * (1 - self.traffic_split)
        limiting = min(units_to_treatment, units_to_control)
        days = int(-(-required_n_per_arm // limiting))  # ceil division
        return max(days, self.min_runtime_days)
