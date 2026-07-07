"""expkit — an A/B testing and experimentation framework.

Modules:
    design       Experiment specification (hypothesis, metrics, guardrails).
    power        Sample-size and power analysis (analytic + simulation).
    frequentist  z/t tests, confidence intervals, relative lift.
    bayesian     Conjugate Bayesian inference: P(B > A), expected loss.
    sequential   Always-valid inference (mSPRT) and peeking simulation.
    validity     SRM, segment (Simpson's paradox) and novelty checks.
    simulate     Synthetic data generators for worked examples and tests.
    report       Assemble a full experiment readout.
"""

from expkit.design import ExperimentDesign, Metric, MetricType
from expkit.power import power_analysis, minimum_detectable_effect, simulated_power
from expkit.frequentist import proportion_ztest, welch_ttest, FrequentistResult
from expkit.bayesian import BetaBinomialTest, BayesianResult
from expkit.sequential import msprt_pvalue, PeekingSimulation
from expkit.validity import srm_check, segment_analysis, lift_over_time
from expkit.report import ExperimentReport

__version__ = "0.1.0"

__all__ = [
    "ExperimentDesign",
    "Metric",
    "MetricType",
    "power_analysis",
    "minimum_detectable_effect",
    "simulated_power",
    "proportion_ztest",
    "welch_ttest",
    "FrequentistResult",
    "BetaBinomialTest",
    "BayesianResult",
    "msprt_pvalue",
    "PeekingSimulation",
    "srm_check",
    "segment_analysis",
    "lift_over_time",
    "ExperimentReport",
]
