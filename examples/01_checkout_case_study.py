"""End-to-end worked example: a checkout-flow experiment where the naive
read is wrong twice.

Scenario: an e-commerce team ships a redesigned checkout button and runs a
14-day experiment. Three analysis traps are planted in the data, and the
framework catches each one:

    1. Peeking     — the PM checks the dashboard daily and wants to stop
                     on day 4 when p < 0.05 flashes green.
    2. Novelty     — the lift decays over the run; the full-period average
                     overstates the long-run effect.
    3. Simpson's   — a companion experiment where the pooled lift is
                     negative while every segment improved.

Run:  PYTHONPATH=src python examples/01_checkout_case_study.py
Outputs charts to examples/output/ and prints the readout.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from expkit import (
    ExperimentDesign,
    Metric,
    MetricType,
    ExperimentReport,
    PeekingSimulation,
    power_analysis,
    segment_analysis,
)
from expkit.design import MetricRole
from expkit.bayesian import BetaBinomialTest
from expkit.simulate import simulate_conversion_experiment, simulate_simpsons_paradox

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. DESIGN — written before any data exists
    # ------------------------------------------------------------------
    design = ExperimentDesign(
        name="checkout_redesign_2026_06",
        hypothesis=(
            "Replacing the two-step checkout confirmation with a single "
            "button will raise checkout conversion by at least 5% relative, "
            "because funnel logs show 18% of drop-off happens on the "
            "confirmation step."
        ),
        unit="user",
        metrics=[
            Metric("checkout_conversion", MetricType.PROPORTION,
                   role=MetricRole.PRIMARY, baseline=0.05, mde_relative=0.05),
            Metric("support_ticket_rate", MetricType.PROPORTION,
                   role=MetricRole.GUARDRAIL, baseline=0.01,
                   guardrail_threshold=0.10),
        ],
        expected_daily_units=40_000,
    )

    pw = power_analysis(baseline=0.05, mde_relative=0.05)
    print("=" * 72)
    print("DESIGN & POWER")
    print("=" * 72)
    print(pw)
    print(f"Runtime at {design.expected_daily_units:,} users/day: "
          f"{design.runtime_days(pw.n_per_arm)} days\n")

    # ------------------------------------------------------------------
    # 2. DATA — 14 days, true +12% lift that decays (novelty effect):
    #    day-0 lift 12%, decaying ~15%/day -> long-run lift ~2-3%
    # ------------------------------------------------------------------
    df = simulate_conversion_experiment(
        n_per_arm=280_000, baseline=0.05, true_lift=0.12,
        n_days=14, novelty_decay=0.15, seed=11,
    )

    # ------------------------------------------------------------------
    # 3. TRAP 1 — peeking. What if we'd stopped at the first green p-value?
    # ------------------------------------------------------------------
    print("=" * 72)
    print("TRAP 1: PEEKING (A/A simulation)")
    print("=" * 72)
    peek = PeekingSimulation(baseline=0.05, n_per_arm=20_000, n_looks=14, seed=11)
    peek_result = peek.run(n_experiments=500)
    print(peek_result.summary())
    print()

    # ------------------------------------------------------------------
    # 4. FULL READOUT — SRM gate, frequentist + Bayesian, novelty check
    # ------------------------------------------------------------------
    print("=" * 72)
    print("FULL READOUT")
    print("=" * 72)
    report = ExperimentReport.from_unit_data(design, df)
    print(report.render())
    print()

    # ------------------------------------------------------------------
    # 5. TRAP 2 — novelty: compare full-period vs late-period estimates
    # ------------------------------------------------------------------
    late = df[df["date"] >= df["date"].max() - np.timedelta64(4, "D")]
    g = late.groupby("arm")["converted"].agg(["sum", "count"])
    late_bayes = BetaBinomialTest().analyze(
        int(g.loc["control", "sum"]), int(g.loc["control", "count"]),
        int(g.loc["treatment", "sum"]), int(g.loc["treatment", "count"]),
        metric_name="checkout_conversion (last 5 days only)",
    )
    print("=" * 72)
    print("TRAP 2: NOVELTY — full-period vs last-5-days estimate")
    print("=" * 72)
    print(f"Full period posterior lift : {report.bayesian.lift_posterior_mean:+.2%}")
    print(f"Last 5 days posterior lift : {late_bayes.lift_posterior_mean:+.2%}")
    print("The full-period average overstates what shipping will deliver.\n")

    # ------------------------------------------------------------------
    # 6. TRAP 3 — Simpson's paradox (companion experiment)
    # ------------------------------------------------------------------
    print("=" * 72)
    print("TRAP 3: SIMPSON'S PARADOX (companion experiment)")
    print("=" * 72)
    sp_df = simulate_simpsons_paradox(seed=11)
    sp = segment_analysis(sp_df, segment_col="segment")
    print(sp.summary())
    print()

    # ------------------------------------------------------------------
    # 7. CHARTS
    # ------------------------------------------------------------------
    _plot_daily_lift(report)
    _plot_posteriors(report)
    _plot_peeking(peek_result)
    print(f"Charts written to {OUT}/")


def _plot_daily_lift(report: ExperimentReport) -> None:
    d = report.trend.daily
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(d["date"], d["relative_lift"] * 100, marker="o", lw=1.5)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("Daily relative lift — novelty decay is visible by week 2")
    ax.set_ylabel("relative lift (%)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "daily_lift.png", dpi=150)
    plt.close(fig)


def _plot_posteriors(report: ExperimentReport) -> None:
    b = report.bayesian
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, (a, be) in (
        ("control", b.control_posterior),
        ("treatment", b.treatment_posterior),
    ):
        x, pdf = BetaBinomialTest.posterior_pdf_grid(a, be)
        mean = a / (a + be)
        mask = (x > mean - 0.004) & (x < mean + 0.004)
        ax.plot(x[mask], pdf[mask], label=label)
        ax.fill_between(x[mask], pdf[mask], alpha=0.2)
    ax.set_title(
        f"Posterior conversion rates — P(treatment > control) = "
        f"{b.prob_treatment_better:.1%}"
    )
    ax.set_xlabel("conversion rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "posteriors.png", dpi=150)
    plt.close(fig)


def _plot_peeking(result) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        ["nominal α", "naive daily peeking", "mSPRT (always-valid)"],
        [result.nominal_alpha,
         result.naive_false_positive_rate,
         result.msprt_false_positive_rate],
        color=["#888888", "#c0392b", "#27ae60"],
    )
    ax.bar_label(bars, fmt="%.3f")
    ax.set_ylabel("false positive rate (A/A)")
    ax.set_title("Peeking inflates false positives; always-valid inference doesn't")
    fig.tight_layout()
    fig.savefig(OUT / "peeking.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
