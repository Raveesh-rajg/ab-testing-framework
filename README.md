# Experiment Decision System | Design, validity, inference, and ship rules

A Python framework covering the full experiment lifecycle: pre-registration and power
analysis, frequentist and Bayesian inference, always-valid sequential testing, and the
validity checks (SRM, Simpson's paradox, novelty effects) that decide whether the lift
numbers can be trusted at all.

Built with scipy and statsmodels. 38 unit tests, including empirical coverage and
false-positive-rate checks — the statistical claims in this README are verified by the
test suite, not just asserted.

## Why this exists

Most A/B test write-ups stop at "run a t-test, check p < 0.05." In practice the test
itself is the easy part. Experiments fail in quieter ways: someone stops the test the
first day the dashboard turns green, a logging bug drops 2% of one arm's traffic, a
mix shift makes the pooled number point the opposite direction from every segment.
This framework treats those failure modes as first-class features, not footnotes.

The worked example (`examples/01_checkout_case_study.py`) plants three of these traps
in simulated data and shows the framework catching each one. All numbers below come
from running it (seeded, reproducible).

**Trap 1 — peeking.** In an A/A simulation (no true effect, 500 experiments, 14 interim
looks), stopping at the first significant naive z-test produced a **24.0% false-positive
rate** against a nominal 5%. The always-valid mSPRT p-value held at **1.2%**.

**Trap 2 — novelty.** The simulated treatment starts at +12% lift and decays. The
full-period estimate reads **+3.67%**, the last-5-days estimate **+1.86%** — shipping
based on the full-period average would roughly double-count the benefit. The framework
flags the decay automatically by comparing first-half vs second-half lift.

**Trap 3 — Simpson's paradox.** In the companion dataset the pooled lift is **−22.9%
(p < 1e-47)** while desktop is **+9.6%** and mobile **+9.8%**. The segment analyzer
detects the sign disagreement and blocks the pooled read.

## Structure

```
src/expkit/
├── design.py       # pre-registration: hypothesis, metrics, guardrails, runtime
├── power.py        # analytic + simulation-based power, MDE inversion
├── frequentist.py  # 2-proportion z, Welch t, absolute + relative-lift CIs
├── bayesian.py     # Beta-Binomial: P(B>A), expected loss, credible intervals
├── sequential.py   # mSPRT always-valid p-values, peeking simulator
├── validity.py     # SRM gate, segment/Simpson's detector, novelty check
├── simulate.py     # unit-level data generators (used by examples + tests)
└── report.py       # gated readout: validity -> primary -> guardrails -> diagnostics
tests/              # 38 tests incl. CI coverage & FPR verification
examples/           # end-to-end case study with charts
dashboard/          # Streamlit app: planner, analyzer, peeking demo
docs/               # design decisions + PM-facing results writeup
```

## Quick start

```bash
pip install -e ".[dev,dashboard]"

pytest                                            # run the test suite
PYTHONPATH=src python examples/01_checkout_case_study.py   # the case study
PYTHONPATH=src streamlit run dashboard/app.py     # the dashboard
```

```python
from expkit import power_analysis, proportion_ztest
from expkit.bayesian import BetaBinomialTest

# How long do we need to run?
print(power_analysis(baseline=0.05, mde_relative=0.05))
# n = 122,107 per arm (244,214 total) ...

# What happened?
freq = proportion_ztest(5000, 100_000, 5300, 100_000)
bayes = BetaBinomialTest().analyze(5000, 100_000, 5300, 100_000)
print(freq.summary())
print(bayes.summary())
```

## Design decisions

The short version — `docs/DECISIONS.md` has the full reasoning:

- **Welch's t by default** for continuous metrics; the equal-variance assumption buys
  nothing and breaks silently.
- **Conjugate Bayesian (Beta-Binomial), not MCMC.** For binary metrics the posterior
  is exact and instant; MCMC would approximate an answer we can compute in closed form.
  The trade-off (no hierarchical models, no heavy-tailed likelihoods) is documented.
- **Expected loss as the Bayesian decision rule**, not a bare P(B>A) threshold — a 95%
  win probability with a large potential downside is worse than 90% with a negligible one.
- **mSPRT for sequential testing** (the method behind Optimizely's Stats Engine) because
  it permits unlimited peeking with a hard validity guarantee; the cost in power at
  fixed n is measured, not hidden.
- **SRM gate blocks the readout entirely.** A sample-ratio mismatch means the data is
  untrustworthy; showing lift numbers next to a failed SRM invites people to use them.

## What I'd do differently at scale

CUPED variance reduction for faster experiments, cluster-robust variance for
session-level randomization with user-level metrics, and a proper metrics store
instead of DataFrames. These are noted in `docs/DECISIONS.md` as known limitations.
