# Design decisions

A record of the non-obvious choices in this framework and why they went the way they
did. Ordered roughly by how often the question comes up.

## 1. Why both frequentist and Bayesian?

They answer different questions. The p-value answers "how surprising is this data if
the treatment did nothing?" — the right question for gating a ship decision against
noise. The posterior answers "given the data, what do we believe the lift is, and what
does each decision cost us in expectation?" — the right question for actually making
the decision, especially for borderline results where "not significant" gets misread
as "no effect."

Showing both also acts as a robustness check. With flat priors and experiment-scale n
they should roughly agree (the test suite verifies P(B>A) ≈ 1 − p/2 for two-sided p),
so a large disagreement signals something structurally wrong with the analysis rather
than a philosophical difference.

## 2. Conjugate Beta-Binomial instead of PyMC/Stan

For binary conversion metrics the Beta-Binomial posterior is exact and closed-form.
Running MCMC there means approximating, slower, with sampler diagnostics to babysit,
an answer available analytically. 200k Monte Carlo draws from two Beta distributions
puts the error on P(B>A) around 0.1% and runs in milliseconds — fine for a dashboard.

What this gives up: hierarchical/multilevel models (e.g., partial pooling across many
markets), non-conjugate likelihoods (heavy-tailed revenue), and priors on derived
quantities. If those become requirements, that's when PyMC earns its complexity.

## 3. Expected loss, not a P(B>A) threshold

A rule like "ship when P(B>A) > 95%" ignores magnitude. P(B>A) = 95% with a fat left
tail (5% chance of a large regression) is a worse bet than P(B>A) = 90% where the
worst case is a rounding error. Expected loss — the average relative regression you'd
eat if you shipped and were wrong — prices magnitude in. Ship when expected loss drops
below a threshold-of-caring ε (default 0.1% relative). This follows the approach
popularized by VWO's Bayesian engine.

## 4. mSPRT for sequential testing rather than group-sequential (O'Brien-Fleming)

Group-sequential designs require choosing the number and timing of looks in advance.
That matches clinical trials, not product dashboards that anyone can refresh at any
time. The mixture SPRT (Johari, Pekelis & Walsh 2017) produces an always-valid p-value
robust to *any* peeking schedule, which matches how experiments are actually monitored.

The cost is power: at a fixed n, mSPRT needs a larger true effect to reject than a
fixed-horizon test — in our A/A simulation it rejected 1.2% of the time against a 5%
budget, i.e., it is conservative. The framework treats fixed-horizon analysis as the
primary readout and mSPRT as the tool for the monitoring period.

Tau (the mixture scale) defaults to the expected effect size scale; misspecifying tau
degrades power but never validity, which is the failure mode you want.

## 5. Pooled SE for the z-statistic, unpooled for the CI

Under H0 the two arms share a rate, so the pooled variance is the correct null
variance for the test. Under H1 (which is where a CI lives) they don't, so the CI uses
unpooled variances. Mixing these up produces small inconsistencies (e.g., a CI that
excludes 0 while p > α). This pairing matches statsmodels' `proportions_ztest`, which
the test suite pins against.

## 6. Relative-lift CI via the delta method

Stakeholders consume "+3.2% [+1.1%, +5.3%]", not absolute point differences. The delta
method gives a cheap first-order variance for the ratio (T−C)/C. It's accurate when C
is many SEs from zero — true for any conversion metric with reasonable n. For metrics
where the denominator is noisy (revenue per session on small segments), bootstrap
instead; this is flagged in the docstring rather than silently absorbed.

## 7. SRM failure blocks the whole readout

The strictest opinion in the codebase: if the sample-ratio check fails, `render()`
prints the failure and *stops*. Rationale: a mismatch means units went missing
non-randomly (redirect bugs, bot filtering, logging loss), so every downstream number
is biased in an unknown direction. Showing "for reference" lift numbers under a
warning banner guarantees someone will use them. The threshold is p < 0.001 rather
than 0.05 because at experiment scale even trivial assignment bugs produce
astronomically small p-values, and a strict threshold nearly eliminates false alarms.

## 8. Novelty detection as a heuristic, not a test

The first-half vs second-half lift comparison is deliberately a flag, not a p-value.
Formally testing "did the effect decay?" (treatment × time interaction) is possible
but invites the same misuse as everything else — dichotomizing at 0.05 and moving on.
The flag's job is to force a human to look at the daily-lift chart before projecting
the full-period average forward.

## 9. Known limitations / what production would add

- **CUPED** (using pre-experiment behavior as a covariate) typically cuts variance
  30–50%, shortening runtimes proportionally. Omitted here because it needs
  pre-period data, which the simulated examples don't model yet.
- **Randomization-unit vs analysis-unit mismatch.** Everything here assumes they
  match (user-randomized, user-level metrics). Session-level metrics under user
  randomization need cluster-robust (delta-method) variance or the CIs are too narrow.
- **Multiple testing across metrics** is handled by convention (one primary metric,
  enforced by `ExperimentDesign`) rather than by correction. With many secondary
  metrics, Benjamini-Hochberg on the secondaries would be the next step.
- **A metrics store.** Definitions live in code per-experiment; at scale they belong
  in a governed semantic layer so "conversion" means one thing everywhere.
