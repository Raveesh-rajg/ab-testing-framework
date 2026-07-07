# Checkout redesign experiment — results

*Audience: product team. No stats background assumed. All numbers reproducible via
`examples/01_checkout_case_study.py` (seeded simulation).*

## Recommendation

Ship the single-button checkout, but plan around a **~+2% long-run conversion lift,
not the +3.7% headline number**. The effect is real and the risk of it being negative
is negligible, but a third of the measured lift was novelty — users reacting to the
change itself — and it faded within two weeks.

## What we tested

Hypothesis: replacing the two-step checkout confirmation with a single button raises
checkout conversion by at least 5% relative, because funnel logs showed 18% of
drop-off happening on the confirmation step. Fourteen days, 50/50 split, 560k users,
user-level randomization. The test was sized before launch: at our traffic we needed
~122k users per arm to reliably detect a 5% relative lift, and we cleared that.

## What we found

Conversion went from 5.00% (control) to 5.19% (treatment): a **+3.7% relative lift,
95% interval +1.3% to +6.0%**. The probability the treatment is genuinely better than
control is over 99.9%. Data quality checks passed — traffic split exactly as designed.

Two caveats matter more than the headline:

**The lift is decaying.** Week one averaged +5.8%; week two +1.5%. Looking at just the
final five days, the estimate is **+1.9%**. This is a classic novelty pattern: some of
the early lift came from the change being new, not better. The honest forecast for
next quarter uses the late-period number.

**We did not stop early, on purpose.** The dashboard showed a "significant" green
result as early as day 4. Our simulations show that stopping the first time the
dashboard turns green produces a false positive **24% of the time** — nearly 5x the 5%
error rate everyone believes they're getting. We committed to the sample size up front
and held to it; for teams that want to monitor continuously, the framework provides an
always-valid p-value that stays honest under daily peeking (1.2% false-positive rate
in the same simulation).

## A warning from a companion experiment

A parallel test produced a pooled result of **−23%** — apparently a disaster — while
desktop improved +9.6% and mobile improved +9.8%. Both device types got better, yet
the total got worse. The cause: the treatment received a much higher share of mobile
traffic, and mobile converts at a fifth of desktop's rate, dragging the blended
average down. The pooled number was answering "what happened to the average of a
different traffic mix," not "did the feature help."

Takeaway for reading any experiment readout: **when the segment mix differs between
arms, the pooled number can lie.** Our framework now flags this pattern automatically,
and the flag blocks the pooled read until the mix difference is explained.

## Charts

- `examples/output/daily_lift.png` — daily lift with the novelty decay visible
- `examples/output/posteriors.png` — belief distributions for each arm's true rate
- `examples/output/peeking.png` — false-positive rates: nominal vs peeking vs always-valid
