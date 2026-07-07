"""Streamlit dashboard for expkit.

Three tabs mirroring the experiment lifecycle:
    1. Plan    — power / sample-size calculator
    2. Analyze — paste your counts, get frequentist + Bayesian readout
    3. Learn   — the peeking demo, run live

Run:  PYTHONPATH=src streamlit run dashboard/app.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from expkit import power_analysis, minimum_detectable_effect, proportion_ztest
from expkit.bayesian import BetaBinomialTest
from expkit.sequential import PeekingSimulation
from expkit.validity import srm_check

st.set_page_config(page_title="expkit — experiment toolkit", layout="wide")
st.title("expkit — A/B testing toolkit")

tab_plan, tab_analyze, tab_learn = st.tabs(["Plan", "Analyze", "Why peeking breaks p-values"])

# ---------------------------------------------------------------- Plan
with tab_plan:
    st.subheader("Sample size & runtime")
    c1, c2, c3, c4 = st.columns(4)
    baseline = c1.number_input("Baseline rate", 0.001, 0.999, 0.05, 0.005, format="%.3f")
    mde = c2.number_input("MDE (relative)", 0.005, 1.0, 0.05, 0.005, format="%.3f")
    alpha = c3.selectbox("Alpha", [0.05, 0.01, 0.10], index=0)
    power = c4.selectbox("Power", [0.80, 0.90, 0.95], index=0)
    daily = st.number_input("Expected daily units (both arms)", 100, 10_000_000, 40_000)

    r = power_analysis(baseline, mde, alpha=alpha, power=power)
    days = max(int(np.ceil(r.n_per_arm / (daily / 2))), 7)
    m1, m2, m3 = st.columns(3)
    m1.metric("n per arm", f"{r.n_per_arm:,}")
    m2.metric("total n", f"{r.total_n:,}")
    m3.metric("runtime (≥7d floor)", f"{days} days")

    st.caption(
        "Or invert it: given the traffic you have, what can you detect?"
    )
    n_avail = st.number_input("Units per arm you can actually get", 100, 100_000_000, 50_000)
    st.metric(
        "Minimum detectable effect at this n",
        f"{minimum_detectable_effect(baseline, n_avail, alpha=alpha, power=power):+.2%}",
    )

# ------------------------------------------------------------- Analyze
with tab_analyze:
    st.subheader("Results readout")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Control**")
        c_n = st.number_input("Control units", 1, None, 100_000)
        c_conv = st.number_input("Control conversions", 0, None, 5_000)
    with cc2:
        st.markdown("**Treatment**")
        t_n = st.number_input("Treatment units", 1, None, 100_000)
        t_conv = st.number_input("Treatment conversions", 0, None, 5_250)

    srm = srm_check(c_n, t_n)
    if not srm.passed:
        st.error(srm.summary())
        st.stop()
    st.success(srm.summary())

    freq = proportion_ztest(c_conv, c_n, t_conv, t_n)
    bayes = BetaBinomialTest().analyze(c_conv, c_n, t_conv, t_n)

    f_col, b_col = st.columns(2)
    with f_col:
        st.markdown("**Frequentist**")
        st.metric("Relative lift", f"{freq.relative_lift:+.2%}",
                  f"95% CI [{freq.ci_low_rel:+.2%}, {freq.ci_high_rel:+.2%}]")
        st.metric("p-value", f"{freq.p_value:.4g}")
        st.write("Significant at α=0.05" if freq.significant else "Not significant at α=0.05")
    with b_col:
        st.markdown("**Bayesian**")
        st.metric("P(treatment > control)", f"{bayes.prob_treatment_better:.1%}")
        st.metric("Expected loss if shipped", f"{bayes.expected_loss_ship:.4%}")
        st.metric("Posterior lift", f"{bayes.lift_posterior_mean:+.2%}",
                  f"95% CrI [{bayes.lift_ci_low:+.2%}, {bayes.lift_ci_high:+.2%}]")

    # posterior chart
    xs, pdf_c = BetaBinomialTest.posterior_pdf_grid(*bayes.control_posterior, points=2000)
    _, pdf_t = BetaBinomialTest.posterior_pdf_grid(*bayes.treatment_posterior, points=2000)
    lo = max(min(c_conv / c_n, t_conv / t_n) * 0.8, 0)
    hi = min(max(c_conv / c_n, t_conv / t_n) * 1.2, 1)
    mask = (xs >= lo) & (xs <= hi)
    chart_df = pd.DataFrame(
        {"rate": xs[mask], "control": pdf_c[mask], "treatment": pdf_t[mask]}
    ).set_index("rate")
    st.line_chart(chart_df)

# --------------------------------------------------------------- Learn
with tab_learn:
    st.subheader("A/A simulation: the cost of peeking")
    st.write(
        "Both arms have the SAME true conversion rate, so every 'significant' "
        "result is a false positive. We analyze each simulated experiment at "
        "several interim looks with a naive z-test vs. an always-valid "
        "p-value (mSPRT)."
    )
    l1, l2, l3 = st.columns(3)
    looks = l1.slider("Interim looks", 2, 30, 14)
    n_exp = l2.slider("Simulated experiments", 100, 1000, 300, step=100)
    base = l3.number_input("True rate (both arms)", 0.01, 0.5, 0.05)

    if st.button("Run simulation"):
        with st.spinner("Simulating..."):
            res = PeekingSimulation(
                baseline=base, n_per_arm=10_000, n_looks=looks, seed=0
            ).run(n_experiments=n_exp)
        k1, k2, k3 = st.columns(3)
        k1.metric("Nominal α", f"{res.nominal_alpha:.0%}")
        k2.metric("Naive peeking FPR", f"{res.naive_false_positive_rate:.1%}",
                  delta=f"{res.naive_false_positive_rate - res.nominal_alpha:+.1%}",
                  delta_color="inverse")
        k3.metric("mSPRT FPR", f"{res.msprt_false_positive_rate:.1%}",
                  delta=f"{res.msprt_false_positive_rate - res.nominal_alpha:+.1%}",
                  delta_color="inverse")
