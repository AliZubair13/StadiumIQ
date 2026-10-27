import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.experiments import analyze_experiment

ui.require_artifacts()
as_of = ui.as_of()
ui.page_header("A/B Experiment: Personalized Ticket Email",
               "A separately simulated randomized experiment, analyzed by intention-to-treat.", as_of)

df = ui.experiment_frame()
if df.empty:
    st.info("No experiment data found.")
    st.stop()

res = analyze_experiment(df)
t, srm, groups = res["test"], res["srm"], res["groups"].set_index("group_name")
start, end = df["assignment_date"].iloc[0], df["experiment_end_date"].iloc[0]

st.markdown(
    f"""
**Design.** Experiment `{df['experiment_id'].iloc[0]}` randomly assigned each eligible fan (signed up before
{ui.fmt_date(start)}) once to **Control — standard ticket email** or **Treatment — personalized ticket email**.
**Primary outcome:** any ticket purchase during the fixed window {ui.fmt_date(start)} – {ui.fmt_date(end)}.
Everyone assigned is analyzed in their assigned group (intention-to-treat), whether or not they opened the email.

*This experiment is simulated independently of the purchase history used elsewhere in the app.*
"""
)

rel = "n/a (control rate is 0)" if t["relative_lift"] is None else f"{t['relative_lift']:+.1%}"
ui.kpi_row([
    ("Assigned (control / treatment)", f"{t['n_control']:,} / {t['n_treatment']:,}", f"{t['n_control'] + t['n_treatment']:,} fans total"),
    ("Conversion: control", ui.fmt_pct(t["rate_control"], 2), f"{t['conversions_control']:,} purchasers"),
    ("Conversion: treatment", ui.fmt_pct(t["rate_treatment"], 2), f"{t['conversions_treatment']:,} purchasers"),
    ("Difference", f"{t['diff_pp']:+.2f} pp", f"Relative lift {rel}"),
])
st.write("")

c1, c2 = st.columns([3, 2])
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[t["diff_pp"]], y=["Treatment − Control"], mode="markers",
        marker=dict(size=14, color=ui.SERIES["xgboost"], line=dict(color="white", width=2)),
        error_x=dict(type="data", symmetric=False, array=[(t["ci_high"] - t["diff"]) * 100],
                     arrayminus=[(t["diff"] - t["ci_low"]) * 100], color=ui.NAVY, thickness=2, width=10),
        name="Difference (95% CI)",
        hovertemplate=f"Difference {t['diff_pp']:+.2f} pp<br>95% CI {t['ci_low'] * 100:+.2f} to {t['ci_high'] * 100:+.2f} pp<extra></extra>"))
    fig.add_vline(x=0, line_color=ui.MUTED, line_dash="dot")
    fig = ui.style_fig(fig, 220, None, "Difference in conversion rate (percentage points)")
    fig.update_yaxes(rangemode="normal", showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor=ui.GRID, zeroline=False)
    fig.update_layout(hovermode="closest", showlegend=False, title="Conversion difference with 95% confidence interval")
    st.plotly_chart(fig, width="stretch")
with c2:
    st.metric("95% CI for difference", f"{t['ci_low'] * 100:+.2f} to {t['ci_high'] * 100:+.2f} pp")
    st.metric("Two-proportion z-test", f"p = {t['p_value']:.3f}", help=f"z = {t['z']:.2f}, two-sided, pooled standard error")
    srm_text = "⚠️ Mismatch detected" if srm["mismatch"] else "✔️ No mismatch"
    st.metric("Sample-ratio check", srm_text,
              help=f"Chi-square vs. 50/50 split: χ² = {srm['chi2']:.2f}, p = {srm['p_value']:.3f}; alarm if p < {srm['alpha']}")

st.markdown("#### Interpretation")
(st.warning if srm["mismatch"] else st.info)(res["interpretation"], icon="🧭")

st.markdown("#### Results by group")
tab = groups.reset_index()
tab = pd.DataFrame({
    "Group": tab["group_name"].map({"control": "Control — standard email", "treatment": "Treatment — personalized email"}),
    "Assigned": tab["assigned"].astype(int),
    "Purchasers": tab["conversions"].astype(int),
    "Conversion rate": tab["conversion_rate"],
    "Revenue": tab["revenue"],
    "Revenue per assigned fan": tab["revenue_per_assigned"],
})
st.dataframe(tab, hide_index=True, width="stretch", column_config={
    "Conversion rate": st.column_config.NumberColumn(format="percent"),
    "Revenue": st.column_config.NumberColumn(format="dollar"),
    "Revenue per assigned fan": st.column_config.NumberColumn(format="$%.2f"),
})
st.caption("Revenue is shown descriptively as a secondary metric; it is skewed and was not part of the primary test.")
ui.download_button(tab, "⬇️ Download group summary CSV", "stadiumiq_experiment_summary.csv", "dl_exp")

with st.expander("📐 Method and assumptions"):
    st.markdown(
        f"""
- **Estimate:** difference in conversion rates, treatment − control, with a 95% Wald confidence interval
  (unpooled standard error). Relative lift = difference ÷ control rate (only when the control rate > 0).
- **Test:** two-sided two-proportion z-test with pooled standard error under "no difference".
- **Assumptions:** independent randomization of individual fans; one assignment per fan; outcomes
  measured over the same fixed window for both groups; large-sample normal approximation
  (each group has ≥ 10 purchasers and non-purchasers: **{'met' if t['normal_approx_ok'] else 'NOT met'}**).
- **Sample-ratio mismatch:** chi-square goodness-of-fit against the planned 50/50 split, alarm at p < {srm['alpha']}
  (observed treatment share {srm['observed_share_treatment']:.2%}).
- **Power:** with these sample sizes, the test had about 80% power to detect a difference of roughly
  {t['mde_80_power'] * 100:.2f} pp; smaller true effects are easily missed.
- **Decision guidance:** a p-value below 0.05 alone is not a reason to ship. Consider the full confidence
  interval, the cost of personalization, and whether the effect would replicate.
"""
    )
with st.expander("⚠️ Limitations"):
    st.markdown(
        """
- The experiment, its effect size and the purchases are simulated; the result is not evidence of real impact.
- One pre-specified primary metric is tested once, at the end of the window, so no multiple-testing or
  peeking correction is applied. Repeated looks or many metrics would require one.
- Purchases in this experiment are not linked to the transaction history used by the other pages.
"""
    )
