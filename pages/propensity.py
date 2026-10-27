import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.config import PROPENSITY_SNAPSHOT_DAYS
from src.features import NO_HISTORY_DAYS

ui.require_artifacts()
as_of = ui.as_of()
ui.page_header(
    "Fan Purchase Propensity",
    f"At {PROPENSITY_SNAPSHOT_DAYS} days before kickoff, which fans who have not bought yet are most likely to buy before kickoff?",
    as_of,
)

pm = ui.report_json("propensity_metrics.json")
live = ui.live_propensity()
sel = pm["selected"]
summary = ui.game_summary().set_index("game_id")

st.markdown(
    f"""
**How to read a score.** Each eligible fan gets a **propensity score** between 0 and 1. Higher means the
model ranks the fan as more likely to buy, based on their purchase history, recent engagement and the
game. Use scores to **rank and prioritize**; they are not guaranteed probabilities (see the calibration
check below). A high score does **not** mean outreach will change the fan's behavior — many high-score
fans would buy anyway.
"""
)

# --- Live list ---------------------------------------------------------------
st.markdown("### Outreach list for the next snapshot game")
if live.empty or not pm.get("live_game_id"):
    st.info(f"No upcoming game kicks off exactly {PROPENSITY_SNAPSHOT_DAYS} days after the as-of date, "
            "or it has no eligible fans (e.g. sold out), so there is no live list.")
else:
    gid = pm["live_game_id"]
    g = summary.loc[gid]
    top = live[live["top_10pct"]]
    st.markdown(f"**{g['game_label']}** · snapshot date {ui.fmt_date(as_of)} "
                f"({PROPENSITY_SNAPSHOT_DAYS} days before kickoff) · model: {pm['selected_family']}")
    ui.kpi_row([
        ("Eligible fans", ui.fmt_int(len(live)), "Signed up, not yet bought for this game"),
        ("Top 10% list", ui.fmt_int(len(top)), f"Scores ≥ {top['propensity_score'].min():.2f}"),
        ("Avg score, top 10%", f"{top['propensity_score'].mean():.2f}", f"vs {live['propensity_score'].mean():.2f} for all eligible"),
        ("Test-season lift", f"{pm['test'][sel]['lift_top10']:.1f}x",
         f"Top-10% buy rate {ui.fmt_pct(pm['test'][sel]['precision_top10'])} vs {ui.fmt_pct(pm['test'][sel]['prevalence'])} overall"),
    ])

    c1, c2 = st.columns([1, 1])
    fig = go.Figure(go.Histogram(x=live["propensity_score"], nbinsx=40, marker_color=ui.SERIES["actual"],
                                 marker_line_color="white", marker_line_width=1, name="Eligible fans",
                                 hovertemplate="Score %{x}<br>%{y:,} fans<extra></extra>"))
    fig.add_vline(x=top["propensity_score"].min(), line_dash="dash", line_color=ui.SERIES["xgboost"])
    fig.add_annotation(x=top["propensity_score"].min(), y=1, yref="paper", text="Top-10% cutoff",
                       showarrow=False, xanchor="left", font=dict(color=ui.NAVY, size=11))
    fig = ui.style_fig(fig, 300, "Fans", "Propensity score")
    fig.update_layout(hovermode="closest", showlegend=False, title="Score distribution")
    c1.plotly_chart(fig, width="stretch")
    seg = live.groupby("fan_segment").agg(fans=("customer_id", "size"), in_top=("top_10pct", "sum")).reset_index()
    seg["share_in_top"] = seg["in_top"] / seg["fans"]
    seg = seg.sort_values("share_in_top")
    fig = go.Figure(go.Bar(y=seg["fan_segment"], x=seg["share_in_top"], orientation="h",
                           marker_color=ui.SERIES["xgboost"], marker_line_width=0,
                           customdata=seg[["in_top", "fans"]],
                           hovertemplate="%{y}: %{x:.1%} (%{customdata[0]:,} of %{customdata[1]:,})<extra></extra>"))
    fig = ui.style_fig(fig, 300, None, "Share of segment in top-10% list")
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(hovermode="closest", title="Who makes the list, by fan segment")
    c2.plotly_chart(fig, width="stretch")

    f1, f2, f3 = st.columns([1, 1, 2])
    only_top = f1.toggle("Top 10% only", value=True)
    segs = f2.multiselect("Fan segment", sorted(live["fan_segment"].unique()))
    channels = f3.multiselect("Suggested contact channel", sorted(live["suggested_contact_channel"].unique()))
    view = live[live["top_10pct"]] if only_top else live
    if segs:
        view = view[view["fan_segment"].isin(segs)]
    if channels:
        view = view[view["suggested_contact_channel"].isin(channels)]
    out = pd.DataFrame({
        "rank": view["rank"],
        "customer_id": view["customer_id"],
        "propensity_score": view["propensity_score"].round(3),
        "fan_segment": view["fan_segment"],
        "suggested_contact_channel": view["suggested_contact_channel"],
        "previous_purchases": view["prev_purchase_count"].astype(int),
        "previous_spend": view["prev_total_spend"].round(2),
        "avg_tickets_per_order": view["avg_ticket_quantity"].round(2),
        "days_since_last_purchase": view["days_since_last_purchase"].where(view["never_purchased"] == 0),
        "bought_previous_home_game": view["bought_previous_home_game"].astype(bool),
        "engagements_last_30d": view["engagement_30d"].astype(int),
        "tenure_days": view["tenure_days"].astype(int),
        "distance_band": view["distance_band"],
    })
    if out.empty:
        st.info("No fans match these filters.")
    else:
        st.dataframe(out, hide_index=True, width="stretch", height=380, column_config={
            "propensity_score": st.column_config.ProgressColumn("Propensity score", format="%.3f", min_value=0, max_value=1),
            "previous_spend": st.column_config.NumberColumn("Previous spend", format="dollar"),
            "days_since_last_purchase": st.column_config.NumberColumn("Days since last purchase", format="%d",
                                                                      help="Blank = never purchased"),
            "rank": st.column_config.NumberColumn("Rank"),
            "customer_id": "Customer ID",
            "fan_segment": "Segment",
            "previous_purchases": st.column_config.NumberColumn("Prior orders"),
            "avg_tickets_per_order": st.column_config.NumberColumn("Avg tickets/order", format="%.2f"),
            "bought_previous_home_game": st.column_config.CheckboxColumn("Bought last home game"),
            "engagements_last_30d": st.column_config.NumberColumn("Engagements (30d)"),
            "tenure_days": st.column_config.NumberColumn("Tenure (days)"),
            "distance_band": "Distance",
            "suggested_contact_channel": st.column_config.TextColumn(
                "Suggested channel*", help="Fan's historical preferred channel — not a proven optimal channel."),
        })
    st.caption(f"{len(out):,} fans shown. *Suggested channel = the fan's historical preferred channel from their profile; "
               "it has not been tested as the most effective channel.")
    ui.download_button(out, "⬇️ Download list as CSV", f"stadiumiq_propensity_{gid}.csv", "dl_prop")

# --- Model comparison -------------------------------------------------------------
st.markdown("### Model comparison")
counts = pm["split_counts"]
st.caption(
    f"Chronological season split — train: {counts['train']['games']} games / {counts['train']['rows']:,} snapshots "
    f"(2020–2022), validation: {counts['validation']['rows']:,} (2023), test: {counts['test']['rows']:,} (2024). "
    "Outcome windows are fully observed for every snapshot used."
)


def metric_frame(block: dict) -> pd.DataFrame:
    rows = []
    for name, m in block.items():
        rows.append({"Model": f"{m['family']} ({name.split('_', 1)[1]})", "PR-AUC (avg precision)": m["average_precision"],
                     "ROC-AUC": m["roc_auc"], "Precision @ top 10%": m["precision_top10"],
                     "Lift @ top 10%": m["lift_top10"], "Base rate": m["prevalence"], "Brier": m["brier"]})
    return pd.DataFrame(rows)


cfg = {"PR-AUC (avg precision)": st.column_config.NumberColumn(format="%.3f"),
       "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
       "Precision @ top 10%": st.column_config.NumberColumn(format="percent"),
       "Lift @ top 10%": st.column_config.NumberColumn(format="%.2fx"),
       "Base rate": st.column_config.NumberColumn(format="percent"),
       "Brier": st.column_config.NumberColumn(format="%.3f")}
st.markdown("**Validation (2023) — used for selection**")
st.dataframe(metric_frame(pm["validation"]), hide_index=True, width="stretch", column_config=cfg)
st.markdown("**Test (2024) — scored once, best of each family**")
st.dataframe(metric_frame(pm["test"]), hide_index=True, width="stretch", column_config=cfg)
st.markdown(
    f"Selected: **{pm['selected_family']}** (`{sel}`) — highest validation PR-AUC. "
    "PR-AUC summarizes how well buyers are ranked above non-buyers when buyers are the minority; "
    "lift @ top 10% says how many times more often the top-10% list buys compared with the average eligible fan."
)

with st.expander("🔢 Confusion matrix at the outreach threshold (test season)"):
    cm = pm["test"][sel]["confusion_matrix"]
    st.markdown(f"Threshold rule: **{pm['threshold_rule']}** — matching how the outreach list is built.")
    st.dataframe(pd.DataFrame({"": ["Actually bought", "Did not buy"],
                               "Flagged (top 10%)": [cm["tp"], cm["fp"]],
                               "Not flagged": [cm["fn"], cm["tn"]]}), hide_index=True)
    st.caption(f"Recall at this threshold: {ui.fmt_pct(pm['test'][sel]['recall_top10'])} of eventual buyers were on the top-10% lists.")

with st.expander("📐 Calibration check (test season)"):
    cal = ui.report_csv("propensity_test_calibration.csv")
    if not cal.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 0.8], y=[0, 0.8], mode="lines", line=dict(color="#C3CAD5", dash="dot", width=1),
                                 name="Perfect calibration", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=cal["mean_score"], y=cal["observed_rate"], mode="lines+markers",
                                 line=dict(color=ui.SERIES["actual"], width=2), marker=dict(size=8),
                                 name="Score decile", hovertemplate="Mean score %{x:.2f}<br>Observed rate %{y:.1%}<extra></extra>"))
        fig = ui.style_fig(fig, 320, "Observed purchase rate", "Average propensity score (decile)")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, width="stretch")
        st.caption("Points below the diagonal mean the scores overstate the actual purchase rate. Because the scores "
                   "are not well calibrated, this app calls them propensity scores and uses them only for ranking.")

with st.expander("🧠 Features and what drives the ranking"):
    imp = pm["importance"]
    st.markdown(
        f"""
All features are computed **as of the snapshot date** (14 days before kickoff) using only
transactions and engagement dated on or before it: previous purchase count and spend, average tickets
per order, days since last purchase, purchases in the past 365 days, whether the fan bought the previous
home game, prior purchases for similar games (same rivalry flag and opponent-popularity tier), tenure,
engagement and ticket-page views in the preceding 30 days, days since last engagement, profile fields
(preferred channel, distance band, fan segment), the game's characteristics and how full it already is.
Fans who never purchased or never engaged get a documented fill value of {NO_HISTORY_DAYS:,} days plus a
`never_purchased` flag. Customer IDs are never model inputs.

Top inputs ({imp['kind']}):
"""
    )
    st.dataframe(pd.DataFrame({"feature": list(imp["values"]), "value": list(imp["values"].values())}),
                 hide_index=True, width="stretch")

with st.expander("⚠️ Limitations"):
    st.markdown(
        """
- Synthetic data: the relationships (e.g. engagement before purchase) were built into the simulator.
- Propensity is **not uplift**: it predicts who buys, not who buys *because of* outreach. Measuring that
  requires a randomized holdout (see the A/B Experiment page).
- Labels are inventory-constrained: fans who wanted tickets to sold-out games appear as non-buyers.
- Scores are not calibrated probabilities; use them for ranking.
"""
    )
