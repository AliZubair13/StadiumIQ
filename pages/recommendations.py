import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.config import FORECAST_HORIZON, NEAR_CAPACITY_SHARE
from src.recommendations import build_recommendations, typical_share_curve

ui.require_artifacts()
as_of = ui.as_of()
ui.page_header("Recommendations", "Transparent, rule-based actions derived from sales pace, forecasts and propensity scores.", as_of)

up = ui.upcoming_with_status()
live = ui.live_propensity()
recs = build_recommendations(up, live, as_of)

PRIORITY_COLORS = {"High": ("#FEE4E2", "#B42318"), "Medium": ("#FEF0C7", "#B54708"), "Low": ("#E6F4EA", "#067647")}
counts = pd.Series([r["priority"] for r in recs]).value_counts()
ui.kpi_row([
    ("Recommendations", str(len(recs)), f"for {len(up)} upcoming games"),
    ("High priority", str(int(counts.get("High", 0))), "Large pace shortfall (>10%)"),
    ("Games at risk", str(int((up["forecast_status"] == "At risk").sum())), "Projected below 7-day checkpoint"),
    ("Near capacity / sold out", str(int(up["forecast_status"].isin(["Near capacity", "Sold out"]).sum())),
     f"≤ {NEAR_CAPACITY_SHARE:.0%} inventory projected"),
])
st.write("")

if not recs:
    st.success("No games are behind pace or near capacity, and no outreach list is available.")
for r in recs:
    bg, fg = PRIORITY_COLORS[r["priority"]]
    st.markdown(
        f"""<div class="siq-rec">
<span class="tag" style="background:{bg};color:{fg}">{r['priority'].upper()}</span>
<span class="tag" style="background:#EEF2F7;color:{ui.NAVY}">{r['category']}</span>
<b>{r['game']}</b>
<div style="margin-top:.5rem"><b>Trigger:</b> {r['trigger_metric']}</div>
<div><b>Threshold:</b> {r['threshold']}</div>
<div><b>Why:</b> {r['explanation']}</div>
<div><b>Suggested action:</b> {r['action']}</div>
</div>""",
        unsafe_allow_html=True,
    )
if recs:
    ui.download_button(pd.DataFrame(recs), "⬇️ Download recommendations CSV", f"stadiumiq_recommendations_{as_of:%Y%m%d}.csv", "dl_recs")

st.markdown("### How the seven-day checkpoint target is derived")
st.markdown(
    f"""
A 7-day forecast only covers the next week, so it is compared with **where sales should be in 7 days**,
never with the final game target:

1. **Checkpoint date** = as-of date + {FORECAST_HORIZON} days (or kickoff, if sooner).
2. **Typical share** = median, over all completed games, of the share of final sales already sold at the
   checkpoint's days-until-kickoff.
3. **Checkpoint target** = final sales target × typical share.
4. **Projected at checkpoint** = tickets sold to date + next-7-day forecast (capped at capacity).
5. **At risk** if projected < checkpoint target; **Near capacity** if ≤ {NEAR_CAPACITY_SHARE:.0%} of seats are
   projected to remain.
"""
)
curve = typical_share_curve(ui.pace_curves()).reset_index()
fig = go.Figure(go.Scatter(x=curve["days_until_kickoff"], y=curve["share_of_final"], mode="lines",
                           line=dict(color=ui.SERIES["actual"], width=2), name="Median share of final sales",
                           hovertemplate="%{x} days out: %{y:.1%} of final sales<extra></extra>"))
cp = up.dropna(subset=["typical_share_at_checkpoint"])
fig.add_trace(go.Scatter(x=cp["checkpoint_lead_days"], y=cp["typical_share_at_checkpoint"], mode="markers",
                         marker=dict(size=10, color=ui.SERIES["xgboost"], line=dict(color="white", width=2)),
                         name="Upcoming game checkpoints", customdata=cp[["game_label"]],
                         hovertemplate="%{customdata[0]}<br>Checkpoint %{x} days out: %{y:.1%}<extra></extra>"))
fig = ui.style_fig(fig, 340, "Share of final sales", "Days until kickoff")
fig.update_xaxes(autorange="reversed")
fig.update_yaxes(tickformat=".0%")
fig.update_layout(hovermode="closest")
st.plotly_chart(fig, width="stretch")

table = pd.DataFrame({
    "Game": up["game_label"],
    "Checkpoint": up["checkpoint_date"].dt.strftime("%b %d"),
    "Days out at checkpoint": up["checkpoint_lead_days"],
    "Typical share": up["typical_share_at_checkpoint"],
    "Final target": up["sales_target"],
    "Checkpoint target": up["checkpoint_target"],
    "Sold to date": up["tickets_sold"],
    "Next 7d forecast": up["forecast_next_7d"].round(0),
    "Projected": up["projected_at_checkpoint"].round(0),
    "Gap": up["checkpoint_gap"].round(0),
    "Status": up["forecast_status"].map(ui.status_label),
})
st.dataframe(table, hide_index=True, width="stretch", column_config={
    "Typical share": st.column_config.NumberColumn(format="percent"),
    "Checkpoint target": st.column_config.NumberColumn(format="%d"),
    "Next 7d forecast": st.column_config.NumberColumn(format="%d"),
    "Projected": st.column_config.NumberColumn(format="%d"),
    "Gap": st.column_config.NumberColumn(format="%+d"),
})

with st.expander("⚠️ What these rules do not do"):
    st.markdown(
        """
- They do **not** optimize ticket prices or estimate causal price elasticity.
- A high propensity score does **not** identify the fans most *influenced* by outreach (that is uplift,
  which needs a randomized holdout to measure).
- The historical share curve is built from realized, inventory-capped sales; sell-outs make some games
  look as if they finished selling earlier than demand alone would imply.
- Thresholds (7-day checkpoint, 10% shortfall for "High", 5% near-capacity) are documented business
  rules, not optimized values.
"""
    )
