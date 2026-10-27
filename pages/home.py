import streamlit as st

from src import ui
from src.config import CURRENT_SEASON, TEAM_NAME, VENUE_NAME

ui.require_artifacts()
as_of = ui.as_of()
ui.page_header(
    "StadiumIQ — Ticket Demand & Fan Insights",
    f"Analytics workspace for the ticket operations team of the fictional {TEAM_NAME} ({VENUE_NAME}).",
    as_of,
)

summary = ui.game_summary()
season = summary[summary["season"] == CURRENT_SEASON]
upcoming = ui.upcoming_with_status()
fm = ui.report_json("forecast_metrics.json")
pm = ui.report_json("propensity_metrics.json")
sel = pm["selected"]

ui.kpi_row([
    (f"{CURRENT_SEASON} tickets sold", ui.fmt_int(season["tickets_sold"].sum()),
     f"{ui.fmt_pct(season['tickets_sold'].sum() / season['stadium_capacity'].sum())} of season capacity"),
    (f"{CURRENT_SEASON} revenue", ui.fmt_money(season["revenue"].sum()), "Season to date"),
    ("Upcoming games", str(len(upcoming)),
     f"{(upcoming['forecast_status'] == 'At risk').sum()} flagged at risk"),
    ("Forecast test WAPE", ui.fmt_pct(fm["test"]["xgboost"]["wape"]),
     f"vs {ui.fmt_pct(fm['test']['baseline']['wape'])} for 7-day-average baseline"),
])

st.markdown("### What this workspace answers")
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
- **📊 Sales Overview** — How are tickets and revenue tracking, and how does each game's pace
  compare with similar completed games at the same lead time?
- **📈 7-Day Forecast** — How many tickets will each upcoming game sell over the next seven days?
- **🎯 Fan Propensity** — Which eligible fans look most likely to buy for the game 14 days out?
"""
    )
with c2:
    st.markdown(
        """
- **🧭 Recommendations** — Which games are behind their seven-day checkpoint, close to capacity,
  or ready for an outreach list?
- **🧪 A/B Experiment** — Did the personalized email convert better than the standard email?
- **📚 Data & Methods** — Schema, simulation assumptions, leakage safeguards and limitations.
"""
    )

st.markdown("### Headline results from this run")
st.markdown(
    f"""
- **Forecasting:** on rolling 7-day backtests of the untouched {fm['test_season']} season, XGBoost had
  MAE {fm['test']['xgboost']['mae']:.1f} tickets/day (WAPE {ui.fmt_pct(fm['test']['xgboost']['wape'])}) vs
  MAE {fm['test']['baseline']['mae']:.1f} (WAPE {ui.fmt_pct(fm['test']['baseline']['wape'])}) for the
  recent 7-day-average baseline. The approach used operationally was selected on the {fm['validation_season']} validation season.
- **Propensity:** the selected {pm['selected_family']} ranks fans so that the top 10% per game bought at
  {ui.fmt_pct(pm['test'][sel]['precision_top10'])} vs a {ui.fmt_pct(pm['test'][sel]['prevalence'])} base rate
  (lift {pm['test'][sel]['lift_top10']:.1f}x) on the {fm['test_season']} test season.
"""
)
st.info(
    "Every number in this app comes from a **synthetic** dataset generated with a fixed seed. The models "
    "recover relationships the simulator was built with; they say nothing about real-world performance, "
    "and no revenue improvement has been demonstrated.",
    icon="ℹ️",
)
