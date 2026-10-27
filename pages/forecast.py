import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.config import PROPENSITY_SNAPSHOT_DAYS

ui.require_artifacts()
as_of = ui.as_of()
ui.page_header("7-Day Sales Forecast", "Daily ticket sales forecast for every upcoming game, from the as-of date.", as_of)

fm = ui.report_json("forecast_metrics.json")
fc = ui.operational_forecast()
up = ui.upcoming_with_status()
selected_name = "XGBoost" if fm["selected_approach"] != "baseline" else "7-day-average baseline"

if fc.empty or up.empty:
    st.info("There are no upcoming games with open sales windows to forecast.")
    st.stop()

st.markdown(
    f"**Operational model:** {selected_name} (selected because it had the lowest WAPE on the "
    f"{fm['validation_season']} validation season). Forecast origin: end of day {ui.fmt_date(as_of)}; "
    "forecasts cover the next 7 days and stop at kickoff."
)

default_id = up.loc[up["days_until_kickoff"] == PROPENSITY_SNAPSHOT_DAYS, "game_id"]
ids = up["game_id"].tolist()
labels = dict(zip(up["game_id"], up["game_label"]))
game_id = st.selectbox("Upcoming game", ids, index=ids.index(default_id.iloc[0]) if len(default_id) else 0,
                       format_func=lambda g: labels[g])
g = up.set_index("game_id").loc[game_id]
gfc = fc[fc["game_id"] == game_id]

ui.kpi_row([
    ("Next-7-day forecast", ui.fmt_int(gfc["selected_forecast"].sum()),
     f"{len(gfc)} forecast day(s) before kickoff"),
    ("Sold to date", ui.fmt_int(g["tickets_sold"]), f"{ui.fmt_pct(g['sell_through'])} of capacity"),
    ("Remaining inventory", ui.fmt_int(g["remaining_inventory"]),
     f"Projected {ui.fmt_int(max(g['projected_remaining_at_checkpoint'], 0))} after 7 days"),
    ("Kickoff", ui.fmt_date(g["game_date"]), f"{int(g['days_until_kickoff'])} days away · {ui.status_label(g['forecast_status'])}"),
])

hist = ui.cumulative([game_id])
hist = hist[hist["sales_date"] > pd.Timestamp(as_of) - pd.Timedelta(days=28)]
fig = go.Figure()
fig.add_trace(go.Bar(x=hist["sales_date"], y=hist["tickets_sold"], name="Actual (last 28 days)",
                     marker_color=ui.SERIES["actual"], marker_line_width=0,
                     hovertemplate="%{y:,.0f} tickets<extra>Actual</extra>"))
fig.add_trace(go.Scatter(x=gfc["sales_date"], y=gfc["xgboost"], name="XGBoost forecast", mode="lines+markers",
                         line=dict(color=ui.SERIES["xgboost"], width=2.5), marker=dict(size=8),
                         hovertemplate="%{y:,.0f} tickets<extra>XGBoost</extra>"))
fig.add_trace(go.Scatter(x=gfc["sales_date"], y=gfc["baseline"], name="Baseline (recent 7-day average)",
                         mode="lines+markers", line=dict(color=ui.SERIES["baseline"], width=2, dash="dash"),
                         marker=dict(size=8, symbol="diamond"),
                         hovertemplate="%{y:,.0f} tickets<extra>Baseline</extra>"))
fig.add_vline(x=pd.Timestamp(as_of), line_dash="dot", line_color=ui.MUTED)
fig.add_annotation(x=pd.Timestamp(as_of), y=1, yref="paper", text="As-of", showarrow=False,
                   xanchor="right", font=dict(color=ui.MUTED, size=11))
fig.add_vline(x=pd.Timestamp(g["game_date"]), line_dash="dash", line_color=ui.NAVY)
fig.add_annotation(x=pd.Timestamp(g["game_date"]), y=1, yref="paper", text="Kickoff", showarrow=False,
                   xanchor="left", font=dict(color=ui.NAVY, size=11))
fig = ui.style_fig(fig, 400, "Tickets per day")
fig.update_layout(title=f"Daily sales and 7-day forecast — {g['game_label']}")
st.plotly_chart(fig, width="stretch")

st.markdown("#### Forecast table — all upcoming games")
table = fc.merge(up[["game_id", "game_label"]], on="game_id")
table = pd.DataFrame({
    "game_id": table["game_id"],
    "game": table["game_label"],
    "forecast_date": table["sales_date"].dt.strftime("%Y-%m-%d"),
    "horizon_days": table["horizon"],
    "days_until_kickoff": table["days_until_kickoff"],
    "forecast_tickets": table["selected_forecast"].round(1),
    "xgboost_tickets": table["xgboost"].round(1),
    "baseline_tickets": table["baseline"].round(1),
    "operational_model": table["selected_model"],
})
st.dataframe(table, hide_index=True, width="stretch", height=300)
ui.download_button(table, "⬇️ Download forecast CSV", f"stadiumiq_forecast_{as_of:%Y%m%d}.csv", "dl_fc")
st.caption("Forecasts are realized (inventory-constrained) ticket sales, not unconstrained demand.")

with st.expander("📏 Model evaluation (rolling-origin backtests)"):
    st.markdown(
        f"""
**Protocol.** Every 7 days through a season's sales period (an *origin*), the model is retrained on all
days whose sales are known by that origin, then forecasts days 1–7 recursively. Validation season:
**{fm['validation_season']}** ({fm['n_validation_origins']} origins) — used to choose between the baseline
and two XGBoost settings. Test season: **{fm['test_season']}** ({fm['n_test_origins']} origins,
{fm['n_test_points']:,} game-day forecasts) — the latest completed season, scored once after selection.

**WAPE** = total absolute error ÷ total actual tickets (lower is better). **MAE** = average absolute error in tickets per game-day.
"""
    )
    val = pd.DataFrame({"Approach": list(fm["validation_wape"]),
                        "Validation MAE": [fm["validation_mae"][k] for k in fm["validation_wape"]],
                        "Validation WAPE": list(fm["validation_wape"].values())})
    test = pd.DataFrame({"Approach": ["baseline", f"xgboost ({fm['best_xgb_config']})"],
                         "Test MAE": [fm["test"]["baseline"]["mae"], fm["test"]["xgboost"]["mae"]],
                         "Test WAPE": [fm["test"]["baseline"]["wape"], fm["test"]["xgboost"]["wape"]]})
    c1, c2 = st.columns(2)
    c1.dataframe(val, hide_index=True, width="stretch",
                 column_config={"Validation WAPE": st.column_config.NumberColumn(format="percent"),
                                "Validation MAE": st.column_config.NumberColumn(format="%.1f")})
    c2.dataframe(test, hide_index=True, width="stretch",
                 column_config={"Test WAPE": st.column_config.NumberColumn(format="percent"),
                                "Test MAE": st.column_config.NumberColumn(format="%.1f")})
    st.caption(f"Selected on validation: **{fm['selected_approach']}**.")

    hz = ui.report_csv("forecast_test_by_horizon.csv")
    if not hz.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hz["horizon"], y=hz["xgboost_wape"], name="XGBoost", mode="lines+markers",
                                 line=dict(color=ui.SERIES["xgboost"], width=2), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=hz["horizon"], y=hz["baseline_wape"], name="Baseline", mode="lines+markers",
                                 line=dict(color=ui.SERIES["baseline"], width=2, dash="dash"),
                                 marker=dict(size=8, symbol="diamond")))
        fig = ui.style_fig(fig, 320, "WAPE", "Forecast horizon (days ahead)")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(title=f"Test-season WAPE by horizon ({fm['test_season']})")
        st.plotly_chart(fig, width="stretch")

    bg = ui.report_csv("forecast_test_by_game.csv")
    if not bg.empty:
        names = ui.game_summary().set_index("game_id")["game_label"]
        bg = bg.assign(game=bg["game_id"].map(names))
        st.markdown("**Test-season error by game**")
        st.dataframe(bg[["game", "n_points", "actual_total", "baseline_mae", "xgboost_mae", "baseline_wape", "xgboost_wape"]],
                     hide_index=True, width="stretch", column_config={
                         "baseline_wape": st.column_config.NumberColumn("Baseline WAPE", format="percent"),
                         "xgboost_wape": st.column_config.NumberColumn("XGBoost WAPE", format="percent"),
                         "baseline_mae": st.column_config.NumberColumn("Baseline MAE", format="%.1f"),
                         "xgboost_mae": st.column_config.NumberColumn("XGBoost MAE", format="%.1f"),
                         "actual_total": st.column_config.NumberColumn("Actual tickets", format="%d"),
                         "n_points": st.column_config.NumberColumn("Forecast points")})

with st.expander("⚙️ How the forecast works"):
    imp = pd.Series(fm["feature_importance"]).head(8)
    st.markdown(
        """
- **Target:** tickets sold per game per day (0–90 days before kickoff, including zero-sale days).
- **Inputs:** days until kickoff, days since sales opened, day of week, lagged sales (1, 2, 7 days),
  rolling 7- and 28-day averages, tickets sold so far and remaining inventory, opponent popularity,
  rivalry and weekend flags, and the base ticket price set before sales open.
- **Leakage safeguards:** lags and rolling means are computed within each game and shifted so a day's
  own sales never enter its inputs. At the origin, all later actual sales are discarded; each forecast
  day's prediction is fed back as the next day's lag and subtracted from inventory (recursive forecasting).
- **Constraints:** Poisson objective (non-negative counts); each day is clipped to the remaining
  inventory; the calendar ends at kickoff, so no forecast is produced after the game.
- **Baseline:** the game's average over its last 7 observed days, held flat and capped at remaining inventory.
"""
    )
    st.markdown("**Most-used inputs (XGBoost gain-based importance):** " +
                ", ".join(f"`{k}` ({v:.0%})" for k, v in imp.items()))

with st.expander("⚠️ Limitations"):
    st.markdown(
        """
- Data is synthetic: accuracy reflects how learnable the simulator is, not real-world performance.
- Observed sales are capped by inventory; the model forecasts realized sales, not unconstrained demand.
- Price is an input, but the club set prices with knowledge of demand (higher for popular games), so the
  model's response to price is **not** a causal price elasticity and must not be used to set prices.
- Recursive forecasts compound errors across the 7 days; errors grow with horizon.
- Games whose sales window opens during the forecast week are not forecast (no history yet).
"""
    )
