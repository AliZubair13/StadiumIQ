import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.config import CURRENT_SEASON

ui.require_artifacts()
as_of = ui.as_of()
ui.page_header("Sales Overview", "Ticket sales, revenue and pacing by season and game.", as_of)

summary = ui.game_summary()
seasons = sorted(summary["season"].unique(), reverse=True)
f1, f2 = st.columns([1, 3])
season = f1.selectbox("Season", seasons, index=seasons.index(CURRENT_SEASON),
                      format_func=lambda s: f"{s} (current)" if s == CURRENT_SEASON else str(s))
games = summary[summary["season"] == season].reset_index(drop=True)
options = ["All games"] + games["game_id"].tolist()
labels = dict(zip(games["game_id"], games["game_label"] + " · " + games["status"]))
game_id = f2.selectbox("Game", options, format_func=lambda g: "All games in season" if g == "All games" else labels[g])

scope = games if game_id == "All games" else games[games["game_id"] == game_id]
sold, cap = scope["tickets_sold"].sum(), scope["stadium_capacity"].sum()
target = scope["sales_target"].sum()
ui.kpi_row([
    ("Tickets sold", ui.fmt_int(sold), f"{ui.fmt_pct(sold / target)} of sales target ({ui.fmt_int(target)})"),
    ("Revenue", ui.fmt_money(scope["revenue"].sum()),
     f"Avg {ui.fmt_money(scope['revenue'].sum() / sold if sold else float('nan'), 2)} per ticket"),
    ("Sell-through", ui.fmt_pct(sold / cap), f"Capacity {ui.fmt_int(cap)} seats"),
    ("Remaining inventory", ui.fmt_int(scope["remaining_inventory"].sum()),
     f"{(scope['remaining_inventory'] == 0).sum()} of {len(scope)} game(s) sold out"),
])
st.caption(f"Observed through {ui.fmt_date(as_of)}. Upcoming games show sales to date, not final sales.")

# --- Daily and cumulative sales -------------------------------------------------
st.markdown("#### Daily and cumulative sales")
if game_id == "All games":
    series = ui.season_daily(season).rename(columns={"tickets_sold": "daily"})
    series = series.rename(columns={"cumulative_tickets": "cumulative"})
else:
    series = ui.cumulative([game_id]).rename(columns={"tickets_sold": "daily", "cumulative_tickets": "cumulative"})

if series.empty:
    st.info("No sales observed yet for this selection.")
else:
    c1, c2 = st.columns(2)
    fig = go.Figure(go.Bar(x=series["sales_date"], y=series["daily"], marker_color=ui.SERIES["actual"],
                           marker_line_width=0, name="Tickets sold",
                           hovertemplate="%{x|%b %d, %Y}: %{y:,} tickets<extra></extra>"))
    c1.plotly_chart(ui.style_fig(fig, 320, "Tickets per day").update_layout(title="Daily tickets sold"),
                    width="stretch")
    fig = go.Figure(go.Scatter(x=series["sales_date"], y=series["cumulative"], mode="lines",
                               line=dict(color=ui.SERIES["actual"], width=2), name="Cumulative",
                               hovertemplate="%{x|%b %d, %Y}: %{y:,} tickets<extra></extra>"))
    if game_id != "All games":
        g = scope.iloc[0]
        fig.add_hline(y=g["sales_target"], line_dash="dot", line_color=ui.MUTED,
                      annotation_text=f"Target {g['sales_target']:,}", annotation_position="top left")
        fig.add_hline(y=g["stadium_capacity"], line_dash="dash", line_color=ui.NAVY,
                      annotation_text=f"Capacity {g['stadium_capacity']:,}", annotation_position="bottom left")
    c2.plotly_chart(ui.style_fig(fig, 320, "Cumulative tickets").update_layout(title="Cumulative tickets sold"),
                    width="stretch")

if game_id == "All games":
    st.markdown("#### Sell-through by game")
    short = games["game_date"].dt.strftime("%b %d") + "<br>" + games["opponent"].str.replace(" ", "<br>", n=1)
    fig = go.Figure(go.Bar(
        x=short, y=games["sell_through"], marker_color=ui.SERIES["actual"], marker_line_width=0,
        name="Sell-through", customdata=games[["tickets_sold", "status", "game_label"]],
        hovertemplate="%{customdata[2]}<br>%{y:.1%} sold (%{customdata[0]:,} tickets)<br>%{customdata[1]}<extra></extra>"))
    fig.add_trace(go.Scatter(x=short, y=games["sales_target"] / games["stadium_capacity"],
                             mode="markers", marker=dict(symbol="line-ew-open", size=26, color=ui.NAVY, line_width=2),
                             name="Sales target", hovertemplate="Target: %{y:.1%}<extra></extra>"))
    fig = ui.style_fig(fig, 360, "Share of capacity")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.08])
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, width="stretch")

# --- Pace vs comparable games --------------------------------------------------
st.markdown("#### Sales pace by days until kickoff")
upcoming_all = summary[summary["status"] == "Upcoming"]
pace_game = game_id if game_id != "All games" else (
    upcoming_all["game_id"].iloc[0] if season == CURRENT_SEASON and len(upcoming_all) else games["game_id"].iloc[-1])
pg = summary.set_index("game_id").loc[pace_game]
if game_id == "All games":
    st.caption(f"Showing {pg['game_label']} — pick a game above to change it.")
lead = int(max(pg["days_until_kickoff"], 0))
comps = ui.comparables(pace_game, lead)
pace = ui.pace_curves()
this = ui.cumulative([pace_game])

fig = go.Figure()
comp_curves = pace[pace["game_id"].isin(comps["game_id"])]
for i, (gid, part) in enumerate(comp_curves.groupby("game_id")):
    fig.add_trace(go.Scatter(x=part["days_until_kickoff"], y=part["cum_sell_through"], mode="lines",
                             line=dict(color="#C3CAD5", width=1), name="Comparable completed games",
                             legendgroup="comp", showlegend=i == 0, hoverinfo="skip"))
if not comp_curves.empty:
    med = comp_curves.groupby("days_until_kickoff")["cum_sell_through"].median().reset_index()
    fig.add_trace(go.Scatter(x=med["days_until_kickoff"], y=med["cum_sell_through"], mode="lines",
                             line=dict(color=ui.SERIES["comparable"], width=2, dash="dash"),
                             name="Comparable median",
                             hovertemplate="%{x} days out: %{y:.1%}<extra>Comparable median</extra>"))
if not this.empty:
    fig.add_trace(go.Scatter(x=this["days_until_kickoff"], y=this["cumulative_sell_through"], mode="lines",
                             line=dict(color=ui.SERIES["actual"], width=2.5), name=pg["game_label"],
                             hovertemplate="%{x} days out: %{y:.1%}<extra>This game</extra>"))
fig = ui.style_fig(fig, 380, "Cumulative sell-through", "Days until kickoff")
fig.update_xaxes(autorange="reversed")
fig.update_yaxes(tickformat=".0%")
c1, c2 = st.columns([2, 1])
c1.plotly_chart(fig, width="stretch")

with c2:
    at_lead_this = this.loc[this["days_until_kickoff"] >= lead, "cumulative_sell_through"]
    this_val = at_lead_this.iloc[-1] if len(at_lead_this) else float("nan")
    st.markdown(f"**At {lead} days before kickoff**")
    if comps.empty:
        st.info("No completed games match this game's rivalry status and opponent popularity (±0.15).")
    else:
        med_val = comps["sell_through_at_lead"].median()
        st.metric("This game", ui.fmt_pct(this_val),
                  delta=f"{(this_val - med_val) * 100:+.1f} pts vs comparable median")
        st.metric("Comparable median", ui.fmt_pct(med_val),
                  help="Median cumulative sell-through of comparable completed games at the same lead time.")
        st.caption(f"{len(comps)} comparable completed games: same rivalry flag, opponent popularity within ±0.15. "
                   f"Range {ui.fmt_pct(comps['sell_through_at_lead'].min())}–{ui.fmt_pct(comps['sell_through_at_lead'].max())}.")
if not comps.empty:
    with st.expander(f"Comparable games ({len(comps)})"):
        show = comps.assign(game_date=pd.to_datetime(comps["game_date"]).dt.strftime("%b %d, %Y"))
        st.dataframe(
            show[["season", "game_date", "opponent", "opponent_popularity", "sell_through_at_lead", "final_sell_through"]],
            hide_index=True, width="stretch",
            column_config={
                "sell_through_at_lead": st.column_config.ProgressColumn(f"Sell-through at {lead}d", format="percent", min_value=0, max_value=1),
                "final_sell_through": st.column_config.ProgressColumn("Final sell-through", format="percent", min_value=0, max_value=1),
                "opponent_popularity": st.column_config.NumberColumn("Opp. popularity", format="%.2f"),
            })

# --- Upcoming games ------------------------------------------------------------
st.markdown("#### Upcoming games")
up = ui.upcoming_with_status()
if up.empty:
    st.info("No upcoming games after the as-of date.")
else:
    table = pd.DataFrame({
        "Game": up["game_label"],
        "Days out": up["days_until_kickoff"],
        "Sold": up["tickets_sold"],
        "Remaining": up["remaining_inventory"],
        "Sell-through": up["sell_through"],
        "Final target": up["sales_target"],
        "Next 7d forecast": up["forecast_next_7d"].round(0),
        "Checkpoint target": up["checkpoint_target"],
        "Projected at checkpoint": up["projected_at_checkpoint"].round(0),
        "Status": up["forecast_status"].map(ui.status_label),
    })
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Sell-through": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Next 7d forecast": st.column_config.NumberColumn(format="%d"),
        "Projected at checkpoint": st.column_config.NumberColumn(format="%d"),
        "Checkpoint target": st.column_config.NumberColumn(format="%d"),
    })
    st.caption("Checkpoint = 7 days after the as-of date (or kickoff, if sooner). Checkpoint target = final target × the "
               "typical share of final sales completed games had reached at that lead time. See Recommendations.")
