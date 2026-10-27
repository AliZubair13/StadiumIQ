"""Transparent, rule-based recommendations.

Seven-day checkpoint target
---------------------------
A 7-day forecast covers only the next 7 days, so it is never compared with the
final game target directly. Instead:

    checkpoint date        = min(as-of + 7 days, kickoff)
    checkpoint lead time   = days until kickoff at the checkpoint date
    typical share by lead  = median, across completed games, of
                             (cumulative tickets at that lead time / final tickets)
    checkpoint target      = sales_target x typical share at checkpoint lead
    projected at checkpoint= tickets sold to date + 7-day forecast (to kickoff)

A game is "At risk" when projected < checkpoint target. Because sell-outs cap
final sales, the historical share curve slightly overstates early pacing for
sold-out games; this is documented as a limitation.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.config import FORECAST_HORIZON, NEAR_CAPACITY_SHARE, TOP_FRACTION


def typical_share_curve(pace: pd.DataFrame) -> pd.Series:
    """Median cumulative share of final sales by days-until-kickoff (0..90)."""
    curve = pace.dropna(subset=["share_of_final"]).groupby("days_until_kickoff")["share_of_final"].median()
    return curve.sort_index()


def checkpoint_table(upcoming: pd.DataFrame, forecast: pd.DataFrame, curve: pd.Series, as_of: date) -> pd.DataFrame:
    """Add 7-day forecast, checkpoint target and pace status for upcoming games."""
    df = upcoming.copy()
    fc7 = forecast.groupby("game_id")["selected_forecast"].sum()
    df["forecast_next_7d"] = df["game_id"].map(fc7).fillna(0.0)
    df["checkpoint_date"] = [
        min(pd.Timestamp(as_of) + pd.Timedelta(days=FORECAST_HORIZON), gd) for gd in df["game_date"]
    ]
    df["checkpoint_lead_days"] = (df["game_date"] - df["checkpoint_date"]).dt.days
    df["typical_share_at_checkpoint"] = df["checkpoint_lead_days"].map(curve)
    df["checkpoint_target"] = np.round(df["sales_target"] * df["typical_share_at_checkpoint"])
    df["projected_at_checkpoint"] = np.minimum(df["tickets_sold"] + df["forecast_next_7d"], df["stadium_capacity"])
    df["checkpoint_gap"] = df["projected_at_checkpoint"] - df["checkpoint_target"]
    df["projected_remaining_at_checkpoint"] = df["stadium_capacity"] - df["projected_at_checkpoint"]

    def status(r) -> str:
        if r["remaining_inventory"] <= 0:
            return "Sold out"
        if r["projected_remaining_at_checkpoint"] <= NEAR_CAPACITY_SHARE * r["stadium_capacity"]:
            return "Near capacity"
        if r["projected_at_checkpoint"] < r["checkpoint_target"]:
            return "At risk"
        return "On pace"

    df["forecast_status"] = df.apply(status, axis=1)
    return df


def build_recommendations(
    checkpoints: pd.DataFrame, propensity_live: pd.DataFrame | None, as_of: date
) -> list[dict]:
    recs: list[dict] = []
    for r in checkpoints.sort_values("game_date").itertuples(index=False):
        label = f"{r.game_date:%b %d} vs {r.opponent}"
        if r.forecast_status == "At risk":
            shortfall = r.checkpoint_target - r.projected_at_checkpoint
            recs.append(
                {
                    "priority": "High" if shortfall / r.checkpoint_target > 0.10 else "Medium",
                    "category": "Sales pace",
                    "game": label,
                    "game_id": r.game_id,
                    "trigger_metric": f"Projected tickets by {r.checkpoint_date:%b %d}: {r.projected_at_checkpoint:,.0f} "
                    f"({r.tickets_sold:,.0f} sold + {r.forecast_next_7d:,.0f} forecast)",
                    "threshold": f"Checkpoint target {r.checkpoint_target:,.0f} = sales target {r.sales_target:,} x "
                    f"{r.typical_share_at_checkpoint:.0%} (typical share of final sales reached {r.checkpoint_lead_days} days before kickoff)",
                    "explanation": f"At the current forecast this game will be about {shortfall:,.0f} tickets "
                    f"({shortfall / r.checkpoint_target:.0%}) behind the pace historically needed to reach its target.",
                    "action": "Prioritize this game in the next marketing push (email/SMS to past buyers of similar games, "
                    "group-sales outreach) and re-check pace after 7 days.",
                }
            )
        elif r.forecast_status in ("Near capacity", "Sold out"):
            recs.append(
                {
                    "priority": "Medium" if r.forecast_status == "Near capacity" else "Low",
                    "category": "Inventory",
                    "game": label,
                    "game_id": r.game_id,
                    "trigger_metric": f"Remaining inventory: {r.remaining_inventory:,.0f} now; "
                    f"projected {max(r.projected_remaining_at_checkpoint, 0):,.0f} by {r.checkpoint_date:%b %d}",
                    "threshold": f"Near-capacity threshold: {NEAR_CAPACITY_SHARE:.0%} of capacity "
                    f"({NEAR_CAPACITY_SHARE * r.stadium_capacity:,.0f} seats)",
                    "explanation": "This game is sold out." if r.forecast_status == "Sold out"
                    else "The forecast suggests this game will be close to full within the next week.",
                    "action": "Pause discount promotions for this game, shift marketing spend to slower games, and "
                    "prepare waitlist / resale communication. (This is not a price-optimization recommendation.)",
                }
            )
    if propensity_live is not None and not propensity_live.empty:
        top = propensity_live[propensity_live["top_10pct"]]
        game = propensity_live["game_id"].iloc[0]
        row = checkpoints.loc[checkpoints["game_id"] == game]
        label = f"{row['game_date'].iloc[0]:%b %d} vs {row['opponent'].iloc[0]}" if len(row) else game
        recs.append(
            {
                "priority": "Medium",
                "category": "Fan outreach",
                "game": label,
                "game_id": game,
                "trigger_metric": f"{len(top):,} of {len(propensity_live):,} eligible fans in the top {TOP_FRACTION:.0%} "
                f"by propensity score (score >= {top['propensity_score'].min():.2f})",
                "threshold": f"Top {TOP_FRACTION:.0%} of eligible fans at the 14-day snapshot ({as_of:%b %d, %Y})",
                "explanation": "These fans look most likely to buy based on purchase history, recent engagement and "
                "game characteristics. High propensity does NOT mean they are the fans most influenced by outreach "
                "- many would buy anyway.",
                "action": "Use the list for reminder/early-access messaging via each fan's historical preferred channel. "
                "To measure impact, hold out a random share of the list as a control group.",
            }
        )
    order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(recs, key=lambda d: order[d["priority"]])
