"""Leakage-safe feature engineering for both models.

Forecasting
-----------
Each row is (game, sales_date) and the target is tickets sold that day. Every
feature for date t uses only sales on dates < t of the *same game*: lags are
grouped by game and shifted, rolling means are shifted by one day before
rolling, and inventory is computed from sales strictly before t.

Multi-day forecasts are produced recursively: at an origin, all sales after the
origin are discarded, then each day's prediction is written back into the
series so the next day's lags, rolling means and remaining inventory are built
from predictions rather than from future actuals.

Propensity
----------
One snapshot per eligible (customer, game) pair, taken 14 days before kickoff.
All features are computed from transactions and engagement dated on or before
the snapshot date.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from src.config import FORECAST_HORIZON, PROPENSITY_SNAPSHOT_DAYS

# ---------------------------------------------------------------------------
# Forecasting features
# ---------------------------------------------------------------------------

FORECAST_FEATURES = [
    "days_until_kickoff",
    "days_since_open",
    "day_of_week",
    "lag_1",
    "lag_2",
    "lag_7",
    "roll_mean_7",
    "roll_mean_28",
    "cum_sold_before",
    "remaining_before",
    "remaining_share",
    "opponent_popularity",
    "rivalry_indicator",
    "weekend_indicator",
    "base_ticket_price",
]


def add_forecast_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add lag/rolling/inventory features computed strictly from earlier days.

    `daily` needs game_id, sales_date, tickets_sold (NaN = unknown), the
    calendar columns and static game attributes. Days before sales open do not
    exist for a game, so lags reaching before the opening day are set to 0
    (no sales were possible), while lags that hit a genuinely unknown day stay NaN.
    """
    df = daily.sort_values(["game_id", "sales_date"]).reset_index(drop=True)
    y = df["tickets_sold"].astype(float)
    grp = y.groupby(df["game_id"])
    pos = df.groupby("game_id").cumcount()

    df["days_since_open"] = pos
    df["day_of_week"] = df["sales_date"].dt.dayofweek
    for k in (1, 2, 7):
        lag = grp.shift(k)
        df[f"lag_{k}"] = lag.where(pos >= k, 0.0)
    shifted = grp.shift(1)
    for w in (7, 28):
        roll = shifted.groupby(df["game_id"]).rolling(w, min_periods=1).mean()
        df[f"roll_mean_{w}"] = roll.reset_index(level=0, drop=True).where(pos >= 1, 0.0)
    # Sum of sales strictly before t; NaN if any earlier day is unknown.
    cum_through = y.fillna(0.0).groupby(df["game_id"]).cumsum()
    known_through = y.notna().astype(int).groupby(df["game_id"]).cummin()
    cum_before = cum_through.groupby(df["game_id"]).shift(1)
    prev_known = known_through.groupby(df["game_id"]).shift(1) == 1
    df["cum_sold_before"] = cum_before.where(prev_known, np.nan).where(pos >= 1, 0.0)
    df["remaining_before"] = df["stadium_capacity"] - df["cum_sold_before"]
    df["remaining_share"] = df["remaining_before"] / df["stadium_capacity"]
    return df


def training_rows(features: pd.DataFrame, cutoff: pd.Timestamp | date) -> pd.DataFrame:
    """Rows whose label (sales date) is on or before the cutoff and observed."""
    cutoff = pd.Timestamp(cutoff)
    mask = (features["sales_date"] <= cutoff) & features["tickets_sold"].notna()
    return features.loc[mask].dropna(subset=FORECAST_FEATURES)


def _forecast_frame(daily: pd.DataFrame, origin: pd.Timestamp, horizon: int, game_ids=None):
    """History up to the origin plus empty placeholder rows for the horizon."""
    active = daily[(daily["sales_open_date"] <= origin) & (daily["game_date"] > origin)]
    if game_ids is not None:
        active = active[active["game_id"].isin(game_ids)]
    hist = active[active["sales_date"] <= origin].copy()
    if hist["tickets_sold"].isna().any():
        raise ValueError("History before the forecast origin contains unobserved days.")
    future = active[(active["sales_date"] > origin) & (active["sales_date"] <= origin + pd.Timedelta(days=horizon))].copy()
    future["tickets_sold"] = np.nan  # never read future actuals
    return hist, future


def recursive_forecast(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    daily: pd.DataFrame,
    origin: pd.Timestamp | date,
    horizon: int = FORECAST_HORIZON,
    game_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Recursive multi-day forecast from `origin` (end of day) for active games.

    Stops at kickoff because the calendar ends on game day. Each prediction is
    clipped to [0, remaining inventory] where inventory already reflects the
    earlier predicted days.
    """
    origin = pd.Timestamp(origin)
    hist, future = _forecast_frame(daily, origin, horizon, game_ids)
    if future.empty:
        return pd.DataFrame()
    # Kept sorted so add_forecast_features returns rows in the same order.
    frame = pd.concat([hist, future]).sort_values(["game_id", "sales_date"]).reset_index(drop=True)
    for h in range(1, horizon + 1):
        target_date = origin + pd.Timedelta(days=h)
        feats = add_forecast_features(frame)
        mask = (feats["sales_date"] == target_date).to_numpy()
        if not mask.any():
            break
        rows = feats.loc[mask]
        pred = np.asarray(predict_fn(rows[FORECAST_FEATURES]), dtype=float)
        pred = np.clip(pred, 0.0, rows["remaining_before"].to_numpy())
        frame.loc[mask, "tickets_sold"] = pred
    out = frame[frame["sales_date"] > origin][["game_id", "sales_date", "days_until_kickoff", "tickets_sold"]]
    out = out.rename(columns={"tickets_sold": "forecast"})
    out["origin"] = origin
    out["horizon"] = (out["sales_date"] - origin).dt.days
    return out.reset_index(drop=True)


def baseline_forecast(
    daily: pd.DataFrame,
    origin: pd.Timestamp | date,
    horizon: int = FORECAST_HORIZON,
    game_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Flat forecast equal to the game's average over its last 7 observed days.

    If sales opened fewer than 7 days ago, the average uses the days available.
    The cumulative forecast is capped at the inventory remaining at the origin.
    """
    origin = pd.Timestamp(origin)
    hist, future = _forecast_frame(daily, origin, horizon, game_ids)
    if future.empty:
        return pd.DataFrame()
    hist = hist.sort_values(["game_id", "sales_date"])
    avg = hist.groupby("game_id")["tickets_sold"].apply(lambda s: s.tail(7).mean())
    sold = hist.groupby("game_id")["tickets_sold"].sum()
    cap = hist.groupby("game_id")["stadium_capacity"].first()
    future = future.sort_values(["game_id", "sales_date"]).copy()
    future["forecast"] = future["game_id"].map(avg).astype(float)
    remaining = (cap - sold).reindex(future["game_id"]).to_numpy()
    cum = future.groupby("game_id")["forecast"].cumsum().to_numpy()
    prev = cum - future["forecast"].to_numpy()
    future["forecast"] = np.clip(np.minimum(future["forecast"].to_numpy(), remaining - prev), 0.0, None)
    future["origin"] = origin
    future["horizon"] = (future["sales_date"] - origin).dt.days
    return future[["game_id", "sales_date", "days_until_kickoff", "forecast", "origin", "horizon"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Propensity snapshot features
# ---------------------------------------------------------------------------

PROPENSITY_NUMERIC = [
    "prev_purchase_count",
    "prev_total_spend",
    "avg_ticket_quantity",
    "days_since_last_purchase",
    "never_purchased",
    "purchases_last_365d",
    "bought_previous_home_game",
    "prior_similar_game_purchases",
    "tenure_days",
    "engagement_30d",
    "ticket_page_views_30d",
    "days_since_last_engagement",
    "opponent_popularity",
    "rivalry_indicator",
    "weekend_indicator",
    "base_ticket_price",
    "game_sell_through_at_snapshot",
]
PROPENSITY_CATEGORICAL = ["preferred_channel", "distance_band", "fan_segment"]
NO_HISTORY_DAYS = 3650  # documented fill value for "never purchased / never engaged"


def _pop_tier(pop: pd.Series | float):
    return np.digitize(pop, [0.4, 0.65])


def snapshot_date_for(game_date) -> pd.Timestamp:
    return pd.Timestamp(game_date) - pd.Timedelta(days=PROPENSITY_SNAPSHOT_DAYS)


def build_propensity_snapshots(
    games: pd.DataFrame,
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    engagement: pd.DataFrame,
    as_of: date,
    game_ids: list[str] | None = None,
    require_label: bool = True,
) -> pd.DataFrame:
    """Build customer-game snapshots 14 days before kickoff.

    require_label=True keeps only games whose whole outcome window
    (snapshot, kickoff] is on or before the as-of date (training/evaluation).
    require_label=False is used for live scoring and returns no label.
    Games already sold out at the snapshot are skipped (no one can buy).
    """
    as_of_ts = pd.Timestamp(as_of)
    g_all = games.copy()
    g_all["game_date"] = pd.to_datetime(g_all["game_date"])
    tx = transactions.copy()
    tx["purchase_date"] = pd.to_datetime(tx["purchase_date"])
    tx = tx[tx["purchase_date"] <= as_of_ts]  # defensive: nothing after as-of
    tx = tx.merge(g_all[["game_id", "rivalry_indicator", "opponent_popularity"]], on="game_id")
    tx["pop_tier"] = _pop_tier(tx["opponent_popularity"])
    eng = engagement.copy()
    eng["engagement_date"] = pd.to_datetime(eng["engagement_date"])
    eng = eng[eng["engagement_date"] <= as_of_ts]
    cust = customers.copy()
    cust["signup_date"] = pd.to_datetime(cust["signup_date"])

    targets = g_all if game_ids is None else g_all[g_all["game_id"].isin(game_ids)]
    frames = []
    for g in targets.itertuples(index=False):
        snap = snapshot_date_for(g.game_date)
        if snap > as_of_ts:
            continue  # snapshot is in the future
        if require_label and g.game_date > as_of_ts:
            continue  # outcome window not fully observed
        past = tx[tx["purchase_date"] <= snap]
        game_past = past[past["game_id"] == g.game_id]
        sold = game_past["ticket_quantity"].sum()
        if sold >= g.stadium_capacity:
            continue
        already = set(game_past["customer_id"])
        elig = cust[(cust["signup_date"] <= snap) & ~cust["customer_id"].isin(already)].copy()
        if elig.empty:
            continue

        agg = past.groupby("customer_id").agg(
            prev_purchase_count=("transaction_id", "size"),
            prev_total_spend=("revenue", "sum"),
            avg_ticket_quantity=("ticket_quantity", "mean"),
            last_purchase=("purchase_date", "max"),
        )
        recent = past[past["purchase_date"] > snap - pd.Timedelta(days=365)]
        agg["purchases_last_365d"] = recent.groupby("customer_id").size()
        similar = past[
            (past["rivalry_indicator"] == g.rivalry_indicator)
            & (past["pop_tier"] == _pop_tier(g.opponent_popularity))
        ]
        agg["prior_similar_game_purchases"] = similar.groupby("customer_id").size()
        prev_games = g_all[g_all["game_date"] <= snap].sort_values("game_date")
        if len(prev_games):
            prev_id = prev_games["game_id"].iloc[-1]
            agg["bought_previous_home_game"] = past[past["game_id"] == prev_id].groupby("customer_id").size().gt(0)

        e_past = eng[eng["engagement_date"] <= snap]
        e30 = e_past[e_past["engagement_date"] > snap - pd.Timedelta(days=30)]
        e_agg = pd.DataFrame(
            {
                "engagement_30d": e30.groupby("customer_id").size(),
                "ticket_page_views_30d": e30[e30["engagement_type"] == "ticket_page_view"].groupby("customer_id").size(),
                "last_engagement": e_past.groupby("customer_id")["engagement_date"].max(),
            }
        )

        df = elig.merge(agg, left_on="customer_id", right_index=True, how="left")
        df = df.merge(e_agg, left_on="customer_id", right_index=True, how="left")
        for col in ["prev_purchase_count", "prev_total_spend", "purchases_last_365d",
                    "prior_similar_game_purchases", "engagement_30d", "ticket_page_views_30d"]:
            df[col] = df[col].fillna(0).astype(float)
        if "bought_previous_home_game" not in df:
            df["bought_previous_home_game"] = 0.0
        df["bought_previous_home_game"] = df["bought_previous_home_game"].astype("boolean").fillna(False).astype(float)
        df["avg_ticket_quantity"] = df["avg_ticket_quantity"].fillna(0.0)
        df["never_purchased"] = df["last_purchase"].isna().astype(float)
        df["days_since_last_purchase"] = (snap - df["last_purchase"]).dt.days.fillna(NO_HISTORY_DAYS).astype(float)
        df["days_since_last_engagement"] = (snap - df["last_engagement"]).dt.days.fillna(NO_HISTORY_DAYS).astype(float)
        df["tenure_days"] = (snap - df["signup_date"]).dt.days.astype(float)
        df["game_id"] = g.game_id
        df["season"] = g.season
        df["game_date"] = g.game_date
        df["snapshot_date"] = snap
        df["opponent_popularity"] = g.opponent_popularity
        df["rivalry_indicator"] = float(g.rivalry_indicator)
        df["weekend_indicator"] = float(g.weekend_indicator)
        df["base_ticket_price"] = float(g.base_ticket_price)
        df["game_sell_through_at_snapshot"] = sold / g.stadium_capacity

        if require_label:
            window = tx[
                (tx["game_id"] == g.game_id)
                & (tx["purchase_date"] > snap)
                & (tx["purchase_date"] <= g.game_date)
            ]
            df["label"] = df["customer_id"].isin(set(window["customer_id"])).astype(int)
        frames.append(df.drop(columns=["last_purchase", "last_engagement", "signup_date"]))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
