"""Reproducible synthetic data generator for the fictional Miami Waves.

Everything here is invented. The relationships the models later "discover"
exist only because this generator puts them there, deliberately mixed with
noise and unobserved factors (latent fan loyalty, yearly fan interest shocks,
per-game "buzz", marketing spikes) so that predictions are imperfect.

Simulation design (high level):
* Each fan has an unobserved loyalty score that drives both engagement and
  purchase probability, plus a yearly interest shock.
* For every home game, each signed-up fan buys with a probability that depends
  on opponent popularity, rivalry, weekend scheduling, price relative to a
  reference price, distance from the stadium, habit (bought the previous home
  game) and an unobserved game-level shock.
* Buyers receive a purchase day from a segment-specific lead-time profile
  (early on-sale buyers vs last-minute buyers) perturbed by daily noise.
* Purchases are processed chronologically and stop when the venue is full, so
  realized sales are inventory-constrained.
* Current-season data after the as-of date is simulated (so sell-outs and
  timing are coherent) but then discarded and never stored.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.config import (
    EXPERIMENT_END,
    EXPERIMENT_ID,
    EXPERIMENT_START,
    SALES_WINDOW_DAYS,
    SCHEDULE_OFFSETS,
    SimulationConfig,
)

# Fictional opponents: (name, base popularity 0-1, rival flag)
OPPONENTS = [
    ("Palm Harbor Pelicans", 0.80, 1),
    ("Everglade Storm", 0.72, 1),
    ("Coastal Kings", 0.85, 0),
    ("Bay Ridge Hammers", 0.65, 0),
    ("Desert Comets", 0.45, 0),
    ("Lakeshore Lynx", 0.55, 0),
    ("Northern Pines", 0.30, 0),
    ("Capital Sentinels", 0.70, 0),
    ("Red Mesa Roadrunners", 0.35, 0),
    ("Granite Peak Rams", 0.40, 0),
    ("Silver Valley Stallions", 0.60, 0),
    ("Prairie Twisters", 0.25, 0),
    ("Harborview Mariners", 0.50, 0),
    ("Iron Range Miners", 0.20, 0),
]

SEGMENTS = ["Loyalist", "Regular", "Family", "Casual"]
SEGMENT_PROBS = [0.15, 0.30, 0.15, 0.40]
SEGMENT_LOYALTY = {"Loyalist": 1.6, "Regular": 0.6, "Family": 0.2, "Casual": -0.6}
SEGMENT_QTY_MEAN = {"Loyalist": 2.0, "Regular": 2.0, "Family": 3.3, "Casual": 1.6}
# Lead-time mixture shares: (on-sale burst, steady trickle, late ramp)
SEGMENT_TIMING = {
    "Loyalist": (0.45, 0.30, 0.25),
    "Regular": (0.25, 0.35, 0.40),
    "Family": (0.20, 0.35, 0.45),
    "Casual": (0.08, 0.32, 0.60),
}
CHANNELS = ["Email", "SMS", "App push"]
CHANNEL_PROBS = [0.5, 0.2, 0.3]
DISTANCE_BANDS = ["0-10 mi", "10-30 mi", "30-60 mi", "60+ mi"]
DISTANCE_PROBS = [0.35, 0.35, 0.20, 0.10]
DISTANCE_EFFECT = {"0-10 mi": 0.3, "10-30 mi": 0.1, "30-60 mi": -0.3, "60+ mi": -0.8}

ENGAGEMENT_CHANNELS = ["Email", "SMS", "App push", "Web", "Social"]
ENGAGEMENT_TYPES = {
    "Email": ["email_open", "email_click"],
    "SMS": ["sms_click"],
    "App push": ["app_open"],
    "Web": ["site_visit", "ticket_page_view"],
    "Social": ["social_interaction"],
}

PURCHASE_INTERCEPT = -0.95
FIRST_SIGNUP = date(2014, 1, 1)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def reference_price(pop: float, rivalry: int, weekend: int) -> float:
    """Price (2020 dollars) the club would 'normally' charge for a matchup."""
    return 52 + 40 * pop + 15 * rivalry + 8 * weekend


def first_saturday_of_september(season: int) -> date:
    d = date(season, 9, 1)
    while d.weekday() != 5:
        d += timedelta(days=1)
    return d


def generate_games(cfg: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    rivals = [o for o in OPPONENTS if o[2] == 1]
    others = [o for o in OPPONENTS if o[2] == 0]
    for season in cfg.seasons:
        picks = rivals + [others[i] for i in rng.choice(len(others), size=8, replace=False)]
        order = rng.permutation(len(picks))
        start = first_saturday_of_september(season)
        for k, (offset, idx) in enumerate(zip(SCHEDULE_OFFSETS, order)):
            name, base_pop, rival = picks[idx]
            gdate = start + timedelta(days=offset)
            pop = float(np.clip(base_pop + rng.normal(0, 0.07), 0.05, 0.98))
            weekend = int(gdate.weekday() >= 5)
            price = reference_price(pop, rival, weekend) * 1.035 ** (season - 2020)
            price *= rng.uniform(0.93, 1.07)
            target_rate = np.clip(0.74 + 0.16 * pop + 0.05 * rival, 0.74, 0.97)
            rows.append(
                {
                    "game_id": f"G{season}{k + 1:02d}",
                    "season": season,
                    "opponent": name,
                    "opponent_popularity": round(pop, 3),
                    "game_date": gdate,
                    "sales_open_date": gdate - timedelta(days=SALES_WINDOW_DAYS),
                    "stadium_capacity": cfg.capacity,
                    "base_ticket_price": float(round(price)),
                    "weekend_indicator": weekend,
                    "rivalry_indicator": int(rival),
                    "sales_target": int(round(cfg.capacity * target_rate, -1)),
                }
            )
    return pd.DataFrame(rows).sort_values("game_date").reset_index(drop=True)


def generate_customers(cfg: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_customers
    early_span = (date(2019, 12, 31) - FIRST_SIGNUP).days
    late_span = (cfg.as_of - timedelta(days=14) - date(2020, 1, 1)).days
    early = rng.random(n) < 0.7
    offsets = np.where(
        early,
        rng.integers(0, early_span + 1, n),
        (date(2020, 1, 1) - FIRST_SIGNUP).days + rng.integers(0, late_span + 1, n),
    )
    signup = [FIRST_SIGNUP + timedelta(days=int(o)) for o in offsets]
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
            "signup_date": signup,
            "preferred_channel": rng.choice(CHANNELS, n, p=CHANNEL_PROBS),
            "distance_band": rng.choice(DISTANCE_BANDS, n, p=DISTANCE_PROBS),
            "fan_segment": rng.choice(SEGMENTS, n, p=SEGMENT_PROBS),
        }
    )


def _timing_profiles(rng: np.random.Generator, game_date: date) -> dict[str, np.ndarray]:
    """Probability over days-until-kickoff (index 0..90) for each segment."""
    d = np.arange(SALES_WINDOW_DAYS + 1)
    onsale = np.exp(-(SALES_WINDOW_DAYS - d) / 4.0)
    steady = np.ones_like(d, dtype=float)
    late = np.exp(-d / 7.0)
    comps = [c / c.sum() for c in (onsale, steady, late)]
    # Day-level noise shared by all segments for this game (news, weather, promos).
    mult = rng.lognormal(0.0, 0.25, d.size)
    promo_days = rng.choice(np.arange(15, 76), size=2, replace=False)
    mult[promo_days] *= 2.0
    mult[0] *= 0.6  # fewer purchases on game day itself
    weekday = np.array([(game_date - timedelta(days=int(x))).weekday() for x in d])
    mult *= np.where(weekday >= 5, 0.85, 1.0)
    out = {}
    for seg, shares in SEGMENT_TIMING.items():
        p = sum(s * c for s, c in zip(shares, comps)) * mult
        out[seg] = p / p.sum()
    return out


def generate_transactions(
    cfg: SimulationConfig,
    games: pd.DataFrame,
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate full-season purchases; return (observable, hidden-future)."""
    n = len(customers)
    seg = customers["fan_segment"].to_numpy()
    loyalty = np.array([SEGMENT_LOYALTY[s] for s in seg]) + rng.normal(0, 0.9, n)
    qty_lam = np.array([SEGMENT_QTY_MEAN[s] - 1 for s in seg]) * rng.lognormal(0, 0.3, n)
    dist = customers["distance_band"].to_numpy()
    dist_eff = np.array([DISTANCE_EFFECT[b] for b in dist])
    far = np.isin(dist, ["30-60 mi", "60+ mi"])
    signup = pd.to_datetime(customers["signup_date"]).to_numpy()
    cust_ids = customers["customer_id"].to_numpy()

    bought_prev = np.zeros(n, dtype=bool)
    season_shock = np.zeros(n)
    current_season = None
    records = []
    for g in games.itertuples(index=False):
        if g.season != current_season:
            current_season = g.season
            season_shock = rng.normal(0, 0.5, n)
        gdate = pd.Timestamp(g.game_date)
        eligible = signup <= gdate.to_datetime64()
        price_ratio = g.base_ticket_price / reference_price(
            g.opponent_popularity, g.rivalry_indicator, g.weekend_indicator
        )
        logit = (
            PURCHASE_INTERCEPT
            + loyalty
            + season_shock
            + 1.2 * (g.opponent_popularity - 0.5)
            + 0.7 * g.rivalry_indicator
            + 0.3 * g.weekend_indicator
            + dist_eff
            - 0.6 * far * (1 - g.weekend_indicator)
            - 1.0 * np.log(price_ratio)
            + 0.6 * bought_prev
            + rng.normal(0, 0.35)  # unobserved game-level buzz
        )
        buy = eligible & (rng.random(n) < _sigmoid(logit))
        idx = np.flatnonzero(buy)

        profiles = _timing_profiles(rng, g.game_date)
        dtk = np.empty(idx.size, dtype=int)
        for s, p in profiles.items():
            m = seg[idx] == s
            dtk[m] = rng.choice(p.size, size=int(m.sum()), p=p)
        # Fans who sign up mid-window cannot buy before their signup date.
        max_dtk = ((gdate.to_datetime64() - signup[idx]) / np.timedelta64(1, "D")).astype(int)
        late_signup = dtk > max_dtk
        dtk[late_signup] = (rng.random(late_signup.sum()) * (max_dtk[late_signup] + 1)).astype(int)

        qty = np.minimum(1 + rng.poisson(qty_lam[idx]), 8)
        order = np.lexsort((rng.random(idx.size), -dtk))  # chronological
        idx, dtk, qty = idx[order], dtk[order], qty[order]
        cum = np.cumsum(qty)
        before = cum - qty
        qty = np.clip(np.minimum(qty, g.stadium_capacity - before), 0, None)
        keep = qty > 0
        idx, dtk, qty = idx[keep], dtk[keep], qty[keep]

        late_premium = 1 + 0.006 * np.clip(21 - dtk, 0, None)
        member = np.where(seg[idx] == "Loyalist", 0.92, 1.0)
        price = np.round(g.base_ticket_price * late_premium * member * rng.uniform(0.97, 1.03, idx.size), 2)
        records.append(
            pd.DataFrame(
                {
                    "customer_id": cust_ids[idx],
                    "game_id": g.game_id,
                    "purchase_date": [g.game_date - timedelta(days=int(x)) for x in dtk],
                    "ticket_quantity": qty.astype(int),
                    "price_per_ticket": price,
                }
            )
        )
        bought_prev = np.zeros(n, dtype=bool)
        bought_prev[idx] = True

    tx = pd.concat(records, ignore_index=True).sort_values(["purchase_date", "game_id"], kind="stable")
    tx["revenue"] = np.round(tx["ticket_quantity"] * tx["price_per_ticket"], 2)
    tx = tx.reset_index(drop=True)
    future = tx["purchase_date"] > cfg.as_of
    observed = tx.loc[~future].copy()
    observed.insert(0, "transaction_id", np.arange(1, len(observed) + 1))
    return observed.reset_index(drop=True), tx.loc[future].reset_index(drop=True)


def generate_engagement(
    cfg: SimulationConfig,
    customers: pd.DataFrame,
    all_purchases: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Background engagement plus browsing bursts in the 3 weeks before purchases.

    Bursts before (possibly future) purchases are kept only when the event itself
    falls on or before the as-of date: intent leaves observable footprints, but
    nothing after the as-of date is stored.
    """
    n = len(customers)
    seg = customers["fan_segment"].to_numpy()
    loyalty = np.array([SEGMENT_LOYALTY[s] for s in seg]) + rng.normal(0, 0.8, n)
    rate = 0.02 * np.exp(0.5 * loyalty)
    signup = pd.to_datetime(customers["signup_date"])
    as_of = pd.Timestamp(cfg.as_of)
    active_days = (as_of - signup).dt.days.to_numpy() + 1
    counts = rng.poisson(rate * active_days)
    owner = np.repeat(np.arange(n), counts)
    bg_dates = signup.to_numpy()[owner] + (rng.random(owner.size) * active_days[owner]).astype(int) * np.timedelta64(1, "D")

    pos = pd.Series(np.arange(n), index=customers["customer_id"])
    p_owner = pos.loc[all_purchases["customer_id"]].to_numpy()
    k = rng.poisson(1.8, len(all_purchases))
    b_owner = np.repeat(p_owner, k)
    b_dates = pd.to_datetime(np.repeat(all_purchases["purchase_date"].to_numpy(), k)).to_numpy()
    b_dates = b_dates - rng.integers(0, 21, b_owner.size) * np.timedelta64(1, "D")

    owners = np.concatenate([owner, b_owner])
    dates = np.concatenate([bg_dates.astype("datetime64[ns]"), b_dates.astype("datetime64[ns]")])
    is_burst = np.concatenate([np.zeros(owner.size, bool), np.ones(b_owner.size, bool)])
    valid = (dates >= signup.to_numpy()[owners]) & (dates <= as_of.to_datetime64())
    owners, dates, is_burst = owners[valid], dates[valid], is_burst[valid]

    pref = customers["preferred_channel"].to_numpy()[owners]
    use_pref = rng.random(owners.size) < 0.6
    channel = np.where(use_pref, pref, rng.choice(ENGAGEMENT_CHANNELS, owners.size))
    channel = np.where(is_burst & (rng.random(owners.size) < 0.4), "Web", channel)
    etype = np.empty(owners.size, dtype=object)
    for ch, types in ENGAGEMENT_TYPES.items():
        m = channel == ch
        etype[m] = rng.choice(types, int(m.sum()))
    burst_web = is_burst & (channel == "Web")
    etype[burst_web] = "ticket_page_view"

    eng = pd.DataFrame(
        {
            "customer_id": customers["customer_id"].to_numpy()[owners],
            "engagement_date": pd.to_datetime(dates).date,
            "channel": channel,
            "engagement_type": etype,
        }
    ).sort_values(["engagement_date", "customer_id"], kind="stable")
    eng.insert(0, "engagement_id", np.arange(1, len(eng) + 1))
    return eng.reset_index(drop=True)


def generate_experiment(cfg: SimulationConfig, customers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Independent simulated randomized email test (not linked to transactions)."""
    rng = np.random.default_rng([cfg.seed, 2025])
    pool = customers[pd.to_datetime(customers["signup_date"]) <= pd.Timestamp(EXPERIMENT_START)]
    n = len(pool)
    group = np.where(rng.random(n) < 0.5, "treatment", "control")
    base = pool["fan_segment"].map({"Loyalist": 0.13, "Regular": 0.08, "Family": 0.07, "Casual": 0.04}).to_numpy()
    odds = base / (1 - base) * np.exp(rng.normal(0, 0.3, n))
    odds = np.where(group == "treatment", odds * 1.18, odds)  # assumed true effect
    purchased = (rng.random(n) < odds / (1 + odds)).astype(int)
    qty = 1 + rng.poisson(1.2, n)
    revenue = np.round(purchased * qty * rng.uniform(55, 120, n), 2)
    assignments = pd.DataFrame(
        {
            "experiment_id": EXPERIMENT_ID,
            "customer_id": pool["customer_id"].to_numpy(),
            "assignment_date": EXPERIMENT_START,
            "experiment_end_date": EXPERIMENT_END,
            "group_name": group,
        }
    )
    outcomes = pd.DataFrame(
        {
            "experiment_id": EXPERIMENT_ID,
            "customer_id": pool["customer_id"].to_numpy(),
            "purchased": purchased,
            "revenue": revenue,
        }
    )
    return assignments, outcomes


def generate_all(cfg: SimulationConfig | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)
    games = generate_games(cfg, rng)
    customers = generate_customers(cfg, rng)
    transactions, hidden_future = generate_transactions(cfg, games, customers, rng)
    all_purchases = pd.concat([transactions, hidden_future], ignore_index=True)
    engagement = generate_engagement(cfg, customers, all_purchases, rng)
    assignments, outcomes = generate_experiment(cfg, customers)
    # hidden_future is intentionally dropped here: it never reaches the database.
    return {
        "games": games,
        "customers": customers,
        "transactions": transactions,
        "engagement": engagement,
        "experiment_assignments": assignments,
        "experiment_outcomes": outcomes,
    }
