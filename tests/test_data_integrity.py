"""Integrity of the generated data and SQLite database."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from src import database
from src.config import AS_OF_DATE, CURRENT_SEASON, SALES_WINDOW_DAYS
from src.generate_data import generate_all
from tests.conftest import SMALL


def test_foreign_keys_hold(db_conn, loaded):
    assert db_conn.execute("PRAGMA foreign_key_check").fetchall() == []
    tx, cust, games = loaded["transactions"], loaded["customers"], loaded["games"]
    assert tx["customer_id"].isin(cust["customer_id"]).all()
    assert tx["game_id"].isin(games["game_id"]).all()
    assert loaded["engagement"]["customer_id"].isin(cust["customer_id"]).all()
    orphan = db_conn.execute("""
        SELECT COUNT(*) FROM experiment_outcomes o LEFT JOIN experiment_assignments a
        USING (experiment_id, customer_id) WHERE a.customer_id IS NULL""").fetchone()[0]
    assert orphan == 0


def test_revenue_is_quantity_times_price(loaded):
    tx = loaded["transactions"]
    assert (tx["ticket_quantity"] > 0).all()
    assert (tx["price_per_ticket"] >= 0).all() and (tx["revenue"] >= 0).all()
    np.testing.assert_allclose(tx["revenue"], (tx["ticket_quantity"] * tx["price_per_ticket"]).round(2), atol=0.006)


def test_capacity_never_exceeded(db_conn):
    over = db_conn.execute("""
        SELECT g.game_id FROM games g JOIN transactions t USING (game_id)
        GROUP BY g.game_id HAVING SUM(t.ticket_quantity) > MAX(g.stadium_capacity)""").fetchall()
    assert over == []


def test_small_simulation_has_sellouts_to_exercise_capacity(db_conn):
    summary = database.game_summary(db_conn, AS_OF_DATE)
    assert (summary["remaining_inventory"] == 0).any()
    assert (summary["remaining_inventory"] >= 0).all()


def test_timestamps_are_coherent(loaded):
    tx = loaded["transactions"].merge(loaded["games"], on="game_id").merge(loaded["customers"], on="customer_id")
    assert (tx["purchase_date"] >= tx["sales_open_date"]).all()
    assert (tx["purchase_date"] <= tx["game_date"]).all()
    assert (tx["purchase_date"] >= tx["signup_date"]).all()
    eng = loaded["engagement"].merge(loaded["customers"], on="customer_id")
    assert (eng["engagement_date"] >= eng["signup_date"]).all()
    as_of = pd.Timestamp(AS_OF_DATE)
    assert tx["purchase_date"].max() <= as_of
    assert eng["engagement_date"].max() <= as_of
    assert (loaded["customers"]["signup_date"] <= as_of).all()


def test_daily_calendar_is_complete_and_distinguishes_missing(db_conn, daily, loaded):
    per_game = daily.groupby("game_id").size()
    assert (per_game == SALES_WINDOW_DAYS + 1).all()
    as_of = pd.Timestamp(AS_OF_DATE)
    observed = daily[daily["sales_date"] <= as_of]
    future = daily[daily["sales_date"] > as_of]
    assert observed["tickets_sold"].notna().all() and (observed["is_observed"] == 1).all()
    assert future["tickets_sold"].isna().all() and (future["is_observed"] == 0).all()
    assert (observed["tickets_sold"] == 0).any(), "zero-sale days must be present"
    # Daily totals reconcile with transactions.
    tx_tot = loaded["transactions"].groupby("game_id")["ticket_quantity"].sum()
    day_tot = observed.groupby("game_id")["tickets_sold"].sum()
    pd.testing.assert_series_equal(day_tot.reindex(tx_tot.index).astype(int), tx_tot.astype(int), check_names=False)


def test_schedule_relative_to_as_of(loaded):
    games = loaded["games"]
    cur = games[games["season"] == CURRENT_SEASON]
    as_of = pd.Timestamp(AS_OF_DATE)
    assert (cur["game_date"] < as_of).any() and (cur["game_date"] > as_of).any()
    assert (games["game_date"] == as_of + timedelta(days=14)).sum() == 1
    assert not (games["game_date"] == as_of).any()
    assert games.groupby("season").size().eq(10).all()
    assert set(games["season"]) == {2020, 2021, 2022, 2023, 2024, 2025}


def test_experiment_one_assignment_per_customer(tables):
    a = tables["experiment_assignments"]
    assert not a["customer_id"].duplicated().any()
    assert set(a["group_name"]) == {"control", "treatment"}
    assert (pd.to_datetime(a["experiment_end_date"]) > pd.to_datetime(a["assignment_date"])).all()


def test_generation_is_reproducible(tables):
    again = generate_all(SMALL)
    for name, df in tables.items():
        pd.testing.assert_frame_equal(df.reset_index(drop=True), again[name].reset_index(drop=True))
