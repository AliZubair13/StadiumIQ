"""Shared fixtures: a small but complete simulation written to a temp database."""
from __future__ import annotations

import pandas as pd
import pytest

from src import database
from src.config import AS_OF_DATE, SimulationConfig
from src.features import add_forecast_features
from src.generate_data import generate_all

SMALL = SimulationConfig(n_customers=400, capacity=400)


@pytest.fixture(scope="session")
def tables():
    return generate_all(SMALL)


@pytest.fixture(scope="session")
def db_conn(tables, tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "test.db"
    database.create_database(tables, path, SMALL.as_of, SMALL.seed)
    conn = database.connect(path)
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def daily(db_conn):
    return database.daily_sales(db_conn)


@pytest.fixture(scope="session")
def loaded(db_conn):
    return {t: database.load_table(db_conn, t) for t in ["games", "customers", "transactions", "engagement"]}


@pytest.fixture
def toy_daily():
    """Two games, 10-day windows, hand-written sales for exact feature checks."""
    rows = []
    for gid, start, sales in [("A", "2024-01-01", [5, 3, 0, 4, 2, 6, 1, 0, 7, 8]),
                              ("B", "2024-01-03", [100, 90, 80, 70, 60, 50, 40, 30, 20, 10])]:
        dates = pd.date_range(start, periods=len(sales))
        for i, (d, y) in enumerate(zip(dates, sales)):
            rows.append({
                "game_id": gid, "sales_date": d, "tickets_sold": float(y),
                "days_until_kickoff": len(sales) - 1 - i, "season": 2024,
                "game_date": dates[-1], "sales_open_date": dates[0],
                "stadium_capacity": 60 if gid == "A" else 1000,
                "opponent_popularity": 0.5, "rivalry_indicator": 0, "weekend_indicator": 1,
                "base_ticket_price": 80.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def as_of():
    return AS_OF_DATE
