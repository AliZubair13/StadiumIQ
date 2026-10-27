"""SQLite schema, loading, and the analytical SQL used by the dashboard."""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

from src.config import AS_OF_DATE, DB_PATH, SEED

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE simulation_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE games (
    game_id             TEXT PRIMARY KEY,
    season              INTEGER NOT NULL,
    opponent            TEXT NOT NULL,
    opponent_popularity REAL NOT NULL CHECK (opponent_popularity BETWEEN 0 AND 1),
    game_date           TEXT NOT NULL,
    sales_open_date     TEXT NOT NULL,
    stadium_capacity    INTEGER NOT NULL CHECK (stadium_capacity > 0),
    base_ticket_price   REAL NOT NULL CHECK (base_ticket_price > 0),
    weekend_indicator   INTEGER NOT NULL CHECK (weekend_indicator IN (0, 1)),
    rivalry_indicator   INTEGER NOT NULL CHECK (rivalry_indicator IN (0, 1)),
    sales_target        INTEGER NOT NULL CHECK (sales_target > 0)
);

CREATE TABLE customers (
    customer_id       TEXT PRIMARY KEY,
    signup_date       TEXT NOT NULL,
    preferred_channel TEXT NOT NULL,
    distance_band     TEXT NOT NULL,
    fan_segment       TEXT NOT NULL
);

CREATE TABLE transactions (
    transaction_id   INTEGER PRIMARY KEY,
    customer_id      TEXT NOT NULL REFERENCES customers(customer_id),
    game_id          TEXT NOT NULL REFERENCES games(game_id),
    purchase_date    TEXT NOT NULL,
    ticket_quantity  INTEGER NOT NULL CHECK (ticket_quantity > 0),
    price_per_ticket REAL NOT NULL CHECK (price_per_ticket >= 0),
    revenue          REAL NOT NULL CHECK (revenue >= 0)
);

CREATE TABLE engagement (
    engagement_id   INTEGER PRIMARY KEY,
    customer_id     TEXT NOT NULL REFERENCES customers(customer_id),
    engagement_date TEXT NOT NULL,
    channel         TEXT NOT NULL,
    engagement_type TEXT NOT NULL
);

CREATE TABLE experiment_assignments (
    experiment_id       TEXT NOT NULL,
    customer_id         TEXT NOT NULL REFERENCES customers(customer_id),
    assignment_date     TEXT NOT NULL,
    experiment_end_date TEXT NOT NULL,
    group_name          TEXT NOT NULL CHECK (group_name IN ('control', 'treatment')),
    PRIMARY KEY (experiment_id, customer_id)
);

CREATE TABLE experiment_outcomes (
    experiment_id TEXT NOT NULL,
    customer_id   TEXT NOT NULL,
    purchased     INTEGER NOT NULL CHECK (purchased IN (0, 1)),
    revenue       REAL NOT NULL CHECK (revenue >= 0),
    PRIMARY KEY (experiment_id, customer_id),
    FOREIGN KEY (experiment_id, customer_id)
        REFERENCES experiment_assignments(experiment_id, customer_id)
);

-- Complete daily calendar per game. tickets_sold is NULL (missing) for days after
-- the as-of date and 0 for observed days with no sales: the two are never mixed.
CREATE TABLE daily_sales (
    game_id            TEXT NOT NULL REFERENCES games(game_id),
    sales_date         TEXT NOT NULL,
    days_until_kickoff INTEGER NOT NULL,
    tickets_sold       INTEGER,
    revenue            REAL,
    is_observed        INTEGER NOT NULL CHECK (is_observed IN (0, 1)),
    PRIMARY KEY (game_id, sales_date)
);

CREATE INDEX ix_tx_game_date ON transactions(game_id, purchase_date);
CREATE INDEX ix_tx_customer ON transactions(customer_id, purchase_date);
CREATE INDEX ix_eng_customer ON engagement(customer_id, engagement_date);
"""

DAILY_SALES_SQL = """
WITH RECURSIVE calendar(game_id, sales_date, game_date) AS (
    SELECT game_id, sales_open_date, game_date FROM games
    UNION ALL
    SELECT game_id, date(sales_date, '+1 day'), game_date
    FROM calendar WHERE sales_date < game_date
),
tx AS (
    SELECT game_id, purchase_date, SUM(ticket_quantity) AS tickets, SUM(revenue) AS revenue
    FROM transactions GROUP BY game_id, purchase_date
)
INSERT INTO daily_sales
SELECT c.game_id,
       c.sales_date,
       CAST(julianday(c.game_date) - julianday(c.sales_date) AS INTEGER),
       CASE WHEN c.sales_date <= :as_of THEN COALESCE(tx.tickets, 0) END,
       CASE WHEN c.sales_date <= :as_of THEN ROUND(COALESCE(tx.revenue, 0), 2) END,
       CASE WHEN c.sales_date <= :as_of THEN 1 ELSE 0 END
FROM calendar c
LEFT JOIN tx ON tx.game_id = c.game_id AND tx.purchase_date = c.sales_date;
"""

DATE_COLUMNS = {
    "games": ["game_date", "sales_open_date"],
    "customers": ["signup_date"],
    "transactions": ["purchase_date"],
    "engagement": ["engagement_date"],
    "experiment_assignments": ["assignment_date", "experiment_end_date"],
    "daily_sales": ["sales_date"],
}
TABLE_ORDER = [
    "games",
    "customers",
    "transactions",
    "engagement",
    "experiment_assignments",
    "experiment_outcomes",
]


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _to_sql_frame(df: pd.DataFrame, table: str) -> pd.DataFrame:
    out = df.copy()
    for col in DATE_COLUMNS.get(table, []):
        out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")
    return out


def create_database(
    tables: dict[str, pd.DataFrame],
    db_path: Path | str = DB_PATH,
    as_of: date = AS_OF_DATE,
    seed: int = SEED,
) -> Path:
    """(Re)create the SQLite database from generated DataFrames."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        for name in TABLE_ORDER:
            df = _to_sql_frame(tables[name], name)
            cols = ", ".join(df.columns)
            marks = ", ".join("?" for _ in df.columns)
            conn.executemany(
                f"INSERT INTO {name} ({cols}) VALUES ({marks})",
                df.itertuples(index=False, name=None),
            )
        conn.execute(DAILY_SALES_SQL, {"as_of": as_of.isoformat()})
        conn.executemany(
            "INSERT INTO simulation_metadata VALUES (?, ?)",
            [
                ("as_of_date", as_of.isoformat()),
                ("seed", str(seed)),
                ("data_origin", "synthetic (src/generate_data.py)"),
            ],
        )
    return db_path


def read_sql(sql: str, conn: sqlite3.Connection, params: dict | tuple | None = None) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def load_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    df = read_sql(f"SELECT * FROM {table}", conn)
    for col in DATE_COLUMNS.get(table, []):
        df[col] = pd.to_datetime(df[col])
    return df


def get_as_of(conn: sqlite3.Connection) -> date:
    value = conn.execute("SELECT value FROM simulation_metadata WHERE key = 'as_of_date'").fetchone()[0]
    return date.fromisoformat(value)


# ---------------------------------------------------------------------------
# Dashboard queries
# ---------------------------------------------------------------------------

def game_summary(conn: sqlite3.Connection, as_of: date) -> pd.DataFrame:
    """One row per game with observed sales, revenue, inventory and status."""
    sql = """
    SELECT g.game_id, g.season, g.opponent, g.opponent_popularity, g.game_date,
           g.sales_open_date, g.stadium_capacity, g.base_ticket_price,
           g.weekend_indicator, g.rivalry_indicator, g.sales_target,
           COALESCE(SUM(t.ticket_quantity), 0)             AS tickets_sold,
           ROUND(COALESCE(SUM(t.revenue), 0), 2)            AS revenue,
           COUNT(t.transaction_id)                          AS orders,
           g.stadium_capacity - COALESCE(SUM(t.ticket_quantity), 0) AS remaining_inventory,
           CASE WHEN g.game_date <= :as_of THEN 'Completed' ELSE 'Upcoming' END AS status,
           CAST(julianday(g.game_date) - julianday(:as_of) AS INTEGER) AS days_until_kickoff
    FROM games g
    LEFT JOIN transactions t ON t.game_id = g.game_id AND t.purchase_date <= :as_of
    GROUP BY g.game_id
    ORDER BY g.game_date
    """
    df = read_sql(sql, conn, {"as_of": as_of.isoformat()})
    df["sell_through"] = df["tickets_sold"] / df["stadium_capacity"]
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["sales_open_date"] = pd.to_datetime(df["sales_open_date"])
    df["game_label"] = (
        df["game_date"].dt.strftime("%b %d, %Y") + " vs " + df["opponent"]
    )
    return df


def daily_sales(conn: sqlite3.Connection, game_ids: list[str] | None = None) -> pd.DataFrame:
    """Daily calendar joined with game attributes (NULL = not yet observed)."""
    sql = """
    SELECT d.game_id, d.sales_date, d.days_until_kickoff, d.tickets_sold, d.revenue,
           d.is_observed, g.season, g.game_date, g.sales_open_date, g.opponent,
           g.opponent_popularity, g.rivalry_indicator, g.weekend_indicator,
           g.base_ticket_price, g.stadium_capacity, g.sales_target
    FROM daily_sales d JOIN games g USING (game_id)
    """
    params: tuple = ()
    if game_ids:
        sql += f" WHERE d.game_id IN ({', '.join('?' for _ in game_ids)})"
        params = tuple(game_ids)
    sql += " ORDER BY d.game_id, d.sales_date"
    df = read_sql(sql, conn, params)
    for col in ["sales_date", "game_date", "sales_open_date"]:
        df[col] = pd.to_datetime(df[col])
    return df


def cumulative_sales(conn: sqlite3.Connection, game_ids: list[str]) -> pd.DataFrame:
    """Observed daily and cumulative sales per game using a window function."""
    marks = ", ".join("?" for _ in game_ids)
    sql = f"""
    SELECT d.game_id, d.sales_date, d.days_until_kickoff, d.tickets_sold, d.revenue,
           SUM(d.tickets_sold) OVER (PARTITION BY d.game_id ORDER BY d.sales_date) AS cumulative_tickets,
           SUM(d.revenue) OVER (PARTITION BY d.game_id ORDER BY d.sales_date)      AS cumulative_revenue,
           1.0 * SUM(d.tickets_sold) OVER (PARTITION BY d.game_id ORDER BY d.sales_date)
               / g.stadium_capacity AS cumulative_sell_through
    FROM daily_sales d JOIN games g USING (game_id)
    WHERE d.is_observed = 1 AND d.game_id IN ({marks})
    ORDER BY d.game_id, d.sales_date
    """
    df = read_sql(sql, conn, tuple(game_ids))
    df["sales_date"] = pd.to_datetime(df["sales_date"])
    return df


def season_daily_sales(conn: sqlite3.Connection, season: int) -> pd.DataFrame:
    sql = """
    SELECT d.sales_date, SUM(d.tickets_sold) AS tickets_sold, ROUND(SUM(d.revenue), 2) AS revenue
    FROM daily_sales d JOIN games g USING (game_id)
    WHERE g.season = ? AND d.is_observed = 1
    GROUP BY d.sales_date ORDER BY d.sales_date
    """
    df = read_sql(sql, conn, (int(season),))
    df["sales_date"] = pd.to_datetime(df["sales_date"])
    df["cumulative_tickets"] = df["tickets_sold"].cumsum()
    df["cumulative_revenue"] = df["revenue"].cumsum()
    return df


def completed_pace_curves(conn: sqlite3.Connection, as_of: date) -> pd.DataFrame:
    """Cumulative sales by days-until-kickoff for every completed game.

    Includes cumulative share of the game's *final realized* sales (used for
    checkpoint targets) and cumulative sell-through of capacity.
    """
    sql = """
    WITH c AS (
        SELECT d.game_id, d.days_until_kickoff,
               SUM(d.tickets_sold) OVER (PARTITION BY d.game_id ORDER BY d.sales_date) AS cum_tickets,
               SUM(d.tickets_sold) OVER (PARTITION BY d.game_id) AS final_tickets
        FROM daily_sales d JOIN games g USING (game_id)
        WHERE g.game_date <= :as_of
    )
    SELECT c.game_id, g.season, g.opponent, g.opponent_popularity, g.rivalry_indicator,
           g.weekend_indicator, g.stadium_capacity, c.days_until_kickoff, c.cum_tickets,
           c.final_tickets,
           1.0 * c.cum_tickets / NULLIF(c.final_tickets, 0) AS share_of_final,
           1.0 * c.cum_tickets / g.stadium_capacity          AS cum_sell_through
    FROM c JOIN games g USING (game_id)
    ORDER BY c.game_id, c.days_until_kickoff DESC
    """
    return read_sql(sql, conn, {"as_of": as_of.isoformat()})


def comparable_games_at_lead(
    conn: sqlite3.Connection, game_id: str, lead_days: int, as_of: date, pop_tolerance: float = 0.15
) -> pd.DataFrame:
    """Completed games similar to `game_id` and their sell-through at the same lead time.

    Comparable = same rivalry flag and opponent popularity within +/- tolerance.
    """
    sql = """
    WITH target AS (SELECT * FROM games WHERE game_id = :gid),
    comps AS (
        SELECT g.* FROM games g, target t
        WHERE g.game_date <= :as_of AND g.game_id <> t.game_id
          AND g.rivalry_indicator = t.rivalry_indicator
          AND ABS(g.opponent_popularity - t.opponent_popularity) <= :tol
    )
    SELECT c.game_id, c.season, c.opponent, c.game_date, c.opponent_popularity,
           c.stadium_capacity,
           SUM(CASE WHEN d.days_until_kickoff >= :lead THEN d.tickets_sold ELSE 0 END) AS tickets_at_lead,
           SUM(d.tickets_sold) AS final_tickets
    FROM comps c JOIN daily_sales d ON d.game_id = c.game_id
    GROUP BY c.game_id ORDER BY c.game_date
    """
    df = read_sql(sql, conn, {"gid": game_id, "as_of": as_of.isoformat(), "tol": pop_tolerance, "lead": lead_days})
    df["sell_through_at_lead"] = df["tickets_at_lead"] / df["stadium_capacity"]
    df["final_sell_through"] = df["final_tickets"] / df["stadium_capacity"]
    return df


def experiment_data(conn: sqlite3.Connection) -> pd.DataFrame:
    sql = """
    SELECT a.experiment_id, a.customer_id, a.assignment_date, a.experiment_end_date,
           a.group_name, COALESCE(o.purchased, 0) AS purchased, COALESCE(o.revenue, 0) AS revenue
    FROM experiment_assignments a
    LEFT JOIN experiment_outcomes o
           ON o.experiment_id = a.experiment_id AND o.customer_id = a.customer_id
    """
    return read_sql(sql, conn)
