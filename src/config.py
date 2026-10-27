"""Central configuration for StadiumIQ.

Every module reads dates, paths and modelling constants from here so the
dashboard, training and tests agree on a single simulation timeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
DB_PATH = DATA_DIR / "stadiumiq.db"

TEAM_NAME = "Miami Waves"
VENUE_NAME = "Waves Field"
SYNTHETIC_BADGE = "Synthetic portfolio demonstration"

# Fixed "today" of the simulation. All observable data stops at the end of this
# day; nothing in the project reads the real system clock.
AS_OF_DATE = date(2025, 10, 18)

SEED = 42
COMPLETED_SEASONS = [2020, 2021, 2022, 2023, 2024]
CURRENT_SEASON = 2025
ALL_SEASONS = COMPLETED_SEASONS + [CURRENT_SEASON]

SALES_WINDOW_DAYS = 90          # sales open 90 days before kickoff
N_CUSTOMERS = 5_000
BASE_CAPACITY = 5_000           # seats for the full-size simulation

# Home-game offsets (days) from the first Saturday of September.
# Produces Saturday games plus a few Thursday/Monday night games.
SCHEDULE_OFFSETS = [0, 14, 26, 35, 47, 56, 70, 79, 91, 105]

# Chronological splits (by season) shared by both models.
TRAIN_SEASONS = [2020, 2021, 2022]
VALIDATION_SEASON = 2023
TEST_SEASON = 2024              # latest completed season, untouched until final report

FORECAST_HORIZON = 7
ROLLING_ORIGIN_STEP_DAYS = 7

PROPENSITY_SNAPSHOT_DAYS = 14   # snapshot taken 14 days before kickoff
TOP_FRACTION = 0.10             # outreach list = top 10% of eligible fans per game

NEAR_CAPACITY_SHARE = 0.05      # flag games with <= 5% inventory left
SRM_ALPHA = 0.001               # sample-ratio-mismatch alarm level

EXPERIMENT_ID = "EXP-2025-EMAIL-01"
EXPERIMENT_START = date(2025, 8, 4)
EXPERIMENT_END = date(2025, 8, 24)


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for the synthetic generator (tests use smaller values)."""

    seed: int = SEED
    n_customers: int = N_CUSTOMERS
    capacity: int = BASE_CAPACITY
    as_of: date = AS_OF_DATE
    seasons: list[int] = field(default_factory=lambda: list(ALL_SEASONS))
