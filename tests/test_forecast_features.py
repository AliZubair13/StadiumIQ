"""Leakage safety of forecast features and recursive forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import TEST_SEASON, VALIDATION_SEASON
from src.features import (
    FORECAST_FEATURES,
    add_forecast_features,
    baseline_forecast,
    recursive_forecast,
    training_rows,
)
from src.train_forecast import XGB_CONFIGS, fit_xgb, season_origins


def test_lags_and_rolling_are_shifted_and_grouped(toy_daily):
    f = add_forecast_features(toy_daily).set_index(["game_id", "sales_date"])
    a = f.loc["A"]
    sales_a = [5, 3, 0, 4, 2, 6, 1, 0, 7, 8]
    # First day of each game: no history, so no bleed from the other game.
    assert a.iloc[0][["lag_1", "lag_2", "lag_7", "roll_mean_7", "cum_sold_before"]].eq(0).all()
    assert f.loc["B"].iloc[0]["lag_1"] == 0
    for t in range(1, 10):
        row = a.iloc[t]
        assert row["lag_1"] == sales_a[t - 1]
        assert row["roll_mean_7"] == pytest.approx(np.mean(sales_a[max(0, t - 7):t]))
        assert row["cum_sold_before"] == sum(sales_a[:t])
        assert row["remaining_before"] == 60 - sum(sales_a[:t])
    # Changing today's sales never changes today's features.
    bumped = toy_daily.copy()
    bumped.loc[(bumped.game_id == "A") & (bumped.sales_date == "2024-01-05"), "tickets_sold"] = 999
    f2 = add_forecast_features(bumped).set_index(["game_id", "sales_date"])
    pd.testing.assert_series_equal(f.loc[("A", pd.Timestamp("2024-01-05")), FORECAST_FEATURES],
                                   f2.loc[("A", pd.Timestamp("2024-01-05")), FORECAST_FEATURES])


def test_unknown_history_is_not_treated_as_zero(toy_daily):
    d = toy_daily.copy()
    d.loc[(d.game_id == "A") & (d.sales_date >= "2024-01-06"), "tickets_sold"] = np.nan
    f = add_forecast_features(d).set_index(["game_id", "sales_date"]).loc["A"]
    assert np.isnan(f.loc["2024-01-07", "lag_1"])
    assert np.isnan(f.loc["2024-01-08", "cum_sold_before"])


def test_training_rows_respect_cutoff(daily):
    f = add_forecast_features(daily)
    cutoff = pd.Timestamp("2023-09-15")
    rows = training_rows(f, cutoff)
    assert rows["sales_date"].max() <= cutoff
    assert rows["tickets_sold"].notna().all()


def test_rolling_origins_lie_in_evaluation_season(daily):
    for season in (VALIDATION_SEASON, TEST_SEASON):
        origins = season_origins(daily, season)
        s = daily[daily.season == season]
        assert origins[0] > s["sales_open_date"].min()
        assert origins[-1] < s["game_date"].max()
        # Every training label for the first origin predates it; nothing from later seasons.
        f = add_forecast_features(daily)
        rows = training_rows(f, origins[0])
        assert rows["season"].max() <= season and rows["sales_date"].max() <= origins[0]


def _lag_model(rows: pd.DataFrame) -> np.ndarray:
    return rows["lag_1"].to_numpy() + 1.0


def test_forecast_ignores_future_actuals(daily):
    origin = pd.Timestamp("2024-10-01")
    model = fit_xgb(add_forecast_features(daily), origin, XGB_CONFIGS["xgb_depth3"])
    fc1 = recursive_forecast(model.predict, daily, origin)
    tampered = daily.copy()
    after = tampered["sales_date"] > origin
    tampered.loc[after, "tickets_sold"] = tampered.loc[after, "tickets_sold"].fillna(0) * 50 + 77
    fc2 = recursive_forecast(model.predict, tampered, origin)
    pd.testing.assert_frame_equal(fc1, fc2)
    b1, b2 = baseline_forecast(daily, origin), baseline_forecast(tampered, origin)
    pd.testing.assert_frame_equal(b1, b2)


def test_recursion_feeds_predictions_back(toy_daily):
    origin = pd.Timestamp("2024-01-05")
    fc = recursive_forecast(_lag_model, toy_daily, origin, game_ids=["B"])
    # Day 1 uses the actual sales on the origin day; later days use the previous prediction.
    actual_at_origin = toy_daily.query("game_id=='B' and sales_date==@origin")["tickets_sold"].iloc[0]
    expected = actual_at_origin + np.arange(1, len(fc) + 1)
    np.testing.assert_allclose(fc["forecast"], expected)


def test_forecast_respects_inventory_and_kickoff(toy_daily, daily):
    huge = lambda rows: np.full(len(rows), 1e6)  # noqa: E731
    origin = pd.Timestamp("2024-01-06")
    fc = recursive_forecast(huge, toy_daily, origin)
    games = toy_daily.groupby("game_id").agg(cap=("stadium_capacity", "first"), kick=("game_date", "first"))
    sold = toy_daily[toy_daily.sales_date <= origin].groupby("game_id")["tickets_sold"].sum()
    for gid, part in fc.groupby("game_id"):
        assert (part["forecast"] >= 0).all()
        assert part["forecast"].sum() <= games.loc[gid, "cap"] - sold[gid] + 1e-9
        assert part["sales_date"].max() <= games.loc[gid, "kick"]
    # Game A kicks off 4 days after the origin: exactly 4 forecast days, not 7.
    assert len(fc[fc.game_id == "A"]) == 4
    assert fc["horizon"].max() <= 7
    base = baseline_forecast(toy_daily, origin)
    for gid, part in base.groupby("game_id"):
        assert part["forecast"].sum() <= games.loc[gid, "cap"] - sold[gid] + 1e-9


def test_operational_style_forecast_on_real_calendar(daily, as_of):
    model = fit_xgb(add_forecast_features(daily), as_of, XGB_CONFIGS["xgb_depth3"])
    fc = recursive_forecast(model.predict, daily, as_of)
    assert not fc.empty and fc["forecast"].notna().all()
    assert fc["sales_date"].min() > pd.Timestamp(as_of)
    kick = daily.groupby("game_id")["game_date"].first()
    assert (fc["sales_date"] <= fc["game_id"].map(kick)).all()
