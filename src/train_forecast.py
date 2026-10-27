"""Seven-day ticket-sales forecasting: baseline vs XGBoost.

Evaluation protocol
-------------------
* Rolling origins every 7 days through each evaluation season's sales period.
* At each origin the XGBoost model is refit on rows whose label date is on or
  before the origin (expanding window), then forecasts days 1-7 recursively.
* Validation season (2023) selects the approach/hyper-parameters.
* Test season (2024, the latest completed season) is scored once, afterwards,
  for the baseline and the validation-selected XGBoost configuration.
* The operational forecast uses the validation-selected approach, refit on all
  observed data up to the as-of date.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src import database
from src.config import (
    FORECAST_HORIZON,
    MODELS_DIR,
    REPORTS_DIR,
    ROLLING_ORIGIN_STEP_DAYS,
    SEED,
    TEST_SEASON,
    VALIDATION_SEASON,
)
from src.evaluate import forecast_metrics
from src.features import (
    FORECAST_FEATURES,
    add_forecast_features,
    baseline_forecast,
    recursive_forecast,
    training_rows,
)

XGB_CONFIGS = {
    "xgb_depth3": dict(max_depth=3, n_estimators=400, learning_rate=0.05, min_child_weight=5),
    "xgb_depth5": dict(max_depth=5, n_estimators=300, learning_rate=0.05, min_child_weight=10),
}


def make_xgb(params: dict) -> XGBRegressor:
    # Poisson objective: non-negative count predictions without ad-hoc clipping
    # of the loss; inventory clipping is still applied at forecast time.
    return XGBRegressor(
        objective="count:poisson",
        tree_method="hist",
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=4,
        **params,
    )


def fit_xgb(features: pd.DataFrame, cutoff, params: dict) -> XGBRegressor:
    train = training_rows(features, cutoff)
    model = make_xgb(params)
    model.fit(train[FORECAST_FEATURES], train["tickets_sold"])
    return model


def season_origins(daily: pd.DataFrame, season: int, step: int = ROLLING_ORIGIN_STEP_DAYS) -> list[pd.Timestamp]:
    s = daily[daily["season"] == season]
    start = s["sales_open_date"].min() + pd.Timedelta(days=14)
    end = s["game_date"].max() - pd.Timedelta(days=1)
    return list(pd.date_range(start, end, freq=f"{step}D"))


def rolling_origin_evaluation(
    daily: pd.DataFrame, features: pd.DataFrame, season: int, configs: dict[str, dict]
) -> pd.DataFrame:
    """Forecast points (origin, game, horizon) with actuals for one season."""
    game_ids = sorted(daily.loc[daily["season"] == season, "game_id"].unique())
    actual = daily.set_index(["game_id", "sales_date"])["tickets_sold"]
    parts = []
    for origin in season_origins(daily, season):
        base = baseline_forecast(daily, origin, FORECAST_HORIZON, game_ids)
        if base.empty:
            continue
        base = base.rename(columns={"forecast": "baseline"})
        for name, params in configs.items():
            model = fit_xgb(features, origin, params)
            fc = recursive_forecast(model.predict, daily, origin, FORECAST_HORIZON, game_ids)
            base = base.merge(
                fc[["game_id", "sales_date", "forecast"]].rename(columns={"forecast": name}),
                on=["game_id", "sales_date"],
            )
        parts.append(base)
    points = pd.concat(parts, ignore_index=True)
    points["actual"] = actual.reindex(pd.MultiIndex.from_frame(points[["game_id", "sales_date"]])).to_numpy()
    if points["actual"].isna().any():
        raise ValueError("Evaluation season contains unobserved days.")
    points["season"] = season
    return points


def run(conn: sqlite3.Connection, as_of: date) -> dict:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    daily = database.daily_sales(conn)
    features = add_forecast_features(daily)

    # 1) Validation: choose between baseline and XGBoost configurations.
    print(f"[forecast] rolling-origin validation on {VALIDATION_SEASON} ...")
    val_points = rolling_origin_evaluation(daily, features, VALIDATION_SEASON, XGB_CONFIGS)
    candidates = ["baseline", *XGB_CONFIGS]
    val_overall = forecast_metrics(val_points, candidates).iloc[0]
    val_wape = {c: float(val_overall[f"{c}_wape"]) for c in candidates}
    selected = min(val_wape, key=val_wape.get)
    best_xgb = min(XGB_CONFIGS, key=lambda c: val_wape[c])
    print(f"[forecast] validation WAPE: { {k: round(v, 4) for k, v in val_wape.items()} } -> selected {selected}")

    # 2) Test: score the pre-specified baseline and best XGBoost config once.
    print(f"[forecast] rolling-origin test on {TEST_SEASON} ...")
    test_points = rolling_origin_evaluation(daily, features, TEST_SEASON, {best_xgb: XGB_CONFIGS[best_xgb]})
    test_points = test_points.rename(columns={best_xgb: "xgboost"})
    val_points = val_points.rename(columns={best_xgb: "xgboost"})
    cols = ["baseline", "xgboost"]
    test_overall = forecast_metrics(test_points, cols).iloc[0]
    by_horizon = forecast_metrics(test_points, cols, by="horizon")
    by_game = forecast_metrics(test_points, cols, by="game_id")
    val_by_horizon = forecast_metrics(val_points, cols, by="horizon")

    # 3) Operational model: selected approach refit on all data up to as-of.
    op_model = fit_xgb(features, as_of, XGB_CONFIGS[best_xgb])
    base_fc = baseline_forecast(daily, as_of).rename(columns={"forecast": "baseline"})
    xgb_fc = recursive_forecast(op_model.predict, daily, as_of)
    op = base_fc.merge(xgb_fc[["game_id", "sales_date", "forecast"]].rename(columns={"forecast": "xgboost"}),
                       on=["game_id", "sales_date"])
    op["selected_forecast"] = op["baseline"] if selected == "baseline" else op["xgboost"]
    op["selected_model"] = "baseline" if selected == "baseline" else "xgboost"
    op.to_csv(REPORTS_DIR / "forecast_operational.csv", index=False)

    joblib.dump(
        {"model": op_model, "features": FORECAST_FEATURES, "config_name": best_xgb,
         "params": XGB_CONFIGS[best_xgb], "selected_approach": selected, "trained_through": as_of.isoformat()},
        MODELS_DIR / "forecast_xgb.joblib",
    )
    importance = pd.Series(op_model.feature_importances_, index=FORECAST_FEATURES).sort_values(ascending=False)

    test_points.to_csv(REPORTS_DIR / "forecast_test_points.csv", index=False)
    by_horizon.to_csv(REPORTS_DIR / "forecast_test_by_horizon.csv", index=False)
    by_game.to_csv(REPORTS_DIR / "forecast_test_by_game.csv", index=False)
    val_by_horizon.to_csv(REPORTS_DIR / "forecast_validation_by_horizon.csv", index=False)

    def _clean(x):
        return None if pd.isna(x) else round(float(x), 4)

    metrics = {
        "validation_season": VALIDATION_SEASON,
        "test_season": TEST_SEASON,
        "n_validation_origins": int(val_points["origin"].nunique()),
        "n_test_origins": int(test_points["origin"].nunique()),
        "n_validation_points": int(len(val_points)),
        "n_test_points": int(len(test_points)),
        "validation_wape": {k: _clean(v) for k, v in val_wape.items()},
        "validation_mae": {c: _clean(val_overall[f"{c}_mae"]) for c in candidates},
        "best_xgb_config": best_xgb,
        "selected_approach": selected,
        "test": {c: {"mae": _clean(test_overall[f"{c}_mae"]), "wape": _clean(test_overall[f"{c}_wape"])} for c in cols},
        "feature_importance": {k: _clean(v) for k, v in importance.items()},
        "objective": "count:poisson, predictions clipped to [0, remaining inventory]",
    }
    (REPORTS_DIR / "forecast_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[forecast] test: {metrics['test']}")
    return metrics
