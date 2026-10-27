"""Fan purchase propensity: Logistic Regression vs Random Forest.

Task: 14 days before kickoff, for a fan who has not yet bought tickets to the
game, predict whether they buy before kickoff.

Splits (by season of the game, never random rows):
  train = 2020-2022, validation = 2023 (model/hyper-parameter selection),
  test = 2024 (scored once for reporting). The operational model is the
  validation-selected candidate refit on every fully observed snapshot
  (2020-2025 completed games) and scores the live game whose snapshot date is
  the as-of date.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import database
from src.config import (
    MODELS_DIR,
    PROPENSITY_SNAPSHOT_DAYS,
    REPORTS_DIR,
    SEED,
    TEST_SEASON,
    TOP_FRACTION,
    TRAIN_SEASONS,
    VALIDATION_SEASON,
)
from src.evaluate import calibration_table, classification_metrics
from src.features import PROPENSITY_CATEGORICAL, PROPENSITY_NUMERIC, build_propensity_snapshots

SKEWED = [
    "prev_purchase_count",
    "prev_total_spend",
    "days_since_last_purchase",
    "purchases_last_365d",
    "prior_similar_game_purchases",
    "tenure_days",
    "engagement_30d",
    "ticket_page_views_30d",
    "days_since_last_engagement",
]
UNSKEWED = [c for c in PROPENSITY_NUMERIC if c not in SKEWED]
FEATURES = PROPENSITY_NUMERIC + PROPENSITY_CATEGORICAL


def make_logistic(C: float) -> Pipeline:
    prep = ColumnTransformer(
        [
            ("log_scaled", Pipeline([("log1p", FunctionTransformer(np.log1p)), ("scale", StandardScaler())]), SKEWED),
            ("scaled", StandardScaler(), UNSKEWED),
            ("onehot", OneHotEncoder(handle_unknown="ignore"), PROPENSITY_CATEGORICAL),
        ]
    )
    return Pipeline([("prep", prep), ("clf", LogisticRegression(C=C, max_iter=3000))])


def make_forest(min_samples_leaf: int) -> Pipeline:
    prep = ColumnTransformer(
        [
            ("num", "passthrough", PROPENSITY_NUMERIC),
            ("onehot", OneHotEncoder(handle_unknown="ignore"), PROPENSITY_CATEGORICAL),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        max_samples=0.5,
        n_jobs=-1,
        random_state=SEED,
    )
    return Pipeline([("prep", prep), ("clf", clf)])


CANDIDATES = {
    "logistic_C0.1": ("Logistic Regression", lambda: make_logistic(0.1)),
    "logistic_C1.0": ("Logistic Regression", lambda: make_logistic(1.0)),
    "forest_leaf20": ("Random Forest", lambda: make_forest(20)),
    "forest_leaf80": ("Random Forest", lambda: make_forest(80)),
}


def season_split(snaps: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": snaps[snaps["season"].isin(TRAIN_SEASONS)],
        "validation": snaps[snaps["season"] == VALIDATION_SEASON],
        "test": snaps[snaps["season"] == TEST_SEASON],
    }


def _score(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURES])[:, 1]


def _metrics(model: Pipeline, df: pd.DataFrame) -> dict:
    return classification_metrics(df["label"], _score(model, df), df["game_id"], TOP_FRACTION)


def live_game_id(games: pd.DataFrame, as_of: date) -> str | None:
    target = pd.Timestamp(as_of) + pd.Timedelta(days=PROPENSITY_SNAPSHOT_DAYS)
    match = games.loc[pd.to_datetime(games["game_date"]) == target, "game_id"]
    return None if match.empty else str(match.iloc[0])


def run(conn: sqlite3.Connection, as_of: date) -> dict:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    games = database.load_table(conn, "games")
    customers = database.load_table(conn, "customers")
    transactions = database.load_table(conn, "transactions")
    engagement = database.load_table(conn, "engagement")

    print("[propensity] building snapshots ...")
    snaps = build_propensity_snapshots(games, customers, transactions, engagement, as_of)
    split = season_split(snaps)

    val_results = {}
    for name, (family, factory) in CANDIDATES.items():
        model = factory().fit(split["train"][FEATURES], split["train"]["label"])
        val_results[name] = {"family": family, **_metrics(model, split["validation"])}
        print(f"[propensity] {name}: validation AP={val_results[name]['average_precision']:.4f}")
    selected = max(val_results, key=lambda n: val_results[n]["average_precision"])
    best_by_family = {}
    for name, res in val_results.items():
        fam = res["family"]
        if fam not in best_by_family or res["average_precision"] > val_results[best_by_family[fam]]["average_precision"]:
            best_by_family[fam] = name

    # Final test report: best config of each family, refit on train+validation, scored once.
    dev = pd.concat([split["train"], split["validation"]])
    test_results, test_scores = {}, {}
    for fam, name in best_by_family.items():
        model = CANDIDATES[name][1]().fit(dev[FEATURES], dev["label"])
        test_scores[name] = _score(model, split["test"])
        test_results[name] = {"family": fam, **classification_metrics(
            split["test"]["label"], test_scores[name], split["test"]["game_id"], TOP_FRACTION)}
    calib = calibration_table(split["test"]["label"], test_scores[selected])
    calib.to_csv(REPORTS_DIR / "propensity_test_calibration.csv", index=False)

    # Operational model: selected candidate refit on all fully observed snapshots.
    op_model = CANDIDATES[selected][1]().fit(snaps[FEATURES], snaps["label"])
    live_id = live_game_id(games, as_of)
    live = pd.DataFrame()
    if live_id:
        live = build_propensity_snapshots(
            games, customers, transactions, engagement, as_of, game_ids=[live_id], require_label=False
        )
    if not live.empty:
        live["propensity_score"] = _score(op_model, live)
        live = live.sort_values("propensity_score", ascending=False).reset_index(drop=True)
        live["rank"] = np.arange(1, len(live) + 1)
        live["top_10pct"] = live["rank"] <= int(np.ceil(TOP_FRACTION * len(live)))
        live["suggested_contact_channel"] = live["preferred_channel"]
        live.to_csv(REPORTS_DIR / "propensity_scores_live.csv", index=False)

    importance = _importance(op_model)
    joblib.dump(
        {"model": op_model, "features": FEATURES, "selected": selected, "family": CANDIDATES[selected][0],
         "trained_through": as_of.isoformat()},
        MODELS_DIR / "propensity.joblib",
    )
    counts = {
        k: {"rows": int(len(v)), "games": int(v["game_id"].nunique()), "prevalence": round(float(v["label"].mean()), 4)}
        for k, v in split.items()
    }
    metrics = {
        "task": f"At {PROPENSITY_SNAPSHOT_DAYS} days before kickoff, will a fan who has not yet bought tickets purchase before kickoff?",
        "split_counts": counts,
        "operational_training_rows": int(len(snaps)),
        "validation": _round(val_results),
        "selected": selected,
        "selected_family": CANDIDATES[selected][0],
        "best_by_family": best_by_family,
        "test": _round(test_results),
        "threshold_rule": f"Flag the top {int(TOP_FRACTION * 100)}% of eligible fans by score within each game",
        "live_game_id": live_id,
        "live_eligible_fans": int(len(live)),
        "importance": importance,
    }
    (REPORTS_DIR / "propensity_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[propensity] selected {selected}; test: "
          f"{ {k: round(v['average_precision'], 4) for k, v in test_results.items()} }")
    return metrics


def _importance(model: Pipeline) -> dict:
    names = model.named_steps["prep"].get_feature_names_out()
    clf = model.named_steps["clf"]
    values = clf.coef_[0] if hasattr(clf, "coef_") else clf.feature_importances_
    kind = "standardized coefficient" if hasattr(clf, "coef_") else "impurity importance"
    s = pd.Series(values, index=[n.split("__", 1)[-1] for n in names])
    s = s.reindex(s.abs().sort_values(ascending=False).index)
    return {"kind": kind, "values": {k: round(float(v), 4) for k, v in s.head(15).items()}}


def _round(results: dict) -> dict:
    out = {}
    for name, res in results.items():
        out[name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}
    return out
