"""Metric helpers shared by training scripts, tests and the dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


def mae(actual, pred) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    if actual.size == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - pred)))


def wape(actual, pred) -> float:
    """Weighted absolute percentage error = sum|error| / sum|actual|.

    Returns NaN when total actual volume is zero (undefined), rather than
    dividing by zero or silently returning 0.
    """
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    denom = np.abs(actual).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(actual - pred).sum() / denom)


def forecast_metrics(df: pd.DataFrame, pred_cols: list[str], by: str | list[str] | None = None) -> pd.DataFrame:
    """MAE/WAPE per prediction column, optionally grouped (e.g. by horizon)."""

    def _one(part: pd.DataFrame) -> pd.Series:
        out = {"n_points": len(part), "actual_total": part["actual"].sum()}
        for c in pred_cols:
            out[f"{c}_mae"] = mae(part["actual"], part[c])
            out[f"{c}_wape"] = wape(part["actual"], part[c])
        return pd.Series(out)

    if by is None:
        return _one(df).to_frame().T
    return df.groupby(by).apply(_one, include_groups=False).reset_index()


def top_fraction_mask(scores: np.ndarray, groups: np.ndarray | None, fraction: float) -> np.ndarray:
    """Flag the top `fraction` of scores within each group (at least one per group)."""
    scores = np.asarray(scores, float)
    groups = np.zeros(len(scores)) if groups is None else np.asarray(groups)
    flag = np.zeros(len(scores), dtype=bool)
    for gval in pd.unique(groups):
        idx = np.flatnonzero(groups == gval)
        k = max(1, int(np.ceil(fraction * idx.size)))
        top = idx[np.argsort(-scores[idx], kind="stable")[:k]]
        flag[top] = True
    return flag


def classification_metrics(y, scores, groups=None, fraction: float = 0.10) -> dict:
    """Ranking metrics plus a confusion matrix for the per-game top-fraction rule."""
    y = np.asarray(y, int)
    scores = np.asarray(scores, float)
    prevalence = float(y.mean()) if y.size else float("nan")
    flag = top_fraction_mask(scores, groups, fraction)
    precision_top = float(y[flag].mean()) if flag.any() else float("nan")
    tn, fp, fn, tp = confusion_matrix(y, flag.astype(int), labels=[0, 1]).ravel()
    two_classes = len(np.unique(y)) == 2
    return {
        "n": int(y.size),
        "prevalence": prevalence,
        "average_precision": float(average_precision_score(y, scores)) if two_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y, scores)) if two_classes else float("nan"),
        "precision_top10": precision_top,
        "lift_top10": precision_top / prevalence if prevalence > 0 else float("nan"),
        "recall_top10": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "brier": float(brier_score_loss(y, np.clip(scores, 0, 1))),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def calibration_table(y, scores, n_bins: int = 10) -> pd.DataFrame:
    """Mean score vs observed purchase rate by score decile."""
    df = pd.DataFrame({"y": np.asarray(y, int), "score": np.asarray(scores, float)})
    df["bin"] = pd.qcut(df["score"].rank(method="first"), n_bins, labels=False) + 1
    return (
        df.groupby("bin")
        .agg(n=("y", "size"), mean_score=("score", "mean"), observed_rate=("y", "mean"))
        .reset_index()
    )
