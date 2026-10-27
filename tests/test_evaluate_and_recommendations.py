"""Metric edge cases and recommendation rules."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.evaluate import classification_metrics, mae, top_fraction_mask, wape
from src.recommendations import build_recommendations, checkpoint_table, typical_share_curve


def test_wape_and_mae():
    assert wape([10, 0, 10], [8, 1, 12]) == pytest.approx(5 / 20)
    assert np.isnan(wape([0, 0], [1, 2]))  # undefined, not 0 or inf
    assert mae([1, 2], [2, 4]) == 1.5


def test_top_fraction_is_per_group():
    scores = np.array([0.9, 0.1, 0.2, 0.3, 0.8, 0.7, 0.6, 0.5, 0.4, 0.05])
    groups = np.array(["a"] * 5 + ["b"] * 5)
    flag = top_fraction_mask(scores, groups, 0.2)
    assert flag.tolist() == [True, False, False, False, False, True, False, False, False, False]


def test_lift_and_confusion():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    s = np.linspace(1, 0, 10)
    m = classification_metrics(y, s, None, 0.1)
    assert m["precision_top10"] == 1.0 and m["lift_top10"] == pytest.approx(5.0)
    assert m["confusion_matrix"] == {"tn": 8, "fp": 0, "fn": 1, "tp": 1}


def _checkpoint_inputs():
    pace = pd.DataFrame({"days_until_kickoff": [10, 10, 3, 3], "share_of_final": [0.5, 0.7, 0.8, 0.9]})
    upcoming = pd.DataFrame({
        "game_id": ["G1", "G2", "G3"], "opponent": ["X", "Y", "Z"],
        "game_date": pd.to_datetime(["2025-01-18", "2025-01-18", "2025-01-14"]),
        "sales_target": [1000, 1000, 1000], "stadium_capacity": [1000, 1000, 1000],
        "tickets_sold": [300, 800, 990], "remaining_inventory": [700, 200, 10],
    })
    forecast = pd.DataFrame({"game_id": ["G1", "G2", "G3"], "selected_forecast": [100.0, 100.0, 5.0]})
    return pace, upcoming, forecast


def test_checkpoint_target_uses_lead_time_share():
    pace, upcoming, forecast = _checkpoint_inputs()
    curve = typical_share_curve(pace)
    cp = checkpoint_table(upcoming, forecast, curve, date(2025, 1, 1)).set_index("game_id")
    # Checkpoint = Jan 8 -> 10 days before Jan 18 kickoff -> median share 0.6.
    assert cp.loc["G1", "checkpoint_lead_days"] == 10
    assert cp.loc["G1", "checkpoint_target"] == 600
    assert cp.loc["G1", "projected_at_checkpoint"] == 400
    assert cp.loc["G1", "forecast_status"] == "At risk"
    assert cp.loc["G2", "forecast_status"] == "On pace"
    # G3 kicks off before the 7-day checkpoint: lead 0 is not in the toy curve -> no target, but near capacity.
    assert cp.loc["G3", "forecast_status"] == "Near capacity"


def test_recommendations_have_required_fields_and_disclaimers():
    pace, upcoming, forecast = _checkpoint_inputs()
    cp = checkpoint_table(upcoming, forecast, typical_share_curve(pace), date(2025, 1, 1))
    live = pd.DataFrame({"game_id": "G1", "customer_id": ["a", "b"], "propensity_score": [0.9, 0.1],
                         "top_10pct": [True, False]})
    recs = build_recommendations(cp, live, date(2025, 1, 1))
    cats = {r["category"] for r in recs}
    assert {"Sales pace", "Inventory", "Fan outreach"} <= cats
    for r in recs:
        assert all(r[k] for k in ["trigger_metric", "threshold", "explanation", "action"])
    outreach = next(r for r in recs if r["category"] == "Fan outreach")
    assert "NOT" in outreach["explanation"]
