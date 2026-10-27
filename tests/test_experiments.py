"""Experiment statistics checked against hand-computed small examples."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.experiments import analyze_experiment, srm_check, two_proportion_test


def test_two_proportion_known_example():
    r = two_proportion_test(50, 1000, 65, 1000)
    assert r["rate_control"] == pytest.approx(0.05)
    assert r["rate_treatment"] == pytest.approx(0.065)
    assert r["diff_pp"] == pytest.approx(1.5)
    assert r["relative_lift"] == pytest.approx(0.30)
    p_pool = 115 / 2000
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (2 / 1000))
    assert r["z"] == pytest.approx(0.015 / se_pool)
    assert r["z"] == pytest.approx(1.4408, abs=1e-3)
    assert r["p_value"] == pytest.approx(0.1496, abs=1e-3)
    se = math.sqrt(0.05 * 0.95 / 1000 + 0.065 * 0.935 / 1000)
    assert r["ci_low"] == pytest.approx(0.015 - 1.959964 * se, abs=1e-6)
    assert r["ci_high"] == pytest.approx(0.015 + 1.959964 * se, abs=1e-6)


def test_zero_control_rate_has_no_relative_lift():
    r = two_proportion_test(0, 100, 5, 100)
    assert r["relative_lift"] is None
    assert r["normal_approx_ok"] is False


def test_identical_groups():
    r = two_proportion_test(0, 50, 0, 50)
    assert r["diff"] == 0 and r["p_value"] == 1.0


def test_srm_check():
    assert srm_check(500, 500)["p_value"] == pytest.approx(1.0)
    borderline = srm_check(450, 550)
    assert borderline["chi2"] == pytest.approx(10.0)
    assert not borderline["mismatch"]  # p ~ 0.0016 > 0.001
    assert srm_check(400, 600)["mismatch"]


def test_itt_counts_everyone_assigned():
    df = pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(8)],
        "group_name": ["control"] * 4 + ["treatment"] * 4,
        "purchased": [1, 0, 0, 0, 1, 1, 0, 0],
        "revenue": [10, 0, 0, 0, 20, 20, 0, 0],
    })
    out = analyze_experiment(df)
    g = out["groups"].set_index("group_name")
    assert g.loc["control", "assigned"] == 4 and g.loc["treatment", "conversion_rate"] == 0.5
    assert g.loc["treatment", "revenue_per_assigned"] == 10
    assert "simulated" in out["interpretation"]
    with pytest.raises(ValueError):
        analyze_experiment(pd.concat([df, df.iloc[[0]]]))
