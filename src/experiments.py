"""A/B test statistics for the simulated email experiment (intention-to-treat).

Primary outcome: binary purchase within the fixed observation window.
* Difference in conversion rates with a 95% Wald (unpooled) confidence interval.
* Two-sided two-proportion z-test using the pooled standard error under H0.
* Sample-ratio-mismatch (SRM) check: chi-square goodness of fit vs. 50/50.

Assumptions: customers are randomized independently, each customer is counted
once in the group they were assigned to (regardless of opening the email), and
counts are large enough for the normal approximation (checked explicitly).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from src.config import SRM_ALPHA


def two_proportion_test(conv_c: int, n_c: int, conv_t: int, n_t: int, alpha: float = 0.05) -> dict:
    if min(n_c, n_t) <= 0:
        raise ValueError("Both groups need at least one assigned customer.")
    p_c, p_t = conv_c / n_c, conv_t / n_t
    diff = p_t - p_c
    z_crit = stats.norm.ppf(1 - alpha / 2)
    se_unpooled = math.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    p_pool = (conv_c + conv_t) / (n_c + n_t)
    se_pooled = math.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    if se_pooled > 0:
        z = diff / se_pooled
        p_value = float(2 * stats.norm.sf(abs(z)))
    else:  # both groups all-0 or all-1: no evidence of a difference
        z, p_value = 0.0, 1.0
    # Minimum detectable effect (80% power, two-sided alpha) at the pooled rate.
    mde = (z_crit + stats.norm.ppf(0.80)) * se_pooled
    return {
        "n_control": n_c,
        "n_treatment": n_t,
        "conversions_control": conv_c,
        "conversions_treatment": conv_t,
        "rate_control": p_c,
        "rate_treatment": p_t,
        "diff": diff,
        "diff_pp": diff * 100,
        "ci_low": diff - z_crit * se_unpooled,
        "ci_high": diff + z_crit * se_unpooled,
        "relative_lift": diff / p_c if p_c > 0 else None,
        "z": float(z),
        "p_value": p_value,
        "alpha": alpha,
        "mde_80_power": float(mde),
        "normal_approx_ok": min(conv_c, n_c - conv_c, conv_t, n_t - conv_t) >= 10,
    }


def srm_check(n_c: int, n_t: int, expected_share_treatment: float = 0.5, alpha: float = SRM_ALPHA) -> dict:
    total = n_c + n_t
    expected = [total * (1 - expected_share_treatment), total * expected_share_treatment]
    chi2, p = stats.chisquare([n_c, n_t], f_exp=expected)
    return {
        "observed_share_treatment": n_t / total if total else float("nan"),
        "expected_share_treatment": expected_share_treatment,
        "chi2": float(chi2),
        "p_value": float(p),
        "mismatch": bool(p < alpha),
        "alpha": alpha,
    }


def summarize_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Per-group ITT summary: everyone assigned is in the denominator."""
    out = (
        df.groupby("group_name")
        .agg(assigned=("customer_id", "nunique"), conversions=("purchased", "sum"), revenue=("revenue", "sum"))
        .reindex(["control", "treatment"])
        .fillna(0)
    )
    out["conversion_rate"] = out["conversions"] / out["assigned"]
    out["revenue_per_assigned"] = out["revenue"] / out["assigned"]
    return out.reset_index()


def analyze_experiment(df: pd.DataFrame) -> dict:
    if df["customer_id"].duplicated().any():
        raise ValueError("Each customer must be assigned exactly once.")
    groups = summarize_groups(df).set_index("group_name")
    test = two_proportion_test(
        int(groups.loc["control", "conversions"]),
        int(groups.loc["control", "assigned"]),
        int(groups.loc["treatment", "conversions"]),
        int(groups.loc["treatment", "assigned"]),
    )
    srm = srm_check(int(groups.loc["control", "assigned"]), int(groups.loc["treatment", "assigned"]))
    return {"groups": groups.reset_index(), "test": test, "srm": srm, "interpretation": interpret(test, srm)}


def interpret(test: dict, srm: dict) -> str:
    lo, hi = test["ci_low"] * 100, test["ci_high"] * 100
    parts = []
    if srm["mismatch"]:
        parts.append(
            "Warning: the group sizes differ more than randomization would explain (sample-ratio mismatch). "
            "Investigate the assignment pipeline before trusting the result."
        )
    parts.append(
        f"Treatment converted at {test['rate_treatment']:.2%} vs {test['rate_control']:.2%} for control, "
        f"a difference of {test['diff_pp']:+.2f} percentage points (95% CI {lo:+.2f} to {hi:+.2f} pp)."
    )
    if test["ci_low"] > 0:
        parts.append(
            "The interval excludes zero, so the data are consistent with a positive effect, but its plausible size "
            f"ranges from {lo:.2f} to {hi:.2f} pp. Weigh the low end against the cost of personalization before "
            "rolling out."
        )
        if test["ci_low"] < 0.25 * test["diff"]:
            parts.append(
                "The lower bound sits close to zero, so the evidence is modest; replicating the test before a "
                "full rollout would be prudent."
            )
    elif test["ci_high"] < 0:
        parts.append("The interval is entirely below zero: the personalized email appears to perform worse.")
    else:
        parts.append(
            f"The interval includes zero (p = {test['p_value']:.3f}), so we cannot distinguish the effect from "
            "random variation. The result is inconclusive rather than proof of no effect: effects as large as "
            f"{max(abs(lo), abs(hi)):.2f} pp remain plausible. A larger sample would narrow the interval "
            f"(this test had ~80% power only for effects of about {test['mde_80_power'] * 100:.2f} pp)."
        )
    parts.append("This is a simulated experiment; it does not demonstrate real business impact.")
    return " ".join(parts)
