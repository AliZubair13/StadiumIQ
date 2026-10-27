"""Propensity snapshots: eligibility, outcome windows, point-in-time features, splits."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from src.config import TEST_SEASON, TRAIN_SEASONS, VALIDATION_SEASON
from src.features import PROPENSITY_NUMERIC, build_propensity_snapshots, snapshot_date_for
from src.train_propensity import FEATURES, live_game_id, season_split


@pytest.fixture(scope="module")
def snaps(loaded, as_of):
    return build_propensity_snapshots(loaded["games"], loaded["customers"], loaded["transactions"],
                                      loaded["engagement"], as_of)


def _game(loaded, as_of, season=2023, idx=4):
    g = loaded["games"]
    return g[g.season == season].sort_values("game_date").iloc[idx]


def test_eligibility_and_labels_match_definition(snaps, loaded):
    tx = loaded["transactions"]
    cust = loaded["customers"].set_index("customer_id")
    for gid, part in snaps.groupby("game_id"):
        snap = part["snapshot_date"].iloc[0]
        kick = part["game_date"].iloc[0]
        assert snap == kick - timedelta(days=14)
        gtx = tx[tx.game_id == gid]
        bought_before = set(gtx.loc[gtx.purchase_date <= snap, "customer_id"])
        assert not set(part["customer_id"]) & bought_before
        assert (cust.loc[part["customer_id"], "signup_date"] <= snap).all()
        positives = set(gtx.loc[(gtx.purchase_date > snap) & (gtx.purchase_date <= kick), "customer_id"])
        assert set(part.loc[part.label == 1, "customer_id"]) == positives & set(part["customer_id"])
    assert not snaps.duplicated(["game_id", "customer_id"]).any()


def test_only_fully_observed_outcome_windows(snaps, as_of):
    assert (snaps["game_date"] <= pd.Timestamp(as_of)).all()


def test_features_ignore_future_transactions_and_engagement(loaded, as_of):
    g = _game(loaded, as_of)
    snap = snapshot_date_for(g.game_date)
    args = (loaded["games"], loaded["customers"])
    base = build_propensity_snapshots(*args, loaded["transactions"], loaded["engagement"], as_of, [g.game_id])
    cid = base["customer_id"].iloc[0]
    # Inject activity strictly after the snapshot (another game + engagement burst).
    other = loaded["games"][loaded["games"].game_date > snap]["game_id"].iloc[0]
    extra_tx = pd.DataFrame([{"transaction_id": 10**9, "customer_id": cid, "game_id": other,
                              "purchase_date": snap + timedelta(days=1), "ticket_quantity": 4,
                              "price_per_ticket": 500.0, "revenue": 2000.0}])
    extra_eng = pd.DataFrame([{"engagement_id": 10**9 + i, "customer_id": cid,
                               "engagement_date": snap + timedelta(days=1 + i % 5),
                               "channel": "Web", "engagement_type": "ticket_page_view"} for i in range(20)])
    tx2 = pd.concat([loaded["transactions"], extra_tx], ignore_index=True)
    eng2 = pd.concat([loaded["engagement"], extra_eng], ignore_index=True)
    after = build_propensity_snapshots(*args, tx2, eng2, as_of, [g.game_id])
    cols = ["customer_id"] + PROPENSITY_NUMERIC
    pd.testing.assert_frame_equal(base[cols], after[cols])

    # Sanity: the same activity *before* the snapshot does change the features.
    extra_tx["purchase_date"] = snap - timedelta(days=1)
    extra_eng["engagement_date"] = snap - timedelta(days=1)
    before = build_propensity_snapshots(*args, pd.concat([loaded["transactions"], extra_tx]),
                                        pd.concat([loaded["engagement"], extra_eng]), as_of, [g.game_id])
    row = before.set_index("customer_id").loc[cid]
    assert row["engagement_30d"] >= 20
    assert row["prev_total_spend"] >= 2000


def test_no_ids_or_labels_in_model_features():
    assert "customer_id" not in FEATURES and "game_id" not in FEATURES
    assert "label" not in FEATURES


def test_season_split_is_chronological(snaps):
    split = season_split(snaps)
    assert set(split["train"]["season"]) == set(TRAIN_SEASONS)
    assert set(split["validation"]["season"]) == {VALIDATION_SEASON}
    assert set(split["test"]["season"]) == {TEST_SEASON}
    assert split["train"]["game_date"].max() < split["validation"]["game_date"].min()
    assert split["validation"]["game_date"].max() < split["test"]["game_date"].min()


def test_live_snapshot_has_no_label_and_uses_as_of(loaded, as_of):
    gid = live_game_id(loaded["games"], as_of)
    assert gid is not None
    live = build_propensity_snapshots(loaded["games"], loaded["customers"], loaded["transactions"],
                                      loaded["engagement"], as_of, [gid], require_label=False)
    if not live.empty:  # empty only if the small simulation sold out the game
        assert "label" not in live
        assert (live["snapshot_date"] == pd.Timestamp(as_of)).all()
    # The same game is excluded when labels are required (window not observed yet).
    labeled = build_propensity_snapshots(loaded["games"], loaded["customers"], loaded["transactions"],
                                         loaded["engagement"], as_of, [gid])
    assert labeled.empty
