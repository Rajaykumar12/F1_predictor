"""Tests for the honest position-model training split logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from pipeline.train import (
    _forward_chain_split,
    _mean_vif,
    select_numeric_features,
)


def _frame(n_rounds=10, per_round=18, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(1, n_rounds + 1):
        for _ in range(per_round):
            g = rng.integers(1, per_round + 1)
            rows.append({
                "Race": r,
                "Position": int(np.clip(g + rng.integers(-3, 4), 1, per_round)),
                "grid": float(g),
                "form": g + rng.normal(0, 2),
                "dup": g + rng.normal(0, 2),   # collinear-ish with form
                "noise": rng.normal(),
            })
    df = pd.DataFrame(rows)
    df["dup"] = df["form"] + rng.normal(0, 1e-3, len(df))  # near-duplicate of form
    return df


# --------------------------------------------------------------------------- #
# forward-chaining holdout
# --------------------------------------------------------------------------- #
def test_forward_chain_split_holds_out_the_last_k_rounds():
    df = _frame(n_rounds=10).reset_index(drop=True)
    train_idx, test_idx = _forward_chain_split(df, holdout_rounds=3)

    train_rounds = set(df.loc[train_idx, "Race"])
    test_rounds = set(df.loc[test_idx, "Race"])
    assert test_rounds == {8, 9, 10}
    assert train_rounds == {1, 2, 3, 4, 5, 6, 7}
    # every training round is strictly earlier than every test round
    assert max(train_rounds) < min(test_rounds)
    # partition, no overlap, nothing lost
    assert train_rounds.isdisjoint(test_rounds)
    assert len(train_idx) + len(test_idx) == len(df)


def test_forward_chain_split_shrinks_holdout_when_few_rounds():
    df = _frame(n_rounds=4).reset_index(drop=True)
    train_idx, test_idx = _forward_chain_split(df, holdout_rounds=3)
    # with only 4 rounds a 3-round holdout would leave 1 training round — shrink it
    assert len(set(df.loc[train_idx, "Race"])) >= 2
    assert len(test_idx) > 0


# --------------------------------------------------------------------------- #
# GroupKFold(by Race) never splits a race across a fold boundary
# --------------------------------------------------------------------------- #
def test_groupkfold_by_race_never_shares_a_round():
    df = _frame(n_rounds=10).reset_index(drop=True)
    groups = df["Race"]
    gkf = GroupKFold(n_splits=5)
    for train_idx, test_idx in gkf.split(df, df["Position"], groups):
        tr = set(groups.iloc[train_idx])
        te = set(groups.iloc[test_idx])
        assert tr.isdisjoint(te), "a Race appeared in both train and test"


# --------------------------------------------------------------------------- #
# feature selection
# --------------------------------------------------------------------------- #
def test_select_none_keeps_everything():
    df = _frame()
    kept, dropped = select_numeric_features(
        df, ["grid", "form", "dup", "noise"], "Position", "none", 10.0
    )
    assert kept == ["grid", "form", "dup", "noise"]
    assert dropped == []


def test_select_vif_drops_the_near_duplicate():
    df = _frame(n_rounds=14)
    kept, dropped = select_numeric_features(
        df, ["grid", "form", "dup", "noise"], "Position", "vif", 10.0
    )
    # form and dup are ~identical -> exactly one of them must go
    assert ("form" in dropped) ^ ("dup" in dropped)
    assert "noise" in kept
    assert len(kept) >= 3


def test_select_lasso_returns_subset_and_drops_noise_biased():
    df = _frame(n_rounds=16)
    kept, dropped = select_numeric_features(
        df, ["grid", "form", "dup", "noise"], "Position", "lasso", 10.0
    )
    assert set(kept) | set(dropped) == {"grid", "form", "dup", "noise"}
    assert set(kept).isdisjoint(dropped)
    assert len(kept) >= 3


def test_mean_vif_is_low_for_orthogonal_and_high_for_collinear():
    df = _frame(n_rounds=14)
    low = _mean_vif(df, ["grid", "noise"])
    high = _mean_vif(df, ["grid", "form", "dup", "noise"])
    assert low < 5
    assert high > low
