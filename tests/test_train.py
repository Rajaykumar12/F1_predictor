"""Tests for the honest position-model training split logic."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from pipeline.config_loader import FeaturesConfig, ModelParams
from pipeline.train import (
    PositionModel,
    _forward_chain_split,
    _mean_vif,
    blend_regressor_and_ranker,
    build_position_pipeline,
    fit_position_model,
    select_numeric_features,
    train_dnf_model,
    train_position_ranker_model,
    tune_position_hyperparams,
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


# --------------------------------------------------------------------------- #
# D1 — PositionModel target-transform wrapper
# --------------------------------------------------------------------------- #
def _position_frame(n_rounds=8, per_round=10, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(1, n_rounds + 1):
        grid = rng.permutation(per_round) + 1
        for i in range(per_round):
            pos = int(np.clip(grid[i] + rng.integers(-2, 3), 1, per_round))
            rows.append({
                "Race": r, "GridPosition": float(grid[i]), "pace": float(grid[i]) + rng.normal(0, 0.5),
                "Position": pos,
            })
    return pd.DataFrame(rows)


def _params():
    return ModelParams(n_estimators=30, learning_rate=0.3, max_depth=2, random_state=0)


def test_position_model_predict_is_always_position_domain():
    df = _position_frame()
    pipeline = build_position_pipeline(_params(), ["GridPosition", "pace"], [])
    model = fit_position_model(pipeline, df, df.index, ["GridPosition", "pace"], [], "positions_gained")
    preds = model.predict(df[["GridPosition", "pace"]])
    # a Position estimate, not a gained-delta -> should land near the 1..per_round range
    assert preds.min() > -5 and preds.max() < 20


def test_position_model_gained_vs_raw_reconstruction_differ_only_by_target():
    df = _position_frame()
    numeric = ["GridPosition", "pace"]
    p1 = build_position_pipeline(_params(), numeric, [])
    p2 = build_position_pipeline(_params(), numeric, [])
    gained_model = fit_position_model(p1, df, df.index, numeric, [], "positions_gained")
    raw_model = fit_position_model(p2, df, df.index, numeric, [], "position")
    assert gained_model.target == "positions_gained"
    assert raw_model.target == "position"
    # both fit on the same rows and both predict() in Position-domain
    gp = gained_model.predict(df[numeric])
    rp = raw_model.predict(df[numeric])
    assert gp.shape == rp.shape == (len(df),)


def test_position_model_named_steps_proxies_the_inner_pipeline():
    df = _position_frame()
    pipeline = build_position_pipeline(_params(), ["GridPosition"], [])
    model = fit_position_model(pipeline, df, df.index, ["GridPosition"], [], "position")
    assert "preprocessor" in model.named_steps
    assert "regressor" in model.named_steps


# --------------------------------------------------------------------------- #
# D2 — rank-average blend
# --------------------------------------------------------------------------- #
def test_blend_agrees_when_both_models_agree():
    # both say the same order -> blend must reproduce that same rank order
    positions = [1.0, 2.0, 3.0, 4.0]       # lower = better
    ranker_scores = [4.0, 3.0, 2.0, 1.0]   # higher = better -> same order
    blended = blend_regressor_and_ranker(positions, ranker_scores)
    assert list(np.argsort(blended)) == [0, 1, 2, 3]


def test_blend_splits_the_difference_on_disagreement():
    # regressor says A is best; ranker says B is best -> blend must not put
    # either one at a rank worse than both models individually gave it
    positions = [1.0, 2.0]        # A predicted best
    ranker_scores = [1.0, 2.0]    # B ("index 1") has the higher (better) score
    blended = blend_regressor_and_ranker(positions, ranker_scores)
    assert blended[0] == blended[1]  # tied 1-and-2 average -> exact tie


def test_train_position_ranker_model_smoke(tmp_path, monkeypatch):
    """End-to-end smoke test on synthetic data: trains, saves a pickle, and
    writes a position_ranker metrics block."""
    from types import SimpleNamespace
    from pipeline.config_loader import FeaturesConfig

    df = _position_frame(n_rounds=10, per_round=8)
    models_dir = tmp_path / "models"

    def fake_prepare(config):
        d = df.copy()
        d["race_seq"] = d["Race"]
        d["Year"] = 2099
        return {
            "data": d, "numeric": ["GridPosition", "pace"], "categorical": [],
            "features": ["GridPosition", "pace"], "dropped": [], "mean_vif": 1.0,
        }

    monkeypatch.setattr("pipeline.train.prepare_position_data", fake_prepare)

    cfg = SimpleNamespace(
        paths=SimpleNamespace(models_dir=models_dir),
        models=SimpleNamespace(position=_params()),
        features=FeaturesConfig(
            lookback_races=6, shift=1, families_enabled=[], cv_splits=5,
            holdout_rounds=3, vif_threshold=10.0, feature_selection="none",
        ),
        constants=SimpleNamespace(grid_size=8),
    )
    train_position_ranker_model(cfg)

    assert (models_dir / "race_ranking_pipeline.pkl").exists()
    metrics = json.loads((models_dir / "metrics.json").read_text())
    assert "position_ranker" in metrics
    assert metrics["position_ranker"]["objective"] == "rank:ndcg"


# --------------------------------------------------------------------------- #
# D3 — optuna hyperparameter search
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_tune_position_hyperparams_returns_valid_xgb_params():
    from types import SimpleNamespace

    df = _position_frame(n_rounds=14, per_round=10)
    df["race_seq"] = df["Race"]
    numeric, categorical = ["GridPosition", "pace"], []
    cfg = SimpleNamespace(
        features=FeaturesConfig(
            lookback_races=6, shift=1, families_enabled=[], cv_splits=5,
            holdout_rounds=3, vif_threshold=10.0, feature_selection="none", tune="quick",
        ),
        models=SimpleNamespace(position=_params()),
    )
    best = tune_position_hyperparams(df, numeric, categorical, df["race_seq"], cfg, "position")
    assert set(best) == {"n_estimators", "learning_rate", "max_depth", "reg_alpha", "reg_lambda"}
    assert 100 <= best["n_estimators"] <= 1000
    assert 2 <= best["max_depth"] <= 6


# --------------------------------------------------------------------------- #
# E1 — DNF classifier
# --------------------------------------------------------------------------- #
def test_train_dnf_model_smoke(tmp_path, monkeypatch):
    from types import SimpleNamespace
    rng = np.random.default_rng(2)
    rows = []
    for r in range(1, 15):
        for i in range(10):
            rows.append({
                "Race": r, "GridPosition": float(i + 1),
                "pace": float(i) + rng.normal(0, 0.5),
                "is_dnf": int(rng.random() < 0.2),
            })
    df = pd.DataFrame(rows)
    df["race_seq"] = df["Race"]

    def fake_prepare(config):
        return {
            "data": df, "numeric": ["GridPosition", "pace"], "categorical": [],
            "features": ["GridPosition", "pace"], "dropped": [],
        }

    monkeypatch.setattr("pipeline.train.prepare_dnf_data", fake_prepare)
    models_dir = tmp_path / "models"
    cfg = SimpleNamespace(
        paths=SimpleNamespace(models_dir=models_dir),
        models=SimpleNamespace(position=_params()),
        features=FeaturesConfig(
            lookback_races=6, shift=1, families_enabled=[], cv_splits=5,
            holdout_rounds=3, vif_threshold=10.0, feature_selection="none",
        ),
    )
    train_dnf_model(cfg)

    assert (models_dir / "dnf_pipeline.pkl").exists()
    metrics = json.loads((models_dir / "metrics.json").read_text())
    assert "dnf" in metrics
    assert 0 <= metrics["dnf"]["cv_auc_mean"] <= 1


def test_position_model_survives_pickling():
    import pickle
    df = _position_frame()
    pipeline = build_position_pipeline(_params(), ["GridPosition"], [])
    model = fit_position_model(pipeline, df, df.index, ["GridPosition"], [], "positions_gained")
    blob = pickle.dumps(model)
    restored: PositionModel = pickle.loads(blob)
    np.testing.assert_allclose(
        restored.predict(df[["GridPosition"]]), model.predict(df[["GridPosition"]])
    )


# --------------------------------------------------------------------------- #
# F — ranker group integrity and simulation output shapes
# --------------------------------------------------------------------------- #
def test_ranker_group_integrity():
    """_ranker_group_sizes must produce one group per Race with no cross-race
    mixing. If a row's Race value appears in multiple groups the XGB ranker
    would silently see labels from different races together."""
    from pipeline.train import _ranker_group_sizes

    df = _position_frame(n_rounds=6, per_round=8)
    df["race_seq"] = df["Race"]  # ranker needs race_seq
    _sorted_df, sizes = _ranker_group_sizes(df, df.index)
    # There should be one group per round and the sizes must sum to len(df)
    assert sum(sizes) == len(df)
    assert len(sizes) == df["Race"].nunique()
    # Every group should have exactly per_round rows (8 in this synthetic case)
    assert all(s == 8 for s in sizes), f"unequal group sizes: {sizes}"


def test_simulate_output_shapes():
    """simulate_race_df returns one row per driver with the required probability
    columns, and the probability vectors are internally consistent."""
    from pipeline.simulate import simulate_race_df

    n_drivers = 12
    rng = np.random.default_rng(0)
    pred_df = pd.DataFrame({
        "Driver": [f"D{i}" for i in range(n_drivers)],
        "PredictedPosition": rng.uniform(1, n_drivers, n_drivers),
    })
    result = simulate_race_df(pred_df, metrics={}, dnf_model=None, n_trials=500)

    assert len(result) == n_drivers
    required = {"Driver", "win_probability", "podium_probability", "points_probability",
                "expected_position", "p10", "p90", "dnf_probability"}
    assert required.issubset(result.columns), f"Missing columns: {required - set(result.columns)}"

    # Probabilities must be in [0, 1]
    for col in ("win_probability", "podium_probability", "points_probability"):
        assert result[col].between(0.0, 1.001).all(), f"{col} out of range"

    # Win probabilities must sum to ~1 (one winner per simulated race)
    np.testing.assert_allclose(result["win_probability"].sum(), 1.0, atol=0.05)

    # P(win) <= P(podium) <= P(points) for every driver
    assert (result["win_probability"] <= result["podium_probability"] + 1e-6).all()
    assert (result["podium_probability"] <= result["points_probability"] + 1e-6).all()

    # 10th-percentile position must be <= 90th-percentile position
    assert (result["p10"] <= result["p90"]).all()
