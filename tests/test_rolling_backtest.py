"""Tests for the walk-forward season scorecard (docs/prediction-improvement-plan.md A2)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rolling_backtest as rb  # noqa: E402
from pipeline.config_loader import ModelParams  # noqa: E402


def _synthetic_prepared(n_rounds=6, drivers=("A", "B", "C", "D", "E", "F"), year=2099):
    rng = np.random.default_rng(0)
    rows = []
    for r in range(1, n_rounds + 1):
        grid = rng.permutation(len(drivers)) + 1
        for i, drv in enumerate(drivers):
            noise = rng.integers(-1, 2)
            rows.append({
                "Year": year,
                "Race": r,
                "Driver": drv,
                "Team": "T1" if i % 2 == 0 else "T2",
                "grid_strength": float(grid[i]),
                "GridPosition": float(grid[i]),  # D1 needs this to reconstruct Position
                "Position": int(np.clip(grid[i] + noise, 1, len(drivers))),
            })
    data = pd.DataFrame(rows)
    data["race_seq"] = data["Race"]  # single synthetic season -> race_seq == Race
    return {
        "data": data,
        "numeric": ["grid_strength", "GridPosition"],
        "categorical": ["Driver", "Team"],
        "features": ["grid_strength", "GridPosition", "Driver", "Team"],
        "dropped": [],
        "mean_vif": 1.0,
    }


def _fake_config(min_lookback=2, position_target="positions_gained"):
    return SimpleNamespace(
        pipeline=SimpleNamespace(min_lookback=min_lookback, season=2099),
        models=SimpleNamespace(position=ModelParams(
            n_estimators=20, learning_rate=0.3, max_depth=2, random_state=0
        )),
        features=SimpleNamespace(position_target=position_target),
        paths=SimpleNamespace(plots_dir=Path("/tmp/does_not_matter")),
    )


def test_run_rolling_backtest_skips_rounds_without_enough_history(monkeypatch):
    prepared = _synthetic_prepared(n_rounds=5)
    monkeypatch.setattr(rb, "prepare_position_data", lambda config: prepared)

    cfg = _fake_config(min_lookback=2)
    scorecard = rb.run_rolling_backtest(cfg, min_train_races=2)

    # rounds 1, 2 can never have >= 2 prior races; 3, 4, 5 can
    assert sorted(scorecard["round"]) == [3, 4, 5]
    assert (scorecard["n_train_races"] >= 2).all()


def test_run_rolling_backtest_never_trains_on_the_predicted_round_or_later(monkeypatch):
    prepared = _synthetic_prepared(n_rounds=6)
    monkeypatch.setattr(rb, "prepare_position_data", lambda config: prepared)

    calls = []
    real_predict_round = rb._predict_round

    def spy(pipeline, data, numeric, categorical, train_idx, test_idx, position_target="positions_gained"):
        train_rounds = set(data.loc[train_idx, "Race"])
        test_round = data.loc[test_idx, "Race"].iloc[0]
        calls.append((max(train_rounds), test_round))
        return real_predict_round(pipeline, data, numeric, categorical, train_idx, test_idx, position_target)

    monkeypatch.setattr(rb, "_predict_round", spy)
    cfg = _fake_config(min_lookback=2)
    rb.run_rolling_backtest(cfg, min_train_races=2)

    assert calls  # at least one round was backtested
    for max_train_round, test_round in calls:
        assert max_train_round < test_round


def test_season_summary_empty_scorecard():
    assert rb._season_summary(pd.DataFrame()) == {}


def test_season_summary_averages_present_columns():
    sc = pd.DataFrame([
        {"position_mae": 2.0, "spearman": 0.5, "winner_correct": True,
         "podium_overlap": 2, "winner_logloss": 1.0, "podium_brier": 0.1, "points_brier": 0.2},
        {"position_mae": 4.0, "spearman": 0.7, "winner_correct": False,
         "podium_overlap": 1, "winner_logloss": 2.0, "podium_brier": 0.2, "points_brier": 0.3},
    ])
    summary = rb._season_summary(sc)
    assert summary["rounds_backtested"] == 2
    assert summary["position_mae"] == pytest.approx(3.0)
    assert summary["winner_hit_rate"] == pytest.approx(0.5)
