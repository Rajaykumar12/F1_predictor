from dataclasses import dataclass

import pytest
from click.testing import CliRunner

from main import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_help_lists_commands(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for cmd in ("fetch", "train", "predict-race", "score-race", "evaluate-position", "serve", "run-all"):
        assert cmd in result.output


def test_cli_fetch_calls_run_fetch(runner, monkeypatch):
    called = {}
    monkeypatch.setattr("pipeline.fetch.run_fetch", lambda cfg: called.setdefault("hit", True))
    result = runner.invoke(cli, ["fetch"])
    assert result.exit_code == 0
    assert called.get("hit") is True


def test_cli_train_passes_model(runner, monkeypatch):
    seen = {}
    monkeypatch.setattr("pipeline.train.run_training", lambda cfg, models=None: seen.setdefault("models", models))
    result = runner.invoke(cli, ["train", "--model", "position"])
    assert result.exit_code == 0
    assert seen["models"] == ["position"]


@dataclass
class _F:
    pred_rank: int
    driver: str
    team: str
    predicted_position: float
    raw_predicted_position: float
    confidence: float
    recent_form: dict


@dataclass
class _Result:
    forecasts: list
    race_label: str = "Testonia Grand Prix"
    using_real_qualifying: bool = True
    model_r2: float = 0.5
    lookback: int = 6
    as_of_round: int = 13
    prediction_date: str = "2026-09-19 15:00"
    next_race: str = "Testonia Grand Prix — real qualifying, last 6 races form"
    bias_applied: dict = None


def _fake_bundle():
    @dataclass
    class B:
        race_model: object
        metrics: dict
    return B(race_model=object(), metrics={"position": {"r2": 0.5}})


def _canned_result():
    return _Result(forecasts=[
        _F(1, "A DRIVER", "Team X", 1.4, 1.4, 0.4, {"avg_position": 2, "wins": 1, "podiums": 3, "dnfs": 0}),
        _F(2, "B DRIVER", "Team Y", 2.6, 2.6, 0.4, {"avg_position": 4, "wins": 0, "podiums": 1, "dnfs": 1}),
    ])


def test_cli_predict_race_prints_table(runner, monkeypatch):
    monkeypatch.setattr("pipeline.model_registry.load_bundle", lambda cfg: _fake_bundle())
    monkeypatch.setattr("pipeline.predict.predict_race", lambda *a, **k: _canned_result())
    monkeypatch.setattr("pipeline.fetch.fetch_upcoming_qualifying", lambda cfg, race_round=None: None)
    monkeypatch.setattr("pipeline.fetch.get_completed_race_rounds", lambda season: [1, 2, 3])
    monkeypatch.setattr("pipeline.orchestrate.compute_bias", lambda cfg: {})

    result = runner.invoke(cli, ["predict-race", "--round", "4", "--no-save"])
    assert result.exit_code == 0, result.output
    assert "A DRIVER" in result.output
    assert "PredPos" in result.output
    assert "Testonia Grand Prix" in result.output


def test_cli_predict_race_no_model_errors(runner, monkeypatch):
    @dataclass
    class B:
        race_model: object
        metrics: dict
    monkeypatch.setattr("pipeline.model_registry.load_bundle", lambda cfg: B(race_model=None, metrics={}))
    monkeypatch.setattr("pipeline.fetch.fetch_upcoming_qualifying", lambda cfg, race_round=None: None)
    monkeypatch.setattr("pipeline.fetch.get_completed_race_rounds", lambda season: [1, 2, 3])

    result = runner.invoke(cli, ["predict-race", "--round", "4", "--no-save"])
    assert result.exit_code != 0
    assert "train" in result.output.lower()


def test_cli_score_race_scorecard(runner, monkeypatch):
    canned = {
        "race_round": 13,
        "metrics": {"winner_correct": False, "podium_overlap": 1, "podium_exact": 1,
                    "top5": 2, "top10": 7, "spearman": 0.4, "position_mae": 4.6,
                    "position_rmse": 7.2, "n_drivers": 20},
        "rolling_scorecard": {"races": 3, "winner_hit_rate": 0.33, "position_mae_avg": 4.1, "spearman_avg": 0.5},
        "drift": {"retrain_recommended": True, "reasons": ["rolling position MAE 4.60 > 4.0"], "auto_retrain": False},
    }
    monkeypatch.setattr("pipeline.orchestrate.score_race", lambda cfg, rnd, fetch_if_missing=True: canned)
    result = runner.invoke(cli, ["score-race", "--round", "13"])
    assert result.exit_code == 0, result.output
    assert "Spearman" in result.output
    assert "retrain recommended" in result.output.lower()


def test_cli_race_weekend_prints_routine(runner):
    result = runner.invoke(cli, ["race-weekend"])
    assert result.exit_code == 0
    assert "predict-race" in result.output and "score-race" in result.output
