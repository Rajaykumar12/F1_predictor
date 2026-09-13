import json
from dataclasses import dataclass

import pandas as pd
import pytest

from pipeline import feedback


# --------------------------------------------------------------------------- #
# score_prediction
# --------------------------------------------------------------------------- #
def _pred(order):
    return pd.DataFrame(
        {"PredRank": range(1, len(order) + 1), "Driver": order,
         "PredictedPosition": [float(i) for i in range(1, len(order) + 1)]}
    )


def _actual(order):
    return pd.DataFrame({"Driver": order, "Position": range(1, len(order) + 1)})


def test_score_prediction_perfect():
    drivers = [f"D{i}" for i in range(1, 13)]
    m = feedback.score_prediction(_pred(drivers), _actual(drivers))
    assert m["winner_correct"] is True
    assert m["podium_overlap"] == 3 and m["podium_exact"] == 3
    assert m["top5"] == 5 and m["top10"] == 10
    assert m["spearman"] == pytest.approx(1.0)
    assert m["position_mae"] == 0.0
    assert m["unmatched"] == []


def test_score_prediction_partial():
    pred = _pred(["A", "B", "C", "D", "E"])
    actual = _actual(["B", "A", "C", "E", "D"])  # winner wrong, C exact
    m = feedback.score_prediction(pred, actual)
    assert m["winner_correct"] is False
    assert m["podium_overlap"] == 3      # {A,B,C} both sides
    assert m["podium_exact"] == 1        # only C in place
    assert m["n_drivers"] == 5


def test_score_prediction_few_drivers_nan_rank_metrics():
    pred = _pred(["A", "B"])
    actual = _actual(["A", "B"])
    m = feedback.score_prediction(pred, actual)
    assert m["winner_correct"] is True          # set metrics still computed
    assert m["position_mae"] != m["position_mae"]  # NaN (fewer than 3 matched)


# --------------------------------------------------------------------------- #
# sharp-end metrics — winner_logloss / podium_brier / points_brier
# --------------------------------------------------------------------------- #
def test_score_prediction_perfect_order_has_low_winner_logloss():
    drivers = [f"D{i}" for i in range(1, 13)]
    m = feedback.score_prediction(_pred(drivers), _actual(drivers))
    # predicted winner == actual winner and the field is a well-formed probability
    assert m["winner_logloss"] >= 0
    assert 0 <= m["podium_brier"] <= 1
    assert 0 <= m["points_brier"] <= 1


def test_score_prediction_wrong_winner_worse_logloss_than_right_winner():
    drivers = [f"D{i}" for i in range(1, 6)]
    right = feedback.score_prediction(_pred(drivers), _actual(drivers))
    # swap the predicted P1/P2 so the model's top pick is wrong
    swapped = _pred(drivers)
    swapped.loc[[0, 1], "Driver"] = swapped.loc[[1, 0], "Driver"].to_numpy()
    wrong = feedback.score_prediction(swapped, _actual(drivers))
    assert wrong["winner_logloss"] > right["winner_logloss"]


def test_sharp_end_metrics_by_race_averages_across_races():
    pred = [1, 2, 3, 4] + [1, 2, 3, 4]
    actual = [1, 2, 3, 4] + [1, 2, 3, 4]
    race = [1, 1, 1, 1, 2, 2, 2, 2]
    out = feedback.sharp_end_metrics_by_race(pred, actual, race)
    assert out["winner_logloss"] >= 0
    assert 0 <= out["podium_brier"] <= 1
    assert 0 <= out["points_brier"] <= 1


def test_per_driver_errors_sign():
    pred = _pred(["A", "B", "C", "D", "E"])            # A predicted rank 1
    actual = _actual(["E", "D", "C", "B", "A"])        # A actually finished 5th
    errs = {e["driver"]: e["signed_error"] for e in feedback.per_driver_errors(pred, actual)}
    assert errs["A"] == 1 - 5  # -4: model placed A too high
    assert errs["E"] == 5 - 1  # +4: model placed E too low


# --------------------------------------------------------------------------- #
# driver_bias
# --------------------------------------------------------------------------- #
def test_driver_bias_weights_recent_more():
    scored = [
        {"round": 1, "per_driver": [{"driver": "X", "signed_error": 0}]},
        {"round": 2, "per_driver": [{"driver": "X", "signed_error": 6}]},
    ]
    bias = feedback.driver_bias(scored, halflife=1.0, max_abs=10.0)
    # latest round (error 6) has weight 1, older (error 0) weight 0.5 -> 6*1 / 1.5 = 4.0
    assert bias["X"] == pytest.approx(4.0)


def test_driver_bias_clip():
    scored = [{"round": r, "per_driver": [{"driver": "X", "signed_error": 10}]} for r in (1, 2, 3)]
    bias = feedback.driver_bias(scored, halflife=3.0, max_abs=3.0)
    assert bias["X"] == 3.0


# --------------------------------------------------------------------------- #
# apply_bias_correction
# --------------------------------------------------------------------------- #
@dataclass
class _F:
    pred_rank: int
    driver: str
    team: str
    predicted_position: float
    raw_predicted_position: float
    confidence: float = 0.5
    recent_form: dict = None


def test_apply_bias_correction_reorders():
    fc = [
        _F(1, "A", "T", 2.0, 2.0),
        _F(2, "B", "T", 3.0, 3.0),
    ]
    # bias is subtracted from the raw position; a negative offset pushes A down
    out = feedback.apply_bias_correction(fc, {"A": -5.0})
    assert out[0].driver == "B"
    assert out[1].driver == "A"
    assert out[1].pred_rank == 2
    assert out[1].predicted_position == pytest.approx(7.0)

    # a positive offset pulls a chronically under-rated driver up (floored at 1.0)
    up = feedback.apply_bias_correction(fc, {"B": 5.0})
    assert up[0].driver == "B" and up[0].predicted_position == 1.0


# --------------------------------------------------------------------------- #
# should_retrain
# --------------------------------------------------------------------------- #
@dataclass
class _FB:
    enabled: bool = True
    window_races: int = 5
    min_scored_races: int = 5
    max_position_mae: float = 4.0
    min_spearman: float = 0.30
    min_winner_hit_rate: float = 0.15


def _hist(n, mae, spearman=0.8, winner=True):
    return [
        {"position_mae": mae, "position_rmse": mae, "spearman": spearman,
         "winner_correct": winner, "podium_overlap": 2, "top5": 3, "top10": 6}
        for _ in range(n)
    ]


def test_should_retrain_gated_by_min_scored():
    ok, reasons = feedback.should_retrain(_hist(3, mae=9.0), _FB())
    assert ok is False
    assert "need 5" in reasons[0]


def test_should_retrain_triggers_on_mae():
    ok, reasons = feedback.should_retrain(_hist(5, mae=9.0), _FB())
    assert ok is True
    assert any("MAE" in r for r in reasons)


def test_should_retrain_clean_history():
    ok, reasons = feedback.should_retrain(_hist(6, mae=2.0), _FB())
    assert ok is False
    assert reasons == []


# --------------------------------------------------------------------------- #
# stores round-trip
# --------------------------------------------------------------------------- #
def test_history_roundtrip(tmp_path):
    p = tmp_path / "sub" / "score_history.json"
    for i in range(3):
        feedback.append_history(p, {"round": i, "position_mae": float(i)})
    hist = feedback.load_history(p)
    assert [h["round"] for h in hist] == [0, 1, 2]


def test_load_history_missing(tmp_path):
    assert feedback.load_history(tmp_path / "nope.json") == []


def test_append_history_replaces_same_round(tmp_path):
    p = tmp_path / "score_history.json"
    feedback.append_history(p, {"season": 2026, "round": 5, "position_mae": 9.0})
    feedback.append_history(p, {"season": 2026, "round": 6, "position_mae": 2.0})
    feedback.append_history(p, {"season": 2026, "round": 5, "position_mae": 3.0})  # re-score r5
    hist = feedback.load_history(p)
    assert [(h["round"], h["position_mae"]) for h in hist] == [(5, 3.0), (6, 2.0)]


def _tmp_config(tmp_path):
    import yaml
    from pipeline.config_loader import get_config
    data = {
        "paths": {"data_dir": "data", "models_dir": "models", "cache_dir": "cache", "plots_dir": "plots"},
        "pipeline": {"season": 2026, "lookback_races": 6, "min_lookback": 3, "max_lookback": 12, "api_sleep_seconds": 2},
        "models": {k: {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 3, "random_state": 42}
                   for k in ("laptime", "racewin", "position")},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data))
    return get_config(p)


def test_prediction_log_roundtrip(tmp_path):
    cfg = _tmp_config(tmp_path)

    @dataclass
    class _Result:
        forecasts: list
        race_label: str = "Test GP"
        using_real_qualifying: bool = True
        model_r2: float = 0.5
        lookback: int = 6
        as_of_round: int = 13
        prediction_date: str = "2026-09-19 15:00"
        bias_applied: dict = None

    forecasts = [_F(1, "A", "T", 1.5, 1.7, 0.4, {}), _F(2, "B", "T", 2.4, 2.4, 0.4, {})]
    result = _Result(forecasts=forecasts)

    path = feedback.write_prediction_log(cfg, result, round_no=14)
    assert path.exists()
    loaded = feedback.load_prediction_log(cfg, 14)
    assert loaded["round"] == 14
    assert loaded["as_of_round"] == 13
    assert loaded["forecasts"][0]["raw_predicted_position"] == 1.7
    assert loaded["scored"] is None

    feedback.update_prediction_log_scored(cfg, 14, {"scored_at": "x", "metrics": {}, "per_driver": []})
    assert feedback.load_prediction_log(cfg, 14)["scored"]["scored_at"] == "x"

    logs = feedback.list_prediction_logs(cfg)
    assert len(logs) == 1 and logs[0]["round"] == 14
