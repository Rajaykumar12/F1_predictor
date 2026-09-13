"""
API endpoint tests using FastAPI's TestClient.
Models must be trained before running (python main.py train).
Tests that require trained models are skipped if models aren't loaded.
"""
import pytest
from fastapi.testclient import TestClient

from app import app, win_model, laptime_pipeline, race_model

client = TestClient(app)

needs_win_model = pytest.mark.skipif(win_model is None, reason="win model not loaded")
needs_laptime_model = pytest.mark.skipif(laptime_pipeline is None, reason="laptime model not loaded")
needs_race_model = pytest.mark.skipif(race_model is None, reason="race position model not loaded")


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models_loaded" in data
    assert "data_age_hours" in data


def test_health_has_all_model_keys():
    response = client.get("/health")
    data = response.json()
    assert "race_winner" in data["models_loaded"]
    assert "lap_time" in data["models_loaded"]
    assert "race_position" in data["models_loaded"]


def test_predict_rejects_invalid_grid_position():
    payload = {
        "Team": "Red Bull",
        "Position": 1,
        "GridPosition": 0,  # invalid: must be >= 1
        "driver_win_rate": 50.0,
        "team_reliability": 90.0,
        "BestQualifyingTime": 83.5,
        "GapToPole": 0.0,
        "QualifyingPerformance": 5.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_out_of_range_win_rate():
    payload = {
        "Team": "Red Bull",
        "Position": 1,
        "GridPosition": 1,
        "driver_win_rate": 150.0,  # invalid: must be <= 100
        "team_reliability": 90.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_laptime_rejects_negative_tire_age():
    payload = {
        "Race": "Monaco",
        "Driver": "VER",
        "Team": "Red Bull",
        "Position": 1,
        "TireCompound": "SOFT",
        "TireAge": -1,  # invalid
        "driver_win_rate": 50.0,
        "team_reliability": 90.0,
    }
    response = client.post("/predict_laptime", json=payload)
    assert response.status_code == 422


def test_predict_returns_503_when_model_unloaded(monkeypatch):
    monkeypatch.setattr("app.win_model", None)
    payload = {
        "Team": "Red Bull",
        "Position": 1,
        "GridPosition": 1,
        "driver_win_rate": 50.0,
        "team_reliability": 90.0,
        "BestQualifyingTime": 83.5,
        "GapToPole": 0.0,
        "QualifyingPerformance": 5.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 503


@needs_win_model
def test_predict_valid_input_returns_prediction():
    """Passes valid input and expects either a prediction (200) or a model-mismatch
    error (400) if the loaded model was trained with old data. 422 means our own
    validation fired incorrectly — that should never happen here."""
    payload = {
        "Team": "Red Bull",
        "Position": 1,
        "GridPosition": 1,
        "driver_win_rate": 50.0,
        "team_reliability": 90.0,
        "BestQualifyingTime": 83.5,
        "GapToPole": 0.0,
        "QualifyingPerformance": 5.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in (200, 400), (
        f"Unexpected status {response.status_code}: {response.text}"
    )
    if response.status_code == 200:
        data = response.json()
        assert "will_win" in data
        assert "win_probability" in data
        assert 0.0 <= data["win_probability"] <= 1.0


@needs_laptime_model
def test_predict_laptime_valid_input():
    payload = {
        "Race": "Monaco",
        "Driver": "VER",
        "Team": "Red Bull",
        "Position": 1,
        "TireCompound": "SOFT",
        "TireAge": 5,
        "driver_win_rate": 50.0,
        "team_reliability": 90.0,
    }
    response = client.post("/predict_laptime", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_laptime_seconds" in data
    assert data["predicted_laptime_seconds"] > 0


# ---------------------------------------------------------------------------
# Pipeline job endpoints
# ---------------------------------------------------------------------------

import time as _time

import app as app_module


@pytest.fixture(autouse=True)
def _clear_jobs():
    """Ensure each test starts with no in-flight job (the executor has 1 worker,
    so a leftover 'running' job would 409 every subsequent submission)."""
    app_module._jobs.clear()
    yield
    app_module._jobs.clear()


def _wait_for_job(job_id: str, timeout: float = 5.0) -> dict:
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        job = app_module._jobs[job_id]
        if job["status"] in ("success", "failed"):
            return job
        _time.sleep(0.02)
    raise TimeoutError(f"job {job_id} did not finish in {timeout}s")


def test_pipeline_fetch_submits_job(monkeypatch):
    monkeypatch.setattr(app_module, "run_fetch", lambda cfg: None)
    response = client.post("/pipeline/fetch")
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "queued"
    job = _wait_for_job(data["job_id"])
    assert job["status"] == "success"


def test_pipeline_rejects_concurrent_job(monkeypatch):
    import threading

    release = threading.Event()

    def _slow_fetch(cfg):
        release.wait(timeout=5)

    monkeypatch.setattr(app_module, "run_fetch", _slow_fetch)
    try:
        first = client.post("/pipeline/fetch")
        assert first.status_code == 202

        second = client.post("/pipeline/clean")
        assert second.status_code == 409
    finally:
        release.set()
        _wait_for_job(first.json()["job_id"])


def test_job_status_404_for_unknown_id():
    response = client.get("/jobs/does-not-exist")
    assert response.status_code == 404


def test_pipeline_train_reloads_models(monkeypatch):
    # _job_train now delegates to pipeline.orchestrate.train_models
    monkeypatch.setattr(
        app_module.orchestrate,
        "train_models",
        lambda cfg, model: {"trained": [model], "metrics": {}},
    )
    monkeypatch.setattr(app_module, "_reload_models", lambda: None)
    response = client.post("/pipeline/train", json={"model": "racewin"})
    assert response.status_code == 202
    job = _wait_for_job(response.json()["job_id"])
    assert job["status"] == "success"
    assert job["result"]["trained"] == ["racewin"]


# ---------------------------------------------------------------------------
# Feedback loop endpoints
# ---------------------------------------------------------------------------
def test_pipeline_evaluate_position_submits_job(monkeypatch):
    monkeypatch.setattr(
        app_module.orchestrate, "evaluate_position_job",
        lambda cfg, race: {"race_round": race, "drivers_evaluated": 20, "position_mae": 3.2},
    )
    response = client.post("/pipeline/evaluate-position", json={"race": 13})
    assert response.status_code == 202
    job = _wait_for_job(response.json()["job_id"])
    assert job["status"] == "success"
    assert job["result"]["drivers_evaluated"] == 20


def test_pipeline_score_race_requires_round():
    assert client.post("/pipeline/score-race", json={}).status_code == 422


def test_pipeline_score_race_runs(monkeypatch):
    canned = {
        "race_round": 13,
        "metrics": {"winner_correct": False, "position_mae": 4.6},
        "rolling_scorecard": {"races": 1},
        "drift": {"retrain_recommended": False, "reasons": [], "auto_retrain": False},
    }
    monkeypatch.setattr(app_module.orchestrate, "score_race", lambda cfg, race, fetch_if_missing=True: canned)
    response = client.post("/pipeline/score-race", json={"race": 13})
    assert response.status_code == 202
    job = _wait_for_job(response.json()["job_id"])
    assert job["status"] == "success"
    assert "drift" in job["result"]


def test_score_history_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module.feedback, "load_history", lambda path: [])
    data = client.get("/score-history").json()
    assert data["history"] == []
    assert "rolling_scorecard" in data


def _fake_prediction_log(round_no: int) -> dict:
    return {
        "season": 2026,
        "round": round_no,
        "race_label": f"Round {round_no}",
        "lookback": 6,
        "as_of_round": round_no - 1,
        "predicted_at": "2026-01-01T00:00:00",
        "model_trained_at": None,
        "model_r2": None,
        "using_real_qualifying": False,
        "bias_applied": None,
        "forecasts": [],
        "scored": None,
    }


def test_predictions_endpoint_shape(monkeypatch):
    monkeypatch.setattr(
        app_module.feedback, "list_prediction_logs",
        lambda cfg: [_fake_prediction_log(12), _fake_prediction_log(13)],
    )
    data = client.get("/predictions").json()
    assert isinstance(data, list) and data[0]["round"] == 13  # newest first


def test_health_includes_feedback_key():
    data = client.get("/health").json()
    assert "feedback" in data  # may be None when no history


@needs_race_model
def test_predict_next_race_parity():
    import re
    r = client.get("/predict_next_race?lookback_races=6")
    assert r.status_code == 200
    body = r.json()
    # E3 added simulation_n_trials to the top-level response
    expected_keys = {"predictions", "prediction_date", "next_race", "model_r2", "simulation_n_trials"}
    assert set(body) == expected_keys, f"Unexpected keys: {set(body) ^ expected_keys}"
    assert body["predictions"]
    assert re.search(r" — (real qualifying|historical grid positions), last 6 races form$", body["next_race"])
    # E3 — each driver prediction should carry probability fields (may be None if DNF model absent)
    first = body["predictions"][0]
    prob_fields = {"win_probability", "podium_probability", "points_probability", "p10", "p90", "dnf_probability"}
    assert prob_fields.issubset(first), f"Missing probability fields: {prob_fields - set(first)}"
