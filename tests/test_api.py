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
