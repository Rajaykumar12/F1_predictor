import pandas as pd
import pytest
from pipeline.features import (
    engineer_lap_features,
    engineer_result_features,
    create_historical_features,
    _compute_team_reliability,
)
from pipeline.clean import (
    handle_missing_lap_times,
    detect_and_remove_outliers,
    clean_laps,
)


def _make_laps(n=50, race=1, driver="VER", team="Red Bull"):
    return pd.DataFrame({
        "Year": 2026,
        "Race": race,
        "Driver": driver,
        "Team": team,
        "LapNumber": range(1, n + 1),
        "LapTime_seconds": [90.0 + i * 0.05 for i in range(n)],
        "Position": [1] * n,
        "TireCompound": ["MEDIUM"] * n,
        "TireAge": list(range(1, n + 1)),
    })


def _make_results():
    return pd.DataFrame({
        "Year": 2026,
        "Race": [1, 1, 2, 2],
        "Driver": ["VER", "HAM", "VER", "HAM"],
        "Team": ["Red Bull", "Mercedes", "Red Bull", "Mercedes"],
        "Position": [1, 2, 2, 1],
        "GridPosition": [1, 3, 2, 1],
        "Points": [25, 18, 18, 25],
        "Status": ["Finished", "Finished", "+1 Lap", "Finished"],
    })


def test_engineer_lap_features_adds_columns():
    laps = _make_laps()
    out = engineer_lap_features(laps)
    for col in ["start_position", "positions_gained", "tire_degradation", "race_phase",
                "TireCompound_encoded", "IsFreshTire", "LapNumber_normalized",
                "FuelLoadProxy", "IsOutlap", "IsInlap", "RollingAvgLapTime_3"]:
        assert col in out.columns, f"Missing column: {col}"


def test_tire_compound_encoded_values():
    laps = _make_laps()
    out = engineer_lap_features(laps)
    assert (out["TireCompound_encoded"] == 2).all()  # MEDIUM=2


def test_isoutlap_on_first_lap():
    laps = _make_laps()
    out = engineer_lap_features(laps)
    assert out.iloc[0]["IsOutlap"] == 1
    assert out.iloc[1]["IsOutlap"] == 0


def test_lap_number_normalized_range():
    laps = _make_laps(n=60)
    out = engineer_lap_features(laps)
    assert out["LapNumber_normalized"].max() <= 1.0
    assert out["LapNumber_normalized"].min() > 0.0


def test_team_reliability_uses_completed_statuses():
    results = _make_results()
    rel_strict = _compute_team_reliability(results, completed_statuses=["Finished"])
    rel_lenient = _compute_team_reliability(results, completed_statuses=["Finished", "+1 Lap"])
    # Red Bull: 1 Finished + 1 "+1 Lap" out of 2 races
    assert rel_strict["Red Bull"] == 50.0
    assert rel_lenient["Red Bull"] == 100.0


def test_create_historical_features_dnf_counts():
    results = _make_results()
    out = create_historical_features(results, n_previous=6, completed_statuses=["Finished"])
    ver = out[out["Driver"] == "VER"]
    # VER race 2: status "+1 Lap" — with strict completed_statuses, should count as DNF
    assert ver.iloc[1]["dnf_last"] >= 1


def test_create_historical_features_lenient_statuses():
    results = _make_results()
    out = create_historical_features(results, n_previous=6, completed_statuses=["Finished", "+1 Lap"])
    ver = out[out["Driver"] == "VER"]
    # With lenient statuses "+1 Lap" is NOT a DNF
    assert ver.iloc[1]["dnf_last"] == 0


def _make_results_3races():
    return pd.DataFrame({
        "Year": 2026,
        "Race": [1, 2, 3, 1, 3],
        "Driver": ["VER", "VER", "VER", "HAM", "ROOKIE"],
        "Team": ["Red Bull", "Red Bull", "Red Bull", "Mercedes", "Sauber"],
        "Position": [2, 4, 6, 1, 10],
        "GridPosition": [1, 3, 5, 2, 12],
        "Points": [18, 12, 8, 25, 1],
        "Status": ["Finished"] * 5,
    })


def test_create_historical_features_as_of_round_filters():
    results = _make_results_3races()
    out = create_historical_features(results, n_previous=6, as_of_round=2)
    ver = out[out["Driver"] == "VER"]
    # only races 1 and 2 survive the cutoff -> last row's rolling mean = mean(2, 4)
    assert len(ver) == 2
    assert ver.iloc[-1]["avg_position_last"] == pytest.approx(3.0)
    # ROOKIE's only race is round 3 -> dropped entirely
    assert "ROOKIE" not in out["Driver"].values


def test_create_historical_features_default_unchanged():
    results = _make_results_3races()
    a = create_historical_features(results, n_previous=6)
    b = create_historical_features(results, n_previous=6, as_of_round=None)
    pd.testing.assert_frame_equal(a, b)


def test_handle_missing_lap_times_fills_with_median():
    laps = _make_laps()
    laps.loc[5, "LapTime_seconds"] = None
    out = handle_missing_lap_times(laps)
    assert out["LapTime_seconds"].isna().sum() == 0


def test_outlier_removal_removes_extreme_values():
    laps = _make_laps()
    laps.loc[10, "LapTime_seconds"] = 9999.0  # extreme outlier
    out = detect_and_remove_outliers(laps, "LapTime_seconds")
    assert 9999.0 not in out["LapTime_seconds"].values
    assert len(out) < len(laps)


def test_clean_laps_uses_default_tire(tmp_path):
    laps = _make_laps()
    laps["TireCompound"] = None  # all nulls
    out = clean_laps(laps)
    assert out["TireCompound"].isna().sum() == 0
