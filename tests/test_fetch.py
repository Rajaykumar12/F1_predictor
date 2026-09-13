"""Tests for the multi-season fetch loop (B1) and the Position fallback."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from pipeline import fetch


def _laps_df():
    return pd.DataFrame({
        "Driver": ["VER", "VER", "HAM", "HAM"],
        "LapNumber": [1, 2, 1, 2],
        "Position": [1.0, 1.0, 2.0, 2.0],
    })


def test_final_lap_positions_uses_last_lap_per_driver():
    session = SimpleNamespace(laps=_laps_df())
    out = fetch._final_lap_positions(session)
    assert out == {"VER": 1.0, "HAM": 2.0}


def test_final_lap_positions_empty_laps():
    session = SimpleNamespace(laps=pd.DataFrame())
    assert fetch._final_lap_positions(session) == {}


def test_collect_multiple_races_loops_every_season(monkeypatch):
    seen_seasons = []

    def fake_completed(season):
        seen_seasons.append(season)
        return [1, 2]

    def fake_collect(year, race):
        return (
            [{"Year": year, "Race": race, "row": "lap"}],
            [{"Year": year, "Race": race, "row": "result"}],
            [{"Year": year, "Race": race, "row": "quali"}],
        )

    monkeypatch.setattr(fetch, "get_completed_race_rounds", fake_completed)
    monkeypatch.setattr(fetch, "collect_single_race", fake_collect)

    config = SimpleNamespace(
        pipeline=SimpleNamespace(history_start_season=2023, season=2025, api_sleep_seconds=0),
    )
    laps, results, qualifying = fetch.collect_multiple_races(config)

    assert seen_seasons == [2023, 2024, 2025]
    assert len(laps) == 6  # 3 seasons x 2 rounds
    assert {r["Year"] for r in results} == {2023, 2024, 2025}


def test_collect_multiple_races_skips_seasons_with_no_completed_rounds(monkeypatch):
    def fake_completed(season):
        return [] if season == 2023 else [1]

    monkeypatch.setattr(fetch, "get_completed_race_rounds", fake_completed)
    monkeypatch.setattr(
        fetch, "collect_single_race",
        lambda year, race: ([{"Year": year}], [{"Year": year}], [{"Year": year}]),
    )

    config = SimpleNamespace(
        pipeline=SimpleNamespace(history_start_season=2023, season=2024, api_sleep_seconds=0),
    )
    laps, results, qualifying = fetch.collect_multiple_races(config)
    assert {r["Year"] for r in results} == {2024}


# --------------------------------------------------------------------------- #
# save_data(merge=True) / run_fetch incremental saving
# --------------------------------------------------------------------------- #
def test_merge_and_dedup_new_rows_win_over_stale_ones(tmp_path):
    path = tmp_path / "results.csv"
    pd.DataFrame({"Year": [2024], "Race": [1], "Driver": ["VER"], "Position": [1]}).to_csv(path, index=False)

    new = pd.DataFrame({"Year": [2024], "Race": [1], "Driver": ["VER"], "Position": [2]})
    out = fetch._merge_and_dedup(new, path, ["Year", "Race", "Driver"])
    assert len(out) == 1
    assert out.iloc[0]["Position"] == 2  # the re-fetched row wins


def test_merge_and_dedup_keeps_rows_not_in_the_new_batch(tmp_path):
    path = tmp_path / "results.csv"
    pd.DataFrame({
        "Year": [2023, 2024], "Race": [1, 1], "Driver": ["VER", "VER"], "Position": [1, 1],
    }).to_csv(path, index=False)

    new = pd.DataFrame({"Year": [2025], "Race": [1], "Driver": ["VER"], "Position": [3]})
    out = fetch._merge_and_dedup(new, path, ["Year", "Race", "Driver"])
    assert sorted(out["Year"]) == [2023, 2024, 2025]


def test_run_fetch_saves_after_each_season_even_if_a_later_one_fails(tmp_path, monkeypatch):
    config = SimpleNamespace(
        pipeline=SimpleNamespace(history_start_season=2023, season=2025, api_sleep_seconds=0),
        paths=SimpleNamespace(data_dir=tmp_path, cache_dir=tmp_path / "cache"),
    )
    monkeypatch.setattr(fetch, "_setup_cache", lambda cfg: None)

    def fake_completed(season):
        if season == 2025:
            raise fetch.fastf1.exceptions.RateLimitExceededError({"info": "rate limited"}) \
                if hasattr(fetch.fastf1, "exceptions") else RuntimeError("rate limited")
        return [1]

    monkeypatch.setattr(fetch, "get_completed_race_rounds", fake_completed)
    monkeypatch.setattr(
        fetch, "collect_single_race",
        lambda year, race: (
            [{"Year": year, "Race": race, "Driver": "VER", "LapNumber": 1}],
            [{"Year": year, "Race": race, "Driver": "VER"}],
            [{"Year": year, "Race": race, "Driver": "VER"}],
        ),
    )

    with pytest.raises(Exception):
        fetch.run_fetch(config)

    # 2023 and 2024 must have been saved even though 2025 blew up
    results = pd.read_csv(tmp_path / "f1_results_simple.csv")
    assert sorted(results["Year"].unique().tolist()) == [2023, 2024]
