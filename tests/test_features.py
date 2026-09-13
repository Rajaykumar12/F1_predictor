"""Tests for the leakage-safe position-model feature builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import feature_registry as fr
from pipeline.config_loader import get_config
from pipeline.features import (
    build_championship_features,
    build_circuit_features,
    build_form_features,
    build_history_features,
    build_position_features,
    build_quali_features,
    build_racecraft_features,
    build_reliability_features,
    build_team_features,
    compute_race_seq,
    create_historical_features,
    feature_manifest,
    regulation_era,
)

_STATUSES = ["Finished", "+1 Lap"]


def _season(n_races=8, drivers=("VER", "HAM", "NOR", "LEC"), seed=0):
    """A small synthetic season: each race a permutation of finishing positions,
    2 drivers per team, monotone-ish qualifying."""
    rng = np.random.default_rng(seed)
    teams = {"VER": "RB", "HAM": "RB", "NOR": "McL", "LEC": "McL"}
    rows = []
    for r in range(1, n_races + 1):
        order = rng.permutation(len(drivers)) + 1
        for d, pos in zip(drivers, order):
            grid = int(np.clip(pos + rng.integers(-1, 2), 1, len(drivers)))
            best_q = 78.0 + 0.2 * grid + rng.normal(0, 0.05)
            rows.append({
                "Year": 2026, "Race": r, "Driver": d, "Team": teams[d],
                "Position": int(pos), "GridPosition": grid,
                "Points": max(0, len(drivers) - pos) * 5,
                "Status": "Finished" if rng.random() > 0.1 else "Accident",
                "BestQualifyingTime": best_q,
                "QualifyingPerformance": grid / 20 * 100,
                "q3_reached": int(grid <= 3),
            })
    df = pd.DataFrame(rows)
    df["GapToPole"] = df["BestQualifyingTime"] - df.groupby("Race")["BestQualifyingTime"].transform("min")
    return df


@pytest.fixture(scope="module")
def cfg():
    return get_config()


# --------------------------------------------------------------------------- #
# family builders
# --------------------------------------------------------------------------- #
def test_quali_features_columns_and_teammate_delta(cfg):
    df = build_quali_features(_season(), cfg)
    for c in ("quali_gap_to_pole_pct", "quali_gap_to_teammate_s",
              "quali_beat_teammate", "q3_reached"):
        assert c in df.columns
    # exactly one driver per (race, team) pair beats the team-mate
    per = df.groupby(["Race", "Team"])["quali_beat_teammate"].sum()
    assert (per == 1).all()


def test_quali_gap_to_pole_pct_zero_for_pole_sitter(cfg):
    df = build_quali_features(_season(), cfg)
    poled = df.loc[df.groupby("Race")["BestQualifyingTime"].idxmin()]
    assert np.allclose(poled["quali_gap_to_pole_pct"], 0.0)


def test_racecraft_is_shift_before_roll(cfg):
    df = create_historical_features(_season(), n_previous=6, completed_statuses=_STATUSES)
    df = build_racecraft_features(df, cfg)
    # first race per driver has no prior grid->finish delta
    firsts = df.sort_values("Race").groupby("Driver").head(1)
    assert firsts["hist_positions_gained_s5"].isna().all()


def test_reliability_rate_is_pre_race(cfg):
    df = _season()
    df = build_reliability_features(df, cfg)
    firsts = df.sort_values("Race").groupby("Driver").head(1)
    assert firsts["driver_dnf_rate_todate"].isna().all()
    ok = df["driver_dnf_rate_todate"].dropna()
    assert ((ok >= 0) & (ok <= 1)).all()


def test_team_quali_pace_rank_is_dense_rank_within_race(cfg):
    df = build_team_features(_season(), cfg)
    for _, g in df.groupby("Race"):
        ranks = set(g["team_quali_pace_rank"].dropna().unique())
        assert ranks <= {1.0, 2.0}  # two teams


def test_championship_gap_nonnegative_and_zero_at_round1(cfg):
    df = build_championship_features(_season(), cfg)
    assert (df["driver_points_gap_to_leader_before"] >= 0).all()
    assert (df.loc[df["Race"] == 1, "driver_points_gap_to_leader_before"] == 0).all()


def test_circuit_features_merge_by_round(cfg):
    df = _season(n_races=3)
    out = build_circuit_features(df, cfg)
    for c in ("circuit_overtaking_index", "circuit_is_street", "circuit_sc_probability"):
        assert c in out.columns
        assert out[c].notna().all()
    # round 6 == Monaco: street, hardest to overtake
    monaco = build_circuit_features(_season(n_races=6), cfg)
    m = monaco[monaco["Race"] == 6].iloc[0]
    assert m["circuit_is_street"] == 1
    assert m["circuit_overtaking_index"] == 1


def test_circuit_features_idempotent(cfg):
    """Re-running on a frame that already has the columns must not raise / dup."""
    df = build_circuit_features(_season(n_races=3), cfg)
    again = build_circuit_features(df, cfg)
    assert list(again.columns).count("circuit_overtaking_index") == 1


# --------------------------------------------------------------------------- #
# the assembled feature path
# --------------------------------------------------------------------------- #
def test_build_position_features_has_every_registry_column(cfg):
    df = build_position_features(_season(n_races=10), cfg)
    for name in fr.feature_names(fr.enabled_features(cfg)):
        assert name in df.columns, name
    assert "PositionChange" not in df.columns


def test_build_position_features_is_idempotent(cfg):
    once = build_position_features(_season(n_races=6), cfg)
    twice = build_position_features(once, cfg)
    names = [n for n in fr.feature_names(fr.enabled_features(cfg)) if n in once.columns]
    pd.testing.assert_frame_equal(
        once[names].reset_index(drop=True), twice[names].reset_index(drop=True)
    )


def test_as_of_safety_future_race_does_not_change_past_rows(cfg):
    """The core leakage guarantee: adding race N+1 must leave every earlier row's
    feature values byte-identical."""
    full = _season(n_races=9, seed=3)
    early = full[full["Race"] <= 8].copy()

    feats_full = build_position_features(full, cfg)
    feats_early = build_position_features(early, cfg)

    names = [n for n in fr.numeric_names(fr.enabled_features(cfg)) if n in feats_full.columns]
    a = (feats_full[feats_full["Race"] <= 8]
         .sort_values(["Race", "Driver"]).reset_index(drop=True)[names])
    b = feats_early.sort_values(["Race", "Driver"]).reset_index(drop=True)[names]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_feature_manifest_schema(cfg):
    df = build_position_features(_season(n_races=10), cfg)
    man = feature_manifest(df, cfg)
    assert man["registry_version"] == fr.REGISTRY_VERSION
    assert man["shift"] == cfg.features.shift
    assert set(man["features"]) == set(fr.feature_names(fr.enabled_features(cfg)))
    assert man["as_of_unsafe"] == []
    assert man["features_missing_from_frame"] == []
    assert 0.0 <= min(man["null_rate"].values()) <= max(man["null_rate"].values()) <= 1.0


def test_disabling_a_family_removes_its_columns(cfg, monkeypatch):
    monkeypatch.setattr(cfg.features, "families_enabled", ["quali", "form", "racecraft"])
    df = build_position_features(_season(n_races=6), cfg)
    assert "circuit_overtaking_index" not in df.columns
    assert "driver_points_gap_to_leader_before" not in df.columns
    assert "quali_gap_to_pole_pct" in df.columns


# --------------------------------------------------------------------------- #
# B2 — cross-season ordering (race_seq)
# --------------------------------------------------------------------------- #
def _multi_season(n_races=4, years=(2024, 2025), drivers=("VER", "HAM"), seed=0):
    """Two mini seasons with the SAME round numbers each year — the exact
    shape that breaks any builder still grouping/sorting by 'Race' alone."""
    rng = np.random.default_rng(seed)
    teams = {"VER": "RB", "HAM": "Merc"}
    rows = []
    for year in years:
        for r in range(1, n_races + 1):
            order = rng.permutation(len(drivers)) + 1
            for d, pos in zip(drivers, order):
                rows.append({
                    "Year": year, "Race": r, "Driver": d, "Team": teams[d],
                    "Position": int(pos), "GridPosition": int(pos),
                    "Points": max(0, len(drivers) - pos) * 5,
                    "Status": "Finished",
                    "BestQualifyingTime": 78.0 + pos + rng.normal(0, 0.05),
                })
    df = pd.DataFrame(rows)
    df["GapToPole"] = df["BestQualifyingTime"] - df.groupby(["Year", "Race"])["BestQualifyingTime"].transform("min")
    return df


def test_race_seq_is_monotonic_across_seasons():
    df = _multi_season(n_races=3, years=(2024, 2025))
    df["race_seq"] = compute_race_seq(df)
    # every 2024 row's race_seq must be < every 2025 row's race_seq
    assert df.loc[df["Year"] == 2024, "race_seq"].max() < df.loc[df["Year"] == 2025, "race_seq"].min()
    # dense-ranked: 2 years x 3 rounds -> 6 distinct race_seq values
    assert sorted(df["race_seq"].unique()) == [1, 2, 3, 4, 5, 6]


def test_create_historical_features_does_not_leak_across_same_round_number(cfg):
    """Round 1 of 2025 must never see round 1 of 2024 as 'the same race' —
    the historical bug this class of test guards against."""
    df = _multi_season(n_races=4, years=(2024, 2025))
    out = create_historical_features(df, n_previous=6, completed_statuses=["Finished"])
    # round 1 of the SECOND season has real prior history (round 2-4 of 2024)
    r1_2025 = out[(out["Year"] == 2025) & (out["Race"] == 1)]
    assert r1_2025["avg_position_last"].notna().all()
    # round 1 of the FIRST season has none
    r1_2024 = out[(out["Year"] == 2024) & (out["Race"] == 1)]
    assert r1_2024["avg_position_last"].isna().all()


def test_championship_gap_resets_each_season(cfg):
    df = _multi_season(n_races=4, years=(2024, 2025))
    out = build_championship_features(df, cfg)
    r1_2025 = out[(out["Year"] == 2025) & (out["Race"] == 1)]
    # round 1 of a new season -> no points banked yet by anyone -> gap is 0
    assert (r1_2025["driver_points_gap_to_leader_before"] == 0).all()


def test_reliability_todate_resets_each_season(cfg):
    df = _multi_season(n_races=4, years=(2024, 2025))
    out = build_reliability_features(df, cfg)
    r1_2025 = out[(out["Year"] == 2025) & (out["Race"] == 1)]
    # first race of a new season -> no prior-season tally carried in -> NaN
    assert r1_2025["driver_dnf_rate_todate"].isna().all()


def test_as_of_round_resolves_within_the_latest_season(cfg):
    df = _multi_season(n_races=4, years=(2024, 2025))
    out = create_historical_features(
        df, n_previous=6, completed_statuses=["Finished"], as_of_round=2,
    )
    # as_of_round defaults to the LATEST season present (2025) — so 2025
    # rounds 3-4 must be excluded, but all of 2024 stays (it's strictly earlier)
    assert set(out.loc[out["Year"] == 2025, "Race"]) <= {1, 2}
    assert set(out.loc[out["Year"] == 2024, "Race"]) == {1, 2, 3, 4}


# --------------------------------------------------------------------------- #
# regulation-era boundary — team/driver "form" must not cross a technical
# regulation reset, but MUST still bridge an ordinary same-era season change
# --------------------------------------------------------------------------- #
def test_regulation_era_buckets_years_correctly():
    resets = [2022, 2026]
    assert regulation_era(2022, resets) == 2022
    assert regulation_era(2023, resets) == 2022
    assert regulation_era(2025, resets) == 2022
    assert regulation_era(2026, resets) == 2026
    assert regulation_era(2030, resets) == 2026


def test_team_form_resets_at_regulation_era_boundary(cfg):
    # 2022 -> 2026 crosses cfg's default regulation_reset_seasons ([2022, 2026])
    df = _multi_season(n_races=4, years=(2022, 2026))
    out = build_team_features(df, cfg)
    r1_2026 = out[(out["Year"] == 2026) & (out["Race"] == 1)]
    assert r1_2026["team_form_avg_finish_s5"].isna().all(), (
        "team_form_avg_finish_s5 must be NaN at the first race of a new "
        "regulation era — it must not carry in the prior era's form"
    )


def test_team_form_still_bridges_a_same_era_season_boundary(cfg):
    # 2024 -> 2025 is entirely inside the same era (both < 2026) — ordinary
    # season-spanning form must still work, this is not a blanket per-season reset.
    df = _multi_season(n_races=4, years=(2024, 2025))
    out = build_team_features(df, cfg)
    r1_2025 = out[(out["Year"] == 2025) & (out["Race"] == 1)]
    assert r1_2025["team_form_avg_finish_s5"].notna().all(), (
        "team_form_avg_finish_s5 must still bridge an ordinary same-era "
        "season boundary — only a real regulation reset should break it"
    )


def test_form_features_reset_at_regulation_era_boundary(cfg):
    df = _multi_season(n_races=4, years=(2022, 2026))
    out = build_form_features(df, cfg)
    r1_2026 = out[(out["Year"] == 2026) & (out["Race"] == 1)]
    assert r1_2026["form_avg_finish_s5"].isna().all()
    # form_trend's neutral-fill (NaN -> 0.0) happens in build_position_features's
    # post-processing, not inside build_form_features itself — called standalone
    # here, it's correctly NaN, same as form_avg_finish_s5.
    assert r1_2026["form_trend"].isna().all()


def test_history_family_prior_season_stays_neutral_across_era_boundary(cfg):
    """driver_prior_season_avg_finish etc. must NOT draw a "prior" from a
    different regulation era — 2026 has no valid prior under the default
    [2022, 2026] reset config, since 2025 is missing AND even if it existed
    it would be pre-reset data relative to 2026."""
    df = _multi_season(n_races=4, years=(2022, 2026))
    out = build_history_features(df, cfg)
    r1_2026 = out[(out["Year"] == 2026) & (out["Race"] == 1)]
    # neutral-filled to a constant (config.constants.grid_size / 2), not a
    # real value derived from 2022
    assert r1_2026["driver_prior_season_avg_finish"].nunique() == 1


def test_history_family_prior_season_activates_across_same_era_years(cfg):
    """A genuine same-era year-to-year prior (2024 -> 2025, both pre-2026)
    must produce a real, non-neutral, non-constant value."""
    df = _multi_season(n_races=4, years=(2024, 2025))
    out = build_history_features(df, cfg)
    r1_2025 = out[(out["Year"] == 2025) & (out["Race"] == 1)]
    # 2 drivers with different 2024 season averages -> at least 2 distinct
    # non-neutral prior values expected
    assert r1_2025["driver_prior_season_avg_finish"].nunique() >= 2


def test_circuit_calendar_distinguishes_same_round_different_years(cfg):
    """Round 13 is a different circuit in 2022 (Hungarian GP) than in 2026
    (Italian GP) — a plain Race==round join must not conflate them."""
    from pipeline.features import load_circuit_calendar

    df = pd.DataFrame({"Year": [2022, 2026], "Race": [13, 13]})
    calendar = load_circuit_calendar(cfg, df)
    keyed = df.merge(calendar, left_on=["Year", "Race"], right_on=["Year", "Round"], how="left")
    keys = dict(zip(keyed["Year"], keyed["circuit_key"]))
    assert keys[2022] != keys[2026], (
        f"round 13 resolved to the same circuit_key in both years: {keys}"
    )
