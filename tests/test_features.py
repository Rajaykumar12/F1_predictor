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
    build_position_features,
    build_quali_features,
    build_racecraft_features,
    build_reliability_features,
    build_team_features,
    create_historical_features,
    feature_manifest,
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
