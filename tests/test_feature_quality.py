"""Phase-5 re-validation gate for the redesigned position-model feature set.

Runs the leakage-aware battery from ``pipeline.analyze`` on the *real*
``data/f1_results_features.csv`` and fails if the redesign has regressed:

* no feature (combination) reconstructs the target (identity leak),
* every numeric feature the deployed model uses has VIF < the configured cap,
* the clean frame carries no ``leaky`` flag,
* at least a handful of features carry genuine model-based signal,
* the forward-chaining holdout MAE is in a sane range and reported next to the
  documented baseline.

Marked ``slow`` — run with ``-m slow``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import analyze
from pipeline import feature_registry as fr
from pipeline.config_loader import get_config

pytestmark = pytest.mark.slow

_ROOT = Path(__file__).resolve().parents[1]
_CSV = _ROOT / "data" / "f1_results_features.csv"
_METRICS = _ROOT / "models" / "metrics.json"

pytest.importorskip("statsmodels")
skip_no_data = pytest.mark.skipif(not _CSV.exists(), reason="run `python main.py features` first")


@pytest.fixture(scope="module")
def cfg():
    return get_config()


@pytest.fixture(scope="module")
def frames(cfg):
    results = pd.read_csv(_CSV)
    raw = analyze.build_raw_frame(results, cfg)
    clean = analyze.build_clean_frame(results, cfg)
    return raw, clean


@skip_no_data
def test_no_identity_leak(frames, cfg):
    raw, clean = frames
    rep = analyze.leakage_report(raw, clean, cfg)
    idy = rep["identity"]
    # in-sample OLS of Position on all numeric features must not reconstruct it
    assert idy["r2"] < 0.95, idy
    # and no single feature is a near-perfect proxy (the old PositionChange was ~1)
    assert (idy["max_single_feature_r2"] or 0.0) < 0.9, idy


@skip_no_data
def test_clean_frame_has_no_leaky_flag(frames):
    _raw, clean = frames
    specs = analyze.specs_for_frame(clean, "clean")
    assert specs, "clean frame produced no known features"
    assert not any(s.leaky for s in specs)
    assert "PositionChange" not in clean.columns


@skip_no_data
def test_deployed_features_are_below_vif_cap(frames, cfg):
    _raw, clean = frames
    cap = cfg.features.vif_threshold

    # the feature set the trained model actually uses (post-selection)
    used = None
    if _METRICS.exists():
        used = json.loads(_METRICS.read_text()).get("position", {}).get("features_used")
    numeric = [
        c for c in (used or fr.numeric_names(fr.enabled_features(cfg)))
        if c in clean.columns and c not in ("Driver", "Team")
    ]
    vt = analyze.vif_table(clean, numeric).set_index("feature")["vif"]
    offenders = vt[vt >= cap]
    assert offenders.empty, f"VIF >= {cap}: {offenders.to_dict()}"


@skip_no_data
def test_enough_features_carry_model_signal(frames, cfg):
    _raw, clean = frames
    specs = [s for s in analyze.specs_for_frame(clean, "clean")]
    mb = analyze.model_based_table(
        clean, specs, cfg, n_splits=3, n_repeats=5,
        rng=np.random.default_rng(0),
    )
    real = mb[(mb["p_perm_importance"] < 0.05) | (mb["delta_cv_mae"] > 0)]
    assert len(real) >= 5, mb[["feature", "p_perm_importance", "delta_cv_mae"]].to_string()


@skip_no_data
def test_shift_guard_shows_leak_removed(frames, cfg):
    raw, clean = frames
    rep = analyze.leakage_report(raw, clean, cfg)
    g = rep["shift_guard_cv_mae"]
    # shift=0 (cheating) should not be *worse* than the production shift
    assert g["cv_mae_shift0"] <= g["cv_mae_production"] + 0.5


@skip_no_data
def test_forward_chain_holdout_is_sane_and_reported():
    assert _METRICS.exists(), "train the position model first"
    m = json.loads(_METRICS.read_text())["position"]
    fc = m["forward_chain_mae"]
    assert 1.0 < fc < 6.0, m                     # better than random (~7), not leaky-perfect
    assert "documented_baseline" in m
    assert m["registry_version"] == fr.REGISTRY_VERSION


# --------------------------------------------------------------------------- #
# F — race_seq ordering and season-reset feature property tests
# --------------------------------------------------------------------------- #
@skip_no_data
def test_race_seq_ordering_never_leaks_future(frames, cfg):
    """race_seq must be a monotone dense-rank of (Year, Race) and, with shift=1,
    every form feature on round R must only aggregate rounds < R.
    """
    _raw, clean = frames
    assert "race_seq" in clean.columns, "race_seq missing from feature frame"

    # race_seq must be monotone in (Year, Race) order
    order = (
        clean.drop_duplicates(["Year", "Race", "race_seq"])
        .sort_values(["Year", "Race"])
    )
    seqs = order["race_seq"].tolist()
    assert seqs == sorted(seqs), \
        "race_seq is NOT monotone with (Year, Race): ordering is broken"

    # With shift >= 1, avg_position_last on round R may not equal the mean
    # of Position from the *same or future* rounds for any driver.
    if "avg_position_last" not in clean.columns:
        return
    for driver, g in clean.groupby("Driver", sort=False):
        g = g.sort_values("race_seq")
        for _, row in g.iterrows():
            s = int(row["race_seq"])
            val = row["avg_position_last"]
            if pd.isna(val):
                continue
            # positions from this round onward — none of these should be in val
            future = g[g["race_seq"] >= s]["Position"]
            if len(future) > 1:
                future_mean = float(future.mean())
                assert not np.isclose(val, future_mean, atol=0.01), \
                    (f"Possible future-leakage for {driver} at seq={s}: "
                     f"avg_position_last={val:.2f} ≈ future mean {future_mean:.2f}")


@skip_no_data
def test_season_reset_features_are_zero_at_round_1(frames, cfg):
    """Season-start rows (Race == 1 for each Year) must not carry over accumulated
    stats from the previous season — those to-date totals should be NaN or 0.
    """
    _raw, clean = frames
    season_starts = clean[clean["Race"] == 1]
    if season_starts.empty:
        return

    for col in ("driver_dnf_rate_todate", "driver_points_gap_to_leader_before"):
        if col not in clean.columns:
            continue
        non_zero = season_starts[
            season_starts[col].notna() & (season_starts[col] != 0.0)
        ]
        assert non_zero.empty, (
            f"Season-reset feature '{col}' has non-zero/non-NaN values at Race=1:\n"
            + non_zero[["Year", "Race", "Driver", col]].to_string()
        )
