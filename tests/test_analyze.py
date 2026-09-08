"""Synthetic-frame tests for pipeline.analyze — no network, no real CSVs.

The synthetic frame: ``R`` races x ``D`` drivers; ``Position`` is a per-race
permutation of 1..D; ``Team`` is derived from the driver; ``signal_feat`` tracks
Position with noise; ``noise_feat`` is pure noise; ``dup_feat`` duplicates the
signal; ``const_in_race_feat`` is constant per driver across the season.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import analyze
from pipeline.analyze import FeatureSpec

_STATUSES = ["Finished", "+1 Lap"]


def make_frame(R=13, D=18, seed=0, signal_sd=2.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    driver_const = {f"D{d:02d}": rng.normal() for d in range(D)}
    for r in range(1, R + 1):
        order = rng.permutation(D) + 1  # positions 1..D assigned to drivers
        for d in range(D):
            drv = f"D{d:02d}"
            pos = int(order[d])
            rows.append({
                "Year": 2026, "Race": r, "Driver": drv, "Team": f"T{d // 2:02d}",
                "Position": pos,
                "GridPosition": int(np.clip(pos + rng.integers(-3, 4), 1, D)),
                "Points": max(0, D - pos), "Status": "Finished",
                "BestQualifyingTime": 80 + pos * 0.1 + rng.normal(0, 0.05),
                "GapToPole": pos * 0.1 + rng.normal(0, 0.02),
                "QualifyingPerformance": rng.normal(),
                "signal_feat": pos + rng.normal(0, signal_sd),
                "noise_feat": rng.normal(),
                "const_in_race_feat": driver_const[drv],
            })
    df = pd.DataFrame(rows)
    df["dup_feat"] = df["signal_feat"]
    df["bin_target"] = (df["Position"] <= 3).astype(int)
    return df


NUMERIC_SPECS = [
    FeatureSpec("signal_feat", "numeric", False, ""),
    FeatureSpec("noise_feat", "numeric", False, ""),
    FeatureSpec("const_in_race_feat", "numeric", False, ""),
]


# --------------------------------------------------------------------------- #
# 1. univariate_tests
# --------------------------------------------------------------------------- #
def test_univariate_flags_signal_not_noise():
    df = make_frame(seed=1)
    rng = np.random.default_rng(1)
    out = analyze.univariate_tests(df, NUMERIC_SPECS, "Position", n_perm=300, rng=rng)

    sp = out[out["test"] == "spearman"].set_index("feature")
    assert sp.loc["signal_feat", "q_perm_bh"] < 0.05
    assert sp.loc["signal_feat", "significant"]
    assert "worse finish" in sp.loc["signal_feat", "direction"]  # positive rho
    assert not sp.loc["noise_feat", "significant"]
    assert sp.loc["noise_feat", "q_perm_bh"] > 0.05


def test_univariate_binary_target_runs():
    df = make_frame(seed=2)
    rng = np.random.default_rng(2)
    out = analyze.univariate_tests(df, NUMERIC_SPECS, "bin_target", n_perm=200, rng=rng)
    pb = out[out["test"] == "point_biserial"].set_index("feature")
    assert pb.loc["signal_feat", "p_perm"] < 0.05
    mw = out[out["test"] == "mann_whitney"]
    assert set(mw["feature"]) >= {"signal_feat", "noise_feat"}


# --------------------------------------------------------------------------- #
# 2. bh_fdr
# --------------------------------------------------------------------------- #
def test_bh_fdr_properties():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50)
    q = analyze.bh_fdr(p)
    assert np.all(q >= p - 1e-12)
    assert np.all((q >= 0) & (q <= 1))
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)  # monotone in sorted p


def test_bh_fdr_passes_through_nan():
    q = analyze.bh_fdr([0.01, np.nan, 0.5])
    assert np.isnan(q[1])
    assert q[0] >= 0.01


# --------------------------------------------------------------------------- #
# 3. within_group_permutation_p — calibrated under the null
# --------------------------------------------------------------------------- #
def test_permutation_p_uniform_under_null():
    from scipy import stats

    def abs_spearman(v, t):
        return abs(stats.spearmanr(v, t).correlation)

    ps = []
    for s in range(120):
        rng = np.random.default_rng(1000 + s)
        df = make_frame(R=10, D=14, seed=s)
        x = df["noise_feat"].to_numpy()
        y = df["Position"].to_numpy(float)
        obs = abs_spearman(x, y)
        ps.append(analyze.within_group_permutation_p(
            x, y, df["Race"], abs_spearman, obs, n_perm=100, rng=rng))
    assert 0.35 < float(np.mean(ps)) < 0.65


# --------------------------------------------------------------------------- #
# 4. vif_table
# --------------------------------------------------------------------------- #
def test_vif_detects_duplicate():
    df = make_frame(seed=3)
    vt = analyze.vif_table(df, ["signal_feat", "dup_feat", "noise_feat"]).set_index("feature")
    assert vt.loc["dup_feat", "vif"] > 1e3
    assert vt.loc["noise_feat", "vif"] < 5


# --------------------------------------------------------------------------- #
# 5. ols_cluster_robust
# --------------------------------------------------------------------------- #
def test_ols_cluster_robust_recovers_signal():
    df = make_frame(seed=4)
    res = analyze.ols_cluster_robust(df, ["signal_feat", "noise_feat"], "Position")
    terms = {t["term"]: t for t in res["terms"]}
    assert terms["signal_feat"]["coef"] > 0
    assert terms["signal_feat"]["p"] < 0.05
    assert res["summary"]["n_clusters"] == df["Driver"].nunique()


def test_ols_noise_usually_not_significant():
    hits = 0
    for s in range(20):
        df = make_frame(seed=100 + s)
        res = analyze.ols_cluster_robust(df, ["signal_feat", "noise_feat"], "Position")
        p = {t["term"]: t["p"] for t in res["terms"]}["noise_feat"]
        hits += p < 0.05
    assert hits <= 4  # ~5% false-positive rate, allow slack


# --------------------------------------------------------------------------- #
# 6. gee_logit
# --------------------------------------------------------------------------- #
def test_gee_logit_runs_and_finds_signal():
    df = make_frame(R=16, D=20, seed=5)
    res = analyze.gee_logit(df, ["signal_feat", "noise_feat"], "bin_target", categorical=())
    assert res["summary"]["converged"]
    terms = {t["term"]: t for t in res["terms"]}
    # bin_target = Position <= 3; higher signal_feat => higher Position => less likely
    assert terms["signal_feat"]["coef"] < 0
    assert terms["signal_feat"]["p"] < 0.05


# --------------------------------------------------------------------------- #
# 7. add_shifted_history / build_clean_frame
# --------------------------------------------------------------------------- #
def test_add_shifted_history_excludes_current_race():
    df = make_frame(R=12, D=1, seed=6)
    sh = analyze.add_shifted_history(df, n_previous=6, completed_statuses=_STATUSES)
    sh = sh.sort_values("Race").reset_index(drop=True)
    pos = sh["Position"].to_numpy()
    k = 9  # 0-based row index
    expected = pos[max(0, k - 6):k].mean()
    assert sh.loc[k, "avg_position_last"] == pytest.approx(expected)
    assert np.isnan(sh.loc[0, "avg_position_last"])  # first race has no history


def test_build_clean_frame_drops_leakage(tmp_path):
    df = make_frame(seed=7)
    cfg = _tmp_config(tmp_path)
    clean = analyze.build_clean_frame(df, cfg)
    assert "PositionChange" not in clean.columns
    # shifted rollup present and finite
    assert clean["avg_position_last"].notna().all()


# --------------------------------------------------------------------------- #
# 8. permutation scheme fallback
# --------------------------------------------------------------------------- #
def test_driver_constant_feature_uses_within_driver_scheme():
    df = make_frame(seed=8)
    specs = [FeatureSpec("const_in_race_feat", "numeric", True, "driver-constant")]
    rng = np.random.default_rng(8)
    out = analyze.univariate_tests(df, specs, "Position", n_perm=50, rng=rng)
    assert (out["perm_scheme"] == "within_driver").all()


# --------------------------------------------------------------------------- #
# 9. run_feature_analysis smoke
# --------------------------------------------------------------------------- #
_DATA_CSV = Path(__file__).resolve().parents[1] / "data" / "f1_results_features.csv"


@pytest.mark.skipif(not _DATA_CSV.exists(), reason="feature CSV not present")
def test_run_feature_analysis_smoke_no_model():
    from pipeline.config_loader import get_config

    res = analyze.run_feature_analysis(get_config(), run_model=False, n_perm=100, seed=42)
    for key in ("frames", "univariate", "multivariable", "model", "leakage", "markdown"):
        assert key in res
    assert not res["univariate"].empty
    assert res["markdown"].startswith("# Statistical hypothesis testing")
    u = res["univariate"]
    assert u["p_perm"].dropna().between(1e-9, 1.0).all()
    assert (u["q_perm_bh"].dropna() >= u.loc[u["q_perm_bh"].notna(), "p_perm"] - 1e-9).all()


@pytest.mark.skipif(not _DATA_CSV.exists(), reason="feature CSV not present")
def test_run_feature_analysis_smoke_tiny_model():
    from pipeline.config_loader import get_config

    cfg = get_config()
    cfg.models.position = dataclasses.replace(cfg.models.position, n_estimators=10)
    res = analyze.run_feature_analysis(
        cfg, targets=("Position",), frames=("clean",), run_model=True,
        n_perm=50, perm_importance_repeats=3, cv_splits=3, seed=1,
    )
    assert not res["model"].empty
    assert res["model"]["p_perm_importance"].between(0, 1).all()


# --------------------------------------------------------------------------- #
def _tmp_config(tmp_path):
    import yaml
    from pipeline.config_loader import get_config

    data = {
        "paths": {"data_dir": "data", "models_dir": "models", "cache_dir": "cache",
                  "plots_dir": "plots"},
        "pipeline": {"season": 2026, "lookback_races": 6, "min_lookback": 3,
                     "max_lookback": 12, "api_sleep_seconds": 2},
        "models": {k: {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 3,
                       "random_state": 42}
                   for k in ("laptime", "racewin", "position")},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data))
    return get_config(p)
