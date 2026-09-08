"""Statistical hypothesis testing of the position model's engineered features.

For every engineered feature, this module asks: is it significantly associated
with race outcome, and does it contribute to the position model — while
accounting for the two things that break a naive Pearson-heatmap read of this
data:

* **Target leakage.** ``PositionChange == GridPosition - Position`` exactly;
  every ``create_historical_features`` rolling window is *unshifted* (includes
  the current race); ``driver_win_rate`` / ``team_reliability`` are whole-season
  aggregates that include the target row. See ``pipeline/features.py``.
* **Repeated measures.** ~23 drivers x 13 rounds; within a race the finishing
  positions are a fixed permutation of 1..22. Naive OLS/logit SEs and random
  K-fold CV are anticonservative.

The response: a dual raw/clean frame design, within-``Race`` permutation
p-values, BH-FDR, cluster-robust / GEE / mixed-effects multivariable inference,
and GroupKFold(by Race) model-based importance.

All functions are pure or narrow-IO so they unit-test with synthetic frames
(see ``tests/test_analyze.py``). The CLI wrapper is
``scripts/feature_hypothesis_tests.py``.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline.config_loader import Config, get_config
from pipeline.features import create_historical_features

logger = logging.getLogger(__name__)

BINARY_TARGETS = ("podium_finish", "points_finish", "race_winner")
ALL_TARGETS = ("Position",) + BINARY_TARGETS
CATEGORICAL = ("Driver", "Team")
RACE_COL = "Race"

# The leaky trio quantified in leakage_report / Phase 5.
LEAKY_TRIO = ("PositionChange", "driver_win_rate", "team_reliability")

# Base series -> rolled feature name, for the unshifted rollups in
# create_historical_features. add_shifted_history reproduces these with .shift(1).
_ROLLUPS: dict[str, str] = {
    "avg_position_last": "Position",
    "best_position_last": "Position",
    "avg_grid_last": "GridPosition",
    "avg_positions_gained": "positions_gained",
    "podiums_last": "Position",
    "wins_last": "Position",
    "points_last": "Points",
    "dnf_last": "is_dnf",
    "reliability_rate": "is_dnf",
    "avg_quali_time": "BestQualifyingTime",
    "avg_gap_to_pole": "GapToPole",
}


@dataclass
class FeatureSpec:
    name: str
    kind: str  # "numeric" | "categorical"
    leaky: bool
    leak_reason: str


# --------------------------------------------------------------------------- #
# Phase 1 — data prep
# --------------------------------------------------------------------------- #
def _position_race_features(processed: pd.DataFrame) -> list[str]:
    """The position model's candidate feature list (train.py:198-206)."""
    race_features = [
        "Driver", "Team", "GridPosition",
        "driver_win_rate", "team_reliability", "QualifyingPerformance", "PositionChange",
        "avg_position_last", "best_position_last", "avg_grid_last",
        "dnf_last", "reliability_rate", "avg_positions_gained",
        "podiums_last", "wins_last", "points_last", "form_trend",
    ]
    if "avg_quali_time" in processed.columns:
        race_features += ["avg_quali_time", "avg_gap_to_pole"]
    return [f for f in race_features if f in processed.columns]


def build_raw_frame(results: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Reproduce ``train_position_model``'s ``processed`` + dropna exactly, then
    carry ``Race`` / ``Driver`` / ``Team`` alongside for grouping and clustering.

    The 4 targets and the leaky ``PositionChange`` are all retained; ``feature_specs``
    flags what is leaky.
    """
    processed = create_historical_features(
        results,
        n_previous=config.pipeline.lookback_races,
        completed_statuses=config.constants.completed_statuses,
    )
    available = _position_race_features(processed)

    _cross_check_against_pickle(available, config)

    keep = list(dict.fromkeys(available + list(ALL_TARGETS) + [RACE_COL, "Driver", "Team"]))
    keep = [c for c in keep if c in processed.columns]
    frame = processed[keep].dropna(subset=[c for c in available if c in processed.columns] + ["Position"])
    return frame.reset_index(drop=True)


def _cross_check_against_pickle(available: list[str], config: Config) -> None:
    pkl = config.paths.models_dir / "race_prediction_pipeline.pkl"
    if not pkl.exists():
        logger.warning("Trained position model %s missing — skipping feature cross-check.", pkl)
        return
    try:
        from pipeline.model_registry import load_bundle, model_feature_columns

        model = load_bundle(config).race_model
        if model is None:
            return
        trained_cols = set(model_feature_columns(model))
        got = set(available)
        if trained_cols != got:
            logger.warning(
                "Feature list drift vs trained model: only-in-frame=%s only-in-model=%s",
                sorted(got - trained_cols), sorted(trained_cols - got),
            )
    except Exception as e:  # noqa: BLE001 — cross-check is advisory only
        logger.warning("Could not cross-check features against the pickle: %s", e)


def add_shifted_history(
    df: pd.DataFrame, n_previous: int, completed_statuses: list[str]
) -> pd.DataFrame:
    """Local, leakage-free reimplementation of the 11 rollups in
    ``create_historical_features`` (features.py:134-158): per driver sorted by
    ``Race``, ``.shift(1)`` **before** ``.rolling(n_previous, min_periods=1)`` so
    the current race is never in its own window. ``form_trend`` keeps its
    existing ``.shift(3)`` construction.

    Column names match ``features.py`` so downstream code is frame-agnostic.
    """
    out = []
    for _driver, g in df.groupby("Driver", sort=False):
        d = g.sort_values(RACE_COL).reset_index(drop=True).copy()

        d["is_dnf"] = (~d["Status"].isin(completed_statuses)).astype(int)
        d["positions_gained"] = d["GridPosition"] - d["Position"]

        roll = lambda s: s.shift(1).rolling(n_previous, min_periods=1)  # noqa: E731

        d["avg_position_last"] = roll(d["Position"]).mean()
        d["best_position_last"] = roll(d["Position"]).min()
        d["avg_grid_last"] = roll(d["GridPosition"]).mean()
        d["dnf_last"] = roll(d["is_dnf"]).sum()
        d["reliability_rate"] = 1 - (d["dnf_last"] / n_previous)
        d["avg_positions_gained"] = roll(d["positions_gained"]).mean()
        d["podiums_last"] = roll((d["Position"] <= 3).astype(int)).sum()
        d["wins_last"] = roll((d["Position"] == 1).astype(int)).sum()
        d["points_last"] = roll(d["Points"]).sum()

        if "BestQualifyingTime" in d.columns:
            d["avg_quali_time"] = roll(d["BestQualifyingTime"]).mean()
            d["avg_gap_to_pole"] = roll(d["GapToPole"]).mean()

        recent_avg = d["Position"].rolling(3, min_periods=1).mean()
        if n_previous > 3:
            older_avg = d["Position"].shift(3).rolling(n_previous - 3, min_periods=1).mean()
            d["form_trend"] = older_avg - recent_avg
        else:
            d["form_trend"] = 0.0

        out.append(d)
    return pd.concat(out, ignore_index=True)


def _expanding_pre_race_rate(df: pd.DataFrame, group: str, event: pd.Series) -> pd.Series:
    """Mean of ``event`` over the group's rows in races **strictly before** the
    current race, as a percent (matching ``_compute_driver_win_rate`` /
    ``_compute_team_reliability`` scale). First race for a group -> NaN.

    Race-aware: a team's two same-race rows never see each other, so the estimate
    is genuinely pre-race.
    """
    tmp = pd.DataFrame({"_g": df[group].to_numpy(), "_r": df[RACE_COL].to_numpy(),
                        "_e": np.asarray(event, dtype=float)}, index=df.index)
    # per (group, race): sum and count of the event
    per = tmp.groupby(["_g", "_r"])["_e"].agg(["sum", "count"]).sort_index()
    cum_sum = per.groupby(level=0)["sum"].cumsum() - per["sum"]
    cum_cnt = per.groupby(level=0)["count"].cumsum() - per["count"]
    rate = (cum_sum / cum_cnt.replace(0, np.nan)) * 100.0
    rate.name = "_rate"
    joined = tmp.join(rate, on=["_g", "_r"])
    return joined["_rate"]


def build_clean_frame(
    results: pd.DataFrame, config: Config, drop_rates: bool = False
) -> pd.DataFrame:
    """Leakage-controlled twin of ``build_raw_frame``:

    * **drop** ``PositionChange`` (it is ``GridPosition - Position``);
    * **replace** ``driver_win_rate`` / ``team_reliability`` with expanding
      *pre-race* rates (per group sorted by ``Race``: cumulative rate over races
      strictly before this one; first race -> NaN). ``drop_rates=True`` drops
      both instead;
    * use the ``.shift(1)`` rollups from :func:`add_shifted_history`;
    * keep ``GridPosition``, ``QualifyingPerformance``, ``Driver``, ``Team``,
      ``form_trend``.
    """
    n_prev = config.pipeline.lookback_races
    completed = config.constants.completed_statuses
    shifted = add_shifted_history(results, n_prev, completed)

    shifted = shifted.drop(columns=[c for c in ("PositionChange",) if c in shifted.columns])

    if drop_rates:
        shifted = shifted.drop(
            columns=[c for c in ("driver_win_rate", "team_reliability") if c in shifted.columns]
        )
    else:
        shifted = shifted.sort_values([RACE_COL, "Driver"]).reset_index(drop=True)
        win = (shifted["Position"] == 1).astype(float)
        rel = shifted["Status"].isin(completed).astype(float)
        shifted["driver_win_rate"] = _expanding_pre_race_rate(shifted, "Driver", win)
        shifted["team_reliability"] = _expanding_pre_race_rate(shifted, "Team", rel)

    feats = [s.name for s in feature_specs("clean") if s.name in shifted.columns]
    keep = list(dict.fromkeys(feats + list(ALL_TARGETS) + [RACE_COL, "Driver", "Team"]))
    keep = [c for c in keep if c in shifted.columns]
    frame = shifted[keep].dropna(subset=feats + ["Position"])
    return frame.reset_index(drop=True)


def feature_specs(frame_kind: str) -> list[FeatureSpec]:
    """Feature list + leakage flags for ``frame_kind`` in {"raw", "clean"}."""
    if frame_kind not in ("raw", "clean"):
        raise ValueError(f"frame_kind must be 'raw' or 'clean', got {frame_kind!r}")

    unshifted = ("includes the current race in its own rolling window "
                 "(create_historical_features windows are unshifted)")
    specs: list[FeatureSpec] = [
        FeatureSpec("GridPosition", "numeric", False, ""),
        FeatureSpec("QualifyingPerformance", "numeric", False, ""),
        FeatureSpec("form_trend", "numeric", False, ""),
        FeatureSpec("Driver", "categorical", False, ""),
        FeatureSpec("Team", "categorical", False, ""),
    ]
    rollups = [
        "avg_position_last", "best_position_last", "avg_grid_last",
        "dnf_last", "reliability_rate", "avg_positions_gained",
        "podiums_last", "wins_last", "points_last",
        "avg_quali_time", "avg_gap_to_pole",
    ]
    if frame_kind == "raw":
        specs.append(FeatureSpec(
            "PositionChange", "numeric", True,
            "equals GridPosition - Position exactly (R^2 ~ 1)"))
        specs.append(FeatureSpec(
            "driver_win_rate", "numeric", True,
            "whole-season aggregate including the target row; constant within Driver"))
        specs.append(FeatureSpec(
            "team_reliability", "numeric", True,
            "whole-season aggregate including the target row; constant within Team"))
        specs += [FeatureSpec(r, "numeric", True, unshifted) for r in rollups]
    else:
        specs.append(FeatureSpec(
            "driver_win_rate", "numeric", False,
            "expanding pre-race win rate (races strictly before the current one)"))
        specs.append(FeatureSpec(
            "team_reliability", "numeric", False,
            "expanding pre-race finish rate (races strictly before the current one)"))
        specs += [FeatureSpec(r, "numeric", False, "") for r in rollups]
    return specs


def specs_for_frame(frame: pd.DataFrame, frame_kind: str) -> list[FeatureSpec]:
    """feature_specs filtered to columns actually present in ``frame``."""
    return [s for s in feature_specs(frame_kind) if s.name in frame.columns]


# --------------------------------------------------------------------------- #
# Phase 2 — univariate association tests
# --------------------------------------------------------------------------- #
def bh_fdr(pvalues) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values. NaNs pass through. The result is
    monotone in sorted raw p and satisfies ``q >= p`` elementwise."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return out
    adj = stats.false_discovery_control(p[mask], method="bh")
    out[mask] = np.clip(adj, 0.0, 1.0)
    return out


def _group_indices(groups: pd.Series) -> list[np.ndarray]:
    codes = pd.Categorical(groups).codes
    return [np.where(codes == c)[0] for c in np.unique(codes)]


def within_group_permutation_p(
    values, target, groups, stat_fn, observed: float, n_perm: int, rng
) -> float:
    """Permutation p-value where the target is shuffled **within each group**
    (repeated-measures valid): ``p = (1 + #{stat_perm >= stat_obs}) / (n_perm + 1)``.

    ``stat_fn(values, permuted_target) -> float`` must return a non-negative,
    larger-is-more-extreme statistic (e.g. ``abs(rho)``, ``H``). NaN statistics
    count as 0.
    """
    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=float)
    groups = pd.Series(np.asarray(groups)).reset_index(drop=True)
    blocks = _group_indices(groups)
    obs = 0.0 if not np.isfinite(observed) else float(observed)

    ge = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_perm):
            perm = target.copy()
            for idx in blocks:
                perm[idx] = rng.permutation(perm[idx])
            s = stat_fn(values, perm)
            if not np.isfinite(s):
                s = 0.0
            if s >= obs - 1e-12:
                ge += 1
    return (1 + ge) / (n_perm + 1)


def _rank_biserial(values: np.ndarray, binary: np.ndarray) -> tuple[float, float, int]:
    """Mann-Whitney U (feature split by binary target) + rank-biserial effect
    size ``r = 1 - 2U/(n1 n0)`` (positive => feature higher when target == 1)."""
    a = values[binary == 1]
    b = values[binary == 0]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, 0
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    rb = 1.0 - (2.0 * u) / (len(a) * len(b))
    return float(rb), float(p), int(len(a) + len(b))


def _epsilon_sq(h: float, k: int, n: int) -> float:
    if n - k <= 0:
        return np.nan
    return float((h - k + 1) / (n - k))


def _cramers_v(table: np.ndarray, chi2: float) -> float:
    n = table.sum()
    if n == 0:
        return np.nan
    r, c = table.shape
    denom = n * (min(r, c) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan


def _direction_numeric(stat: float, target: str, feature: str) -> str:
    if not np.isfinite(stat) or abs(stat) < 1e-9:
        return "no monotone association"
    if target == "Position":
        return (f"higher {feature} -> worse finish (higher position number)"
                if stat > 0 else f"higher {feature} -> better finish")
    return (f"higher {feature} associated with {target} = 1"
            if stat > 0 else f"higher {feature} associated with {target} = 0")


def _constant_within(df: pd.DataFrame, feature: str, group: str) -> bool:
    return bool(df.groupby(group)[feature].nunique(dropna=False).max() <= 1)


def univariate_tests(
    df: pd.DataFrame,
    specs: list[FeatureSpec],
    target: str,
    race_col: str = RACE_COL,
    n_perm: int = 2000,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """One row per (feature, statistical test) for a single target.

    numeric x Position    : Spearman rho (primary), Kendall tau, Pearson r
    numeric x binary      : point-biserial r, Mann-Whitney U (rank-biserial)
    categorical x Position: Kruskal-Wallis H, one-way ANOVA F (eps^2 / eta^2)
    categorical x binary  : chi-square, Cramer's V

    Every row also carries a within-``race_col`` permutation p (fallback:
    within-Driver, when the feature is constant in every race) and ``n``.
    """
    rng = np.random.default_rng() if rng is None else rng
    rows: list[dict] = []

    for spec in specs:
        cols = list(dict.fromkeys([spec.name, target, race_col, "Driver"]))
        sub = df[cols].dropna()
        if len(sub) < 10 or sub[target].nunique() < 2:
            continue
        x = sub[spec.name].to_numpy() if spec.kind == "categorical" else sub[spec.name].to_numpy(float)
        y = sub[target].to_numpy(float)
        groups = sub[race_col]

        # Permutation scheme: shuffle the target within each Race (repeated-measures
        # valid). But a feature that is constant within Driver (e.g. the raw
        # season-wide rate features) has *only* between-driver variation, which is
        # exactly the leakage confound — a within-Race shuffle would still credit
        # it. Fall back to a within-Driver shuffle there, which neutralizes it.
        if spec.kind == "numeric" and _constant_within(sub, spec.name, "Driver"):
            perm_scheme = "within_driver"
            perm_groups = sub["Driver"]
        else:
            perm_scheme = "within_race"
            perm_groups = groups

        tests = _numeric_tests(x, y, target) if spec.kind == "numeric" \
            else _categorical_tests(x, y, target)

        for t in tests:
            if t["stat_fn"] is not None and np.isfinite(t["statistic"]):
                p_perm = within_group_permutation_p(
                    x if spec.kind == "numeric" else pd.Categorical(x).codes,
                    y, perm_groups, t["stat_fn"], t["perm_observed"], n_perm, rng,
                )
            else:
                p_perm = np.nan
            rows.append({
                "feature": spec.name,
                "feature_kind": spec.kind,
                "leaky": spec.leaky,
                "leak_reason": spec.leak_reason,
                "target": target,
                "test": t["test"],
                "statistic": t["statistic"],
                "effect_size_name": t["effect_size_name"],
                "effect_size": t["effect_size"],
                "n": int(len(sub)),
                "p_analytic": t["p_analytic"],
                "perm_scheme": perm_scheme,
                "p_perm": p_perm,
                "n_perm": int(n_perm),
                "direction": t["direction"](spec.name),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_analytic_bh"] = np.nan
    out["q_perm_bh"] = np.nan
    for _test, g in out.groupby("test"):
        out.loc[g.index, "q_analytic_bh"] = bh_fdr(g["p_analytic"].to_numpy())
        out.loc[g.index, "q_perm_bh"] = bh_fdr(g["p_perm"].to_numpy())
    out["significant"] = out["q_perm_bh"] < 0.05
    cols = ["feature", "feature_kind", "leaky", "leak_reason", "target", "test",
            "statistic", "effect_size_name", "effect_size", "n", "p_analytic",
            "q_analytic_bh", "perm_scheme", "p_perm", "q_perm_bh", "n_perm",
            "direction", "significant"]
    return out[cols]


def _numeric_tests(x: np.ndarray, y: np.ndarray, target: str) -> list[dict]:
    if target == "Position":
        rho = stats.spearmanr(x, y).correlation
        tau = stats.kendalltau(x, y).correlation
        r = stats.pearsonr(x, y)
        return [
            {"test": "spearman", "statistic": float(rho),
             "effect_size_name": "rho", "effect_size": float(rho),
             "p_analytic": float(stats.spearmanr(x, y).pvalue),
             "stat_fn": lambda v, t: abs(stats.spearmanr(v, t).correlation),
             "perm_observed": abs(rho),
             "direction": lambda f: _direction_numeric(rho, target, f)},
            {"test": "kendall", "statistic": float(tau),
             "effect_size_name": "tau", "effect_size": float(tau),
             "p_analytic": float(stats.kendalltau(x, y).pvalue),
             "stat_fn": lambda v, t: abs(stats.kendalltau(v, t).correlation),
             "perm_observed": abs(tau),
             "direction": lambda f: _direction_numeric(tau, target, f)},
            {"test": "pearson", "statistic": float(r.statistic),
             "effect_size_name": "r", "effect_size": float(r.statistic),
             "p_analytic": float(r.pvalue),
             "stat_fn": lambda v, t: abs(np.corrcoef(v, t)[0, 1]),
             "perm_observed": abs(r.statistic),
             "direction": lambda f: _direction_numeric(r.statistic, target, f)},
        ]
    # binary target
    pb = stats.pointbiserialr(y, x)
    rb, p_mw, _n = _rank_biserial(x, y.astype(int))
    return [
        {"test": "point_biserial", "statistic": float(pb.statistic),
         "effect_size_name": "r_pb", "effect_size": float(pb.statistic),
         "p_analytic": float(pb.pvalue),
         "stat_fn": lambda v, t: abs(stats.pointbiserialr(t, v).statistic),
         "perm_observed": abs(pb.statistic),
         "direction": lambda f: _direction_numeric(pb.statistic, target, f)},
        {"test": "mann_whitney", "statistic": float(rb),
         "effect_size_name": "rank_biserial", "effect_size": float(rb),
         "p_analytic": float(p_mw),
         "stat_fn": lambda v, t: abs(_rank_biserial(v, t.astype(int))[0]),
         "perm_observed": abs(rb) if np.isfinite(rb) else np.nan,
         "direction": lambda f: _direction_numeric(rb, target, f)},
    ]


def _categorical_tests(x: np.ndarray, y: np.ndarray, target: str) -> list[dict]:
    codes = pd.Categorical(x).codes
    # precompute per-level row-index arrays once; the permutation closures below
    # reuse them instead of re-scanning x on every one of B iterations
    level_idx = [np.where(codes == c)[0] for c in np.unique(codes) if np.sum(codes == c) > 0]
    levels = [y[idx] for idx in level_idx]
    k = len(levels)
    n = len(y)
    if target == "Position":
        h, p_h = stats.kruskal(*levels)
        f, p_f = stats.f_oneway(*levels)
        eps2 = _epsilon_sq(h, k, n)
        # eta^2 from ANOVA sums of squares
        grand = y.mean()
        ss_between = sum(len(lv) * (lv.mean() - grand) ** 2 for lv in levels)
        ss_total = ((y - grand) ** 2).sum()
        eta2 = float(ss_between / ss_total) if ss_total > 0 else np.nan

        def kruskal_stat(_v, t):
            parts = [t[idx] for idx in level_idx]
            return stats.kruskal(*parts).statistic if len(parts) > 1 else 0.0

        def anova_stat(_v, t):
            parts = [t[idx] for idx in level_idx]
            return stats.f_oneway(*parts).statistic if len(parts) > 1 else 0.0

        return [
            {"test": "kruskal_wallis", "statistic": float(h),
             "effect_size_name": "epsilon_sq", "effect_size": eps2,
             "p_analytic": float(p_h), "stat_fn": kruskal_stat, "perm_observed": float(h),
             "direction": lambda f: f"finishing position varies by {f}"},
            {"test": "anova", "statistic": float(f),
             "effect_size_name": "eta_sq", "effect_size": eta2,
             "p_analytic": float(p_f), "stat_fn": anova_stat, "perm_observed": float(f),
             "direction": lambda f: f"mean finishing position varies by {f}"},
        ]
    # binary target -> chi-square. Build the contingency table by bincount on the
    # precomputed level codes (much faster than pd.crosstab in the permutation loop).
    y_int = y.astype(int)
    n_lvl = int(codes.max()) + 1

    def _table(t_int: np.ndarray) -> np.ndarray:
        pos = np.bincount(codes[t_int == 1], minlength=n_lvl)
        neg = np.bincount(codes[t_int == 0], minlength=n_lvl)
        return np.vstack([neg, pos]).T  # rows = levels, cols = [0, 1]

    table = _table(y_int)
    if table.shape[0] < 2 or (table.sum(axis=0) == 0).any():
        return []
    chi2, p_chi, _dof, _exp = stats.chi2_contingency(table)
    v = _cramers_v(table, chi2)

    def chi2_stat(_v, t):
        tab = _table(t.astype(int))
        if (tab.sum(axis=0) == 0).any() or (tab.sum(axis=1) == 0).any():
            return 0.0
        return stats.chi2_contingency(tab)[0]

    return [
        {"test": "chi_square", "statistic": float(chi2),
         "effect_size_name": "cramers_v", "effect_size": v,
         "p_analytic": float(p_chi), "stat_fn": chi2_stat, "perm_observed": float(chi2),
         "direction": lambda f: f"{f} distribution differs by {target}"},
    ]


# --------------------------------------------------------------------------- #
# Phase 3 — multivariable inference (statsmodels)
# --------------------------------------------------------------------------- #
def _standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        s = out[c].astype(float)
        sd = s.std(ddof=0)
        out[c] = (s - s.mean()) / sd if sd > 0 else 0.0
    return out


def _formula(target: str, numeric: list[str], categorical: tuple[str, ...]) -> str:
    rhs = " + ".join(numeric + [f"C({c})" for c in categorical]) or "1"
    return f"Q('{target}') ~ {rhs}" if not target.isidentifier() else f"{target} ~ {rhs}"


def _drop_to_full_rank(df: pd.DataFrame, numeric: list[str], tol: float = 1e-8) -> tuple[list[str], list[str]]:
    """Greedily drop numeric predictors that are (near-)exact linear combinations
    of the ones already kept, so statsmodels gets a full-rank design. In the raw
    frame this removes ``PositionChange`` (== GridPosition - Position); the clean
    frame is normally untouched. Returns (kept, dropped)."""
    z = _standardize(df[numeric].dropna(), numeric)
    kept: list[str] = []
    dropped: list[str] = []
    basis = np.empty((len(z), 0))
    for c in numeric:
        v = z[c].to_numpy(dtype=float).reshape(-1, 1)
        if basis.shape[1] == 0:
            resid_norm = np.linalg.norm(v)
        else:
            coef, *_ = np.linalg.lstsq(basis, v, rcond=None)
            resid_norm = np.linalg.norm(v - basis @ coef)
        if resid_norm <= tol * max(1.0, np.linalg.norm(v)):
            dropped.append(c)
        else:
            kept.append(c)
            basis = np.column_stack([basis, v])
    return kept, dropped


def vif_table(df: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    """Variance inflation factor per numeric predictor (on standardized columns
    plus an intercept)."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    z = _standardize(df[numeric].dropna(), numeric)
    x = np.column_stack([np.ones(len(z))] + [z[c].to_numpy() for c in numeric])
    rows = []
    for i, c in enumerate(numeric, start=1):
        try:
            vif = float(variance_inflation_factor(x, i))
        except Exception:  # noqa: BLE001
            vif = np.nan
        rows.append({"feature": c, "vif": vif})
    return pd.DataFrame(rows)


def ols_cluster_robust(
    df: pd.DataFrame, numeric: list[str], target: str,
    categorical: tuple[str, ...] = ("Team",), cluster: str = "Driver",
) -> dict:
    """OLS with cluster-robust (by ``cluster``) SEs + type-II partial F + VIF."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    cols = numeric + list(categorical) + [target, cluster]
    d = _standardize(df[cols].dropna(), numeric)
    fit = smf.ols(_formula(target, numeric, categorical), data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d[cluster]}
    )
    try:
        aov = anova_lm(smf.ols(_formula(target, numeric, categorical), data=d).fit(), typ=2)
    except Exception:  # noqa: BLE001
        aov = None
    vif = vif_table(d, numeric).set_index("feature")["vif"].to_dict()

    terms = []
    for term in fit.params.index:
        base = term.split("[")[0].replace("C(", "").replace(")", "")
        terms.append({
            "term": term,
            "coef": float(fit.params[term]),
            "se": float(fit.bse[term]),
            "stat": float(fit.tvalues[term]),
            "p": float(fit.pvalues[term]),
            "vif": vif.get(term, np.nan),
            "partial_f": _aov_get(aov, base, "F"),
            "p_partial_f": _aov_get(aov, base, "PR(>F)"),
        })
    return {
        "model": "ols_cluster",
        "terms": terms,
        "summary": {
            "n": int(fit.nobs), "k": int(len(fit.params)),
            "r2": float(fit.rsquared), "adj_r2": float(fit.rsquared_adj),
            "f": float(fit.fvalue) if fit.fvalue is not None else np.nan,
            "p_f": float(fit.f_pvalue) if fit.f_pvalue is not None else np.nan,
            "n_clusters": int(d[cluster].nunique()),
            "cond_number": float(fit.condition_number),
            # a near-perfect fit means a leaky predictor reconstructs the target
            # (raw frame: Position == GridPosition - PositionChange) — per-term
            # p-values are not interpretable here.
            "degenerate": bool(fit.rsquared > 0.9999),
            "converged": True,
        },
    }


def _aov_get(aov, name: str, col: str) -> float:
    if aov is None:
        return np.nan
    for idx in aov.index:
        if idx.split("[")[0].replace("C(", "").replace(")", "") == name:
            try:
                return float(aov.loc[idx, col])
            except Exception:  # noqa: BLE001
                return np.nan
    return np.nan


def mixedlm_driver(
    df: pd.DataFrame, numeric: list[str], target: str,
    categorical: tuple[str, ...] = ("Team",),
) -> dict:
    """``target ~ z-numeric + C(Team)`` with a random intercept by Driver."""
    import statsmodels.formula.api as smf

    cols = numeric + list(categorical) + [target, "Driver"]
    d = _standardize(df[cols].dropna(), numeric)
    converged = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(
                _formula(target, numeric, categorical), data=d, groups=d["Driver"]
            ).fit()
        converged = bool(fit.converged)
        terms = [
            {"term": t, "coef": float(fit.params[t]), "se": float(fit.bse[t]),
             "stat": float(fit.tvalues[t]), "p": float(fit.pvalues[t]),
             "vif": np.nan, "partial_f": np.nan, "p_partial_f": np.nan}
            for t in fit.params.index if t != "Group Var"
        ]
        group_var = float(fit.cov_re.iloc[0, 0]) if fit.cov_re is not None else np.nan
    except Exception as e:  # noqa: BLE001
        logger.warning("mixedlm failed for %s: %s", target, e)
        return {"model": "mixedlm", "terms": [], "summary": {"converged": False, "error": str(e)}}
    return {
        "model": "mixedlm",
        "terms": terms,
        "summary": {
            "n": int(fit.nobs), "k": int(len(terms)),
            "group_var": group_var, "n_clusters": int(d["Driver"].nunique()),
            "converged": converged,
        },
    }


def gee_logit(
    df: pd.DataFrame, numeric: list[str], target: str,
    categorical: tuple[str, ...] = ("Team",), cluster: str = "Driver",
) -> dict:
    """Population-averaged (GEE, exchangeable) panel logistic with cluster-robust
    p-values. Drops ``C(Team)`` and retries if the full model will not converge."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    cols = numeric + list(categorical) + [target, cluster]
    d = _standardize(df[cols].dropna(), numeric)
    n_pos = int(d[target].sum())
    low_power = n_pos < 20 or n_pos > len(d) - 20

    for cats in (categorical, ()):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.gee(
                    _formula(target, numeric, cats), groups=cluster, data=d,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable(),
                ).fit()
            # Reject only when the fit is genuinely unusable: non-finite SEs, or
            # a *numeric* predictor of interest blown up by separation. A large
            # coefficient on a rare Team dummy is flagged (separation_terms) but
            # does not sink an otherwise-fittable model (e.g. points_finish).
            num_blown = any(
                abs(float(fit.params[t])) > 15 for t in fit.params.index
                if not t.startswith("C(") and t != "Intercept"
            )
            if not np.isfinite(fit.bse.to_numpy()).all() or num_blown:
                raise RuntimeError("non-finite SEs / separation on a numeric predictor")
            terms = [
                {"term": t, "coef": float(fit.params[t]), "se": float(fit.bse[t]),
                 "stat": float(fit.tvalues[t]), "p": float(fit.pvalues[t]),
                 "vif": np.nan, "partial_f": np.nan, "p_partial_f": np.nan}
                for t in fit.params.index
            ]
            separation_terms = [t["term"] for t in terms if abs(t["coef"]) > 10]
            return {
                "model": "gee_logit",
                "terms": terms,
                "summary": {
                    "n": int(fit.nobs), "k": int(len(terms)),
                    "n_clusters": int(d[cluster].nunique()),
                    "n_positive": n_pos,
                    "dropped_team": cats == (),
                    "qic": _safe_qic(fit),
                    "low_power": bool(low_power),
                    "separation_terms": ",".join(separation_terms) or None,
                    "converged": True,
                },
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("GEE (cats=%s) failed for %s: %s", cats, target, e)
    cause = ("low power / quasi-separation" if low_power
             else "separation (a predictor near-perfectly splits the outcome — "
                  "expected in the raw frame from the unshifted leaky rollups)")
    return {
        "model": "gee_logit", "terms": [],
        "summary": {
            "n": int(len(d)), "n_positive": n_pos, "n_clusters": int(d[cluster].nunique()),
            "low_power": bool(low_power), "converged": False,
            "note": f"no stable GEE fit — {cause} ({n_pos} positives in {len(d)} rows)",
        },
    }


def _safe_qic(fit) -> float:
    try:
        q = fit.qic()
        return float(q[0] if isinstance(q, (tuple, list, np.ndarray)) else q)
    except Exception:  # noqa: BLE001
        return np.nan


def multivariable_table(df: pd.DataFrame, specs: list[FeatureSpec], target: str) -> pd.DataFrame:
    """Run the appropriate multivariable models for ``target`` and flatten them
    into the ``_multivariable.csv`` schema (term rows + summary rows)."""
    numeric_full = [s.name for s in specs if s.kind == "numeric" and s.name in df.columns]
    numeric, dropped_collinear = _drop_to_full_rank(df, numeric_full)
    results: list[dict] = []

    if target == "Position":
        models = [ols_cluster_robust(df, numeric, target), mixedlm_driver(df, numeric, target)]
    else:
        models = [gee_logit(df, numeric, target)]

    for m in models:
        for t in m["terms"]:
            results.append({
                "target": target, "model": m["model"], "term": t["term"],
                "coef": t["coef"], "se": t["se"], "stat": t["stat"], "p": t["p"],
                "vif": t.get("vif", np.nan),
                "partial_f": t.get("partial_f", np.nan),
                "p_partial_f": t.get("p_partial_f", np.nan),
            })
        s = m["summary"]
        results.append({
            "target": target, "model": m["model"], "term": "__summary__",
            "coef": np.nan, "se": np.nan, "stat": np.nan, "p": np.nan, "vif": np.nan,
            "partial_f": np.nan, "p_partial_f": np.nan,
            "dropped_collinear": ",".join(dropped_collinear) or None,
            **{k: s.get(k) for k in
               ("n", "k", "r2", "adj_r2", "f", "p_f", "group_var", "n_clusters",
                "converged", "cond_number", "qic", "n_positive", "dropped_team",
                "degenerate", "low_power", "separation_terms", "note")},
        })
    return pd.DataFrame(results)


# --------------------------------------------------------------------------- #
# Phase 4 — model-based importance & significance (target = Position)
# --------------------------------------------------------------------------- #
def _position_pipeline(feature_cols: list[str], config: Config) -> Pipeline:
    """The exact position Pipeline from train.py:215-228."""
    import xgboost as xgb

    p = config.models.position
    categorical = [c for c in CATEGORICAL if c in feature_cols]
    numerical = [c for c in feature_cols if c not in categorical]
    pre = ColumnTransformer([
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([
        ("preprocessor", pre),
        ("regressor", xgb.XGBRegressor(
            n_estimators=p.n_estimators, learning_rate=p.learning_rate,
            max_depth=p.max_depth, random_state=p.random_state,
        )),
    ])


def _grouped_cv_mae(df: pd.DataFrame, feature_cols: list[str], n_splits: int, config: Config) -> float:
    x, y, groups = df[feature_cols], df["Position"], df[RACE_COL]
    n_splits = min(n_splits, int(groups.nunique()))
    scores = cross_val_score(
        _position_pipeline(feature_cols, config), x, y,
        cv=GroupKFold(n_splits=n_splits), groups=groups,
        scoring="neg_mean_absolute_error",
    )
    return float(-scores.mean())


def grouped_permutation_importance(
    df: pd.DataFrame, feature_cols: list[str], n_splits: int = 5,
    n_repeats: int = 50, config: Config | None = None, rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Per held-out GroupKFold(by Race) fold, permutation importance
    (neg-MAE scoring). Pool the ``n_splits * n_repeats`` per-feature MAE-increase
    deltas -> mean, std, one-sided empirical p. Native gain importance
    (OHE collapsed back to Driver/Team) is reported alongside.
    """
    config = get_config() if config is None else config
    rng = np.random.default_rng(42) if rng is None else rng
    seed = int(rng.integers(0, 2**31 - 1))

    x, y, groups = df[feature_cols], df["Position"], df[RACE_COL]
    n_splits = min(n_splits, int(groups.nunique()))
    gkf = GroupKFold(n_splits=n_splits)

    pooled: dict[str, list[float]] = {f: [] for f in feature_cols}
    for tr, te in gkf.split(x, y, groups):
        pipe = _position_pipeline(feature_cols, config)
        pipe.fit(x.iloc[tr], y.iloc[tr])
        r = permutation_importance(
            pipe, x.iloc[te], y.iloc[te], n_repeats=n_repeats,
            scoring="neg_mean_absolute_error", random_state=seed,
        )
        # permutation_importance reports (baseline - permuted) score; with
        # neg-MAE that is (-MAE_base) - (-MAE_perm) = MAE_perm - MAE_base, so a
        # positive delta means permuting the feature made MAE worse.
        for i, f in enumerate(feature_cols):
            pooled[f].extend(r.importances[i].tolist())

    gain = _gain_importance(df, feature_cols, config)
    n = n_splits * n_repeats
    rows = []
    for f in feature_cols:
        deltas = np.asarray(pooled[f], dtype=float)
        rows.append({
            "feature": f,
            "perm_importance_mean": float(np.mean(deltas)),
            "perm_importance_std": float(np.std(deltas)),
            "p_perm_importance": float((1 + np.sum(deltas <= 0)) / (n + 1)),
            "gain_importance": gain.get(f, np.nan),
        })
    return pd.DataFrame(rows)


def _gain_importance(df: pd.DataFrame, feature_cols: list[str], config: Config) -> dict[str, float]:
    pipe = _position_pipeline(feature_cols, config)
    pipe.fit(df[feature_cols], df["Position"])
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    imp = pipe.named_steps["regressor"].feature_importances_
    collapsed: dict[str, float] = {f: 0.0 for f in feature_cols}
    for name, val in zip(names, imp):
        raw = name.split("__", 1)[1] if "__" in name else name
        for f in feature_cols:
            if raw == f or raw.startswith(f + "_"):
                collapsed[f] += float(val)
                break
    return collapsed


def drop_one_cv_mae(
    df: pd.DataFrame, feature_cols: list[str], n_splits: int = 5, config: Config | None = None
) -> pd.DataFrame:
    """Full-model GroupKFold(by Race) CV MAE, then per feature: drop it, re-run
    the same CV, record ``cv_mae_without`` and ``delta_cv_mae``."""
    config = get_config() if config is None else config
    full = _grouped_cv_mae(df, feature_cols, n_splits, config)
    rows = []
    for f in feature_cols:
        reduced = [c for c in feature_cols if c != f]
        without = _grouped_cv_mae(df, reduced, n_splits, config)
        rows.append({
            "feature": f, "cv_mae_full": full, "cv_mae_without": without,
            "delta_cv_mae": without - full,
        })
    return pd.DataFrame(rows)


def model_based_table(
    df: pd.DataFrame, specs: list[FeatureSpec], config: Config,
    n_splits: int = 5, n_repeats: int = 50, rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    feats = [s.name for s in specs if s.name in df.columns]
    perm = grouped_permutation_importance(df, feats, n_splits, n_repeats, config, rng)
    drop = drop_one_cv_mae(df, feats, n_splits, config)
    return perm.merge(drop, on="feature", how="outer")


# --------------------------------------------------------------------------- #
# Phase 5 — leakage quantification
# --------------------------------------------------------------------------- #
def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def leakage_report(raw_df: pd.DataFrame, clean_df: pd.DataFrame, config: Config) -> dict:
    """The four Phase-5 checks; JSON-serializable."""
    out: dict = {}

    # 1. Position == GridPosition - PositionChange identity
    if {"GridPosition", "PositionChange"}.issubset(raw_df.columns):
        recon = (raw_df["GridPosition"] - raw_df["PositionChange"]).to_numpy(float)
        pos = raw_df["Position"].to_numpy(float)
        out["identity"] = {
            "expr": "Position == GridPosition - PositionChange",
            "r2": _r2(pos, recon),
            "max_abs_residual": float(np.max(np.abs(pos - recon))),
        }

    # 2. Position-model GroupKFold(by Race) MAE with vs without the leaky trio
    feats = [s.name for s in specs_for_frame(raw_df, "raw")]
    trio = [f for f in LEAKY_TRIO if f in feats]
    without_trio = [f for f in feats if f not in trio]
    mae_with = _grouped_cv_mae(raw_df, feats, 5, config)
    mae_without = _grouped_cv_mae(raw_df, without_trio, 5, config)
    out["leaky_trio_cv_mae"] = {
        "trio": trio,
        "cv_mae_with_trio": mae_with,
        "cv_mae_without_trio": mae_without,
        "ratio_without_over_with": mae_without / mae_with if mae_with else np.nan,
    }

    # 3. each unshifted rollup vs its shifted twin
    completed = config.constants.completed_statuses
    n_prev = config.pipeline.lookback_races
    shifted = add_shifted_history(raw_df, n_prev, completed) if "Status" in raw_df.columns else None
    rollup_rows = []
    for name in _ROLLUPS:
        if name not in raw_df.columns:
            continue
        row = {"rollup": name, "corr_unshifted_vs_position": _safe_corr(raw_df[name], raw_df["Position"])}
        if shifted is not None and name in shifted.columns:
            m = raw_df[["Driver", RACE_COL, name, "Position"]].merge(
                shifted[["Driver", RACE_COL, name]], on=["Driver", RACE_COL],
                suffixes=("_unshifted", "_shifted"),
            )
            row["corr_shifted_vs_position"] = _safe_corr(m[f"{name}_shifted"], m["Position"])
            row["corr_shifted_vs_unshifted"] = _safe_corr(m[f"{name}_shifted"], m[f"{name}_unshifted"])
        rollup_rows.append(row)
    out["rollup_shift_comparison"] = rollup_rows

    # 4. driver_win_rate / team_reliability degeneracy
    deg = {}
    for feat, grp in (("driver_win_rate", "Driver"), ("team_reliability", "Team")):
        if feat in raw_df.columns:
            nun = raw_df.groupby(grp)[feat].nunique(dropna=False)
            deg[feat] = {
                "group": grp,
                "constant_within_group": bool(nun.max() <= 1),
                "max_distinct_per_group": int(nun.max()),
            }
    if "driver_win_rate" in raw_df.columns:
        mean_pos = raw_df.groupby("Driver")["Position"].mean()
        wr = raw_df.groupby("Driver")["driver_win_rate"].first()
        deg["driver_win_rate"]["corr_with_driver_mean_finish"] = _safe_corr(wr, mean_pos)
    out["rate_feature_degeneracy"] = deg

    return json.loads(json.dumps(out, default=_json_default))


def _safe_corr(a, b) -> float:
    a = pd.Series(np.asarray(a, dtype=float))
    b = pd.Series(np.asarray(b, dtype=float))
    m = a.notna() & b.notna()
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


# --------------------------------------------------------------------------- #
# Phase 6 — orchestration + report
# --------------------------------------------------------------------------- #
def run_feature_analysis(
    config: Config | None = None,
    *,
    targets: tuple[str, ...] = ALL_TARGETS,
    frames: tuple[str, ...] = ("raw", "clean"),
    run_model: bool = True,
    n_perm: int = 2000,
    perm_importance_repeats: int = 50,
    cv_splits: int = 5,
    seed: int = 42,
    make_plots: bool = False,
    drop_rates: bool = False,
    out_dir=None,
) -> dict:
    """Run the whole battery and return a dict of DataFrames plus a rendered
    markdown report:

        {"univariate": df, "multivariable": df, "model": df, "leakage": dict,
         "frames": {kind: df}, "markdown": str}
    """
    config = get_config() if config is None else config
    rng = np.random.default_rng(seed)

    results = pd.read_csv(config.paths.data_dir / "f1_results_features.csv")

    built: dict[str, pd.DataFrame] = {}
    if "raw" in frames:
        built["raw"] = build_raw_frame(results, config)
    if "clean" in frames:
        built["clean"] = build_clean_frame(results, config, drop_rates=drop_rates)

    uni_parts, multi_parts, model_parts = [], [], []
    for kind, frame in built.items():
        specs = specs_for_frame(frame, kind)
        for target in targets:
            if target not in frame.columns or frame[target].nunique() < 2:
                continue
            u = univariate_tests(frame, specs, target, n_perm=n_perm, rng=rng)
            if not u.empty:
                u.insert(0, "frame", kind)
                uni_parts.append(u)
            try:
                m = multivariable_table(frame, specs, target)
                if not m.empty:
                    m.insert(0, "frame", kind)
                    multi_parts.append(m)
            except Exception as e:  # noqa: BLE001
                logger.warning("multivariable failed (%s/%s): %s", kind, target, e)

        if run_model and "Position" in frame.columns:
            try:
                mb = model_based_table(
                    frame, specs, config,
                    n_splits=cv_splits, n_repeats=perm_importance_repeats, rng=rng,
                )
                mb.insert(0, "frame", kind)
                model_parts.append(mb)
            except Exception as e:  # noqa: BLE001
                logger.warning("model-based phase failed (%s): %s", kind, e)

    leakage = {}
    if "raw" in built and "clean" in built:
        leakage = leakage_report(built["raw"], built["clean"], config)

    uni = pd.concat(uni_parts, ignore_index=True) if uni_parts else pd.DataFrame()
    multi = pd.concat(multi_parts, ignore_index=True) if multi_parts else pd.DataFrame()
    model = pd.concat(model_parts, ignore_index=True) if model_parts else pd.DataFrame()

    markdown = _render_report(built, uni, multi, model, leakage, config,
                              targets=targets, n_perm=n_perm, seed=seed, run_model=run_model)

    if make_plots:
        try:
            _make_plots(built, uni, model, out_dir or (config.paths.plots_dir / "feature_stats"))
        except Exception as e:  # noqa: BLE001
            logger.warning("plotting failed: %s", e)

    return {
        "frames": built,
        "univariate": uni,
        "multivariable": multi,
        "model": model,
        "leakage": leakage,
        "markdown": markdown,
    }


def _overall_call(row: pd.Series, model_df: pd.DataFrame, multi_df: pd.DataFrame) -> str:
    if row["leaky"] and row.get("significant"):
        return "leakage-driven"
    uni_sig = bool(row.get("significant"))
    m = model_df[(model_df.get("frame") == row["frame"]) & (model_df["feature"] == row["feature"])] \
        if not model_df.empty else pd.DataFrame()
    model_sig = (not m.empty and (m["p_perm_importance"].iloc[0] < 0.05)
                 and (m["delta_cv_mae"].iloc[0] > 0))
    if uni_sig and model_sig:
        return "strong"
    if uni_sig or model_sig:
        return "moderate"
    if row.get("p_perm", 1.0) < 0.10:
        return "weak"
    return "none"


def _render_report(built, uni, multi, model, leakage, config, *, targets, n_perm, seed, run_model):
    L: list[str] = []
    w = L.append
    w("# Statistical hypothesis testing of the F1_predictor features\n")
    w(f"_seed={seed}, permutations={n_perm}, model phase={'on' if run_model else 'off'}, "
      f"statsmodels-backed multivariable inference._\n")

    # 1. caveats
    w("## 1. Statistical validity limits\n")
    for line in _CAVEATS:
        w(f"- {line}")
    w("")

    # 2. data
    w("## 2. Data\n")
    w("| frame | rows | drivers | teams | races |")
    w("| --- | --- | --- | --- | --- |")
    for kind, f in built.items():
        w(f"| {kind} | {len(f)} | {f['Driver'].nunique()} | {f['Team'].nunique()} "
          f"| {f[RACE_COL].nunique()} |")
    w("")
    for t in targets:
        for kind, f in built.items():
            if t in f.columns and t != "Position":
                rate = f[t].mean()
                w(f"- `{t}` ({kind}): base rate {rate:.1%} ({int(f[t].sum())}/{len(f)})")
    w("")

    # 3. verdict table for Position
    w("## 3. Verdict — target `Position`\n")
    if not uni.empty:
        prim = uni[(uni["target"] == "Position") & (uni["test"].isin(["spearman", "kruskal_wallis"]))]
        w("| feature | frame | q_perm | effect | direction | leaky | overall |")
        w("| --- | --- | --- | --- | --- | --- | --- |")
        for _, r in prim.sort_values(["frame", "q_perm_bh"]).iterrows():
            call = _overall_call(r, model, multi)
            q = r["q_perm_bh"]
            w(f"| {r['feature']} | {r['frame']} | {q:.3g} | "
              f"{r['effect_size_name']}={_fmt(r['effect_size'])} | {r['direction']} | "
              f"{'yes' if r['leaky'] else 'no'} | {call} |")
    w("")

    # 4. univariate full tables
    w("## 4. Univariate association tests\n")
    if not uni.empty:
        for (kind, t), g in uni.groupby(["frame", "target"]):
            w(f"### {kind} / {t}\n")
            w("| feature | test | stat | effect | n | p_analytic | q_analytic | "
              "perm | p_perm | q_perm | sig |")
            w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for _, r in g.iterrows():
                w(f"| {r['feature']} | {r['test']} | {_fmt(r['statistic'])} | "
                  f"{_fmt(r['effect_size'])} | {r['n']} | {_fmt(r['p_analytic'])} | "
                  f"{_fmt(r['q_analytic_bh'])} | {r['perm_scheme']} | {_fmt(r['p_perm'])} | "
                  f"{_fmt(r['q_perm_bh'])} | {'*' if r['significant'] else ''} |")
            w("")

    # 5. multivariable
    w("## 5. Multivariable inference\n")
    if not multi.empty:
        for (kind, t, mdl), g in multi.groupby(["frame", "target", "model"]):
            w(f"### {kind} / {t} — {mdl}\n")
            terms = g[g["term"] != "__summary__"]
            w("| term | coef | se | stat | p | vif |")
            w("| --- | --- | --- | --- | --- | --- |")
            for _, r in terms.iterrows():
                w(f"| {r['term']} | {_fmt(r['coef'])} | {_fmt(r['se'])} | {_fmt(r['stat'])} "
                  f"| {_fmt(r['p'])} | {_fmt(r['vif'])} |")
            s = g[g["term"] == "__summary__"]
            if not s.empty:
                sr = s.iloc[0].to_dict()
                bits = [f"{k}={_fmt(sr[k])}" for k in
                        ("n", "k", "r2", "adj_r2", "f", "p_f", "group_var",
                         "n_clusters", "converged", "cond_number", "qic", "n_positive",
                         "dropped_team", "degenerate", "low_power")
                        if k in sr and sr[k] is not None and pd.notna(sr[k])]
                w("\n_" + ", ".join(bits) + "_")
                if sr.get("dropped_collinear"):
                    w(f"\n_dropped (collinear): {sr['dropped_collinear']}_")
                if sr.get("separation_terms"):
                    w(f"\n_quasi-separated terms (|coef|>10, unreliable): {sr['separation_terms']}_")
                if sr.get("note"):
                    w(f"\n_{sr['note']}_")
                if sr.get("degenerate"):
                    w("\n_**degenerate fit** — a leaky predictor reconstructs the target; "
                      "per-term p-values are not interpretable._")
                w("")

    # 6. model-based importance
    w("## 6. Model-based importance (GroupKFold by Race, target `Position`)\n")
    if not model.empty:
        for kind, g in model.groupby("frame"):
            w(f"### {kind}\n")
            w("| feature | perm_imp_mean | perm_imp_std | p | gain | cv_mae_full | "
              "cv_mae_without | delta_cv_mae |")
            w("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for _, r in g.sort_values("perm_importance_mean", ascending=False).iterrows():
                w(f"| {r['feature']} | {_fmt(r['perm_importance_mean'])} | "
                  f"{_fmt(r['perm_importance_std'])} | {_fmt(r['p_perm_importance'])} | "
                  f"{_fmt(r['gain_importance'])} | {_fmt(r['cv_mae_full'])} | "
                  f"{_fmt(r['cv_mae_without'])} | {_fmt(r['delta_cv_mae'])} |")
            w("")
    else:
        w("_model phase not run._\n")

    # 7. leakage
    w("## 7. Leakage\n")
    if leakage:
        idy = leakage.get("identity", {})
        if idy:
            w(f"- Identity `{idy['expr']}`: R^2 = {_fmt(idy['r2'])}, "
              f"max abs residual = {_fmt(idy['max_abs_residual'])}")
        lt = leakage.get("leaky_trio_cv_mae", {})
        if lt:
            w(f"- Position-model CV MAE **with** leaky trio {lt['trio']} = "
              f"{_fmt(lt['cv_mae_with_trio'])}; **without** = {_fmt(lt['cv_mae_without_trio'])} "
              f"(ratio {_fmt(lt['ratio_without_over_with'])})")
        w("")
        w("| rollup | corr(unshifted, Position) | corr(shifted, Position) | corr(shifted, unshifted) |")
        w("| --- | --- | --- | --- |")
        for r in leakage.get("rollup_shift_comparison", []):
            w(f"| {r['rollup']} | {_fmt(r.get('corr_unshifted_vs_position'))} | "
              f"{_fmt(r.get('corr_shifted_vs_position'))} | {_fmt(r.get('corr_shifted_vs_unshifted'))} |")
        w("")
        for feat, d in leakage.get("rate_feature_degeneracy", {}).items():
            w(f"- `{feat}`: constant within {d['group']} = {d['constant_within_group']}"
              + (f", corr with {d['group']} mean finish = {_fmt(d.get('corr_with_driver_mean_finish'))}"
                 if "corr_with_driver_mean_finish" in d else ""))
    w("")

    # 8. method notes
    w("## 8. Method notes\n")
    for line in _METHOD_NOTES:
        w(f"- {line}")
    w("")
    return "\n".join(L) + "\n"


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == 0:
        return "0"
    if abs(f) < 1e-3 or abs(f) >= 1e5:
        return f"{f:.2e}"
    return f"{f:.3f}"


_CAVEATS = [
    "n ~ 215-286, **one** 2026 season, 13 rounds -> no genuine temporal hold-out; "
    "GroupKFold(by Race) is the best available and still optimistic.",
    "Repeated measures: ~23 driver / ~11 team clusters; within a race the finishing "
    "positions are a fixed permutation of 1..22. Hence within-Race permutation p, "
    "cluster-robust (by Driver) SEs, GEE for binary panels, and the (1|Driver) mixed model.",
    "With only ~13-23 clusters even cluster-robust SEs are somewhat optimistic — treat "
    "p just under 0.05 as \"suggestive\".",
    "~19 features x 4 targets -> BH-FDR within each (frame, target, test-family); "
    "report raw p and q, decide on q.",
    "The data looks simulated (single clean synthetic-looking season) — findings may "
    "reflect the generator, not real F1.",
    "Leakage dominates the naive analysis by construction; the dual raw/clean design "
    "is what makes the honest conclusions readable.",
]

_METHOD_NOTES = [
    "Permutation p: target shuffled within each Race (within Driver when the feature is "
    "race-constant, e.g. the rate features); p = (1 + #{stat_perm >= stat_obs}) / (B + 1).",
    "BH-FDR (scipy.stats.false_discovery_control, method='bh') applied within each "
    "(frame, target, test) family; analytic and permutation p are separate families.",
    "Multivariable: numeric predictors z-scored (matches the model's StandardScaler); "
    "Team via C(Team); Driver excluded from the regressions (23 levels on ~215 rows).",
    "OLS SEs cluster-robust by Driver; mixed model adds a random intercept by Driver; "
    "binary targets use population-averaged GEE logistic (exchangeable working correlation).",
    "Model-based importance rebuilds the exact position Pipeline (StandardScaler + "
    "OneHotEncoder + XGBRegressor) under GroupKFold(by Race), replacing train.py's leaky cv=5.",
]


def _make_plots(built, uni, model, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not model.empty:
        for kind, g in model.groupby("frame"):
            g = g.sort_values("perm_importance_mean")
            fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(g))))
            ax.barh(g["feature"], g["perm_importance_mean"], xerr=g["perm_importance_std"])
            ax.set_title(f"Permutation importance (MAE increase) — {kind}")
            ax.set_xlabel("mean MAE increase when permuted")
            fig.tight_layout()
            fig.savefig(out / f"perm_importance_{kind}.png", dpi=120)
            plt.close(fig)

    if not uni.empty:
        g = uni[uni["target"] == "Position"].dropna(subset=["p_perm", "q_perm_bh"])
        if not g.empty:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(g["p_perm"], g["q_perm_bh"], s=12)
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.set_xlabel("p_perm")
            ax.set_ylabel("q_perm_bh")
            ax.set_title("p vs q (BH-FDR) — Position")
            fig.tight_layout()
            fig.savefig(out / "p_vs_q_position.png", dpi=120)
            plt.close(fig)

    for kind, f in built.items():
        num = [c for c in ("GridPosition", "avg_grid_last", "form_trend") if c in f.columns]
        if not num:
            continue
        fig, axes = plt.subplots(1, len(num), figsize=(4 * len(num), 4), squeeze=False)
        for ax, c in zip(axes[0], num):
            ax.scatter(f[c], f["Position"], s=10, alpha=0.5)
            ax.set_xlabel(c)
            ax.set_ylabel("Position")
        fig.suptitle(f"feature vs Position — {kind}")
        fig.tight_layout()
        fig.savefig(out / f"scatter_{kind}.png", dpi=120)
        plt.close(fig)


def write_outputs(analysis: dict, out_dir) -> dict:
    """Persist the analysis dict to ``out_dir``; return {name: path}."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    md = out / "feature_hypothesis_tests.md"
    md.write_text(analysis["markdown"])
    paths["markdown"] = str(md)

    for key, fname in (
        ("univariate", "feature_hypothesis_tests.csv"),
        ("multivariable", "feature_hypothesis_tests_multivariable.csv"),
        ("model", "feature_hypothesis_tests_model.csv"),
    ):
        df = analysis.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(out / fname, index=False)
            paths[key] = str(out / fname)

    lj = out / "feature_hypothesis_tests_leakage.json"
    lj.write_text(json.dumps(analysis.get("leakage", {}), indent=2, default=_json_default))
    paths["leakage"] = str(lj)
    return paths


def _asdict_specs(specs: list[FeatureSpec]) -> list[dict]:
    return [asdict(s) for s in specs]
