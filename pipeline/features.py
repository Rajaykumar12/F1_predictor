"""Feature engineering.

Two independent feature paths live here:

* **Lap-time model** — :func:`engineer_lap_features` (unchanged).
* **Position model** — :func:`build_position_features`, the single code path used
  by training, prediction, evaluation and backtest. It is *leakage-safe by
  construction*: every rolling window is ``.shift(shift)``-ed per driver before it
  is computed, every season-to-date rate is expanding over races **strictly
  before** the target race, and there is no target-derived column. The exact set
  of columns it emits is declared in :mod:`pipeline.feature_registry`.

The legacy whole-season ``driver_win_rate`` / ``team_reliability`` columns are
still written by :func:`engineer_result_features` because the *race-win* and
*lap-time* models consume them; the **position** model no longer does (it reads
its feature list from the registry). Reworking those two models is out of scope
for the feature redesign — see ``docs/feature-engineering-redesign-plan.md``.
"""

from __future__ import annotations

import json
import logging
from typing import List

import numpy as np
import pandas as pd

from pipeline import feature_registry as fr
from pipeline.config_loader import Config, get_config

logger = logging.getLogger(__name__)

_TIRE_ENCODE = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4, "WET": 5}


# =========================================================================== #
# Lap-time model features (unchanged)
# =========================================================================== #
def engineer_lap_features(df: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    phase_bins = config.constants.race_phase_bins if config else [0, 15, 40, 100]
    pos_bins = config.constants.position_bins if config else [0, 5, 10, 15, 20]

    df = df.copy()

    df["start_position"] = df.groupby(["Driver", "Race"])["Position"].transform("first")
    df["positions_gained"] = df["start_position"] - df["Position"]

    df["tire_degradation"] = df.groupby(
        ["Driver", "Race", "TireCompound"]
    )["LapTime_seconds"].diff()
    df["tire_degradation"] = df["tire_degradation"].fillna(df["tire_degradation"].median())

    df["race_phase"] = pd.cut(
        df["LapNumber"],
        bins=phase_bins,
        labels=["Early", "Middle", "Late"],
    )

    df["PositionGroup"] = pd.cut(
        df["Position"],
        bins=pos_bins,
        labels=["Top 5", "6-10", "11-15", "16-20"],
    )

    # --- Additional features for the laptime model ---
    df["TireCompound_encoded"] = df["TireCompound"].str.upper().map(_TIRE_ENCODE).fillna(2)
    df["IsFreshTire"] = (df["TireAge"] <= 3).astype(int)
    df["StintLapNumber"] = df["TireAge"]

    total_laps = df.groupby(["Driver", "Race"])["LapNumber"].transform("max")
    df["LapNumber_normalized"] = df["LapNumber"] / total_laps.clip(lower=1)
    df["FuelLoadProxy"] = (total_laps - df["LapNumber"]) / total_laps.clip(lower=1)

    # Outlap: first lap on a new set of tires (TireAge == 1)
    df["IsOutlap"] = (df["TireAge"] == 1).astype(int)
    # Inlap: the lap immediately before an outlap (next lap is TireAge == 1)
    next_tire_age = df.groupby(["Driver", "Race"])["TireAge"].shift(-1)
    df["IsInlap"] = (next_tire_age == 1).fillna(False).astype(int)

    # Rolling lap time statistics per driver per race
    grp = df.groupby(["Driver", "Race"])["LapTime_seconds"]
    df["RollingAvgLapTime_3"] = grp.transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["RollingAvgLapTime_5"] = grp.transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["LapTimeStd_5"] = grp.transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0))

    return df


# =========================================================================== #
# Legacy whole-season rates — still feed the race-win / lap-time models only
# =========================================================================== #
def _compute_driver_win_rate(results: pd.DataFrame) -> pd.Series:
    wins = results[results["Position"] == 1]["Driver"].value_counts()
    total = results["Driver"].value_counts()
    return (wins / total * 100).fillna(0).rename("driver_win_rate")


def _compute_team_reliability(
    results: pd.DataFrame, completed_statuses: List[str] | None = None
) -> pd.Series:
    if completed_statuses is None:
        completed_statuses = ["Finished"]
    return (
        results.groupby("Team")["Status"]
        .apply(lambda x: x.isin(completed_statuses).mean() * 100)
        .rename("team_reliability")
    )


def engineer_result_features(
    laps: pd.DataFrame,
    results: pd.DataFrame,
    completed_statuses: List[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach the legacy whole-season rate columns (race-win / lap-time models)
    and the binary targets. The old ``PositionChange`` column — which equalled
    ``GridPosition - Position`` exactly and leaked the target — is **not** written
    any more.
    """
    driver_win_rate = _compute_driver_win_rate(results)
    team_reliability = _compute_team_reliability(results, completed_statuses)

    laps = laps.merge(driver_win_rate, left_on="Driver", right_index=True, how="left")
    results = results.merge(driver_win_rate, left_on="Driver", right_index=True, how="left")
    laps["driver_win_rate"] = laps["driver_win_rate"].fillna(0)
    results["driver_win_rate"] = results["driver_win_rate"].fillna(0)

    laps = laps.merge(team_reliability, left_on="Team", right_index=True, how="left")
    results = results.merge(team_reliability, left_on="Team", right_index=True, how="left")

    results["race_winner"] = (results["Position"] == 1).astype(int)
    results["podium_finish"] = (results["Position"] <= 3).astype(int)
    results["points_finish"] = (results["Position"] <= 10).astype(int)

    return laps, results


# =========================================================================== #
# Position model — leakage-safe rolling history
# =========================================================================== #
def _shifted_roll(
    df: pd.DataFrame,
    value_col: str,
    window: int,
    shift: int,
    how: str = "mean",
    group: str = "Driver",
    order: str = "Race",
) -> pd.Series:
    """Per ``group`` (sorted by ``order``), apply ``.shift(shift)`` **before** a
    ``.rolling(window, min_periods=1).<how>()``. With ``shift >= 1`` the current
    row is never inside its own window, so the value is strictly historical. The
    first ``shift`` rows of each group are NaN by construction."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, g in df.groupby(group, sort=False):
        g = g.sort_values(order)
        rolled = getattr(g[value_col].shift(shift).rolling(window, min_periods=1), how)()
        out.loc[g.index] = rolled.to_numpy()
    return out


def _expanding_pre_race_rate(df: pd.DataFrame, group: str, event: pd.Series) -> pd.Series:
    """Mean of ``event`` (0/1) over ``group``'s rows in races **strictly before**
    the current race. Race-aware: a team's two same-race rows never see each
    other. First race for a group -> NaN. Returns a 0-1 rate."""
    tmp = pd.DataFrame(
        {"_g": df[group].to_numpy(), "_r": df["Race"].to_numpy(),
         "_e": np.asarray(event, dtype=float)},
        index=df.index,
    )
    per = tmp.groupby(["_g", "_r"])["_e"].agg(["sum", "count"]).sort_index()
    cum_sum = per.groupby(level=0)["sum"].cumsum() - per["sum"]
    cum_cnt = per.groupby(level=0)["count"].cumsum() - per["count"]
    rate = (cum_sum / cum_cnt.replace(0, np.nan))
    rate.name = "_rate"
    return tmp.join(rate, on=["_g", "_r"])["_rate"]


def create_historical_features(
    df: pd.DataFrame,
    n_previous: int = 6,
    completed_statuses: List[str] | None = None,
    as_of_round: int | None = None,
    shift: int = 1,
) -> pd.DataFrame:
    """Per-driver rolling form, **shift-before-roll**.

    For every driver (sorted by ``Race``) a ``.shift(shift)`` is applied before
    each ``.rolling(n_previous, min_periods=1)`` window, so with the default
    ``shift=1`` the race being predicted is never part of its own history.
    ``shift=0`` reproduces the old leaky windows and is used only by the
    hypothesis-testing "raw" regression guard.

    ``as_of_round`` truncates every driver's history to ``Race <= as_of_round``
    before the windows are computed (an honest "form going into the next round"
    snapshot).

    The legacy rollup column names (``avg_position_last``, ``dnf_last``,
    ``podiums_last`` …) are still produced for display / backward compatibility
    (``predict.py`` recent-form panel, ``evaluate.py`` row filter). The position
    **model** no longer selects them — it reads
    :func:`pipeline.feature_registry.enabled_features`.
    """
    if completed_statuses is None:
        completed_statuses = ["Finished"]

    frames = []
    for driver in df["Driver"].unique():
        d = df[df["Driver"] == driver].copy()
        d = d.sort_values("Race").reset_index(drop=True)

        if as_of_round is not None and "Race" in d.columns:
            d = d[d["Race"] <= as_of_round].reset_index(drop=True)
            if d.empty:
                continue

        def roll(series: pd.Series, how: str = "mean", w: int = n_previous):
            return getattr(series.shift(shift).rolling(w, min_periods=1), how)()

        d["is_dnf"] = (~d["Status"].isin(completed_statuses)).astype(int)
        d["positions_gained"] = d["GridPosition"] - d["Position"]

        d["avg_position_last"] = roll(d["Position"], "mean")
        d["best_position_last"] = roll(d["Position"], "min")
        d["avg_grid_last"] = roll(d["GridPosition"], "mean")
        d["dnf_last"] = roll(d["is_dnf"], "sum")
        d["reliability_rate"] = 1 - (d["dnf_last"] / n_previous)
        d["avg_positions_gained"] = roll(d["positions_gained"], "mean")
        d["podiums_last"] = roll((d["Position"] <= 3).astype(int), "sum")
        d["wins_last"] = roll((d["Position"] == 1).astype(int), "sum")
        d["points_last"] = roll(d["Points"], "sum")

        if "BestQualifyingTime" in d.columns:
            d["avg_quali_time"] = roll(d["BestQualifyingTime"], "mean")
            d["avg_gap_to_pole"] = roll(d["GapToPole"], "mean")

        # form_trend keeps its own shift(3) construction: older-window minus
        # recent-window mean Position. Larger => improving form.
        recent_avg = d["Position"].rolling(3, min_periods=1).mean()
        if n_previous > 3:
            older_avg = d["Position"].shift(3).rolling(n_previous - 3, min_periods=1).mean()
            d["form_trend"] = older_avg - recent_avg
        else:
            d["form_trend"] = 0.0

        frames.append(d)

    return pd.concat(frames, ignore_index=True)


# =========================================================================== #
# Registry-driven family builders — each is pure: (df, config) -> df + new cols
# =========================================================================== #
def build_quali_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Qualifying-derived signals. All known once qualifying is complete, before
    the race — every column is as-of-safe."""
    df = df.copy()

    if "BestQualifyingTime" in df.columns:
        pole = df.groupby("Race")["BestQualifyingTime"].transform("min")
        gap = df["GapToPole"] if "GapToPole" in df.columns else (df["BestQualifyingTime"] - pole)
        with np.errstate(divide="ignore", invalid="ignore"):
            df["quali_gap_to_pole_pct"] = np.where(pole > 0, gap / pole * 100.0, np.nan)

        # teammate delta: driver best-quali minus the mean of same-race, same-team
        # team-mates (2-car teams => just the one team-mate).
        team_mean = df.groupby(["Race", "Team"])["BestQualifyingTime"].transform("mean")
        team_cnt = df.groupby(["Race", "Team"])["BestQualifyingTime"].transform("count")
        # mean of *others* = (sum - self) / (count - 1)
        others_mean = np.where(
            team_cnt > 1,
            (team_mean * team_cnt - df["BestQualifyingTime"]) / (team_cnt - 1),
            np.nan,
        )
        df["quali_gap_to_teammate_s"] = df["BestQualifyingTime"] - others_mean
        df["quali_beat_teammate"] = np.where(
            df["quali_gap_to_teammate_s"].notna(),
            (df["quali_gap_to_teammate_s"] < 0).astype(float),
            np.nan,
        )
    else:
        for c in ("quali_gap_to_pole_pct", "quali_gap_to_teammate_s", "quali_beat_teammate"):
            df[c] = np.nan

    if "q3_reached" not in df.columns:
        # Fallback when the qualifying merge did not carry it: infer from grid /
        # quali rank <= 10. Logged so the data path can be fixed upstream.
        if "QualifyingPerformance" in df.columns:
            df["q3_reached"] = (df["QualifyingPerformance"] <= 50.0).astype(float)
        elif "GridPosition" in df.columns:
            df["q3_reached"] = (df["GridPosition"] <= 10).astype(float)
        else:
            df["q3_reached"] = np.nan
        logger.warning("q3_reached not present from qualifying merge — inferred from rank.")
    else:
        df["q3_reached"] = df["q3_reached"].astype(float)

    return df


def build_form_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Per-driver rolling finishing form (shift-before-roll)."""
    df = df.copy()
    s = config.features.shift
    completed = config.constants.completed_statuses

    df["form_avg_finish_s5"] = _shifted_roll(df, "Position", 5, s, "mean")

    if "is_dnf" not in df.columns:
        df["is_dnf"] = (~df["Status"].isin(completed)).astype(int)
    df["form_dnf_rate_s8"] = _shifted_roll(df, "is_dnf", 8, s, "mean")

    if "form_trend" not in df.columns:
        # create_historical_features normally supplies this; recompute if a
        # caller invoked the builder directly.
        n_prev = config.features.lookback_races
        parts = []
        for _, g in df.groupby("Driver", sort=False):
            g = g.sort_values("Race")
            recent = g["Position"].rolling(3, min_periods=1).mean()
            if n_prev > 3:
                older = g["Position"].shift(3).rolling(n_prev - 3, min_periods=1).mean()
                g = g.assign(form_trend=older - recent)
            else:
                g = g.assign(form_trend=0.0)
            parts.append(g)
        df = pd.concat(parts).sort_index()

    return df


def build_racecraft_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Causal replacement for the deleted, target-identical ``PositionChange``:
    the driver's grid-to-finish delta averaged over *prior* races only."""
    df = df.copy()
    s = config.features.shift
    if "positions_gained" not in df.columns:
        df["positions_gained"] = df["GridPosition"] - df["Position"]
    df["hist_positions_gained_s5"] = _shifted_roll(df, "positions_gained", 5, s, "mean")
    df["hist_grid_finish_consistency_s5"] = _shifted_roll(df, "positions_gained", 5, s, "std")
    return df


def build_team_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Team-level (both cars) strength going into the round."""
    df = df.copy()

    # mean finishing Position of the team over the previous up-to-5 races
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, g in df.groupby("Team", sort=False):
        race_mean = g.groupby("Race")["Position"].mean().sort_index()
        rolled = race_mean.shift(1).rolling(5, min_periods=1).mean()
        out.loc[g.index] = g["Race"].map(rolled).to_numpy()
    df["team_form_avg_finish_s5"] = out

    # this round's team qualifying pace, ranked across teams (1 = fastest)
    if "GapToPole" in df.columns:
        team_pace = df.groupby(["Race", "Team"])["GapToPole"].transform("mean")
        df["_team_pace"] = team_pace
        df["team_quali_pace_rank"] = (
            df.groupby("Race")["_team_pace"].rank(method="dense", ascending=True)
        )
        df = df.drop(columns="_team_pace")
    else:
        df["team_quali_pace_rank"] = np.nan
    return df


def build_reliability_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Season-to-date reliability, expanding over races strictly before this one."""
    df = df.copy()
    completed = config.constants.completed_statuses
    if "is_dnf" not in df.columns:
        df["is_dnf"] = (~df["Status"].isin(completed)).astype(int)
    finished = df["Status"].isin(completed).astype(float)

    df["driver_dnf_rate_todate"] = _expanding_pre_race_rate(df, "Driver", df["is_dnf"].astype(float))
    df["team_reliability_todate"] = _expanding_pre_race_rate(df, "Team", finished)
    return df


def load_circuits(config: Config) -> pd.DataFrame:
    """Static circuit reference (``data/circuits.csv``), keyed by season round."""
    path = config.paths.data_dir / "circuits.csv"
    if not path.exists():
        logger.warning("data/circuits.csv missing — circuit features will be NaN/median-filled.")
        return pd.DataFrame(columns=["round", "circuit_overtaking_index",
                                     "circuit_is_street", "circuit_sc_probability"])
    return pd.read_csv(path)


def build_circuit_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Attach curated circuit metadata by round. A round missing from
    ``circuits.csv`` gets the column median and a logged warning."""
    df = df.copy()
    circuits = load_circuits(config)
    cols = ["circuit_overtaking_index", "circuit_is_street", "circuit_sc_probability"]

    # Idempotent: build_position_features may run on a frame that already carries
    # these columns (baked into f1_results_features.csv). Drop them before the
    # merge so we don't get _x / _y suffix collisions.
    df = df.drop(columns=[c for c in cols if c in df.columns])

    if circuits.empty:
        for c in cols:
            df[c] = np.nan
        return df

    merged = df.merge(
        circuits[["round"] + cols], left_on="Race", right_on="round", how="left"
    ).drop(columns=["round"])

    missing_rounds = sorted(set(df["Race"]) - set(circuits["round"]))
    if missing_rounds:
        logger.warning("circuits.csv has no row for round(s) %s — median-filling.", missing_rounds)
    for c in cols:
        med = circuits[c].median()
        merged[c] = merged[c].fillna(med)
    merged.index = df.index
    return merged


def build_championship_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Points gap to the championship leader **before** the current round —
    fully causal (uses only completed earlier races)."""
    df = df.copy()
    d = df.sort_values(["Driver", "Race"])
    cum_before = d.groupby("Driver")["Points"].cumsum() - d["Points"]
    cum_before = cum_before.reindex(df.index)
    leader_before = cum_before.groupby(df["Race"]).transform("max")
    df["driver_points_gap_to_leader_before"] = leader_before - cum_before
    return df


_BUILDERS = {
    "quali": build_quali_features,
    "form": build_form_features,
    "racecraft": build_racecraft_features,
    "team": build_team_features,
    "reliability": build_reliability_features,
    "circuit": build_circuit_features,
    "championship": build_championship_features,
}


# =========================================================================== #
# The single position-model feature path
# =========================================================================== #
def build_position_features(
    results_df: pd.DataFrame,
    config: Config | None = None,
    as_of_round: int | None = None,
    lookback: int | None = None,
    shift: int | None = None,
) -> pd.DataFrame:
    """Assemble every enabled registry feature on ``results_df``.

    This is the ONE code path shared by training (:mod:`pipeline.train`),
    prediction (:mod:`pipeline.predict`), evaluation (:mod:`pipeline.evaluate`)
    and backtest (``scripts/backtest.py``), so train-time and predict-time
    features can never diverge. ``as_of_round`` forwards to
    :func:`create_historical_features` for an honest pre-round snapshot.
    """
    config = config or get_config()
    if shift is not None and shift != config.features.shift:
        import dataclasses
        config = dataclasses.replace(
            config, features=dataclasses.replace(config.features, shift=shift)
        )

    df = create_historical_features(
        results_df,
        n_previous=lookback or config.features.lookback_races,
        completed_statuses=config.constants.completed_statuses,
        as_of_round=as_of_round,
        shift=config.features.shift,
    )

    for key in fr.by_builder(fr.enabled_features(config)):
        if key == "context":
            continue
        df = _BUILDERS[key](df, config)

    # Neutral fills for "not enough history yet" quantities: a trend / volatility
    # of 0 means "no movement observed", which is the honest default and avoids
    # discarding a driver's early-season rows over a derived NaN. Level features
    # (avg finish, rates) keep their NaN — those rows genuinely have no form.
    for col, neutral in (("form_trend", 0.0), ("hist_grid_finish_consistency_s5", 0.0)):
        if col in df.columns:
            df[col] = df[col].fillna(neutral)

    return df


def feature_manifest(df: pd.DataFrame, config: Config) -> dict:
    """Provenance sidecar for ``data/f1_results_features.csv``."""
    feats = fr.enabled_features(config)
    names = fr.feature_names(feats)
    present = [n for n in names if n in df.columns]
    return {
        "registry_version": fr.REGISTRY_VERSION,
        "season": config.pipeline.season,
        "row_count": int(len(df)),
        "shift": config.features.shift,
        "lookback_races": config.features.lookback_races,
        "families_enabled": sorted({f.family for f in feats}),
        "features": names,
        "features_missing_from_frame": [n for n in names if n not in df.columns],
        "as_of_unsafe": fr.as_of_unsafe(feats),
        "null_rate": {n: round(float(df[n].isna().mean()), 4) for n in present},
    }


def run_feature_engineering(config: Config | None = None) -> None:
    if config is None:
        config = get_config()

    data_dir = config.paths.data_dir
    completed_statuses = config.constants.completed_statuses
    n_previous = config.features.lookback_races

    laps = pd.read_csv(data_dir / "f1_laps_cleaned.csv")
    results = pd.read_csv(data_dir / "f1_results_cleaned.csv")

    laps = engineer_lap_features(laps, config)
    laps, results = engineer_result_features(laps, results, completed_statuses)

    # Bake the leakage-safe position-model features straight into the results CSV
    # so training just selects columns; prediction re-runs this with a cutoff.
    results = build_position_features(results, config)

    try:
        from pipeline.visualize import plot_feature_report
        plot_feature_report(laps, results, config.paths.plots_dir)
    except Exception as e:
        logger.warning("Visualization skipped: %s", e)

    laps.to_csv(data_dir / "f1_laps_features.csv", index=False)
    results.to_csv(data_dir / "f1_results_features.csv", index=False)

    manifest = feature_manifest(results, config)
    (data_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info(
        "Feature engineering complete: %d lap rows, %d result rows saved. "
        "(lookback=%d, shift=%d, %d position features, manifest v%s)",
        len(laps), len(results), n_previous, config.features.shift,
        len(manifest["features"]), manifest["registry_version"],
    )
