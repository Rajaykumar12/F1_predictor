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


def compute_race_seq(df: pd.DataFrame) -> pd.Series:
    """Dense rank of ``(Year, Race)`` in chronological order — monotonic across
    seasons, so 2024's round 13 sorts before 2025's round 1, unlike the
    season-local ``Race`` column which resets every year (B2, multi-season
    data). Falls back to ``Race`` alone when ``Year`` is absent (legacy
    single-season callers / ad-hoc frames)."""
    if "Year" not in df.columns:
        return df["Race"]
    pairs = sorted(set(zip(df["Year"], df["Race"])))
    mapping = {p: i + 1 for i, p in enumerate(pairs)}
    keys = list(zip(df["Year"], df["Race"]))
    return pd.Series(keys, index=df.index).map(mapping)


def _resolve_as_of_seq(df: pd.DataFrame, as_of_round: int, season: int | None = None) -> int | None:
    """Translate a current-season ``as_of_round`` cutoff into the ``race_seq``
    threshold it corresponds to, so callers keep passing a familiar
    "round N of this season" number while every builder underneath orders by
    the season-spanning ``race_seq``. Defaults ``season`` to the latest season
    present in ``df`` (the single-season behaviour predict/evaluate/backtest
    already rely on)."""
    if season is None:
        if "Year" not in df.columns or df.empty:
            return None
        season = int(df["Year"].max())
    prior_this_season = df[(df["Year"] == season) & (df["Race"] <= as_of_round)] \
        if "Year" in df.columns else df[df["Race"] <= as_of_round]
    if not prior_this_season.empty:
        return int(prior_this_season["race_seq"].max())
    earlier_seasons = df[df["Year"] < season] if "Year" in df.columns else pd.DataFrame()
    if not earlier_seasons.empty:
        return int(earlier_seasons["race_seq"].max())
    return None


def _ensure_race_seq(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``race_seq`` if a caller invoked a builder directly on a frame that
    never went through :func:`create_historical_features` (unit tests, ad-hoc
    use). Idempotent — a no-op once the column exists."""
    if "race_seq" not in df.columns:
        df = df.copy()
        df["race_seq"] = compute_race_seq(df)
    return df


def regulation_era(year: int, reset_seasons: list[int]) -> int:
    """The regulation era a season belongs to — the most recent reset year
    ``<= year``. With ``reset_seasons=[2022, 2026]``: 2022..2025 all map to
    era 2022 (one continuous technical package); 2026 maps to era 2026 (a
    fresh reset, distinct even from the immediately preceding season).

    A year before the first listed reset has no defined era (returns that
    year itself, so it never spuriously matches any real era) — this project
    has no data before 2022 so it's a defensive fallback, not a real case.
    """
    era = reset_seasons[0]
    for r in reset_seasons:
        if r <= year:
            era = r
        else:
            break
    return era


def _ensure_era(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Add an ``_era`` column (regulation era per row, see :func:`regulation_era`)
    if missing. Idempotent. Falls back to a single constant era when ``Year``
    is absent (legacy single-season frames), matching how ``_ensure_race_seq``
    degrades for the same case."""
    if "_era" not in df.columns:
        df = df.copy()
        resets = config.pipeline.regulation_reset_seasons
        if "Year" in df.columns:
            df["_era"] = df["Year"].map(lambda y: regulation_era(int(y), resets))
        else:
            df["_era"] = 0
    return df


def _status_col(df: pd.DataFrame) -> pd.Series:
    """The classified-status column to compare against ``completed_statuses``.

    Prefers ``status_canon`` (written by :func:`pipeline.clean.normalize_status`
    on every real pipeline frame — folds legacy Ergast strings and current
    fastf1 strings onto one vocabulary). Falls back to the raw ``Status`` column
    for ad-hoc frames (unit tests) that never went through ``clean.py``.
    """
    return df["status_canon"] if "status_canon" in df.columns else df["Status"]


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
        _status_col(results).groupby(results["Team"])
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
    order: str = "race_seq",
    extra_group: str | None = None,
) -> pd.Series:
    """Per ``group`` (sorted by ``order``), apply ``.shift(shift)`` **before** a
    ``.rolling(window, min_periods=1).<how>()``. With ``shift >= 1`` the current
    row is never inside its own window, so the value is strictly historical. The
    first ``shift`` rows of each group are NaN by construction.

    ``extra_group`` (e.g. ``"_era"``, see :func:`_ensure_era`) partitions the
    rolling window further so it never crosses a regulation-era boundary —
    without it, a driver/team's "recent form" at the start of a new
    regulation package would silently include races run under a since-
    superseded technical formula (the ``team_form_avg_finish_s5`` bug this
    was added to fix: Red Bull's rolling form at 2026 Round 1 was drawing
    entirely on their 2022 form).
    """
    keys = group if extra_group is None else [group, extra_group]
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, g in df.groupby(keys, sort=False):
        g = g.sort_values(order)
        rolled = getattr(g[value_col].shift(shift).rolling(window, min_periods=1), how)()
        out.loc[g.index] = rolled.to_numpy()
    return out


def _expanding_pre_race_rate(
    df: pd.DataFrame, group: str, event: pd.Series, within_season: bool = False
) -> pd.Series:
    """Mean of ``event`` (0/1) over ``group``'s rows in races **strictly before**
    the current race, ordered by ``race_seq`` (B2 — monotonic across seasons).
    Race-aware: a team's two same-race rows never see each other. First race
    for a group -> NaN. Returns a 0-1 rate.

    ``within_season=True`` (B3) resets the cumulative count at every season
    boundary — for "to-date" rates like reliability, where a new season starts
    from zero, not carrying a prior season's tally forward. Rolling *form*
    (a fixed-window rolling mean, built with :func:`_shifted_roll` instead)
    stays season-spanning on purpose — recent form is recent form.
    """
    group_key = (
        list(zip(df[group], df["Year"])) if within_season and "Year" in df.columns
        else df[group].to_numpy()
    )
    tmp = pd.DataFrame(
        {"_g": group_key, "_r": df["race_seq"].to_numpy(),
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
    as_of_seq: int | None = None,
    shift: int = 1,
    season: int | None = None,
) -> pd.DataFrame:
    """Per-driver rolling form, **shift-before-roll**.

    For every driver (sorted by ``race_seq`` — the dense rank of ``(Year,
    Race)``, monotonic across seasons; see :func:`compute_race_seq`, B2) a
    ``.shift(shift)`` is applied before each ``.rolling(n_previous,
    min_periods=1)`` window, so with the default ``shift=1`` the race being
    predicted is never part of its own history. ``shift=0`` reproduces the old
    leaky windows and is used only by the hypothesis-testing "raw" regression
    guard.

    ``as_of_round`` truncates every driver's history to races at or before
    round ``as_of_round`` **of the current/latest season** (an honest "form
    going into the next round" snapshot) — resolved to the equivalent
    ``race_seq`` cutoff via :func:`_resolve_as_of_seq` so it still means the
    same thing once prior seasons are in the frame. Pass ``as_of_seq``
    directly to cut off by ``race_seq`` instead (e.g. a rolling backtest
    walking round-by-round across seasons).

    The legacy rollup column names (``avg_position_last``, ``dnf_last``,
    ``podiums_last`` …) are still produced for display / backward compatibility
    (``predict.py`` recent-form panel, ``evaluate.py`` row filter). The position
    **model** no longer selects them — it reads
    :func:`pipeline.feature_registry.enabled_features`.
    """
    if completed_statuses is None:
        completed_statuses = ["Finished"]

    df = _ensure_race_seq(df)

    cutoff_seq = as_of_seq
    if cutoff_seq is None and as_of_round is not None:
        cutoff_seq = _resolve_as_of_seq(df, as_of_round, season)
        if cutoff_seq is None:
            return df.iloc[0:0]

    frames = []
    for driver in df["Driver"].unique():
        d = df[df["Driver"] == driver].copy()
        d = d.sort_values("race_seq").reset_index(drop=True)

        if cutoff_seq is not None:
            d = d[d["race_seq"] <= cutoff_seq].reset_index(drop=True)
            if d.empty:
                continue

        def roll(series: pd.Series, how: str = "mean", w: int = n_previous):
            return getattr(series.shift(shift).rolling(w, min_periods=1), how)()

        d["is_dnf"] = (~_status_col(d).isin(completed_statuses)).astype(int)
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

    if not frames:
        return df.iloc[0:0]
    return pd.concat(frames, ignore_index=True)


# =========================================================================== #
# Registry-driven family builders — each is pure: (df, config) -> df + new cols
# =========================================================================== #
def build_quali_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Qualifying-derived signals. All known once qualifying is complete, before
    the race — every column is as-of-safe."""
    df = _ensure_race_seq(df)

    if "BestQualifyingTime" in df.columns:
        pole = df.groupby("race_seq")["BestQualifyingTime"].transform("min")
        gap = df["GapToPole"] if "GapToPole" in df.columns else (df["BestQualifyingTime"] - pole)
        with np.errstate(divide="ignore", invalid="ignore"):
            df["quali_gap_to_pole_pct"] = np.where(pole > 0, gap / pole * 100.0, np.nan)

        # teammate delta: driver best-quali minus the mean of same-race, same-team
        # team-mates (2-car teams => just the one team-mate).
        team_mean = df.groupby(["race_seq", "Team"])["BestQualifyingTime"].transform("mean")
        team_cnt = df.groupby(["race_seq", "Team"])["BestQualifyingTime"].transform("count")
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

    # C1: GridPosition is already post-penalty (58/286 rows differ from
    # QualifyingPosition in the current data, range -4..+12) — grid_penalty
    # captures recovery drives / grid drops the model was previously blind to.
    if "QualifyingPosition" in df.columns and "GridPosition" in df.columns:
        df["grid_penalty"] = df["GridPosition"] - df["QualifyingPosition"]
    else:
        df["grid_penalty"] = 0.0

    return df


def build_form_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Per-driver rolling finishing form (shift-before-roll).

    Every window here is additionally partitioned by regulation era (see
    :func:`_ensure_era`) — a driver's "recent form" must not reach back
    across a technical-regulation reset into a car/formula that no longer
    exists, the same reasoning as ``team_form_avg_finish_s5`` in
    :func:`build_team_features`.
    """
    df = _ensure_race_seq(df)
    df = _ensure_era(df, config)
    s = config.features.shift
    completed = config.constants.completed_statuses

    df["form_avg_finish_s5"] = _shifted_roll(df, "Position", 5, s, "mean", extra_group="_era")

    if "is_dnf" not in df.columns:
        df["is_dnf"] = (~_status_col(df).isin(completed)).astype(int)
    df["form_dnf_rate_s8"] = _shifted_roll(df, "is_dnf", 8, s, "mean", extra_group="_era")

    if "form_trend" not in df.columns:
        # create_historical_features normally supplies this; recompute if a
        # caller invoked the builder directly.
        n_prev = config.features.lookback_races
        parts = []
        for _, g in df.groupby(["Driver", "_era"], sort=False):
            g = g.sort_values("race_seq")
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
    the driver's grid-to-finish delta averaged over *prior* races only.
    Era-partitioned for the same reason as :func:`build_form_features`."""
    df = _ensure_race_seq(df)
    df = _ensure_era(df, config)
    s = config.features.shift
    if "positions_gained" not in df.columns:
        df["positions_gained"] = df["GridPosition"] - df["Position"]
    df["hist_positions_gained_s5"] = _shifted_roll(
        df, "positions_gained", 5, s, "mean", extra_group="_era"
    )
    df["hist_grid_finish_consistency_s5"] = _shifted_roll(
        df, "positions_gained", 5, s, "std", extra_group="_era"
    )
    return df


def build_team_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Team-level (both cars) strength going into the round."""
    df = _ensure_race_seq(df)
    df = _ensure_era(df, config)

    # mean finishing Position of the team over the previous up-to-5 races,
    # ordered by race_seq (B2, prevents mixing via a repeated round number)
    # AND never crossing a regulation-era boundary (grouping by "_era" too) —
    # a team's competitiveness under a since-superseded technical formula is
    # not "recent form" for the current one. Concretely: without the era
    # split, Red Bull's dominant 2022 rolling form (era 2022) was still
    # feeding "team_form_avg_finish_s5" for 2026 Round 1 (era 2026), the
    # single feature the model relies on most.
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, g in df.groupby(["Team", "_era"], sort=False):
        race_mean = g.groupby("race_seq")["Position"].mean().sort_index()
        rolled = race_mean.shift(1).rolling(5, min_periods=1).mean()
        out.loc[g.index] = g["race_seq"].map(rolled).to_numpy()
    df["team_form_avg_finish_s5"] = out

    # this round's team qualifying pace, ranked across teams (1 = fastest)
    if "GapToPole" in df.columns:
        team_pace = df.groupby(["race_seq", "Team"])["GapToPole"].transform("mean")
        df["_team_pace"] = team_pace
        df["team_quali_pace_rank"] = (
            df.groupby("race_seq")["_team_pace"].rank(method="dense", ascending=True)
        )
        df = df.drop(columns="_team_pace")
    else:
        df["team_quali_pace_rank"] = np.nan
    return df


def build_reliability_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Season-to-date reliability, expanding over races strictly before this
    one, reset at every season boundary (B3) — a driver's reliability record
    doesn't carry a prior season's tally into a new one."""
    df = _ensure_race_seq(df)
    completed = config.constants.completed_statuses
    if "is_dnf" not in df.columns:
        df["is_dnf"] = (~_status_col(df).isin(completed)).astype(int)
    finished = _status_col(df).isin(completed).astype(float)

    df["driver_dnf_rate_todate"] = _expanding_pre_race_rate(
        df, "Driver", df["is_dnf"].astype(float), within_season=True
    )
    df["team_reliability_todate"] = _expanding_pre_race_rate(
        df, "Team", finished, within_season=True
    )
    return df


def load_circuits(config: Config) -> pd.DataFrame:
    """Static circuit reference (``data/circuits.csv``), keyed by
    ``circuit_key`` — a circuit's physical properties (invariant across
    years). ``round`` is kept as a documentation/legacy column for
    single-season (no ``Year``) callers only; multi-season lookups must go
    through :func:`load_circuit_calendar`, never straight off ``round``,
    since round numbers are reassigned to different circuits year to year
    (round 13 was the Hungarian GP in 2022, the Italian GP in 2026)."""
    path = config.paths.data_dir / "circuits.csv"
    if not path.exists():
        logger.warning("data/circuits.csv missing — circuit features will be NaN/median-filled.")
        return pd.DataFrame(columns=["round", "circuit_key", "location",
                                     "circuit_overtaking_index",
                                     "circuit_is_street", "circuit_sc_probability"])
    return pd.read_csv(path)


_LOCATION_ALIASES = {
    # fastf1's schedule Location (city/area) -> this project's circuit_key
    # slug, for the cases where they legitimately differ (a city hosting a
    # circuit not named after the city itself, or a label that changed
    # across seasons for the same physical track). Only add an alias when
    # it's genuinely the same track — an unmatched location correctly
    # becomes a new, unknown circuit_key (median-filled) rather than being
    # force-matched to something it isn't.
    "miami_gardens": "miami",
    "monte_carlo": "monaco",
    "sao_paulo": "interlagos",
    "yas_island": "yas_marina",
    "spa_francorchamps": "spa",
}


def _slugify(text) -> str:
    """ASCII-lowercase, underscore-separated slug — 'Montréal' -> 'montreal',
    'São Paulo' -> 'sao_paulo', 'Spa-Francorchamps' -> 'spa_francorchamps' —
    so accented/hyphenated fastf1 location names compare equal to this
    project's plain-ASCII, underscore-separated circuit_key slugs."""
    import re
    import unicodedata
    ascii_text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return "_".join(re.split(r"[\s\-]+", ascii_text.strip().lower())).strip("_")


def _normalize_location(loc) -> str:
    slug = _slugify(loc)
    return _LOCATION_ALIASES.get(slug, slug)


def build_circuit_calendar(config: Config, years: list[int]) -> pd.DataFrame:
    """(Year, Round) -> circuit_key for each requested year, built from the
    fastf1 event schedule and matched against ``data/circuits.csv``'s
    ``circuit_key`` column (already a plain-ASCII city/venue slug — e.g.
    "monza", "budapest" — which is what fastf1's ``Location`` field
    normalizes to, NOT this file's ``location`` column, which holds the
    venue's proper name like "Hungaroring"). A location absent from
    ``circuits.csv`` gets a fresh, unmatched circuit_key (``unknown_<slug>``)
    rather than silently reusing another circuit's stats — every downstream
    consumer already median-fills an unrecognized circuit_key the same way
    it does a missing round today."""
    import fastf1

    circuits = load_circuits(config)
    loc_to_key = {}
    if not circuits.empty and "circuit_key" in circuits.columns:
        loc_to_key = dict(zip(circuits["circuit_key"], circuits["circuit_key"]))

    rows = []
    for year in years:
        try:
            sched = fastf1.get_event_schedule(int(year), include_testing=False)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not load %s event schedule for circuit calendar: %s", year, e)
            continue
        for _, r in sched.iterrows():
            norm = _normalize_location(r["Location"])
            circuit_key = loc_to_key.get(norm, f"unknown_{norm.replace(' ', '_')}")
            rows.append({"Year": int(year), "Round": int(r["RoundNumber"]), "circuit_key": circuit_key})
    return pd.DataFrame(rows, columns=["Year", "Round", "circuit_key"])


def load_circuit_calendar(config: Config, df: pd.DataFrame) -> pd.DataFrame:
    """Cached wrapper around :func:`build_circuit_calendar` — persists to
    ``data/circuit_calendar.csv`` so a normal feature build doesn't hit the
    fastf1 schedule API every run, and extends the cache automatically when
    a year not yet covered shows up in ``df``."""
    path = config.paths.data_dir / "circuit_calendar.csv"
    years_needed = set(int(y) for y in df["Year"].dropna().unique()) if "Year" in df.columns else set()
    cached = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["Year", "Round", "circuit_key"])
    missing_years = sorted(years_needed - set(cached["Year"].unique()) if not cached.empty else years_needed)
    if not missing_years:
        return cached
    fresh = build_circuit_calendar(config, missing_years)
    combined = pd.concat([cached, fresh], ignore_index=True).drop_duplicates(["Year", "Round"])
    try:
        combined.to_csv(path, index=False)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not cache circuit calendar to %s: %s", path, e)
    return combined


def build_circuit_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Attach curated circuit metadata by real circuit identity (via
    :func:`load_circuit_calendar`), not round number — a round missing a
    calendar match, or a circuit_key missing from ``circuits.csv``, gets the
    column median and a logged warning, same convention as before."""
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

    if "Year" in df.columns:
        calendar = load_circuit_calendar(config, df)
        keyed = df.merge(
            calendar, left_on=["Year", "Race"], right_on=["Year", "Round"], how="left"
        ).drop(columns=["Round"])
        merged = keyed.merge(circuits[["circuit_key"] + cols], on="circuit_key", how="left")
        missing = sorted(set(
            zip(df.loc[merged["circuit_key"].isna(), "Year"], df.loc[merged["circuit_key"].isna(), "Race"])
        ))
        if missing:
            logger.warning("No circuit calendar/circuits.csv match for (Year, Round) %s — median-filling.", missing)
        merged = merged.drop(columns=["circuit_key"])
    else:
        # Legacy single-season fallback (no Year column, e.g. ad-hoc frames) —
        # round-number join is the best available without a season to key on.
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
    fully causal (uses only completed earlier races). Ordered by ``race_seq``
    (B2) and reset every season (B3) — championship points don't carry over
    into a new year."""
    df = _ensure_race_seq(df)
    season_key = df["Year"] if "Year" in df.columns else 0
    d = df.assign(_season_key=season_key).sort_values(["Driver", "race_seq"])
    cum_before = d.groupby(["Driver", "_season_key"])["Points"].cumsum() - d["Points"]
    cum_before = cum_before.reindex(df.index)
    leader_before = cum_before.groupby(df["race_seq"]).transform("max")
    df["driver_points_gap_to_leader_before"] = leader_before - cum_before
    return df


def build_history_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Frozen prior-season priors (B4) — known before round 1 of a season, so
    unlike every other family these are causal even at the very first race.
    Falls back to all-NaN (median-filled downstream, like ``circuit``) for
    single-season data with no prior year to draw on.

    ``driver_circuit_avg_finish_prior`` keys "this circuit" by real circuit
    identity (:func:`load_circuit_calendar`, matched via location — round
    number is NOT a stable circuit identity across seasons: round 13 was the
    Hungarian GP in 2022, the Italian GP in 2026) and expands over every
    STRICTLY earlier instance of that circuit *within the same regulation
    era* — a driver's finishing position is heavily confounded by
    car/team competitiveness, so a same-circuit result from a superseded
    technical formula is not a meaningful "prior" for the current one, same
    reasoning as ``team_form_avg_finish_s5``. In the current 2022+2026
    dataset (two different eras, no same-era multi-year overlap yet) this
    means the feature is neutral-filled for every row today, exactly like
    the three prior-season features above — it will start producing real
    values once genuine same-era multi-year data exists.

    ``season_progress_weight`` (``min(1, Race/8)``) is exposed as a plain
    feature rather than used to blend two code paths — the model is left to
    learn how much to lean on priors vs. in-season form itself.
    """
    df = _ensure_race_seq(df)
    cols = [
        "driver_prior_season_avg_finish", "driver_prior_season_dnf_rate",
        "team_prior_season_points_rank", "driver_circuit_avg_finish_prior",
    ]
    # Idempotent: build_position_features may run on a frame that already
    # carries these columns (baked into f1_results_features.csv). Drop before
    # the merges below so we don't get _x / _y suffix collisions.
    df = df.drop(columns=[c for c in cols + ["season_progress_weight"] if c in df.columns])
    orig_index = df.index
    if "Year" not in df.columns:
        for c in cols:
            df[c] = np.nan
        df["season_progress_weight"] = (df["Race"] / 8).clip(upper=1.0)
        return df

    completed = config.constants.completed_statuses
    is_dnf = (~_status_col(df).isin(completed)).astype(int)
    work = df.assign(_is_dnf=is_dnf)

    # A "prior season" is only a valid prior when it's in the SAME regulation
    # era as the season it's predicting — otherwise it's not a prior, it's a
    # different formula's result (e.g. 2026 has no valid prior under this
    # rule, since 2026 is itself an era-start year; this makes that
    # currently-accidental all-neutral behaviour intentional and documented,
    # and lets it correctly activate once real same-era multi-year data
    # exists, e.g. a future 2024 row validly drawing on 2023).
    resets = config.pipeline.regulation_reset_seasons
    def _valid_prior_year(year: int) -> bool:
        return regulation_era(int(year), resets) == regulation_era(int(year) - 1, resets)

    driver_season = (
        work.groupby(["Driver", "Year"])
        .agg(driver_prior_season_avg_finish=("Position", "mean"),
             driver_prior_season_dnf_rate=("_is_dnf", "mean"))
        .reset_index()
    )
    driver_season["Year"] += 1  # this season's stats become NEXT season's "prior"
    driver_season = driver_season[driver_season["Year"].map(_valid_prior_year)]
    df = df.merge(driver_season, on=["Driver", "Year"], how="left")

    team_season = work.groupby(["Team", "Year"])["Points"].sum().reset_index()
    team_season["team_prior_season_points_rank"] = (
        team_season.groupby("Year")["Points"].rank(ascending=False, method="min")
    )
    team_season["Year"] += 1
    team_season = team_season[team_season["Year"].map(_valid_prior_year)]
    df = df.merge(
        team_season[["Team", "Year", "team_prior_season_points_rank"]],
        on=["Team", "Year"], how="left",
    )
    df.index = orig_index
    df = _ensure_era(df, config)

    calendar = load_circuit_calendar(config, df)
    circuit_base = df[["Driver", "Year", "Race", "race_seq", "_era", "Position"]]
    circuit_hist = circuit_base.merge(
        calendar, left_on=["Year", "Race"], right_on=["Year", "Round"], how="left"
    ).drop(columns=["Round"])
    circuit_hist.index = circuit_base.index  # merge resets to RangeIndex; how="left" preserves row order
    circuit_hist = circuit_hist.sort_values("race_seq")
    circuit_hist["driver_circuit_avg_finish_prior"] = (
        circuit_hist.groupby(["Driver", "circuit_key", "_era"])["Position"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    df["driver_circuit_avg_finish_prior"] = circuit_hist.sort_index()["driver_circuit_avg_finish_prior"]

    # Median-fill the "no prior season yet" rows (single-season data, or a
    # driver's rookie season) — same convention as build_circuit_features,
    # so `history` can be enabled by default without starving prepare_position
    # _data's dropna of rows before B1's multi-season fetch has run. A truly
    # all-NaN column (only one season anywhere in the frame) falls back to a
    # neutral mid-grid estimate rather than leaving NaN.
    neutral = {
        "driver_prior_season_avg_finish": config.constants.grid_size / 2,
        "driver_prior_season_dnf_rate": 0.2,
        "team_prior_season_points_rank": 5.0,
        "driver_circuit_avg_finish_prior": config.constants.grid_size / 2,
    }
    for c in cols:
        med = df[c].median()
        df[c] = df[c].fillna(med if pd.notna(med) else neutral[c])

    df["season_progress_weight"] = (df["Race"] / 8).clip(upper=1.0)
    return df


def load_context_csv(config: Config) -> pd.DataFrame:
    """Curated pre-race context (``data/context.csv``, B6), keyed by
    ``(Year, Race)``: dominant tyre allocation, wet-race flag (from the
    pre-race forecast), a team-upgrade flag, and a rookie flag per driver."""
    path = config.paths.data_dir / "context.csv"
    if not path.exists():
        logger.warning("data/context.csv missing — raceday features will be neutral-filled.")
        return pd.DataFrame(columns=[
            "Year", "Race", "Driver", "context_wet_race_forecast",
            "context_team_upgrade", "context_rookie",
        ])
    return pd.read_csv(path)


def build_raceday_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Attach curated per-(Year, Race[, Driver]) context (B6). Missing rows get
    neutral defaults (0 — "no wet forecast / no upgrade / not a rookie") and a
    logged warning rather than NaN, since these are binary situational flags,
    not a level that's honestly unknown."""
    df = df.copy()
    context = load_context_csv(config)
    flag_cols = ["context_wet_race_forecast", "context_team_upgrade", "context_rookie"]
    df = df.drop(columns=[c for c in flag_cols if c in df.columns])

    if context.empty or "Year" not in df.columns:
        for c in flag_cols:
            df[c] = 0.0
        return df

    keys = ["Year", "Race"] + (["Driver"] if "Driver" in context.columns else [])
    orig_index = df.index
    merged = df.merge(context[keys + flag_cols], on=keys, how="left")
    merged.index = orig_index
    for c in flag_cols:
        merged[c] = merged[c].fillna(0.0)
    return merged


def build_racewin_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """C2: blend the separately-trained racewin classifier's ``predict_proba``
    in as a feature.

    CAUTION (why this family is opt-in, not in the default
    ``families_enabled``): the racewin model's OTHER inputs — legacy
    ``driver_win_rate`` / ``team_reliability`` from
    :func:`engineer_result_features` — are whole-history rates computed over
    ALL of a driver/team's rows at once, not shift-before-roll / as-of-safe
    like every position-model feature. Reworking the racewin model itself is
    out of scope here (see this module's docstring); blending its output
    trades that known leakage risk for whatever signal ``predict_proba``
    adds, so it's available to explore but excluded from the honest
    backtest by default. (The far more serious leak — the racewin model's
    OWN training previously included the actual finishing ``Position`` as an
    input, target-identical to what it predicts — was removed from
    :func:`pipeline.train.train_racewin_model` as a hard prerequisite for
    this feature existing at all.)
    """
    df = df.copy()
    neutral = 1.0 / config.constants.grid_size
    try:
        from pipeline.model_registry import load_bundle, model_feature_columns

        bundle = load_bundle(config)
        model = bundle.win_model
        if model is None:
            raise RuntimeError("racewin model not trained")
        cols = model_feature_columns(model)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"missing racewin input column(s): {missing}")
        df["racewin_probability"] = model.predict_proba(df[cols])[:, 1]
    except Exception as e:  # noqa: BLE001 — any failure means "neutral-fill"
        logger.warning("racewin blend unavailable (%s) — neutral-filling racewin_probability.", e)
        df["racewin_probability"] = neutral
    return df


def build_circuit_similarity_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """C3 — rolling form over *similar* circuits (street vs non-street grouping).

    ``form_avg_finish_similar_circuit_s5`` = shift(1) then rolling(5) mean of
    ``Position`` over prior races whose circuit shares the same
    ``circuit_is_street`` flag as the target race.

    Street status is looked up via the real circuit identity
    (:func:`load_circuit_calendar`), not round number — round 13 is a
    different circuit in different seasons. Falls back to all-NaN
    (median-filled downstream) when ``circuits.csv`` is absent or the
    calendar has no match, exactly like ``build_circuit_features``.

    The season-spanning ordering from :func:`compute_race_seq` is preserved:
    2024-round-5 is treated as older than 2025-round-1 (B2) — but never
    across a regulation-era boundary (see :func:`_ensure_era`): finishing
    Position is heavily confounded by car/team competitiveness, so a
    driver's similar-circuit results from a superseded technical formula are
    not "recent form" for the current one, same reasoning as
    ``team_form_avg_finish_s5``.
    """
    df = _ensure_race_seq(df)
    df = _ensure_era(df, config)
    s = config.features.shift

    circuits = load_circuits(config)
    if circuits.empty or "circuit_is_street" not in circuits.columns:
        df["form_avg_finish_similar_circuit_s5"] = np.nan
        return df

    df = df.copy()
    if "Year" in df.columns:
        calendar = load_circuit_calendar(config, df)
        street_by_key = dict(zip(circuits["circuit_key"], circuits["circuit_is_street"].astype(float)))
        keyed = df.merge(
            calendar, left_on=["Year", "Race"], right_on=["Year", "Round"], how="left"
        )
        df["_circuit_type"] = keyed["circuit_key"].map(street_by_key).to_numpy()
    else:
        # Legacy single-season fallback — same as build_circuit_features.
        street_map: dict[int, float] = dict(zip(
            circuits["round"].astype(int), circuits["circuit_is_street"].astype(float),
        ))
        df["_circuit_type"] = df["Race"].map(street_map)  # NaN for unknown rounds

    out = pd.Series(np.nan, index=df.index, dtype=float)
    for (driver, _era), g in df.groupby(["Driver", "_era"], sort=False):
        g = g.sort_values("race_seq")
        result_arr = np.full(len(g), np.nan)
        for i, (_, row) in enumerate(g.iterrows()):
            ct = row["_circuit_type"]
            if np.isnan(ct):
                continue  # unknown circuit type — leave NaN for this row
            # Restrict to prior rows (shift=s means exclude last s rows of self)
            # with the same circuit type; treat shift as # rows to exclude from tail.
            prior = g.iloc[: max(0, i - s + 1)]
            same_type = prior[prior["_circuit_type"] == ct]["Position"]
            if len(same_type) == 0:
                continue
            window = same_type.iloc[-5:]  # last up to 5 same-type finishes
            result_arr[i] = float(window.mean())
        out.loc[g.index] = result_arr

    df["form_avg_finish_similar_circuit_s5"] = out
    df = df.drop(columns=["_circuit_type"])
    return df


_BUILDERS = {
    "quali": build_quali_features,
    "form": build_form_features,
    "racecraft": build_racecraft_features,
    "team": build_team_features,
    "reliability": build_reliability_features,
    "circuit": build_circuit_features,
    "championship": build_championship_features,
    "history": build_history_features,
    "raceday": build_raceday_features,
    "racewin": build_racewin_features,
    "circuit_sim": build_circuit_similarity_features,
}


# =========================================================================== #
# The single position-model feature path
# =========================================================================== #
def build_position_features(
    results_df: pd.DataFrame,
    config: Config | None = None,
    as_of_round: int | None = None,
    as_of_seq: int | None = None,
    season: int | None = None,
    lookback: int | None = None,
    shift: int | None = None,
) -> pd.DataFrame:
    """Assemble every enabled registry feature on ``results_df``.

    This is the ONE code path shared by training (:mod:`pipeline.train`),
    prediction (:mod:`pipeline.predict`), evaluation (:mod:`pipeline.evaluate`)
    and backtest (``scripts/backtest.py``, ``scripts/rolling_backtest.py``), so
    train-time and predict-time features can never diverge. ``as_of_round``
    (a round number of ``season``, default the latest season in the frame) or
    ``as_of_seq`` (a direct ``race_seq`` cutoff — B2) forward to
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
        as_of_seq=as_of_seq,
        season=season,
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
