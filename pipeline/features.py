from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from pipeline.config_loader import Config, get_config

logger = logging.getLogger(__name__)

_TIRE_ENCODE = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4, "WET": 5}


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
        .apply(lambda x: x.isin(completed_statuses).mean() * 100, include_groups=False)
        .rename("team_reliability")
    )


def engineer_result_features(
    laps: pd.DataFrame,
    results: pd.DataFrame,
    completed_statuses: List[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    driver_win_rate = _compute_driver_win_rate(results)
    team_reliability = _compute_team_reliability(results, completed_statuses)

    laps = laps.merge(driver_win_rate, left_on="Driver", right_index=True, how="left")
    results = results.merge(driver_win_rate, left_on="Driver", right_index=True, how="left")
    laps["driver_win_rate"] = laps["driver_win_rate"].fillna(0)
    results["driver_win_rate"] = results["driver_win_rate"].fillna(0)

    laps = laps.merge(team_reliability, left_on="Team", right_index=True, how="left")
    results = results.merge(team_reliability, left_on="Team", right_index=True, how="left")

    results["PositionChange"] = results["GridPosition"] - results["Position"]
    results["race_winner"] = (results["Position"] == 1).astype(int)
    results["podium_finish"] = (results["Position"] <= 3).astype(int)
    results["points_finish"] = (results["Position"] <= 10).astype(int)

    return laps, results


def create_historical_features(
    df: pd.DataFrame,
    n_previous: int = 6,
    completed_statuses: List[str] | None = None,
) -> pd.DataFrame:
    """Compute per-driver rolling statistics over the previous n_previous races."""
    if completed_statuses is None:
        completed_statuses = ["Finished"]

    frames = []
    for driver in df["Driver"].unique():
        d = df[df["Driver"] == driver].copy().reset_index(drop=True)

        d["avg_position_last"] = d["Position"].rolling(n_previous, min_periods=1).mean()
        d["best_position_last"] = d["Position"].rolling(n_previous, min_periods=1).min()
        d["avg_grid_last"] = d["GridPosition"].rolling(n_previous, min_periods=1).mean()

        d["is_dnf"] = (~d["Status"].isin(completed_statuses)).astype(int)
        d["dnf_last"] = d["is_dnf"].rolling(n_previous, min_periods=1).sum()
        d["reliability_rate"] = 1 - (d["dnf_last"] / n_previous)

        d["positions_gained"] = d["GridPosition"] - d["Position"]
        d["avg_positions_gained"] = d["positions_gained"].rolling(n_previous, min_periods=1).mean()

        d["podiums_last"] = (d["Position"] <= 3).astype(int).rolling(n_previous, min_periods=1).sum()
        d["wins_last"] = (d["Position"] == 1).astype(int).rolling(n_previous, min_periods=1).sum()
        d["points_last"] = d["Points"].rolling(n_previous, min_periods=1).sum()

        if "BestQualifyingTime" in d.columns:
            d["avg_quali_time"] = d["BestQualifyingTime"].rolling(n_previous, min_periods=1).mean()
            d["avg_gap_to_pole"] = d["GapToPole"].rolling(n_previous, min_periods=1).mean()

        recent_avg = d["Position"].rolling(3, min_periods=1).mean()
        if n_previous > 3:
            older_avg = d["Position"].shift(3).rolling(n_previous - 3, min_periods=1).mean()
            d["form_trend"] = older_avg - recent_avg
        else:
            d["form_trend"] = 0.0

        frames.append(d)

    return pd.concat(frames, ignore_index=True)


def run_feature_engineering(config: Config | None = None) -> None:
    if config is None:
        config = get_config()

    data_dir = config.paths.data_dir
    completed_statuses = config.constants.completed_statuses
    n_previous = config.pipeline.lookback_races

    laps = pd.read_csv(data_dir / "f1_laps_cleaned.csv")
    results = pd.read_csv(data_dir / "f1_results_cleaned.csv")

    laps = engineer_lap_features(laps, config)
    laps, results = engineer_result_features(laps, results, completed_statuses)

    try:
        from pipeline.visualize import plot_feature_report
        plot_feature_report(laps, results, config.paths.plots_dir)
    except Exception as e:
        logger.warning("Visualization skipped: %s", e)

    laps.to_csv(data_dir / "f1_laps_features.csv", index=False)
    results.to_csv(data_dir / "f1_results_features.csv", index=False)

    logger.info(
        "Feature engineering complete: %d lap rows, %d result rows saved. "
        "(lookback=%d, completed_statuses=%s)",
        len(laps), len(results), n_previous, completed_statuses,
    )
