from __future__ import annotations

import logging
from typing import List

import pandas as pd

from pipeline.config_loader import Config, get_config

logger = logging.getLogger(__name__)


def load_raw_data(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = config.paths.data_dir
    laps = pd.read_csv(data_dir / "f1_laps_simple.csv")
    results = pd.read_csv(data_dir / "f1_results_simple.csv")

    qualifying_path = data_dir / "f1_qualifying_simple.csv"
    if qualifying_path.exists():
        qualifying = pd.read_csv(qualifying_path)
        logger.info(
            "Loaded %d laps, %d results, %d qualifying records.",
            len(laps), len(results), len(qualifying),
        )
    else:
        qualifying = pd.DataFrame()
        logger.warning("No qualifying data found — qualifying features will be skipped.")

    return laps, results, qualifying


def handle_missing_lap_times(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LapTime_seconds"] = df.groupby(["Driver", "Race"])["LapTime_seconds"].transform(
        lambda x: x.fillna(x.median())
    )
    global_median = df["LapTime_seconds"].median()
    df["LapTime_seconds"] = df["LapTime_seconds"].fillna(global_median)
    return df


def detect_and_remove_outliers(df: pd.DataFrame, col: str = "LapTime_seconds") -> pd.DataFrame:
    values = df[col].dropna()
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    before = len(df)
    df = df[(df[col].isna()) | ((df[col] >= lower) & (df[col] <= upper))].copy()
    logger.info(
        "Outlier removal on '%s': %d → %d rows (removed %d).",
        col, before, len(df), before - len(df),
    )
    return df


def clean_laps(df: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    default_tire = config.constants.default_tire if config else "MEDIUM"
    unknown_position = config.constants.unknown_position if config else 15

    df = handle_missing_lap_times(df)
    df["TireCompound"] = df.groupby("Race")["TireCompound"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else default_tire)
    )
    df["TireAge"] = df["TireAge"].fillna(0)
    df["Position"] = df.groupby(["Driver", "Race"])["Position"].ffill()
    df["Position"] = df["Position"].fillna(unknown_position)
    df = detect_and_remove_outliers(df, "LapTime_seconds")
    df["Driver"] = df["Driver"].astype("category")
    df["Team"] = df["Team"].astype("category")
    return df


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Status"] = df["Status"].astype("category")
    df["Driver"] = df["Driver"].astype("category")
    df["Team"] = df["Team"].astype("category")
    return df


def clean_qualifying(df: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    grid_size = config.constants.grid_size if config else 20

    df = df.copy()

    # Capture "reached Q3" from the RAW (pre-imputation) Q3 column — after the
    # median-fill below every driver has a Q3 time and the signal is lost.
    if "Q3" in df.columns:
        df["q3_reached"] = df["Q3"].notna().astype(int)

    for q_col in ["Q1", "Q2", "Q3"]:
        if q_col in df.columns:
            df[q_col] = df.groupby("Race")[q_col].transform(
                lambda x: x.fillna(x.median())
            )

    quali_times = df[["Q1", "Q2", "Q3"]].apply(pd.to_numeric, errors="coerce")
    df["BestQualifyingTime"] = quali_times.min(axis=1)

    for race in df["Race"].unique():
        mask = df["Race"] == race
        pole_time = df.loc[mask, "BestQualifyingTime"].min()
        df.loc[mask, "GapToPole"] = df.loc[mask, "BestQualifyingTime"] - pole_time

    df["QualifyingPerformance"] = (df["QualifyingPosition"] / grid_size) * 100
    df["Driver"] = df["Driver"].astype("category")
    df["Team"] = df["Team"].astype("category")
    return df


def merge_qualifying_into_results(
    results: pd.DataFrame, qualifying: pd.DataFrame
) -> pd.DataFrame:
    carry = ["Year", "Race", "Driver", "BestQualifyingTime", "GapToPole",
             "QualifyingPerformance"]
    if "q3_reached" in qualifying.columns:
        carry.append("q3_reached")
    quali_features = qualifying[carry]
    merged = results.merge(quali_features, on=["Year", "Race", "Driver"], how="left")
    logger.info("Qualifying features merged into results.")
    return merged


def run_cleaning(config: Config | None = None) -> None:
    if config is None:
        config = get_config()

    laps, results, qualifying = load_raw_data(config)

    laps_raw = laps.copy()

    laps = clean_laps(laps, config)
    results = clean_results(results)

    if not qualifying.empty:
        qualifying = clean_qualifying(qualifying, config)
        results = merge_qualifying_into_results(results, qualifying)
        qualifying.to_csv(config.paths.data_dir / "f1_qualifying_cleaned.csv", index=False)
        logger.info("Saved f1_qualifying_cleaned.csv")

    try:
        from pipeline.visualize import plot_cleaning_report
        plot_cleaning_report(
            laps_raw, laps, results,
            qualifying if not qualifying.empty else None,
            config.paths.plots_dir,
        )
    except Exception as e:
        logger.warning("Visualization skipped: %s", e)

    laps.to_csv(config.paths.data_dir / "f1_laps_cleaned.csv", index=False)
    results.to_csv(config.paths.data_dir / "f1_results_cleaned.csv", index=False)
    logger.info(
        "Cleaning complete: %d laps, %d results saved.",
        len(laps), len(results),
    )
