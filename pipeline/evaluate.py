from __future__ import annotations

import logging
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from pipeline.config_loader import Config, get_config

logger = logging.getLogger(__name__)


def evaluate_laptime(config: Config, race_round: int | None = None) -> pd.DataFrame:
    """
    Compare predicted vs actual lap times for a completed race.

    Parameters
    ----------
    race_round : int | None
        Race round number to evaluate. Defaults to the most recent race in the data.

    Returns the comparison DataFrame.
    """
    data_dir = config.paths.data_dir
    models_dir = config.paths.models_dir
    plots_dir = config.paths.plots_dir / "evaluation"

    # Load feature data
    laps = pd.read_csv(data_dir / "f1_laps_features.csv")

    available_races = sorted(laps["Race"].unique())
    if not available_races:
        raise RuntimeError("No lap data found. Run: python main.py features")

    if race_round is None:
        race_round = int(available_races[-1])
        logger.info("No race specified — defaulting to most recent: Round %d", race_round)

    if race_round not in available_races:
        raise ValueError(
            f"Round {race_round} not in data. Available rounds: {available_races}"
        )

    race_laps = laps[laps["Race"] == race_round].copy()
    logger.info("Evaluating Round %d — %d laps", race_round, len(race_laps))

    # Load model and predict
    with open(models_dir / "xgb_laptime_pipeline.pkl", "rb") as f:
        model = pickle.load(f)

    # Use only columns the model was trained on
    pre = model.named_steps["preprocessor"]
    numerical_cols = pre.transformers_[0][2]
    categorical_cols = pre.transformers_[1][2]
    feature_cols = numerical_cols + categorical_cols

    missing = [c for c in feature_cols if c not in race_laps.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns in data: {missing}")

    X = race_laps[feature_cols]
    y_actual = race_laps["LapTime_seconds"]

    y_pred = model.predict(X)

    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)

    logger.info("MAE: %.3f seconds  |  R²: %.4f", mae, r2)

    # Build comparison table
    results = race_laps[["Driver", "Team", "LapNumber", "TireCompound", "TireAge",
                          "Position", "LapTime_seconds"]].copy()
    results["Predicted_seconds"] = np.round(y_pred, 3)
    results["Error_seconds"] = np.round(results["Predicted_seconds"] - results["LapTime_seconds"], 3)
    results["Abs_Error_seconds"] = results["Error_seconds"].abs()
    results = results.rename(columns={"LapTime_seconds": "Actual_seconds"})

    # Print summary
    print(f"\n{'='*65}")
    print(f"  Lap Time Evaluation — Round {race_round} ({config.pipeline.season} Season)")
    print(f"{'='*65}")
    print(f"  Laps evaluated : {len(results)}")
    print(f"  MAE            : {mae:.3f} seconds")
    print(f"  R²             : {r2:.4f}")
    print(f"{'='*65}")

    # Per-driver breakdown
    driver_summary = (
        results.groupby("Driver")
        .agg(
            Laps=("Actual_seconds", "count"),
            Actual_avg=("Actual_seconds", "mean"),
            Predicted_avg=("Predicted_seconds", "mean"),
            MAE=("Abs_Error_seconds", "mean"),
        )
        .round(3)
        .sort_values("MAE")
    )
    print("\nPer-Driver Breakdown (sorted by MAE):")
    print(driver_summary.to_string())

    # Sample rows
    print("\nSample Laps (10 random):")
    sample = results.sample(min(10, len(results)), random_state=42)[
        ["Driver", "LapNumber", "TireCompound", "TireAge",
         "Actual_seconds", "Predicted_seconds", "Error_seconds"]
    ].sort_values("LapNumber")
    print(sample.to_string(index=False))

    # Save plots
    _save_evaluation_plots(results, race_round, config.pipeline.season, mae, r2, plots_dir)

    return results


def _save_evaluation_plots(
    results: pd.DataFrame,
    race_round: int,
    season: int,
    mae: float,
    r2: float,
    plots_dir,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scatter: actual vs predicted
    try:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(results["Actual_seconds"], results["Predicted_seconds"],
                   alpha=0.35, s=12, color="steelblue")
        lims = [
            min(results["Actual_seconds"].min(), results["Predicted_seconds"].min()) - 1,
            max(results["Actual_seconds"].max(), results["Predicted_seconds"].max()) + 1,
        ]
        ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
        ax.set_xlabel("Actual Lap Time (s)")
        ax.set_ylabel("Predicted Lap Time (s)")
        ax.set_title(f"Actual vs Predicted — Round {race_round} ({season})\nMAE={mae:.3f}s  R²={r2:.4f}")
        ax.legend()
        path = plots_dir / f"round{race_round}_actual_vs_predicted.png"
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        logger.info("Saved %s", path.name)
    except Exception as e:
        logger.warning("Skipping scatter plot: %s", e)

    # 2. Residuals histogram
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(results["Error_seconds"], bins=50, color="coral", edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Prediction Error (Predicted − Actual, seconds)")
        ax.set_ylabel("Count")
        ax.set_title(f"Residuals Distribution — Round {race_round} ({season})")
        path = plots_dir / f"round{race_round}_residuals.png"
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        logger.info("Saved %s", path.name)
    except Exception as e:
        logger.warning("Skipping residuals plot: %s", e)

    # 3. MAE per driver bar chart
    try:
        driver_mae = (
            results.groupby("Driver")["Abs_Error_seconds"]
            .mean()
            .sort_values()
        )
        fig, ax = plt.subplots(figsize=(8, max(4, len(driver_mae) * 0.35)))
        ax.barh(driver_mae.index.astype(str), driver_mae.values, color="steelblue")
        ax.axvline(mae, color="red", linestyle="--", linewidth=1, label=f"Overall MAE ({mae:.3f}s)")
        ax.set_xlabel("MAE (seconds)")
        ax.set_title(f"Per-Driver MAE — Round {race_round} ({season})")
        ax.legend()
        path = plots_dir / f"round{race_round}_driver_mae.png"
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        logger.info("Saved %s", path.name)
    except Exception as e:
        logger.warning("Skipping driver MAE plot: %s", e)
