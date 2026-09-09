from __future__ import annotations

import logging
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from pipeline.config_loader import Config
from pipeline.features import build_position_features, create_historical_features

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


def evaluate_position(config: Config, race_round: int | None = None) -> pd.DataFrame:
    """Compare the position model's predicted finishing order vs the actual result
    for a completed race. Mirrors ``evaluate_laptime``.

    NOTE: this is a post-hoc audit. The *features* are leakage-safe (build_position
    _features applies shift-before-roll), but the loaded model was trained on data
    that INCLUDES this race, so the score is still optimistic. Use
    ``scripts/backtest.py`` for an honest held-out-model score.
    """
    import pickle

    from pipeline.feedback import score_prediction

    data_dir = config.paths.data_dir
    models_dir = config.paths.models_dir
    plots_dir = config.paths.plots_dir / "evaluation"
    season = config.pipeline.season

    results_df = pd.read_csv(data_dir / "f1_results_features.csv")
    season_df = results_df[results_df["Year"] == season].copy()
    available_races = sorted(int(r) for r in season_df["Race"].unique())
    if not available_races:
        raise RuntimeError("No results feature data found. Run: python main.py features")

    if race_round is None:
        race_round = available_races[-1]
        logger.info("No race specified — defaulting to most recent: Round %d", race_round)
    if race_round not in available_races:
        raise ValueError(
            f"Round {race_round} not in data. Available rounds: {available_races}"
        )

    processed = build_position_features(season_df, config)
    rows = (
        processed[processed["Race"] == race_round]
        .groupby("Driver")
        .last()
        .reset_index()
    )
    rows = rows[rows["form_avg_finish_s5"].notna()].reset_index(drop=True)
    if rows.empty:
        raise RuntimeError(f"No usable driver rows for Round {race_round}.")

    with open(models_dir / "race_prediction_pipeline.pkl", "rb") as f:
        model = pickle.load(f)

    from pipeline.model_registry import model_feature_columns

    feature_cols = model_feature_columns(model)
    missing = [c for c in feature_cols if c not in rows.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns in data: {missing}")

    y_pred = model.predict(rows[feature_cols])

    out = rows[["Driver", "Team", "GridPosition", "Position"]].copy()
    out = out.rename(columns={"Position": "ActualPosition"})
    out["PredictedPosition"] = np.round(y_pred, 2)
    out = out.sort_values("PredictedPosition").reset_index(drop=True)
    out["PredRank"] = out.index + 1
    out["Error"] = np.round(out["PredRank"] - out["ActualPosition"], 2)
    out["AbsError"] = out["Error"].abs()

    metrics = score_prediction(
        out[["PredRank", "Driver", "PredictedPosition"]],
        out[["Driver"]].assign(Position=out["ActualPosition"]),
    )

    print(f"\n{'='*65}")
    print(f"  Finishing-Position Evaluation — Round {race_round} ({season} Season)")
    print(f"{'='*65}")
    print(f"  Drivers evaluated : {metrics['n_drivers']}")
    print(f"  Winner correct    : {metrics['winner_correct']}")
    print(f"  Podium overlap    : {metrics['podium_overlap']}/3   (exact {metrics['podium_exact']}/3)")
    print(f"  Top-5 / Top-10    : {metrics['top5']}/5   {metrics['top10']}/10")
    print(f"  Spearman          : {metrics['spearman']:.3f}")
    print(f"  Position MAE      : {metrics['position_mae']:.2f}   RMSE {metrics['position_rmse']:.2f}")
    print(f"{'='*65}")
    print("\nPer-Driver (sorted by abs error):")
    print(
        out.sort_values("AbsError", ascending=False)[
            ["PredRank", "Driver", "Team", "PredictedPosition", "ActualPosition", "Error"]
        ].to_string(index=False)
    )
    print("\n  (post-hoc audit — features are leakage-safe, but the model was")
    print("   trained on this race; see scripts/backtest.py for a held-out score.)")

    _save_position_eval_plots(out, race_round, season, metrics, plots_dir)
    return out


def _save_position_eval_plots(out, race_round, season, metrics, plots_dir) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(out["ActualPosition"], out["PredRank"], color="steelblue", s=30)
        lim = [0, max(out["ActualPosition"].max(), out["PredRank"].max()) + 1]
        ax.plot(lim, lim, "r--", linewidth=1, label="Perfect")
        for _, r in out.iterrows():
            ax.annotate(str(r["Driver"]), (r["ActualPosition"], r["PredRank"]), fontsize=7)
        ax.set_xlabel("Actual finishing position")
        ax.set_ylabel("Predicted rank")
        ax.set_title(
            f"Predicted vs Actual — Round {race_round} ({season})\n"
            f"MAE={metrics['position_mae']:.2f}  Spearman={metrics['spearman']:.3f}"
        )
        ax.legend()
        fig.savefig(plots_dir / f"position_round{race_round}_pred_vs_actual.png", bbox_inches="tight", dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        logger.warning("Skipping position pred-vs-actual plot: %s", e)

    try:
        ordered = out.sort_values("PredRank")
        fig, ax = plt.subplots(figsize=(8, max(4, len(ordered) * 0.35)))
        ax.barh(ordered["Driver"].astype(str), ordered["Error"], color="coral")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Rank error (predicted rank − actual position)")
        ax.set_title(f"Per-Driver Rank Error — Round {race_round} ({season})")
        ax.invert_yaxis()
        fig.savefig(plots_dir / f"position_round{race_round}_rank_error.png", bbox_inches="tight", dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        logger.warning("Skipping position rank-error plot: %s", e)


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
