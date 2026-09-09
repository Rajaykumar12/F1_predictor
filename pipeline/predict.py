"""Framework-agnostic race finishing-order prediction.

This is the shared core behind both ``GET /predict_next_race`` (app.py) and the
``python main.py predict-race`` CLI command. The logic is lifted verbatim from
the original route handler so the API response is unchanged; the CLI calls the
same function instead of re-implementing it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from pipeline import feature_registry as fr
from pipeline.config_loader import Config
from pipeline.features import build_position_features
from pipeline.model_registry import model_feature_columns

logger = logging.getLogger(__name__)

_QUALI_OVERRIDE_COLS = ["GridPosition", "BestQualifyingTime", "GapToPole", "QualifyingPerformance"]


class ModelNotLoadedError(RuntimeError):
    """The position model pickle is missing / unreadable."""


class PredictionDataMissingError(FileNotFoundError):
    """The engineered results feature CSV does not exist yet."""


@dataclass
class DriverForecast:
    pred_rank: int
    driver: str
    team: str
    predicted_position: float        # post bias-correction, 2 dp
    raw_predicted_position: float     # model output before bias-correction
    confidence: float
    recent_form: dict


@dataclass
class PredictionResult:
    forecasts: list[DriverForecast]
    race_label: str
    using_real_qualifying: bool
    model_r2: float | None
    lookback: int
    as_of_round: int | None
    prediction_date: str
    next_race: str
    bias_applied: dict[str, float] | None = None


def predict_race(
    config: Config,
    race_model,
    metrics: dict,
    *,
    lookback: int,
    results_features_path=None,
    qualifying_path=None,
    as_of_round: int | None = None,
    bias: dict[str, float] | None = None,
) -> PredictionResult:
    """Predict the finishing order for the next (or a held-out) race.

    Raises ``ModelNotLoadedError`` / ``PredictionDataMissingError`` / ``ValueError``
    which each caller maps to its own error convention.
    """
    if race_model is None:
        raise ModelNotLoadedError(
            "Race position model not loaded. Train it first (position model)."
        )

    results_path = results_features_path or (config.paths.data_dir / "f1_results_features.csv")
    try:
        f1_results = pd.read_csv(results_path)
    except FileNotFoundError as e:
        raise PredictionDataMissingError(
            f"Results feature data not found at {results_path}. Run feature engineering."
        ) from e

    season = config.pipeline.season
    season_data = f1_results[f1_results["Year"] == season].copy()
    logger.info("Using %d records from %s season.", len(season_data), season)

    # One code path with training: build_position_features assembles the exact
    # leakage-safe registry feature set (create_historical_features + family
    # builders), honouring the as-of-round cutoff for an honest snapshot.
    processed = build_position_features(
        season_data, config, as_of_round=as_of_round, lookback=lookback
    )
    if processed.empty:
        raise ValueError("No data after processing.")

    latest = processed.groupby("Driver").last().reset_index()
    latest = latest[latest["avg_position_last"].notna()].reset_index(drop=True)

    upcoming_path = Path(qualifying_path) if qualifying_path else (config.paths.data_dir / "upcoming_qualifying.csv")
    using_real_qualifying = False
    race_label = "Next Grand Prix"
    if upcoming_path.exists():
        quali = pd.read_csv(upcoming_path)
        race_label = quali["RaceName"].iloc[0] if "RaceName" in quali.columns else "Next Grand Prix"
        for _, q_row in quali.iterrows():
            driver_mask = latest["Driver"] == q_row["Driver"]
            if not driver_mask.any():
                continue
            for col in _QUALI_OVERRIDE_COLS:
                if col in quali.columns:
                    latest.loc[driver_mask, col] = q_row[col]
        using_real_qualifying = True
        logger.info("Applied real qualifying data from %s (%d drivers).", race_label, len(quali))
    else:
        logger.info("No upcoming_qualifying.csv found — using historical grid positions.")

    # The feature list is read straight off the fitted model, so it can never
    # drift from what train.py used.
    race_features = model_feature_columns(race_model)
    # Forecast-time guard: never feed a feature that needs post-race information
    # (none in the registry today, but this keeps a future as_of_safe=False
    # feature out of a real prediction automatically).
    unsafe = set(fr.as_of_unsafe(fr.all_features()))
    dropped_unsafe = [f for f in race_features if f in unsafe]
    if dropped_unsafe:
        logger.warning("Dropping as-of-unsafe features from the forecast: %s", dropped_unsafe)
        race_features = [f for f in race_features if f not in unsafe]
    missing = [f for f in race_features if f not in latest.columns]
    if missing:
        logger.warning("Missing features for prediction: %s", missing)
    available = [f for f in race_features if f in latest.columns]

    raw_predictions = race_model.predict(latest[available])

    model_r2 = metrics.get("position", {}).get("r2")
    bias = bias or None

    forecasts: list[DriverForecast] = []
    for idx, row in latest.iterrows():
        try:
            form_consistency = 1.0 - min(1.0, abs(float(row.get("form_trend", 0))) / 5.0)
            driver_confidence = round(
                (model_r2 if model_r2 is not None else 0.5) * form_consistency, 3
            )
            recent_form = {
                "avg_position": round(float(row["avg_position_last"]), 2),
                "best_position": int(row["best_position_last"]) if "best_position_last" in row else None,
                "podiums": int(row["podiums_last"]),
                "wins": int(row["wins_last"]) if "wins_last" in row else 0,
                "dnfs": int(row["dnf_last"]),
                "reliability": round(float(row["reliability_rate"]) * 100, 1) if "reliability_rate" in row else None,
                "form_trend": round(float(row["form_trend"]), 2) if pd.notna(row.get("form_trend")) else None,
                "driver_win_rate": round(float(row["driver_win_rate"]) * 100, 1) if "driver_win_rate" in row else None,
                "team_reliability": round(float(row["team_reliability"]), 1) if "team_reliability" in row else None,
                "quali_performance": round(float(row["QualifyingPerformance"]), 2) if "QualifyingPerformance" in row else None,
            }
            driver = str(row["Driver"])
            raw_pos = round(float(raw_predictions[idx]), 2)
            adj = raw_pos
            if bias and driver in bias:
                adj = max(1.0, round(raw_pos - float(bias[driver]), 2))
            forecasts.append(DriverForecast(
                pred_rank=0,  # filled after sort
                driver=driver,
                team=str(row["Team"]),
                predicted_position=adj,
                raw_predicted_position=raw_pos,
                confidence=driver_confidence,
                recent_form=recent_form,
            ))
        except Exception as e:  # noqa: BLE001 — skip a bad row, mirror original
            logger.error("Error processing %s: %s", row.get("Driver", "Unknown"), e)

    if not forecasts:
        raise ValueError("No valid predictions generated.")

    forecasts.sort(key=lambda f: f.predicted_position)
    for rank, f in enumerate(forecasts, start=1):
        f.pred_rank = rank
    forecasts = forecasts[:30]

    quali_note = "real qualifying" if using_real_qualifying else "historical grid positions"
    return PredictionResult(
        forecasts=forecasts,
        race_label=race_label,
        using_real_qualifying=using_real_qualifying,
        model_r2=model_r2,
        lookback=lookback,
        as_of_round=as_of_round,
        prediction_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        next_race=f"{race_label} — {quali_note}, last {lookback} races form",
        bias_applied=bias,
    )
