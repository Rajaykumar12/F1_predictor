from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime

import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline.config_loader import Config, get_config
from pipeline.features import create_historical_features

logger = logging.getLogger(__name__)


def _save_metrics(models_dir, key: str, metrics: dict) -> None:
    path = models_dir / "metrics.json"
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            pass
    existing[key] = {**metrics, "trained_at": datetime.now().isoformat(timespec="seconds")}
    path.write_text(json.dumps(existing, indent=2))


def train_laptime_model(config: Config) -> None:
    data_dir = config.paths.data_dir
    models_dir = config.paths.models_dir
    models_dir.mkdir(exist_ok=True)
    p = config.models.laptime

    laps = pd.read_csv(data_dir / "f1_laps_features.csv")

    all_features = [
        # Base
        "Race", "Driver", "Team", "Position", "TireCompound", "TireAge",
        "driver_win_rate", "team_reliability",
        # Tire (engineered in features.py)
        "TireCompound_encoded", "IsFreshTire", "StintLapNumber",
        # Race progression (engineered in features.py)
        "FuelLoadProxy", "LapNumber_normalized", "IsOutlap", "IsInlap",
        # Position dynamics
        "positions_gained", "tire_degradation",
        # Rolling form
        "RollingAvgLapTime_3", "RollingAvgLapTime_5", "LapTimeStd_5",
    ]
    features = [f for f in all_features if f in laps.columns]
    missing = [f for f in all_features if f not in laps.columns]
    if missing:
        logger.warning("Laptime model: %d features unavailable (skipped): %s", len(missing), missing)
    logger.info("Training lap time model with %d / %d features.", len(features), len(all_features))

    data = laps[features + ["LapTime_seconds"]].dropna()
    X, y = data[features], data["LapTime_seconds"]

    categorical = [c for c in ["Race", "Driver", "Team", "TireCompound"] if c in features]
    numerical = [c for c in features if c not in categorical]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", xgb.XGBRegressor(
            n_estimators=p.n_estimators,
            learning_rate=p.learning_rate,
            max_depth=p.max_depth,
            random_state=p.random_state,
        )),
    ])

    # Cross-validation metrics
    cv_mae = cross_val_score(pipeline, X, y, cv=5, scoring="neg_mean_absolute_error")
    cv_r2 = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
    logger.info(
        "Laptime CV (5-fold) — MAE: %.3f ± %.3f  R²: %.4f ± %.4f",
        -cv_mae.mean(), cv_mae.std(), cv_r2.mean(), cv_r2.std(),
    )

    # Final model on full data; hold-out for quick sanity check
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=p.random_state
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    holdout_mae = mean_absolute_error(y_test, y_pred)
    holdout_r2 = r2_score(y_test, y_pred)
    logger.info("Laptime holdout — MAE: %.3fs  R²: %.4f", holdout_mae, holdout_r2)

    with open(models_dir / "xgb_laptime_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    feature_info = {"features": features, "categorical": categorical, "numerical": numerical}
    with open(models_dir / "xgb_laptime_features.pkl", "wb") as f:
        pickle.dump(feature_info, f)

    _save_metrics(models_dir, "laptime", {
        "mae": round(holdout_mae, 4),
        "r2": round(holdout_r2, 4),
        "cv_mae_mean": round(-cv_mae.mean(), 4),
        "cv_mae_std": round(cv_mae.std(), 4),
        "cv_r2_mean": round(cv_r2.mean(), 4),
        "cv_r2_std": round(cv_r2.std(), 4),
        "features_used": len(features),
    })
    logger.info("Saved xgb_laptime_pipeline.pkl  (features=%d)", len(features))


def train_racewin_model(config: Config) -> None:
    data_dir = config.paths.data_dir
    models_dir = config.paths.models_dir
    models_dir.mkdir(exist_ok=True)
    p = config.models.racewin

    results = pd.read_csv(data_dir / "f1_results_features.csv")

    features = ["Team", "Position", "GridPosition", "driver_win_rate", "team_reliability"]
    if "BestQualifyingTime" in results.columns:
        features.extend(["BestQualifyingTime", "GapToPole", "QualifyingPerformance"])
        logger.info("Using qualifying features for race win model.")

    data = results[features + ["race_winner"]].dropna()
    X, y = data[features], data["race_winner"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), ["Team"]),
        ("num", "passthrough", [f for f in features if f != "Team"]),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb.XGBClassifier(
            n_estimators=p.n_estimators,
            learning_rate=p.learning_rate,
            max_depth=p.max_depth,
            random_state=p.random_state,
        )),
    ])

    # Stratified CV to handle imbalanced class (only 1 winner per race)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=p.random_state)
    cv_acc = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy")
    cv_f1 = cross_val_score(pipeline, X, y, cv=skf, scoring="f1")
    logger.info(
        "Racewin CV (5-fold stratified) — Accuracy: %.4f ± %.4f  F1: %.4f ± %.4f",
        cv_acc.mean(), cv_acc.std(), cv_f1.mean(), cv_f1.std(),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=p.random_state, stratify=y
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    holdout_acc = accuracy_score(y_test, y_pred)
    holdout_f1 = f1_score(y_test, y_pred, zero_division=0)
    logger.info(
        "Racewin holdout — Accuracy: %.4f  F1: %.4f", holdout_acc, holdout_f1
    )

    with open(models_dir / "xgb_racewin_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    _save_metrics(models_dir, "racewin", {
        "accuracy": round(holdout_acc, 4),
        "f1": round(holdout_f1, 4),
        "cv_accuracy_mean": round(cv_acc.mean(), 4),
        "cv_accuracy_std": round(cv_acc.std(), 4),
        "cv_f1_mean": round(cv_f1.mean(), 4),
        "cv_f1_std": round(cv_f1.std(), 4),
    })
    logger.info("Saved xgb_racewin_pipeline.pkl")


def train_position_model(config: Config) -> None:
    data_dir = config.paths.data_dir
    models_dir = config.paths.models_dir
    models_dir.mkdir(exist_ok=True)
    p = config.models.position
    n_prev = config.pipeline.lookback_races
    completed_statuses = config.constants.completed_statuses

    results = pd.read_csv(data_dir / "f1_results_features.csv")
    processed = create_historical_features(
        results, n_previous=n_prev, completed_statuses=completed_statuses
    )

    race_features = [
        "Driver", "Team", "GridPosition",
        "driver_win_rate", "team_reliability", "QualifyingPerformance", "PositionChange",
        "avg_position_last", "best_position_last", "avg_grid_last",
        "dnf_last", "reliability_rate", "avg_positions_gained",
        "podiums_last", "wins_last", "points_last", "form_trend",
    ]
    if "avg_quali_time" in processed.columns:
        race_features.extend(["avg_quali_time", "avg_gap_to_pole"])

    available = [f for f in race_features if f in processed.columns]
    data = processed[available + ["Position"]].dropna()
    X, y = data[available], data["Position"]

    categorical = ["Driver", "Team"]
    numerical = [f for f in available if f not in categorical]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", xgb.XGBRegressor(
            n_estimators=p.n_estimators,
            learning_rate=p.learning_rate,
            max_depth=p.max_depth,
            random_state=p.random_state,
        )),
    ])

    # Cross-validation
    cv_mae = cross_val_score(pipeline, X, y, cv=5, scoring="neg_mean_absolute_error")
    cv_r2 = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
    logger.info(
        "Position CV (5-fold) — MAE: %.3f ± %.3f  R²: %.4f ± %.4f",
        -cv_mae.mean(), cv_mae.std(), cv_r2.mean(), cv_r2.std(),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=p.random_state
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    holdout_mae = mean_absolute_error(y_test, y_pred)
    holdout_r2 = r2_score(y_test, y_pred)
    logger.info(
        "Position holdout — MAE: %.3f  R²: %.4f", holdout_mae, holdout_r2
    )

    with open(models_dir / "race_prediction_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    feature_info = {
        "n_features": len(available),
        "lookback_races": n_prev,
        "features": available,
        "has_qualifying": "avg_quali_time" in available,
    }
    with open(models_dir / "race_position_feature_info.pkl", "wb") as f:
        pickle.dump(feature_info, f)

    _save_metrics(models_dir, "position", {
        "mae": round(holdout_mae, 4),
        "r2": round(holdout_r2, 4),
        "cv_mae_mean": round(-cv_mae.mean(), 4),
        "cv_mae_std": round(cv_mae.std(), 4),
        "cv_r2_mean": round(cv_r2.mean(), 4),
        "cv_r2_std": round(cv_r2.std(), 4),
        "features_used": len(available),
    })
    logger.info(
        "Saved race_prediction_pipeline.pkl  (lookback=%d, features=%d)",
        n_prev, len(available),
    )


def run_training(config: Config | None = None, models: list[str] | None = None) -> None:
    if config is None:
        config = get_config()
    if models is None or "all" in models:
        models = ["laptime", "racewin", "position"]

    dispatchers = {
        "laptime": train_laptime_model,
        "racewin": train_racewin_model,
        "position": train_position_model,
    }

    for name in models:
        if name not in dispatchers:
            logger.error(
                "Unknown model '%s'. Choose from: laptime, racewin, position, all.", name
            )
            continue
        logger.info("--- Training %s model ---", name)
        dispatchers[name](config)
    logger.info("Training complete.")
