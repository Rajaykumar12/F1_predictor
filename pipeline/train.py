from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline import feature_registry as fr
from pipeline.config_loader import Config, get_config
from pipeline.features import build_position_features, create_historical_features
from pipeline.model_registry import MODEL_NAMES

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


def _mean_vif(frame: pd.DataFrame, cols: list[str]) -> float:
    """Mean variance-inflation factor across ``cols`` (standardized + intercept).
    Returns NaN if it cannot be computed (too few rows / singular)."""
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except Exception:  # noqa: BLE001
        return float("nan")
    z = frame[cols].dropna()
    if len(z) <= len(cols) + 1:
        return float("nan")
    z = (z - z.mean()) / z.std(ddof=0).replace(0, np.nan)
    z = z.dropna(axis=1)
    if z.shape[1] < 2:
        return float("nan")
    x = np.column_stack([np.ones(len(z))] + [z[c].to_numpy() for c in z.columns])
    vifs = []
    for i in range(1, x.shape[1]):
        try:
            vifs.append(float(variance_inflation_factor(x, i)))
        except Exception:  # noqa: BLE001
            pass
    return float(np.mean(vifs)) if vifs else float("nan")


def select_numeric_features(
    frame: pd.DataFrame, numeric: list[str], target: str, method: str, vif_threshold: float
) -> tuple[list[str], list[str]]:
    """Pick the numeric feature subset for the position pipeline.

    ``none``  -> keep all.
    ``vif``   -> greedily drop the highest-VIF feature while any VIF exceeds
                 ``vif_threshold`` (never below 3 features).
    ``lasso`` -> LassoCV on standardized numerics; keep non-zero coefficients
                 (fall back to all if L1 zeroes everything).
    Returns ``(kept, dropped)``.
    """
    d = frame[numeric + [target]].dropna()
    if method == "none" or len(d) <= len(numeric) + 2:
        return list(numeric), []

    if method == "lasso":
        from sklearn.linear_model import LassoCV

        z = (d[numeric] - d[numeric].mean()) / d[numeric].std(ddof=0).replace(0, np.nan)
        z = z.fillna(0.0)
        model = LassoCV(cv=5, random_state=0, n_alphas=50, max_iter=5000).fit(z, d[target])
        kept = [c for c, coef in zip(numeric, model.coef_) if abs(coef) > 1e-8]
        if len(kept) < 3:
            logger.warning("Lasso selection kept <3 features — keeping all numerics.")
            return list(numeric), []
        return kept, [c for c in numeric if c not in kept]

    if method == "vif":
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        kept = list(numeric)
        dropped: list[str] = []
        while len(kept) > 3:
            z = (d[kept] - d[kept].mean()) / d[kept].std(ddof=0).replace(0, np.nan)
            z = z.dropna(axis=1)
            if z.shape[1] < 2:
                break
            x = np.column_stack([np.ones(len(z))] + [z[c].to_numpy() for c in z.columns])
            vifs = {c: float(variance_inflation_factor(x, i + 1)) for i, c in enumerate(z.columns)}
            worst, worst_vif = max(vifs.items(), key=lambda kv: kv[1])
            if worst_vif <= vif_threshold:
                break
            kept.remove(worst)
            dropped.append(worst)
            logger.info("VIF-drop: %s (VIF=%.1f)", worst, worst_vif)
        return kept, dropped

    raise ValueError(f"unknown feature_selection method {method!r}")


def _forward_chain_split(df: pd.DataFrame, holdout_rounds: int) -> tuple[pd.Index, pd.Index]:
    """Train on the earliest rounds, test on the last ``holdout_rounds`` — the
    only honest temporal split available in a single growing season."""
    rounds = sorted(df["Race"].unique())
    if len(rounds) <= holdout_rounds + 1:
        holdout_rounds = max(1, len(rounds) // 4)
    test_rounds = set(rounds[-holdout_rounds:])
    test_idx = df.index[df["Race"].isin(test_rounds)]
    train_idx = df.index[~df["Race"].isin(test_rounds)]
    return train_idx, test_idx


def train_position_model(config: Config) -> None:
    data_dir = config.paths.data_dir
    models_dir = config.paths.models_dir
    models_dir.mkdir(exist_ok=True)
    p = config.models.position
    ft = config.features

    results = pd.read_csv(data_dir / "f1_results_features.csv")

    # The results feature CSV already carries the leakage-safe registry features
    # (baked in by run_feature_engineering); re-assemble only if a column is
    # somehow absent (e.g. an older CSV).
    feats = fr.enabled_features(config)
    want = fr.feature_names(feats)
    if any(c not in results.columns for c in want):
        logger.info("Registry columns missing from CSV — re-running build_position_features.")
        results = build_position_features(results, config)

    categorical = [c for c in fr.categorical_names(feats) if c in results.columns]
    numeric_all = [c for c in fr.numeric_names(feats) if c in results.columns]

    data = results[["Race"] + numeric_all + categorical + ["Position"]].dropna(
        subset=numeric_all + ["Position"]
    ).reset_index(drop=True)
    logger.info(
        "Position training rows: %d (from %d; %d dropped for missing history).",
        len(data), len(results), len(results) - len(data),
    )

    # --- optional feature selection (L1 / greedy-VIF) on the numerics --------- #
    numeric, dropped = select_numeric_features(
        data, numeric_all, "Position", ft.feature_selection, ft.vif_threshold
    )
    if dropped:
        logger.info("feature_selection=%s dropped: %s", ft.feature_selection, dropped)
    features = numeric + categorical
    mean_vif = _mean_vif(data, numeric)

    X, y, groups = data[features], data["Position"], data["Race"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
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

    # --- honest CV: GroupKFold(by Race) — a race is never split across folds -- #
    n_splits = min(ft.cv_splits, int(groups.nunique()))
    gkf = GroupKFold(n_splits=n_splits)
    cv_mae = cross_val_score(pipeline, X, y, cv=gkf, groups=groups,
                             scoring="neg_mean_absolute_error")
    cv_r2 = cross_val_score(pipeline, X, y, cv=gkf, groups=groups, scoring="r2")
    logger.info(
        "Position GroupKFold(%d, by Race) — MAE: %.3f ± %.3f  R²: %.4f ± %.4f",
        n_splits, -cv_mae.mean(), cv_mae.std(), cv_r2.mean(), cv_r2.std(),
    )

    # --- headline metric: forward-chaining holdout on the last k rounds ------ #
    train_idx, test_idx = _forward_chain_split(data, ft.holdout_rounds)
    pipeline.fit(X.loc[train_idx], y.loc[train_idx])
    y_hat = pipeline.predict(X.loc[test_idx])
    fc_mae = mean_absolute_error(y.loc[test_idx], y_hat)
    fc_r2 = r2_score(y.loc[test_idx], y_hat)
    fc_spearman = float(spearmanr(y.loc[test_idx], y_hat).correlation)
    held_rounds = sorted(data.loc[test_idx, "Race"].unique().tolist())
    logger.info(
        "Position forward-chain holdout (rounds %s) — MAE: %.3f  R²: %.4f  Spearman: %.3f",
        held_rounds, fc_mae, fc_r2, fc_spearman,
    )

    # --- final model on all rows ------------------------------------------- #
    pipeline.fit(X, y)
    with open(models_dir / "race_prediction_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    # The feature list lives inside the fitted pipeline's ColumnTransformer;
    # model_registry.model_feature_columns() reads it back at predict/evaluate
    # time, so it isn't persisted separately.

    _save_metrics(models_dir, "position", {
        "mae": round(fc_mae, 4),                 # headline = forward-chain holdout
        "r2": round(fc_r2, 4),
        "forward_chain_mae": round(fc_mae, 4),
        "forward_chain_r2": round(fc_r2, 4),
        "forward_chain_spearman": round(fc_spearman, 4),
        "forward_chain_holdout_rounds": held_rounds,
        "cv_mae_mean": round(-cv_mae.mean(), 4),
        "cv_mae_std": round(cv_mae.std(), 4),
        "cv_r2_mean": round(cv_r2.mean(), 4),
        "cv_r2_std": round(cv_r2.std(), 4),
        "cv": f"GroupKFold({n_splits}, by Race)",
        "registry_version": fr.REGISTRY_VERSION,
        "feature_selection": ft.feature_selection,
        "features_used": features,
        "features_dropped_by_selection": dropped,
        "mean_vif": None if np.isnan(mean_vif) else round(mean_vif, 3),
        "documented_baseline": {"leaky_cv_mae": 1.21, "honest_groupkfold_mae": 3.59},
    })
    logger.info(
        "Saved race_prediction_pipeline.pkl  (%d features, selection=%s, mean VIF=%.2f)",
        len(features), ft.feature_selection, mean_vif,
    )


def run_training(config: Config | None = None, models: list[str] | None = None) -> None:
    if config is None:
        config = get_config()
    if models is None or "all" in models:
        models = list(MODEL_NAMES)

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
