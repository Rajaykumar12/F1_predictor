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
from pipeline.feedback import sharp_end_metrics_by_race
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

    # NOTE: "Position" was previously in this list — the actual finishing
    # position, target-identical to race_winner (Position == 1). That let the
    # model trivially "predict" the winner from its own answer and is exactly
    # the leakage class this project's redesign otherwise eliminated (see
    # docs/feature-engineering-redesign-plan.md); removed here as a C2
    # prerequisite, since its predict_proba is about to be blended INTO the
    # position model — leaving it in would leak the position target right
    # back in through the side door.
    features = ["Team", "GridPosition", "driver_win_rate", "team_reliability"]
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
    """Train on the earliest races, test on the last ``holdout_rounds`` — the
    only honest temporal split available in a single growing season. Orders by
    ``race_seq`` (B2 — monotonic across seasons) when present, falling back to
    the season-local ``Race`` column for single-season frames / older CSVs."""
    order_col = "race_seq" if "race_seq" in df.columns else "Race"
    rounds = sorted(df[order_col].unique())
    if len(rounds) <= holdout_rounds + 1:
        holdout_rounds = max(1, len(rounds) // 4)
    test_rounds = set(rounds[-holdout_rounds:])
    test_idx = df.index[df[order_col].isin(test_rounds)]
    train_idx = df.index[~df[order_col].isin(test_rounds)]
    return train_idx, test_idx


def prepare_position_data(config: Config, results: pd.DataFrame | None = None) -> dict:
    """Load (or reuse) the leakage-safe position-model frame, select the
    registry's numeric/categorical columns, drop rows with missing history,
    and run feature selection. Shared by :func:`train_position_model` (the
    production fit) and ``scripts/rolling_backtest.py`` (per-round
    walk-forward retrain) so both train on identical columns."""
    data_dir = config.paths.data_dir
    ft = config.features

    if results is None:
        results = pd.read_csv(data_dir / "f1_results_features.csv")

    # The results feature CSV already carries the leakage-safe registry features
    # (baked in by run_feature_engineering); re-assemble only if a column is
    # somehow absent (e.g. an older CSV).
    feats = fr.enabled_features(config)
    want = fr.feature_names(feats)
    if any(c not in results.columns for c in want) or "race_seq" not in results.columns:
        logger.info("Registry columns missing from CSV — re-running build_position_features.")
        results = build_position_features(results, config)

    categorical = [c for c in fr.categorical_names(feats) if c in results.columns]
    numeric_all = [c for c in fr.numeric_names(feats) if c in results.columns]

    # race_seq (B2) — the dense-rank (Year, Race) key — is what GroupKFold and
    # the forward-chain split order/group by; raw "Race" repeats across
    # seasons and would wrongly fold e.g. 2024-round-5 with 2025-round-5.
    meta_cols = ["race_seq", "Race"] + (["Year"] if "Year" in results.columns else [])
    data = results[meta_cols + numeric_all + categorical + ["Position"]].dropna(
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

    return {
        "data": data,
        "numeric": numeric,
        "categorical": categorical,
        "features": numeric + categorical,
        "dropped": dropped,
        "mean_vif": _mean_vif(data, numeric),
    }


def prepare_dnf_data(config: Config, results: pd.DataFrame | None = None) -> dict:
    """E1 — same leakage-safe registry columns as :func:`prepare_position_data`
    (no ``Position`` needed as a feature; DNF is knowable pre-race only from
    reliability/circuit/history signals, never the finishing order), target
    is ``is_dnf`` (canonical status, see :func:`pipeline.clean.normalize_status`)
    instead of ``Position``. Unlike the position target, a DNF row's own
    ``Position`` may be missing/irrelevant, so rows are kept as long as the
    FEATURE columns have history — not gated on ``Position`` being present."""
    data_dir = config.paths.data_dir
    ft = config.features

    if results is None:
        results = pd.read_csv(data_dir / "f1_results_features.csv")

    feats = fr.enabled_features(config)
    want = fr.feature_names(feats)
    if any(c not in results.columns for c in want) or "race_seq" not in results.columns:
        results = build_position_features(results, config)

    from pipeline.features import _status_col  # local import: internal helper

    completed = config.constants.completed_statuses
    results = results.assign(is_dnf=(~_status_col(results).isin(completed)).astype(int))

    categorical = [c for c in fr.categorical_names(feats) if c in results.columns]
    numeric_all = [c for c in fr.numeric_names(feats) if c in results.columns]

    meta_cols = ["race_seq", "Race"] + (["Year"] if "Year" in results.columns else [])
    data = results[meta_cols + numeric_all + categorical + ["is_dnf"]].dropna(
        subset=numeric_all
    ).reset_index(drop=True)
    logger.info(
        "DNF training rows: %d (from %d; %d dropped for missing history). Positive rate: %.1f%%",
        len(data), len(results), len(results) - len(data),
        100 * data["is_dnf"].mean() if len(data) else float("nan"),
    )

    numeric, dropped = select_numeric_features(
        data, numeric_all, "is_dnf", ft.feature_selection, ft.vif_threshold
    )
    if dropped:
        logger.info("DNF feature_selection=%s dropped: %s", ft.feature_selection, dropped)

    return {
        "data": data,
        "numeric": numeric,
        "categorical": categorical,
        "features": numeric + categorical,
        "dropped": dropped,
    }


def build_position_pipeline(
    model_params, numeric: list[str], categorical: list[str], extra_xgb_params: dict | None = None,
) -> Pipeline:
    """Fresh, unfit preprocessor + XGBRegressor pipeline for the position model.
    ``extra_xgb_params`` (D3) overrides/extends the four base hyperparameters —
    e.g. optuna's ``reg_alpha``/``reg_lambda``, which aren't in ``ModelParams``."""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    xgb_params = {
        "n_estimators": model_params.n_estimators,
        "learning_rate": model_params.learning_rate,
        "max_depth": model_params.max_depth,
        "random_state": model_params.random_state,
        **(extra_xgb_params or {}),
    }
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", xgb.XGBRegressor(**xgb_params)),
    ])


class PositionModel:
    """D1 — wraps a fitted preprocessor+regressor ``Pipeline`` so every
    external caller (``predict.py``, ``evaluate.py``, ``scripts/backtest.py``,
    ``scripts/rolling_backtest.py``, ``model_registry``) keeps getting a raw
    finishing-**Position** estimate from ``.predict()``, regardless of what
    the pipeline is actually trained to predict internally.

    With ``target="positions_gained"`` the wrapped pipeline predicts
    ``GridPosition - Position`` — removing the dominant grid-position
    variance so the model has to learn pace/racecraft/strategy instead of
    mostly memorizing the grid slot — and ``.predict()`` reconstructs
    ``finish = GridPosition - Δ̂`` before returning. With ``target="position"``
    it's a passthrough (the pre-D1 behaviour). Every caller of ``.predict()``
    is unchanged either way; only :func:`fit_position_model` and
    :func:`train_position_model` know the target was ever transformed.
    """

    def __init__(self, pipeline: Pipeline, target: str = "position",
                 grid_col: str = "GridPosition"):
        self.pipeline = pipeline
        self.target = target
        self.grid_col = grid_col

    @property
    def named_steps(self):
        # model_registry.model_feature_columns() reads this straight off the
        # fitted ColumnTransformer — proxy it so callers never need to know
        # whether they hold a raw Pipeline or a PositionModel.
        return self.pipeline.named_steps

    def predict(self, X) -> np.ndarray:
        raw = np.asarray(self.pipeline.predict(X))
        if self.target != "positions_gained":
            return raw
        grid = np.asarray(X[self.grid_col], dtype=float)
        return grid - raw


def fit_position_model(
    pipeline: Pipeline, data: pd.DataFrame, idx, numeric: list[str], categorical: list[str],
    position_target: str = "position",
) -> PositionModel:
    """Fit ``pipeline`` on ``data.loc[idx]`` against the configured target (D1)
    and return it wrapped in :class:`PositionModel` so ``.predict()`` always
    yields a Position estimate. Shared by :func:`train_position_model` (the
    production fit) and ``scripts/rolling_backtest.py`` (per-round
    walk-forward retrain)."""
    features = numeric + categorical
    X = data.loc[idx, features]
    if position_target == "positions_gained":
        y = data.loc[idx, "GridPosition"] - data.loc[idx, "Position"]
    else:
        y = data.loc[idx, "Position"]
    pipeline.fit(X, y)
    return PositionModel(pipeline, target=position_target)


def tune_position_hyperparams(
    data: pd.DataFrame, numeric: list[str], categorical: list[str], groups: pd.Series,
    config: Config, position_target: str,
) -> dict:
    """D3 — optuna search over XGBRegressor hyperparameters, optimizing mean
    GroupKFold(by race) MAE in the model's own target domain. Gated by
    ``config.features.tune``: ``"quick"`` (15 trials) | ``"full"`` (50).
    ``config.models.position`` (n_estimators=1000, lr=0.01, depth=5, fixed) is
    over-parameterised for ~260 rows of single-season data — expect the
    search to prefer ``max_depth`` 2-4 and non-zero ``reg_alpha``/``reg_lambda``."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_trials = {"quick": 15, "full": 50}[config.features.tune]
    ft = config.features
    y = (data["GridPosition"] - data["Position"]) if position_target == "positions_gained" else data["Position"]
    X = data[numeric + categorical]
    n_splits = min(ft.cv_splits, int(groups.nunique()))
    gkf = GroupKFold(n_splits=n_splits)
    base_random_state = config.models.position.random_state

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": base_random_state,
        }
        pipeline = build_position_pipeline(config.models.position, numeric, categorical, params)
        scores = cross_val_score(
            pipeline, X, y, cv=gkf, groups=groups, scoring="neg_mean_absolute_error"
        )
        return float(-scores.mean())

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=base_random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(
        "Optuna tune (%s, %d trials) — best target-domain MAE %.3f, params %s",
        ft.tune, n_trials, study.best_value, study.best_params,
    )
    return study.best_params


def train_position_model(config: Config) -> None:
    data_dir = config.paths.data_dir
    models_dir = config.paths.models_dir
    models_dir.mkdir(exist_ok=True)
    p = config.models.position
    ft = config.features

    prepared = prepare_position_data(config)
    data = prepared["data"]
    numeric, categorical, features = prepared["numeric"], prepared["categorical"], prepared["features"]
    dropped, mean_vif = prepared["dropped"], prepared["mean_vif"]

    # Group by race_seq (B2), not the season-local "Race" — otherwise
    # 2024-round-5 and 2025-round-5 would wrongly fold together as "one race".
    # D1: the model's actual training TARGET is config-selectable
    # (position_target: positions_gained | position); CV below fits/scores in
    # that same target's own domain (labelled in the log), while the
    # headline forward-chain holdout always reports true Position-domain
    # metrics — via PositionModel's reconstruction — so it stays comparable
    # across target modes and to the documented pre-D1 baseline.
    target_mode = ft.position_target
    X, groups = data[features], data["race_seq"]
    y_cv = (data["GridPosition"] - data["Position"]) if target_mode == "positions_gained" else data["Position"]

    # --- D3: optional optuna search, gated by features.tune ------------------ #
    tuned_params: dict | None = None
    if ft.tune != "none":
        tuned_params = tune_position_hyperparams(data, numeric, categorical, groups, config, target_mode)

    pipeline = build_position_pipeline(p, numeric, categorical, tuned_params)

    # --- honest CV: GroupKFold(by race) — a race is never split across folds -- #
    n_splits = min(ft.cv_splits, int(groups.nunique()))
    gkf = GroupKFold(n_splits=n_splits)
    cv_mae = cross_val_score(pipeline, X, y_cv, cv=gkf, groups=groups,
                             scoring="neg_mean_absolute_error")
    cv_r2 = cross_val_score(pipeline, X, y_cv, cv=gkf, groups=groups, scoring="r2")
    logger.info(
        "Position GroupKFold(%d, by race, target=%s) — MAE: %.3f ± %.3f  R²: %.4f ± %.4f",
        n_splits, target_mode, -cv_mae.mean(), cv_mae.std(), cv_r2.mean(), cv_r2.std(),
    )

    # --- headline metric: forward-chaining holdout on the last k rounds ------ #
    train_idx, test_idx = _forward_chain_split(data, ft.holdout_rounds)
    pipeline = build_position_pipeline(p, numeric, categorical, tuned_params)
    fc_model = fit_position_model(pipeline, data, train_idx, numeric, categorical, target_mode)
    y_hat = fc_model.predict(X.loc[test_idx])            # always Position-domain
    y_true = data.loc[test_idx, "Position"]
    fc_mae = mean_absolute_error(y_true, y_hat)
    fc_r2 = r2_score(y_true, y_hat)
    fc_spearman = float(spearmanr(y_true, y_hat).correlation)
    if "Year" in data.columns:
        held_rounds = sorted(
            {(int(y_), int(r)) for y_, r in data.loc[test_idx, ["Year", "Race"]].itertuples(index=False)}
        )
    else:
        held_rounds = sorted(data.loc[test_idx, "Race"].unique().tolist())
    sharp = sharp_end_metrics_by_race(y_hat, y_true, data.loc[test_idx, "race_seq"])
    logger.info(
        "Position forward-chain holdout (rounds %s, target=%s) — MAE: %.3f  R²: %.4f  Spearman: %.3f  "
        "winner_logloss: %.3f  podium_brier: %.3f  points_brier: %.3f",
        held_rounds, target_mode, fc_mae, fc_r2, fc_spearman,
        sharp["winner_logloss"], sharp["podium_brier"], sharp["points_brier"],
    )

    # --- final model on all rows ------------------------------------------- #
    pipeline = build_position_pipeline(p, numeric, categorical)
    final_model = fit_position_model(pipeline, data, data.index, numeric, categorical, target_mode)
    with open(models_dir / "race_prediction_pipeline.pkl", "wb") as f:
        pickle.dump(final_model, f)
    # The feature list lives inside the fitted pipeline's ColumnTransformer;
    # model_registry.model_feature_columns() reads it back at predict/evaluate
    # time (through PositionModel.named_steps), so it isn't persisted separately.

    _save_metrics(models_dir, "position", {
        "mae": round(fc_mae, 4),                 # headline = forward-chain holdout
        "r2": round(fc_r2, 4),
        "forward_chain_mae": round(fc_mae, 4),
        "forward_chain_r2": round(fc_r2, 4),
        "forward_chain_spearman": round(fc_spearman, 4),
        "forward_chain_holdout_rounds": held_rounds,
        "forward_chain_winner_logloss": round(sharp["winner_logloss"], 4),
        "forward_chain_podium_brier": round(sharp["podium_brier"], 4),
        "forward_chain_points_brier": round(sharp["points_brier"], 4),
        "cv_mae_mean": round(-cv_mae.mean(), 4),
        "cv_mae_std": round(cv_mae.std(), 4),
        "cv_r2_mean": round(cv_r2.mean(), 4),
        "cv_r2_std": round(cv_r2.std(), 4),
        "cv": f"GroupKFold({n_splits}, by race)",
        "position_target": target_mode,
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


def build_position_ranker_pipeline(model_params, numeric: list[str], categorical: list[str]) -> Pipeline:
    """D2 — a learning-to-rank head alongside the D1 regressor. Same
    preprocessing, ``rank:ndcg`` objective instead of squared-error."""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("ranker", xgb.XGBRanker(
            objective="rank:ndcg",
            n_estimators=model_params.n_estimators,
            learning_rate=model_params.learning_rate,
            max_depth=model_params.max_depth,
            random_state=model_params.random_state,
        )),
    ])


def _ranker_group_sizes(data: pd.DataFrame, idx) -> tuple[pd.DataFrame, np.ndarray]:
    """XGBRanker requires rows of the same group (race) contiguous, in the
    same order as the ``group`` sizes array. Sort ``data.loc[idx]`` by
    ``race_seq`` and return it alongside that ordering's per-race group
    sizes."""
    d = data.loc[idx].sort_values("race_seq")
    sizes = d.groupby("race_seq", sort=True).size().to_numpy()
    return d, sizes


def blend_regressor_and_ranker(
    position_estimates, ranker_scores, grid_size: int | None = None
) -> np.ndarray:
    """D2's rank-average blend: convert the D1 regressor's Position estimates
    and the D2 ranker's relevance scores (higher = better, opposite polarity
    of Position) to WITHIN-RACE ranks and average them. Returns a blended
    score — lower is still better, sortable exactly like a Position estimate
    — so callers that already sort ascending need no other change.

    Both inputs are 1-D arrays over the SAME race (one row per driver);
    blending across multiple races at once would rank drivers from different
    races against each other, which is meaningless — callers must call this
    once per race.
    """
    pos = np.asarray(position_estimates, dtype=float)
    rank = np.asarray(ranker_scores, dtype=float)
    pos_rank = pd.Series(pos).rank(method="average")
    # higher ranker score = better = should get the LOW rank number, hence
    # ascending=False before .rank() (equivalently: rank the negative score)
    rank_rank = pd.Series(-rank).rank(method="average")
    return ((pos_rank + rank_rank) / 2.0).to_numpy()


def train_position_ranker_model(config: Config) -> None:
    """D2 — trains the learning-to-rank head as a second, complementary model
    to :func:`train_position_model`. Relevance = ``grid_size - Position``
    (higher = better finish); CV scorer is Spearman rank correlation between
    the ranker's score and true finishing order, per held-out race."""
    models_dir = config.paths.models_dir
    models_dir.mkdir(exist_ok=True)
    p = config.models.position
    ft = config.features
    grid_size = config.constants.grid_size

    prepared = prepare_position_data(config)
    data = prepared["data"]
    numeric, categorical, features = prepared["numeric"], prepared["categorical"], prepared["features"]

    train_idx, test_idx = _forward_chain_split(data, ft.holdout_rounds)

    d_train, group_sizes = _ranker_group_sizes(data, train_idx)
    y_train = (grid_size - d_train["Position"]).round().astype(int).clip(lower=0)
    pipeline = build_position_ranker_pipeline(p, numeric, categorical)
    pipeline.fit(d_train[features], y_train, ranker__group=group_sizes)

    # --- headline: per-race Spearman on the held-out rounds ------------------ #
    d_test, _ = _ranker_group_sizes(data, test_idx)
    scores = pipeline.predict(d_test[features])
    d_test = d_test.assign(_score=scores)
    per_race_spearman = [
        float(spearmanr(-g["_score"], g["Position"]).correlation)
        for _, g in d_test.groupby("race_seq") if len(g) >= 3
    ]
    fc_spearman = float(np.nanmean(per_race_spearman)) if per_race_spearman else float("nan")
    logger.info(
        "Position ranker forward-chain holdout — mean per-race Spearman: %.3f (%d races)",
        fc_spearman, len(per_race_spearman),
    )

    # --- final model on all rows ------------------------------------------- #
    d_all, group_sizes_all = _ranker_group_sizes(data, data.index)
    y_all = (grid_size - d_all["Position"]).round().astype(int).clip(lower=0)
    pipeline = build_position_ranker_pipeline(p, numeric, categorical)
    pipeline.fit(d_all[features], y_all, ranker__group=group_sizes_all)
    with open(models_dir / "race_ranking_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    _save_metrics(models_dir, "position_ranker", {
        "forward_chain_spearman": round(fc_spearman, 4) if not np.isnan(fc_spearman) else None,
        "forward_chain_holdout_races": len(per_race_spearman),
        "registry_version": fr.REGISTRY_VERSION,
        "features_used": features,
        "objective": "rank:ndcg",
    })
    logger.info("Saved race_ranking_pipeline.pkl  (%d features)", len(features))


def train_dnf_model(config: Config) -> None:
    """E1 — gradient-boosted DNF classifier using the now-correct reliability
    features (post-A1) + circuit_sc_probability + prior-season DNF rate +
    team/driver incident history — all already in the registry. Feeds
    :func:`pipeline.simulate.simulate_race`'s per-driver P(DNF) sampling."""
    from sklearn.model_selection import StratifiedGroupKFold

    models_dir = config.paths.models_dir
    models_dir.mkdir(exist_ok=True)
    p = config.models.position  # reuse the position model's tree-depth/lr defaults
    ft = config.features

    prepared = prepare_dnf_data(config)
    data, numeric, categorical, features = (
        prepared["data"], prepared["numeric"], prepared["categorical"], prepared["features"]
    )

    X, y, groups = data[features], data["is_dnf"], data["race_seq"]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb.XGBClassifier(
            n_estimators=p.n_estimators, learning_rate=p.learning_rate,
            max_depth=min(p.max_depth, 4), random_state=p.random_state,
            eval_metric="logloss",
        )),
    ])

    n_splits = min(ft.cv_splits, int(groups.nunique()), int(y.value_counts().min()) or 1)
    n_splits = max(n_splits, 2)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=p.random_state)
    cv_auc = cross_val_score(pipeline, X, y, cv=sgkf, groups=groups, scoring="roc_auc")
    cv_brier = -cross_val_score(pipeline, X, y, cv=sgkf, groups=groups, scoring="neg_brier_score")
    logger.info(
        "DNF StratifiedGroupKFold(%d) — ROC-AUC: %.3f ± %.3f  Brier: %.3f ± %.3f",
        n_splits, cv_auc.mean(), cv_auc.std(), cv_brier.mean(), cv_brier.std(),
    )

    pipeline.fit(X, y)
    with open(models_dir / "dnf_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    _save_metrics(models_dir, "dnf", {
        "cv_auc_mean": round(float(cv_auc.mean()), 4),
        "cv_auc_std": round(float(cv_auc.std()), 4),
        "cv_brier_mean": round(float(cv_brier.mean()), 4),
        "cv_brier_std": round(float(cv_brier.std()), 4),
        "cv": f"StratifiedGroupKFold({n_splits})",
        "positive_rate": round(float(y.mean()), 4),
        "registry_version": fr.REGISTRY_VERSION,
        "features_used": features,
    })
    logger.info("Saved dnf_pipeline.pkl  (%d features, positive rate %.1f%%)", len(features), 100 * y.mean())


def run_training(config: Config | None = None, models: list[str] | None = None) -> None:
    if config is None:
        config = get_config()
    if models is None or "all" in models:
        models = list(MODEL_NAMES)

    dispatchers = {
        "laptime": train_laptime_model,
        "racewin": train_racewin_model,
        "position": train_position_model,
        "position_ranker": train_position_ranker_model,
        "dnf": train_dnf_model,
    }

    for name in models:
        if name not in dispatchers:
            logger.error(
                "Unknown model '%s'. Choose from: %s, all.", name, ", ".join(dispatchers),
            )
            continue
        logger.info("--- Training %s model ---", name)
        dispatchers[name](config)
    logger.info("Training complete.")
