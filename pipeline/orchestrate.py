"""Framework-agnostic orchestration of multi-step pipeline actions.

These functions take a ``Config`` and return plain dicts. They are shared by the
FastAPI background jobs (``app.py`` ``_job_*``) and the terminal CLI (``main.py``)
so there is one implementation of "train + report", "run the whole pipeline",
"score a race", etc.
"""

from __future__ import annotations

import logging

import pandas as pd

from pipeline import feedback
from pipeline.config_loader import Config
from pipeline.evaluate import evaluate_laptime, evaluate_position
from pipeline.fetch import fetch_upcoming_qualifying, run_fetch
from pipeline.clean import run_cleaning
from pipeline.features import run_feature_engineering
from pipeline.model_registry import MODEL_NAMES, load_metrics
from pipeline.train import run_training

logger = logging.getLogger(__name__)


def train_models(config: Config, model: str) -> dict:
    run_training(config, models=[model])
    metrics = load_metrics(config)
    trained = list(MODEL_NAMES) if model == "all" else [model]
    return {"trained": trained, "metrics": {k: metrics.get(k) for k in trained}}


def evaluate_laptime_job(config: Config, race) -> dict:
    results = evaluate_laptime(config, race_round=race)
    abs_error = (results["Predicted_seconds"] - results["Actual_seconds"]).abs()
    return {
        "race_round": race,
        "laps_evaluated": int(len(results)),
        "mae": round(float(abs_error.mean()), 3),
    }


def evaluate_position_job(config: Config, race) -> dict:
    out = evaluate_position(config, race_round=race)
    return {
        "race_round": race,
        "drivers_evaluated": int(len(out)),
        "position_mae": round(float(out["AbsError"].mean()), 3),
    }


def run_all(config: Config) -> dict:
    run_fetch(config)
    run_cleaning(config)
    run_feature_engineering(config)
    run_training(config)
    qualifying_fetched = True
    try:
        fetch_upcoming_qualifying(config)
    except Exception as e:  # noqa: BLE001
        logger.warning("run-all: fetch-qualifying skipped: %s", e)
        qualifying_fetched = False
    return {"trained": list(MODEL_NAMES), "qualifying_fetched": qualifying_fetched}


def _actual_results(config: Config, race_round: int, fetch_if_missing: bool) -> pd.DataFrame:
    path = config.paths.data_dir / "f1_results_simple.csv"
    if path.exists():
        df = pd.read_csv(path)
        got = df[df["Race"] == race_round]
        if not got.empty and got["Position"].notna().any():
            return got
    if fetch_if_missing:
        logger.info("Round %d results not on disk — running fetch.", race_round)
        run_fetch(config)
        df = pd.read_csv(path)
        return df[df["Race"] == race_round]
    return pd.DataFrame(columns=["Driver", "Position"])


def score_race(config: Config, race_round: int, *, fetch_if_missing: bool = True) -> dict:
    log = feedback.load_prediction_log(config, race_round)
    if log is None:
        raise FileNotFoundError(
            f"No prediction log for round {race_round}. Run: python main.py predict-race --round {race_round} --save"
        )

    pred_df = pd.DataFrame(
        [
            {"PredRank": f["pred_rank"], "Driver": f["driver"], "PredictedPosition": f["predicted_position"]}
            for f in log["forecasts"]
        ]
    )
    actual = _actual_results(config, race_round, fetch_if_missing)
    if actual.empty or actual["Position"].isna().all():
        raise RuntimeError(
            f"No actual results available for round {race_round} yet (FastF1 may not have published)."
        )

    metrics = feedback.score_prediction(pred_df, actual[["Driver", "Position"]])
    pd_errors = feedback.per_driver_errors(pred_df, actual[["Driver", "Position"]])

    scored_at = feedback.now_str()
    feedback.update_prediction_log_scored(
        config, race_round,
        {"scored_at": scored_at, "metrics": metrics, "per_driver": pd_errors},
    )

    history = [
        h for h in feedback.load_history(config.feedback.score_history_path)
        if (h.get("season"), h.get("round")) != (config.pipeline.season, int(race_round))
    ]
    record = {
        "season": config.pipeline.season,
        "round": int(race_round),
        "scored_at": scored_at,
        "model_trained_at": log.get("model_trained_at"),
        "winner_correct": metrics["winner_correct"],
        "podium_overlap": metrics["podium_overlap"],
        "podium_exact": metrics["podium_exact"],
        "top5": metrics["top5"],
        "top10": metrics["top10"],
        "spearman": metrics["spearman"],
        "position_mae": metrics["position_mae"],
        "position_rmse": metrics["position_rmse"],
        "winner_logloss": metrics["winner_logloss"],
        "podium_brier": metrics["podium_brier"],
        "points_brier": metrics["points_brier"],
        "n_drivers": metrics["n_drivers"],
    }
    history_plus = history + [record]
    retrain, reasons = feedback.should_retrain(history_plus, config.feedback)
    record["retrain_triggered"] = bool(retrain and config.feedback.auto_retrain)
    record["retrain_reasons"] = reasons
    feedback.append_history(config.feedback.score_history_path, record)

    return {
        "race_round": int(race_round),
        "metrics": metrics,
        "rolling_scorecard": feedback.rolling_scorecard(history_plus, config.feedback.window_races),
        "drift": {
            "retrain_recommended": bool(retrain),
            "reasons": reasons,
            "auto_retrain": bool(config.feedback.auto_retrain),
        },
    }


def compute_bias(config: Config) -> dict:
    fb = config.feedback
    if not fb.bias_correction_enabled:
        return {}
    scored = [
        {"round": l["round"], "per_driver": l["scored"]["per_driver"]}
        for l in feedback.list_prediction_logs(config)
        if l.get("scored") and l["scored"].get("per_driver")
    ]
    if len(scored) < fb.min_scored_races:
        return {}
    return feedback.driver_bias(scored, fb.bias_halflife_races, fb.bias_max_abs)
