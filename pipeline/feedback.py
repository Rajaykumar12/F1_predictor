"""Model feedback loop: prediction logging, post-race scoring, rolling
drift scorecard, and per-driver bias correction.

All functions here are pure or narrow-IO so they unit-test with synthetic
frames (see tests/test_feedback.py). Orchestration that ties them to the
pipeline lives in pipeline/orchestrate.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Scoring — lifted from scripts/backtest_monza.py:score()
# --------------------------------------------------------------------------- #
def _overlap(a, b) -> int:
    return len(set(a) & set(b))


# --------------------------------------------------------------------------- #
# Sharp-end metrics — winner_logloss / podium_brier
#
# Position MAE ~2.2 hides that the model is usually wrong about who's on the
# podium (see docs/prediction-improvement-plan.md A3). Neither the point
# forecast nor a calibrated probability model exists yet (that's Phase E's
# Monte-Carlo simulation), so these read a proxy probability off the
# continuous predicted position: drivers closer to the front of the predicted
# order get more win/podium mass, via a softmax / logistic transform scaled by
# the race's own predicted-position spread. This is NOT a calibrated
# probability — it exists so a race with a wrong winner scores worse than one
# with a merely-reordered podium, which MAE and Spearman alone don't capture.
# --------------------------------------------------------------------------- #
def _pseudo_rank_probabilities(pred_positions: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    pred_positions = pd.Series(pred_positions).astype(float)
    spread = pred_positions.std()
    scale = float(spread) if spread and spread > 1e-6 else 1.0

    z = (pred_positions - pred_positions.mean()) / scale
    win_score = np.exp(-z)
    p_win = (win_score / win_score.sum()).to_numpy()

    p_podium = (1.0 / (1.0 + np.exp((pred_positions.to_numpy() - 3.5) / scale)))
    p_points = (1.0 / (1.0 + np.exp((pred_positions.to_numpy() - 10.5) / scale)))
    return p_win, p_podium, p_points


def _winner_logloss(p_win: np.ndarray, actual_positions: pd.Series, eps: float = 1e-9) -> float:
    actual_positions = pd.Series(actual_positions).astype(float)
    winner_mask = (actual_positions == actual_positions.min()).to_numpy()
    if not winner_mask.any():
        return float("nan")
    p = float(np.asarray(p_win)[winner_mask].sum())
    return float(-np.log(np.clip(p, eps, 1.0)))


def _brier(p: np.ndarray, actual_positions: pd.Series, threshold: float) -> float:
    actual = (pd.Series(actual_positions).astype(float) <= threshold).astype(float).to_numpy()
    return float(np.mean((np.asarray(p) - actual) ** 2))


def sharp_end_metrics_by_race(
    pred_positions, actual_positions, race
) -> dict:
    """Average ``winner_logloss`` / ``podium_brier`` / ``points_brier`` across
    races — the metric ``pipeline/train.py``'s forward-chain holdout and
    ``scripts/rolling_backtest.py`` report alongside MAE."""
    df = pd.DataFrame({
        "pred": np.asarray(pred_positions, dtype=float),
        "actual": np.asarray(actual_positions, dtype=float),
        "race": np.asarray(race),
    })
    logloss, podium_brier, points_brier = [], [], []
    for _, g in df.groupby("race"):
        p_win, p_podium, p_points = _pseudo_rank_probabilities(g["pred"])
        logloss.append(_winner_logloss(p_win, g["actual"]))
        podium_brier.append(_brier(p_podium, g["actual"], 3))
        points_brier.append(_brier(p_points, g["actual"], 10))
    return {
        "winner_logloss": float(np.nanmean(logloss)) if logloss else float("nan"),
        "podium_brier": float(np.nanmean(podium_brier)) if podium_brier else float("nan"),
        "points_brier": float(np.nanmean(points_brier)) if points_brier else float("nan"),
    }


def score_prediction(pred_df: pd.DataFrame, actual_df: pd.DataFrame) -> dict:
    """Score a predicted finishing order against the actual result.

    pred_df   : columns PredRank (int), Driver, PredictedPosition (float)
    actual_df : columns Driver, Position   (from f1_results_simple.csv, Race == N)
    """
    actual = actual_df.copy()
    actual["Position"] = pd.to_numeric(actual["Position"], errors="coerce")
    actual = actual.dropna(subset=["Position"])
    actual_order = actual.sort_values("Position")[["Driver", "Position"]]

    merged = pred_df.merge(actual_order, on="Driver", how="inner")
    merged["AbsError"] = (merged["PredRank"] - merged["Position"]).abs()

    pred_order = pred_df.sort_values("PredictedPosition")["Driver"].tolist()
    act_order = actual_order["Driver"].tolist()

    winner_correct = bool(pred_order and act_order and pred_order[0] == act_order[0])

    pred_podium, act_podium = pred_order[:3], act_order[:3]
    podium_overlap = _overlap(pred_podium, act_podium)
    podium_exact = int(sum(p == a for p, a in zip(pred_podium, act_podium)))

    top5 = _overlap(pred_order[:5], act_order[:5])
    top10 = _overlap(pred_order[:10], act_order[:10])

    if len(merged) >= 3:
        rho, _ = spearmanr(merged["PredRank"], merged["Position"])
        mae = float(merged["AbsError"].mean())
        rmse = float(np.sqrt((merged["AbsError"] ** 2).mean()))
        rho = float(rho)
    else:
        rho = mae = rmse = float("nan")

    unmatched = sorted(set(pred_df["Driver"]) - set(actual_order["Driver"]))

    if len(merged) >= 1:
        p_win, p_podium, p_points = _pseudo_rank_probabilities(merged["PredictedPosition"])
        winner_logloss = _winner_logloss(p_win, merged["Position"])
        podium_brier = _brier(p_podium, merged["Position"], 3)
        points_brier = _brier(p_points, merged["Position"], 10)
    else:
        winner_logloss = podium_brier = points_brier = float("nan")

    return {
        "winner_correct": winner_correct,
        "podium_overlap": podium_overlap,
        "podium_exact": podium_exact,
        "top5": top5,
        "top10": top10,
        "spearman": rho,
        "position_mae": mae,
        "position_rmse": rmse,
        "winner_logloss": winner_logloss,
        "podium_brier": podium_brier,
        "points_brier": points_brier,
        "n_drivers": int(len(merged)),
        "unmatched": unmatched,
    }


def per_driver_errors(pred_df: pd.DataFrame, actual_df: pd.DataFrame) -> list[dict]:
    """Signed rank error per driver. Positive = model placed them too low."""
    actual = actual_df.copy()
    actual["Position"] = pd.to_numeric(actual["Position"], errors="coerce")
    actual = actual.dropna(subset=["Position"])[["Driver", "Position"]]
    merged = pred_df.merge(actual, on="Driver", how="inner")
    out = []
    for _, r in merged.iterrows():
        out.append({
            "driver": str(r["Driver"]),
            "pred_rank": int(r["PredRank"]),
            "actual_position": int(r["Position"]),
            "signed_error": int(r["PredRank"] - r["Position"]),
        })
    return out


# --------------------------------------------------------------------------- #
# Stores
# --------------------------------------------------------------------------- #
def load_history(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not read score history %s: %s", path, e)
        return []


def append_history(path, record: dict) -> None:
    """Append a score record. If one already exists for the same (season, round),
    it is replaced in place so re-scoring a race doesn't double-count it."""
    path = Path(path)
    history = load_history(path)
    key = (record.get("season"), record.get("round"))
    if key != (None, None):
        history = [h for h in history if (h.get("season"), h.get("round")) != key]
    history.append(record)
    history.sort(key=lambda h: (h.get("season", 0), h.get("round", 0)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2))


def prediction_log_path(cfg, season: int, round_no: int) -> Path:
    return cfg.feedback.prediction_log_dir / f"{season}_r{int(round_no):02d}.json"


def _forecast_to_dict(f) -> dict:
    return {
        "pred_rank": f.pred_rank,
        "driver": f.driver,
        "team": f.team,
        "predicted_position": f.predicted_position,
        "raw_predicted_position": f.raw_predicted_position,
        "confidence": f.confidence,
        "recent_form": f.recent_form,
    }


def _model_trained_at(cfg) -> str | None:
    path = cfg.paths.models_dir / "metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("position", {}).get("trained_at")
    except Exception:  # noqa: BLE001
        return None


def write_prediction_log(cfg, result, round_no: int) -> Path:
    """Persist a PredictionResult so it can be scored after the race."""
    season = cfg.pipeline.season
    path = prediction_log_path(cfg, season, round_no)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "season": season,
        "round": int(round_no),
        "race_label": result.race_label,
        "lookback": result.lookback,
        "as_of_round": result.as_of_round,
        "predicted_at": result.prediction_date,
        "model_trained_at": _model_trained_at(cfg),
        "model_r2": result.model_r2,
        "using_real_qualifying": result.using_real_qualifying,
        "bias_applied": result.bias_applied or None,
        "forecasts": [_forecast_to_dict(f) for f in result.forecasts],
        "scored": None,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_prediction_log(cfg, round_no: int) -> dict | None:
    path = prediction_log_path(cfg, cfg.pipeline.season, round_no)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not read prediction log %s: %s", path, e)
        return None


def update_prediction_log_scored(cfg, round_no: int, scored: dict) -> None:
    log = load_prediction_log(cfg, round_no)
    if log is None:
        raise FileNotFoundError(f"No prediction log for round {round_no}")
    log["scored"] = scored
    path = prediction_log_path(cfg, cfg.pipeline.season, round_no)
    path.write_text(json.dumps(log, indent=2))


def list_prediction_logs(cfg) -> list[dict]:
    d = cfg.feedback.prediction_log_dir
    if not d.exists():
        return []
    logs = []
    for p in sorted(d.glob(f"{cfg.pipeline.season}_r*.json")):
        try:
            logs.append(json.loads(p.read_text()))
        except Exception as e:  # noqa: BLE001
            logger.warning("Skipping unreadable prediction log %s: %s", p, e)
    logs.sort(key=lambda x: x.get("round", 0))
    return logs


# --------------------------------------------------------------------------- #
# Rolling scorecard + drift
# --------------------------------------------------------------------------- #
def _nanmean(values) -> float | None:
    """Mean of the non-None values, skipping NaNs too. ``None`` (not NaN) when
    nothing is left — NaN isn't valid JSON and every one of these fields is
    exposed verbatim over the API (``/health``, ``/score-history``)."""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return None
    return float(np.nanmean(vals))


def rolling_scorecard(history: list[dict], window: int) -> dict:
    recent = history[-window:] if window > 0 else list(history)
    if not recent:
        return {"races": 0}
    return {
        "races": len(recent),
        "winner_hit_rate": _nanmean([1.0 if r.get("winner_correct") else 0.0 for r in recent]),
        "podium_overlap_avg": _nanmean([r.get("podium_overlap") for r in recent]),
        "top5_avg": _nanmean([r.get("top5") for r in recent]),
        "top10_avg": _nanmean([r.get("top10") for r in recent]),
        "spearman_avg": _nanmean([r.get("spearman") for r in recent]),
        "position_mae_avg": _nanmean([r.get("position_mae") for r in recent]),
        "position_rmse_avg": _nanmean([r.get("position_rmse") for r in recent]),
        "winner_logloss_avg": _nanmean([r.get("winner_logloss") for r in recent]),
        "podium_brier_avg": _nanmean([r.get("podium_brier") for r in recent]),
        "points_brier_avg": _nanmean([r.get("points_brier") for r in recent]),
    }


def should_retrain(history: list[dict], fb) -> tuple[bool, list[str]]:
    if not fb.enabled:
        return False, ["feedback disabled"]
    if len(history) < fb.min_scored_races:
        return False, [f"only {len(history)} scored race(s); need {fb.min_scored_races}"]

    card = rolling_scorecard(history, fb.window_races)
    reasons: list[str] = []

    mae = card.get("position_mae_avg")
    if mae is not None and not np.isnan(mae) and mae > fb.max_position_mae:
        reasons.append(f"rolling position MAE {mae:.2f} > {fb.max_position_mae}")

    rho = card.get("spearman_avg")
    if rho is not None and not np.isnan(rho) and rho < fb.min_spearman:
        reasons.append(f"rolling Spearman {rho:.2f} < {fb.min_spearman}")

    whr = card.get("winner_hit_rate")
    if whr is not None and not np.isnan(whr) and whr < fb.min_winner_hit_rate:
        reasons.append(f"rolling winner hit-rate {whr:.2f} < {fb.min_winner_hit_rate}")

    return (len(reasons) > 0), reasons


# --------------------------------------------------------------------------- #
# Per-driver bias correction (exp-weighted signed error)
# --------------------------------------------------------------------------- #
def driver_bias(scored_rounds: list[dict], halflife: float, max_abs: float) -> dict[str, float]:
    """Exp-weighted mean signed rank error per driver, clipped to +/- max_abs.

    scored_rounds : chronological list of {"round": int,
                    "per_driver": [{"driver", "signed_error"}, ...]}
    weight for a round k positions before the latest = 0.5 ** (k / halflife)
    """
    if not scored_rounds:
        return {}
    ordered = sorted(scored_rounds, key=lambda r: r.get("round", 0))
    latest_idx = len(ordered) - 1

    acc: dict[str, list[tuple[float, float]]] = {}
    for i, rnd in enumerate(ordered):
        k = latest_idx - i
        w = 0.5 ** (k / halflife)
        for pd_row in rnd.get("per_driver", []):
            drv = pd_row["driver"]
            acc.setdefault(drv, []).append((w, float(pd_row["signed_error"])))

    bias: dict[str, float] = {}
    for drv, pairs in acc.items():
        wsum = sum(w for w, _ in pairs)
        if wsum <= 0:
            continue
        est = sum(w * e for w, e in pairs) / wsum
        bias[drv] = float(np.clip(est, -max_abs, max_abs))
    return bias


def apply_bias_correction(forecasts: list, bias: dict[str, float]) -> list:
    """Return a re-ranked copy of forecasts with bias subtracted from raw positions."""
    if not bias:
        return list(forecasts)
    adjusted = []
    for f in forecasts:
        delta = bias.get(f.driver, 0.0)
        new_pos = max(1.0, round(f.raw_predicted_position - delta, 2))
        adjusted.append(replace(f, predicted_position=new_pos))
    adjusted.sort(key=lambda f: f.predicted_position)
    for rank, f in enumerate(adjusted, start=1):
        f.pred_rank = rank
    return adjusted


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")
