"""Monte-Carlo race simulation (E2, docs/prediction-improvement-plan.md).

Turns the point-estimate finishing order (D1 regressor, optionally D2-blended)
into per-driver probabilities: P(win), P(podium), P(points), E[position], and
a 10-90 band. Inputs are all already computed elsewhere, so this stays a pure
aggregation step:

* a point-estimate predicted position per driver — the "skill" signal
* P(DNF) per driver from the E1 classifier (falls back to a neutral rate if
  that model isn't trained/loaded)
* a residual noise scale calibrated from the position model's own
  forward-chain holdout MAE (``models/metrics.json``), so trial-to-trial
  variance reflects how wrong the model actually is, not an arbitrary
  constant.

    from pipeline.simulate import simulate_race_df
    probs = simulate_race_df(pred_df, metrics, dnf_model=bundle.dnf_model)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_N_TRIALS = 5000
DEFAULT_DNF_PROBABILITY = 0.15  # neutral fallback ~ observed season DNF rate
_DNF_SCORE_PENALTY = 1000.0     # sorts a DNF after every classified finisher


@dataclass
class DriverSimResult:
    driver: str
    win_probability: float
    podium_probability: float
    points_probability: float
    expected_position: float
    p10: float
    p90: float
    dnf_probability: float


def residual_std_from_metrics(metrics: dict, default: float = 2.5) -> float:
    """Pull a calibrated noise scale off the position model's own
    forward-chain holdout MAE (``models/metrics.json``) — a model that's
    typically off by ~3 places should produce wider simulated bands than one
    off by ~1. Falls back to ``default`` when metrics aren't available.
    MAE (mean ABSOLUTE error) -> the std of the half-normal with that mean
    is ``mae * sqrt(pi/2)``."""
    mae = (metrics or {}).get("position", {}).get("forward_chain_mae")
    if mae is None:
        return default
    return float(mae) * 1.2533


def simulate_race(
    driver: list[str],
    predicted_position,
    dnf_probability=None,
    residual_std: float = 2.5,
    n_trials: int = DEFAULT_N_TRIALS,
    points_paying_positions: int = 10,
    random_state: int = 42,
) -> list[DriverSimResult]:
    """Monte-Carlo the race ``n_trials`` times.

    Each trial: sample a DNF per driver from ``dnf_probability``; draw a
    noisy finishing SCORE (``predicted_position + Normal(0, residual_std)``)
    for every driver; a sampled DNF is pushed to the back of the field
    (order among simultaneous DNFs is arbitrary — the model has no signal on
    *when* each one retires). Resolve the trial's order by sorting scores;
    aggregate across trials into win/podium/points probabilities, expected
    position, and a 10-90 band.
    """
    n = len(driver)
    if n == 0:
        return []
    pred = np.asarray(predicted_position, dtype=float)
    if pred.shape != (n,):
        raise ValueError(f"predicted_position must have {n} entries, got {pred.shape}")

    dnf_p = (
        np.clip(np.asarray(dnf_probability, dtype=float), 0.0, 0.95)
        if dnf_probability is not None
        else np.full(n, DEFAULT_DNF_PROBABILITY)
    )
    if dnf_p.shape != (n,):
        raise ValueError(f"dnf_probability must have {n} entries, got {dnf_p.shape}")

    rng = np.random.default_rng(random_state)
    noise = rng.normal(0.0, residual_std, size=(n_trials, n))
    dnf_draw = rng.random((n_trials, n)) < dnf_p  # broadcasts dnf_p over trials

    score = pred[None, :] + noise
    score = np.where(dnf_draw, score + _DNF_SCORE_PENALTY, score)

    order = np.argsort(score, axis=1, kind="stable")
    positions = np.empty_like(order)
    trial_idx = np.arange(n_trials)[:, None]
    positions[trial_idx, order] = np.arange(1, n + 1)[None, :]

    results = []
    for i, drv in enumerate(driver):
        p = positions[:, i]
        results.append(DriverSimResult(
            driver=str(drv),
            win_probability=float(np.mean(p == 1)),
            podium_probability=float(np.mean(p <= 3)),
            points_probability=float(np.mean(p <= points_paying_positions)),
            expected_position=float(np.mean(p)),
            p10=float(np.percentile(p, 10)),
            p90=float(np.percentile(p, 90)),
            dnf_probability=float(dnf_p[i]),
        ))
    return results


def _dnf_probabilities(df: pd.DataFrame, dnf_model) -> np.ndarray | None:
    if dnf_model is None:
        return None
    try:
        from pipeline.model_registry import model_feature_columns

        cols = model_feature_columns(dnf_model)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            logger.warning("DNF model missing feature(s) %s — using neutral P(DNF).", missing)
            return None
        return dnf_model.predict_proba(df[cols])[:, 1]
    except Exception as e:  # noqa: BLE001 — best-effort; simulation still runs
        logger.warning("DNF model prediction failed (%s) — using neutral P(DNF).", e)
        return None


def simulate_race_df(
    df: pd.DataFrame,
    metrics: dict | None = None,
    dnf_model=None,
    n_trials: int = DEFAULT_N_TRIALS,
    random_state: int = 42,
) -> pd.DataFrame:
    """Convenience wrapper over :func:`simulate_race`. ``df`` needs a
    ``Driver`` and a ``PredictedPosition`` column (plus whatever feature
    columns ``dnf_model`` expects, if one is given — typically the same
    frame ``predict.py`` already built). Returns one row per driver, ready to
    merge into a ``PredictionResult`` / API response."""
    driver = df["Driver"].tolist()
    pred = df["PredictedPosition"].to_numpy(dtype=float)
    dnf_p = _dnf_probabilities(df, dnf_model)
    residual_std = residual_std_from_metrics(metrics)

    sims = simulate_race(
        driver, pred, dnf_p, residual_std=residual_std,
        n_trials=n_trials, random_state=random_state,
    )
    return pd.DataFrame([{
        "Driver": s.driver,
        "win_probability": round(s.win_probability, 4),
        "podium_probability": round(s.podium_probability, 4),
        "points_probability": round(s.points_probability, 4),
        "expected_position": round(s.expected_position, 2),
        "p10": s.p10,
        "p90": s.p90,
        "dnf_probability": round(s.dnf_probability, 4),
    } for s in sims])
