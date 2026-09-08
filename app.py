from __future__ import annotations

import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import feedback, orchestrate
from pipeline.clean import run_cleaning
from pipeline.config_loader import get_config
from pipeline.features import run_feature_engineering
from pipeline.fetch import fetch_upcoming_qualifying, get_completed_race_rounds, run_fetch
from pipeline.model_registry import _load_pickle, load_bundle, load_metrics  # noqa: F401 (re-export)
from pipeline.predict import (
    ModelNotLoadedError,
    PredictionDataMissingError,
    predict_race,
)
from pipeline.train import run_training

_cfg = get_config()

logging.basicConfig(
    level=getattr(logging, _cfg.logging.level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_RACES = _cfg.pipeline.lookback_races
MIN_LOOKBACK = _cfg.pipeline.min_lookback
MAX_LOOKBACK = _cfg.pipeline.max_lookback

app = FastAPI(
    title="F1 Race Winner Prediction API",
    description="API for predicting F1 race winners",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cfg.api.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# _load_pickle is imported from pipeline.model_registry (kept importable as app._load_pickle).
_b = load_bundle(_cfg)
win_model = _b.win_model
laptime_pipeline = _b.laptime_pipeline
race_model = _b.race_model
_laptime_features = _b.laptime_features
_metrics: dict = _b.metrics


def _reload_metrics() -> None:
    global _metrics
    _metrics = load_metrics(_cfg)


def _reload_models() -> None:
    """Re-read model pickles + metrics from disk (e.g. after a /pipeline/train job)."""
    global win_model, laptime_pipeline, race_model, _laptime_features
    b = load_bundle(_cfg)
    win_model = b.win_model
    laptime_pipeline = b.laptime_pipeline
    race_model = b.race_model
    _laptime_features = b.laptime_features
    _reload_metrics()
    logger.info("Models reloaded from disk.")


# ---------------------------------------------------------------------------
# Background job runner for pipeline stages (fetch/clean/features/train/...)
# ---------------------------------------------------------------------------

_executor = ThreadPoolExecutor(max_workers=1)  # single worker: pipeline jobs write shared files
_jobs: dict[str, dict] = {}
_JOB_HISTORY_LIMIT = 50


def _active_job_id() -> Optional[str]:
    for job_id, job in _jobs.items():
        if job["status"] in ("queued", "running"):
            return job_id
    return None


def _run_job(job_id: str, fn, *args, **kwargs) -> None:
    job = _jobs[job_id]
    job["status"] = "running"
    job["started_at"] = datetime.now().isoformat()
    try:
        result = fn(*args, **kwargs)
        job["status"] = "success"
        job["result"] = result
    except Exception as e:
        logger.error("Pipeline job %s (%s) failed: %s", job_id, job["kind"], e)
        job["status"] = "failed"
        job["error"] = str(e)
    finally:
        job["finished_at"] = datetime.now().isoformat()


def _submit_job(kind: str, fn, *args, **kwargs) -> str:
    active = _active_job_id()
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=f"A pipeline job is already in progress ({_jobs[active]['kind']}, id={active}). "
                   f"Wait for it to finish or check GET /jobs/{active}.",
        )
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "id": job_id,
        "kind": kind,
        "status": "queued",
        "started_at": None,
        "finished_at": None,
        "error": None,
        "result": None,
    }
    # Trim history so the in-memory dict doesn't grow unbounded
    if len(_jobs) > _JOB_HISTORY_LIMIT:
        oldest = sorted(_jobs.values(), key=lambda j: j["id"])[0]["id"]
        if oldest != job_id:
            del _jobs[oldest]
    _executor.submit(_run_job, job_id, fn, *args, **kwargs)
    return job_id


class JobSubmitted(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    id: str
    kind: str
    status: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict] = None


@app.get("/jobs", response_model=list[JobStatus], tags=["pipeline"])
def list_jobs():
    """List recent pipeline jobs, most recently created first."""
    return list(reversed(list(_jobs.values())))


@app.get("/jobs/{job_id}", response_model=JobStatus, tags=["pipeline"])
def get_job(job_id: str):
    """Poll the status of a pipeline job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


class RaceInput(BaseModel):
    Team: str
    Position: int = Field(..., ge=1, le=26)
    GridPosition: int = Field(..., ge=1, le=26)
    driver_win_rate: float = Field(..., ge=0.0, le=100.0)
    team_reliability: float = Field(..., ge=0.0, le=100.0)
    BestQualifyingTime: Optional[float] = None
    GapToPole: Optional[float] = Field(default=None, ge=0.0)
    QualifyingPerformance: Optional[float] = Field(default=None, ge=0.0, le=100.0)


@app.post("/predict")
def predict(race: RaceInput):
    """Predict race winner probability."""
    if win_model is None:
        raise HTTPException(status_code=503, detail="Race win model not loaded. Run: POST /pipeline/train {'model': 'racewin'}")
    try:
        input_data = race.model_dump()
        df = pd.DataFrame([input_data])
        if df.isnull().any().any():
            raise HTTPException(
                status_code=422,
                detail="Input contains missing values. All features including qualifying data are required.",
            )

        prediction = win_model.predict(df)[0]
        probability = float(win_model.predict_proba(df)[0][1])

        confidence = "high" if probability > 0.7 or probability < 0.3 else "medium"
        logger.info(
            "predict | team=%s grid=%d | will_win=%s prob=%.4f",
            race.Team, race.GridPosition, bool(prediction), probability,
        )
        return {
            "will_win": bool(prediction),
            "win_probability": round(probability, 4),
            "confidence": confidence,
            "features_used": len(input_data),
            "includes_qualifying": True,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


class LapTimeInput(BaseModel):
    Race: str
    Driver: str
    Team: str
    Position: int = Field(..., ge=1, le=26)
    TireCompound: str
    TireAge: int = Field(..., ge=0)
    driver_win_rate: float = Field(..., ge=0.0, le=100.0)
    team_reliability: float = Field(..., ge=0.0, le=100.0)
    TireCompound_encoded: Optional[float] = None
    IsFreshTire: Optional[int] = None
    StintLapNumber: Optional[int] = None
    FuelLoadProxy: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    LapNumber_normalized: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    IsOutlap: Optional[int] = 0
    IsInlap: Optional[int] = 0
    positions_gained: Optional[float] = 0.0
    tire_degradation: Optional[float] = 0.0
    RollingAvgLapTime_3: Optional[float] = None
    RollingAvgLapTime_5: Optional[float] = None
    LapTimeStd_5: Optional[float] = 0.5


_TIRE_COMPOUND_MAP = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4, "WET": 5}
_TIRE_LIFE_MAP = {"SOFT": 40, "MEDIUM": 50, "HARD": 60}
_DEFAULT_LAPTIME = 95.0


@app.post("/predict_laptime")
def predict_laptime(lap: LapTimeInput):
    """Predict lap time (auto-computes derived fields if not provided)."""
    if laptime_pipeline is None:
        raise HTTPException(status_code=503, detail="Lap time model not loaded. Run: POST /pipeline/train {'model': 'laptime'}")
    try:
        d = lap.model_dump()
        compound = lap.TireCompound.upper()
        max_tire_life = _TIRE_LIFE_MAP.get(compound, 50)

        if d["TireCompound_encoded"] is None:
            d["TireCompound_encoded"] = _TIRE_COMPOUND_MAP.get(compound, 2)
        if d["IsFreshTire"] is None:
            d["IsFreshTire"] = 1 if lap.TireAge <= 3 else 0
        if d["StintLapNumber"] is None:
            d["StintLapNumber"] = lap.TireAge
        if d["RollingAvgLapTime_3"] is None:
            d["RollingAvgLapTime_3"] = _DEFAULT_LAPTIME
        if d["RollingAvgLapTime_5"] is None:
            d["RollingAvgLapTime_5"] = _DEFAULT_LAPTIME

        # Only pass the features the model was actually trained on
        trained_features = _laptime_features["features"] if _laptime_features else list(d.keys())
        df_input = pd.DataFrame([{k: d[k] for k in trained_features if k in d}])
        predicted = float(laptime_pipeline.predict(df_input)[0])

        minutes = int(predicted // 60)
        seconds = predicted % 60
        tire_wear_pct = min(100, (lap.TireAge / max_tire_life) * 100)

        logger.info(
            "predict_laptime | driver=%s compound=%s age=%d | predicted=%.3fs",
            lap.Driver, compound, lap.TireAge, predicted,
        )
        return {
            "predicted_laptime_seconds": round(predicted, 3),
            "predicted_laptime_formatted": f"{minutes}:{seconds:06.3f}",
            "tire_compound": lap.TireCompound,
            "tire_age": lap.TireAge,
            "tire_wear_pct": round(tire_wear_pct, 1),
            "is_fresh_tire": bool(d["IsFreshTire"]),
        }
    except Exception as e:
        logger.error("Lap time prediction error: %s", e)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


class DriverPrediction(BaseModel):
    predicted_position: float
    driver: str
    team: str
    confidence: float
    recent_form: dict


class RacePrediction(BaseModel):
    predictions: list[DriverPrediction]
    prediction_date: str
    next_race: str
    model_r2: Optional[float] = None


def _maybe_bias() -> Optional[dict]:
    """Per-driver bias correction, only when enabled in config. Never raises."""
    if not getattr(_cfg.feedback, "bias_correction_enabled", False):
        return None
    try:
        return orchestrate.compute_bias(_cfg) or None
    except Exception as e:  # noqa: BLE001
        logger.warning("bias correction skipped: %s", e)
        return None


def _next_round_guess() -> Optional[int]:
    """Best-effort 'which round is this prediction for' — for the prediction log."""
    try:
        up = _cfg.paths.data_dir / "upcoming_qualifying.csv"
        if up.exists():
            return int(pd.read_csv(up)["Race"].iloc[0])
        done = get_completed_race_rounds(_cfg.pipeline.season)
        return (max(done) + 1) if done else None
    except Exception:  # noqa: BLE001
        return None


@app.get("/predict_next_race", response_model=RacePrediction)
def predict_next_race(
    lookback_races: int = Query(
        default=DEFAULT_LOOKBACK_RACES,
        ge=MIN_LOOKBACK,
        le=MAX_LOOKBACK,
        description="Number of previous races to consider (optimal: 5-8)",
    ),
    save: bool = Query(
        default=False,
        description="Persist this prediction to the prediction log for later scoring.",
    ),
):
    """Predict next race finishing positions for all drivers."""
    try:
        result = predict_race(
            _cfg, race_model, _metrics,
            lookback=lookback_races,
            bias=_maybe_bias(),
        )
    except ModelNotLoadedError:
        raise HTTPException(
            status_code=503,
            detail="Race position model not loaded. Run: POST /pipeline/train {'model': 'position'}",
        )
    except PredictionDataMissingError:
        raise HTTPException(status_code=404, detail="Results data not found. Run: POST /pipeline/features")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.error("Error in predict_next_race: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    if save:
        try:
            rnd = _next_round_guess()
            if rnd is not None:
                feedback.write_prediction_log(_cfg, result, round_no=rnd)
        except Exception as e:  # noqa: BLE001
            logger.warning("prediction log write skipped: %s", e)

    logger.info(
        "predict_next_race | race=%s lookback=%d | %d drivers predicted",
        result.race_label, lookback_races, len(result.forecasts),
    )
    return RacePrediction(
        predictions=[
            DriverPrediction(
                predicted_position=f.predicted_position,
                driver=f.driver,
                team=f.team,
                confidence=f.confidence,
                recent_form=f.recent_form,
            )
            for f in result.forecasts
        ],
        prediction_date=result.prediction_date,
        next_race=result.next_race,
        model_r2=result.model_r2,
    )


@app.get("/health")
def health():
    """Health check endpoint with model and data status."""
    try:
        results_path = _cfg.paths.data_dir / "f1_results_features.csv"
        data_age_hours = None
        if results_path.exists():
            age_seconds = time.time() - os.path.getmtime(results_path)
            data_age_hours = round(age_seconds / 3600, 1)

        freshness_threshold = _cfg.api.data_freshness_hours
        data_stale = data_age_hours is not None and data_age_hours > freshness_threshold

        status = "degraded" if (
            win_model is None or race_model is None or laptime_pipeline is None or data_stale
        ) else "healthy"

        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "models_loaded": {
                "race_winner": win_model is not None,
                "lap_time": laptime_pipeline is not None,
                "race_position": race_model is not None,
            },
            "data_available": {
                "results": results_path.exists(),
                "laps": (_cfg.paths.data_dir / "f1_laps_cleaned.csv").exists(),
                "qualifying": (_cfg.paths.data_dir / "f1_qualifying_cleaned.csv").exists(),
            },
            "data_age_hours": data_age_hours,
            "data_fresh": not data_stale,
            "model_metrics": _metrics or None,
            "feedback": _feedback_health(),
            "config": {
                "season": _cfg.pipeline.season,
                "default_lookback_races": DEFAULT_LOOKBACK_RACES,
                "lookback_range": f"{MIN_LOOKBACK}-{MAX_LOOKBACK}",
            },
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


def _feedback_health():
    """Informational drift block for /health. Never raises; drift does not flip status."""
    try:
        history = feedback.load_history(_cfg.feedback.score_history_path)
        if not history or not _cfg.feedback.enabled:
            return None
        retrain, reasons = feedback.should_retrain(history, _cfg.feedback)
        return {
            "scored_races": len(history),
            "rolling_scorecard": feedback.rolling_scorecard(history, _cfg.feedback.window_races),
            "retrain_recommended": bool(retrain),
            "reasons": reasons,
            "bias_correction_enabled": _cfg.feedback.bias_correction_enabled,
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Pipeline management — run data/training stages as background jobs, polled
# via GET /jobs/{job_id}. Replaces the old CLI subcommands.
# ---------------------------------------------------------------------------

TrainModel = Literal["laptime", "racewin", "position", "all"]


class TrainRequest(BaseModel):
    model: TrainModel = "all"


class RaceRoundRequest(BaseModel):
    race: Optional[int] = Field(default=None, description="Race round number. Auto-detects if omitted.")


def _job_train(model: TrainModel) -> dict:
    result = orchestrate.train_models(_cfg, model)
    _reload_models()
    return result


def _job_evaluate_laptime(race: Optional[int]) -> dict:
    return orchestrate.evaluate_laptime_job(_cfg, race)


def _job_evaluate_position(race: Optional[int]) -> dict:
    return orchestrate.evaluate_position_job(_cfg, race)


def _job_run_all() -> dict:
    result = orchestrate.run_all(_cfg)
    _reload_models()
    return result


def _job_score_race(race: int) -> dict:
    out = orchestrate.score_race(_cfg, race)
    if out["drift"]["retrain_recommended"] and _cfg.feedback.auto_retrain:
        run_training(_cfg)
        _reload_models()
        out["retrained"] = True
    return out


@app.post("/pipeline/fetch", response_model=JobSubmitted, status_code=202, tags=["pipeline"])
def pipeline_fetch():
    """Fetch raw F1 data from the fastf1 API and save raw CSVs. Runs as a background job."""
    job_id = _submit_job("fetch", run_fetch, _cfg)
    return {"job_id": job_id, "status": "queued"}


@app.post("/pipeline/clean", response_model=JobSubmitted, status_code=202, tags=["pipeline"])
def pipeline_clean():
    """Clean and preprocess raw CSV data. Runs as a background job."""
    job_id = _submit_job("clean", run_cleaning, _cfg)
    return {"job_id": job_id, "status": "queued"}


@app.post("/pipeline/features", response_model=JobSubmitted, status_code=202, tags=["pipeline"])
def pipeline_features():
    """Run feature engineering on cleaned data. Runs as a background job."""
    job_id = _submit_job("features", run_feature_engineering, _cfg)
    return {"job_id": job_id, "status": "queued"}


@app.post("/pipeline/train", response_model=JobSubmitted, status_code=202, tags=["pipeline"])
def pipeline_train(req: TrainRequest):
    """Train ML model(s) and save pipelines to models/. Reloads the API's in-memory
    models automatically once the job succeeds. Runs as a background job."""
    job_id = _submit_job("train", _job_train, req.model)
    return {"job_id": job_id, "status": "queued"}


@app.post("/pipeline/fetch-qualifying", response_model=JobSubmitted, status_code=202, tags=["pipeline"])
def pipeline_fetch_qualifying(req: RaceRoundRequest):
    """Fetch qualifying results for the next (or given) race. Runs as a background job."""
    job_id = _submit_job("fetch-qualifying", fetch_upcoming_qualifying, _cfg, race_round=req.race)
    return {"job_id": job_id, "status": "queued"}


@app.post("/pipeline/evaluate-laptime", response_model=JobSubmitted, status_code=202, tags=["pipeline"])
def pipeline_evaluate_laptime(req: RaceRoundRequest):
    """Compare predicted vs actual lap times for a completed race. Runs as a background job."""
    job_id = _submit_job("evaluate-laptime", _job_evaluate_laptime, req.race)
    return {"job_id": job_id, "status": "queued"}


@app.post("/pipeline/run-all", response_model=JobSubmitted, status_code=202, tags=["pipeline"])
def pipeline_run_all():
    """Run the full pipeline: fetch -> clean -> features -> train -> fetch-qualifying.
    Runs as a single background job."""
    job_id = _submit_job("run-all", _job_run_all)
    return {"job_id": job_id, "status": "queued"}


@app.post("/pipeline/evaluate-position", response_model=JobSubmitted, status_code=202, tags=["pipeline"])
def pipeline_evaluate_position(req: RaceRoundRequest):
    """Audit the position model's predicted order vs actual for a completed race. Background job."""
    job_id = _submit_job("evaluate-position", _job_evaluate_position, req.race)
    return {"job_id": job_id, "status": "queued"}


@app.post("/pipeline/score-race", response_model=JobSubmitted, status_code=202, tags=["pipeline"])
def pipeline_score_race(req: RaceRoundRequest):
    """Score a saved prediction (see /predict_next_race?save=true) against the actual
    result, update the feedback loop, and report drift. Background job."""
    if req.race is None:
        raise HTTPException(status_code=422, detail="score-race requires an explicit 'race' round number.")
    job_id = _submit_job("score-race", _job_score_race, req.race)
    return {"job_id": job_id, "status": "queued"}


# ---------------------------------------------------------------------------
# Feedback loop — read-only views (synchronous)
# ---------------------------------------------------------------------------
@app.get("/predictions", tags=["feedback"])
def list_predictions():
    """All saved race predictions for the configured season, newest first."""
    return list(reversed(feedback.list_prediction_logs(_cfg)))


@app.get("/predictions/{round_no}", tags=["feedback"])
def get_prediction(round_no: int):
    """The saved prediction (and, once scored, the result) for one round."""
    log = feedback.load_prediction_log(_cfg, round_no)
    if log is None:
        raise HTTPException(status_code=404, detail=f"No prediction log for round {round_no}.")
    return log


@app.get("/score-history", tags=["feedback"])
def score_history():
    """Per-race scores plus the rolling drift scorecard."""
    history = feedback.load_history(_cfg.feedback.score_history_path)
    retrain, reasons = feedback.should_retrain(history, _cfg.feedback) if history else (False, [])
    return {
        "history": history,
        "rolling_scorecard": feedback.rolling_scorecard(history, _cfg.feedback.window_races),
        "drift": {"retrain_recommended": bool(retrain), "reasons": reasons},
    }
