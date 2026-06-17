from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline.config_loader import get_config
from pipeline.features import create_historical_features

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


def _load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning("Could not load model %s: %s", path, e)
        return None


win_model = _load_pickle(_cfg.paths.models_dir / "xgb_racewin_pipeline.pkl")
laptime_pipeline = _load_pickle(_cfg.paths.models_dir / "xgb_laptime_pipeline.pkl")
race_model = _load_pickle(_cfg.paths.models_dir / "race_prediction_pipeline.pkl")
_laptime_features = _load_pickle(_cfg.paths.models_dir / "xgb_laptime_features.pkl")

_metrics: dict = {}
_metrics_path = _cfg.paths.models_dir / "metrics.json"
if _metrics_path.exists():
    try:
        _metrics = json.loads(_metrics_path.read_text())
    except Exception as e:
        logger.warning("Could not load metrics.json: %s", e)


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
        raise HTTPException(status_code=503, detail="Race win model not loaded. Run: python main.py train --model racewin")
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
        raise HTTPException(status_code=503, detail="Lap time model not loaded. Run: python main.py train --model laptime")
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


@app.get("/predict_next_race", response_model=RacePrediction)
async def predict_next_race(
    lookback_races: int = Query(
        default=DEFAULT_LOOKBACK_RACES,
        ge=MIN_LOOKBACK,
        le=MAX_LOOKBACK,
        description="Number of previous races to consider (optimal: 5-8)",
    )
):
    """Predict next race positions for all drivers."""
    if race_model is None:
        raise HTTPException(status_code=503, detail="Race position model not loaded. Run: python main.py train --model position")
    try:
        results_path = _cfg.paths.data_dir / "f1_results_features.csv"
        try:
            f1_results = pd.read_csv(results_path)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="Results data not found. Run: python main.py features",
            )

        season = _cfg.pipeline.season
        season_data = f1_results[f1_results["Year"] == season].copy()
        logger.info("Using %d records from %s season.", len(season_data), season)

        completed_statuses = _cfg.constants.completed_statuses
        processed = create_historical_features(
            season_data, n_previous=lookback_races, completed_statuses=completed_statuses
        )
        if processed.empty:
            raise HTTPException(status_code=500, detail="No data after processing.")

        latest = processed.groupby("Driver").last().reset_index()
        latest = latest[latest["avg_position_last"].notna()]

        # Override with real qualifying data if available
        upcoming_path = _cfg.paths.data_dir / "upcoming_qualifying.csv"
        using_real_qualifying = False
        race_label = "Next Grand Prix"
        if upcoming_path.exists():
            quali = pd.read_csv(upcoming_path)
            race_label = quali["RaceName"].iloc[0] if "RaceName" in quali.columns else "Next Grand Prix"
            for _, q_row in quali.iterrows():
                driver_mask = latest["Driver"] == q_row["Driver"]
                if not driver_mask.any():
                    continue
                for col in ["GridPosition", "BestQualifyingTime", "GapToPole", "QualifyingPerformance"]:
                    if col in quali.columns:
                        latest.loc[driver_mask, col] = q_row[col]
            using_real_qualifying = True
            logger.info("Applied real qualifying data from %s (%d drivers).", race_label, len(quali))
        else:
            logger.info("No upcoming_qualifying.csv found — using historical grid positions.")

        race_features = [
            "Driver", "Team", "GridPosition",
            "driver_win_rate", "team_reliability", "QualifyingPerformance", "PositionChange",
            "avg_position_last", "best_position_last", "avg_grid_last",
            "dnf_last", "reliability_rate", "avg_positions_gained",
            "podiums_last", "wins_last", "points_last", "form_trend",
        ]
        if "avg_quali_time" in latest.columns:
            race_features.extend(["avg_quali_time", "avg_gap_to_pole"])

        available = [f for f in race_features if f in latest.columns]
        missing = [f for f in race_features if f not in latest.columns]
        if missing:
            logger.warning("Missing features for prediction: %s", missing)

        predictions = race_model.predict(latest[available])

        # Use stored R² as model-level confidence; use form consistency for per-driver confidence
        model_r2 = _metrics.get("position", {}).get("r2")

        results_list = []
        for idx, row in latest.iterrows():
            try:
                # Per-driver confidence: higher when form is consistent (low std across recent positions)
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
                results_list.append(DriverPrediction(
                    predicted_position=round(float(predictions[idx]), 2),
                    driver=str(row["Driver"]),
                    team=str(row["Team"]),
                    confidence=driver_confidence,
                    recent_form=recent_form,
                ))
            except Exception as e:
                logger.error("Error processing %s: %s", row.get("Driver", "Unknown"), e)

        if not results_list:
            raise HTTPException(status_code=500, detail="No valid predictions generated.")

        results_list.sort(key=lambda x: x.predicted_position)
        logger.info(
            "predict_next_race | race=%s lookback=%d | %d drivers predicted",
            race_label, lookback_races, len(results_list),
        )

        quali_note = "real qualifying" if using_real_qualifying else "historical grid positions"
        return RacePrediction(
            predictions=results_list[:20],
            prediction_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            next_race=f"{race_label} — {quali_note}, last {lookback_races} races form",
            model_r2=model_r2,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in predict_next_race: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


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
            "config": {
                "season": _cfg.pipeline.season,
                "default_lookback_races": DEFAULT_LOOKBACK_RACES,
                "lookback_range": f"{MIN_LOOKBACK}-{MAX_LOOKBACK}",
            },
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}
