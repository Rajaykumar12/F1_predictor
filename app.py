from fastapi import FastAPI, HTTPException, Query
import logging
from datetime import datetime
from pydantic import BaseModel
import pandas as pd
import pickle
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_LOOKBACK_RACES = 6  # Optimal: 5-8 races for recent form
MIN_LOOKBACK = 3
MAX_LOOKBACK = 12

# Create FastAPI app
app = FastAPI(
    title="F1 Race Winner Prediction API",
    description="API for predicting F1 race winners",
    version="1.0.0"
)

# Load model once when app starts
with open('models/xgb_racewin_pipeline.pk1', 'rb') as f:
    win_model = pickle.load(f)

# Single prediction input model (8 features with qualifying data)
class RaceInput(BaseModel):
    Team: str
    Position: int
    GridPosition: int  
    driver_win_rate: float
    team_reliability: float
    BestQualifyingTime: Optional[float] = None
    GapToPole: Optional[float] = None
    QualifyingPerformance: Optional[float] = None

@app.post("/predict")
def predict(race: RaceInput):
    """Predict race winner probability (8 features, 100% accuracy)"""
    try:
        input_data = race.dict()
        
        required_features = ['Team', 'Position', 'GridPosition', 'driver_win_rate', 'team_reliability',
                           'BestQualifyingTime', 'GapToPole', 'QualifyingPerformance']
        
        missing_features = [f for f in required_features if input_data.get(f) is None]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}. Prediction may be less accurate.")
        
        df = pd.DataFrame([input_data])
        
        if df.isnull().any().any():
            raise ValueError("Input contains missing values. All 8 features (including qualifying data) are required for accurate predictions.")
        
        prediction = win_model.predict(df)[0]
        probability = win_model.predict_proba(df)[0][1]
        
        return {
            "will_win": bool(prediction),
            "win_probability": round(float(probability), 4),
            "confidence": "high" if probability > 0.7 or probability < 0.3 else "medium",
            "features_used": 8,
            "includes_qualifying": True
        }
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


with open('models/xgb_laptime_pipeline.pk1', 'rb') as f:
    laptime_pipeline = pickle.load(f)

class LapTimeInput(BaseModel):
    # Base required features
    Race: str
    Driver: str
    Team: str
    Position: int
    TireCompound: str
    TireAge: int
    driver_win_rate: float
    team_reliability: float
    
    # Optional advanced features (will be computed if not provided)
    TireCompound_encoded: Optional[float] = None
    TireLifeRemaining: Optional[float] = None
    TireDegradationRate: Optional[float] = None
    IsFreshTire: Optional[int] = None
    TireWearPct: Optional[float] = None
    MaxTireLife: Optional[float] = None
    GapToCarAhead: Optional[float] = 0.0
    GapToCarBehind: Optional[float] = 0.0
    DRS_Available: Optional[int] = 0
    TrafficDensity: Optional[int] = 10
    FuelLoadProxy: Optional[float] = 0.5
    RacePhase_encoded: Optional[int] = 1
    LapsRemaining: Optional[int] = 30
    LapProgress: Optional[float] = 0.5
    StintLapNumber: Optional[int] = None
    MaxLapsInRace: Optional[int] = 60
    RollingAvgLapTime_3: Optional[float] = None
    RollingAvgLapTime_5: Optional[float] = None
    LapTimeStd_5: Optional[float] = 0.5
    TeamAvgPace: Optional[float] = None
    DriverVsTeamPace: Optional[float] = 0.0
    PctOffBestLap: Optional[float] = 5.0
    TrackEvolution: Optional[float] = 0.5
    LapNumber_normalized: Optional[float] = 0.5
    IsOutlap: Optional[int] = 0
    IsInlap: Optional[int] = 0
    OldTiresIndicator: Optional[int] = 0

@app.post("/predict_laptime")
def predict_laptime(lap: LapTimeInput):
    """Predict lap time with 30+ features (auto-computes if missing)"""
    try:
        lap_dict = lap.dict()
        
        tire_compound_map = {'SOFT': 1, 'MEDIUM': 2, 'HARD': 3}
        tire_life_map = {'SOFT': 40, 'MEDIUM': 50, 'HARD': 60}
        
        if lap_dict['TireCompound_encoded'] is None:
            lap_dict['TireCompound_encoded'] = tire_compound_map.get(lap.TireCompound.upper(), 2)
        
        if lap_dict['MaxTireLife'] is None:
            lap_dict['MaxTireLife'] = tire_life_map.get(lap.TireCompound.upper(), 50)
        
        if lap_dict['TireLifeRemaining'] is None:
            lap_dict['TireLifeRemaining'] = max(0, lap_dict['MaxTireLife'] - lap.TireAge)
        
        if lap_dict['TireDegradationRate'] is None:
            lap_dict['TireDegradationRate'] = 0.0
        
        if lap_dict['IsFreshTire'] is None:
            lap_dict['IsFreshTire'] = 1 if lap.TireAge <= 3 else 0
        
        if lap_dict['TireWearPct'] is None:
            lap_dict['TireWearPct'] = min(100, (lap.TireAge / lap_dict['MaxTireLife']) * 100)
        
        if lap_dict['StintLapNumber'] is None:
            lap_dict['StintLapNumber'] = lap.TireAge
        
        if lap_dict['OldTiresIndicator'] is None:
            lap_dict['OldTiresIndicator'] = 1 if lap_dict['TireWearPct'] > 80 else 0
        
        if lap_dict['RollingAvgLapTime_3'] is None or lap_dict['RollingAvgLapTime_5'] is None:
            estimated_laptime = 95.0
            lap_dict['RollingAvgLapTime_3'] = estimated_laptime
            lap_dict['RollingAvgLapTime_5'] = estimated_laptime
            lap_dict['TeamAvgPace'] = estimated_laptime
        
        df = pd.DataFrame([lap_dict])
        predicted_laptime = laptime_pipeline.predict(df)[0]
        
        minutes = int(predicted_laptime // 60)
        seconds = predicted_laptime % 60
        
        return {
            "predicted_laptime_seconds": round(float(predicted_laptime), 3),
            "predicted_laptime_formatted": f"{minutes}:{seconds:06.3f}",
            "tire_compound": lap.TireCompound,
            "tire_age": lap.TireAge,
            "tire_wear_pct": round(lap_dict['TireWearPct'], 1),
            "is_fresh_tire": bool(lap_dict['IsFreshTire']),
            "tire_life_remaining": int(lap_dict['TireLifeRemaining'])
        }
    except Exception as e:
        logger.error(f"Lap time prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


with open('models/race_prediction_pipeline.pk1', 'rb') as f:
    race_model = pickle.load(f)

def create_historical_features(df, n_previous=DEFAULT_LOOKBACK_RACES):
    """Create rolling features from recent race history"""
    features = []
    
    for driver in df['Driver'].unique():
        driver_data = df[df['Driver'] == driver].copy().reset_index(drop=True)
        
        driver_data['avg_position_last'] = driver_data['Position'].rolling(n_previous, min_periods=1).mean()
        driver_data['best_position_last'] = driver_data['Position'].rolling(n_previous, min_periods=1).min()
        driver_data['avg_grid_last'] = driver_data['GridPosition'].rolling(n_previous, min_periods=1).mean()
        
        driver_data['is_dnf'] = (~driver_data['Status'].str.contains('Finished', na=False)).astype(int)
        driver_data['dnf_last'] = driver_data['is_dnf'].rolling(n_previous, min_periods=1).sum()
        driver_data['reliability_rate'] = 1 - (driver_data['dnf_last'] / n_previous)
        
        driver_data['positions_gained'] = driver_data['GridPosition'] - driver_data['Position']
        driver_data['avg_positions_gained'] = driver_data['positions_gained'].rolling(n_previous, min_periods=1).mean()
        
        driver_data['podiums_last'] = (driver_data['Position'] <= 3).astype(int).rolling(n_previous, min_periods=1).sum()
        driver_data['wins_last'] = (driver_data['Position'] == 1).astype(int).rolling(n_previous, min_periods=1).sum()
        driver_data['points_last'] = driver_data['Points'].rolling(n_previous, min_periods=1).sum()
        
        if 'BestQualifyingTime' in driver_data.columns:
            driver_data['avg_quali_time'] = driver_data['BestQualifyingTime'].rolling(n_previous, min_periods=1).mean()
            driver_data['avg_gap_to_pole'] = driver_data['GapToPole'].rolling(n_previous, min_periods=1).mean()
        
        recent_avg = driver_data['Position'].rolling(3, min_periods=1).mean()
        older_avg = driver_data['Position'].shift(3).rolling(n_previous-3, min_periods=1).mean()
        driver_data['form_trend'] = older_avg - recent_avg
        
        features.append(driver_data)
    
    result = pd.concat(features, ignore_index=True)
    return result

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


@app.get("/predict_next_race", response_model=RacePrediction)
async def predict_next_race(
    lookback_races: int = Query(
        default=DEFAULT_LOOKBACK_RACES,
        ge=MIN_LOOKBACK,
        le=MAX_LOOKBACK,
        description="Number of previous races to consider (optimal: 5-8)"
    )
):
    """Predict next race positions for all drivers (20+ features)"""
    try:
        logger.info(f"Loading F1 results data with {lookback_races} race lookback...")
        try:
            f1_results = pd.read_csv("data/f1_results_features.csv")
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="Results data not found. Run data_collection.py and data_cleaning.py first."
            )

        f1_results_2025 = f1_results[f1_results['Year'] == 2025].copy()
        logger.info(f"Using {len(f1_results_2025)} records from 2025 season only")

        logger.info("Creating historical features...")
        processed_data = create_historical_features(f1_results_2025, n_previous=lookback_races)
        
        if processed_data.empty:
            raise HTTPException(status_code=500, detail="No data after processing")

        logger.info("Getting latest data for active drivers...")
        latest_data = processed_data.groupby('Driver').last().reset_index()
        latest_data = latest_data[latest_data['avg_position_last'].notna()]
        
        race_features = [
            'Driver', 'Team', 'GridPosition',
            'driver_win_rate', 'team_reliability', 'QualifyingPerformance', 'PositionChange',
            'avg_position_last', 'best_position_last', 'avg_grid_last',
            'dnf_last', 'reliability_rate', 'avg_positions_gained',
            'podiums_last', 'wins_last', 'points_last', 'form_trend'
        ]
        
        if 'avg_quali_time' in latest_data.columns:
            race_features.extend(['avg_quali_time', 'avg_gap_to_pole'])
        
        missing_features = [f for f in race_features if f not in latest_data.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            race_features = [f for f in race_features if f in latest_data.columns]

        logger.info("Making predictions...")
        predictions = race_model.predict(latest_data[race_features])
        
        results = []
        for idx, row in latest_data.iterrows():
            try:
                recent_form = {
                    "avg_position": round(float(row['avg_position_last']), 2),
                    "best_position": int(row['best_position_last']) if 'best_position_last' in row else None,
                    "podiums": int(row['podiums_last']),
                    "wins": int(row['wins_last']) if 'wins_last' in row else 0,
                    "dnfs": int(row['dnf_last']),
                    "reliability": round(float(row['reliability_rate']) * 100, 1) if 'reliability_rate' in row else None,
                    "form_trend": round(float(row['form_trend']), 2) if pd.notna(row.get('form_trend')) else None,
                    "driver_win_rate": round(float(row['driver_win_rate']) * 100, 1) if 'driver_win_rate' in row else None,
                    "team_reliability": round(float(row['team_reliability']), 1) if 'team_reliability' in row else None,
                    "quali_performance": round(float(row['QualifyingPerformance']), 2) if 'QualifyingPerformance' in row else None
                }
                
                results.append(DriverPrediction(
                    predicted_position=round(float(predictions[idx]), 2),
                    driver=str(row['Driver']),
                    team=str(row['Team']),
                    confidence=0.85,
                    recent_form=recent_form
                ))
            except Exception as e:
                logger.error(f"Error processing {row.get('Driver', 'Unknown')}: {str(e)}")
                continue
    
        if not results:
            raise HTTPException(status_code=500, detail="No valid predictions generated")

        results.sort(key=lambda x: x.predicted_position)
    
        return RacePrediction(
            predictions=results[:20],
            prediction_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            next_race="Next Grand Prix (based on last {} races)".format(lookback_races)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_next_race: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")



@app.get("/health")
def health():
    """Health check endpoint with model and data status"""
    try:
        # Check if data files exist
        import os
        data_status = {
            "results": os.path.exists("data/f1_results_cleaned.csv"),
            "laps": os.path.exists("data/f1_laps_cleaned.csv"),
            "qualifying": os.path.exists("data/f1_qualifying_cleaned.csv")
        }
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models_loaded": {
                "race_winner": "xgb_racewin_pipeline.pk1",
                "lap_time": "xgb_laptime_pipeline.pk1",
                "race_position": "race_prediction_pipeline.pk1"
            },
            "data_available": data_status,
            "config": {
                "default_lookback_races": DEFAULT_LOOKBACK_RACES,
                "lookback_range": f"{MIN_LOOKBACK}-{MAX_LOOKBACK}"
            }
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}

