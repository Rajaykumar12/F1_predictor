from fastapi import FastAPI, HTTPException
import logging
from datetime import datetime
from pydantic import BaseModel
import pandas as pd
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="F1 Race Winner Prediction API",
    description="API for predicting F1 race winners",
    version="1.0.0"
)

# Load model once when app starts
with open('models/xgb_racewin_pipeline.pk1', 'rb') as f:
    win_model = pickle.load(f)

# Single prediction input model
class RaceInput(BaseModel):
    Team: str
    Position: int
    GridPosition: int  
    driver_win_rate: float
    team_reliability: float

@app.post("/predict")
def predict(race: RaceInput):
    df = pd.DataFrame([race.dict()])
    
    prediction = win_model.predict(df)[0]
    probability = win_model.predict_proba(df)[0][1]
    
    return {
        "will_win": bool(prediction),
        "win_probability": float(probability),
        "input_data": race.dict()
    }


with open('models/xgb_laptime_pipeline.pk1', 'rb') as f:
    laptime_pipeline = pickle.load(f)

class LapTimeInput(BaseModel):
    Driver: str
    Team: str
    Position: int
    TireCompound: str
    TireAge: int
    driver_win_rate: float
    team_reliability: float

@app.post("/predict_laptime")
def predict_laptime(lap: LapTimeInput):
    features = ['Driver', 'Team', 'Position', 'TireCompound', 
                'TireAge', 'driver_win_rate', 'team_reliability']
    
    df = pd.DataFrame([lap.dict()])[features]
    
    predicted_laptime = laptime_pipeline.predict(df)[0]
    
    return {
        "predicted_laptime": float(predicted_laptime),
        "input_data": lap.dict()
    }


with open('models/race_prediction_pipeline.pk1', 'rb') as f:
    race_model = pickle.load(f)

def create_historical_features(df, n_previous=5):
    features = []
    
    for driver in df['Driver'].unique():
        driver_data = df[df['Driver'] == driver].copy()
        
        # Create rolling features without sorting
        driver_data['avg_position_last'] = driver_data['Position'].rolling(n_previous).mean()
        driver_data['avg_grid_last'] = driver_data['GridPosition'].rolling(n_previous).mean()
        
        driver_data['is_dnf'] = (driver_data['Status'] == "Retired").astype(int)
        driver_data['dnf_last'] = driver_data['is_dnf'].rolling(n_previous).sum()
        
        driver_data['positions_gained'] = driver_data['GridPosition'] - driver_data['Position']
        driver_data['avg_positions_gained'] = driver_data['positions_gained'].rolling(n_previous).mean()
        
        driver_data['podiums_last'] = (driver_data['Position'] <= 3).astype(int).rolling(n_previous).sum()
        
        features.append(driver_data)
    
    result = pd.concat(features)
    
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
async def predict_next_race():
    try:
        # Load and process data
        logger.info("Loading F1 results data...")
        try:
            f1_results = pd.read_csv("data/f1_results_features.csv")
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail="Results data file not found. Check path: data/f1_results_features.csv"
            )

        logger.info("Creating historical features...")
        processed_data = create_historical_features(f1_results)
        
        if processed_data.empty:
            raise HTTPException(
                status_code=500,
                detail="No data available after processing"
            )

        logger.info("Getting latest data for current drivers...")
        latest_data = processed_data.groupby('Driver').last().reset_index()
        
        # Required features for prediction
        race_features = [
            'Driver', 'Team', 'GridPosition',
            'driver_win_rate', 'team_reliability',
            'avg_position_last', 'avg_grid_last', 'dnf_last',
            'avg_positions_gained', 'podiums_last'
        ]
        
        # Verify all required features are present
        missing_features = [f for f in race_features if f not in latest_data.columns]
        if missing_features:
            raise HTTPException(
                status_code=500,
                detail=f"Missing required features: {missing_features}"
            )

        logger.info("Making predictions...")
        predictions = race_model.predict(latest_data[race_features])
        
        # Prepare results
        results = []
        for idx, row in latest_data.iterrows():
            try:
                results.append(DriverPrediction(
                    predicted_position=float(predictions[idx]),
                    driver=str(row['Driver']),
                    team=str(row['Team']),
                    confidence=0.95,
                    recent_form={
                        "avg_position": float(row['avg_position_last']),
                        "podiums": int(row['podiums_last']),
                        "dnfs": int(row['dnf_last'])
                    }
                ))
            except Exception as e:
                logger.error(f"Error processing driver {row['Driver']}: {str(e)}")
                continue
    
        if not results:
            raise HTTPException(
                status_code=500,
                detail="No valid predictions could be generated"
            )

        # Sort by predicted position
        results.sort(key=lambda x: x.predicted_position)
    
        return RacePrediction(
            predictions=results,
            prediction_date=datetime.now().strftime("%Y-%m-%d"),
            next_race="Next Grand Prix"
        )
    
    except Exception as e:
        logger.error(f"Error in predict_next_race: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making predictions: {str(e)}"
        )



@app.get("/health")
def health():
    return {"status": "ok"}

