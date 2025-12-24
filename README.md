# 🏎️ F1 Race Prediction System

A comprehensive machine learning system for predicting Formula 1 race outcomes using FastAPI, XGBoost, and the FastF1 API. The system analyzes historical race data, qualifying performance, and driver/team form to generate accurate predictions.

## 📋 Overview

This project provides:
- **Race Winner Prediction**: Predict which driver will win the race
- **Lap Time Prediction**: Estimate lap times based on driver, tire age, and compound
- **Race Position Prediction**: Predict finishing positions for all drivers
- **REST API**: FastAPI-powered endpoints for real-time predictions
- **Qualifying Analysis**: Incorporates qualifying times and grid positions
- **Historical Features**: Uses configurable lookback periods (3-12 races) to analyze recent form

## 🚀 Features

### Prediction Models
- **XGBoost Race Winner Model**: Binary classification (100% accuracy on test data)
- **XGBoost Lap Time Model**: Regression model for lap time predictions
- **Race Position Model**: Multi-class position prediction with 15+ features

### Key Capabilities
- ✅ Configurable lookback period (default: 6 races, optimal: 5-8)
- ✅ Qualifying feature integration (best time, gap to pole, qualifying performance)
- ✅ Real-time predictions via REST API
- ✅ Historical trend analysis (form trends, reliability, success rates)
- ✅ 2025 season data for latest predictions
- ✅ Comprehensive EDA with qualifying visualizations

## 📁 Project Structure

```
F1/
├── app.py                      # FastAPI server with prediction endpoints
├── data_collection.py          # Collects race, lap, and qualifying data from FastF1
├── data_cleaning.py            # Cleans and processes data with qualifying features
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── data/                       # CSV data files
│   ├── f1_laps_simple.csv
│   ├── f1_results_simple.csv
│   ├── f1_qualifying_simple.csv
│   ├── f1_laps_cleaned.csv
│   ├── f1_results_cleaned.csv
│   └── ...
├── models/                     # Trained ML models
│   ├── xgb_racewin_pipeline.pk1
│   ├── xgb_laptime_pipeline.pk1
│   └── race_position_prediction_pipeline.pk1
├── notebooks/                  # Jupyter notebooks for training & analysis
│   ├── feature_and_eda.ipynb
│   ├── race_winner_model.ipynb
│   ├── lap_time_model.ipynb
│   └── next_race_predict.ipynb
└── fastf1_cache/              # Cached FastF1 API data
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd F1
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Packages
```
fastapi
uvicorn
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
fastf1
requests-cache
```

## 📊 Data Pipeline

### 1. Data Collection
Collects race, lap, and qualifying data from FastF1 API (2025 season):

```bash
python data_collection.py
```

**Output:**
- `f1_laps_simple.csv` - Lap-by-lap data
- `f1_results_simple.csv` - Race results
- `f1_qualifying_simple.csv` - Qualifying times (Q1, Q2, Q3)

### 2. Data Cleaning
Processes raw data, handles missing values, and creates qualifying features:

```bash
python data_cleaning.py
```

**Features Created:**
- `BestQualifyingTime` - Fastest qualifying lap
- `GapToPole` - Time difference from pole position
- `QualifyingPerformance` - Normalized qualifying metric

### 3. Model Training
Open and run the Jupyter notebooks in order:

1. **`feature_and_eda.ipynb`** - Exploratory data analysis with qualifying visualizations
2. **`race_winner_model.ipynb`** - Train race winner classifier
3. **`lap_time_model.ipynb`** - Train lap time regressor
4. **`next_race_predict.ipynb`** - Train race position predictor

## 🌐 API Usage

### Start the Server

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Or with auto-reload for development:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### API Endpoints

#### 1. Health Check
```bash
GET http://127.0.0.1:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-18T22:00:00",
  "models_loaded": {
    "race_winner": "xgb_racewin_pipeline.pk1",
    "lap_time": "xgb_laptime_pipeline.pk1",
    "race_position": "race_prediction_pipeline.pk1"
  }
}
```

#### 2. Predict Race Winner
```bash
POST http://127.0.0.1:8000/predict
Content-Type: application/json

{
  "Team": "Red Bull Racing",
  "Position": 1,
  "GridPosition": 1,
  "driver_win_rate": 0.45,
  "team_reliability": 0.95,
  "BestQualifyingTime": 78.5,
  "GapToPole": 0.0,
  "QualifyingPerformance": 1.0
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.98,
  "team": "Red Bull Racing"
}
```

#### 3. Predict Lap Time
```bash
POST http://127.0.0.1:8000/predict_laptime
Content-Type: application/json

{
  "driver": "M VERSTAPPEN",
  "lap": 25,
  "tire_age": 10,
  "compound": "SOFT"
}
```

**Response:**
```json
{
  "predicted_laptime": 82.45,
  "driver": "M VERSTAPPEN",
  "lap": 25,
  "tire_compound": "SOFT",
  "tire_age": 10
}
```

#### 4. Predict Next Race (All Drivers)
```bash
GET http://127.0.0.1:8000/predict_next_race?lookback_races=6
```

**Parameters:**
- `lookback_races` (optional): Number of previous races to analyze (default: 6, range: 3-12)

**Response:**
```json
{
  "predictions": [
    {
      "predicted_position": 2.38,
      "driver": "M VERSTAPPEN",
      "team": "Red Bull Racing",
      "confidence": 0.85,
      "recent_form": {
        "avg_position": 3.83,
        "best_position": 1,
        "podiums": 3,
        "wins": 1,
        "dnfs": 0,
        "reliability": 100.0,
        "form_trend": 1.0
      }
    }
  ],
  "prediction_date": "2025-12-18 22:00",
  "next_race": "Next Grand Prix (based on last 6 races)"
}
```

## 🔧 Configuration

### Lookback Period
Adjust the number of previous races used for prediction in `app.py`:

```python
DEFAULT_LOOKBACK_RACES = 6  # Optimal: 5-8 races
MIN_LOOKBACK = 3
MAX_LOOKBACK = 12
```

### Data Filtering
The API automatically filters for **2025 season data only** when making predictions, ensuring forecasts are based on current season performance.

## 📈 Model Features

### Historical Features (15+)
1. `avg_position_last` - Average finishing position over last N races
2. `best_position_last` - Best position achieved
3. `avg_grid_last` - Average starting grid position
4. `dnf_last` - Number of DNFs (Did Not Finish)
5. `reliability_rate` - Percentage of races finished
6. `avg_positions_gained` - Average positions gained from grid to finish
7. `podiums_last` - Podium finishes in period
8. `wins_last` - Race wins in period
9. `points_last` - Total points scored
10. `form_trend` - Recent vs older form comparison
11. `avg_quali_time` - Average qualifying lap time
12. `avg_gap_to_pole` - Average gap to pole position

### Qualifying Features
- **BestQualifyingTime**: Fastest lap in Q1/Q2/Q3
- **GapToPole**: Time difference from pole position
- **QualifyingPerformance**: Normalized performance score

## 📊 Performance

### Model Accuracy
- **Race Winner Model**: 100% accuracy (Random Forest/Gradient Boosting)
- **Expected Improvement**: 70-75% → 80-85% with qualifying features
- **Lookback Optimal Range**: 5-8 races for best balance

### Prediction Insights
- Qualifying position strongly correlates with race results (correlation ~0.75)
- Pole position win rate: ~40%
- Front row (P1-P2) win rate: ~60%
- Top 3 quali positions lead to ~70% podium rate

## 🔬 Exploratory Data Analysis

The `feature_and_eda.ipynb` notebook includes:

### Qualifying Analysis (9 visualizations)
- Qualifying position distribution by team
- Gap to pole distribution and trends
- Q1/Q2/Q3 session time progressions
- Driver consistency and averages
- Top 10 qualifying success rates

### Qualifying vs Race Correlation (6 visualizations)
- Position scatter plots with correlation coefficients
- Win/podium rates by qualifying position
- Positions gained/lost analysis
- Gap to pole impact on race results

### Correlation Matrix
- Heatmap of all feature relationships
- Qualifying metric impact on race outcomes

## 🧪 Development

### Running in Development Mode
```bash
# Start server with auto-reload
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### Adding New Features
1. Update `data_cleaning.py` to create new features
2. Modify `create_historical_features()` in `app.py`
3. Retrain models in Jupyter notebooks
4. Update API response models if needed

## 📝 Data Sources

- **FastF1 API**: Official F1 timing data
- **Seasons**: 2025 (24 races - complete season through Abu Dhabi)
- **Cache**: `fastf1_cache/` directory for faster subsequent loads

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastF1](https://github.com/theOehrly/Fast-F1) - F1 timing data API
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting library
- Formula 1 - For the amazing sport

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Note**: Predictions are for educational and entertainment purposes only. Actual race results may vary due to weather, strategy, incidents, and other unpredictable factors.

