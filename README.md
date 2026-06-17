# F1 Race Prediction System

A machine learning system for predicting Formula 1 race outcomes. Uses XGBoost models trained on FastF1 telemetry data, served through a FastAPI REST API. The entire workflow — data fetching through race-day prediction — runs from a single CLI.

## Project Structure

```
F1_predictor/
├── main.py                     # CLI entry point (run everything from here)
├── app.py                      # FastAPI prediction server
├── config.yaml                 # Central config: paths, season, constants, hyperparameters
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Dev/test dependencies (jupyter, pytest, etc.)
├── Dockerfile                  # Container image for production deployment
├── docker-compose.yml          # Docker Compose service definition
│
├── pipeline/
│   ├── config_loader.py        # Typed config dataclasses + validation + get_config()
│   ├── fetch.py                # FastF1 API → raw CSVs + upcoming qualifying fetch
│   ├── clean.py                # Raw CSVs → cleaned CSVs + data health plots
│   ├── features.py             # Cleaned CSVs → feature CSVs + insight plots
│   ├── train.py                # Feature CSVs → trained model pipelines + metrics.json
│   ├── evaluate.py             # Predicted vs actual lap time comparison
│   └── visualize.py            # Headless matplotlib/seaborn plot generation
│
├── tests/
│   ├── test_api.py             # FastAPI endpoint tests (validation, 503, health)
│   ├── test_pipeline.py        # Unit tests for clean.py and features.py
│   └── test_config.py          # Config loading, validation, and defaults
│
├── data/                       # Generated CSV artifacts (git-ignored)
│   ├── f1_laps_simple.csv
│   ├── f1_results_simple.csv
│   ├── f1_qualifying_simple.csv
│   ├── f1_laps_cleaned.csv
│   ├── f1_results_cleaned.csv
│   ├── f1_qualifying_cleaned.csv
│   ├── f1_laps_features.csv
│   ├── f1_results_features.csv
│   └── upcoming_qualifying.csv     # Real qualifying result for the next race
│
├── models/                     # Trained sklearn pipelines (git-ignored)
│   ├── xgb_racewin_pipeline.pkl
│   ├── xgb_laptime_pipeline.pkl
│   ├── xgb_laptime_features.pkl    # Feature list used at train time (for predict)
│   ├── race_prediction_pipeline.pkl
│   ├── race_position_feature_info.pkl
│   └── metrics.json                # CV scores and holdout metrics for all models
│
├── outputs/plots/              # Auto-generated visualizations (git-ignored)
│   ├── cleaning/               # Data health plots after clean stage
│   ├── features/               # Feature insight plots after features stage
│   └── evaluation/             # Predicted vs actual plots after evaluate
│
└── fastf1_cache/               # FastF1 API cache (git-ignored)
```

## Setup

**Requirements:** Python 3.10+

```bash
git clone <repository-url>
cd F1_predictor

# Production
pip install -r requirements.txt

# Development (includes jupyter, pytest, matplotlib)
pip install -r requirements-dev.txt
```

## Race Weekend Routine

This is the week-to-week workflow once the system is set up.

**Saturday — after qualifying:**
```bash
python main.py fetch-qualifying
```
Fetches real grid positions from today's qualifying session and saves them to `data/upcoming_qualifying.csv`. The prediction endpoint and CLI command automatically use these instead of historical averages.

**Sunday — before the race:**
```bash
# Terminal output (no server needed)
python main.py predict

# Or via the API
python main.py serve
curl -s http://localhost:8000/predict_next_race | python -m json.tool
```

**Sunday — after the race:**
```bash
python main.py run-all
```
Ingests the completed race, retrains all models on the updated season data, and attempts to fetch qualifying for the next race weekend. By Monday the system is ready for the following round.

## All CLI Commands

```bash
# Full pipeline (fetch → clean → features → train → fetch-qualifying)
python main.py run-all

# Individual pipeline stages
python main.py fetch                         # Pull completed race data from FastF1
python main.py clean                         # Clean raw CSVs + save data health plots
python main.py features                      # Engineer features + save insight plots
python main.py train                         # Train all three models
python main.py train --model laptime         # Train a specific model only
python main.py train --model racewin
python main.py train --model position

# Qualifying and prediction
python main.py fetch-qualifying              # Fetch qualifying for the next race (auto-detects round)
python main.py fetch-qualifying --race 8     # Fetch qualifying for a specific round
python main.py predict                       # Print predicted race finishing order to terminal
python main.py predict --lookback 8         # Use a custom form window (default: config value)

# Evaluation
python main.py evaluate-laptime              # Compare predicted vs actual (most recent race)
python main.py evaluate-laptime --race 3     # Evaluate a specific round

# Serve the prediction API
python main.py serve
python main.py serve --port 8080 --reload    # Dev mode with auto-reload
```

## How Predictions Work

### `predict` CLI and `GET /predict_next_race`

Builds a per-driver feature set from their last N completed races — rolling averages of finishing position, wins, podiums, DNF rate, qualifying performance, and form trend — and feeds it into the race position model (XGBoost regressor).

**Without qualifying data** — uses each driver's historical average grid position. Useful as a form guide but not race-specific.

**With qualifying data** (after `fetch-qualifying`) — overrides grid position, gap to pole, and qualifying performance with the real values from this weekend's session. This is what makes it a genuine race-day prediction.

The `next_race` field in the API response shows which mode was active:
```
"Australian Grand Prix — real qualifying, last 6 races form"
"Next Grand Prix — historical grid positions, last 6 races form"
```

### Confidence scores

Per-driver confidence in `/predict_next_race` is computed as:

```
confidence = model_R² × (1 - |form_trend| / 5)
```

`model_R²` comes from `models/metrics.json` (written at train time). Drivers with erratic recent form get lower confidence than drivers with consistent results.

## Configuration

All paths, season, grid constants, and model hyperparameters live in `config.yaml`. Nothing is hardcoded in the pipeline code.

```yaml
paths:
  data_dir: data
  models_dir: models
  cache_dir: fastf1_cache
  plots_dir: outputs/plots

pipeline:
  season: 2026
  lookback_races: 6      # rolling window for historical features (optimal: 5-8)
  min_lookback: 3
  max_lookback: 12
  api_sleep_seconds: 2

constants:
  grid_size: 20                  # max grid positions (used for QualifyingPerformance %)
  race_phase_bins: [0, 15, 40, 100]
  position_bins: [0, 5, 10, 15, 20]
  default_tire: "MEDIUM"
  unknown_position: 15
  completed_statuses:            # statuses NOT counted as DNF
    - "Finished"
    - "+1 Lap"
    - "+2 Laps"
    - "+3 Laps"
    - "+4 Laps"
    - "+5 Laps"

logging:
  level: INFO

api:
  cors_origins: ["*"]
  data_freshness_hours: 48       # /health reports "degraded" if data is older than this

models:
  laptime:
    n_estimators: 1000
    learning_rate: 0.01
    max_depth: 5
    random_state: 42
  racewin:                       # same keys, different values
    ...
  position:
    ...
```

To switch seasons, change `season:` in `config.yaml` and run `python main.py run-all`.

## Models

Three XGBoost models saved as full sklearn `Pipeline` objects (preprocessor + model in one pickle). Training metrics are saved to `models/metrics.json` and exposed at `/health`.

| Model file | Task | Training evaluation |
|------------|------|---------------------|
| `xgb_racewin_pipeline.pkl` | Binary classification — will this driver win? | Accuracy + F1, stratified 5-fold CV |
| `xgb_laptime_pipeline.pkl` | Regression — predicted lap time (seconds) | MAE + R², 5-fold CV |
| `race_prediction_pipeline.pkl` | Regression — predicted finishing position | MAE + R², 5-fold CV |

### Lap time features (20 total)

Base: `Race`, `Driver`, `Team`, `Position`, `TireCompound`, `TireAge`, `driver_win_rate`, `team_reliability`

Engineered: `TireCompound_encoded`, `IsFreshTire`, `StintLapNumber`, `LapNumber_normalized`, `FuelLoadProxy`, `IsOutlap`, `IsInlap`, `positions_gained`, `tire_degradation`, `RollingAvgLapTime_3`, `RollingAvgLapTime_5`, `LapTimeStd_5`

### Race position features (up to 19)

Rolling per-driver over last N races: `avg_position_last`, `best_position_last`, `avg_grid_last`, `dnf_last`, `reliability_rate`, `avg_positions_gained`, `podiums_last`, `wins_last`, `points_last`, `form_trend`

Plus qualifying: `GridPosition`, `QualifyingPerformance`, `PositionChange`, `avg_quali_time`, `avg_gap_to_pole`

## API Reference

Start the server: `python main.py serve`

### `GET /health`

```json
{
  "status": "healthy",
  "timestamp": "2026-06-17T12:00:00",
  "models_loaded": {
    "race_winner": true,
    "lap_time": true,
    "race_position": true
  },
  "data_available": {
    "results": true,
    "laps": true,
    "qualifying": true
  },
  "data_age_hours": 2.4,
  "data_fresh": true,
  "model_metrics": {
    "laptime": { "mae": 0.586, "r2": 0.976, "cv_mae_mean": 1.097, "features_used": 20 },
    "racewin": { "accuracy": 1.0, "f1": 1.0, "cv_accuracy_mean": 1.0 },
    "position": { "mae": 2.959, "r2": 0.677, "cv_mae_mean": 2.137 }
  },
  "config": { "season": 2026, "default_lookback_races": 6, "lookback_range": "3-12" }
}
```

`status` is `"degraded"` if any model failed to load or if data is older than `data_freshness_hours`.

### `GET /predict_next_race?lookback_races=6`

Returns predicted finishing positions for all drivers with recent form breakdown. `lookback_races` is configurable (3–12, default from config).

Uses real qualifying data automatically if `fetch-qualifying` has been run for this weekend. Response includes `model_r2` from the last training run.

### `POST /predict` — Race winner probability

```json
{
  "Team": "Ferrari",
  "Position": 1,
  "GridPosition": 1,
  "driver_win_rate": 14.3,
  "team_reliability": 85.7,
  "BestQualifyingTime": 78.792,
  "GapToPole": 0.0,
  "QualifyingPerformance": 5.0
}
```

Returns `will_win` (bool), `win_probability` (0–1 float), `confidence` (high/medium).

Field constraints: `GridPosition` and `Position` must be 1–26; rates must be 0–100. Returns HTTP 422 if violated.

### `POST /predict_laptime` — Lap time prediction

```json
{
  "Race": "Monaco",
  "Driver": "HAM",
  "Team": "Ferrari",
  "Position": 1,
  "TireCompound": "MEDIUM",
  "TireAge": 12,
  "driver_win_rate": 14.3,
  "team_reliability": 85.7
}
```

Returns `predicted_laptime_seconds`, `predicted_laptime_formatted` (M:SS.mmm), `tire_wear_pct`, and `is_fresh_tire`. All derived tire fields are auto-computed if omitted.

All endpoints return HTTP 503 with an actionable message if the relevant model is not loaded.

## Automated Visualizations

Running `clean` or `features` automatically saves diagnostic PNGs to `outputs/plots/`.

**After `python main.py clean` → `outputs/plots/cleaning/`**

| File | What it shows |
|------|---------------|
| `lap_time_distribution.png` | Raw vs cleaned lap time histograms overlaid |
| `outliers_boxplot.png` | Lap time spread per race after outlier removal |
| `missing_data_heatmap.png` | % missing per column per race before cleaning |
| `data_completeness.png` | Lap count retained per Grand Prix weekend |

**After `python main.py features` → `outputs/plots/features/`**

| File | What it shows |
|------|---------------|
| `correlation_matrix.png` | Numeric feature correlations vs race targets |
| `driver_win_rate.png` | Win rate per driver (sorted) |
| `tire_degradation_by_compound.png` | Lap-to-lap delta per compound |
| `race_phase_distribution.png` | Lap count by Early / Middle / Late phase |
| `team_reliability.png` | % races finished per team |

**After `python main.py evaluate-laptime` → `outputs/plots/evaluation/`**

| File | What it shows |
|------|---------------|
| `round{N}_actual_vs_predicted.png` | Scatter of real vs predicted lap times |
| `round{N}_residuals.png` | Error distribution histogram |
| `round{N}_driver_mae.png` | Per-driver MAE bar chart |

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ -v --cov=pipeline --cov=app --cov-report=term-missing
```

Tests cover config validation, pipeline unit logic (feature engineering, cleaning, status handling), and API endpoint behaviour (input validation, 503 on missing models, health check shape). Tests that require trained models or live FastF1 data are marked to skip gracefully when those aren't present.

## Deployment

### Docker

```bash
# Build and run
docker build -t f1-predictor .
docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models f1-predictor

# Or with Docker Compose
docker-compose up
```

The container runs gunicorn with 2 uvicorn workers. Mount `data/` and `models/` as volumes so trained models and feature CSVs persist outside the container.

### Manual (systemd / VPS)

```bash
gunicorn app:app --worker-class uvicorn.workers.UvicornWorker --workers 2 --bind 0.0.0.0:8000
```

## Data Sources

- **FastF1**: Official F1 timing and telemetry API
- **Season**: 2026 (new regulations — 2025 data not used)
- **Cache**: `fastf1_cache/` stores session data locally after first fetch, making reruns fast

## License

MIT — see [LICENSE](LICENSE).
