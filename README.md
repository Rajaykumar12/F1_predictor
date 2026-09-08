# F1 Race Prediction System

A machine learning system for predicting Formula 1 race outcomes. Uses XGBoost models trained on FastF1 telemetry data. The full workflow — data fetching, training, race-day prediction, and post-race scoring — runs **either from the terminal** (`python main.py <command>`) **or through the FastAPI Swagger UI** (`/docs`). Both call the same pipeline code, so they stay in sync. A prediction/scoring feedback loop tracks accuracy per race and flags when the model has drifted.

## Project Structure

```
F1_predictor/
├── main.py                     # CLI: pipeline stages, predict-race, score-race, serve
├── app.py                      # FastAPI server: predictions + pipeline + feedback endpoints
├── config.yaml                 # Central config: paths, season, constants, hyperparameters
├── requirements.txt            # All dependencies (API, pipeline, plotting, tests)
├── Dockerfile                  # Container image for production deployment
├── docker-compose.yml          # Docker Compose service definition
│
├── pipeline/
│   ├── config_loader.py        # Typed config dataclasses + validation + get_config()
│   ├── fetch.py                # FastF1 API → raw CSVs + upcoming qualifying fetch
│   ├── clean.py                # Raw CSVs → cleaned CSVs + data health plots
│   ├── features.py             # Cleaned CSVs → feature CSVs + insight plots
│   ├── train.py                # Feature CSVs → trained model pipelines + metrics.json
│   ├── model_registry.py       # Single loader for models/*.pkl + metrics.json
│   ├── predict.py              # Shared finishing-order inference core (CLI + API)
│   ├── evaluate.py             # Predicted vs actual: lap time + finishing position
│   ├── feedback.py             # Scoring, rolling drift scorecard, per-driver bias
│   ├── orchestrate.py          # Multi-step actions shared by CLI + API jobs
│   └── visualize.py            # Headless matplotlib/seaborn plot generation
│
├── scripts/
│   └── backtest.py             # Honest held-out backtest for any completed round
│
├── tests/
│   ├── test_api.py             # FastAPI endpoint tests (validation, 503, health, feedback)
│   ├── test_cli.py             # CLI command tests (CliRunner)
│   ├── test_feedback.py        # Scoring / bias / drift pure-function tests
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
│   ├── upcoming_qualifying.csv     # Real qualifying result for the target race
│   └── predictions/               # Saved predictions: <season>_r<NN>.json (+ scored result)
│
├── models/                     # Trained sklearn pipelines (git-ignored)
│   ├── xgb_racewin_pipeline.pkl
│   ├── xgb_laptime_pipeline.pkl
│   ├── xgb_laptime_features.pkl    # Lap-time model's train-time feature list
│   ├── race_prediction_pipeline.pkl
│   ├── metrics.json                # CV scores and holdout metrics for all models
│   └── score_history.json          # One record per scored race (feedback loop)
│
├── outputs/plots/              # Auto-generated visualizations (git-ignored)
│   ├── cleaning/               # Data health plots after clean stage
│   ├── features/               # Feature insight plots after features stage
│   └── evaluation/             # Predicted vs actual plots (laptime + position)
│
└── fastf1_cache/               # FastF1 API cache (git-ignored)
```

## Setup

**Requirements:** Python 3.10+

```bash
git clone <repository-url>
cd F1_predictor

pip install -r requirements.txt
```

## Race Weekend Routine

Run it from the terminal, or make the equivalent Swagger call — pick whichever is
handier. `N` is the round number.

| When | Terminal | Swagger (`/docs`) |
|---|---|---|
| **Sat — after qualifying** | `python main.py fetch-qualifying --race N` | `POST /pipeline/fetch-qualifying` `{"race": N}` |
| **Sun — before the race** | `python main.py predict-race --round N --save` | `GET /predict_next_race?save=true` |
| **Sun/Mon — after the race** | `python main.py score-race --round N` | `POST /pipeline/score-race` `{"race": N}` |
| **Periodically / on drift** | `python main.py run-all` | `POST /pipeline/run-all` |

`predict-race` fetches this weekend's qualifying if it is not already on disk,
builds each driver's form from rounds `≤ N-1` (honest — the race hasn't happened
yet), applies the real grid, and prints a ranked finishing order. `--save` writes
it to `data/predictions/<season>_r<NN>.json`.

`score-race` joins that saved prediction against the actual result, appends a
record to `models/score_history.json`, prints a scorecard + rolling drift check,
and (if `feedback.auto_retrain: true`) retrains on drift. See
**Model feedback loop** below.

Wrap the two commands in `cron`, a systemd timer, or the `/loop` skill to run
them automatically each weekend — there is no built-in scheduler.

## Running the pipeline

### From the terminal

`python main.py <command> --help` for options.

| Command | What it does |
|---|---|
| `fetch` / `clean` / `features` | Individual pipeline stages |
| `train --model laptime\|racewin\|position\|all` | Train model(s) → `models/*.pkl` |
| `fetch-qualifying --race N` | Fetch qualifying → `data/upcoming_qualifying.csv` |
| `predict-race --round N [--save] [--apply-bias]` | Ranked finishing-order prediction |
| `score-race --round N [--plots]` | Score a saved prediction; update the feedback loop |
| `evaluate-laptime --race N` / `evaluate-position --race N` | Post-hoc predicted-vs-actual audit |
| `run-all` | fetch → clean → features → train → fetch-qualifying |
| `race-weekend` | Print the routine above |
| `serve [--host --port --reload]` | Start the FastAPI server |

### From Swagger UI

Start the server (`python main.py serve`), open `http://localhost:8000/docs`.
Pipeline stages run as **background jobs** — the `POST` returns a `job_id`; poll
`GET /jobs/{job_id}` for `queued` → `running` → `success`/`failed`. Only one job
runs at a time (a second `POST` returns `409 Conflict`).

| Endpoint | Body | What it does |
|---|---|---|
| `POST /pipeline/run-all` | — | Full pipeline (qualifying step is non-fatal) |
| `POST /pipeline/fetch` / `clean` / `features` | — | Individual stages |
| `POST /pipeline/train` | `{"model": "laptime"\|"racewin"\|"position"\|"all"}` | Train; API models hot-reload on success |
| `POST /pipeline/fetch-qualifying` | `{"race": <round>\|null}` | Fetch qualifying (auto-detects next round if `null`) |
| `POST /pipeline/evaluate-laptime` | `{"race": <round>\|null}` | Predicted vs actual lap times |
| `POST /pipeline/evaluate-position` | `{"race": <round>\|null}` | Predicted vs actual finishing order (audit) |
| `POST /pipeline/score-race` | `{"race": <round>}` | Score a saved prediction + run the drift check |
| `GET /predictions` / `GET /predictions/{round}` | — | Saved predictions (and their scores) |
| `GET /score-history` | — | Per-race scores + rolling drift scorecard |
| `GET /jobs` / `GET /jobs/{job_id}` | — | Job list / single job status |

To switch seasons, change `season:` in `config.yaml` and run `run-all`.

## How Predictions Work

### `predict-race` / `GET /predict_next_race`

Builds a per-driver feature set from their last N completed races — rolling averages of finishing position, wins, podiums, DNF rate, qualifying performance, and form trend — and feeds it into the race position model (XGBoost regressor). The CLI's `predict-race` and the API's `GET /predict_next_race` share the exact same code (`pipeline/predict.py`).

**Without qualifying data** — uses each driver's historical average grid position. Useful as a form guide but not race-specific.

**With qualifying data** (after `fetch-qualifying`) — overrides grid position, gap to pole, and qualifying performance with the real values from this weekend's session. This is what makes it a genuine race-day prediction.

**`as_of_round` cutoff** — `predict-race --round N` builds each driver's form from rounds `≤ N-1` only, so predicting a race never sees its own result. (`GET /predict_next_race` uses whatever is in `f1_results_features.csv`, which on a normal weekend already stops at the last completed round.)

**Bias correction** — when `feedback.bias_correction_enabled` is on, each driver's learned recent error offset is subtracted from the raw model output before the final ranking (see **Model Feedback Loop**). The prediction log keeps both `predicted_position` and `raw_predicted_position`.

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

## Model Feedback Loop

Every `predict-race --save` (or `GET /predict_next_race?save=true`) writes the
full ranking to `data/predictions/<season>_r<NN>.json`. After the race,
`score-race --round N` (or `POST /pipeline/score-race`) joins it against the
actual result and:

1. **Scores it** — winner correct, podium overlap, top-5 / top-10 overlap,
   Spearman rank correlation, position MAE/RMSE — and stores the per-driver
   signed rank error back into the prediction file.
2. **Appends** a record to `models/score_history.json` (one per scored race).
3. **Checks for drift** — over the last `feedback.window_races` scored races, if
   the rolling position MAE, Spearman, or winner hit-rate crosses its threshold,
   it reports "retrain recommended" (and retrains automatically if
   `feedback.auto_retrain: true`). Drift is measured against these **real
   post-race scores**, never against `metrics.json` (whose R² is optimistic —
   the rolling-form features aren't time-shifted).
4. **Learns a per-driver bias** — when `feedback.bias_correction_enabled: true`,
   an exponentially-weighted mean of each driver's recent signed error is
   subtracted from that driver's next prediction (clamped to
   `± bias_max_abs` places). Both the drift check and the bias estimate stay
   inert until `feedback.min_scored_races` races have been scored, since a single
   season is only ~24 races.

`GET /score-history` and the `feedback` block in `GET /health` expose the current
scorecard. Only the **finishing-position** model is scored — the `racewin`
classifier takes the actual finishing position as an input feature and so cannot
predict before a race.

`scripts/backtest.py --round N` runs the same scoring for a past race using
`as_of_round = N-1`, without touching the live prediction log. For a *fully*
honest score the model must also have been trained on data that excludes round N
(the report prints `model trained_at` so you can check).

## Configuration

All paths, season, grid constants, model hyperparameters, and feedback-loop
thresholds live in `config.yaml`. Nothing is hardcoded in the pipeline code.

```yaml
feedback:
  enabled: true
  window_races: 5           # rolling window for the drift scorecard
  min_scored_races: 5       # drift + bias stay inert below this many scored races
  max_position_mae: 4.0     # drift if rolling |pred rank - actual pos| exceeds this
  min_spearman: 0.30        # drift if rolling rank correlation drops below this
  min_winner_hit_rate: 0.15 # drift if rolling winner-correct rate drops below this
  auto_retrain: false       # true = score-race retrains automatically on drift
  bias_correction_enabled: false
  bias_halflife_races: 3.0  # recency half-life for the per-driver bias estimate
  bias_max_abs: 3.0         # clamp each driver's correction to +/- this many places
  prediction_log_dir: data/predictions
  score_history_path: models/score_history.json
```

The rest of the file:

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

To switch seasons, change `season:` in `config.yaml` and run `run-all` (`python main.py run-all` or `POST /pipeline/run-all`).

## Models

Three XGBoost models saved as full sklearn `Pipeline` objects (preprocessor + model in one pickle). Training metrics are saved to `models/metrics.json` and exposed at `/health`.

| Model file | Task | Training evaluation |
|------------|------|---------------------|
| `xgb_racewin_pipeline.pkl` | Binary classification — will this driver win? | Accuracy + F1, stratified 5-fold CV |
| `xgb_laptime_pipeline.pkl` | Regression — predicted lap time (seconds) | MAE + R², 5-fold CV |
| `race_prediction_pipeline.pkl` | Regression — predicted finishing position | MAE + R², 5-fold CV; scored against real results by the feedback loop |

**The `predict-race` / `/predict_next_race` prediction is the position model only.** The `racewin` classifier takes the actual finishing `Position` as an input feature, so it cannot be used to predict a winner *before* a race (its 1.0 accuracy is that leakage, not real skill) — it and the lap-time model are inference utilities exposed via `POST /predict` and `POST /predict_laptime`. Likewise `model_metrics` R² is an optimistic in-sample number; the feedback loop's rolling scorecard is the honest measure.

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
  "feedback": {
    "scored_races": 4,
    "rolling_scorecard": { "races": 4, "winner_hit_rate": 0.25, "position_mae_avg": 3.9, "spearman_avg": 0.44 },
    "retrain_recommended": false,
    "reasons": [],
    "bias_correction_enabled": false
  },
  "config": { "season": 2026, "default_lookback_races": 6, "lookback_range": "3-12" }
}
```

`status` is `"degraded"` if any model failed to load or if data is older than `data_freshness_hours`. The `feedback` block is `null` until at least one race has been scored; **drift never changes `status`** (it is informational).

### `GET /predict_next_race?lookback_races=6&save=false`

Returns predicted finishing positions for all drivers with recent form breakdown. `lookback_races` is configurable (3–12, default from config). `save=true` also writes the prediction to `data/predictions/<season>_r<NN>.json` for later scoring.

Uses real qualifying data automatically if `fetch-qualifying` has been run for this weekend. Response includes `model_r2` from the last training run. Response shape: `{ predictions[], prediction_date, next_race, model_r2 }`.

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

Running `POST /pipeline/clean` or `POST /pipeline/features` automatically saves diagnostic PNGs to `outputs/plots/`.

**After `POST /pipeline/clean` → `outputs/plots/cleaning/`**

| File | What it shows |
|------|---------------|
| `lap_time_distribution.png` | Raw vs cleaned lap time histograms overlaid |
| `outliers_boxplot.png` | Lap time spread per race after outlier removal |
| `missing_data_heatmap.png` | % missing per column per race before cleaning |
| `data_completeness.png` | Lap count retained per Grand Prix weekend |

**After `POST /pipeline/features` → `outputs/plots/features/`**

| File | What it shows |
|------|---------------|
| `correlation_matrix.png` | Numeric feature correlations vs race targets |
| `driver_win_rate.png` | Win rate per driver (sorted) |
| `tire_degradation_by_compound.png` | Lap-to-lap delta per compound |
| `race_phase_distribution.png` | Lap count by Early / Middle / Late phase |
| `team_reliability.png` | % races finished per team |

**After `evaluate-laptime` → `outputs/plots/evaluation/`**

| File | What it shows |
|------|---------------|
| `round{N}_actual_vs_predicted.png` | Scatter of real vs predicted lap times |
| `round{N}_residuals.png` | Error distribution histogram |
| `round{N}_driver_mae.png` | Per-driver MAE bar chart |

**After `evaluate-position` (or `score-race --plots`) → `outputs/plots/evaluation/`**

| File | What it shows |
|------|---------------|
| `position_round{N}_pred_vs_actual.png` | Predicted rank vs actual finishing position, per driver |
| `position_round{N}_rank_error.png` | Per-driver rank error (predicted − actual) |

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ -v --cov=pipeline --cov=app --cov-report=term-missing
```

Tests cover config validation (including the `feedback` section), pipeline unit logic (feature engineering incl. `as_of_round`, cleaning, status handling), the feedback loop (scoring, drift, per-driver bias — pure functions), CLI commands (`click.testing.CliRunner`), and API endpoint behaviour (input validation, 503 on missing models, health shape, the feedback endpoints). Tests that require trained models or live FastF1 data skip gracefully when those aren't present.

## Deployment

### Docker

```bash
# Build and run the API
docker build -t f1-predictor .
docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models f1-predictor

# Or with Docker Compose
docker-compose up

# Run a CLI command in the container (image includes main.py)
docker run --rm -v $(pwd)/data:/app/data:rw -v $(pwd)/models:/app/models:rw \
  f1-predictor python main.py score-race --round 13
```

The container runs gunicorn with 2 uvicorn workers. Mount `data/` and `models/`
so trained models and CSVs persist. Two caveats:

- **`--workers 2` means per-process state.** The job registry and hot-reloaded
  models live in each worker; a `score-race` retrain only reaches the worker that
  ran it. `GET /health` and `GET /score-history` re-read from disk each request.
  Use `--workers 1` if you rely on the write-capable feedback endpoints.
- `docker-compose.yml` mounts `data/` and `models/` **read-only** — fine for the
  API's read paths, but `--save`, `score-race`, and `auto_retrain` need `:rw`
  mounts (as above) or a host-side CLI run.

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
