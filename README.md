# F1 Race Prediction System

A machine learning system for predicting Formula 1 race outcomes. Uses XGBoost models (regression, ranking, and classification) trained on multi-season FastF1 data, plus a Monte-Carlo simulation layer for per-driver win/podium/points probabilities. The full workflow — data fetching, training, race-day prediction, and post-race scoring — runs **either from the terminal** (`python main.py <command>`) **or through the FastAPI Swagger UI** (`/docs`). Both call the same pipeline code, so they stay in sync. A prediction/scoring feedback loop tracks accuracy per race and flags when the model has drifted.

See `docs/prediction-improvement-plan.md` for the full history of the leakage-free, multi-season redesign, including honestly-measured accuracy numbers and two harness bugs found and fixed on 2026-09-13 — read that status block before trusting any single backtest number at face value.

## Project Structure

```
F1_predictor/
├── main.py                     # CLI: pipeline stages, predict-race, score-race, serve
├── app.py                      # FastAPI server: predictions + pipeline + feedback endpoints
├── config.yaml                 # Central config: paths, season, feature families, hyperparameters
├── requirements.txt            # All dependencies (API, pipeline, plotting, tests)
├── Dockerfile                  # Container image for production deployment
├── docker-compose.yml          # Docker Compose service definition
│
├── pipeline/
│   ├── config_loader.py        # Typed config dataclasses + validation + get_config()
│   ├── fetch.py                # FastF1 API → raw CSVs (multi-season loop) + upcoming qualifying fetch
│   ├── clean.py                # Raw CSVs → cleaned CSVs (status normalization, outlier removal)
│   ├── features.py             # Cleaned CSVs → registry-driven, leakage-safe feature CSVs
│   ├── feature_registry.py     # Declarative feature list: family, dtype, leakage-safety flag
│   ├── train.py                # Feature CSVs → trained pipelines (regressor, ranker, DNF classifier)
│   ├── model_registry.py       # Single loader for models/*.pkl + metrics.json
│   ├── predict.py              # Shared finishing-order + probability inference core (CLI + API)
│   ├── simulate.py             # Monte-Carlo race simulation → P(win)/P(podium)/P(points)/band
│   ├── evaluate.py             # Predicted vs actual: lap time + finishing position
│   ├── feedback.py             # Scoring (incl. sharp-end metrics), rolling drift scorecard, per-driver bias
│   ├── orchestrate.py          # Multi-step actions shared by CLI + API jobs
│   ├── analyze.py              # Feature-quality gate: leakage checks, VIF, hypothesis battery
│   └── visualize.py            # Headless matplotlib/seaborn plot generation
│
├── scripts/
│   ├── backtest.py             # Single-round backtest for any completed round (--retrain for a fully honest fresh model)
│   ├── rolling_backtest.py     # Season-wide walk-forward backtest (retrains per round) — the real yardstick
│   └── feature_hypothesis_tests.py  # Registry-driven battery of leakage/signal hypothesis tests
│
├── tests/
│   ├── test_api.py             # FastAPI endpoint tests (validation, 503, health, feedback)
│   ├── test_cli.py             # CLI command tests (CliRunner)
│   ├── test_config.py          # Config loading, validation, and defaults
│   ├── test_pipeline.py        # Unit tests for clean.py and features.py
│   ├── test_features.py        # Feature-builder unit tests (family-by-family)
│   ├── test_feature_registry.py # Registry structure + declared-vs-built consistency
│   ├── test_feature_quality.py # Slow gate: leakage, race_seq monotonicity, VIF, season resets
│   ├── test_fetch.py           # Multi-season collection loop tests
│   ├── test_train.py           # Ranker group integrity, simulate output shapes, target framing
│   ├── test_rolling_backtest.py # Walk-forward backtest unit tests
│   └── test_feedback.py        # Scoring / bias / drift pure-function tests
│
├── data/                       # Generated CSV artifacts (git-ignored except circuits.csv/context.csv)
│   ├── f1_*_simple.csv / *_cleaned.csv / *_features.csv   # Per-stage pipeline output
│   ├── circuits.csv                # Static circuit reference, keyed by circuit_key (overtaking index, street flag, SC prob.)
│   ├── circuit_calendar.csv        # (Year, Round) -> circuit_key, built from fastf1 schedules (auto-generated, cached)
│   ├── context.csv                 # Curated per-race context (tyre allocation, wet flag, rookie flag)
│   ├── feature_manifest.json       # Registry version + feature list snapshot from the last `features` run
│   ├── upcoming_qualifying.csv     # Real qualifying result for the target race (round-tagged)
│   └── predictions/                # Saved predictions: <season>_r<NN>.json (+ scored result)
│
├── models/                     # Trained sklearn pipelines (git-ignored)
│   ├── xgb_racewin_pipeline.pkl
│   ├── xgb_laptime_pipeline.pkl
│   ├── xgb_laptime_features.pkl    # Lap-time model's train-time feature list
│   ├── race_prediction_pipeline.pkl  # Position regressor (D1)
│   ├── race_ranking_pipeline.pkl     # XGBRanker, blended with the regressor (D2)
│   ├── dnf_pipeline.pkl              # DNF classifier, feeds the Monte-Carlo simulation (E1)
│   ├── metrics.json                # CV scores, forward-chain holdout, and sharp-end metrics for all models
│   └── score_history.json          # One record per scored race (feedback loop)
│
├── outputs/                    # Backtest reports + auto-generated visualizations (git-ignored)
│   ├── rolling_backtest.md         # Season-wide honest walk-forward scorecard (scripts/rolling_backtest.py)
│   ├── backtest_r{N}_report.md     # Single-round honest backtest (scripts/backtest.py)
│   └── plots/
│       ├── cleaning/               # Data health plots after clean stage
│       ├── features/               # Feature insight plots after features stage
│       ├── evaluation/             # Predicted vs actual plots (laptime + position)
│       └── rolling_backtest.png    # MAE / Spearman / winner-correct by round
│
├── docs/
│   ├── feature-engineering-redesign-plan.md   # The original leakage-free rebuild (2026-09-09)
│   └── prediction-improvement-plan.md         # Multi-season + probabilistic-output plan, with honest re-measurements
│
└── fastf1_cache/                # FastF1 API cache (git-ignored)
```

## Setup

**Requirements:** Python 3.10+

```bash
git clone <repository-url>
cd F1_predictor

pip install -r requirements.txt
```

## Commands to run the project

First time (or after changing `pipeline.season` / `pipeline.history_start_season` in `config.yaml`):

```bash
python main.py fetch                        # pull raw data for every season in [history_start_season .. season]
python main.py clean                        # normalize status, remove outliers
python main.py features                     # build the registry feature set (race_seq, all families)
python main.py train --model all            # train laptime + racewin + position + position_ranker + dnf
```

Or all four in one go:

```bash
python main.py run-all                      # fetch -> clean -> features -> train -> fetch-qualifying
```

Race weekend, once data/models above already exist:

```bash
python main.py fetch-qualifying             # auto-detects the next upcoming round
python main.py predict-race                 # prints ranked finishing order + P(win)/P(pod)/P(pts)/band
python main.py predict-race --save          # same, and logs it to data/predictions/ for later scoring
python main.py score-race                   # after the race: scores the saved prediction, updates drift/bias
```

Check the model against a specific already-completed round — see the caveat under **Backtesting** below for which of these two to trust:

```bash
python scripts/backtest.py --round 13             # quick check: reuses the currently-trained model
python scripts/backtest.py --round 13 --retrain   # honest check: fits a fresh model on races before round 13 only
```

Run the real accuracy yardstick — walk-forward, retrains for every round, never lets a model see the round it's predicting:

```bash
python scripts/rolling_backtest.py          # writes outputs/rolling_backtest.md + plots/rolling_backtest.png
```

Run the API server instead of the CLI:

```bash
python main.py serve                        # http://localhost:8000/docs
```

Run the tests:

```bash
python -m pytest -q                         # fast suite
python -m pytest -m slow -q                 # feature-quality gate (needs data/f1_results_features.csv)
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

`predict-race` fetches this weekend's qualifying if it is not already on disk
(checking the file's `Race` column actually matches `N` before trusting it —
otherwise it refetches), builds each driver's form from rounds `≤ N-1` (honest
— the race hasn't happened yet, and `race_seq`-ordered so a prior season's
later rounds are never treated as "future" relative to this season's early
ones), applies the real grid, runs a 5,000-trial Monte-Carlo simulation, and
prints a ranked finishing order with P(win)/P(podium)/P(points)/10-90 band per
driver. `--save` writes it to `data/predictions/<season>_r<NN>.json`.

`score-race` joins that saved prediction against the actual result, appends a
record to `models/score_history.json`, prints a scorecard (including the
sharp-end metrics — winner logloss, podium/points Brier) + rolling drift
check, and (if `feedback.auto_retrain: true`) retrains on drift. See
**Model feedback loop** below.

Wrap the two commands in `cron`, a systemd timer, or the `/loop` skill to run
them automatically each weekend — there is no built-in scheduler.

## Running the pipeline

### From the terminal

`python main.py <command> --help` for options.

| Command | What it does |
|---|---|
| `fetch` / `clean` / `features` | Individual pipeline stages (fetch spans `[history_start_season .. season]`) |
| `train --model laptime\|racewin\|position\|position_ranker\|dnf\|all` | Train model(s) → `models/*.pkl` |
| `fetch-qualifying --race N` | Fetch qualifying → `data/upcoming_qualifying.csv` |
| `predict-race --round N [--save] [--apply-bias]` | Ranked finishing-order prediction with Monte-Carlo probabilities |
| `score-race --round N [--plots]` | Score a saved prediction; update the feedback loop |
| `evaluate-laptime --race N` / `evaluate-position --race N` | Post-hoc predicted-vs-actual audit |
| `run-all` | fetch → clean → features → train → fetch-qualifying |
| `race-weekend` | Print the routine above |
| `serve [--host --port --reload]` | Start the FastAPI server |

### From Swagger UI

Start the server (`python main.py serve`), open `http://localhost:8000` (redirects
to `/docs`). Routes are grouped under **prediction**, **pipeline**, **feedback**,
and **system** tags, and the request bodies below are pre-filled as examples in
"Try it out." Pipeline stages run as **background jobs** — the `POST` returns a
`job_id`; poll `GET /jobs/{job_id}` for `queued` → `running` → `success`/`failed`.
Only one job runs at a time (a second `POST` returns `409 Conflict`).

| Endpoint | Body | What it does |
|---|---|---|
| `POST /pipeline/run-all` | — | Full pipeline (qualifying step is non-fatal) |
| `POST /pipeline/fetch` / `clean` / `features` | — | Individual stages |
| `POST /pipeline/train` | `{"model": "laptime"\|"racewin"\|"position"\|"position_ranker"\|"dnf"\|"all"}` | Train; API models hot-reload on success |
| `POST /pipeline/fetch-qualifying` | `{"race": <round>\|null}` | Fetch qualifying (auto-detects next round if `null`) |
| `POST /pipeline/evaluate-laptime` | `{"race": <round>\|null}` | Predicted vs actual lap times |
| `POST /pipeline/evaluate-position` | `{"race": <round>\|null}` | Predicted vs actual finishing order (audit) |
| `POST /pipeline/score-race` | `{"race": <round>}` | Score a saved prediction + run the drift check |
| `GET /predictions` / `GET /predictions/{round}` | — | Saved predictions (and their scores) |
| `GET /score-history` | — | Per-race scores + rolling drift scorecard |
| `GET /jobs` / `GET /jobs/{job_id}` | — | Job list / single job status |

To switch seasons, change `season:` in `config.yaml` (and `history_start_season:`
if you also want more/less history) and run `run-all`.

**Securing the mutating endpoints** — all `POST /pipeline/*` routes above accept
an optional `X-API-Key` header. By default no key is configured and they stay
open (fine for local/dev use). To require one, set the `F1_API_KEY` environment
variable (preferred — never commit a real key) or `api.api_key` in
`config.yaml`; the env var takes precedence. Once set, Swagger UI shows an
**Authorize** button for these routes, and a request without a matching header
gets `401 Unauthorized`. Read-only endpoints (`/predict*`, `/health`,
`/predictions*`, `/score-history`, `/jobs*`) are never gated.

## How Predictions Work

### `predict-race` / `GET /predict_next_race`

Builds a per-driver, leakage-safe feature set from `pipeline/feature_registry.py`'s
families (qualifying, rolling form, racecraft, team, reliability, circuit,
championship, prior-season history priors, raceday context, circuit-similarity
form) via `create_historical_features` in `pipeline/features.py`, and feeds it
into the position model (an XGBoost regressor, optionally blended with the
`XGBRanker` rank head). The CLI's `predict-race` and the API's `GET
/predict_next_race` share the exact same code (`pipeline/predict.py`).

**Without qualifying data** — uses each driver's historical average grid position. Useful as a form guide but not race-specific.

**With qualifying data** (after `fetch-qualifying`) — overrides grid position, gap to pole, and qualifying performance with the real values from this weekend's session. This is what makes it a genuine race-day prediction.

**`as_of_round` / `race_seq` cutoff** — `predict-race --round N` builds each driver's form from races before round `N` only, ordered by the cross-season-monotonic `race_seq` (not the season-local `Race` number), so predicting a race never sees its own result and a later round in an earlier season is never mistaken for "the future." (`GET /predict_next_race` uses whatever is in `f1_results_features.csv`, which on a normal weekend already stops at the last completed round.)

**Monte-Carlo probabilities** — `pipeline/simulate.py` runs 5,000 trials per prediction: each driver's DNF is sampled from the DNF classifier's probability, a finishing score is drawn from the regressor/ranker blend plus residual noise calibrated on holdout error, and the trials are aggregated into `win_probability`, `podium_probability`, `points_probability`, and a `p10`–`p90` band. These ride alongside the point-estimate rank in every forecast; `simulation_n_trials` in the response says how many trials ran (`null` if simulation failed or was disabled).

**Bias correction** — when `feedback.bias_correction_enabled` is on, each driver's learned recent error offset is subtracted from the raw model output before the final ranking (see **Model Feedback Loop**). The prediction log keeps both `predicted_position` and `raw_predicted_position`.

The `next_race` field in the API response shows which mode was active:
```
"Italian Grand Prix — real qualifying, last 6 races form"
"Next Grand Prix — historical grid positions, last 6 races form"
```

### Confidence scores

Per-driver confidence in `/predict_next_race` is computed as:

```
confidence = model_R² × (1 - |form_trend| / 5)
```

`model_R²` comes from `models/metrics.json` (written at train time — this is the optimistic, in-sample number; treat the forward-chain / rolling-backtest metrics in the same file as the honest ones). Drivers with erratic recent form get lower confidence than drivers with consistent results.

## Model Feedback Loop

Every `predict-race --save` (or `GET /predict_next_race?save=true`) writes the
full ranking (including probabilities) to `data/predictions/<season>_r<NN>.json`.
After the race, `score-race --round N` (or `POST /pipeline/score-race`) joins
it against the actual result and:

1. **Scores it** — winner correct, podium overlap, top-5 / top-10 overlap,
   Spearman rank correlation, position MAE/RMSE, and the **sharp-end
   metrics** (winner logloss, podium Brier, points Brier — MAE alone hides
   whether the model gets the front of the grid right) — and stores the
   per-driver signed rank error back into the prediction file.
2. **Appends** a record to `models/score_history.json` (one per scored race).
3. **Checks for drift** — over the last `feedback.window_races` scored races, if
   the rolling position MAE, Spearman, or winner hit-rate crosses its threshold,
   it reports "retrain recommended" (and retrains automatically if
   `feedback.auto_retrain: true`). Drift is measured against these **real
   post-race scores**, never against `metrics.json`'s in-sample R².
4. **Learns a per-driver bias** — when `feedback.bias_correction_enabled: true`,
   an exponentially-weighted mean of each driver's recent signed error is
   subtracted from that driver's next prediction (clamped to
   `± bias_max_abs` places). Both the drift check and the bias estimate stay
   inert until `feedback.min_scored_races` races have been scored, since a single
   season is only ~24 races.

`GET /score-history` and the `feedback` block in `GET /health` expose the current
scorecard. Only the **finishing-position** model is scored directly — the
`racewin` classifier takes the actual finishing position as an input feature
and so cannot predict before a race; the **DNF** classifier's calibration is
implicitly checked via the podium/points Brier scores (a driver it flagged as
DNF-likely who then DNFs should already be priced into those probabilities).

### Backtesting — three options, two honesty levels

- **`scripts/backtest.py --round N`** — scores the round using `as_of_round =
  N-1` feature cutoffs, but reuses the **currently-trained, in-place model**.
  If that model was trained on data that includes round `N` (the normal case
  — training usually runs on everything fetched so far), this is a *quick
  directional check*, not a fully honest score: the model's learned
  parameters have already seen round `N`'s outcome even though its input
  features are cut off. The report prints which model it used (and its
  `trained_at` timestamp) so you can tell. This script also guards against
  `data/upcoming_qualifying.csv` holding a *different* round's grid than the
  one being backtested (it falls back to the round's real historical grid
  and prints a note if so), and filters actual results by `(Year, Race)`,
  not just `Race`, since round numbers repeat across seasons in the
  multi-season dataset.
- **`scripts/backtest.py --round N --retrain`** — the fully honest version of
  the same single-round report. Fits a fresh position model on only the
  races strictly before round `N` (same methodology as
  `scripts/rolling_backtest.py`'s walk-forward loop, just for one round with
  a full per-driver table instead of an aggregate-only row), predicts round
  `N` with it, and reports against that. Slower (one real retrain) but the
  model has never seen round `N` in any form. Doesn't use the ranker blend
  or Monte-Carlo simulation — point-estimate rank only, kept deliberately
  simple to match `rolling_backtest.py`.
- **`scripts/rolling_backtest.py`** — the season-wide honest yardstick. For
  every round (walking forward by `race_seq`), it **retrains from scratch**
  on only the races strictly before it, then predicts. No model here has
  ever seen the round it's scoring. This is the number to trust for overall
  model health; see `docs/prediction-improvement-plan.md`'s status block for
  the current season-wide and Monza-specific results, and rerun it yourself
  after any feature/model change with `python scripts/rolling_backtest.py`.
  Use `backtest.py --retrain` instead when you want the full per-driver
  breakdown for just one race rather than an aggregate row.

## Configuration

All paths, season, feature families, model hyperparameters, and feedback-loop
thresholds live in `config.yaml`. Nothing is hardcoded in the pipeline code.

```yaml
pipeline:
  season: 2026
  history_start_season: 2022   # fetch/train spans [history_start_season .. season]
  lookback_races: 6
  min_lookback: 3
  max_lookback: 12
  api_sleep_seconds: 2
  regulation_reset_seasons: [2022, 2026]   # years F1's technical regulations reset — human-maintained,
                                            # see "Regulation-era boundaries" below

features:
  lookback_races: 6           # rolling-window span for form/racecraft families
  shift: 1                    # .shift(N) applied before every rolling window (leakage guard)
  families_enabled: [quali, form, racecraft, team, reliability, circuit, championship, history, raceday, circuit_sim]
  cv_splits: 5                # GroupKFold(by Race) splits for honest CV
  holdout_rounds: 3           # forward-chaining holdout: last N rounds are the test set
  vif_threshold: 12.0         # hard cap; the quality gate fails above this (raised from 10.0 on
                               # 2026-09-13 after the regulation-era fix left team_form_avg_finish_s5
                               # and form_avg_finish_s5 more tightly — but legitimately — correlated)
  feature_selection: lasso    # lasso | vif | none — vif re-measured worse on 2026-09-13, see docs/prediction-improvement-plan.md
  position_target: position   # positions_gained | position — positions_gained implemented but measured worse so far
  tune: none                  # none | quick | full — optuna hyperparameter search

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

constants:
  grid_size: 20                  # max grid positions (used for QualifyingPerformance %)
  race_phase_bins: [0, 15, 40, 100]
  position_bins: [0, 5, 10, 15, 20]
  default_tire: "MEDIUM"
  unknown_position: 15
  completed_statuses:            # statuses NOT counted as DNF, matched against status_canon
    - "Finished"
    - "Lapped"                   # a classified finish, not a DNF

logging:
  level: INFO

api:
  cors_origins: ["*"]
  data_freshness_hours: 48       # /health reports "degraded" if data is older than this
  # api_key: ""                  # requires X-API-Key on POST /pipeline/*; prefer
                                  # the F1_API_KEY env var instead (it takes precedence)

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

Five XGBoost-backed models, each saved as a full sklearn `Pipeline` object (preprocessor + model in one pickle). Training metrics are saved to `models/metrics.json` and exposed at `/health`.

| Model file | Task | Training evaluation |
|------------|------|---------------------|
| `xgb_racewin_pipeline.pkl` | Binary classification — will this driver win? | Accuracy + F1, stratified 5-fold CV |
| `xgb_laptime_pipeline.pkl` | Regression — predicted lap time (seconds) | MAE + R², 5-fold CV |
| `race_prediction_pipeline.pkl` | Regression — predicted finishing position | GroupKFold(by Race) CV + forward-chain holdout, incl. sharp-end metrics |
| `race_ranking_pipeline.pkl` | `XGBRanker(objective="rank:ndcg")` — blended with the regressor for the final order | Spearman / NDCG under the same CV |
| `dnf_pipeline.pkl` | Binary classification — will this driver DNF? | CV AUC; feeds the Monte-Carlo simulation's per-driver DNF sampling |

**The `predict-race` / `/predict_next_race` prediction is the position model (+ ranker blend + DNF-informed simulation).** The `racewin` classifier takes the actual finishing `Position` as an input feature, so it cannot be used to predict a winner *before* a race (its high in-sample accuracy is that leakage, not real skill) — it and the lap-time model are inference utilities exposed via `POST /predict` and `POST /predict_laptime`. Likewise the position model's in-sample R² is optimistic; the forward-chain holdout, the rolling walk-forward backtest, and the feedback loop's rolling scorecard are the honest measures — see `docs/prediction-improvement-plan.md` for what each currently says and why they can legitimately disagree.

### Lap time features (20 total)

Base: `Race`, `Driver`, `Team`, `Position`, `TireCompound`, `TireAge`, `driver_win_rate`, `team_reliability`

Engineered: `TireCompound_encoded`, `IsFreshTire`, `StintLapNumber`, `LapNumber_normalized`, `FuelLoadProxy`, `IsOutlap`, `IsInlap`, `positions_gained`, `tire_degradation`, `RollingAvgLapTime_3`, `RollingAvgLapTime_5`, `LapTimeStd_5`

### Race position features (registry-driven, `pipeline/feature_registry.py`)

Declared per family (`quali`, `form`, `racecraft`, `team`, `reliability`, `circuit`, `championship`, `history`, `raceday`, `circuit_sim` — toggle with `features.families_enabled`), each flagged leakage-safe or not. Every rolling/expanding feature is shift-before-roll (no target race in its own window) and ordered by `race_seq`, not the season-local `Race` number, so it's safe across season boundaries. Which features actually survive selection for the trained model is in `models/metrics.json`'s `features_used` and `data/feature_manifest.json`.

### Regulation-era boundaries

F1's technical regulations reset periodically (2022: ground-effect return;
2026: new power-unit formula + active aero) — a team's competitiveness under
a superseded formula is not a reliable "recent form" signal for the current
one. `config.yaml: pipeline.regulation_reset_seasons` (human-maintained,
update it when a new reset is announced) declares which years these resets
happen; `pipeline.features.regulation_era()` buckets every season into the
era of its most recent reset. Every rolling/expanding "form" feature
(`team_form_avg_finish_s5`, `form_avg_finish_s5`, `form_trend`,
`form_dnf_rate_s8`, `hist_positions_gained_s5`,
`hist_grid_finish_consistency_s5`, `form_avg_finish_similar_circuit_s5`) and
the "history" family's prior-season joins are partitioned by era — a rolling
window still bridges an ordinary same-era season boundary (e.g. a future
2024→2025), but resets hard at a real regulation change, so an old
formula's results can never again be mistaken for current form. Circuit
identity (`circuit_overtaking_index`, `circuit_is_street`,
`circuit_sc_probability`, `driver_circuit_avg_finish_prior`) is resolved via
a real `(Year, Round) -> circuit_key` calendar
(`pipeline.features.load_circuit_calendar`, cached to
`data/circuit_calendar.csv`, built from fastf1's schedule) rather than round
number, since round numbers are reassigned to different circuits year to
year. See `docs/prediction-improvement-plan.md`, "Regulation-era
contamination fix" for the concrete bug this closed and its measured impact.

## API Reference

Start the server: `python main.py serve`

### `GET /health`

```json
{
  "status": "healthy",
  "timestamp": "2026-09-13T12:00:00",
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
    "laptime": { "mae": 0.613, "r2": 0.983, "cv_mae_mean": 0.905, "features_used": 20 },
    "racewin": { "accuracy": 1.0, "f1": 1.0, "cv_accuracy_mean": 1.0 },
    "position": { "mae": 0.613, "r2": 0.983, "forward_chain_mae": 2.839, "forward_chain_spearman": 0.781, "forward_chain_winner_logloss": 1.943 },
    "position_ranker": { "cv": "GroupKFold(5, by race)" },
    "dnf": { "cv_auc_mean": 0.589, "cv_brier_mean": 0.200, "positive_rate": 0.209 }
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

Returns predicted finishing positions for all drivers, with recent-form breakdown and Monte-Carlo probabilities per driver. `lookback_races` is configurable (3–12, default from config). `save=true` also writes the prediction to `data/predictions/<season>_r<NN>.json` for later scoring.

Uses real qualifying data automatically if `fetch-qualifying` has been run for this weekend. Response includes `model_r2` from the last training run and `simulation_n_trials` (5000, or `null` if simulation didn't run). Each driver in `predictions[]` carries `predicted_position`, `confidence`, `recent_form`, and — when simulation succeeded — `win_probability`, `podium_probability`, `points_probability`, `p10`, `p90`, `dnf_probability`.

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

**After `scripts/rolling_backtest.py` → `outputs/plots/`**

| File | What it shows |
|------|---------------|
| `rolling_backtest.png` | MAE / Spearman / winner-correct, per round, across the whole walk-forward backtest |

## Running Tests

```bash
# Run all tests
python -m pytest -q

# With coverage report
python -m pytest -q --cov=pipeline --cov=app --cov-report=term-missing

# The slow feature-quality gate (leakage checks, race_seq monotonicity, VIF) — needs data/f1_results_features.csv
python -m pytest -m slow -q
```

Tests cover config validation, pipeline unit logic (feature engineering incl.
`as_of_round`/`as_of_seq`, cleaning, status normalization), the feature
registry and quality gate, multi-season fetch collection, training (ranker
group integrity, simulation output shapes, target framing), the feedback loop
(scoring, drift, per-driver bias — pure functions), CLI commands
(`click.testing.CliRunner`), and API endpoint behaviour (input validation, 503
on missing models, health shape, the feedback endpoints). Tests that require
trained models or live FastF1 data skip gracefully when those aren't present.

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
- **Seasons**: `[pipeline.history_start_season .. pipeline.season]` in `config.yaml` (default 2022–2026) — multi-season history is used for rolling form, prior-season priors, and honest walk-forward evaluation; see `docs/prediction-improvement-plan.md` Phase B
- **Cache**: `fastf1_cache/` stores session data locally after first fetch, making reruns fast

## License

MIT — see [LICENSE](LICENSE).
