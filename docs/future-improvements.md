# Future improvements — backlog

Ideas for after the Phase A–F work in `prediction-improvement-plan.md`. Nothing
here is implemented yet. Written 2026-10-04, after round 16 (Bahrain); 7 rounds
of the 2026 season remain.

Baseline to beat (`models/metrics.json`, `models/score_history.json`):

| Model | Metric | Value |
|---|---|---|
| position (forward-chain) | Spearman / MAE | 0.765 / 2.97 |
| position (live, 4 scored races) | winner hit-rate / MAE / Spearman | 0.75 / 4.50 / 0.50 |
| dnf | CV AUC | **0.59** (weak) |
| laptime | R² / MAE | 0.98 / 0.74 s |

Measure every accuracy change with `python scripts/rolling_backtest.py`
(walk-forward, retrains per round), not a single-round backtest.

**Suggested order:** #5 (quick win) → #1 (main accuracy project) → #2 → the rest.

---

## Accuracy

### 1. Practice-session pace features — ⭐ highest expected gain
Same as **B5** in `prediction-improvement-plan.md` (never started).

- **Why:** qualifying measures one-lap speed; FP2 long runs show race pace and
  tyre degradation — the gap behind cases like Norris going from P4 on the grid to a P12 prediction.
- **How:**
  - `pipeline/fetch.py` `collect_single_race`: also load `"FP2"` (+ `"FP3"`) and
    save practice laps (new `f1_practice_simple.csv`, keyed `Year, Race, Driver, Session, LapNumber`).
  - New `practice` family in `pipeline/feature_registry.py` + builder in `features.py`:
    `fp_long_run_pace_rank` (median of green-flag laps in stints with tyre life > 5),
    `fp_short_run_pace_rank`, `practice_vs_quali_delta`.
  - Sprint weekends have only FP1 — fall back to FP1 or leave NaN (median-fill).
- **Watch out:** real refetch of every season → FastF1 hourly rate limit. Fetch
  per season, resume on failure (the `run_fetch` loop already saves per season).
- **Done when:** rolling backtest Spearman / MAE improve vs. baseline; leakage gate
  (`tests/test_feature_quality.py`) still green.

### 2. Stronger DNF model (AUC 0.59)
DNF probabilities drive the Monte-Carlo (`pipeline/simulate.py`), so a weak model
blurs every P(win)/P(podium).

- `team_reliability_todate` already exists — extend rather than duplicate:
  - power-unit manufacturer reliability (Mercedes/Ferrari/Honda/RBPT/Audi customers share failures)
  - circuit safety-car / retirement rate (street circuits: Baku, Singapore, Monaco)
  - first-lap incident risk by grid slot (midfield P8–P16 is riskest)
  - split mechanical vs. accident DNFs if `Status` allows (`pipeline/clean.py` status canon)
- **Where:** `pipeline/train.py` `train_dnf_model` (line ~731), registry `reliability` family.
- **Done when:** CV AUC clearly above 0.59, Brier below 0.20.

### 3. Automatic weather features
`data/context.csv` already has a hand-entered `context_wet_race_forecast` column.

- Pull FastF1 session weather (`session.weather_data` — already downloaded during
  fetch) for qualifying / practice: track temp, air temp, rainfall flag.
- Race-day rain is only knowable *after* the race → must stay `as_of_safe=False`
  (see `feature_registry.py` line ~68); use the forecast / Saturday conditions instead.
- Optional: a weather-forecast API for Sunday rain probability.

### 4. Mid-season driver/team swaps
2026 data shows Lawson listed at Red Bull Racing and Tsunoda at Racing Bulls only
for rounds 12–14. Driver form currently follows the *driver*; much of the
performance is the *car*.

- First verify whether the swap is real or a FastF1 labelling glitch.
- Feature idea: weight the driver's recent form with the form of the team they
  are driving for *this* weekend (team from `upcoming_qualifying.csv`).
- Display bug: the forecast table's `Team` column comes from the driver's latest
  results row, not from the qualifying sheet — take it from qualifying.

---

## New predictions

### 5. Championship odds — ⭐ quick win
Monte-Carlo the **remaining rounds** to get title probabilities for drivers and
constructors ("Antonelli 61%, Verstappen 27%").

- Reuse `simulate.py` `simulate_race` per remaining round; sum F1 points
  (25-18-15-12-10-8-6-4-2-1, plus sprint points on sprint weekends) on top of
  current standings from `f1_results_simple.csv`.
- For future rounds there's no qualifying yet → use each driver's current
  predicted position / historical grid; add extra noise for later rounds.
- Correlate trials across rounds (a car that's fast stays fast) rather than
  independent races — otherwise odds are overconfident.
- Surface as `python main.py championship` + `GET /championship`.

### 6. Pit-strategy simulator
Use the lap-time model (R² 0.98) to compare one-stop vs. two-stop strategies and
the best pit lap per compound sequence for a given circuit.

- Inputs: tyre compound, tyre life, fuel load (lap number), pit-loss time per circuit
  (add a column to `data/circuits.csv`).
- Output: total race time per strategy, fastest strategy, pit window.

### 7. "What if" grid scenarios
Re-run the forecast with a modified grid, e.g. "Verstappen starts from the pit lane"
or a penalty announced after qualifying.

- API: `POST /predict_scenario` with `{"grid_overrides": {"M VERSTAPPEN": 22}}`.
- Implementation: `predict_race` already takes `qualifying_path` → write a modified
  copy of `upcoming_qualifying.csv` to a temp file and pass it.

---

## Measurement & presentation

### 8. Probability calibration check
Are the stated probabilities honest? Bin saved predictions by P(win) / P(podium)
and compare against how often the outcome happened (reliability diagram).

- Data: every `data/predictions/*.json` that has a `scored` block.
- Needs ~10+ scored races to be meaningful; until then use the rolling backtest outputs.
- If miscalibrated: tune `residual_std` in `simulate.py` (`residual_std_from_metrics`).
- Benchmark: compare against bookmaker win odds for an outside yardstick.

### 9. Results dashboard
One page: latest forecast, predicted vs. actual per round, rolling Spearman / MAE /
winner hit-rate over time, per-driver bias.

- Sources: `data/predictions/`, `models/score_history.json`, `outputs/rolling_backtest.csv`.
- Either a published artifact page or a `GET /dashboard` HTML route on the existing FastAPI app.

---

## Housekeeping

### 10. One-command weekend + scheduling
- `score-race` should also run `clean` + `features` after fetching the result
  (today it only appends to the raw `f1_results_simple.csv`, so the next forecast
  reads stale features unless they're run by hand).
- Optional `systemd` timer / cron: Saturday evening `fetch-qualifying`, Sunday
  pre-race `predict-race --save`, Sunday night `score-race` + rebuild.

### 11. Docs consistency
- README **"Race Weekend Routine"** table (around line 168) still omits `clean` /
  `features` after `score-race` — align it with the "Commands to run the project" section.
- `python main.py race-weekend` output and the `predict-race` docstring
  (`main.py` ~157) still describe the old routine.

### 12. Small robustness items
- A full `python main.py fetch` that stops on the rate limit leaves the data
  partially updated; consider a `fetch --since-round N` / "fetch only missing
  rounds" mode instead of re-downloading every season.
- Add a `fetch-race --round N` CLI command wrapping `fetch_single_race`, instead of the
  `python -c` one-liner in the README.
