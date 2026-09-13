# Improving F1 position-model predictions

## Regulation-era contamination fix (2026-09-13, later same day)

A direct question — "F1's 2026 regulations are a major reset, doesn't mixing
2022–2026 training data actively hurt rather than help?" — turned out to be
correct, and the mechanism was concrete, not theoretical. The original
`docs/feature-engineering-redesign-plan.md:34-35` (2026-09-09) explicitly
excluded pre-2026 data for exactly this reason ("2026 is a regulation reset;
pre-2026 cars are a different formula"); Phase B below reversed that decision
without addressing the confound. Investigation found:

- **`team_form_avg_finish_s5`** — the single feature this document's own
  hypothesis battery says the model "leans almost entirely on" (line ~96) —
  had no regulation-era boundary despite its comment claiming season-boundary
  safety. Verified: 2026 Round 1, Red Bull Racing's `team_form_avg_finish_s5`
  = 3.3, entirely sourced from their dominant 2022 form (2022: avg finish
  4.57, real 2026: avg finish 9.08 — team competitiveness fully reshuffled by
  the reset). This directly miscalibrated the model's most important input
  for every early-2026-round prediction.
- The same unbounded-rolling-window pattern was also present in
  `form_avg_finish_s5`, `form_trend`, `form_dnf_rate_s8`,
  `hist_positions_gained_s5`, `hist_grid_finish_consistency_s5`, and
  `form_avg_finish_similar_circuit_s5`.
- `driver_circuit_avg_finish_prior` keyed "this circuit" by round NUMBER, not
  real identity — confirmed round 13 = Hungarian GP (2022) vs Italian GP /
  Monza (2026), completely different tracks. The same round-based lookup bug
  also affected `build_circuit_features` (`circuit_overtaking_index`,
  `circuit_is_street`, `circuit_sc_probability`) and
  `build_circuit_similarity_features`'s street/non-street bucketing — all
  three fixed via a new `(Year, Round) -> circuit_key` calendar
  (`load_circuit_calendar` / `build_circuit_calendar` in `pipeline/features.py`,
  cached to `data/circuit_calendar.csv`), matched against `circuits.csv`'s
  `circuit_key` via fastf1's `Location` field (not the `location` column,
  which holds the venue's proper name, e.g. "Hungaroring" — a naming trap
  worth remembering).
- Separately: 3 of the 4 "history" family features
  (`driver_prior_season_avg_finish`, `driver_prior_season_dnf_rate`,
  `team_prior_season_points_rank`) were already **complete dead constants**
  (1 unique value across all 705 rows) because the `Year += 1` join requires
  the immediately-preceding year to exist, and the dataset only has 2022 and
  2026 — 2023–2025 are entirely absent. Fixed to require the preceding year
  to be both present AND in the same regulation era, making the (currently
  still all-neutral) behavior intentional and correct-by-design rather than
  an accidental side effect of the data gap, so it activates correctly once
  genuine same-era multi-year data exists.

**Fix**: a new `config.yaml: pipeline.regulation_reset_seasons: [2022, 2026]`
(domain knowledge, human-maintained) and a `regulation_era()` helper
(`pipeline/features.py`). Every rolling/expanding "form" feature is now
additionally partitioned by era — resets at a real regulation change, but
(unlike a blanket per-season reset) still bridges an ordinary same-era season
boundary once one exists (verified in `tests/test_features.py`). Registry
bumped to v2026.4.

**Re-measured impact** (`scripts/rolling_backtest.py`, full honest
walk-forward, retrains per round):

| | Before this fix (same day, earlier) | After |
|---|---|---|
| Rounds backtested | 28 | 24 (4 rounds lost all rows to the stricter dropna at era starts — an honest cost, not a bug) |
| Position MAE | 3.80 | **3.62** |
| Spearman | 0.622 | **0.640** |
| Winner hit-rate | 0.36 | **0.58** |
| Training rows surviving dropna (`train --model position`) | 631/705 | 559/705 |

Genuine improvement on every metric, even measured on a harder (smaller,
early-round-heavy) set of test rounds — though the round-count change means
this isn't a perfectly apples-to-apples comparison; re-check after more
2026 rounds complete. `lasso` feature selection now also correctly drops
`driver_circuit_avg_finish_prior` and `form_avg_finish_similar_circuit_s5`
(both currently inert — no same-era multi-year data yet for either) rather
than keeping a feature that was previously contributing spurious, wrong-era
signal.

**Known remaining gap, flagged not fixed**: `f1_results_simple.csv` only has
2022 and 2026 — 2023–2025 were never fetched (or the fetch didn't complete).
Filling that gap would NOT help 2026 predictions directly (2023-2025 share
2022's regulation era, not 2026's — 2026 is necessarily cold-start under a
correct era-aware design regardless of what other years exist); it would
only enrich same-era continuity for evaluating the 2022-2025 era itself.
Worth a follow-up fetch investigation, but don't expect it to fix 2026's
cold start.

> **Status (2026-09-13, re-verified): Phases A–E code-complete; B5 not actually started; Phase F tests green (148 passed).**
>
> The 2026-09-12 status block below (kept in git history) reported numbers that
> turned out to be **stale and, in one case, produced by two bugs in the
> measurement harness itself** (`scripts/backtest.py` — see "Harness bugs found
> 2026-09-13" below). This block replaces it with numbers reproduced today
> after fixing those bugs. The lesson: a "done ✅" status table is a claim
> about the code, not a substitute for rerunning the measurement.
>
> | Phase | Status | Notes |
> |-------|--------|-------|
> | A — bug fixes + harness | ✅ Done | `pytest` green (148 passed) |
> | B1–B4, B6 — multi-season infra | ✅ Done | 2022+2026 data, `race_seq`, history priors, context.csv |
> | B5 — practice sessions | ❌ **Not implemented** | No `FP2`/`FP3`/`practice`/`fp_long_run`/`fp_short_run` symbol exists anywhere in `pipeline/`. The 2026-09-12 note claiming "scaffolding designed, family wired into the builder chain" does not match the code — there is nothing to activate by refetching. |
> | C1 — `grid_penalty` | ✅ Done | Merged from `QualifyingPosition` |
> | C2 — racewin blend | ✅ Done | Opt-in family; off by default (leakage caveat) |
> | C3 — circuit-similarity form | ✅ Done | `form_avg_finish_similar_circuit_s5`, registry v2026.3 |
> | D1 — `positions_gained` target | ✅ Done | Implemented; `position` stays default until more data pays off |
> | D2 — learning-to-rank head | ✅ Done | `XGBRanker` blend, saved as `race_ranking_pipeline.pkl` |
> | D3 — optuna search | ✅ Done | Gated by `config.features.tune: none\|quick\|full` |
> | D3 — `feature_selection: vif` switch | ❌ **Re-measured, reverted** | Re-tested on multi-season data 2026-09-13: `vif` still loses to `lasso` (see table below). Default stays `lasso`. |
> | E1 — DNF classifier | ✅ Done | `dnf_pipeline.pkl`, CV AUC stored in `metrics.json` |
> | E2 — Monte-Carlo simulation | ✅ Done | `pipeline/simulate.py`, 5,000 trials, noise calibrated from holdout MAE |
> | E3 — surface probabilities | ✅ Done | `predict_race()` / `main.py` / `app.py` all emit P(win)/P(pod)/P(pts)/band |
> | F — tests + docs | ✅ Done (tests); docs were stale | 148 passed, 0 failed as of 2026-09-13 (up from the 132 claimed 2026-09-12 — 2 of those 132 were actually failing before today's fixes) |
>
> **`feature_selection` re-measurement (2026-09-13, multi-season data, 705 rows / 631 after history dropna):**
>
> | metric | `lasso` (default) | `vif` | winner |
> |---|---|---|---|
> | forward-chain MAE | **2.8394** | 2.9805 | lasso |
> | forward-chain Spearman | **0.7809** | 0.7446 | lasso |
> | forward-chain winner_logloss | 1.9425 | **1.8776** | vif (marginal) |
> | forward-chain podium_brier | **0.0908** | 0.0954 | lasso |
> | forward-chain points_brier | **0.1375** | 0.1452 | lasso |
> | cv_mae_mean | **3.2544** | 3.365 | lasso |
> | features kept | 16 | 29 | — |
>
> `vif`'s looser pruning (drops only collinear features, not weak ones) keeps
> nearly double the features but overfits relative to `lasso` at ~630 rows.
> Same conclusion as the original single-season measurement — multi-season
> data did not flip it. `lasso` stays default.
>
> **Harness bugs found 2026-09-13 (both fixed in `scripts/backtest.py`):**
> 1. `predict_race()` unconditionally applies whatever race is sitting in
>    `data/upcoming_qualifying.csv` onto the round being predicted, matched by
>    driver name. `main.py predict-race` self-guards against this (it checks
>    the file's `Race` column first and refetches if it doesn't match), but
>    `scripts/backtest.py` didn't — so backtesting a past round after fetching
>    the *next* round's qualifying silently overwrote that past round's real
>    grid with the wrong one. Fixed: the script now checks the file's round
>    before use and ignores it (falling back to the round's real historical
>    grid) if it doesn't match, with a printed note.
> 2. The actual-results lookup filtered only on `Race`, not `(Year, Race)`.
>    Since B1 made round numbers repeat across seasons (2022 and 2026 both
>    have a Round 13), this pooled a different season's result for any driver
>    who raced in both — a many-to-many join that inflated `n_drivers` (36
>    instead of ~22) and corrupted every metric. Fixed: filter now includes
>    `Year == season`.
> 3. Also found: `pipeline/analyze.py:_assemble_frame` silently dropped
>    `race_seq` and `Year` from the feature-quality frame, breaking
>    `test_race_seq_ordering_never_leaks_future` (one of Phase F's own
>    leakage-safety regression tests). Fixed by keeping both columns
>    alongside the model features.
>
> **Honest walk-forward backtest, regenerated 2026-09-13 (28 rounds, 2022+2026, race_seq order):**
> - Season MAE **3.796**, Spearman **0.622** — *worse* than the 2026-09-12
>   doc's claimed 2.839/0.781, because that number was from a stale 9-round
>   (2026-only) run of `outputs/rolling_backtest.md` that was never
>   regenerated after the multi-season data landed. Early 2022 rounds (little
>   accumulated history) drag the honest average down; this is expected, not
>   a regression from any change made today.
> - **Monza (Italian GP, Round 13) honest holdout** — model trained only on
>   the 30 races before Monza, never saw its outcome: Spearman **0.703**, MAE
>   **3.27**, winner_logloss 1.878. Predicted podium Russell/Antonelli/Norris
>   (actual: Antonelli/Russell/Verstappen) — 2/3 podium overlap, winner not
>   exactly right (Antonelli picked 2nd-favourite behind Russell). Leclerc's
>   retirement (P22 from a top-5 predicted finish) was the single biggest
>   miss and is unforeseeable from form/history features — a genuine DNF, not
>   a model failure.
> - The original 2026-09-12 narrative below (Norris/Antonelli/Hamilton top-3,
>   28%/38.3% win probabilities) came from `scripts/backtest.py --round 13`
>   using the **in-place, full-season-trained model** — a "quick directional
>   check" per that script's own docstring, not the fully-honest holdout. It
>   is kept below for the record but should not be read as the accuracy
>   number for Monza.

## Context

The leakage-free feature redesign (`docs/feature-engineering-redesign-plan.md`, 2026-09-09)
made the accuracy number *honest* but did not raise it. A fully honest round-13 (Monza)
holdout — train on rounds 1-12, predict 13 — gave **MAE 2.19 places, Spearman 0.869,
top-10 8/10, but podium 1/3 and the winner missed** (Antonelli predicted P4 finished P1;
Verstappen P5 finished P3). GroupKFold-by-Race MAE is ≈ 3.7. The hypothesis battery shows
the model leans almost entirely on `GridPosition` + `Team`/`Driver` +
`team_form_avg_finish_s5` + `form_trend`; the newer families add little, and the circuit
family adds nothing at all in single-season data.

Two live bugs were found while investigating (see A1). This plan takes **all tiers**, and
**reverses the redesign's "2026-only" decision** — multi-season history is the single biggest
lever on raw accuracy, especially for early-season rounds with no in-season form. Output
becomes **both** a point-estimate finishing order (headline) **and** per-driver probabilities
(`P(win)`, `P(podium)`, `P(points)`, expected position + 10-90 band) from a Monte-Carlo
simulation.

Implementation is sequenced A→F; each phase ends green on `pytest` + the feature-quality gate
and is measured on the new rolling backtest (Phase A) so every later change is judged on true
forward-chain performance.

---

## Phase A — bug fixes + measurement harness ✅ Done

### A1. Fix the stale `completed_statuses` (live bug) ✅
`config.yaml:constants.completed_statuses` lists old Ergast strings (`"+1 Lap"`…`"+5 Laps"`);
fastf1 3.8 returns `Finished / Lapped / Retired / Did not start`. Today
`is_dnf = ~Status.isin(completed_statuses)` flags **151/286 rows (53%) as DNF** vs a true
mechanical/incident rate of **60 (21%)** — 91 *Lapped-but-classified* finishers are wrongly
counted, corrupting `form_dnf_rate_s8`, `driver_dnf_rate_todate`, `team_reliability_todate`.

- Add `pipeline/clean.py:normalize_status(series) -> canonical {finished, lapped, retired, dns, dsq}`
  mapping both the old and new vocabularies; write a `status_canon` column in
  `clean_results` / `clean_qualifying`.
- Set `completed_statuses: ["Finished", "Lapped"]` (Lapped = classified finish, not a DNF).
- Point `create_historical_features` / `build_reliability_features` at `status_canon`.
- Re-run `main.py clean && main.py features && main.py train --model position`.

### A2. `scripts/rolling_backtest.py` — season scorecard ✅
`scripts/backtest.py` is single-round and reuses the all-data model. New script: for each
round r ≥ k, retrain on everything before r (all prior seasons + current-season rounds < r),
predict r, collect `position_mae`, `spearman`, `podium_hit`, `winner_logloss`,
`points_brier`; emit `outputs/rolling_backtest.md` + a plot. Reuses
`pipeline.train.train_position_model` refactored to accept an in-memory train/predict split,
and `pipeline.feedback.score_prediction`. This is the yardstick for every later phase.

### A3. Sharp-end metrics in `feedback.py` / `metrics.json` ✅
Add `winner_logloss` and `podium_brier` to `pipeline/feedback.py:score_prediction` and the
saved `position` metrics — MAE ≈ 2.2 hides that the podium is where the model is wrong.

---

## Phase B — multi-season data foundation

### B1. Multi-season fetch ✅
`pipeline/fetch.py`: `collect_multiple_races` loops `config.pipeline.history_start_season ..
config.pipeline.season` (new config key, default 2022). Every raw/cleaned/feature CSV already
carries `Year`; make **every** merge and group key `(Year, Race, Driver)` — currently
`clean.py:merge_qualifying_into_results` uses `(Year, Race, Driver)` (ok) but
`create_historical_features` / the family builders group by `Driver` and sort by `Race` only.

### B2. Cross-season ordering ✅
Add a monotonic `race_seq` = dense rank of `(Year, Race)` and sort every rolling/expanding
feature by `race_seq` (not `Race`), so 2024-round-13 precedes 2025-round-1. `as_of_round`
becomes `as_of_seq`. Touches `pipeline/features.py` (`create_historical_features`,
`_shifted_roll`, `_expanding_pre_race_rate`, `build_team_features`,
`build_championship_features`) and `pipeline/predict.py` / `evaluate.py` / `backtest.py`
cutoffs.

### B3. Season boundaries for the "to-date" and championship features ✅
`driver_dnf_rate_todate` / `team_reliability_todate` / `driver_points_gap_to_leader_before`
must reset per season (championship points don't carry over). Add a `within_season` guard to
those builders; keep the *rolling* form features (`form_avg_finish_s5`, etc.) spanning the
season boundary (recent form is recent form).

### B4. Frozen prior-season priors (new `history` family in `feature_registry.py`) ✅
Causal — computed from completed prior seasons, known before round 1:
`driver_prior_season_avg_finish`, `team_prior_season_points_rank`,
`driver_prior_season_dnf_rate`, `driver_circuit_avg_finish_prior` (driver-at-this-circuit,
keyed via the round→circuit map — extend `data/circuits.csv` to every season's calendar or
derive from fastf1 `EventName`). New `build_history_features(df, config)` +
`REGISTRY_VERSION` bump. A blend weight `w = min(1, current_round / 8)` shifts reliance from
priors to in-season form as the year progresses (applied as a feature, not in code paths).

### B5. Practice-session pace (new `practice` family) ⏸ Deferred
`pipeline/fetch.py`: also pull `fastf1.get_session(year, rnd, "FP2")` (+ FP3). Derive
`fp_long_run_pace_rank` (median of green stint laps with tyre life > 5, per driver, ranked),
`fp_short_run_pace_rank`, `practice_vs_quali_delta`. FP2 long-run pace is among the best
pre-race predictors and is orthogonal to single-lap quali. Only Qualifying + Race are in
`fastf1_cache` today — this is a real refetch (respect `api_sleep_seconds`).

> **Note (2026-09-12, corrected 2026-09-13):** This note originally claimed the
> fetch/clean/features/registry scaffolding for B5 was designed and the
> `practice` family "wired into the builder chain." That is not true of the
> current code: `grep -rn "FP2\|FP3\|fp_long_run\|fp_short_run\|practice_vs_quali" pipeline/`
> returns zero matches. Nothing needs activating by refetching — the family,
> its builder function, and its registry entries do not exist yet. B5 is
> fully unimplemented, not implemented-but-dormant.

### B6. Curated pre-race context — `data/context.csv` (like `circuits.csv`) ✅
Hand-maintained per `(Year, Race)`: dominant tyre allocation, wet-race flag from the *pre-race
forecast*, major team upgrade flag, rookie flag per driver. Loaded by a
`build_context_features` builder; missing rows → neutral defaults + warning.

---

## Phase C — feature additions on the new data

### C1. `grid_penalty` ✅
`GridPosition` in the data is already **post-penalty** (58/286 rows differ from
`QualifyingPosition`, range −4…+12). Add `grid_penalty = GridPosition − QualifyingPosition`
to the `quali` family (`feature_registry.py` + `build_quali_features`); carry
`QualifyingPosition` through `merge_qualifying_into_results` (recoverable from
`QualifyingPerformance`). Captures recovery drives the model is currently blind to.

### C2. Blend the existing `racewin` classifier ✅
`pipeline/model_registry.py` already loads a trained `racewin` model wired only to the manual
`/predict` endpoint. Add its `predict_proba` as a `context`-family feature to the position
model.

### C3. Circuit-similarity form ✅
`form_avg_finish_similar_circuit_s5` — rolling finish over prior races at circuits sharing
`circuit_is_street` / an overtaking-index bucket. Partially rescues the circuit family within
a season and compounds with B4's driver-at-circuit prior.
Implemented in `pipeline/features.py:build_circuit_similarity_features`, registered in
`feature_registry.py` (family `circuit_sim`), enabled in `config.yaml`.

---

## Phase D — modelling framing

### D1. Predict `positions_gained`, not raw `Position` ✅
Target = `GridPosition − Position`; reconstruct `finish = GridPosition − Δ̂`, then re-rank.
Removes the dominant grid-position variance so the model must learn pace / racecraft /
strategy. Small change in `pipeline/train.py` + `pipeline/predict.py`; the racecraft family
already has the historical version.

> **Note:** Measured WORSE on the single-season dataset (MAE 3.87 vs 3.68). Default remains
> `position`. Switch via `config.features.position_target: positions_gained` once more
> multi-season data is available.

### D2. Learning-to-rank head ✅
Add `XGBRanker(objective="rank:ndcg")` with `group` = rows-per-Race, relevance
`= grid_size − Position`, as a second model. CV scorer → Spearman / NDCG. Keep the D1
regressor; the final order is a **blend** (rank-average of the two), chosen on the rolling
backtest. `xgboost==2.0.2` supports it.

### D3. Hyperparameter search under GroupKFold ✅
`config.models.position` is fixed (`n_estimators=1000, lr=0.01, depth=5`) — over-parameterised
for the current ~220 rows (less so after Phase B). Add an `optuna` search in `train.py` gated
by `config.models.tune: none|quick|full`, optimising forward-chain MAE; expect
`max_depth 2-4`, `reg_alpha/reg_lambda > 0`.

> **`feature_selection: lasso → vif` — tried, reverted (2026-09-13):** flipped
> the default to `vif` per this section's original suggestion, retrained on
> the current 705-row multi-season dataset, and re-measured against the
> `lasso` baseline. `vif` lost on 5 of 7 metrics (forward-chain MAE 2.98 vs
> 2.84, Spearman 0.745 vs 0.781) while keeping almost double the features (29
> vs 16) — its pruning only removes collinear features, not weak ones, so at
> ~630 rows it overfits relative to `lasso`'s more aggressive selection. Same
> conclusion as the original single-season measurement; reverted to `lasso`.
> See the status block at the top of this document for the full table.

---

## Phase E — probabilistic output (Monte-Carlo)

### E1. `train_dnf_model` ✅
Gradient-boosted classifier for canonical `retired`, using the now-correct reliability
features + `circuit_sc_probability` + prior-season DNF rate + team/driver incident history.
Saved as `models/dnf_pipeline.pkl` via `model_registry`.

### E2. `pipeline/simulate.py` ✅
For the target race, run N≈5000 trials: sample each driver's DNF from `P(DNF)`; draw a
finishing score from the D1/D2 blend plus a residual noise term calibrated on holdout errors;
resolve to a full order per trial. Aggregate → `P(win)`, `P(podium)`, `P(points)`,
`E[position]`, 10-90 band.

### E3. Surface both outputs ✅
`pipeline/predict.py:PredictionResult` gains `win_probability`, `podium_probability`,
`points_probability`, `p10`/`p90` per driver. Point rank stays the headline;
`app.py` (`/predict_next_race`) and `main.py predict-race` add the probability columns.
Probabilities are computed via `simulate_race_df` (5,000 Monte-Carlo trials) and attached to
every `DriverForecast`. The API response includes `simulation_n_trials` in the top-level body.

---

## Phase F — gate & docs ✅ Done

- **`tests/test_feature_quality.py`** — added:
  - `test_race_seq_ordering_never_leaks_future`: property test verifying `race_seq` is
    monotone in `(Year, Race)` order and `avg_position_last` at round R does not equal
    the future-round mean for any driver.
  - `test_season_reset_features_are_zero_at_round_1`: verifies `driver_dnf_rate_todate`
    and `driver_points_gap_to_leader_before` are NaN/0 at each season's round 1.
- **`tests/test_train.py`** — added:
  - `test_ranker_group_integrity`: verifies `_ranker_group_sizes` produces exactly one
    contiguous group per race with no cross-race mixing.
  - `test_simulate_output_shapes`: verifies `simulate_race_df` returns correct columns,
    probabilities ∈ [0,1], sum(P(win)) ≈ 1, and p10 ≤ p90 for all drivers.
- **`tests/test_api.py`** — updated `test_predict_next_race_parity` to include the new
  E3 fields (`simulation_n_trials`, per-driver probability columns).
- `data/feature_manifest.json` — updated automatically on `main.py features` (v2026.3).
- This document updated to reflect final status.

## Files (primary)

| File | Phase |
| --- | --- |
| `config.yaml`, `pipeline/config_loader.py` | A1, B1, D3 (new keys: `completed_statuses`, `history_start_season`, `models.tune`) |
| `pipeline/clean.py` | A1 (`normalize_status`), C1 (`QualifyingPosition`) |
| `pipeline/fetch.py` | B1 (season loop), B5 (practice sessions) |
| `pipeline/features.py` | B2 (`race_seq`), B3 (season resets), B4/B5/B6/C1/C3 builders |
| `pipeline/feature_registry.py` | B4/B5/B6/C1/C2/C3 new features + `REGISTRY_VERSION` |
| `pipeline/train.py` | A2 hook, D1 target, D2 ranker, D3 tuning, E1 DNF model |
| `pipeline/predict.py`, `pipeline/evaluate.py`, `scripts/backtest.py` | B2 cutoffs, D1/D2 blend, E3 |
| `pipeline/simulate.py` | E2 (NEW) |
| `pipeline/analyze.py` | B2 (`as_of_seq` in raw/clean frames) |
| `scripts/rolling_backtest.py` | A2 (NEW) |
| `data/circuits.csv`, `data/context.csv` | B4 (all-season calendars), B6 (NEW) |
| `tests/test_*.py` | F |

## Verification

### Commands (run after any refetch/retrain)

```bash
venv/bin/python main.py clean
venv/bin/python main.py features
venv/bin/python main.py train --model position
venv/bin/python scripts/rolling_backtest.py        # → outputs/rolling_backtest.md
venv/bin/python -m pytest -q                        # all tests
venv/bin/python -m pytest -m slow -q               # feature-quality gate (needs features CSV)
venv/bin/python main.py predict-race               # shows P(win)/P(pod)/band per driver
```

### Measured results (2026-09-12 — superseded, kept for the record)

This table was generated once, right after the multi-season retrain, and never
regenerated — by 2026-09-13 it no longer matched what `outputs/rolling_backtest.md`
and `scripts/backtest.py` actually produce (see the status block at the top of
this document for the corrected numbers and the two harness bugs that were
found). It is left here as a historical snapshot, not current fact.

| Metric | Before (A, single-season) | After (B–E, 2022+2026), as reported 2026-09-12 | As re-measured 2026-09-13 |
|--------|--------------------------|------------------------|------------------------|
| Training rows | 286 | 705 (+630 after drop) | 705 (unchanged) |
| Features | 28 (v2026.2) | 30 (v2026.3) | 16 (`lasso`, unchanged model) |
| Forward-chain MAE (season, honest walk-forward) | 3.68 (9 rounds, 2026-only) | 2.839 (claimed, not actually regenerated) | **3.796** (28 rounds, 2022+2026, regenerated 2026-09-13) |
| Forward-chain Spearman (season) | 0.644 | 0.781 (claimed) | **0.622** (regenerated) |
| pytest | 130 passed | "132 passed, 0 failed" (claimed) | 148 passed, 0 failed (2 of the previously-claimed-132 were actually failing until fixed 2026-09-13) |

The apparent regression from 2.84→3.80 MAE is not a real regression introduced
by any change made on 2026-09-13 — it's the difference between a 9-round,
current-season-only rolling backtest and the honest 28-round backtest across
both seasons (early 2022 rounds have little accumulated history and are
harder to predict, dragging the average down). No model or feature code
changed between the two numbers; only the measurement's scope did.

**Monza 2026 (Round 13, Italian GP) — two different numbers, don't conflate them:**

1. **"Quick check" (2026-09-12 narrative below) — uses the in-place, full-season-trained model.**
   Per `scripts/backtest.py`'s own docstring this is a *directional* check only,
   because the model has already seen Monza's outcome during training (it was
   trained on all 705 rows, Monza included). Kept for the record:

   | # | Driver | Predicted | P(win) | P(pod) | Band | Actual |
   |---|--------|-----------|--------|--------|------|--------|
   | 1 | L Norris | 1.00 | 38.3% | 76.3% | [1–5] | P4 |
   | 2 | **K Antonelli** | 2.00 | **28.0%** | 67.0% | [1–6] | **P1 ✅** |
   | 3 | L Hamilton | 4.00 | 11.2% | 44.4% | [1–8] | P6 |
   | 4 | C Leclerc | 4.50 | 9.3% | 38.2% | [2–9] | P22 (retired) |
   | 5 | G Russell | 5.00 | 7.4% | 31.9% | [2–12] | **P2 ✅** |
   | 9 | M Verstappen | 9.50 | 0.2% | 1.2% | [8–21] | **P3** |

2. **Fully honest holdout (2026-09-13, this document's real answer for "how good is the model at predicting Monza") — a model retrained on only the 30 races before Monza, which never saw its outcome:**

   | Pred rank | Driver | Predicted pos | Actual |
   |---|---|---|---|
   | 1 | G Russell | 1.88 | 2 |
   | 2 | K Antonelli | 2.38 | 1 |
   | 3 | L Norris | 3.81 | 4 |
   | 4 | L Hamilton | 4.09 | 6 |
   | 5 | O Piastri | 5.91 | 5 |
   | 6 | C Leclerc | 7.65 | 22 (retired) |
   | 12 | M Verstappen | 12.29 | 3 |

   Spearman **0.703**, MAE **3.27**, winner not correct (Russell picked ahead
   of the actual winner Antonelli), podium overlap 2/3. Leclerc's retirement
   is the single largest miss and is not a model failure — a mechanical DNF
   from a top-5 form-based prediction is unforeseeable from the available
   features. Verstappen's podium finish (predicted 12th) is the other real
   miss — his recent form going into Monza was poor, and the model has no way
   to know a competitive car update or race-specific pace jump was coming.

The gap between these two numbers (0.78 win-probability narrative vs 0.70
Spearman honest holdout) is exactly why `scripts/backtest.py` prints the
`model trained_at` timestamp and warns in its own docstring — always check
whether the loaded model's training data includes the round being evaluated
before trusting a single-round backtest as an accuracy claim.
