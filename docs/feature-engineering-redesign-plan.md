# Robust, leakage-free feature engineering for the F1 position model

## Context

The hypothesis-testing analysis added in `pipeline/analyze.py` /
`scripts/feature_hypothesis_tests.py` (report:
`outputs/feature_hypothesis_tests.md`) found that the current feature set is
**mostly leaky or redundant**:

- `PositionChange` is exactly `GridPosition - Position` (identity R² = 1). Every
  `create_historical_features` rolling window is **unshifted** (includes the race
  being predicted). `driver_win_rate` / `team_reliability` are whole-season
  aggregates that include the target row. Strip the leaky trio and the position
  model's honest GroupKFold-by-Race CV MAE goes **1.21 → 3.59 places**.
- After removing leakage, **only `QualifyingPerformance` clears every bar**
  (univariate q_perm < 0.001, cluster-robust OLS p = 0.011, permutation
  importance p = 0.004). `GridPosition` carries the same signal (VIF ≈ 20 vs
  quali — `QualifyingPerformance` is literally `QualifyingPosition / grid_size`).
  `form_trend` is a weak secondary signal.
- The ~11 historical rollups (`avg_position_last`, `best_position_last`,
  `podiums_last`, `wins_last`, `points_last`, `avg_grid_last`,
  `avg_positions_gained`, `dnf_last`, `reliability_rate`, `avg_quali_time`,
  `avg_gap_to_pole`) are pairwise collinear (VIF 8–30), non-significant in the
  multivariable models, and dropping any one does **not** raise CV MAE.
- The raw per-race **pace** signal `GapToPole` exists in the data but is never a
  direct model feature — only its noisy 6-race rolling mean is.

**Goal.** Rebuild feature engineering so every feature is causal (computable
strictly from data *before* the race), prune the redundant rollups, add a small
set of genuinely new orthogonal signals, and switch training to honest
GroupKFold + forward-chaining CV. Re-validate with the analysis tooling already
in the repo and gate on it.

**Decisions (from planning Q&A).**
- **2026 data only.** 2026 is a regulation reset; pre-2026 cars are a different
  formula. No multi-season fetch. Robustness comes from leakage-free features +
  honest CV, not more rows. This rules out driver-at-circuit history and
  multi-year priors — features must work within one growing season (~13–24
  races).
- **Full redesign**: declarative feature registry, leakage-safe `features.py`
  rebuild, new feature families, GroupKFold + forward-chain training, a
  re-validation gate — *plus* the prune and the top new features.
- Small-n discipline throughout: target **12–18 features**, hard VIF cap, L1 /
  feature-selection step, forward-chain holdout as the headline metric.

## Design principles

1. **As-of-round discipline.** Every feature is a function of rows with
   `Race < r` (or non-target columns of row `r`, i.e. grid/quali which are known
   pre-race). One code path for train and predict.
2. **No target-derived columns.** `PositionChange` is deleted, not shifted.
3. **Shift-before-roll.** Per driver sorted by `Race`: `.shift(1)` before every
   `.rolling(window, min_periods=1)`. `form_trend` keeps its `.shift(3)`.
4. **Expanding pre-race aggregates** for rates (cumulative over races strictly
   before the current one; first appearance → NaN).
5. **Parsimony + orthogonality.** Prefer one representative per collinear group;
   VIF < 10 enforced; a feature-selection step in the pipeline.
6. **Declared provenance.** Each feature carries a family + `as_of_safe` flag +
   leak note in a registry, mirroring `analyze.FeatureSpec`.

## Reuse (already in the repo)

- `pipeline/analyze.add_shifted_history` — prototype of shift-before-roll for all
  11 rollups. Promote this logic into `create_historical_features`.
- `pipeline/analyze._expanding_pre_race_rate` — prototype of the expanding
  pre-race rate (race-aware, teammate rows don't leak). Promote it.
- `pipeline/analyze._grouped_cv_mae` / `GroupKFold(by Race)` — the training CV
  pattern.
- `pipeline/analyze.leakage_report`, `run_feature_analysis` — the validation gate.
- `pipeline/model_registry.model_feature_columns` — feature list already read off
  the fitted pipeline; keep that contract so `predict.py` needs no feature edits.
- `pipeline/clean.py:merge_qualifying_into_results`, `clean_qualifying` — where
  new quali-derived columns attach.

## Phase 1 — feature registry (`pipeline/feature_registry.py`, NEW)

Declarative single source of truth.

```python
@dataclass(frozen=True)
class Feature:
    name: str
    family: str            # "quali" | "form" | "racecraft" | "team" | "reliability"
                           # | "circuit" | "championship" | "context"
    kind: str              # "numeric" | "categorical"
    as_of_safe: bool       # False => must be excluded from a true pre-race forecast
    build: str             # dotted path or key into the builder dispatch
    note: str

REGISTRY: list[Feature] = [ ... ]

def enabled_features(config) -> list[Feature]      # honours config.features.families_enabled
def numeric_names(features) / categorical_names(features)
def as_of_unsafe(features) -> list[str]            # for the forecast-time guard
```

`analyze.feature_specs()` is refactored to read `REGISTRY` (raw frame = "unshifted
regression guard", clean frame = registry as-is) so the two modules can't drift.

## Phase 2 — leakage-safe `pipeline/features.py` rebuild

- `create_historical_features(df, n_previous, completed_statuses, as_of_round=None,
  shift=1)`:
  - apply `.shift(shift)` before every `.rolling(...)` (fold in
    `analyze.add_shifted_history`);
  - **remove** the `PositionChange` write from `engineer_result_features`;
  - keep `race_winner` / `podium_finish` / `points_finish` (targets only).
- Replace `_compute_driver_win_rate` / `_compute_team_reliability` with
  `_expanding_pre_race_rate`-based builders producing:
  `driver_win_rate_todate`, `team_reliability_todate` (season-to-date, pre-race).
- Family builders (each a pure `df -> Series`, unit-tested for as-of-safety):
  `build_quali_features`, `build_form_features`, `build_racecraft_features`,
  `build_team_features`, `build_reliability_features`, `build_circuit_features`,
  `build_championship_features`.
- `run_feature_engineering` assembles only `enabled_features(config)`, writes
  `data/f1_results_features.csv` **plus** a sidecar
  `data/feature_manifest.json` (registry version, season, row count, per-feature
  null rate, `as_of_unsafe` list).

### The new feature set (~15, all 2026-feasible)

| family | feature | why / replaces |
| --- | --- | --- |
| quali | `grid_position` | proven signal (raw) |
| quali | `quali_gap_to_pole_pct` = `GapToPole / pole_time * 100` | **pace**, not rank — currently unused directly |
| quali | `quali_gap_to_teammate_s` | isolates car vs driver |
| quali | `quali_beat_teammate` (0/1) | robust head-to-head |
| quali | `q3_reached` (0/1) | top-10 pace tier |
| form | `form_avg_finish_s5` (shift-1, roll 5) | one representative of the collinear rollups |
| form | `form_trend` (keep, already `.shift(3)`) | weak secondary signal, retained |
| form | `form_dnf_rate_s8` (shift-1, roll 8) | reliability trend, replaces `dnf_last`/`reliability_rate` |
| racecraft | `hist_positions_gained_s5` (shift-1 mean of `grid - finish` over prior races) | **causal** replacement for leaky `PositionChange` |
| racecraft | `hist_grid_finish_consistency_s5` (shift-1 std) | volatility |
| team | `team_form_avg_finish_s5` (both cars, shift-1) | car strength |
| team | `team_quali_pace_rank` (this round's mean team `GapToPole`, rank) | pre-race car pace |
| reliability | `driver_dnf_rate_todate` (expanding, pre-race) | replaces season-wide `driver_win_rate` |
| reliability | `team_reliability_todate` (expanding, pre-race) | replaces leaky `team_reliability` |
| circuit | `circuit_overtaking_index`, `circuit_is_street`, `circuit_sc_probability` | from `data/circuits.csv` (static, hand-curated ~24 rows) |
| championship | `driver_points_gap_to_leader_before` | motivation/pressure, fully causal |
| context | `Driver`, `Team` | categoricals, kept |

Weather / track-temp are **not** added to the position model — not known at true
forecast time. If wanted later they go behind `as_of_safe=False` and are excluded
by the forecast guard.

Drop entirely: `PositionChange`, `best_position_last`, `avg_grid_last`,
`avg_positions_gained`, `podiums_last`, `wins_last`, `points_last`, `dnf_last`,
`reliability_rate`, `avg_quali_time`, `avg_gap_to_pole`, whole-season
`driver_win_rate` / `team_reliability`.

## Phase 3 — `data/circuits.csv` (NEW static reference)

~24 rows keyed by a normalized circuit name (map from fastf1 `EventName`).
Columns: `circuit_key, is_street (0/1), overtaking_index (1–5, curated),
sc_probability (0–1, curated from public history), lap_count`. Loaded in
`build_circuit_features`; a missing circuit → median fill + a logged warning.
`pipeline/fetch.py` already has `EventName` via `get_event_schedule`; add the
round→circuit_key map to the results save path.

## Phase 4 — honest training (`pipeline/train.py`)

`train_position_model`:
- feature list from `feature_registry.numeric_names/categorical_names` (not the
  hand-list);
- **CV**: `GroupKFold(n_splits=config.features.cv_splits)` grouped by `Race`,
  replacing `cross_val_score(cv=5)`;
- **holdout**: forward-chaining — train on rounds `≤ N-k`, test on the last `k`
  (`k = config.features.holdout_rounds`, default 3). Report this MAE/Spearman as
  the headline (the random `train_test_split` is removed);
- pipeline gains an optional selection step:
  `SelectFromModel(Lasso)` on the scaled numerics, or greedy VIF-drop > 10,
  toggled by `config.features.feature_selection`;
- `_save_metrics("position", ...)` also records `feature_manifest` version, the
  kept feature list, forward-chain MAE, and mean VIF.

`config.yaml` gains:
```yaml
features:
  lookback_races: 6
  shift: 1
  families_enabled: [quali, form, racecraft, team, reliability, circuit, championship]
  cv_splits: 5
  holdout_rounds: 3
  vif_threshold: 10
  feature_selection: lasso        # lasso | vif | none
```

## Phase 5 — re-validation gate

- `scripts/feature_hypothesis_tests.py --frames both` on the new set
  (`raw` = `shift=0` regression guard, `clean` = registry).
- **Acceptance** (add as `tests/test_feature_quality.py`, marked slow):
  - `analyze.leakage_report(...)["identity"]` absent or R² < 0.5 (no target
    reconstruction);
  - every numeric feature VIF < `config.features.vif_threshold`;
  - ≥ 5 features with model-based `p_perm_importance < 0.05` **or**
    `delta_cv_mae > 0`;
  - no `FeatureSpec.leaky` true in the clean frame.
- Report the new forward-chain holdout MAE next to the documented baseline
  (leaky 1.2 / honest 3.6).

## Files

| File | Action |
| --- | --- |
| `pipeline/feature_registry.py` | NEW — declarative `Feature` list + selectors |
| `pipeline/features.py` | rebuild: shift-before-roll, expanding rates, drop `PositionChange`, family builders, manifest sidecar |
| `pipeline/analyze.py` | `feature_specs()` reads the registry (keep both modules in sync) |
| `pipeline/train.py` | GroupKFold + forward-chain holdout, selection step, registry-driven feature list, richer metrics |
| `pipeline/clean.py` | carry `pole_time` / teammate quali times through for the new quali features |
| `pipeline/fetch.py` | round → `circuit_key` map in the results save path (no new sessions fetched) |
| `data/circuits.csv` | NEW — static circuit metadata (~24 rows) |
| `config.yaml` | `features:` block |
| `tests/test_features.py` | expand: as-of-safety property test, no-`PositionChange` assertion, manifest schema |
| `tests/test_feature_quality.py` | NEW — the Phase-5 gate (slow) |
| `tests/test_train.py` | forward-chain split correctness, GroupKFold never shares a `Race` |

`pipeline/predict.py` unchanged — it already reads the feature list off the
fitted pipeline via `model_feature_columns`; add only a one-line guard that drops
any `as_of_unsafe` column if one ever enters the registry.

## Verification

```bash
cd "/run/media/rajay/New Volume1/Machine Learning/F1_predictor"
venv/bin/python -m pytest tests/test_features.py tests/test_train.py -q      # as-of-safety, no leakage
venv/bin/python main.py features                                            # writes CSV + feature_manifest.json; no PositionChange column
venv/bin/python main.py train --model position                             # logs GroupKFold CV + forward-chain holdout MAE
venv/bin/python scripts/feature_hypothesis_tests.py --frames both --plots   # gate: no leaky flags, VIF<10, >=5 real features
venv/bin/python -m pytest tests/test_feature_quality.py -q -m slow          # automated gate
venv/bin/python -m pytest -q                                                # nothing else broke
venv/bin/python scripts/backtest.py --round <latest>                        # end-to-end honest backtest on the new model
```

Expected: `feature_manifest.json` lists 12–18 features, all `as_of_safe`;
`create_historical_features` output has **no** `PositionChange`; forward-chain
holdout MAE reported (compare to honest baseline 3.6); leakage identity check
absent/low; all VIF < 10; `predict_race` still returns a full grid.

## Implementation status (2026-09-09)

All five phases landed. Suite green (`pytest -m "not slow"` + `pytest -m slow`).

**Honest headline numbers** (`models/metrics.json` → `position`):

| metric | value | reference |
| --- | --- | --- |
| GroupKFold(5, by Race) MAE | **3.74** ± 0.45 | honest baseline 3.59 |
| forward-chain holdout MAE (rounds 11–13) | **2.92** | leaky CV was 1.21 |
| forward-chain Spearman | **0.787** | — |
| mean VIF (deployed features) | **3.14** | cap 10 |
| identity-probe R² (`Position ~ all numeric`, in-sample) | **0.53** | was ≈1.0 with `PositionChange` |

**Deviations from the plan as written:**

- **`features:` config block** added in Phase 1, not Phase 4 — `feature_registry.enabled_features(config)` depends on it. New `FeaturesConfig` dataclass + validation in `config_loader.py`.
- **`grid_position` stays `GridPosition`** — it is a raw passthrough column already threaded through `clean.py` / `fetch.py` / `predict.py`'s quali override; renaming bought only churn.
- **Legacy `driver_win_rate` / `team_reliability` kept** in `engineer_result_features` — the race-win and lap-time models still consume them. The *position* model no longer selects them (it reads the registry). "Drop entirely" was scoped to the position feature frame.
- **`create_historical_features` kept** (name + legacy display columns like `avg_position_last`, used by `predict.py`'s recent-form panel) but converted to shift-before-roll with a `shift` param (`shift=0` = the old leaky behaviour, used by the analysis "raw" guard).
- **`predict.py` did change** (plan said "unchanged"): it now calls the shared `build_position_features` (one code path with training) and drops any `as_of_unsafe` column. `evaluate.py` moved to the same path.
- **18 model features** (top of the 12–18 band). `feature_selection: lasso` prunes to ~10 on the current 13-round data; the **circuit family and the season-to-date reliability rates get dropped by L1** — circuit metrics are ~constant within a race so a GroupKFold-by-race model cannot learn from them in a single season. They become useful only with multi-season data (already a follow-up below). Switch `feature_selection: none`/`vif` to keep them.
- **`leakage_report` reworked** for the post-redesign world: identity check is now an in-sample OLS R² probe (no `PositionChange` to test); `leaky_trio_cv_mae` → `shift_guard_cv_mae` (shift-0 vs production CV MAE, ~0.4-place inflation); rollup-shift table → history-family shift comparison; rate degeneracy → rate-variation check (`*_todate` rates now vary within group: 11 distinct values, vs the old constant-within-driver whole-season rates).
- **New files:** `pipeline/feature_registry.py`, `data/circuits.csv` (23 rows, 2026 calendar, keyed by round), `data/feature_manifest.json` (sidecar), `tests/test_features.py`, `tests/test_feature_quality.py` (the slow gate), `pytest.ini` (registers the `slow` marker). `.gitignore` gains `!data/circuits.csv`.

## Out of scope / follow-ups

- Multi-season data & driver-at-circuit history (needs pre-2026 fetch; revisit
  once the 2026 regs stabilise or if a later season shares them).
- Weather / safety-car *realised* features (not causal for a true forecast;
  only useful for post-hoc analysis).
- Lap-time model feature rework (this plan is the **position** model only).
