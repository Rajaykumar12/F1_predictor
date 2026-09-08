# Statistical hypothesis testing of the F1_predictor features

## Context

The repo only ever inspects features with a Pearson correlation heatmap
(`pipeline/visualize.py:143`, `notebooks/feature_and_eda.ipynb`) — no p-values, no
control for the data's structure, no check of whether a feature actually helps the
model. The goal is a proper, reproducible **hypothesis-testing analysis of the
model's features**: for every engineered feature, is it significantly associated
with race outcome, and does it contribute to the position model — accounting for
the confirmed **target leakage** and **repeated-measures** structure so the
conclusions aren't dominated by leaky columns.

Confirmed by exploration:
- Data `data/f1_results_features.csv`: 286 rows, one 2026 season, 13 rounds
  (~22 rows each), 23 `Driver`, 11 `Team`. Only nulls: `QualifyingPerformance` (3).
- Targets: `Position` (1–22, the position model's regression target),
  `podium_finish` (13.6%), `points_finish` (45.5%), `race_winner` (4.5%)
  (`pipeline/features.py:99-102`).
- Position model's 19-feature list built in `pipeline/train.py:198-206`; the
  `*_last`/`form_trend`/`avg_quali_time`/`avg_gap_to_pole` columns come from
  `create_historical_features(results, n_previous=6, completed_statuses=<6-val list>)`
  (`pipeline/features.py:107`), not the CSV.
- **Leakage**: `PositionChange` is exactly `GridPosition - Position`
  (`features.py:99`, R²≈1); all `create_historical_features` rolling windows are
  **unshifted** (include the current race) except `form_trend`;
  `driver_win_rate`/`team_reliability` are whole-season aggregates including the
  target row (`features.py:65-80`).
- **Non-independence**: each driver ~13 rows, each team ~26; within a race the 22
  `Position` values are a permutation of 1..22. Naive OLS/logit SEs and random
  K-fold CV are anticonservative.
- Env (`venv/bin/python`; note `venv/bin/pip` shebang is broken — use
  `venv/bin/python -m pip`): has `numpy 1.26.2`, `pandas 2.1.3`, `scipy 1.16.3`,
  `scikit-learn 1.3.2`, `xgboost 2.0.2`, `matplotlib`. **No** `statsmodels`,
  `seaborn`. No hypothesis-test code anywhere.

Decisions:
- **Targets**: all four (`Position`, `podium_finish`, `points_finish`, `race_winner`).
- **Depth**: full — univariate + multivariable inference + model-based importance.
- **Leaky features**: test the real 19-feature set (leakage flagged) **and** a
  parallel "clean" set (leakage removed / windows shifted).
- **Add `statsmodels`** — use it for cluster-robust OLS/Logit, VIF, GEE panel
  logistic, and a mixed-effects `Position ~ features + (1|Driver)` model.

Scope: read-only w.r.t. the pipeline; writes only to `outputs/`. `pipeline/train.py`,
`pipeline/features.py`, `config.yaml`, `main.py`, `app.py` unchanged.

---

## New files

| File | Purpose |
| --- | --- |
| `pipeline/analyze.py` | Importable analysis core + `run_feature_analysis(config, ...)`, module conventions (`from __future__ import annotations`, `logger`, `config: Config \| None = None`). |
| `scripts/feature_hypothesis_tests.py` | Thin CLI (template: `scripts/backtest.py:15-33,71-116`) — argparse, `ROOT`/`sys.path` shim, `get_config()`, calls `run_feature_analysis`, writes `outputs/*` + prints the markdown. |
| `tests/test_analyze.py` | pytest, synthetic frames only (style of `tests/test_feedback.py`). |
| `outputs/feature_hypothesis_tests.md` + `.csv` + `_multivariable.csv` + `_model.csv` + `_leakage.json` | Generated at run time. |
| `outputs/plots/feature_stats/*.png` | Optional (`--plots`), pure matplotlib (not via seaborn-gated `visualize.py`). |

`requirements.txt`: add `statsmodels>=0.14,<0.15` (pulls `patsy`; `scipy` already
present). Optionally pin `scipy==1.16.3` explicitly.

### `pipeline/analyze.py` — signatures

```python
@dataclass
class FeatureSpec:
    name: str; kind: str  # "numeric"|"categorical"
    leaky: bool; leak_reason: str

# Phase 1 — data prep
def build_raw_frame(results, config) -> pd.DataFrame          # reproduce train_position_model's `processed`+dropna
def add_shifted_history(df, n_previous, completed_statuses) -> pd.DataFrame  # local; features.py:134-158 with .shift(1)
def build_clean_frame(results, config, drop_rates=False) -> pd.DataFrame
def feature_specs(frame_kind) -> list[FeatureSpec]            # "raw"|"clean"

# Phase 2 — univariate
def univariate_tests(df, specs, target, race_col="Race", n_perm=2000, rng=None) -> pd.DataFrame
def within_group_permutation_p(values, target, groups, stat_fn, observed, n_perm, rng) -> float
def bh_fdr(pvalues) -> np.ndarray                            # scipy.stats.false_discovery_control(method="bh")

# Phase 3 — multivariable (statsmodels)
def ols_cluster_robust(df, numeric, target, categorical=("Team",), cluster="Driver") -> dict
def mixedlm_driver(df, numeric, target, categorical=("Team",)) -> dict     # Position ~ ... + (1|Driver)
def gee_logit(df, numeric, target, categorical=("Team",), cluster="Driver") -> dict  # binary targets
def vif_table(df, numeric) -> pd.DataFrame                   # statsmodels variance_inflation_factor

# Phase 4 — model-based (Position only)
def grouped_permutation_importance(df, feature_cols, n_splits=5, n_repeats=50, config=None, rng=None) -> pd.DataFrame
def drop_one_cv_mae(df, feature_cols, n_splits=5, config=None) -> pd.DataFrame

# Phase 5 / 6
def leakage_report(raw_df, clean_df, config) -> dict
def run_feature_analysis(config=None, *, targets=("Position","podium_finish","points_finish","race_winner"),
                         frames=("raw","clean"), run_model=True, n_perm=2000, seed=42,
                         make_plots=False, drop_rates=False, out_dir=None) -> dict
```

Single `np.random.default_rng(seed)` in `run_feature_analysis`, threaded down.

---

## Phase 0 — Prerequisites

**Base branch.** The CLI + feedback-loop work is committed on branch
`feature/terminal-cli-and-feedback-loop` (tip `9ee9834`). Build this analysis on
that branch: `git checkout feature/terminal-cli-and-feedback-loop`, confirm
`venv/bin/python -m pytest -q` is green (60 tests). It provides
`pipeline/model_registry.py` (`model_feature_columns`), `pipeline/feedback.py`
(scipy precedent), `scripts/backtest.py` (CLI template), `fastf1==3.8.3`, and the
`pipeline/features.py` pandas-2.1.3 fix that this analysis depends on. New work
lands as further commits on this branch.

**Dependency.** `venv/bin/python -m pip install "statsmodels>=0.14,<0.15"`
(manylinux cp312 wheel; brings `patsy`; `scipy` already present). Add the pin to
`requirements.txt`. Verify:
`venv/bin/python -c "import statsmodels.api as sm, statsmodels.formula.api as smf; print(sm.__version__)"`.

## Phase 1 — Data prep

**Raw frame** — reproduce `train_position_model` (`train.py:193-213`) exactly:
`create_historical_features(results, n_previous=config.pipeline.lookback_races,
completed_statuses=config.constants.completed_statuses)` (no `as_of_round`), 19-col
`race_features`, `available = [f for f in race_features if f in processed.columns]`,
`processed[available + targets].dropna()` (~245 rows after `form_trend`'s `.shift(3)`
NaNs + 3 quali nulls). categorical `["Driver","Team"]`, numeric = rest. Carry
`Race`, `Driver` for grouping/clustering. Cross-check `available` against
`model_feature_columns(load_bundle(config).race_model)` (`model_registry.py:41,73`);
warn if the pickle is missing.

**Clean frame** — `add_shifted_history` is a **local helper in `analyze.py`**, not a
new flag on `create_historical_features` (that fn is on the production path). It
reimplements the 11 rollups (`features.py:134-158`) per driver sorted by `Race`
with `.shift(1)` before `.rolling(6, min_periods=1)`; keeps `form_trend`'s existing
`.shift(3)`. `build_clean_frame`:
- **drop** `PositionChange`;
- **replace** `driver_win_rate` / `team_reliability` with expanding *pre-race*
  versions (per group sorted by `Race`: cumulative count / mean over races
  **strictly before** this one; first race → NaN). `--drop-rates` drops both
  instead. Document both formulas in the report.
- use the shifted rollups; keep `GridPosition`, `QualifyingPerformance`, `Driver`,
  `Team`, `form_trend`. `dropna()` → ~230 rows.

## Phase 2 — Univariate association tests (per feature × target × frame)

| feature | target `Position` | binary target |
| --- | --- | --- |
| numeric | Spearman ρ (`spearmanr`) primary; Kendall τ; Pearson r for continuity | point-biserial (`pointbiserialr`) + Mann–Whitney U (`mannwhitneyu`), rank-biserial effect size |
| categorical (`Driver`,`Team`) | Kruskal–Wallis H (`kruskal`) + one-way ANOVA (`f_oneway`); ε² / η² | χ² (`chi2_contingency`) + Cramér's V |

Always report the coefficient plus a standardized effect size (rank-biserial,
`ε²=(H-k+1)/(n-k)`, `η²`, Cramér's V).

**Within-Race permutation p** (repeated-measures-valid): for each (feature,target),
`B=n_perm` (default 2000) iterations shuffling the target **within each `Race`
group** (`rng`), recompute the same statistic (abs ρ, H, …);
`p_perm = (1 + #{stat_perm ≥ stat_obs}) / (B + 1)`. If the feature is constant
within every race (`groupby("Race")[f].nunique().max() ≤ 1` — true for
`driver_win_rate`, `team_reliability`), fall back to **within-Driver** permutation;
record `perm_scheme`.

**Multiple comparisons**: BH-FDR (`scipy.stats.false_discovery_control(p,
method="bh")`) within each `(frame, target, p-family)` — separate families for
analytic vs permutation p. `bh_fdr` guarantees `q ≥ p`, monotone (unit-tested).
Primary verdict = `q_perm_bh < 0.05`.

**`outputs/feature_hypothesis_tests.csv`**: `frame, feature, feature_kind, leaky,
leak_reason, target, test, statistic, effect_size_name, effect_size, n, p_analytic,
q_analytic_bh, perm_scheme, p_perm, q_perm_bh, n_perm, direction (plain words),
significant`.

## Phase 3 — Multivariable inference (statsmodels)

Per frame. Standardize numeric predictors (matches the model's `StandardScaler`).
`Team` via `C(Team)` (10 dummies). **Exclude `Driver` from the regressions** (23
levels on ~245 rows; `driver_win_rate` near-deterministic in Driver → separation) —
report a `Driver`-fixed-effects sensitivity fit only, flagging rank deficiency.

- **`Position` — OLS, cluster-robust by Driver**:
  `smf.ols("Position ~ <z-numeric> + C(Team)", df).fit(cov_type="cluster",
  cov_kwds={"groups": df["Driver"]})`. Report per-term `coef, std err, t, P>|t|`,
  `rsquared`, `rsquared_adj`, `fvalue`, `f_pvalue`. Type-II partial F via
  `statsmodels.stats.anova.anova_lm(fit, typ=2)`. **VIF** via
  `statsmodels.stats.outliers_influence.variance_inflation_factor` per numeric
  predictor.
- **`Position` — mixed effects**:
  `smf.mixedlm("Position ~ <z-numeric> + C(Team)", df, groups=df["Driver"]).fit()`
  — random intercept by Driver. Report fixed-effect coef + p, `Group Var`, and
  compare significance to the cluster-robust OLS. (Stretch: crossed
  `(1|Driver)+(1|Race)` via `vc_formula` — note as optional.)
- **Binary targets (`podium_finish`, `points_finish`, `race_winner`) — GEE panel
  logistic**:
  `GEE.from_formula("<target> ~ <z-numeric> + C(Team)", groups="Driver", data=df,
  family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()`
  → population-averaged log-odds with cluster-robust p. For `race_winner` (~11
  positives) flag quasi-separation / low power; drop `C(Team)` there if it fails to
  converge.

**`outputs/feature_hypothesis_tests_multivariable.csv`**: `frame, target, model
(ols_cluster/mixedlm/gee_logit), term, coef, se, stat, p, vif, delta_r2_or_qic,
partial_f, p_partial_f`; + summary rows (`n, k, r2, adj_r2, f, p_f, group_var,
n_clusters, converged, cond_number`).

## Phase 4 — Model-based importance & significance (target = `Position` only)

Rebuild the exact position `Pipeline` from `train.py:215-228` (`ColumnTransformer([
("num", StandardScaler(), numerical), ("cat", OneHotEncoder(handle_unknown=
"ignore"), categorical)])` → `XGBRegressor(**config.models.position)`), but:
- `GroupKFold(n_splits=5)` grouped by `Race` (replaces the leaky plain `cv=5` /
  `train_test_split` in `train.py:231-240`).
- Per held-out fold: `sklearn.inspection.permutation_importance(fitted, X_hold,
  y_hold, n_repeats=50, scoring="neg_mean_absolute_error", random_state=<rng>)`.
  Pool the 250 per-feature MAE-increase deltas → mean, std, one-sided empirical
  `p = (1 + #{delta ≤ 0}) / (n_repeats*n_splits + 1)`.
- Side-by-side with native `feature_importances_` (gain), OHE columns collapsed
  back to `Driver`/`Team` via `get_feature_names_out()`.
- **Drop-one GroupKFold(by Race) CV MAE**: per feature, remove it, re-run 5-fold
  grouped CV, record `cv_mae_without` and `delta_cv_mae` vs the full model.

**`outputs/feature_hypothesis_tests_model.csv`**: `frame, feature,
perm_importance_mean, perm_importance_std, p_perm_importance, gain_importance,
cv_mae_full, cv_mae_without, delta_cv_mae`.

## Phase 5 — Leakage quantification (`leakage_report`)

1. `Position` vs `GridPosition - PositionChange`: R² (expect ≥ 0.9999), max abs
   residual (≈ 0).
2. Position-model GroupKFold(by Race) MAE **with** vs **without** the leaky trio
   `{PositionChange, driver_win_rate, team_reliability}` — both MAEs + ratio.
3. Per unshifted rollup vs its shifted twin: corr with `Position` each way + corr
   between the two versions.
4. `driver_win_rate` / `team_reliability`: `nunique` per group == 1; corr of
   `driver_win_rate` with each driver's mean finishing position.

→ `outputs/feature_hypothesis_tests_leakage.json` + a "Leakage" section in the `.md`.

## Phase 6 — Report (`outputs/feature_hypothesis_tests.md`)

Sections: (1) caveats (see below); (2) data — raw vs clean row counts, dropna
accounting, targets, cluster counts; (3) **verdict table** per feature for the
primary target `Position` — `feature | frame | univariate q_perm + direction +
effect | survives multivariable? (cluster-robust p<.05 & VIF<10 & meaningful ΔR²,
plus mixedlm agreement) | survives model test? (perm-importance p<.05 & drop-one
ΔMAE>0) | leakage flag | overall call ∈ {strong, moderate, weak, none,
leakage-driven}`; (4) univariate full tables (all 4 targets, both frames);
(5) multivariable (OLS + mixedlm for Position, GEE for binary); (6) model-based
importance; (7) leakage; (8) method notes (permutation scheme, cluster-robust /
GEE / mixed-effects rationale, BH-FDR families). `.csv`/`.json` mirror it
machine-readably. `--plots`: feature-vs-`Position` scatter + rolling median,
perm-importance bar ± error, VIF bar, p-vs-q calibration → `outputs/plots/feature_stats/`.

## Phase 7 — CLI (`scripts/feature_hypothesis_tests.py`)

```
--target Position,podium_finish,points_finish,race_winner   (comma list; default all four)
--frames raw|clean|both        default both
--no-model                     skip Phase 4
--permutations N               default 2000
--perm-importance-repeats N    default 50
--cv-splits N                  default 5
--drop-rates                   clean frame drops win-rate/reliability entirely
--plots
--seed N                       default 42
--out-dir PATH                 default <repo>/outputs
```
`main()` → `run_feature_analysis(cfg, ...)` returns `{dataframes..., "markdown": str}`;
`main()` writes the files and prints the markdown (like `scripts/backtest.py:113-116`).

## Phase 8 — Tests (`tests/test_analyze.py`, synthetic, no network)

Synthetic: `R` races × `D` drivers, `Position` = per-race permutation of 1..D,
`Team` from driver, `signal_feat = Position + N(0,2)`, `noise_feat = N(0,1)`,
`dup_feat = signal_feat`, `const_in_race_feat` (per-driver constant). Assert:
1. `univariate_tests`: post-BH `signal_feat` `q_perm_bh < 0.05` with right
   `direction`; `noise_feat` not.
2. `bh_fdr`: `q ≥ p` elementwise, monotone in sorted-p, all in [0,1].
3. `within_group_permutation_p` on `noise_feat` over ~200 seeds → mean p in
   [0.35, 0.65].
4. `vif_table`: VIF of `dup_feat` (with `signal_feat`) > 1e3.
5. `ols_cluster_robust`: recovers `signal_feat` sign, `p < 0.05`; `noise_feat`
   `p > 0.05` most seeds; `n_clusters` == distinct drivers.
6. `gee_logit` on a synthetic binary target: runs, returns a coef table, planted
   signal `p < 0.05`.
7. `add_shifted_history`: on a 1-driver frame `avg_position_last` at race k ==
   mean of positions in races `k-6..k-1` (excludes k); `build_clean_frame` output
   has no `PositionChange`.
8. `const_in_race_feat` → `perm_scheme == "within_driver"`.
9. Smoke: `run_feature_analysis(run_model=False, n_perm=200)` returns expected
   keys / non-empty frames; with `config.models.position` monkeypatched tiny
   (`n_estimators=10`), `run_model=True` completes.
Tests use `n_perm=200`, `cv_splits=3`, `n_repeats=5`.

## Phase 9 — Verification (real data)

```bash
cd "/run/media/rajay/New Volume1/Machine Learning/F1_predictor"
venv/bin/python -m pip install "statsmodels>=0.14,<0.15"
venv/bin/python -c "import statsmodels.api as sm; print(sm.__version__)"
venv/bin/python -m pytest tests/test_analyze.py -q
venv/bin/python scripts/feature_hypothesis_tests.py --no-model --frames both          # fast
venv/bin/python scripts/feature_hypothesis_tests.py --target Position --frames both --seed 42
venv/bin/python scripts/feature_hypothesis_tests.py --target podium_finish,points_finish,race_winner --no-model
venv/bin/python scripts/feature_hypothesis_tests.py --frames both --plots
venv/bin/python -m pytest -q          # nothing else broke
```

Expected: `outputs/feature_hypothesis_tests.md` (8 sections, non-empty);
`.csv` ~90–150 rows/target; every `p_perm ∈ (0,1]`, `q_*_bh ≥ p_*`;
`_multivariable.csv` one block per (frame,target,model); `_model.csv` 19 rows/frame;
`_leakage.json` identity R² ≥ 0.9999.

Sanity checks (printed at run end):
- `PositionChange` in the **raw** frame: top by |Spearman| & perm-importance,
  `p_analytic ≈ 0`, `p_perm ≈ 1/(B+1)`, `leaky == True`, overall call
  `leakage-driven`.
- `driver_win_rate` / `team_reliability` flagged `leaky`, constant within group,
  `perm_scheme == "within_driver"`.
- **clean** frame: `PositionChange` absent; `GridPosition` /
  `avg_grid_last_shifted` still significant vs `Position` (the real signal).
- Phase 5 item 2: position-model MAE with the leaky trio ≪ without it (both
  numbers reported).
- GroupKFold folds never share a `Race`.

---

## Statistical validity limits (state prominently in the report)

- n ≈ 245–286, **one** 2026 season, 13 rounds → no genuine temporal hold-out;
  GroupKFold(by Race) is the best available and still optimistic.
- Repeated measures: ~23 driver / ~11 team clusters; within a race the 22
  `Position` values are a fixed permutation of 1..22. Hence within-Race
  permutation p, cluster-robust (by Driver) SEs, GEE for binary panels, and the
  `(1|Driver)` mixed model. With only ~13–23 clusters even cluster-robust SEs are
  somewhat optimistic — treat p just under 0.05 as "suggestive".
- ~19 features × 4 targets → BH-FDR within each `(frame, target, p-family)`;
  report raw p and q, decide on q.
- The data looks **simulated** (single clean synthetic-looking season) — findings
  may reflect the generator, not real F1.
- Leakage dominates the naive analysis by construction; the dual raw/clean design
  is what makes the honest conclusions readable.
