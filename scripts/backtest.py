"""Honest held-out backtest of the position model for any completed race.

Two modes:

    venv/bin/python scripts/backtest.py --round 13
        "Quick check" — reuses the currently-trained, in-place model with
        just a feature cutoff (as_of_round=N-1). If that model was trained on
        data that INCLUDES round N (the normal case), this is only a
        directional check: the model's learned parameters have already seen
        round N's outcome even though its input features are cut off. Fast
        (no retrain) — good for a sanity check between real retrains.

    venv/bin/python scripts/backtest.py --round 13 --retrain
        The actually honest version — fits a FRESH position model on only
        the races strictly before round N (same methodology as
        scripts/rolling_backtest.py's walk-forward loop, just for one round
        with a full per-driver report instead of an aggregate-only row) and
        predicts round N with it. Slower (one full retrain), but the model
        has never seen round N in any form. Does not use the ranker blend or
        Monte-Carlo simulation (kept deliberately simple, matching
        rolling_backtest.py) — point-estimate rank only.

Neither mode trims/overwrites the data CSVs or touches models/*.pkl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import feedback  # noqa: E402
from pipeline.config_loader import get_config  # noqa: E402
from pipeline.fetch import get_completed_race_rounds  # noqa: E402
from pipeline.model_registry import load_bundle  # noqa: E402
from pipeline.predict import predict_race  # noqa: E402
from pipeline.train import build_position_pipeline, fit_position_model, prepare_position_data  # noqa: E402

OUT_DIR = ROOT / "outputs"


def _predict_via_retrain(cfg, rnd: int, season: int):
    """Fit a fresh position model on race_seq < target only, predict round
    `rnd` with it. Returns (forecasts, race_label, trained_at_note) where
    forecasts is a list of dicts with pred_rank/driver/team/predicted_position
    — the same shape the report-building code below needs, so both modes
    can share it."""
    prepared = prepare_position_data(cfg)
    data, numeric, categorical = prepared["data"], prepared["numeric"], prepared["categorical"]

    target_rows = data.loc[(data["Year"] == season) & (data["Race"] == rnd)]
    if target_rows.empty:
        raise SystemExit(
            f"Round {rnd} of {season} has no rows surviving prepare_position_data's "
            "dropna (missing history) — can't retrain-and-predict it."
        )
    target_seq = int(target_rows["race_seq"].iloc[0])
    train_idx = data.index[data["race_seq"] < target_seq]
    test_idx = data.index[data["race_seq"] == target_seq]
    n_train_races = data.loc[train_idx, "race_seq"].nunique()

    pipeline = build_position_pipeline(cfg.models.position, numeric, categorical)
    model = fit_position_model(pipeline, data, train_idx, numeric, categorical, cfg.features.position_target)
    features = numeric + categorical
    y_hat = model.predict(data.loc[test_idx, features])

    order = pd.DataFrame({
        "Driver": data.loc[test_idx, "Driver"].to_numpy(),
        "Team": data.loc[test_idx, "Team"].to_numpy(),
        "PredictedPosition": y_hat,
    }).sort_values("PredictedPosition").reset_index(drop=True)

    forecasts = [
        {"pred_rank": i + 1, "driver": row.Driver, "team": row.Team,
         "predicted_position": float(row.PredictedPosition)}
        for i, row in order.iterrows()
    ]
    race_label = _event_name(season, rnd)
    note = f"freshly trained on {n_train_races} races strictly before round {rnd} (never saw its outcome)"
    return forecasts, race_label, note


def _event_name(season: int, rnd: int) -> str:
    try:
        import fastf1
        sched = fastf1.get_event_schedule(season, include_testing=False)
        row = sched[sched["RoundNumber"] == rnd]
        if not row.empty:
            return row.iloc[0]["EventName"]
    except Exception:
        pass
    return f"Round {rnd}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round", "-r", type=int, default=None, help="Round to backtest (default: latest completed).")
    ap.add_argument("--lookback", "-l", type=int, default=None, help="Form lookback (default: from config).")
    ap.add_argument(
        "--retrain", action="store_true",
        help="Fit a fresh model on only the races before this round instead of "
             "reusing the in-place model — the fully honest, but slower, check.",
    )
    args = ap.parse_args()

    cfg = get_config()
    season = cfg.pipeline.season
    lookback = args.lookback or cfg.pipeline.lookback_races

    completed = get_completed_race_rounds(season)
    rnd = args.round or (max(completed) if completed else None)
    if rnd is None:
        raise SystemExit("No completed rounds found for the season.")

    if args.retrain:
        forecasts, race_label, trained_at = _predict_via_retrain(cfg, rnd, season)
        honesty_note = (
            "_Model was retrained excluding round N — this is the fully honest "
            "score, it has never seen round N in any form._"
        )
    else:
        bundle = load_bundle(cfg)
        if bundle.race_model is None:
            raise SystemExit("position model not trained — run: python main.py train --model position")

        # Guard against data/upcoming_qualifying.csv holding a DIFFERENT round than
        # the one being backtested (e.g. it was last fetched for the next live
        # race). predict_race applies whatever's in that file unconditionally by
        # Driver name — for a historical round that means silently overwriting
        # the real, already-known GridPosition with the wrong race's grid. A
        # completed round already has its true GridPosition baked into
        # f1_results_features.csv, so the fix is simply: don't pass that file at
        # all unless it actually is round `rnd`'s qualifying.
        quali_path = cfg.paths.data_dir / "upcoming_qualifying.csv"
        quali_override = None
        if quali_path.exists():
            try:
                file_round = int(pd.read_csv(quali_path)["Race"].iloc[0])
            except Exception:
                file_round = None
            if file_round != rnd:
                print(
                    f"note: data/upcoming_qualifying.csv is for round {file_round}, "
                    f"not {rnd} — ignoring it and using round {rnd}'s real historical grid."
                )
                quali_override = str(ROOT / "__no_such_file__.csv")

        result = predict_race(
            cfg, bundle.race_model, bundle.metrics,
            lookback=lookback, as_of_round=rnd - 1,
            ranker_model=bundle.ranker_model,
            qualifying_path=quali_override,
        )
        race_label = result.race_label
        if quali_override is not None:
            # race_label falls back to "Next Grand Prix" when no quali file is
            # applied — look the real name up from the season schedule instead.
            race_label = _event_name(season, rnd)
        forecasts = [
            {"pred_rank": f.pred_rank, "driver": f.driver, "team": f.team,
             "predicted_position": f.predicted_position}
            for f in result.forecasts
        ]
        trained_at = f"in-place model, trained_at {feedback._model_trained_at(cfg)} " \
                     "(may include round N — see docstring)"
        honesty_note = (
            "_Features are cut off before round N; for a fully honest score the model "
            "must also have been trained on data that excludes round N — rerun with "
            "--retrain for that._"
        )

    pred_df = pd.DataFrame(
        [{"PredRank": f["pred_rank"], "Driver": f["driver"], "PredictedPosition": f["predicted_position"]}
         for f in forecasts]
    )

    simple = pd.read_csv(cfg.paths.data_dir / "f1_results_simple.csv")
    # Multi-season data (B1) means Race numbers repeat across years (e.g. both
    # 2022 and 2026 have a Round 13) — filter on (Year, Race) or this pools
    # a different season's result for any driver who raced in both, silently
    # corrupting the merge below via a many-to-many join on Driver name.
    actual = simple[(simple["Race"] == rnd) & (simple["Year"] == season)][["Driver", "Position"]]
    if actual.empty:
        raise SystemExit(f"No actual results for round {rnd} in f1_results_simple.csv — run `python main.py fetch`.")

    metrics = feedback.score_prediction(pred_df, actual)
    per_driver = feedback.per_driver_errors(pred_df, actual)

    OUT_DIR.mkdir(exist_ok=True)
    disp = pred_df.merge(
        actual.assign(Position=pd.to_numeric(actual["Position"], errors="coerce")),
        on="Driver", how="left",
    ).rename(columns={"Position": "ActualPosition"})
    disp["AbsError"] = (disp["PredRank"] - disp["ActualPosition"]).abs()
    disp.sort_values("PredRank").to_csv(OUT_DIR / f"backtest_r{rnd}_predictions.csv", index=False)

    lines = [
        f"# Backtest — {race_label} ({season} Round {rnd}, held out at as_of_round={rnd - 1})",
        "",
        f"model: {trained_at}",
        honesty_note,
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Winner correct | {metrics['winner_correct']} |",
        f"| Podium overlap | {metrics['podium_overlap']}/3 (exact {metrics['podium_exact']}/3) |",
        f"| Top-5 / Top-10 | {metrics['top5']}/5  {metrics['top10']}/10 |",
        f"| Spearman | {metrics['spearman']:.3f} |",
        f"| Position MAE | {metrics['position_mae']:.2f} |",
        f"| Position RMSE | {metrics['position_rmse']:.2f} |",
        f"| Winner logloss | {metrics['winner_logloss']:.3f} |",
        f"| Podium Brier | {metrics['podium_brier']:.3f} |",
        f"| Points Brier | {metrics['points_brier']:.3f} |",
        f"| Drivers scored | {metrics['n_drivers']} |",
        "",
        "| Pred | Driver | Team | PredPos | Actual | AbsErr |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for f in forecasts:
        row = disp[disp["Driver"] == f["driver"]]
        act = row["ActualPosition"].iloc[0] if not row.empty else None
        act_s = "DNF/NC" if pd.isna(act) else str(int(act))
        err_s = "" if pd.isna(act) else str(int(abs(f["pred_rank"] - act)))
        lines.append(
            f"| {f['pred_rank']} | {f['driver']} | {f['team']} | "
            f"{f['predicted_position']:.2f} | {act_s} | {err_s} |"
        )
    lines += [
        "",
        "Biggest misses:",
    ]
    for e in sorted(per_driver, key=lambda x: abs(x["signed_error"]), reverse=True)[:5]:
        lines.append(f"- {e['driver']}: predicted P{e['pred_rank']}, finished P{e['actual_position']} ({e['signed_error']:+d})")

    report = OUT_DIR / f"backtest_r{rnd}_report.md"
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
