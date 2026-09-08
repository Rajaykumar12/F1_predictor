"""Honest held-out backtest of the position model for any completed race.

Unlike the old scripts/backtest_monza.py, this does NOT trim/overwrite the
data CSVs or retrain. It uses `create_historical_features(as_of_round=N-1)` for
the cutoff and calls the shared `predict_race` core directly.

    venv/bin/python scripts/backtest.py --round 13
    venv/bin/python scripts/backtest.py            # latest completed round

Requires: models trained on data that EXCLUDES the target round for a truly
honest score (otherwise the model has seen round N in training). For a quick
directional check the in-place models are fine — the report says which.
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

OUT_DIR = ROOT / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round", "-r", type=int, default=None, help="Round to backtest (default: latest completed).")
    ap.add_argument("--lookback", "-l", type=int, default=None, help="Form lookback (default: from config).")
    args = ap.parse_args()

    cfg = get_config()
    season = cfg.pipeline.season
    lookback = args.lookback or cfg.pipeline.lookback_races

    completed = get_completed_race_rounds(season)
    rnd = args.round or (max(completed) if completed else None)
    if rnd is None:
        raise SystemExit("No completed rounds found for the season.")

    bundle = load_bundle(cfg)
    if bundle.race_model is None:
        raise SystemExit("position model not trained — run: python main.py train --model position")

    result = predict_race(
        cfg, bundle.race_model, bundle.metrics,
        lookback=lookback, as_of_round=rnd - 1,
    )
    pred_df = pd.DataFrame(
        [{"PredRank": f.pred_rank, "Driver": f.driver, "PredictedPosition": f.predicted_position}
         for f in result.forecasts]
    )

    simple = pd.read_csv(cfg.paths.data_dir / "f1_results_simple.csv")
    actual = simple[simple["Race"] == rnd][["Driver", "Position"]]
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

    trained_at = feedback._model_trained_at(cfg)
    lines = [
        f"# Backtest — {result.race_label} ({season} Round {rnd}, held out at as_of_round={rnd - 1})",
        "",
        f"model trained_at: {trained_at}",
        "_Features are cut off before round N; for a fully honest score the model "
        "must also have been trained on data that excludes round N._",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Winner correct | {metrics['winner_correct']} |",
        f"| Podium overlap | {metrics['podium_overlap']}/3 (exact {metrics['podium_exact']}/3) |",
        f"| Top-5 / Top-10 | {metrics['top5']}/5  {metrics['top10']}/10 |",
        f"| Spearman | {metrics['spearman']:.3f} |",
        f"| Position MAE | {metrics['position_mae']:.2f} |",
        f"| Position RMSE | {metrics['position_rmse']:.2f} |",
        f"| Drivers scored | {metrics['n_drivers']} |",
        "",
        "| Pred | Driver | Team | PredPos | Actual | AbsErr |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for f in result.forecasts:
        row = disp[disp["Driver"] == f.driver]
        act = row["ActualPosition"].iloc[0] if not row.empty else None
        act_s = "DNF/NC" if pd.isna(act) else str(int(act))
        err_s = "" if pd.isna(act) else str(int(abs(f.pred_rank - act)))
        lines.append(f"| {f.pred_rank} | {f.driver} | {f.team} | {f.predicted_position:.2f} | {act_s} | {err_s} |")
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
