"""Season scorecard: an honest walk-forward (forward-chaining) backtest of the
position model — Phase A2 of docs/prediction-improvement-plan.md.

Unlike ``scripts/backtest.py`` (single round, reuses the in-place model)
this RETRAINS for every round: for each round ``r``, it fits on every row
with ``Race < r`` and predicts ``r``. That is the honest yardstick every
later phase (B: multi-season data, C: new features, D: modelling, E:
probabilistic output) is judged against — a race is never predicted by a
model that has seen it, or any later race, in training.

Reuses ``pipeline.train.prepare_position_data`` / ``build_position_pipeline``
(the exact columns and preprocessing the production model uses) and
``pipeline.feedback.score_prediction`` (the exact scoring the single-round
backtest and post-race scoring use), so this is not a parallel metric
implementation.

    venv/bin/python scripts/rolling_backtest.py
    venv/bin/python scripts/rolling_backtest.py --min-train-races 4

Writes outputs/rolling_backtest.md (per-round table + season summary) and
outputs/plots/rolling_backtest.png (MAE / Spearman / winner-correct by round).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import feedback  # noqa: E402
from pipeline.config_loader import get_config  # noqa: E402
from pipeline.train import build_position_pipeline, fit_position_model, prepare_position_data  # noqa: E402

OUT_DIR = ROOT / "outputs"


def _predict_round(pipeline, data, numeric, categorical, train_idx, test_idx,
                    position_target: str = "position") -> dict:
    """Fit on ``train_idx``, predict ``test_idx``, and score against the
    actual finishing order — the per-round unit of the walk-forward loop.
    Goes through :func:`pipeline.train.fit_position_model` (D1) so the
    backtest trains on whatever target the production model does, and always
    scores in Position-domain regardless."""
    features = numeric + categorical
    model = fit_position_model(pipeline, data, train_idx, numeric, categorical, position_target)
    y_hat = model.predict(data.loc[test_idx, features])

    pred_df = pd.DataFrame({
        "Driver": data.loc[test_idx, "Driver"].to_numpy(),
        "PredictedPosition": y_hat,
    }).sort_values("PredictedPosition").reset_index(drop=True)
    pred_df["PredRank"] = pred_df.index + 1

    actual_df = data.loc[test_idx, ["Driver", "Position"]]
    return feedback.score_prediction(pred_df, actual_df)


def run_rolling_backtest(config, min_train_races: int | None = None) -> pd.DataFrame:
    prepared = prepare_position_data(config)
    data, numeric, categorical = prepared["data"], prepared["numeric"], prepared["categorical"]
    if "Driver" not in categorical:
        raise RuntimeError(
            "rolling_backtest needs the 'Driver' context feature to identify rows; "
            "check features.families_enabled / feature_registry."
        )

    min_train_races = min_train_races or config.pipeline.min_lookback
    # Walk forward by race_seq (B2) — the dense-rank (Year, Race) key — not the
    # season-local "Race" column, so a multi-season frame never confuses
    # 2024-round-5 with 2025-round-5.
    seqs = sorted(data["race_seq"].unique())
    has_year = "Year" in data.columns

    rows = []
    for seq in seqs:
        train_idx = data.index[data["race_seq"] < seq]
        test_idx = data.index[data["race_seq"] == seq]
        n_train_races = data.loc[train_idx, "race_seq"].nunique()
        if n_train_races < min_train_races or test_idx.empty:
            continue

        pipeline = build_position_pipeline(config.models.position, numeric, categorical)
        metrics = _predict_round(
            pipeline, data, numeric, categorical, train_idx, test_idx,
            position_target=config.features.position_target,
        )
        row = {
            "race_seq": int(seq),
            "round": int(data.loc[test_idx, "Race"].iloc[0]),
            "n_train_races": int(n_train_races),
            **metrics,
        }
        if has_year:
            row["season"] = int(data.loc[test_idx, "Year"].iloc[0])
        rows.append(row)

    return pd.DataFrame(rows)


def _season_summary(scorecard: pd.DataFrame) -> dict:
    if scorecard.empty:
        return {}
    return {
        "rounds_backtested": len(scorecard),
        "position_mae": float(scorecard["position_mae"].mean(skipna=True)),
        "spearman": float(scorecard["spearman"].mean(skipna=True)),
        "winner_hit_rate": float(scorecard["winner_correct"].mean()),
        "podium_overlap_avg": float(scorecard["podium_overlap"].mean()),
        "winner_logloss": float(scorecard["winner_logloss"].mean(skipna=True)),
        "podium_brier": float(scorecard["podium_brier"].mean(skipna=True)),
        "points_brier": float(scorecard["points_brier"].mean(skipna=True)),
    }


def _write_plot(scorecard: pd.DataFrame, plots_dir: Path) -> Path | None:
    if scorecard.empty:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"Skipping plot (matplotlib unavailable): {e}")
        return None

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    x = scorecard["race_seq"] if "race_seq" in scorecard.columns else scorecard["round"]
    ax1.plot(x, scorecard["position_mae"], "o-", color="coral", label="Position MAE")
    ax1.axhline(scorecard["position_mae"].mean(skipna=True), color="coral", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_ylabel("Position MAE")
    ax1b = ax1.twinx()
    ax1b.plot(x, scorecard["spearman"], "s-", color="steelblue", label="Spearman")
    ax1b.set_ylabel("Spearman")
    ax1.set_title("Rolling walk-forward backtest — Position MAE / Spearman by race")

    winner = scorecard["winner_correct"].astype(int)
    ax2.bar(x, winner, color=np.where(winner == 1, "seagreen", "lightcoral"))
    ax2.set_ylabel("Winner correct")
    ax2.set_xlabel("Race (chronological, race_seq)")
    ax2.set_yticks([0, 1])
    ax2.set_title("Winner-correct by race")

    fig.tight_layout()
    path = plots_dir / "rolling_backtest.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _write_report(scorecard: pd.DataFrame, summary: dict, config, plot_path: Path | None) -> Path:
    span = (
        f"{config.pipeline.history_start_season}–{config.pipeline.season}"
        if config.pipeline.history_start_season < config.pipeline.season
        else str(config.pipeline.season)
    )
    lines = [
        f"# Rolling walk-forward backtest — {span}",
        "",
        "Each race is predicted by a model retrained on every earlier race only "
        "(ordered by `race_seq`, monotonic across seasons — B2). This is the "
        "honest yardstick for the improvement plan's later phases — see "
        "`docs/prediction-improvement-plan.md` A2.",
        "",
    ]
    if scorecard.empty:
        lines.append(
            "No rounds had enough prior history to backtest "
            f"(need >= {config.pipeline.min_lookback} prior races)."
        )
    else:
        lines += [
            "## Season summary",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Rounds backtested | {summary['rounds_backtested']} |",
            f"| Position MAE | {summary['position_mae']:.3f} |",
            f"| Spearman | {summary['spearman']:.3f} |",
            f"| Winner hit-rate | {summary['winner_hit_rate']:.2f} |",
            f"| Podium overlap (avg /3) | {summary['podium_overlap_avg']:.2f} |",
            f"| Winner logloss | {summary['winner_logloss']:.3f} |",
            f"| Podium Brier | {summary['podium_brier']:.3f} |",
            f"| Points Brier | {summary['points_brier']:.3f} |",
            "",
            "## Per-round",
            "",
            "| Race | Train races | Winner correct | Podium overlap | Spearman "
            "| MAE | Winner logloss | Podium Brier | Points Brier |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        has_season = "season" in scorecard.columns
        for _, row in scorecard.iterrows():
            race_label = f"{int(row['season'])} R{row['round']}" if has_season else str(row["round"])
            lines.append(
                f"| {race_label} | {row['n_train_races']} | {row['winner_correct']} "
                f"| {row['podium_overlap']}/3 | {row['spearman']:.3f} | {row['position_mae']:.2f} "
                f"| {row['winner_logloss']:.3f} | {row['podium_brier']:.3f} | {row['points_brier']:.3f} |"
            )
        if plot_path is not None:
            # report lives in outputs/ — link relative to that, not the repo root
            try:
                rel = plot_path.relative_to(OUT_DIR)
            except ValueError:
                rel = Path("..") / plot_path.relative_to(ROOT)
            lines += ["", f"![Rolling backtest]({rel.as_posix()})"]

    report = OUT_DIR / "rolling_backtest.md"
    OUT_DIR.mkdir(exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--min-train-races", type=int, default=None,
        help="Skip rounds with fewer than this many prior races to train on "
             "(default: config.pipeline.min_lookback).",
    )
    args = ap.parse_args()

    config = get_config()
    scorecard = run_rolling_backtest(config, min_train_races=args.min_train_races)
    summary = _season_summary(scorecard)

    if not scorecard.empty:
        scorecard.to_csv(OUT_DIR / "rolling_backtest.csv", index=False)

    plot_path = _write_plot(scorecard, config.paths.plots_dir)
    report = _write_report(scorecard, summary, config, plot_path)

    if scorecard.empty:
        print("No rounds backtested — not enough prior history yet.")
    else:
        print(
            f"Backtested {summary['rounds_backtested']} round(s) — "
            f"MAE {summary['position_mae']:.2f}, Spearman {summary['spearman']:.3f}, "
            f"winner hit-rate {summary['winner_hit_rate']:.2f}"
        )
    print(f"Wrote {report}")
    if plot_path is not None:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
