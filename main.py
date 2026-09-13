import logging
import sys

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


@click.group()
def cli():
    """F1 Predictor — data pipeline, training, and race-weekend prediction.

    Every command here has an equivalent FastAPI endpoint (see `main.py serve`
    then http://localhost:8000/docs). Run `python main.py <command> --help`.
    """


# --------------------------------------------------------------------------- #
# Pipeline stages
# --------------------------------------------------------------------------- #
@cli.command()
def fetch():
    """Fetch raw F1 data from the FastF1 API and save raw CSVs."""
    from pipeline.config_loader import get_config
    from pipeline.fetch import run_fetch

    run_fetch(get_config())


@cli.command()
def clean():
    """Clean and preprocess raw CSV data."""
    from pipeline.clean import run_cleaning
    from pipeline.config_loader import get_config

    run_cleaning(get_config())


@cli.command()
def features():
    """Run feature engineering on cleaned data."""
    from pipeline.config_loader import get_config
    from pipeline.features import run_feature_engineering

    run_feature_engineering(get_config())


@cli.command()
@click.option(
    "--model",
    type=click.Choice(["laptime", "racewin", "position", "position_ranker", "dnf", "all"]),
    default="all",
    show_default=True,
    help="Which model(s) to train.",
)
def train(model: str):
    """Train ML model(s) and save pipelines to models/."""
    from pipeline.config_loader import get_config
    from pipeline.train import run_training

    run_training(get_config(), models=[model])


@cli.command("fetch-qualifying")
@click.option("--race", type=int, default=None, help="Race round. Auto-detects the next upcoming round if omitted.")
def fetch_qualifying(race):
    """Fetch qualifying results so predictions use the real grid."""
    from pipeline.config_loader import get_config
    from pipeline.fetch import fetch_upcoming_qualifying

    fetch_upcoming_qualifying(get_config(), race_round=race)


@cli.command("evaluate-laptime")
@click.option("--race", type=int, default=None, show_default=True, help="Round to evaluate (default: most recent in data).")
def evaluate_laptime_cmd(race):
    """Compare predicted vs actual lap times for a completed race."""
    from pipeline.config_loader import get_config
    from pipeline.evaluate import evaluate_laptime

    evaluate_laptime(get_config(), race_round=race)


@cli.command("evaluate-position")
@click.option("--race", type=int, default=None, show_default=True, help="Round to evaluate (default: most recent in data).")
def evaluate_position_cmd(race):
    """Compare the position model's predicted order vs the actual result (post-hoc audit)."""
    from pipeline.config_loader import get_config
    from pipeline.evaluate import evaluate_position

    evaluate_position(get_config(), race_round=race)


@cli.command("run-all")
def run_all():
    """Full pipeline: fetch -> clean -> features -> train -> fetch-qualifying."""
    from pipeline import orchestrate
    from pipeline.config_loader import get_config

    result = orchestrate.run_all(get_config())
    click.echo(result)


# --------------------------------------------------------------------------- #
# Race-weekend workflow
# --------------------------------------------------------------------------- #
def _print_forecast_table(result, model_r2):
    has_probs = any(f.win_probability is not None for f in result.forecasts)
    if has_probs:
        header = f"{'#':>2}  {'Driver':<14} {'Team':<16} {'PredPos':>7} {'P(win)':>7} {'P(pod)':>7} {'P(pts)':>7}  Band"
    else:
        header = f"{'#':>2}  {'Driver':<14} {'Team':<16} {'PredPos':>7} {'Conf':>5}  Form"
    click.echo("\n" + "=" * 82)
    click.echo(f"  {result.next_race}")
    if result.simulation_n_trials:
        click.echo(f"  Monte-Carlo simulation: {result.simulation_n_trials:,} trials")
    click.echo("=" * 82)
    click.echo(header)
    click.echo("-" * 82)
    for f in result.forecasts:
        if has_probs:
            band = f"[{f.p10:.0f}–{f.p90:.0f}]" if f.p10 is not None else "-"
            click.echo(
                f"{f.pred_rank:>2}  {f.driver:<14} {str(f.team):<16} "
                f"{f.predicted_position:>7.2f} "
                f"{f.win_probability*100:>6.1f}% "
                f"{f.podium_probability*100:>6.1f}% "
                f"{f.points_probability*100:>6.1f}%  {band}"
            )
        else:
            rf = f.recent_form or {}
            form = f"avg {rf.get('avg_position', '?')}, {rf.get('wins', 0)}W {rf.get('podiums', 0)}P {rf.get('dnfs', 0)}DNF"
            click.echo(
                f"{f.pred_rank:>2}  {f.driver:<14} {str(f.team):<16} "
                f"{f.predicted_position:>7.2f} {f.confidence:>5.2f}  {form}"
            )
    click.echo("=" * 82)
    if model_r2 is not None:
        click.echo(f"  model R^2 (stored, optimistic): {model_r2}")
    if result.bias_applied:
        click.echo(f"  bias correction applied to {len(result.bias_applied)} driver(s)")
    click.echo("")


@cli.command("predict-race")
@click.option("--round", "round_no", type=int, default=None, help="Race round to predict (default: next completed+1).")
@click.option("--lookback", type=int, default=None, help="Recent races for form features (default: from config).")
@click.option("--save/--no-save", default=False, help="Append this prediction to the prediction log for later scoring.")
@click.option("--apply-bias/--no-apply-bias", "apply_bias", default=None,
              help="Override config for per-driver bias correction.")
def predict_race_cmd(round_no, lookback, save, apply_bias):
    """Predict a race's finishing order from qualifying + recent form.

    \b
    Weekend workflow:
      Saturday (after qualifying):  python main.py fetch-qualifying --race N
      Sunday   (before the race):   python main.py predict-race --round N --save
      Monday   (after the race):    python main.py score-race --round N
    """
    from pipeline.config_loader import get_config
    from pipeline.fetch import fetch_upcoming_qualifying, get_completed_race_rounds
    from pipeline import feedback, orchestrate
    from pipeline.model_registry import load_bundle
    from pipeline.predict import predict_race
    import pandas as pd

    cfg = get_config()
    lb = lookback or cfg.pipeline.lookback_races
    if not (cfg.pipeline.min_lookback <= lb <= cfg.pipeline.max_lookback):
        raise click.ClickException(
            f"--lookback must be {cfg.pipeline.min_lookback}..{cfg.pipeline.max_lookback}"
        )

    if round_no is None:
        done = get_completed_race_rounds(cfg.pipeline.season)
        round_no = (max(done) + 1) if done else 1
        click.echo(f"No --round given; predicting round {round_no}.")

    quali_path = cfg.paths.data_dir / "upcoming_qualifying.csv"
    need_fetch = True
    if quali_path.exists():
        try:
            need_fetch = int(pd.read_csv(quali_path)["Race"].iloc[0]) != round_no
        except Exception:
            need_fetch = True
    if need_fetch:
        click.echo(f"Fetching qualifying for round {round_no} ...")
        try:
            fetch_upcoming_qualifying(cfg, race_round=round_no)
        except Exception as e:
            click.echo(f"  warning: could not fetch qualifying ({e}); using historical grid.")

    bundle = load_bundle(cfg)
    if bundle.race_model is None:
        raise click.ClickException(
            "position model not trained — run: python main.py train --model position"
        )

    use_bias = cfg.feedback.bias_correction_enabled if apply_bias is None else apply_bias
    bias = orchestrate.compute_bias(cfg) if use_bias else {}

    result = predict_race(
        cfg, bundle.race_model, bundle.metrics,
        lookback=lb, as_of_round=round_no - 1, bias=bias or None,
        ranker_model=bundle.ranker_model,
        dnf_model=bundle.dnf_model,
    )
    _print_forecast_table(result, result.model_r2)

    if save:
        path = feedback.write_prediction_log(cfg, result, round_no=round_no)
        click.echo(f"Saved prediction log -> {path}")


@cli.command("score-race")
@click.option("--round", "round_no", type=int, default=None, help="Round to score (default: latest unscored prediction log).")
@click.option("--no-fetch", is_flag=True, default=False, help="Do not fetch results if missing.")
@click.option("--plots/--no-plots", default=False, help="Also write per-driver evaluation plots.")
def score_race_cmd(round_no, no_fetch, plots):
    """Score a saved prediction against the actual result and update the feedback loop."""
    from pipeline.config_loader import get_config
    from pipeline import feedback, orchestrate
    from pipeline.evaluate import evaluate_position

    cfg = get_config()
    if round_no is None:
        unscored = [l["round"] for l in feedback.list_prediction_logs(cfg) if not l.get("scored")]
        if not unscored:
            raise click.ClickException("No unscored prediction logs. Run predict-race --save first.")
        round_no = max(unscored)
        click.echo(f"Scoring latest unscored round: {round_no}")

    try:
        out = orchestrate.score_race(cfg, round_no, fetch_if_missing=not no_fetch)
    except (FileNotFoundError, RuntimeError) as e:
        raise click.ClickException(str(e))
    m, card, drift = out["metrics"], out["rolling_scorecard"], out["drift"]

    click.echo("\n" + "=" * 60)
    click.echo(f"  Round {round_no} — prediction vs actual")
    click.echo("=" * 60)
    click.echo(f"  Winner correct   : {m['winner_correct']}")
    click.echo(f"  Podium overlap   : {m['podium_overlap']}/3   (exact {m['podium_exact']}/3)")
    click.echo(f"  Top-5 / Top-10   : {m['top5']}/5   {m['top10']}/10")
    click.echo(f"  Spearman         : {m['spearman']:.3f}")
    click.echo(f"  Position MAE     : {m['position_mae']:.2f}   RMSE {m['position_rmse']:.2f}")
    click.echo(f"  Winner logloss   : {m['winner_logloss']:.3f}   Podium Brier {m['podium_brier']:.3f}"
               f"   Points Brier {m['points_brier']:.3f}")
    click.echo(f"  Drivers scored   : {m['n_drivers']}")
    click.echo("-" * 60)

    def _fmt2(v):
        return "n/a" if v is None else f"{v:.2f}"

    click.echo(f"  Rolling ({card.get('races', 0)} races): "
               f"winner {_fmt2(card.get('winner_hit_rate'))}, "
               f"MAE {_fmt2(card.get('position_mae_avg'))}, "
               f"Spearman {_fmt2(card.get('spearman_avg'))}")
    if drift["retrain_recommended"]:
        click.echo("  DRIFT: retrain recommended —")
        for r in drift["reasons"]:
            click.echo(f"    - {r}")
        if drift["auto_retrain"]:
            click.echo("  auto_retrain is ON — retraining now ...")
            from pipeline.train import run_training
            run_training(cfg)
        else:
            click.echo("  run: python main.py train   (or set feedback.auto_retrain: true)")
    else:
        click.echo("  drift: none — model tracking OK")
    click.echo("=" * 60 + "\n")

    if plots:
        evaluate_position(cfg, race_round=round_no)


@cli.command("race-weekend")
def race_weekend():
    """Print the recommended race-weekend command sequence."""
    click.echo(
        "\nRace-weekend routine:\n"
        "  Saturday (after qualifying):\n"
        "    python main.py fetch-qualifying --race N\n"
        "    python main.py predict-race --round N --save\n"
        "  Sunday/Monday (after the race):\n"
        "    python main.py score-race --round N\n"
        "  Periodically / when drift is flagged:\n"
        "    python main.py run-all        # refetch, retrain on the latest data\n\n"
        "Schedule it: wrap these in cron / a systemd timer / the /loop skill.\n"
    )


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #
@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host.")
@click.option("--port", default=8000, show_default=True, help="Bind port.")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload (dev mode).")
def serve(host: str, port: int, reload: bool):
    """Start the FastAPI prediction server (Swagger UI at /docs)."""
    import uvicorn

    uvicorn.run("app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    cli()
