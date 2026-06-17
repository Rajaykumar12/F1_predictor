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
    """F1 Predictor — single-command ML pipeline."""


@cli.command("run-all")
def run_all():
    """Run the full pipeline: fetch → clean → features → train → fetch-qualifying."""
    from pipeline.config_loader import get_config
    from pipeline.fetch import run_fetch, fetch_upcoming_qualifying
    from pipeline.clean import run_cleaning
    from pipeline.features import run_feature_engineering
    from pipeline.train import run_training

    cfg = get_config()
    run_fetch(cfg)
    run_cleaning(cfg)
    run_feature_engineering(cfg)
    run_training(cfg)
    # Non-fatal: qualifying data for the next race may not be available yet
    try:
        fetch_upcoming_qualifying(cfg)
    except Exception as e:
        logging.warning("fetch-qualifying skipped: %s", e)


@cli.command("fetch")
def fetch():
    """Fetch raw F1 data from the fastf1 API and save raw CSVs."""
    from pipeline.config_loader import get_config
    from pipeline.fetch import run_fetch

    run_fetch(get_config())


@cli.command("clean")
def clean():
    """Clean and preprocess raw CSV data."""
    from pipeline.config_loader import get_config
    from pipeline.clean import run_cleaning

    run_cleaning(get_config())


@cli.command("features")
def features():
    """Run feature engineering on cleaned data."""
    from pipeline.config_loader import get_config
    from pipeline.features import run_feature_engineering

    run_feature_engineering(get_config())


@cli.command("train")
@click.option(
    "--model",
    type=click.Choice(["laptime", "racewin", "position", "all"]),
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
@click.option(
    "--race",
    type=int,
    default=None,
    help="Race round number. Auto-detects next upcoming round if omitted.",
)
def fetch_qualifying(race: int | None):
    """Fetch qualifying results for the next race so predictions use real grid positions."""
    from pipeline.config_loader import get_config
    from pipeline.fetch import fetch_upcoming_qualifying

    fetch_upcoming_qualifying(get_config(), race_round=race)


@cli.command("predict")
@click.option(
    "--lookback",
    type=int,
    default=None,
    help="Number of recent races to use for form features (default: from config).",
)
def predict(lookback: int | None):
    """Predict the next race finishing order using qualifying + historical form.

    \b
    Typical weekend workflow:
      Saturday  →  python main.py fetch-qualifying
      Sunday    →  python main.py predict
    """
    import pickle
    import pandas as pd
    from pipeline.config_loader import get_config
    from pipeline.features import create_historical_features

    cfg = get_config()
    n_prev = lookback or cfg.pipeline.lookback_races

    # Load model
    model_path = cfg.paths.models_dir / "race_prediction_pipeline.pkl"
    if not model_path.exists():
        raise click.ClickException("Position model not found. Run: python main.py train --model position")
    with open(model_path, "rb") as f:
        race_model = pickle.load(f)

    # Load historical feature data
    results_path = cfg.paths.data_dir / "f1_results_features.csv"
    if not results_path.exists():
        raise click.ClickException("Feature data not found. Run: python main.py features")
    f1_results = pd.read_csv(results_path)

    season = cfg.pipeline.season
    season_data = f1_results[f1_results["Year"] == season].copy()
    if season_data.empty:
        raise click.ClickException(f"No data found for {season} season in results features.")

    processed = create_historical_features(
        season_data, n_previous=n_prev, completed_statuses=cfg.constants.completed_statuses
    )
    latest = processed.groupby("Driver").last().reset_index()
    latest = latest[latest["avg_position_last"].notna()]

    # Apply qualifying data if available
    race_label = "Next Grand Prix"
    using_quali = False
    quali_path = cfg.paths.data_dir / "upcoming_qualifying.csv"
    if quali_path.exists():
        quali = pd.read_csv(quali_path)
        race_label = quali["RaceName"].iloc[0] if "RaceName" in quali.columns else race_label
        for _, q_row in quali.iterrows():
            mask = latest["Driver"] == q_row["Driver"]
            if not mask.any():
                continue
            for col in ["GridPosition", "BestQualifyingTime", "GapToPole", "QualifyingPerformance"]:
                if col in quali.columns:
                    latest.loc[mask, col] = q_row[col]
        using_quali = True
        click.echo(f"\nUsing qualifying data from: {race_label}")
    else:
        click.echo(f"\nNo qualifying data found — using historical grid positions.")
        click.echo("Run 'python main.py fetch-qualifying' after Saturday qualifying to improve accuracy.\n")

    race_features = [
        "Driver", "Team", "GridPosition",
        "driver_win_rate", "team_reliability", "QualifyingPerformance", "PositionChange",
        "avg_position_last", "best_position_last", "avg_grid_last",
        "dnf_last", "reliability_rate", "avg_positions_gained",
        "podiums_last", "wins_last", "points_last", "form_trend",
    ]
    if "avg_quali_time" in latest.columns:
        race_features.extend(["avg_quali_time", "avg_gap_to_pole"])

    available = [f for f in race_features if f in latest.columns]
    predictions = race_model.predict(latest[available])

    rows = []
    for idx, row in latest.iterrows():
        rows.append({
            "Predicted": round(float(predictions[idx]), 1),
            "Driver": row["Driver"],
            "Team": row["Team"],
            "Grid": int(row["GridPosition"]) if pd.notna(row.get("GridPosition")) else "—",
            "Avg Pos (last {})".format(n_prev): round(float(row["avg_position_last"]), 1),
            "Wins": int(row.get("wins_last", 0)),
            "Podiums": int(row.get("podiums_last", 0)),
            "DNFs": int(row.get("dnf_last", 0)),
        })

    table = pd.DataFrame(rows).sort_values("Predicted").reset_index(drop=True)
    table.index += 1  # 1-based rank

    quali_note = "real qualifying" if using_quali else "historical grid"
    click.echo(f"\n{'='*70}")
    click.echo(f"  {race_label} — Predicted Finishing Order")
    click.echo(f"  ({quali_note}, last {n_prev} races form)")
    click.echo(f"{'='*70}")
    click.echo(table.to_string())
    click.echo(f"{'='*70}\n")


@cli.command("evaluate-laptime")
@click.option(
    "--race",
    type=int,
    default=None,
    show_default=True,
    help="Race round number to evaluate. Defaults to the most recent race in the data.",
)
def evaluate_laptime(race: int | None):
    """Compare predicted vs actual lap times for a completed race."""
    from pipeline.config_loader import get_config
    from pipeline.evaluate import evaluate_laptime as _evaluate

    _evaluate(get_config(), race_round=race)


@cli.command("serve")
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host.")
@click.option("--port", default=8000, show_default=True, help="Bind port.")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload (dev mode).")
def serve(host: str, port: int, reload: bool):
    """Start the FastAPI prediction server."""
    import uvicorn

    uvicorn.run("app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    cli()
