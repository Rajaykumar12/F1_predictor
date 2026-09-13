from __future__ import annotations

import logging
import time
from datetime import date

import fastf1
import pandas as pd

from pipeline.config_loader import Config, get_config

logger = logging.getLogger(__name__)


def _setup_cache(config: Config) -> None:
    config.paths.cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(config.paths.cache_dir))


def collect_qualifying_data(year: int, race_name: int) -> list[dict]:
    logger.info("  Fetching qualifying data for %s %s...", year, race_name)
    try:
        session = fastf1.get_session(year, race_name, "Q")
        session.load()
        qualifying_data = []
        for _, result in session.results.iterrows():
            qualifying_data.append({
                "Year": year,
                "Race": race_name,
                "Driver": result["BroadcastName"],
                "Team": result["TeamName"],
                "QualifyingPosition": result["Position"],
                "Q1": result["Q1"].total_seconds() if pd.notna(result["Q1"]) else None,
                "Q2": result["Q2"].total_seconds() if pd.notna(result["Q2"]) else None,
                "Q3": result["Q3"].total_seconds() if pd.notna(result["Q3"]) else None,
                "Status": result["Status"],
            })
        logger.info("  Got %d qualifying results", len(qualifying_data))
        return qualifying_data
    except Exception as e:
        logger.warning("  Could not fetch qualifying for %s %s: %s", year, race_name, e)
        return []


def _final_lap_positions(session) -> dict:
    """Driver -> classified Position from their last recorded lap.

    Some historical sessions (older Ergast-backed seasons) return NaN in
    ``session.results.Position`` even though the lap telemetry has the real
    finishing order — this is the fallback ``collect_single_race`` uses for
    those rows."""
    laps = session.laps
    if laps.empty:
        return {}
    last = laps.sort_values("LapNumber").groupby("Driver").tail(1)
    return dict(zip(last["Driver"], last["Position"]))


def collect_single_race(
    year: int, race_name: int
) -> tuple[list[dict], list[dict], list[dict]]:
    logger.info("Fetching race data for %s race %s...", year, race_name)
    try:
        session = fastf1.get_session(year, race_name, "R")
        session.load()

        lap_data = [
            {
                "Year": year,
                "Race": race_name,
                "Driver": lap["Driver"],
                "Team": lap["Team"],
                "LapNumber": lap["LapNumber"],
                "LapTime_seconds": lap["LapTime"].total_seconds() if pd.notna(lap["LapTime"]) else None,
                "Position": lap["Position"],
                "TireCompound": lap["Compound"],
                "TireAge": lap["TyreLife"],
            }
            for _, lap in session.laps.iterrows()
        ]

        fallback_positions = _final_lap_positions(session)
        n_position_fallback = 0
        result_data = []
        for _, result in session.results.iterrows():
            position = result["Position"]
            if pd.isna(position):
                position = fallback_positions.get(result["Abbreviation"])
                if position is not None:
                    n_position_fallback += 1
            result_data.append({
                "Year": year,
                "Race": race_name,
                "Driver": result["BroadcastName"],
                "Team": result["TeamName"],
                "Position": position,
                "GridPosition": result["GridPosition"],
                "Points": result["Points"],
                "Status": result["Status"],
            })
        if n_position_fallback:
            logger.info(
                "  Recovered %d classified Position value(s) from lap telemetry "
                "(missing from session.results — see pipeline.fetch._final_lap_positions).",
                n_position_fallback,
            )

        logger.info("  Got %d laps, %d results", len(lap_data), len(result_data))
        qualifying_data = collect_qualifying_data(year, race_name)
        return lap_data, result_data, qualifying_data

    except Exception as e:
        logger.error("Failed to fetch race %s %s: %s", year, race_name, e)
        return [], [], []


def get_completed_race_rounds(season: int) -> list[int]:
    """Return round numbers for races that have already taken place."""
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    today = date.today()
    completed = schedule[schedule["EventDate"].dt.date < today]
    rounds = completed["RoundNumber"].tolist()
    logger.info(
        "%d of %d rounds completed so far in %s season.",
        len(rounds), len(schedule), season,
    )
    return rounds


def collect_multiple_races(config: Config, seasons: list[int] | None = None) -> tuple[list, list, list]:
    """Collect every completed race across ``seasons`` (default:
    ``[history_start_season .. season]``, B1). Every row carries ``Year`` so
    downstream merges/group-bys are always keyed ``(Year, Race, Driver)``.

    All-in-memory, single return — kept for callers (tests, one-off scripts)
    that want the raw rows. :func:`run_fetch` does NOT use this for a
    multi-season fetch any more (see its own per-season loop) because a
    multi-hour, ~100-race fetch can hit fastf1's hourly rate limit partway
    through, and losing every already-fetched race to an exception at the
    very end is a real failure mode, not a hypothetical one."""
    if seasons is None:
        seasons = list(range(config.pipeline.history_start_season, config.pipeline.season + 1))

    all_laps, all_results, all_qualifying = [], [], []

    for season in seasons:
        rounds = get_completed_race_rounds(season)
        if not rounds:
            logger.warning("No completed races found for %s season yet.", season)
            continue

        for race in rounds:
            laps, results, qualifying = collect_single_race(season, race)
            all_laps.extend(laps)
            all_results.extend(results)
            all_qualifying.extend(qualifying)
            time.sleep(config.pipeline.api_sleep_seconds)

    return all_laps, all_results, all_qualifying


def _merge_and_dedup(new_df: pd.DataFrame, path, subset: list[str]) -> pd.DataFrame:
    """Append ``new_df`` to whatever's already at ``path`` (if anything),
    de-duplicating on ``subset`` with the NEW rows winning — so a re-fetched
    race overwrites its own stale rows instead of doubling up."""
    if path.exists():
        try:
            existing = pd.read_csv(path)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not read existing %s (%s) — overwriting it.", path, e)
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    if existing.empty:
        combined = new_df
    elif new_df.empty:
        combined = existing
    else:
        # new_df LAST -> drop_duplicates(keep="last") makes new rows win.
        combined = pd.concat([existing, new_df], ignore_index=True)

    if not combined.empty:
        before = len(combined)
        combined = combined.drop_duplicates(subset=subset, keep="last")
        if len(combined) < before:
            logger.info("Removed %d duplicate row(s) merging into %s.", before - len(combined), path.name)
    return combined


def save_data(
    lap_data: list[dict],
    result_data: list[dict],
    qualifying_data: list[dict],
    config: Config,
    merge: bool = False,
) -> None:
    """Write the three simple CSVs. ``merge=True`` (used by :func:`run_fetch`'s
    per-season loop, B1) appends to whatever's already on disk instead of
    overwriting it, so a season fetched earlier in the run survives even if
    a later season's fetch fails (rate limit, network) — see
    :func:`_merge_and_dedup`. ``merge=False`` (the original behaviour) is for
    a single-season fetch that's meant to fully replace the file."""
    data_dir = config.paths.data_dir
    data_dir.mkdir(exist_ok=True)

    laps_path = data_dir / "f1_laps_simple.csv"
    results_path = data_dir / "f1_results_simple.csv"
    quali_path = data_dir / "f1_qualifying_simple.csv"

    laps_df = pd.DataFrame(lap_data)
    results_df = pd.DataFrame(result_data)
    quali_df = pd.DataFrame(qualifying_data)

    if merge:
        laps_df = _merge_and_dedup(laps_df, laps_path, ["Year", "Race", "Driver", "LapNumber"])
        results_df = _merge_and_dedup(results_df, results_path, ["Year", "Race", "Driver"])
        quali_df = _merge_and_dedup(quali_df, quali_path, ["Year", "Race", "Driver"])
    else:
        if not laps_df.empty:
            before = len(laps_df)
            laps_df = laps_df.drop_duplicates(subset=["Year", "Race", "Driver", "LapNumber"])
            if len(laps_df) < before:
                logger.info("Removed %d duplicate lap rows.", before - len(laps_df))
        if not results_df.empty:
            before = len(results_df)
            results_df = results_df.drop_duplicates(subset=["Year", "Race", "Driver"])
            if len(results_df) < before:
                logger.info("Removed %d duplicate result rows.", before - len(results_df))
        if not quali_df.empty:
            before = len(quali_df)
            quali_df = quali_df.drop_duplicates(subset=["Year", "Race", "Driver"])
            if len(quali_df) < before:
                logger.info("Removed %d duplicate qualifying rows.", before - len(quali_df))

    laps_df.to_csv(laps_path, index=False)
    results_df.to_csv(results_path, index=False)
    quali_df.to_csv(quali_path, index=False)

    logger.info(
        "Saved: %d laps, %d results, %d qualifying records to %s/",
        len(laps_df), len(results_df), len(quali_df), data_dir,
    )


def fetch_upcoming_qualifying(config: Config, race_round: int | None = None) -> None:
    """
    Fetch qualifying results for the next upcoming race and save to
    data/upcoming_qualifying.csv so predict_next_race can use real
    grid positions instead of historical averages.

    Parameters
    ----------
    race_round : int | None
        Explicit round number. If None, auto-detects the next round
        (first round whose race date is in the future).
    """
    _setup_cache(config)
    season = config.pipeline.season

    if race_round is None:
        schedule = fastf1.get_event_schedule(season, include_testing=False)
        today = date.today()
        upcoming = schedule[schedule["EventDate"].dt.date >= today]
        if upcoming.empty:
            logger.warning("No upcoming races found for %s season.", season)
            return
        race_round = int(upcoming.iloc[0]["RoundNumber"])
        race_name = upcoming.iloc[0]["EventName"]
        logger.info("Auto-detected next race: Round %d — %s", race_round, race_name)
    else:
        schedule = fastf1.get_event_schedule(season, include_testing=False)
        row = schedule[schedule["RoundNumber"] == race_round]
        race_name = row.iloc[0]["EventName"] if not row.empty else f"Round {race_round}"
        logger.info("Fetching qualifying for Round %d — %s", race_round, race_name)

    logger.info("Loading qualifying session...")
    try:
        session = fastf1.get_session(season, race_round, "Q")
        session.load()
    except Exception as e:
        logger.warning(
            "Could not load qualifying session for Round %d: %s. "
            "Run again after qualifying has taken place.",
            race_round, e,
        )
        return

    rows = []
    for _, result in session.results.iterrows():
        q1 = result["Q1"].total_seconds() if pd.notna(result["Q1"]) else None
        q2 = result["Q2"].total_seconds() if pd.notna(result["Q2"]) else None
        q3 = result["Q3"].total_seconds() if pd.notna(result["Q3"]) else None
        best = min(t for t in [q1, q2, q3] if t is not None) if any(
            t is not None for t in [q1, q2, q3]
        ) else None
        rows.append({
            "Year": season,
            "Race": race_round,
            "RaceName": race_name,
            "Driver": result["BroadcastName"],
            "Team": result["TeamName"],
            "GridPosition": result["Position"],
            "Q1": q1,
            "Q2": q2,
            "Q3": q3,
            "BestQualifyingTime": best,
        })

    if not rows:
        logger.warning(
            "Qualifying session loaded but contains no driver data for Round %d — "
            "session may not have started yet. Run again after qualifying.",
            race_round,
        )
        return

    df = pd.DataFrame(rows)

    pole_time = df["BestQualifyingTime"].min()
    df["GapToPole"] = df["BestQualifyingTime"] - pole_time
    df["QualifyingPerformance"] = (df["GridPosition"] / config.constants.grid_size) * 100

    out_path = config.paths.data_dir / "upcoming_qualifying.csv"
    config.paths.data_dir.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)

    logger.info(
        "Saved qualifying for %s Round %d (%d drivers) → %s",
        season, race_round, len(df), out_path,
    )
    print(f"\nQualifying result for {race_name}:")
    print(df[["GridPosition", "Driver", "Team", "BestQualifyingTime", "GapToPole"]]
          .sort_values("GridPosition")
          .to_string(index=False))


def fetch_single_race(config: Config, season: int, race_round: int) -> bool:
    """Fetch and save just one (season, round)'s laps/results/qualifying —
    the targeted counterpart to :func:`run_fetch`'s full multi-season sweep.

    Used by :func:`pipeline.orchestrate._actual_results` (the ``score-race``
    path) so checking whether a single round's result is published doesn't
    require re-walking every season's races (``run_fetch`` with no
    ``seasons`` argument re-fetches every completed race in
    ``[history_start_season .. season]`` — fine for the full pipeline, far
    too slow just to check on one round).

    Returns ``True`` if data was fetched and saved, ``False`` if the round
    hasn't happened yet per the schedule (no API call made in that case) or
    the fetch came back empty.
    """
    _setup_cache(config)
    if race_round not in get_completed_race_rounds(season):
        logger.info(
            "Season %d round %d hasn't happened yet per the schedule — skipping fetch.",
            season, race_round,
        )
        return False
    laps, results, qualifying = collect_single_race(season, race_round)
    if not (laps or results or qualifying):
        return False
    save_data(laps, results, qualifying, config, merge=True)
    logger.info("Fetched and saved season %d round %d.", season, race_round)
    return True


def run_fetch(config: Config | None = None, seasons: list[int] | None = None) -> None:
    """Fetch every completed race across ``seasons`` (default:
    ``[history_start_season .. season]``, B1) and save after EACH season —
    not once at the very end. A ~100-race, multi-hour multi-season fetch can
    hit fastf1's hourly rate limit or drop the network partway through;
    saving incrementally (merge=True — see :func:`save_data`) means that
    failure loses at most the season in progress, and simply re-running
    ``run_fetch`` resumes (already-saved seasons are skipped) instead of
    re-fetching from scratch."""
    if config is None:
        config = get_config()
    _setup_cache(config)
    if seasons is None:
        seasons = list(range(config.pipeline.history_start_season, config.pipeline.season + 1))

    logger.info("Starting data collection for season(s) %s...", seasons)
    for season in seasons:
        rounds = get_completed_race_rounds(season)
        if not rounds:
            logger.warning("No completed races found for %s season yet.", season)
            continue

        laps, results, qualifying = [], [], []
        try:
            for race in rounds:
                l, r, q = collect_single_race(season, race)
                laps.extend(l)
                results.extend(r)
                qualifying.extend(q)
                time.sleep(config.pipeline.api_sleep_seconds)
        finally:
            # Save whatever this season produced even if collect_single_race
            # raised partway through (rate limit, network) — merge=True so it
            # adds to prior seasons' rows instead of overwriting them.
            if laps or results or qualifying:
                save_data(laps, results, qualifying, config, merge=True)
                logger.info("Season %s saved (%d races).", season, len(rounds))

    logger.info("Data collection complete.")
