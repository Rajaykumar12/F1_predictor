from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — must come before any pyplot import
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info("Saved %s", path.name)


# ---------------------------------------------------------------------------
# Cleaning report
# ---------------------------------------------------------------------------

def plot_cleaning_report(
    laps_raw: pd.DataFrame,
    laps_clean: pd.DataFrame,
    results_clean: pd.DataFrame,
    qualifying_clean: pd.DataFrame | None,
    plots_dir: Path,
) -> None:
    """Save data-health PNGs to plots_dir/cleaning/ after the clean stage."""
    out = plots_dir / "cleaning"

    # 1. Lap time distribution: raw vs cleaned overlay
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(
            laps_raw["LapTime_seconds"].dropna(),
            bins=80, alpha=0.55, color="steelblue", label="Raw",
        )
        ax.hist(
            laps_clean["LapTime_seconds"].dropna(),
            bins=80, alpha=0.55, color="coral", label="Cleaned",
        )
        ax.set_xlabel("Lap Time (seconds)")
        ax.set_ylabel("Count")
        ax.set_title("Lap Time Distribution: Raw vs Cleaned")
        ax.legend()
        _save(fig, out / "lap_time_distribution.png")
    except Exception as e:
        logger.warning("Skipping lap_time_distribution.png: %s", e)

    # 2. Outlier boxplot: cleaned lap times per race
    try:
        races = sorted(laps_clean["Race"].unique())
        fig, ax = plt.subplots(figsize=(max(10, len(races) * 0.7), 5))
        data_by_race = [
            laps_clean.loc[laps_clean["Race"] == r, "LapTime_seconds"].dropna().values
            for r in races
        ]
        ax.boxplot(data_by_race, labels=[str(r) for r in races], patch_artist=True)
        ax.set_xlabel("Race")
        ax.set_ylabel("Lap Time (seconds)")
        ax.set_title("Cleaned Lap Time Spread per Race")
        plt.xticks(rotation=45, ha="right")
        _save(fig, out / "outliers_boxplot.png")
    except Exception as e:
        logger.warning("Skipping outliers_boxplot.png: %s", e)

    # 3. Missing data heatmap: % missing per column before cleaning
    try:
        missing_pct = (
            laps_raw.isnull()
            .groupby(laps_raw["Race"])
            .mean()
            .mul(100)
        )
        # Only keep columns that had any missing values
        missing_pct = missing_pct.loc[:, missing_pct.max() > 0]
        if not missing_pct.empty:
            fig, ax = plt.subplots(figsize=(max(8, len(missing_pct.columns) * 1.2), 5))
            sns.heatmap(
                missing_pct.T,
                annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "% Missing"},
            )
            ax.set_title("Missing Data (%) per Column per Race — Before Cleaning")
            ax.set_xlabel("Race")
            ax.set_ylabel("Column")
            _save(fig, out / "missing_data_heatmap.png")
        else:
            logger.info("No missing data found — skipping missing_data_heatmap.png")
    except Exception as e:
        logger.warning("Skipping missing_data_heatmap.png: %s", e)

    # 4. Data completeness: lap count per Grand Prix
    try:
        counts = laps_clean.groupby("Race").size().sort_index()
        fig, ax = plt.subplots(figsize=(max(8, len(counts) * 0.6), 5))
        ax.bar(counts.index.astype(str), counts.values, color="steelblue")
        ax.set_xlabel("Race")
        ax.set_ylabel("Lap Count")
        ax.set_title("Data Completeness: Laps Retained per Grand Prix")
        plt.xticks(rotation=45, ha="right")
        _save(fig, out / "data_completeness.png")
    except Exception as e:
        logger.warning("Skipping data_completeness.png: %s", e)

    logger.info("Cleaning report saved to %s/", out)


# ---------------------------------------------------------------------------
# Feature report
# ---------------------------------------------------------------------------

def plot_feature_report(
    laps_features: pd.DataFrame,
    results_features: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Save feature-insight PNGs to plots_dir/features/ after the features stage."""
    out = plots_dir / "features"

    # 1. Correlation matrix: numeric result features vs targets
    try:
        target_cols = ["race_winner", "podium_finish", "points_finish", "Position"]
        numeric_cols = results_features.select_dtypes(include=[np.number]).columns.tolist()
        # Remove targets and high-cardinality identifiers from feature side
        drop = {"Year", "Race", "Points", "GridPosition"}
        feature_cols = [c for c in numeric_cols if c not in drop and c not in target_cols]
        available_targets = [c for c in target_cols if c in results_features.columns]
        plot_cols = feature_cols + available_targets

        corr = results_features[plot_cols].corr()
        fig, ax = plt.subplots(figsize=(max(10, len(plot_cols) * 0.7), max(8, len(plot_cols) * 0.6)))
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.4, ax=ax, cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Feature Correlation Matrix (Results)")
        _save(fig, out / "correlation_matrix.png")
    except Exception as e:
        logger.warning("Skipping correlation_matrix.png: %s", e)

    # 2. Driver win rate: sorted horizontal bar
    try:
        if "driver_win_rate" in results_features.columns:
            win_rate = (
                results_features.groupby("Driver")["driver_win_rate"]
                .first()
                .sort_values(ascending=True)
            )
            fig, ax = plt.subplots(figsize=(8, max(5, len(win_rate) * 0.35)))
            ax.barh(win_rate.index.astype(str), win_rate.values, color="steelblue")
            ax.set_xlabel("Win Rate (%)")
            ax.set_title("Driver Win Rate")
            _save(fig, out / "driver_win_rate.png")
    except Exception as e:
        logger.warning("Skipping driver_win_rate.png: %s", e)

    # 3. Tire degradation by compound
    try:
        if "tire_degradation" in laps_features.columns and "TireCompound" in laps_features.columns:
            plot_data = laps_features.dropna(subset=["tire_degradation", "TireCompound"])
            # Clip extreme values for readability
            q_low = plot_data["tire_degradation"].quantile(0.02)
            q_high = plot_data["tire_degradation"].quantile(0.98)
            plot_data = plot_data[
                plot_data["tire_degradation"].between(q_low, q_high)
            ]
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(
                data=plot_data, x="TireCompound", y="tire_degradation",
                hue="TireCompound", order=["SOFT", "MEDIUM", "HARD"],
                palette="Set2", legend=False, ax=ax,
            )
            ax.set_xlabel("Tire Compound")
            ax.set_ylabel("Lap-to-Lap Time Delta (seconds)")
            ax.set_title("Tire Degradation Rate by Compound")
            ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
            _save(fig, out / "tire_degradation_by_compound.png")
    except Exception as e:
        logger.warning("Skipping tire_degradation_by_compound.png: %s", e)

    # 4. Race phase distribution: lap count per phase
    try:
        if "race_phase" in laps_features.columns:
            phase_counts = laps_features["race_phase"].value_counts().reindex(
                ["Early", "Middle", "Late"]
            )
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(phase_counts.index.astype(str), phase_counts.values, color=["#4CAF50", "#2196F3", "#FF5722"])
            ax.set_xlabel("Race Phase")
            ax.set_ylabel("Lap Count")
            ax.set_title("Lap Count by Race Phase")
            _save(fig, out / "race_phase_distribution.png")
    except Exception as e:
        logger.warning("Skipping race_phase_distribution.png: %s", e)

    # 5. Team reliability: sorted bar chart
    try:
        if "team_reliability" in results_features.columns:
            reliability = (
                results_features.groupby("Team")["team_reliability"]
                .first()
                .sort_values(ascending=True)
            )
            fig, ax = plt.subplots(figsize=(8, max(5, len(reliability) * 0.4)))
            ax.barh(reliability.index.astype(str), reliability.values, color="coral")
            ax.set_xlabel("Reliability (%)")
            ax.set_title("Team Reliability (% Races Finished)")
            ax.set_xlim(0, 105)
            _save(fig, out / "team_reliability.png")
    except Exception as e:
        logger.warning("Skipping team_reliability.png: %s", e)

    logger.info("Feature report saved to %s/", out)
