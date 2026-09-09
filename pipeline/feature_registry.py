"""Declarative single source of truth for the position model's feature set.

Every feature the position model may consume is described here once — its family,
dtype, whether it is safe to use in a *true* pre-race forecast, which builder in
``pipeline.features`` produces it, and a short provenance note. ``pipeline.train``
reads the feature list off this registry (not a hand-maintained list),
``pipeline.features`` assembles exactly the enabled features, and
``pipeline.analyze.feature_specs`` mirrors it so the hypothesis-testing battery
and the production pipeline can never silently drift apart.

Design rules baked in (see ``docs/feature-engineering-redesign-plan.md``):

* **As-of-round discipline.** Every ``as_of_safe`` feature is computable from rows
  strictly *before* the target race, or from non-target columns of the target row
  itself (grid / qualifying, which are known before the race starts).
* **No target-derived columns.** The old ``PositionChange`` (== ``GridPosition -
  Position``) is gone entirely, not shifted.
* **Shift-before-roll.** Rolling form/racecraft features apply ``.shift(shift)``
  per driver (sorted by ``Race``) before every rolling window.
* **Parsimony + orthogonality.** ~18 features, one representative per collinear
  group, VIF < ``config.features.vif_threshold`` enforced downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

# Builder keys — each maps to a ``build_<key>_features`` function in
# ``pipeline.features``. Grouping features by builder lets the assembler call each
# family builder once.
FAMILIES = (
    "quali",
    "form",
    "racecraft",
    "team",
    "reliability",
    "circuit",
    "championship",
    "context",
)

# Model input columns that are not produced by a family builder — they are raw
# passthrough columns already present on the results frame.
CONTEXT_COLUMNS = ("Driver", "Team", "GridPosition")


@dataclass(frozen=True)
class Feature:
    """One declared model feature.

    Attributes
    ----------
    name:
        Column name as it appears in ``data/f1_results_features.csv`` and as the
        fitted pipeline expects it.
    family:
        One of :data:`FAMILIES`. ``"context"`` = raw passthrough (``Driver`` /
        ``Team`` / ``GridPosition``).
    kind:
        ``"numeric"`` or ``"categorical"`` — drives the ColumnTransformer split.
    as_of_safe:
        ``True``  => computable strictly from pre-race information; safe for a
        real forecast.
        ``False`` => uses information only known *after* the race (weather
        realised, safety-car realised, …). Kept out of the model and dropped by
        the forecast-time guard in ``predict.py``.
    build:
        Builder key (see :data:`FAMILIES`); ``"context"`` for passthrough.
    note:
        Human-readable provenance / what it replaces.
    """

    name: str
    family: str
    kind: str
    as_of_safe: bool
    build: str
    note: str

    def __post_init__(self) -> None:  # pragma: no cover - trivial guard
        if self.family not in FAMILIES:
            raise ValueError(f"{self.name}: unknown family {self.family!r}")
        if self.kind not in ("numeric", "categorical"):
            raise ValueError(f"{self.name}: kind must be numeric|categorical, got {self.kind!r}")
        if self.build not in FAMILIES:
            raise ValueError(f"{self.name}: unknown builder {self.build!r}")


# --------------------------------------------------------------------------- #
# The registry — ~18 features across 7 families + 3 context columns.
# --------------------------------------------------------------------------- #
REGISTRY: list[Feature] = [
    # --- context (raw passthrough) ------------------------------------------ #
    Feature("Driver", "context", "categorical", True, "context",
            "driver identity — one-hot; captures durable driver skill"),
    Feature("Team", "context", "categorical", True, "context",
            "constructor identity — one-hot; captures car performance tier"),
    Feature("GridPosition", "context", "numeric", True, "context",
            "starting grid slot; the single strongest clean signal (raw rank)"),

    # --- quali (all known once qualifying is done, before the race) -------- #
    Feature("quali_gap_to_pole_pct", "quali", "numeric", True, "quali",
            "GapToPole / pole_time * 100 — qualifying *pace* deficit, not rank; "
            "previously only its noisy 6-race rolling mean was used"),
    Feature("quali_gap_to_teammate_s", "quali", "numeric", True, "quali",
            "driver best-quali seconds minus same-race teammate — isolates driver "
            "from car"),
    Feature("quali_beat_teammate", "quali", "numeric", True, "quali",
            "1 if out-qualified the teammate this round — robust head-to-head"),
    Feature("q3_reached", "quali", "numeric", True, "quali",
            "1 if the driver set a Q3 lap — top-10 pace tier"),

    # --- form (per-driver rolling, shift-before-roll) --------------------- #
    Feature("form_avg_finish_s5", "form", "numeric", True, "form",
            "shift(1) then rolling(5) mean finishing Position — one representative "
            "of the ~9 collinear historical rollups that replaced them all"),
    Feature("form_trend", "form", "numeric", True, "form",
            "older-window minus recent-window mean Position (keeps its shift(3)); "
            "weak but retained secondary signal"),
    Feature("form_dnf_rate_s8", "form", "numeric", True, "form",
            "shift(1) then rolling(8) mean DNF flag — reliability trend; replaces "
            "dnf_last / reliability_rate"),

    # --- racecraft (causal replacement for leaky PositionChange) --------- #
    Feature("hist_positions_gained_s5", "racecraft", "numeric", True, "racecraft",
            "shift(1) rolling(5) mean of (GridPosition - Position) over *prior* "
            "races — causal race-start / overtaking skill"),
    Feature("hist_grid_finish_consistency_s5", "racecraft", "numeric", True, "racecraft",
            "shift(1) rolling(5) std of (GridPosition - Position) — race-to-race "
            "volatility"),

    # --- team (both cars) ------------------------------------------------- #
    Feature("team_form_avg_finish_s5", "team", "numeric", True, "team",
            "team's mean finishing Position over the previous up-to-5 races (both "
            "cars) — car strength going in"),
    Feature("team_quali_pace_rank", "team", "numeric", True, "team",
            "rank (1 = fastest) of the team's mean GapToPole this round — pre-race "
            "car pace"),

    # --- reliability (expanding, strictly pre-race) --------------------- #
    Feature("driver_dnf_rate_todate", "reliability", "numeric", True, "reliability",
            "season-to-date DNF rate over races strictly before this one — "
            "replaces the whole-season driver_win_rate"),
    Feature("team_reliability_todate", "reliability", "numeric", True, "reliability",
            "season-to-date finish rate over races strictly before this one — "
            "replaces the leaky whole-season team_reliability"),

    # --- circuit (static reference, data/circuits.csv) ----------------- #
    Feature("circuit_overtaking_index", "circuit", "numeric", True, "circuit",
            "curated 1-5 ease-of-overtaking score for the round's circuit"),
    Feature("circuit_is_street", "circuit", "numeric", True, "circuit",
            "1 if a street circuit — higher variance, grid stickier"),
    Feature("circuit_sc_probability", "circuit", "numeric", True, "circuit",
            "curated historical safety-car probability (0-1) for the circuit"),

    # --- championship (fully causal) ---------------------------------- #
    Feature("driver_points_gap_to_leader_before", "championship", "numeric", True, "championship",
            "leader's cumulative points minus the driver's, *before* this round — "
            "motivation / pressure signal"),
]


# --------------------------------------------------------------------------- #
# Selectors
# --------------------------------------------------------------------------- #
def _families_enabled(config) -> set[str]:
    """The set of families to build. ``context`` is always on. Reads
    ``config.features.families_enabled``; tolerates a config without the block."""
    ft = getattr(config, "features", None)
    enabled = set(getattr(ft, "families_enabled", None) or FAMILIES)
    enabled.add("context")
    return enabled


def enabled_features(config) -> list[Feature]:
    """Registry entries whose family is enabled in ``config.features``."""
    keep = _families_enabled(config)
    return [f for f in REGISTRY if f.family in keep]


def all_features() -> list[Feature]:
    """Every registered feature, regardless of config."""
    return list(REGISTRY)


def numeric_names(features: list[Feature]) -> list[str]:
    return [f.name for f in features if f.kind == "numeric"]


def categorical_names(features: list[Feature]) -> list[str]:
    return [f.name for f in features if f.kind == "categorical"]


def feature_names(features: list[Feature]) -> list[str]:
    return [f.name for f in features]


def as_of_unsafe(features: list[Feature]) -> list[str]:
    """Names of features that must be excluded from a true pre-race forecast."""
    return [f.name for f in features if not f.as_of_safe]


def by_builder(features: list[Feature]) -> dict[str, list[Feature]]:
    """Group features by their builder key, preserving registry order. The
    ``context`` bucket is passthrough (no builder call)."""
    out: dict[str, list[Feature]] = {}
    for f in features:
        out.setdefault(f.build, []).append(f)
    return out


def get(name: str) -> Feature:
    for f in REGISTRY:
        if f.name == name:
            return f
    raise KeyError(name)


# Registry schema version — bumped when features are added/removed/renamed so the
# feature manifest and saved metrics can be matched to the set that produced them.
REGISTRY_VERSION = "2026.1"
