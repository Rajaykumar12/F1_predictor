from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

_DEFAULT_CONSTANTS = {
    "grid_size": 20,
    "race_phase_bins": [0, 15, 40, 100],
    "position_bins": [0, 5, 10, 15, 20],
    "default_tire": "MEDIUM",
    "unknown_position": 15,
    "completed_statuses": ["Finished"],
}

_DEFAULT_FEATURES = {
    "lookback_races": 6,
    "shift": 1,
    "families_enabled": [
        "quali", "form", "racecraft", "team", "reliability", "circuit", "championship",
    ],
    "cv_splits": 5,
    "holdout_rounds": 3,
    "vif_threshold": 10.0,
    "feature_selection": "lasso",
}

_DEFAULT_LOGGING = {"level": "INFO"}
_DEFAULT_API = {"cors_origins": ["*"], "data_freshness_hours": 48}
_DEFAULT_FEEDBACK = {
    "enabled": True,
    "window_races": 5,
    "min_scored_races": 5,
    "max_position_mae": 4.0,
    "min_spearman": 0.30,
    "min_winner_hit_rate": 0.15,
    "auto_retrain": False,
    "bias_correction_enabled": False,
    "bias_halflife_races": 3.0,
    "bias_max_abs": 3.0,
    "prediction_log_dir": "data/predictions",
    "score_history_path": "models/score_history.json",
}


@dataclass
class PathsConfig:
    data_dir: Path
    models_dir: Path
    cache_dir: Path
    plots_dir: Path


@dataclass
class PipelineConfig:
    season: int
    lookback_races: int
    min_lookback: int
    max_lookback: int
    api_sleep_seconds: int


@dataclass
class ConstantsConfig:
    grid_size: int
    race_phase_bins: List[int]
    position_bins: List[int]
    default_tire: str
    unknown_position: int
    completed_statuses: List[str]


@dataclass
class FeaturesConfig:
    lookback_races: int
    shift: int
    families_enabled: List[str]
    cv_splits: int
    holdout_rounds: int
    vif_threshold: float
    feature_selection: str


@dataclass
class LoggingConfig:
    level: str


@dataclass
class ApiConfig:
    cors_origins: List[str]
    data_freshness_hours: int


@dataclass
class FeedbackConfig:
    enabled: bool
    window_races: int
    min_scored_races: int
    max_position_mae: float
    min_spearman: float
    min_winner_hit_rate: float
    auto_retrain: bool
    bias_correction_enabled: bool
    bias_halflife_races: float
    bias_max_abs: float
    prediction_log_dir: Path
    score_history_path: Path


@dataclass
class ModelParams:
    n_estimators: int
    learning_rate: float
    max_depth: int
    random_state: int


@dataclass
class ModelsConfig:
    laptime: ModelParams
    racewin: ModelParams
    position: ModelParams


@dataclass
class Config:
    paths: PathsConfig
    pipeline: PipelineConfig
    constants: ConstantsConfig
    features: FeaturesConfig
    logging: LoggingConfig
    api: ApiConfig
    models: ModelsConfig
    feedback: FeedbackConfig

    def validate(self) -> None:
        if self.pipeline.season < 2018:
            raise ValueError(f"season must be >= 2018, got {self.pipeline.season}")
        if self.pipeline.lookback_races < 1:
            raise ValueError(f"lookback_races must be >= 1, got {self.pipeline.lookback_races}")
        if self.constants.grid_size < 10 or self.constants.grid_size > 26:
            raise ValueError(f"grid_size must be 10–26, got {self.constants.grid_size}")
        if not self.constants.completed_statuses:
            raise ValueError("completed_statuses must not be empty")
        fb = self.feedback
        if fb.window_races < 1:
            raise ValueError(f"feedback.window_races must be >= 1, got {fb.window_races}")
        if fb.min_scored_races < 1:
            raise ValueError(f"feedback.min_scored_races must be >= 1, got {fb.min_scored_races}")
        if fb.max_position_mae <= 0:
            raise ValueError(f"feedback.max_position_mae must be > 0, got {fb.max_position_mae}")
        if not 0 <= fb.min_spearman <= 1:
            raise ValueError(f"feedback.min_spearman must be in [0, 1], got {fb.min_spearman}")
        if not 0 <= fb.min_winner_hit_rate <= 1:
            raise ValueError(
                f"feedback.min_winner_hit_rate must be in [0, 1], got {fb.min_winner_hit_rate}"
            )
        if fb.bias_halflife_races <= 0:
            raise ValueError(
                f"feedback.bias_halflife_races must be > 0, got {fb.bias_halflife_races}"
            )
        ft = self.features
        if ft.lookback_races < 1:
            raise ValueError(f"features.lookback_races must be >= 1, got {ft.lookback_races}")
        if ft.shift < 0:
            raise ValueError(f"features.shift must be >= 0, got {ft.shift}")
        if ft.cv_splits < 2:
            raise ValueError(f"features.cv_splits must be >= 2, got {ft.cv_splits}")
        if ft.holdout_rounds < 1:
            raise ValueError(f"features.holdout_rounds must be >= 1, got {ft.holdout_rounds}")
        if ft.vif_threshold <= 1:
            raise ValueError(f"features.vif_threshold must be > 1, got {ft.vif_threshold}")
        if ft.feature_selection not in ("lasso", "vif", "none"):
            raise ValueError(
                "features.feature_selection must be one of lasso|vif|none, "
                f"got {ft.feature_selection!r}"
            )


def get_config(config_path: Path = _CONFIG_PATH) -> Config:
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Ensure config.yaml exists in the project root."
        )
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file {config_path}: {e}")

    if not isinstance(raw, dict):
        raise ValueError(f"Config file {config_path} is empty or not a mapping.")

    root = Path(config_path).parent

    try:
        paths = PathsConfig(
            data_dir=root / raw["paths"]["data_dir"],
            models_dir=root / raw["paths"]["models_dir"],
            cache_dir=root / raw["paths"]["cache_dir"],
            plots_dir=root / raw["paths"]["plots_dir"],
        )

        pipeline = PipelineConfig(**raw["pipeline"])

        raw_constants = {**_DEFAULT_CONSTANTS, **raw.get("constants", {})}
        constants = ConstantsConfig(**raw_constants)

        raw_features = {**_DEFAULT_FEATURES, **raw.get("features", {})}
        features_cfg = FeaturesConfig(**raw_features)

        raw_logging = {**_DEFAULT_LOGGING, **raw.get("logging", {})}
        logging_cfg = LoggingConfig(**raw_logging)

        raw_api = {**_DEFAULT_API, **raw.get("api", {})}
        api = ApiConfig(**raw_api)

        raw_feedback = {**_DEFAULT_FEEDBACK, **raw.get("feedback", {})}
        feedback = FeedbackConfig(**raw_feedback)
        feedback.prediction_log_dir = root / feedback.prediction_log_dir
        feedback.score_history_path = root / feedback.score_history_path

        models = ModelsConfig(
            laptime=ModelParams(**raw["models"]["laptime"]),
            racewin=ModelParams(**raw["models"]["racewin"]),
            position=ModelParams(**raw["models"]["position"]),
        )
    except KeyError as e:
        raise ValueError(f"Missing required config key: {e}")

    cfg = Config(
        paths=paths,
        pipeline=pipeline,
        constants=constants,
        features=features_cfg,
        logging=logging_cfg,
        api=api,
        models=models,
        feedback=feedback,
    )
    cfg.validate()
    return cfg
