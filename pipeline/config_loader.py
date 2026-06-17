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

_DEFAULT_LOGGING = {"level": "INFO"}
_DEFAULT_API = {"cors_origins": ["*"], "data_freshness_hours": 48}


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
class LoggingConfig:
    level: str


@dataclass
class ApiConfig:
    cors_origins: List[str]
    data_freshness_hours: int


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
    logging: LoggingConfig
    api: ApiConfig
    models: ModelsConfig

    def validate(self) -> None:
        if self.pipeline.season < 2018:
            raise ValueError(f"season must be >= 2018, got {self.pipeline.season}")
        if self.pipeline.lookback_races < 1:
            raise ValueError(f"lookback_races must be >= 1, got {self.pipeline.lookback_races}")
        if self.constants.grid_size < 10 or self.constants.grid_size > 26:
            raise ValueError(f"grid_size must be 10–26, got {self.constants.grid_size}")
        if not self.constants.completed_statuses:
            raise ValueError("completed_statuses must not be empty")


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

        raw_logging = {**_DEFAULT_LOGGING, **raw.get("logging", {})}
        logging_cfg = LoggingConfig(**raw_logging)

        raw_api = {**_DEFAULT_API, **raw.get("api", {})}
        api = ApiConfig(**raw_api)

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
        logging=logging_cfg,
        api=api,
        models=models,
    )
    cfg.validate()
    return cfg
