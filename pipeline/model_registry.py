"""Central loader for the trained model pickles + metrics.

Both the FastAPI app (`app.py`) and the terminal CLI (`main.py`) load models
through this module so there is exactly one code path for reading
`models/*.pkl` and `models/metrics.json` from disk.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field

from pipeline.config_loader import Config

logger = logging.getLogger(__name__)

# The three trainable models, in a single place (was a hard-coded triple in
# several spots in app.py).
MODEL_NAMES = ("laptime", "racewin", "position")

_MODEL_FILES = {
    "racewin": "xgb_racewin_pipeline.pkl",
    "laptime": "xgb_laptime_pipeline.pkl",
    "position": "race_prediction_pipeline.pkl",
    "laptime_features": "xgb_laptime_features.pkl",
}


def _load_pickle(path):
    """Load a pickle, returning None (with a warning) if it is missing/unreadable."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:  # noqa: BLE001 — any failure means "model not available"
        logger.warning("Could not load model %s: %s", path, e)
        return None


def model_feature_columns(model) -> list[str]:
    """Input column names a fitted sklearn Pipeline(ColumnTransformer, ...) expects,
    in transformer order. Single source of truth for both prediction and evaluation
    so the feature list is never hand-maintained in more than one place."""
    pre = model.named_steps["preprocessor"]
    cols: list[str] = []
    for _name, _transformer, columns in pre.transformers_:
        cols.extend(list(columns))
    return cols


def load_metrics(config: Config) -> dict:
    """Read models/metrics.json, returning {} if absent or unreadable."""
    path = config.paths.models_dir / "metrics.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not load metrics.json: %s", e)
        return {}


@dataclass
class ModelBundle:
    win_model: object | None = None
    laptime_pipeline: object | None = None
    race_model: object | None = None
    laptime_features: object | None = None
    metrics: dict = field(default_factory=dict)


def load_bundle(config: Config) -> ModelBundle:
    """Load every model pickle + metrics from ``config.paths.models_dir``."""
    md = config.paths.models_dir
    return ModelBundle(
        win_model=_load_pickle(md / _MODEL_FILES["racewin"]),
        laptime_pipeline=_load_pickle(md / _MODEL_FILES["laptime"]),
        race_model=_load_pickle(md / _MODEL_FILES["position"]),
        laptime_features=_load_pickle(md / _MODEL_FILES["laptime_features"]),
        metrics=load_metrics(config),
    )
