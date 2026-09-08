import pytest
import yaml
from pathlib import Path
from pipeline.config_loader import get_config, Config


def test_get_config_loads_successfully(tmp_path):
    cfg_data = {
        "paths": {"data_dir": "data", "models_dir": "models", "cache_dir": "cache", "plots_dir": "plots"},
        "pipeline": {"season": 2026, "lookback_races": 6, "min_lookback": 3, "max_lookback": 12, "api_sleep_seconds": 2},
        "models": {
            "laptime": {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
            "racewin": {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
            "position": {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg_data))
    cfg = get_config(cfg_path)
    assert isinstance(cfg, Config)
    assert cfg.pipeline.season == 2026
    assert cfg.constants.grid_size == 20  # default
    assert "Finished" in cfg.constants.completed_statuses


def test_get_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        get_config(tmp_path / "nonexistent.yaml")


def test_get_config_invalid_yaml(tmp_path):
    bad_yaml = tmp_path / "config.yaml"
    bad_yaml.write_text(": this is not valid yaml\n  bad: [unterminated")
    with pytest.raises(ValueError, match="Invalid YAML"):
        get_config(bad_yaml)


def test_validation_rejects_old_season(tmp_path):
    cfg_data = {
        "paths": {"data_dir": "data", "models_dir": "models", "cache_dir": "cache", "plots_dir": "plots"},
        "pipeline": {"season": 2010, "lookback_races": 6, "min_lookback": 3, "max_lookback": 12, "api_sleep_seconds": 2},
        "models": {
            "laptime": {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
            "racewin": {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
            "position": {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg_data))
    with pytest.raises(ValueError, match="season must be >= 2018"):
        get_config(cfg_path)


_BASE_CFG = {
    "paths": {"data_dir": "data", "models_dir": "models", "cache_dir": "cache", "plots_dir": "plots"},
    "pipeline": {"season": 2026, "lookback_races": 6, "min_lookback": 3, "max_lookback": 12, "api_sleep_seconds": 2},
    "models": {
        "laptime": {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
        "racewin": {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
        "position": {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
    },
}


def _write_cfg(tmp_path, extra=None):
    data = {**_BASE_CFG}
    if extra:
        data = {**data, **extra}
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data))
    return p


def test_feedback_defaults_when_section_absent(tmp_path):
    cfg = get_config(_write_cfg(tmp_path))
    assert cfg.feedback.enabled is True
    assert cfg.feedback.window_races == 5
    assert cfg.feedback.min_scored_races == 5
    assert cfg.feedback.auto_retrain is False
    assert cfg.feedback.bias_correction_enabled is False
    assert cfg.feedback.prediction_log_dir == tmp_path / "data" / "predictions"
    assert cfg.feedback.score_history_path == tmp_path / "models" / "score_history.json"


def test_feedback_override(tmp_path):
    cfg = get_config(_write_cfg(tmp_path, {"feedback": {"window_races": 8, "auto_retrain": True}}))
    assert cfg.feedback.window_races == 8
    assert cfg.feedback.auto_retrain is True
    assert cfg.feedback.min_spearman == 0.30  # still default


def test_feedback_validation_rejects_bad_spearman(tmp_path):
    with pytest.raises(ValueError, match="min_spearman"):
        get_config(_write_cfg(tmp_path, {"feedback": {"min_spearman": 1.5}}))


def test_constants_defaults_applied(tmp_path):
    cfg_data = {
        "paths": {"data_dir": "data", "models_dir": "models", "cache_dir": "cache", "plots_dir": "plots"},
        "pipeline": {"season": 2026, "lookback_races": 6, "min_lookback": 3, "max_lookback": 12, "api_sleep_seconds": 2},
        "models": {
            "laptime": {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
            "racewin": {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
            "position": {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
        },
        # No constants section — defaults should apply
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg_data))
    cfg = get_config(cfg_path)
    assert cfg.constants.grid_size == 20
    assert cfg.constants.default_tire == "MEDIUM"
    assert cfg.api.data_freshness_hours == 48
