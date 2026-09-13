"""Unit tests for the declarative feature registry."""

from __future__ import annotations

import pytest

from pipeline import feature_registry as fr
from pipeline.config_loader import get_config


def test_registry_is_nonempty_and_frozen():
    assert len(fr.REGISTRY) >= 12
    with pytest.raises(Exception):
        fr.REGISTRY[0].name = "mutated"  # frozen dataclass


def test_every_feature_has_valid_family_kind_builder():
    for f in fr.REGISTRY:
        assert f.family in fr.FAMILIES
        assert f.kind in ("numeric", "categorical")
        assert f.build in fr.FAMILIES
        assert f.note.strip()


def test_no_target_derived_or_positionchange_feature():
    names = {f.name for f in fr.REGISTRY}
    for banned in ("PositionChange", "Position", "race_winner", "podium_finish", "points_finish"):
        assert banned not in names


def test_feature_count_in_planned_band():
    # docs/feature-engineering-redesign-plan.md's original band was 12-18. Phases
    # B/C of docs/prediction-improvement-plan.md deliberately grew the registry
    # (history, raceday, racewin families) — widen, not remove, the sanity check:
    # this still catches an accidental registry explosion, just at the new scale.
    non_context = [f for f in fr.REGISTRY if f.family != "context"]
    assert 12 <= len(non_context) <= 35


def test_all_registered_features_are_as_of_safe_by_default():
    # nothing in the initial registry uses post-race information
    assert fr.as_of_unsafe(fr.all_features()) == []


def test_names_unique():
    names = [f.name for f in fr.REGISTRY]
    assert len(names) == len(set(names))


def test_enabled_features_honours_config():
    cfg = get_config()
    feats = fr.enabled_features(cfg)
    fams = {f.family for f in feats}
    assert "context" in fams
    assert "quali" in fams


def test_enabled_features_can_disable_a_family(monkeypatch):
    cfg = get_config()
    monkeypatch.setattr(cfg.features, "families_enabled", ["quali", "form"])
    feats = fr.enabled_features(cfg)
    fams = {f.family for f in feats}
    assert fams == {"quali", "form", "context"}
    assert "circuit" not in fams


def test_selectors_split_numeric_and_categorical():
    feats = fr.all_features()
    num = set(fr.numeric_names(feats))
    cat = set(fr.categorical_names(feats))
    assert cat == {"Driver", "Team"}
    assert "GridPosition" in num
    assert num.isdisjoint(cat)


def test_by_builder_groups_and_preserves_order():
    grouped = fr.by_builder(fr.all_features())
    assert "quali" in grouped and "circuit" in grouped
    quali_names = [f.name for f in grouped["quali"]]
    assert quali_names == [f.name for f in fr.REGISTRY if f.build == "quali"]


def test_get_raises_keyerror_for_unknown():
    with pytest.raises(KeyError):
        fr.get("does_not_exist")
    assert fr.get("GridPosition").family == "context"
