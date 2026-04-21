"""Tests for retrieval.py — symbol extraction, canonicalization, expansion, scoring."""
from pathlib import Path

import pytest
import yaml

from empty_space.retrieval import (
    canonicalize,
    expand_with_cooccurrence,
    load_synonym_map,
    merge_cooccurrence,
)


# --- canonicalize ---

def test_canonicalize_returns_self_when_not_in_map():
    assert canonicalize("愧疚", {}) == "愧疚"


def test_canonicalize_returns_canonical_when_mapped():
    synonym_map = {"愧疚感": "愧疚", "罪惡感": "愧疚"}
    assert canonicalize("愧疚感", synonym_map) == "愧疚"
    assert canonicalize("罪惡感", synonym_map) == "愧疚"


def test_canonicalize_canonical_points_to_itself():
    synonym_map = {"愧疚": "愧疚", "愧疚感": "愧疚"}
    assert canonicalize("愧疚", synonym_map) == "愧疚"


# --- load_synonym_map ---

def test_load_synonym_map_missing_file_returns_empty(tmp_path):
    result = load_synonym_map(tmp_path / "nonexistent.yaml")
    assert result == {}


def test_load_synonym_map_empty_groups_returns_empty(tmp_path):
    path = tmp_path / "syn.yaml"
    path.write_text("groups: []\n", encoding="utf-8")
    result = load_synonym_map(path)
    assert result == {}


def test_load_synonym_map_expands_groups_correctly(tmp_path):
    path = tmp_path / "syn.yaml"
    path.write_text(
        "groups:\n"
        "  - [愧疚, 愧疚感, 罪惡感]\n"
        "  - [沉默, 不說話]\n",
        encoding="utf-8",
    )
    result = load_synonym_map(path)
    assert result == {
        "愧疚": "愧疚",
        "愧疚感": "愧疚",
        "罪惡感": "愧疚",
        "沉默": "沉默",
        "不說話": "沉默",
    }


def test_load_synonym_map_default_path_uses_config_dir(tmp_path, monkeypatch):
    """When no path arg given, default resolves to config/symbol_synonyms.yaml."""
    project_root = tmp_path / "proj"
    (project_root / "config").mkdir(parents=True)
    (project_root / "config" / "symbol_synonyms.yaml").write_text(
        "groups:\n  - [A, B]\n", encoding="utf-8",
    )
    monkeypatch.setattr("empty_space.retrieval.DEFAULT_SYNONYMS_PATH",
                        project_root / "config" / "symbol_synonyms.yaml")
    result = load_synonym_map()
    assert result == {"A": "A", "B": "A"}


# --- expand_with_cooccurrence ---

def test_expand_empty_cooccurrence_returns_seeds():
    result = expand_with_cooccurrence(
        seed_symbols=["A", "B"],
        cooccurrence={},
        top_neighbors_per_seed=2,
    )
    assert result == ["A", "B"]


def test_expand_seed_without_neighbors_no_addition():
    cooc = {"B": {"X": 1}}  # A has no neighbors
    result = expand_with_cooccurrence(
        seed_symbols=["A"],
        cooccurrence=cooc,
        top_neighbors_per_seed=2,
    )
    assert result == ["A"]


def test_expand_takes_top_k_by_count():
    cooc = {
        "A": {"X": 5, "Y": 3, "Z": 1},
    }
    result = expand_with_cooccurrence(
        seed_symbols=["A"],
        cooccurrence=cooc,
        top_neighbors_per_seed=2,
    )
    assert result == ["A", "X", "Y"]  # Z excluded


def test_expand_tiebreak_alphabetical():
    cooc = {
        "A": {"Y": 2, "X": 2},  # Same count
    }
    result = expand_with_cooccurrence(
        seed_symbols=["A"],
        cooccurrence=cooc,
        top_neighbors_per_seed=2,
    )
    assert result == ["A", "X", "Y"]  # Alphabetical tiebreak


def test_expand_dedups_across_seeds():
    cooc = {
        "A": {"X": 3},
        "B": {"X": 2},
    }
    result = expand_with_cooccurrence(
        seed_symbols=["A", "B"],
        cooccurrence=cooc,
        top_neighbors_per_seed=1,
    )
    assert result == ["A", "B", "X"]  # X only once


def test_expand_preserves_seed_order():
    cooc = {"A": {"X": 1}}
    result = expand_with_cooccurrence(
        seed_symbols=["Z", "A", "M"],
        cooccurrence=cooc,
        top_neighbors_per_seed=1,
    )
    assert result[:3] == ["Z", "A", "M"]  # seeds first
    assert "X" in result[3:]


# --- merge_cooccurrence ---

def test_merge_cooccurrence_disjoint():
    a = {"X": {"Y": 1}}
    b = {"Z": {"W": 2}}
    result = merge_cooccurrence(a, b)
    assert result == {"X": {"Y": 1}, "Z": {"W": 2}}


def test_merge_cooccurrence_overlapping_keys_sums_counts():
    a = {"X": {"Y": 1, "Z": 2}}
    b = {"X": {"Y": 3, "W": 1}}
    result = merge_cooccurrence(a, b)
    assert result == {"X": {"Y": 4, "Z": 2, "W": 1}}


def test_merge_cooccurrence_does_not_mutate_inputs():
    a = {"X": {"Y": 1}}
    b = {"X": {"Y": 2}}
    result = merge_cooccurrence(a, b)
    assert a == {"X": {"Y": 1}}  # unchanged
    assert b == {"X": {"Y": 2}}  # unchanged
    assert result["X"]["Y"] == 3
