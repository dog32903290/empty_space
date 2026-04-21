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


# --- extract_symbols ---

from empty_space.llm import GeminiResponse
from empty_space.retrieval import extract_symbols


class _MockLLMClient:
    """Minimal mock for Flash extract_symbols tests."""
    def __init__(self, response_text: str, tokens_in: int = 100, tokens_out: int = 20, latency_ms: int = 150):
        self.response_text = response_text
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.latency_ms = latency_ms
        self.calls = []

    def generate(self, *, system: str, user: str, model: str = "gemini-2.5-flash") -> GeminiResponse:
        self.calls.append({"system": system, "user": user, "model": model})
        return GeminiResponse(
            content=self.response_text,
            raw=None,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            model=model,
            latency_ms=self.latency_ms,
        )


def test_extract_symbols_parses_clean_yaml_list():
    client = _MockLLMClient(response_text="- 分手\n- 女朋友\n- 拒絕\n")
    symbols, tokens_in, tokens_out, latency = extract_symbols(
        text="你昨晚和女朋友分手。",
        llm_client=client,
    )
    assert symbols == ["分手", "女朋友", "拒絕"]
    assert tokens_in == 100
    assert tokens_out == 20
    assert latency == 150
    assert len(client.calls) == 1


def test_extract_symbols_empty_text_skips_llm_call():
    client = _MockLLMClient(response_text="- x\n")
    symbols, tokens_in, tokens_out, latency = extract_symbols(
        text="",
        llm_client=client,
    )
    assert symbols == []
    assert tokens_in == 0
    assert tokens_out == 0
    assert latency == 0
    assert client.calls == []  # LLM was not called


def test_extract_symbols_whitespace_only_text_skips_llm():
    client = _MockLLMClient(response_text="- x\n")
    symbols, *_ = extract_symbols(
        text="   \n\n  ",
        llm_client=client,
    )
    assert symbols == []
    assert client.calls == []


def test_extract_symbols_invalid_yaml_returns_empty():
    client = _MockLLMClient(response_text="- unclosed [\n  bad")
    symbols, _, _, _ = extract_symbols(
        text="something",
        llm_client=client,
    )
    assert symbols == []


def test_extract_symbols_non_list_yaml_returns_empty():
    client = _MockLLMClient(response_text="key: value\n")
    symbols, _, _, _ = extract_symbols(
        text="something",
        llm_client=client,
    )
    assert symbols == []


def test_extract_symbols_strips_whitespace_and_filters_empty():
    client = _MockLLMClient(response_text="-   愧疚  \n- \n- 沉默\n")
    symbols, _, _, _ = extract_symbols(
        text="something",
        llm_client=client,
    )
    assert symbols == ["愧疚", "沉默"]
