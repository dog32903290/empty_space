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


# --- retrieve_top_n ---

from empty_space.schemas import Ledger, LedgerEntry
from empty_space.retrieval import retrieve_top_n, run_session_start_retrieval


def _make_ledger(
    *,
    speaker: str = "protagonist",
    persona_name: str = "母親",
    entries: list[LedgerEntry] | None = None,
    cooccurrence: dict | None = None,
) -> Ledger:
    entries = entries or []
    cooc = cooccurrence or {}
    # Build symbol_index from entries
    symbol_index: dict[str, list[str]] = {}
    for e in entries:
        for s in e.symbols:
            symbol_index.setdefault(s, []).append(e.id)
    return Ledger(
        relationship="R",
        speaker=speaker,  # type: ignore[arg-type]
        persona_name=persona_name,
        ledger_version=len(entries),
        candidates=entries,
        symbol_index=symbol_index,
        cooccurrence=cooc,
    )


def _make_entry(id: str, text: str, symbols: list[str], created: str = "2026-04-21T10:00:00Z") -> LedgerEntry:
    return LedgerEntry(
        id=id, text=text, symbols=symbols,
        from_run="r/t", from_turn=1, created=created,
    )


def test_retrieve_empty_ledgers_returns_empty():
    result = retrieve_top_n(
        query_symbols=["A", "B"],
        ledger_a=_make_ledger(),
        ledger_b=_make_ledger(persona_name="兒子", speaker="counterpart"),
        synonym_map={},
        top_n=3,
    )
    assert result == []


def test_retrieve_single_ledger_hit():
    e1 = _make_entry("imp_001", "text 1", ["A", "B"])
    e2 = _make_entry("imp_002", "text 2", ["C"])
    ledger_a = _make_ledger(entries=[e1, e2])
    ledger_b = _make_ledger(persona_name="兒子", speaker="counterpart")

    result = retrieve_top_n(
        query_symbols=["A"],
        ledger_a=ledger_a,
        ledger_b=ledger_b,
        synonym_map={},
        top_n=3,
    )
    assert len(result) == 1
    assert result[0].id == "imp_001"
    assert result[0].score == 1
    assert result[0].matched_symbols == ("A",)
    assert result[0].speaker == "protagonist"
    assert result[0].persona_name == "母親"


def test_retrieve_cross_ledger_dedup_by_speaker_and_id():
    # Same id in both ledgers; dedup by (speaker, id)
    e_a = _make_entry("imp_001", "a text", ["A"])
    e_b = _make_entry("imp_001", "b text", ["A"])
    ledger_a = _make_ledger(entries=[e_a])
    ledger_b = _make_ledger(persona_name="兒子", speaker="counterpart", entries=[e_b])

    result = retrieve_top_n(
        query_symbols=["A"],
        ledger_a=ledger_a,
        ledger_b=ledger_b,
        synonym_map={},
        top_n=5,
    )
    # Both kept because (speaker, id) differs
    assert len(result) == 2
    speakers = {r.speaker for r in result}
    assert speakers == {"protagonist", "counterpart"}


def test_retrieve_sorts_by_score_desc():
    e_high = _make_entry("imp_001", "high", ["A", "B", "C"])
    e_low = _make_entry("imp_002", "low", ["A"])
    ledger_a = _make_ledger(entries=[e_high, e_low])
    ledger_b = _make_ledger(persona_name="兒子", speaker="counterpart")

    result = retrieve_top_n(
        query_symbols=["A", "B", "C"],
        ledger_a=ledger_a,
        ledger_b=ledger_b,
        synonym_map={},
        top_n=3,
    )
    assert [r.id for r in result] == ["imp_001", "imp_002"]
    assert result[0].score == 3
    assert result[1].score == 1


def test_retrieve_tiebreak_by_created_desc():
    e_old = _make_entry("imp_001", "old", ["A"], created="2026-04-21T10:00:00Z")
    e_new = _make_entry("imp_002", "new", ["A"], created="2026-04-22T10:00:00Z")
    ledger_a = _make_ledger(entries=[e_old, e_new])
    ledger_b = _make_ledger(persona_name="兒子", speaker="counterpart")

    result = retrieve_top_n(
        query_symbols=["A"],
        ledger_a=ledger_a,
        ledger_b=ledger_b,
        synonym_map={},
        top_n=3,
    )
    # Same score, newer first
    assert result[0].id == "imp_002"
    assert result[1].id == "imp_001"


def test_retrieve_synonym_map_matches_variants():
    synonym_map = {"愧疚": "愧疚", "愧疚感": "愧疚"}
    e = _make_entry("imp_001", "mother's regret", ["愧疚感"])
    ledger_a = _make_ledger(entries=[e])
    ledger_b = _make_ledger(persona_name="兒子", speaker="counterpart")

    result = retrieve_top_n(
        query_symbols=["愧疚"],
        ledger_a=ledger_a,
        ledger_b=ledger_b,
        synonym_map=synonym_map,
        top_n=3,
    )
    assert len(result) == 1
    assert result[0].matched_symbols == ("愧疚",)


def test_retrieve_top_n_truncates():
    entries = [_make_entry(f"imp_{i:03d}", f"t{i}", ["A"]) for i in range(1, 6)]
    ledger_a = _make_ledger(entries=entries)
    ledger_b = _make_ledger(persona_name="兒子", speaker="counterpart")

    result = retrieve_top_n(
        query_symbols=["A"],
        ledger_a=ledger_a,
        ledger_b=ledger_b,
        synonym_map={},
        top_n=3,
    )
    assert len(result) == 3


# --- run_session_start_retrieval ---

def test_run_session_start_retrieval_empty_query_returns_empty(tmp_path, monkeypatch):
    # Redirect LEDGERS_DIR
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path)

    client = _MockLLMClient(response_text="- x\n")
    result = run_session_start_retrieval(
        speaker_role="protagonist",
        persona_name="母親",
        query_text="",
        relationship="母親_x_兒子",
        other_persona_name="兒子",
        synonym_map={},
        llm_client=client,
        top_n=3,
    )
    assert result.query_symbols == []
    assert result.expanded_symbols == []
    assert result.impressions == []
    assert result.flash_tokens_in == 0
    assert client.calls == []


def test_run_session_start_retrieval_full_flow(tmp_path, monkeypatch):
    # Pre-populate a ledger via ledger.append_session_candidates
    from empty_space.ledger import append_session_candidates
    from empty_space.schemas import CandidateImpression

    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path)

    append_session_candidates(
        relationship="母親_x_兒子",
        speaker_role="counterpart",
        persona_name="兒子",
        candidates=[
            (8, CandidateImpression(
                text="她的沉默在這一刻比任何辯解都沉",
                symbols=["沉默", "辯解", "愧疚"],
            )),
        ],
        source_run="prev_exp/2026-04-21T09-00-00",
    )

    # Flash returns "愧疚" among extracted symbols
    client = _MockLLMClient(response_text="- 愧疚\n- 母愛\n")
    result = run_session_start_retrieval(
        speaker_role="protagonist",
        persona_name="母親",
        query_text="你昨夜夢到他小時候被帶走。",
        relationship="母親_x_兒子",
        other_persona_name="兒子",
        synonym_map={},
        llm_client=client,
        top_n=3,
    )

    assert result.query_symbols == ["愧疚", "母愛"]
    assert "愧疚" in result.expanded_symbols
    assert len(result.impressions) == 1
    assert result.impressions[0].text == "她的沉默在這一刻比任何辯解都沉"
    assert result.impressions[0].matched_symbols == ("愧疚",)
    assert result.impressions[0].speaker == "counterpart"
    assert result.flash_tokens_in > 0
