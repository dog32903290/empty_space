"""Tests for ledger.py — append-only per-speaker impression ledger."""
from pathlib import Path

import pytest
import yaml

from empty_space.schemas import CandidateImpression, Ledger
from empty_space.ledger import (
    append_session_candidates,
    ledger_path,
    read_ledger,
)


@pytest.fixture(autouse=True)
def redirect_ledgers_dir(tmp_path, monkeypatch):
    """Redirect LEDGERS_DIR to a tmp path for all tests."""
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path)


def test_read_ledger_missing_file_returns_empty():
    ledger = read_ledger(relationship="母親_x_兒子", persona_name="母親")
    assert ledger.relationship == "母親_x_兒子"
    assert ledger.persona_name == "母親"
    assert ledger.ledger_version == 0
    assert ledger.candidates == []
    assert ledger.symbol_index == {}
    assert ledger.cooccurrence == {}


def test_ledger_path_uses_from_speaker_convention(tmp_path):
    path = ledger_path(relationship="母親_x_兒子", persona_name="母親")
    assert path.name == "母親_x_兒子.from_母親.yaml"
    assert path.parent == tmp_path


def test_append_creates_file_with_candidates():
    candidates = [
        (1, CandidateImpression(text="消毒水的味道很淡", symbols=["消毒水", "淡"])),
        (1, CandidateImpression(text="她的手沒有動", symbols=["手", "不動"])),
    ]
    append_session_candidates(
        relationship="母親_x_兒子",
        speaker_role="protagonist",
        persona_name="母親",
        candidates=candidates,
        source_run="mother_x_son_hospital_v3_001/2026-04-21T10-24-12",
    )

    ledger = read_ledger(relationship="母親_x_兒子", persona_name="母親")
    assert ledger.ledger_version == 1
    assert len(ledger.candidates) == 2
    assert ledger.candidates[0].id == "imp_001"
    assert ledger.candidates[0].text == "消毒水的味道很淡"
    assert ledger.candidates[0].symbols == ["消毒水", "淡"]
    assert ledger.candidates[0].from_turn == 1
    assert ledger.candidates[0].from_run == "mother_x_son_hospital_v3_001/2026-04-21T10-24-12"
    assert ledger.candidates[1].id == "imp_002"


def test_append_updates_symbol_index():
    candidates = [
        (1, CandidateImpression(text="x", symbols=["A", "B"])),
        (2, CandidateImpression(text="y", symbols=["B", "C"])),
    ]
    append_session_candidates(
        relationship="R",
        speaker_role="protagonist",
        persona_name="母親",
        candidates=candidates,
        source_run="r/t",
    )
    ledger = read_ledger(relationship="R", persona_name="母親")
    assert ledger.symbol_index == {
        "A": ["imp_001"],
        "B": ["imp_001", "imp_002"],
        "C": ["imp_002"],
    }


def test_append_updates_cooccurrence_symmetric():
    candidates = [
        (1, CandidateImpression(text="x", symbols=["A", "B", "C"])),
    ]
    append_session_candidates(
        relationship="R",
        speaker_role="protagonist",
        persona_name="母親",
        candidates=candidates,
        source_run="r/t",
    )
    ledger = read_ledger(relationship="R", persona_name="母親")
    # All pairs: (A,B), (A,C), (B,C). Symmetric.
    assert ledger.cooccurrence["A"] == {"B": 1, "C": 1}
    assert ledger.cooccurrence["B"] == {"A": 1, "C": 1}
    assert ledger.cooccurrence["C"] == {"A": 1, "B": 1}


def test_append_single_symbol_candidate_no_cooccurrence():
    candidates = [
        (1, CandidateImpression(text="x", symbols=["ONLY"])),
    ]
    append_session_candidates(
        relationship="R",
        speaker_role="protagonist",
        persona_name="母親",
        candidates=candidates,
        source_run="r/t",
    )
    ledger = read_ledger(relationship="R", persona_name="母親")
    assert ledger.symbol_index == {"ONLY": ["imp_001"]}
    # No cooccurrence entry for ONLY (arity < 2)
    assert "ONLY" not in ledger.cooccurrence


def test_second_append_accumulates_correctly():
    # First append
    append_session_candidates(
        relationship="R",
        speaker_role="protagonist",
        persona_name="母親",
        candidates=[(1, CandidateImpression(text="first", symbols=["A", "B"]))],
        source_run="r/t1",
    )
    # Second append
    append_session_candidates(
        relationship="R",
        speaker_role="protagonist",
        persona_name="母親",
        candidates=[(2, CandidateImpression(text="second", symbols=["B", "C"]))],
        source_run="r/t2",
    )
    ledger = read_ledger(relationship="R", persona_name="母親")
    assert ledger.ledger_version == 2
    assert len(ledger.candidates) == 2
    assert ledger.candidates[0].id == "imp_001"
    assert ledger.candidates[1].id == "imp_002"
    assert ledger.candidates[1].from_run == "r/t2"
    # Cooccurrence accumulates
    assert ledger.cooccurrence["B"] == {"A": 1, "C": 1}


def test_append_empty_candidates_list_still_increments_version():
    append_session_candidates(
        relationship="R",
        speaker_role="protagonist",
        persona_name="母親",
        candidates=[],
        source_run="r/t",
    )
    ledger = read_ledger(relationship="R", persona_name="母親")
    # Empty list: file created, version = 1, no candidates
    assert ledger.ledger_version == 1
    assert ledger.candidates == []


def test_atomic_write_no_corrupt_on_replace_failure(monkeypatch):
    """If os.replace raises, the final file must not exist (tmp left is OK)."""
    def failing_replace(*args, **kwargs):
        raise OSError("simulated disk failure")
    monkeypatch.setattr("os.replace", failing_replace)

    with pytest.raises(OSError, match="simulated disk failure"):
        append_session_candidates(
            relationship="R",
            speaker_role="protagonist",
            persona_name="母親",
            candidates=[(1, CandidateImpression(text="x", symbols=["A"]))],
            source_run="r/t",
        )

    # Final file should not exist; tmp may exist but is not the canonical
    final = ledger_path(relationship="R", persona_name="母親")
    assert not final.exists()


def test_schema_round_trip_preserves_everything():
    """Write a ledger via append, read it back, all fields match."""
    candidates = [
        (5, CandidateImpression(text="原始文字", symbols=["符號1", "符號2"])),
    ]
    append_session_candidates(
        relationship="關係",
        speaker_role="counterpart",
        persona_name="兒子",
        candidates=candidates,
        source_run="exp_x/2026-04-21T12-00-00",
    )
    ledger = read_ledger(relationship="關係", persona_name="兒子")
    entry = ledger.candidates[0]
    assert entry.id == "imp_001"
    assert entry.text == "原始文字"
    assert entry.symbols == ["符號1", "符號2"]
    assert entry.from_run == "exp_x/2026-04-21T12-00-00"
    assert entry.from_turn == 5
    # created is ISO 8601 — just check it exists and looks like ISO
    assert entry.created  # non-empty
    assert "T" in entry.created
