"""Tests for refined ledger I/O — append-only, by-speaker."""
from pathlib import Path

import pytest
import yaml

from empty_space.schemas import RefinedImpressionDraft, RefinedLedger
from empty_space.ledger import (
    append_refined_impressions,
    read_refined_ledger,
    refined_ledger_path,
)


@pytest.fixture(autouse=True)
def redirect_ledgers_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path)


def test_refined_ledger_path_uses_refined_convention(tmp_path):
    path = refined_ledger_path(relationship="母親_x_兒子", persona_name="母親")
    assert path.name == "母親_x_兒子.refined.from_母親.yaml"
    assert path.parent == tmp_path


def test_read_refined_ledger_missing_returns_empty():
    ledger = read_refined_ledger(relationship="R", persona_name="母親")
    assert ledger.ledger_version == 0
    assert ledger.impressions == []
    assert ledger.symbol_index == {}
    assert ledger.cooccurrence == {}


def test_append_refined_first_time_creates_file():
    drafts = [
        RefinedImpressionDraft(
            text="沉默時喉嚨收緊",
            symbols=["沉默", "喉嚨", "收緊"],
            source_raw_ids=["imp_003"],
        ),
    ]
    version = append_refined_impressions(
        relationship="R",
        speaker_role="protagonist",
        persona_name="母親",
        drafts=drafts,
        source_run="exp/2026-04-22T10-00-00",
    )
    assert version == 1

    ledger = read_refined_ledger(relationship="R", persona_name="母親")
    assert ledger.ledger_version == 1
    assert len(ledger.impressions) == 1
    assert ledger.impressions[0].id == "ref_001"
    assert ledger.impressions[0].text == "沉默時喉嚨收緊"
    assert ledger.impressions[0].source_raw_ids == ["imp_003"]
    assert ledger.impressions[0].speaker == "protagonist"
    assert ledger.impressions[0].persona_name == "母親"
    assert ledger.impressions[0].from_run == "exp/2026-04-22T10-00-00"


def test_append_refined_id_increments_and_version_bumps():
    append_refined_impressions(
        relationship="R",
        speaker_role="protagonist",
        persona_name="母親",
        drafts=[RefinedImpressionDraft(text="first", symbols=["A"], source_raw_ids=[])],
        source_run="exp/t1",
    )
    version = append_refined_impressions(
        relationship="R",
        speaker_role="protagonist",
        persona_name="母親",
        drafts=[RefinedImpressionDraft(text="second", symbols=["B"], source_raw_ids=[])],
        source_run="exp/t2",
    )
    assert version == 2
    ledger = read_refined_ledger(relationship="R", persona_name="母親")
    assert [i.id for i in ledger.impressions] == ["ref_001", "ref_002"]


def test_append_refined_updates_symbol_index():
    drafts = [
        RefinedImpressionDraft(text="x", symbols=["A", "B"], source_raw_ids=[]),
        RefinedImpressionDraft(text="y", symbols=["B", "C"], source_raw_ids=[]),
    ]
    append_refined_impressions(
        relationship="R", speaker_role="protagonist", persona_name="母親",
        drafts=drafts, source_run="x/t",
    )
    ledger = read_refined_ledger(relationship="R", persona_name="母親")
    assert ledger.symbol_index == {
        "A": ["ref_001"],
        "B": ["ref_001", "ref_002"],
        "C": ["ref_002"],
    }


def test_append_refined_cooccurrence_symmetric():
    drafts = [
        RefinedImpressionDraft(text="x", symbols=["A", "B", "C"], source_raw_ids=[]),
    ]
    append_refined_impressions(
        relationship="R", speaker_role="protagonist", persona_name="母親",
        drafts=drafts, source_run="x/t",
    )
    ledger = read_refined_ledger(relationship="R", persona_name="母親")
    assert ledger.cooccurrence["A"] == {"B": 1, "C": 1}
    assert ledger.cooccurrence["B"] == {"A": 1, "C": 1}
    assert ledger.cooccurrence["C"] == {"A": 1, "B": 1}


def test_append_refined_single_symbol_no_cooccurrence():
    drafts = [
        RefinedImpressionDraft(text="x", symbols=["ONLY"], source_raw_ids=[]),
    ]
    append_refined_impressions(
        relationship="R", speaker_role="protagonist", persona_name="母親",
        drafts=drafts, source_run="x/t",
    )
    ledger = read_refined_ledger(relationship="R", persona_name="母親")
    assert ledger.symbol_index == {"ONLY": ["ref_001"]}
    assert "ONLY" not in ledger.cooccurrence


def test_append_refined_empty_drafts_is_noop():
    version = append_refined_impressions(
        relationship="R", speaker_role="protagonist", persona_name="母親",
        drafts=[], source_run="x/t",
    )
    # No file created, no version bump
    assert version == 0
    path = refined_ledger_path(relationship="R", persona_name="母親")
    assert not path.exists()
