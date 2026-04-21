"""Tests for writer — run directory init, per-turn append, meta writeout."""
import json
from pathlib import Path

import pytest
import yaml

from empty_space.schemas import (
    ExperimentConfig,
    PersonaRef,
    SettingRef,
    InitialState,
    Termination,
)
from empty_space.writer import init_run


@pytest.fixture
def sample_config() -> ExperimentConfig:
    return ExperimentConfig(
        exp_id="test_exp_001",
        protagonist=PersonaRef(path="六個劇中人/母親", version="v3_tension"),
        counterpart=PersonaRef(path="六個劇中人/兒子", version="v3_tension"),
        setting=SettingRef(path="六個劇中人/環境_醫院.yaml"),
        scene_premise="父親在 ICU。",
        initial_state=InitialState(verb="承受", stage="前置積累", mode="基線"),
        director_events={3: "護士推床進來"},
        max_turns=5,
        termination=Termination(),
    )


def test_init_run_creates_directory_structure(tmp_path, sample_config):
    out_dir = tmp_path / "runs" / "test_exp_001" / "2026-04-21T11-30-15"
    init_run(out_dir, sample_config)
    assert out_dir.is_dir()
    assert (out_dir / "turns").is_dir()


def test_init_run_writes_config_yaml_deep_copy(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    config_file = out_dir / "config.yaml"
    assert config_file.is_file()
    loaded = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert loaded["exp_id"] == "test_exp_001"
    assert loaded["scene_premise"] == "父親在 ICU。"
    assert loaded["director_events"] == {3: "護士推床進來"}
    assert loaded["max_turns"] == 5


def test_init_run_initializes_conversation_md_with_scene(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    conv_md = (out_dir / "conversation.md").read_text(encoding="utf-8")
    assert "test_exp_001" in conv_md
    assert "父親在 ICU。" in conv_md


def test_init_run_creates_empty_conversation_jsonl(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    conv_jsonl = out_dir / "conversation.jsonl"
    assert conv_jsonl.is_file()
    assert conv_jsonl.read_text(encoding="utf-8") == ""


from empty_space.schemas import CandidateImpression, Turn
from empty_space.writer import append_turn


def _make_turn(
    *,
    turn_number: int,
    speaker: str,
    name: str,
    content: str,
    impressions: list[CandidateImpression] | None = None,
    events_active: list[tuple[int, str]] | None = None,
    parse_error: str | None = None,
) -> Turn:
    return Turn(
        turn_number=turn_number,
        speaker=speaker,  # type: ignore[arg-type]
        persona_name=name,
        content=content,
        candidate_impressions=impressions or [],
        prompt_system="(system prompt verbatim)",
        prompt_user="(user message verbatim)",
        raw_response=content + "\n",
        tokens_in=100,
        tokens_out=20,
        model="gemini-2.5-flash",
        latency_ms=300,
        timestamp="2026-04-21T11:30:01Z",
        director_events_active=events_active or [],
        parse_error=parse_error,
    )


def test_append_turn_writes_turn_yaml(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    turn = _make_turn(turn_number=1, speaker="protagonist", name="母親", content="你回來了。")
    append_turn(out_dir, turn)

    turn_file = out_dir / "turns" / "turn_001.yaml"
    assert turn_file.is_file()
    loaded = yaml.safe_load(turn_file.read_text(encoding="utf-8"))
    assert loaded["turn"] == 1
    assert loaded["speaker"] == "protagonist"
    assert loaded["response"]["content"] == "你回來了。"
    assert loaded["response"]["model"] == "gemini-2.5-flash"
    assert loaded["prompt_assembled"]["system"] == "(system prompt verbatim)"
    assert loaded["parse_error"] is None


def test_append_turn_numbers_are_zero_padded_to_three(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    turn = _make_turn(turn_number=12, speaker="counterpart", name="兒子", content="嗯。")
    append_turn(out_dir, turn)
    assert (out_dir / "turns" / "turn_012.yaml").is_file()


def test_append_turn_records_candidate_impressions(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    turn = _make_turn(
        turn_number=1,
        speaker="protagonist",
        name="母親",
        content="話。",
        impressions=[
            CandidateImpression(text="她的沉默很沉", symbols=["沉默", "愧疚"]),
        ],
    )
    append_turn(out_dir, turn)
    loaded = yaml.safe_load((out_dir / "turns" / "turn_001.yaml").read_text(encoding="utf-8"))
    assert len(loaded["candidate_impressions"]) == 1
    assert loaded["candidate_impressions"][0]["text"] == "她的沉默很沉"
    assert loaded["candidate_impressions"][0]["symbols"] == ["沉默", "愧疚"]


def test_append_turn_records_parse_error(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    turn = _make_turn(
        turn_number=1, speaker="protagonist", name="母親",
        content="話。", parse_error="YAML parse error: bad",
    )
    append_turn(out_dir, turn)
    loaded = yaml.safe_load((out_dir / "turns" / "turn_001.yaml").read_text(encoding="utf-8"))
    assert loaded["parse_error"] == "YAML parse error: bad"


def test_append_turn_records_director_events_active(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    turn = _make_turn(
        turn_number=4, speaker="counterpart", name="兒子", content="嗯。",
        events_active=[(3, "護士推床進來")],
    )
    append_turn(out_dir, turn)
    loaded = yaml.safe_load((out_dir / "turns" / "turn_004.yaml").read_text(encoding="utf-8"))
    assert loaded["director_events_active"] == [{"turn": 3, "content": "護士推床進來"}]


def test_append_turn_appends_to_conversation_md(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    append_turn(out_dir, _make_turn(
        turn_number=1, speaker="protagonist", name="母親", content="你回來了。",
    ))
    append_turn(out_dir, _make_turn(
        turn_number=2, speaker="counterpart", name="兒子", content="嗯。",
    ))
    md = (out_dir / "conversation.md").read_text(encoding="utf-8")
    assert "**Turn 1 · 母親**\n你回來了。" in md
    assert "**Turn 2 · 兒子**\n嗯。" in md


def test_append_turn_inserts_director_event_marker_before_triggering_turn(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    # Turn 3 triggered a new event at turn 3 — last element of events_active has turn == turn_number
    append_turn(out_dir, _make_turn(
        turn_number=3, speaker="protagonist", name="母親", content="⋯⋯",
        events_active=[(3, "護士推床進來")],
    ))
    md = (out_dir / "conversation.md").read_text(encoding="utf-8")
    marker_pos = md.find("**[世界] Turn 3：護士推床進來**")
    turn_header_pos = md.find("**Turn 3 · 母親**")
    assert marker_pos != -1
    assert turn_header_pos != -1
    assert marker_pos < turn_header_pos


def test_append_turn_no_event_marker_when_no_new_event_this_turn(tmp_path, sample_config):
    """Turn 5 sees event from Turn 3 in its active list but must not re-print marker."""
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    append_turn(out_dir, _make_turn(
        turn_number=5, speaker="protagonist", name="母親", content="話。",
        events_active=[(3, "護士推床進來")],  # old event, not triggered this turn
    ))
    md = (out_dir / "conversation.md").read_text(encoding="utf-8")
    assert "**[世界] Turn 3：護士推床進來**" not in md
    assert "**Turn 5 · 母親**" in md


def test_append_turn_appends_to_conversation_jsonl(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    append_turn(out_dir, _make_turn(
        turn_number=1, speaker="protagonist", name="母親", content="你回來了。",
    ))
    append_turn(out_dir, _make_turn(
        turn_number=3, speaker="protagonist", name="母親", content="⋯⋯",
        events_active=[(3, "護士推床進來")],
    ))
    lines = (out_dir / "conversation.jsonl").read_text(encoding="utf-8").strip().split("\n")
    # First turn entry, then director_event entry + turn entry for Turn 3
    entries = [json.loads(ln) for ln in lines]
    assert entries[0] == {
        "turn": 1, "speaker": "protagonist", "name": "母親",
        "content": "你回來了。", "timestamp": "2026-04-21T11:30:01Z",
    }
    assert entries[1] == {
        "type": "director_event", "turn": 3, "content": "護士推床進來",
    }
    assert entries[2]["turn"] == 3
    assert entries[2]["content"] == "⋯⋯"
