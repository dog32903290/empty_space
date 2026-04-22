"""Level 4 schema additions: JudgeState, JudgeResult, Persona v3 fields,
SessionResult extensions, InitialState v3 vocabulary validator.
"""
import pytest

from empty_space.schemas import (
    ExperimentConfig,
    InitialState,
    JudgeResult,
    JudgeState,
    Persona,
    PersonaRef,
    SettingRef,
    Termination,
)


def test_judge_state_defaults():
    s = JudgeState(speaker_role="protagonist", stage="前置積累", mode="在")
    assert s.last_why == ""
    assert s.last_verdict == ""
    assert s.move_history == []
    assert s.verdict_history == []
    assert s.hits_history == []


def test_judge_result_defaults():
    r = JudgeResult(
        proposed_stage="前置積累",
        proposed_mode="收",
        proposed_verdict="N/A",
        why="",
        hits=[],
        meta={},
    )
    assert r.proposed_verdict == "N/A"


def test_persona_v3_fields_default_empty():
    p = Persona(name="母親", version="v3_tension", core_text="...")
    assert p.judge_principles_text == ""
    assert p.stage_mode_contexts_parsed == {}


def test_persona_v3_fields_assignable():
    p = Persona(
        name="母親",
        version="v3_tension",
        core_text="...",
        judge_principles_text="鯨的下潛...",
        stage_mode_contexts_parsed={
            "前置積累_收": {"身體傾向": "鯨的下潛", "語聲傾向": "極短", "注意力": "內收"},
        },
    )
    assert "鯨" in p.judge_principles_text
    assert p.stage_mode_contexts_parsed["前置積累_收"]["身體傾向"] == "鯨的下潛"


def test_initial_state_accepts_v3_mode():
    s = InitialState(verb="承受", stage="前置積累", mode="在")
    assert s.mode == "在"


def test_initial_state_migrates_legacy_baseline_to_在():
    """Legacy mode='基線' auto-migrated to '在' (軟遷移)."""
    s = InitialState(verb="承受", stage="前置積累", mode="基線")
    assert s.mode == "在"


def test_initial_state_legacy_migration_preserves_other_modes():
    s = InitialState(verb="承受", stage="前置積累", mode="收")
    assert s.mode == "收"
