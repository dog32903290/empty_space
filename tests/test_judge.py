"""Unit tests for empty_space.judge — constants, parsers, ratchet, prompt, run."""
import pytest

from empty_space.judge import (
    MODES,
    STAGE_ORDER,
    apply_stage_target,
    parse_judge_principles,
    parse_stage_mode_contexts,
)
from empty_space.schemas import JudgeState


def _state(stage="前置積累", mode="在") -> JudgeState:
    return JudgeState(speaker_role="protagonist", stage=stage, mode=mode)


def test_stage_order_length_and_first_last():
    assert len(STAGE_ORDER) == 7
    assert STAGE_ORDER[0] == "前置積累"
    assert STAGE_ORDER[-1] == "基線"


def test_modes_are_three():
    assert MODES == ["收", "放", "在"]


def test_parse_judge_principles_is_identity():
    text = "鯨的下潛——肩內收。"
    assert parse_judge_principles(text) == text


def test_parse_stage_mode_contexts_extracts_cells():
    raw = {
        "前置積累_收": {
            "張力狀態": "拉力 > 推力",
            "身體": "鯨的下潛",
            "語言形態": "極短",
            "碎裂密度": "低",
        },
        "前置積累_在": {
            "身體": "鯨的巡游",
            "語言形態": "極少",
            "碎裂密度": "最低",
        },
    }
    parsed = parse_stage_mode_contexts(raw)
    assert "前置積累_收" in parsed
    assert parsed["前置積累_收"]["身體傾向"] == "鯨的下潛"
    assert parsed["前置積累_收"]["語聲傾向"] == "極短"
    assert parsed["前置積累_收"]["注意力"] == "拉力 > 推力"
    assert parsed["前置積累_在"]["身體傾向"] == "鯨的巡游"
    assert parsed["前置積累_在"]["語聲傾向"] == "極少"
    assert parsed["前置積累_在"]["注意力"] == ""  # 張力狀態 missing → empty


def test_parse_stage_mode_contexts_ignores_non_cell_keys():
    raw = {
        "前置積累_收": {"身體": "A", "語言形態": "B", "張力狀態": "C"},
        "comment": "this is not a cell",
        "metadata": {"version": "v3"},
    }
    parsed = parse_stage_mode_contexts(raw)
    assert set(parsed.keys()) == {"前置積累_收"}


def test_parse_stage_mode_contexts_empty_input():
    assert parse_stage_mode_contexts({}) == {}
    assert parse_stage_mode_contexts(None) == {}


def test_apply_stay():
    last = _state(stage="前置積累", mode="在")
    new, move = apply_stage_target(
        last_state=last, proposed_stage="前置積累",
        proposed_mode="在", proposed_verdict="N/A",
    )
    assert new.stage == "前置積累"
    assert move == "stay"


def test_apply_advance():
    last = _state(stage="前置積累", mode="在")
    new, move = apply_stage_target(
        last_state=last, proposed_stage="初感訊號",
        proposed_mode="收", proposed_verdict="N/A",
    )
    assert new.stage == "初感訊號"
    assert new.mode == "收"
    assert move == "advance"


def test_apply_regress():
    last = _state(stage="初感訊號", mode="收")
    new, move = apply_stage_target(
        last_state=last, proposed_stage="前置積累",
        proposed_mode="在", proposed_verdict="N/A",
    )
    assert new.stage == "前置積累"
    assert move == "regress"


def test_apply_illegal_jump_forces_stay():
    last = _state(stage="前置積累", mode="在")
    new, move = apply_stage_target(
        last_state=last, proposed_stage="明確切換",   # +3 jump, no fire_release
        proposed_mode="放", proposed_verdict="N/A",
    )
    assert new.stage == "前置積累"
    assert move == "illegal_stay"


def test_apply_fire_release_allows_plus_two():
    last = _state(stage="前置積累", mode="在")
    new, move = apply_stage_target(
        last_state=last, proposed_stage="半意識浮現",   # +2
        proposed_mode="放", proposed_verdict="fire_release",
    )
    assert new.stage == "半意識浮現"
    assert move == "fire_advance"


def test_apply_fire_release_does_not_allow_plus_three():
    last = _state(stage="前置積累", mode="在")
    new, move = apply_stage_target(
        last_state=last, proposed_stage="明確切換",   # +3 even under fire
        proposed_mode="放", proposed_verdict="fire_release",
    )
    assert new.stage == "前置積累"
    assert move == "illegal_stay"


def test_apply_basin_lock_forces_stay():
    last = _state(stage="穩定期", mode="在")
    new, move = apply_stage_target(
        last_state=last, proposed_stage="前置積累",   # Judge tried regress
        proposed_mode="收", proposed_verdict="basin_lock",
    )
    assert new.stage == "穩定期"
    assert move == "basin_stay"


def test_apply_mode_fallback_when_unknown():
    last = _state(stage="前置積累", mode="在")
    new, _ = apply_stage_target(
        last_state=last, proposed_stage="前置積累",
        proposed_mode="壓抑",   # not in MODES
        proposed_verdict="N/A",
    )
    assert new.mode == "在"   # fallback to last


def test_apply_mode_free_switch_within_legal():
    last = _state(stage="前置積累", mode="在")
    new, _ = apply_stage_target(
        last_state=last, proposed_stage="前置積累",
        proposed_mode="放", proposed_verdict="N/A",
    )
    assert new.mode == "放"


def test_apply_unknown_stage_name_forces_stay():
    last = _state(stage="前置積累", mode="在")
    new, move = apply_stage_target(
        last_state=last, proposed_stage="緩和期",   # not in STAGE_ORDER
        proposed_mode="收", proposed_verdict="N/A",
    )
    assert new.stage == "前置積累"
    assert move == "illegal_stay"


def test_apply_appends_move_history():
    last = _state()
    last.move_history = ["stay", "advance"]
    new, _ = apply_stage_target(
        last_state=last, proposed_stage="前置積累",
        proposed_mode="在", proposed_verdict="N/A",
    )
    assert new.move_history == ["stay", "advance", "stay"]
