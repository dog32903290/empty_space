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


from empty_space.judge import parse_judge_output


def test_parse_judge_output_happy_path():
    text = """STAGE: 初感訊號
MODE: 收
WHY: 母親縮肩且答話變短
VERDICT: N/A
HITS: 肩往下; 「嗯。」; 沉默 4 秒
"""
    last = _state(stage="前置積累", mode="在")
    r = parse_judge_output(text, last_state=last)
    assert r.proposed_stage == "初感訊號"
    assert r.proposed_mode == "收"
    assert r.proposed_verdict == "N/A"
    assert r.why == "母親縮肩且答話變短"
    assert r.hits == ["肩往下", "「嗯。」", "沉默 4 秒"]
    assert r.meta["parse_status"] == "ok"


def test_parse_judge_output_full_width_colons():
    text = """STAGE：初感訊號
MODE：收
WHY：母親縮肩
VERDICT：N/A
HITS：line1
"""
    last = _state()
    r = parse_judge_output(text, last_state=last)
    assert r.proposed_stage == "初感訊號"
    assert r.proposed_mode == "收"
    assert r.why == "母親縮肩"


def test_parse_judge_output_preamble_ignored():
    text = """好的，我來判斷：

STAGE: 前置積累
MODE: 在
WHY: 對話剛開始
VERDICT: N/A
HITS: -
"""
    last = _state()
    r = parse_judge_output(text, last_state=last)
    assert r.proposed_stage == "前置積累"
    assert r.proposed_mode == "在"


def test_parse_judge_output_stage_fuzzy_match():
    text = """STAGE: 明確切換期
MODE: 放
WHY: 爆發
VERDICT: fire_release
HITS: line1
"""
    last = _state()
    r = parse_judge_output(text, last_state=last)
    # Fuzzy substring match → 明確切換
    assert r.proposed_stage == "明確切換"


def test_parse_judge_output_missing_hits_line():
    text = """STAGE: 前置積累
MODE: 收
WHY: 沒線索
VERDICT: N/A
"""
    last = _state()
    r = parse_judge_output(text, last_state=last)
    assert r.hits == []
    assert r.meta["parse_status"] == "ok"


def test_parse_judge_output_totally_broken_falls_back():
    text = "the model said nothing useful"
    last = _state(stage="半意識浮現", mode="收")
    r = parse_judge_output(text, last_state=last)
    # Everything falls back to last_state
    assert r.proposed_stage == "半意識浮現"
    assert r.proposed_mode == "收"
    assert r.proposed_verdict == "N/A"
    assert r.meta["parse_status"] == "fallback_used"


def test_parse_judge_output_unknown_mode_falls_back():
    text = """STAGE: 前置積累
MODE: 壓抑
WHY: mode 詞彙錯
VERDICT: N/A
HITS: x
"""
    last = _state(stage="前置積累", mode="收")
    r = parse_judge_output(text, last_state=last)
    assert r.proposed_mode == "收"   # fallback


def test_parse_judge_output_unknown_verdict_becomes_na():
    text = """STAGE: 前置積累
MODE: 收
WHY: x
VERDICT: ignition
HITS: x
"""
    last = _state()
    r = parse_judge_output(text, last_state=last)
    assert r.proposed_verdict == "N/A"


def test_parse_judge_output_single_char_stage_falls_back():
    """Short junk like '前' must not match '前置積累' — should fallback to last."""
    text = """STAGE: 前
MODE: 收
WHY: x
VERDICT: N/A
HITS: -
"""
    last = _state(stage="半意識浮現", mode="在")
    r = parse_judge_output(text, last_state=last)
    assert r.proposed_stage == "半意識浮現"   # fallback


from empty_space.judge import (
    build_judge_prompt,
    is_basin_lock,
    is_fire_release,
    run_judge,
)
from empty_space.llm import GeminiResponse


class _MockLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.explode = False

    def generate(self, *, system, user, model="gemini-2.5-flash"):
        self.calls.append({"system": system, "user": user, "model": model})
        if self.explode:
            raise RuntimeError("network down")
        content = self.responses.pop(0)
        return GeminiResponse(
            content=content, raw=None,
            tokens_in=100, tokens_out=30, model=model, latency_ms=80,
        )


def test_build_judge_prompt_includes_last_state_and_persona():
    last = _state(stage="前置積累", mode="在")
    last.last_why = "上一句只說嗯"
    system, user = build_judge_prompt(
        last_state=last,
        principles_text="鯨的下潛——肩內收",
        stage_mode_contexts_text="前置積累_在：巡游",
        recent_turns_text="[Turn 1 母親] 嗯。",
        speaker_role="protagonist",
        persona_name="母親",
    )
    assert "STAGE:" in system
    assert "VERDICT:" in system
    assert "鯨的下潛" in user
    assert "巡游" in user
    assert "前置積累" in user
    assert "上一句只說嗯" in user
    assert "[Turn 1 母親] 嗯。" in user
    assert "母親" in user


def test_run_judge_happy_path():
    mock = _MockLLM(responses=[
        "STAGE: 初感訊號\nMODE: 收\nWHY: ok\nVERDICT: N/A\nHITS: -\n",
    ])
    last = _state(stage="前置積累", mode="在")
    result = run_judge(
        last_state=last,
        principles_text="p",
        stage_mode_contexts_text="c",
        recent_turns_text="t",
        speaker_role="protagonist",
        persona_name="母親",
        llm_client=mock,
    )
    assert result.proposed_stage == "初感訊號"
    assert result.proposed_mode == "收"
    assert result.meta["parse_status"] == "ok"
    assert result.meta["model"] == "gemini-2.5-flash"
    assert result.meta["tokens_in"] == 100


def test_run_judge_llm_exception_returns_fallback_result():
    mock = _MockLLM(responses=[])
    mock.explode = True
    last = _state(stage="半意識浮現", mode="收")
    result = run_judge(
        last_state=last,
        principles_text="p",
        stage_mode_contexts_text="c",
        recent_turns_text="t",
        speaker_role="protagonist",
        persona_name="母親",
        llm_client=mock,
    )
    # Fallback: last state preserved, verdict N/A, error recorded
    assert result.proposed_stage == "半意識浮現"
    assert result.proposed_mode == "收"
    assert result.proposed_verdict == "N/A"
    assert "error" in result.meta
    assert "network down" in result.meta["error"]


def test_is_fire_release_and_basin_lock():
    s = _state()
    s.last_verdict = "fire_release"
    assert is_fire_release(s) is True
    assert is_basin_lock(s) is False
    s.last_verdict = "basin_lock"
    assert is_fire_release(s) is False
    assert is_basin_lock(s) is True
    s.last_verdict = "N/A"
    assert is_fire_release(s) is False
    assert is_basin_lock(s) is False


# --- Level 4.4: Judge reads refined ledger ---

def test_build_judge_prompt_includes_refined_excerpt_when_provided():
    """When refined_excerpt is non-empty, user prompt contains the section header and text."""
    last = _state(stage="前置積累", mode="在")
    excerpt = "- ref_012: 十幾年後在醫院走廊見到他，視線只敢落在他身上一瞬。 [symbols: 十幾年, 醫院, 走廊]"
    system, user = build_judge_prompt(
        last_state=last,
        principles_text="p",
        stage_mode_contexts_text="c",
        recent_turns_text="t",
        speaker_role="protagonist",
        persona_name="母親",
        refined_excerpt=excerpt,
    )
    assert "角色的過去印象" in user
    assert "ref_012" in user
    assert "十幾年後在醫院走廊" in user


def test_build_judge_prompt_omits_section_when_excerpt_empty():
    """When refined_excerpt is empty string, the section is absent from user prompt."""
    last = _state(stage="前置積累", mode="在")
    system, user = build_judge_prompt(
        last_state=last,
        principles_text="p",
        stage_mode_contexts_text="c",
        recent_turns_text="t",
        speaker_role="protagonist",
        persona_name="母親",
        refined_excerpt="",
    )
    assert "角色的過去印象" not in user
    assert "refined ledger" not in user
