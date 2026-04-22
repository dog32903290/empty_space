"""Unit tests for empty_space.judge — constants, parsers, ratchet, prompt, run."""
import pytest

from empty_space.judge import (
    MODES,
    STAGE_ORDER,
    parse_judge_principles,
    parse_stage_mode_contexts,
)


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
