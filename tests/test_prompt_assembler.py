"""Tests for prompt_assembler.build_system_prompt + build_user_message.

Covers spec §4 structure and §8.2 assertions.
"""
import pytest

from empty_space.prompt_assembler import build_system_prompt
from empty_space.schemas import (
    Persona,
    Setting,
    InitialState,
)


@pytest.fixture
def mother_persona() -> Persona:
    return Persona(
        name="母親",
        version="v3_tension",
        core_text="## 貫通軸\n動作詞: 承受\n",
        relationship_texts={"兒子": "## 關係語境\n母愛的退讓\n"},
    )


@pytest.fixture
def son_persona() -> Persona:
    return Persona(
        name="兒子",
        version="v3_tension",
        core_text="## 貫通軸\n動作詞: 迴避\n",
        relationship_texts={"母親": "## 關係語境\n不參與\n"},
    )


@pytest.fixture
def hospital_setting() -> Setting:
    return Setting(
        name="環境_醫院",
        content="## 設定\n壓低、懸吊、剝除\n",
    )


@pytest.fixture
def initial_state() -> InitialState:
    return InitialState(verb="承受（靠近）", stage="前置積累", mode="基線")


# --- build_system_prompt ---

def test_system_prompt_has_five_blocks_in_order(
    mother_persona, son_persona, hospital_setting, initial_state
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
    )
    # All five block headers present
    assert "## 貫通軸" in prompt
    assert "## 關係層：對兒子" in prompt
    assert "## 此刻" in prompt
    assert "## 現場" in prompt
    assert "## 輸出格式" in prompt
    # In correct order
    assert prompt.index("## 貫通軸") < prompt.index("## 關係層：對兒子")
    assert prompt.index("## 關係層：對兒子") < prompt.index("## 此刻")
    assert prompt.index("## 此刻") < prompt.index("## 現場")
    assert prompt.index("## 現場") < prompt.index("## 輸出格式")


def test_system_prompt_embeds_persona_core_text_verbatim(
    mother_persona, son_persona, hospital_setting, initial_state
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
    )
    assert mother_persona.core_text in prompt


def test_system_prompt_embeds_correct_relationship_text(
    mother_persona, son_persona, hospital_setting, initial_state
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
    )
    assert mother_persona.relationship_texts["兒子"] in prompt


def test_system_prompt_this_moment_block_formatting(
    mother_persona, son_persona, hospital_setting, initial_state
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
    )
    assert "動作詞：承受（靠近）" in prompt
    assert "階段：前置積累" in prompt
    assert "模式：基線" in prompt


def test_system_prompt_scene_premise_sub_block_omitted_when_none(
    mother_persona, son_persona, hospital_setting, initial_state
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
    )
    assert "### 場景前提" not in prompt


def test_system_prompt_scene_premise_appears_when_given(
    mother_persona, son_persona, hospital_setting, initial_state
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise="父親在 ICU。",
        initial_state=initial_state,
        active_events=[],
    )
    assert "### 場景前提" in prompt
    assert "父親在 ICU。" in prompt


def test_system_prompt_events_sub_block_omitted_when_empty(
    mother_persona, son_persona, hospital_setting, initial_state
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
    )
    assert "### 已發生的事" not in prompt


def test_system_prompt_events_listed_in_turn_order(
    mother_persona, son_persona, hospital_setting, initial_state
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[(3, "護士推空床進病房"), (10, "走廊傳來長音")],
    )
    assert "### 已發生的事" in prompt
    # Events appear in order, each with its turn prefix
    event3_pos = prompt.find("Turn 3：護士推空床進病房")
    event10_pos = prompt.find("Turn 10：走廊傳來長音")
    assert event3_pos != -1
    assert event10_pos != -1
    assert event3_pos < event10_pos


def test_system_prompt_output_format_contains_marker(
    mother_persona, son_persona, hospital_setting, initial_state
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
    )
    assert "---IMPRESSIONS---" in prompt


def test_system_prompt_setting_content_embedded(
    mother_persona, son_persona, hospital_setting, initial_state
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
    )
    assert hospital_setting.content in prompt
