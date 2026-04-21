"""Tests for prompt_assembler.build_system_prompt + build_user_message.

Covers spec §4 structure and §8.2 assertions.
"""
import pytest

from empty_space.prompt_assembler import build_system_prompt, build_user_message
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


# --- build_user_message ---

def _make_turn(n: int, speaker: str, name: str, content: str):
    from empty_space.schemas import Turn
    return Turn(
        turn_number=n,
        speaker=speaker,  # type: ignore[arg-type]
        persona_name=name,
        content=content,
        candidate_impressions=[],
        prompt_system="",
        prompt_user="",
        raw_response="",
        tokens_in=0,
        tokens_out=0,
        model="gemini-2.5-flash",
        latency_ms=0,
        timestamp="2026-04-21T11:30:00Z",
        director_events_active=[],
        parse_error=None,
    )


def test_user_message_turn_1_is_scene_opening():
    msg = build_user_message(history=[])
    assert msg == "（場景開始。）"


def test_user_message_turn_2_single_history_line():
    history = [_make_turn(1, "protagonist", "母親", "你回來了。")]
    msg = build_user_message(history=history)
    assert msg == "[Turn 1 母親] 你回來了。"


def test_user_message_turn_3_two_history_lines_in_order():
    history = [
        _make_turn(1, "protagonist", "母親", "你回來了。"),
        _make_turn(2, "counterpart", "兒子", "嗯。"),
    ]
    msg = build_user_message(history=history)
    assert msg == "[Turn 1 母親] 你回來了。\n[Turn 2 兒子] 嗯。"


def test_user_message_uses_persona_name_not_role_code():
    history = [_make_turn(1, "protagonist", "母親", "x")]
    msg = build_user_message(history=history)
    assert "母親" in msg
    assert "protagonist" not in msg


def test_user_message_has_no_tail_anchor_or_instruction():
    history = [
        _make_turn(1, "protagonist", "母親", "你回來了。"),
        _make_turn(2, "counterpart", "兒子", "嗯。"),
    ]
    msg = build_user_message(history=history)
    # No directive like "說第 N 句" or "你是 母親" appended
    assert "說第" not in msg
    assert "你是" not in msg
    assert msg.endswith("嗯。")


# --- Level 2: 你的內在 block ---

from empty_space.schemas import RetrievedImpression


def _make_retrieved(text: str, speaker: str = "counterpart") -> RetrievedImpression:
    return RetrievedImpression(
        id="imp_001",
        text=text,
        symbols=("x",),
        speaker=speaker,  # type: ignore[arg-type]
        persona_name="兒子",
        from_run="r/t",
        from_turn=1,
        score=1,
        matched_symbols=("x",),
    )


def test_system_prompt_omits_inner_block_when_no_prelude_and_no_retrieved(
    mother_persona, son_persona, hospital_setting, initial_state,
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
        prelude=None,
        retrieved_impressions=[],
    )
    assert "## 你的內在" not in prompt


def test_system_prompt_inner_block_with_only_prelude(
    mother_persona, son_persona, hospital_setting, initial_state,
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
        prelude="你昨夜夢到他小時候被帶走。",
        retrieved_impressions=[],
    )
    assert "## 你的內在" in prompt
    assert "你昨夜夢到他小時候被帶走。" in prompt
    assert "你可能想起的：" not in prompt


def test_system_prompt_inner_block_with_only_retrieved(
    mother_persona, son_persona, hospital_setting, initial_state,
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
        prelude=None,
        retrieved_impressions=[
            _make_retrieved("她的沉默在這一刻比任何辯解都沉"),
            _make_retrieved("他看著鞋帶不看她的眼"),
        ],
    )
    assert "## 你的內在" in prompt
    assert "你可能想起的：" in prompt
    assert "- 她的沉默在這一刻比任何辯解都沉" in prompt
    assert "- 他看著鞋帶不看她的眼" in prompt


def test_system_prompt_inner_block_with_both_prelude_and_retrieved(
    mother_persona, son_persona, hospital_setting, initial_state,
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
        prelude="你昨夜夢到他小時候被帶走。",
        retrieved_impressions=[_make_retrieved("她的沉默比辯解都沉")],
    )
    assert "## 你的內在" in prompt
    assert "你昨夜夢到他小時候被帶走。" in prompt
    assert "你可能想起的：" in prompt
    assert "- 她的沉默比辯解都沉" in prompt
    # prelude should come before retrieved within the block
    prelude_pos = prompt.find("你昨夜夢到他小時候被帶走。")
    recall_pos = prompt.find("你可能想起的：")
    assert 0 <= prelude_pos < recall_pos


def test_system_prompt_inner_block_position_after_xianchang(
    mother_persona, son_persona, hospital_setting, initial_state,
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise="父親 ICU。",
        initial_state=initial_state,
        active_events=[],
        prelude="你在想他。",
        retrieved_impressions=[],
    )
    # Order: 貫通軸 → 關係層 → 此刻 → 現場 → 你的內在 → 輸出格式
    assert prompt.index("## 現場") < prompt.index("## 你的內在")
    assert prompt.index("## 你的內在") < prompt.index("## 輸出格式")


def test_system_prompt_retrieved_impressions_rendered_as_bullet_list(
    mother_persona, son_persona, hospital_setting, initial_state,
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
        prelude=None,
        retrieved_impressions=[
            _make_retrieved("印象一"),
            _make_retrieved("印象二"),
            _make_retrieved("印象三"),
        ],
    )
    # Exact bullet format
    assert "你可能想起的：\n- 印象一\n- 印象二\n- 印象三" in prompt
