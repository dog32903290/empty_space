"""Pure functions that build the system prompt and user message for each turn.

Spec §4: five-block system prompt (貫通軸 / 關係層 / 此刻 / 現場 / 輸出格式),
user message is verbatim dialogue history with no tail anchor.
"""
from empty_space.schemas import (
    InitialState,
    Persona,
    Setting,
    Turn,
)


_OUTPUT_FORMAT_INSTRUCTION = """\
先寫你要說的話。說完之後，另起一行寫 "---IMPRESSIONS---"，然後以 YAML list 格式
列出你這輪浮現的印象句（若無，省略整段 ---IMPRESSIONS--- 區塊）。

範例：
---
她低著頭，沒有回答。

---IMPRESSIONS---
- text: "她的沉默在這一刻比任何辯解都沉"
  symbols: [沉默, 辯解, 愧疚]
- text: "她的手在膝上動了一下，又停了"
  symbols: [遲疑, 克制]
---"""


def build_system_prompt(
    persona: Persona,
    counterpart_name: str,
    setting: Setting,
    scene_premise: str | None,
    initial_state: InitialState,
    active_events: list[tuple[int, str]],
    ambient_echo: list[str] | None = None,
) -> str:
    """Assemble the system prompt for one role's turn.

    Block order (spec §4.1): 貫通軸 → 關係層 → 此刻 → 現場 → 輸出格式.
    `ambient_echo` is a Phase 4 hook; Phase 2 callers pass None or [].
    """
    _ = ambient_echo  # reserved for Phase 4
    relationship_text = persona.relationship_texts.get(counterpart_name, "")

    blocks: list[str] = []

    blocks.append(f"## 貫通軸\n{persona.core_text.rstrip()}")

    blocks.append(f"## 關係層：對{counterpart_name}\n{relationship_text.rstrip()}")

    blocks.append(
        "## 此刻\n"
        f"動作詞：{initial_state.verb}\n"
        f"階段：{initial_state.stage}\n"
        f"模式：{initial_state.mode}"
    )

    scene_parts: list[str] = [setting.content.rstrip()]
    if scene_premise:
        scene_parts.append(f"### 場景前提\n{scene_premise.rstrip()}")
    if active_events:
        event_lines = "\n".join(
            f"Turn {turn}：{content}" for turn, content in active_events
        )
        scene_parts.append(f"### 已發生的事\n{event_lines}")
    blocks.append("## 現場\n" + "\n\n".join(scene_parts))

    blocks.append(f"## 輸出格式\n{_OUTPUT_FORMAT_INSTRUCTION}")

    return "\n\n".join(blocks)
