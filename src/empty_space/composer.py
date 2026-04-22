"""Composer: session-end Pro bake that consolidates raw candidates into
refined impressions (short, atomic, first-person, register-aligned).

Output goes to two refined ledgers (by speaker). Called by runner at session end.
"""
from pathlib import Path

import yaml

from empty_space.ledger import read_refined_ledger
from empty_space.schemas import (
    CandidateImpression,
    ComposerInput,
    RefinedImpressionDraft,
    Turn,
)


COMPOSER_MODEL = "gemini-2.5-pro"


def parse_composer_output(
    raw_yaml: str,
    *,
    protagonist_name: str,
    counterpart_name: str,
) -> tuple[list[RefinedImpressionDraft], list[RefinedImpressionDraft], str | None]:
    """Parse Composer's YAML output. Returns (protagonist_drafts, counterpart_drafts, parse_error).

    Graceful degradation:
    - YAML parse failure → ([], [], error_msg)
    - Non-dict root → ([], [], error_msg)
    - Section key fuzzy match: if exact protagonist_name/counterpart_name not found,
      tries any key whose first character matches.
    - Item without `text` → silently skipped.
    - `symbols` missing → default [].
    - `source_raw_ids` missing → default [].
    """
    try:
        parsed = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as e:
        return [], [], f"YAML parse error: {e}"

    if not isinstance(parsed, dict):
        return [], [], f"composer output root is not a dict: {type(parsed).__name__}"

    # Find sections — exact match first, then fuzzy by first character
    p_section = _find_section(parsed, protagonist_name)
    c_section = _find_section(parsed, counterpart_name)

    p_drafts = _parse_section(p_section) if p_section is not None else []
    c_drafts = _parse_section(c_section) if c_section is not None else []

    return p_drafts, c_drafts, None


def _find_section(parsed: dict, name: str):
    """Find section by exact key, then by first-character fuzzy match."""
    if name in parsed:
        return parsed[name]
    # Fuzzy: any key starting with the same character
    if name:
        first_char = name[0]
        for key, val in parsed.items():
            if isinstance(key, str) and key and key[0] == first_char:
                return val
    return None


def _parse_section(section) -> list[RefinedImpressionDraft]:
    """Parse one section's list of impression items."""
    if not isinstance(section, list):
        return []
    drafts: list[RefinedImpressionDraft] = []
    for item in section:
        if not isinstance(item, dict) or "text" not in item:
            continue
        symbols = item.get("symbols") or []
        if not isinstance(symbols, list):
            symbols = []
        source_raw_ids = item.get("source_raw_ids") or []
        if not isinstance(source_raw_ids, list):
            source_raw_ids = []
        drafts.append(RefinedImpressionDraft(
            text=str(item["text"]),
            symbols=[str(s) for s in symbols],
            source_raw_ids=[str(s) for s in source_raw_ids],
        ))
    return drafts


_COMPOSER_SYSTEM_PROMPT = """\
你是劇場記憶的 consolidator。

你剛看完一段對話（兩個角色的 session）。兩個角色在 turn 之中各自產出了
自己的「候選印象」——原始、未經整理的感受片段。你的工作是把這些原料
**精煉**成簡短、atomic 的意象，讓下次他們再相遇時，這些精華會浮現在
他們的內在。

你**不是** summarize 對話。你是**提煉**他們內在留下的痕跡。

---

## 產出規則

**1. 第一人稱視角保持**
- 母親的 refined 用「你」第一人稱內在感受
- 兒子的 refined 同上
- 嚴禁第三人稱。若 atomic image 描述某人的狀態，歸屬給該人並用第一人稱重寫

**2. Atomic 原則**
- 每條控制在 15 字內
- 原始感官 / 動作 / 體感 — 不是 judgment 或 analysis
- 壞例：「她的沉默比辯解都沉」（judgment）
- 好例：「沉默時喉嚨收緊」（體感）
- 壞例：「他感覺到自己的牆沒有用」（analysis）
- 好例：「手指捏著衣角」（動作）

**3. 歸屬判斷**
- 每條 refined 歸屬給其中一個角色
- 原則：這條 refined 是誰的內在感受

**4. 不保留 judgment / analysis raw**
- Raw 中充滿判斷性、反思性的句子（「他的存在本身是一種重量」）
- 不要精煉這些。只提煉真正會沉到身體裡的片段

**5. Merge 或保留**
- 同 session 多個 raw 講同個瞬間 → 各自 refine 成不同 atomic image
- 不強求 merge 跨 speaker

**6. Symbols**
- 每條 refined 帶 2-4 個 symbols
- 和 raw 的 symbol 體系盡量對齊

**7. 精簡數量**
- 每角色產出 3-6 條 refined
- 若這 session 沒什麼沉澱，少產幾條也 OK

---

## 輸出格式

只輸出 YAML，不加任何解釋：

```
母親:
  - text: "沉默時喉嚨收緊"
    symbols: [沉默, 喉嚨, 收緊]
    source_raw_ids: [imp_003, imp_007]

兒子:
  - text: "背靠著牆坐"
    symbols: [背, 牆, 坐]
    source_raw_ids: [imp_002]
```

`source_raw_ids` 必填（若沒直接對應 raw，用 []）。
"""


def gather_composer_input(
    *,
    relationship: str,
    protagonist_name: str,
    counterpart_name: str,
    out_dir: Path,
    session_turns: list[Turn],
    new_raw_ids: dict[str, list[str]],
) -> ComposerInput:
    """Gather all materials Composer needs from the session."""
    conversation_text = (out_dir / "conversation.md").read_text(encoding="utf-8")

    new_candidates: dict[str, list[CandidateImpression]] = {
        "protagonist": [],
        "counterpart": [],
    }
    for turn in session_turns:
        new_candidates[turn.speaker].extend(turn.candidate_impressions)

    existing_p = read_refined_ledger(
        relationship=relationship, persona_name=protagonist_name,
    )
    existing_c = read_refined_ledger(
        relationship=relationship, persona_name=counterpart_name,
    )

    return ComposerInput(
        relationship=relationship,
        protagonist_name=protagonist_name,
        counterpart_name=counterpart_name,
        conversation_text=conversation_text,
        new_candidates=new_candidates,
        new_candidate_ids=new_raw_ids,
        existing_refined={
            "protagonist": existing_p.impressions[-30:],
            "counterpart": existing_c.impressions[-30:],
        },
    )


def build_composer_prompt(input: ComposerInput) -> tuple[str, str]:
    """Return (system_prompt, user_message) for Pro bake."""
    user_parts: list[str] = []

    user_parts.append("## Session 對話\n" + input.conversation_text.rstrip())

    # Raw candidates with ids
    p_raws = input.new_candidates["protagonist"]
    p_ids = input.new_candidate_ids.get("protagonist", [])
    user_parts.append(
        f"## {input.protagonist_name}的 Raw Candidates（本 session 新產出）\n"
        + _format_raw_list(p_raws, p_ids)
    )

    c_raws = input.new_candidates["counterpart"]
    c_ids = input.new_candidate_ids.get("counterpart", [])
    user_parts.append(
        f"## {input.counterpart_name}的 Raw Candidates（本 session 新產出）\n"
        + _format_raw_list(c_raws, c_ids)
    )

    # Existing refined (context)
    user_parts.append(
        f"## {input.protagonist_name}的既有 Refined Impressions（供參考語氣）\n"
        + _format_refined_list(input.existing_refined.get("protagonist", []))
    )
    user_parts.append(
        f"## {input.counterpart_name}的既有 Refined Impressions（供參考語氣）\n"
        + _format_refined_list(input.existing_refined.get("counterpart", []))
    )

    user_parts.append("---\n\n開始精煉。")

    return _COMPOSER_SYSTEM_PROMPT, "\n\n".join(user_parts)


def _format_raw_list(raws: list[CandidateImpression], ids: list[str]) -> str:
    if not raws:
        return "（無）"
    lines = []
    for i, raw in enumerate(raws):
        raw_id = ids[i] if i < len(ids) else f"imp_?{i}"
        symbols_str = ", ".join(raw.symbols) if raw.symbols else ""
        lines.append(f"- {raw_id}: {raw.text} [symbols: {symbols_str}]")
    return "\n".join(lines)


def _format_refined_list(refined) -> str:
    if not refined:
        return "（空）"
    lines = []
    for imp in refined:
        symbols_str = ", ".join(imp.symbols) if imp.symbols else ""
        lines.append(f"- {imp.id}: {imp.text} [symbols: {symbols_str}]")
    return "\n".join(lines)
