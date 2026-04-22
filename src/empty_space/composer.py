"""Composer: session-end Pro bake that consolidates raw candidates into
refined impressions (short, atomic, first-person, register-aligned).

Output goes to two refined ledgers (by speaker). Called by runner at session end.
"""
from pathlib import Path

import yaml

from empty_space.ledger import append_refined_impressions, read_refined_ledger
from empty_space.schemas import (
    CandidateImpression,
    ComposerInput,
    ComposerSessionResult,
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
    # Strip markdown code fence that Pro often adds
    cleaned = _strip_code_fence(raw_yaml)
    try:
        parsed = yaml.safe_load(cleaned)
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


def _strip_code_fence(text: str) -> str:
    """Strip leading/trailing markdown code fence if present.

    Handles variants:
    - ``` ... ```
    - ```yaml ... ```
    - ```YAML ... ```
    - leading/trailing whitespace around fence
    """
    s = text.strip()
    if not s.startswith("```"):
        return text
    lines = s.split("\n")
    # Remove first fence line
    lines = lines[1:]
    # Remove trailing fence line if present
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


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

## Raw 優先級
部分 raws 帶有狀態標記：
- 🔥fire_release：角色情緒釋放時刻。優先從這些 raws 取材精煉
- 🪨basin_lock：盆地鎖住時刻。代表穩態下的核心印象，也值得優先
- 高張力-半意識浮現 / 高張力-明確切換：情緒弧線高張力區
- 無標記的 raws 屬於常態狀態，優先級次之

精煉時不要只從無標記 raws 取材；至少 50% 的 refined 要有標記 raws 參與。

---

你剛看完一段對話（兩個角色的 session）。兩個角色在 turn 之中各自產出了
自己的「候選印象」——雜亂、片段的感受記錄。你的工作是**從整段上下文**
提取**敘事單位**：這段時間裡發生了什麼事件，角色的身體如何承接它。

這些敘事單位會成為下次他們再相遇時的記憶 anchor。不是零散的感官瞬間，
而是「那天發生了 X，我身體是這樣的」的完整記憶片段。

---

## 產出原則

**1. 事件 anchor + 身體殘留**

每條 refined impression 包含兩個元素：
- **事件** — 這段時間發生了什麼（從 scene + 對話推出：父親過世、十幾年重逢、女朋友分手等）
- **身體感受** — 這個事件在角色身體上留下的殘留

壞例（純感官，無 anchor，像飄的）：
- 「喉嚨發緊」（什麼時候？因為什麼？無 anchor）
- 「視線停在他身上」

好例（事件 + 感受）：
- 「父親 ICU 那天，我喉嚨發緊，聲音又細又小」
- 「十幾年後的走廊，我的視線停在他身上」
- 「他十幾年後第一次出現，我的手縮回膝蓋上」

**2. 句型**

每條 25-40 字，結構：
- 「(事件/時空錨)，(身體感受/動作)」
- 或：「(事件)，我/他(如何)」

**3. 第一人稱保持**

母親的 refined 用「我」寫敘事；兒子的 refined 同上。不要第三人稱。

壞例：「她看著他，喉嚨發緊」（第三人稱）
好例：「他坐下那瞬間，我喉嚨發緊」（第一人稱）

**4. Symbols 混合事件詞 + 身體詞**

每條 2-5 個 symbols，混合三類詞彙：
- **事件詞**：父親、醫院、ICU、離開、分手、十幾年、相見、走廊
- **人物角色詞**：兒子、母親、她、他
- **身體/感受詞**：手、喉嚨、視線、發緊、沉默、躲藏

範例對應：
- 「父親 ICU 那天，我喉嚨發緊」→ symbols: [父親, ICU, 喉嚨, 發緊]
- 「十幾年後的走廊，我的視線停在他身上」→ symbols: [十幾年, 走廊, 兒子, 視線]

**這很重要**——混合 register 讓未來檢索能用事件詞（如「父親」「離開」）查到這些記憶。

**5. 不保留 raw 的 judgment / 敘述者 voice**

Raw 裡有些是敘述者視角的判斷（「她的苦是這個空間裡空氣密度的改變」、「他的存在本身是一種重量」）。這些**不要**。只留角色身體內在真實感受到的。

**6. 精簡數量**

每角色產 3-6 條 refined。少 OK。不強求填滿。

**7. 歸屬**

基於「這個敘事是誰的感受」。多數時候沿用 raw 的 speaker；但若某 raw 是兒子觀察母親的姿態（第三人稱），可以重寫成母親的第一人稱內在（歸給母親）。

---

## 輸入結構

你會收到：
- 這 session 的完整對話（conversation.md 格式；包含 scene_premise 和 prelude 作為 context）
- 母親這 session 產出的 raw candidates（帶 imp_XXX id）
- 兒子這 session 產出的 raw candidates（帶 imp_XXX id）
- （若有）母親現有的 refined impressions
- （若有）兒子現有的 refined impressions

參考既有 refined 的語氣，保持連續性。

---

## 輸出格式

只輸出 YAML，不加任何解釋或 markdown fence：

```
母親:
  - text: "父親 ICU 那天，我坐在走廊等，兒子出現時我視線停在他身上"
    symbols: [父親, ICU, 走廊, 兒子, 視線]
    source_raw_ids: [imp_003, imp_007]
  - text: "他坐下那瞬間，我喉嚨發緊，話吞回去了"
    symbols: [兒子, 坐下, 喉嚨, 發緊]
    source_raw_ids: [imp_004, imp_012]

兒子:
  - text: "接到電話那天走進醫院，她已經在走廊等了"
    symbols: [電話, 醫院, 走廊, 母親]
    source_raw_ids: [imp_001, imp_006]
  - text: "她靠近時我的背貼得更死，手握成拳"
    symbols: [母親, 靠近, 背, 手]
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
    candidate_states: dict[str, list[dict]] | None = None,
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
        new_candidate_states=candidate_states or {},
    )


def build_composer_prompt(input: ComposerInput) -> tuple[str, str]:
    """Return (system_prompt, user_message) for Pro bake."""
    user_parts: list[str] = []

    user_parts.append("## Session 對話\n" + input.conversation_text.rstrip())

    # Raw candidates with ids and state tags
    p_raws = input.new_candidates["protagonist"]
    p_ids = input.new_candidate_ids.get("protagonist", [])
    p_states = input.new_candidate_states.get("protagonist", [])
    user_parts.append(
        f"## {input.protagonist_name}的 Raw Candidates（本 session 新產出）\n"
        + _format_raw_list_with_states(p_raws, p_ids, p_states)
    )

    c_raws = input.new_candidates["counterpart"]
    c_ids = input.new_candidate_ids.get("counterpart", [])
    c_states = input.new_candidate_states.get("counterpart", [])
    user_parts.append(
        f"## {input.counterpart_name}的 Raw Candidates（本 session 新產出）\n"
        + _format_raw_list_with_states(c_raws, c_ids, c_states)
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


def _format_raw_with_state(raw_id: str, imp: CandidateImpression, state: dict | None) -> str:
    """Format one raw candidate line with optional state tag."""
    state_tag = ""
    if state:
        verdict = state.get("verdict", "N/A")
        stage = state.get("stage", "")
        tag_parts = []
        if verdict == "fire_release":
            tag_parts.append("🔥fire_release")
        elif verdict == "basin_lock":
            tag_parts.append("🪨basin_lock")
        if stage in ("半意識浮現", "明確切換"):
            tag_parts.append(f"高張力-{stage}")
        if tag_parts:
            state_tag = f" [{'|'.join(tag_parts)}]"
    symbols_str = ", ".join(imp.symbols) if imp.symbols else ""
    return f"- {raw_id}{state_tag}: {imp.text} [symbols: {symbols_str}]"


def _format_raw_list_with_states(
    raws: list[CandidateImpression],
    ids: list[str],
    states: list[dict],
) -> str:
    if not raws:
        return "（無）"
    lines = []
    for i, raw in enumerate(raws):
        raw_id = ids[i] if i < len(ids) else f"imp_?{i}"
        state = states[i] if i < len(states) else None
        lines.append(_format_raw_with_state(raw_id, raw, state))
    return "\n".join(lines)


def _format_raw_list(raws: list[CandidateImpression], ids: list[str]) -> str:
    """Legacy helper (no state tags). Kept for backward compatibility."""
    return _format_raw_list_with_states(raws, ids, [])


def _format_refined_list(refined) -> str:
    if not refined:
        return "（空）"
    lines = []
    for imp in refined:
        symbols_str = ", ".join(imp.symbols) if imp.symbols else ""
        lines.append(f"- {imp.id}: {imp.text} [symbols: {symbols_str}]")
    return "\n".join(lines)


def _enrich_drafts_with_states(
    drafts: list[RefinedImpressionDraft],
    state_map: dict[str, dict],
) -> list[RefinedImpressionDraft]:
    """For each draft, look up source_states from state_map via source_raw_ids."""
    enriched = []
    for draft in drafts:
        source_states = [
            state_map[rid] for rid in draft.source_raw_ids if rid in state_map
        ]
        enriched.append(RefinedImpressionDraft(
            text=draft.text,
            symbols=draft.symbols,
            source_raw_ids=draft.source_raw_ids,
            source_states=source_states,
        ))
    return enriched


def run_composer(
    *,
    relationship: str,
    protagonist_name: str,
    counterpart_name: str,
    out_dir: Path,
    session_turns: list[Turn],
    new_raw_ids: dict[str, list[str]],
    source_run: str,
    llm_client,
    state_maps: dict[str, dict[str, dict]] | None = None,
    candidate_states: dict[str, list[dict]] | None = None,
) -> ComposerSessionResult:
    """Top-level Composer orchestrator. Called by runner at session end.

    On any exception: returns ComposerSessionResult with parse_error set,
    tokens zero, no refined appended. Raw ledgers remain intact upstream.

    On success: produces 0-6 refined impressions per speaker, appends to
    two refined ledgers, returns counts and tokens for meta.yaml.

    state_maps: optional {"protagonist": {raw_id: state_dict}, "counterpart": {...}}
    When provided, enriches each refined draft with source_states metadata.

    candidate_states: optional {"protagonist": [state_dict, ...], "counterpart": [...]}
    When provided, Composer sees each raw's state tags (fire_release/basin_lock/high-stage).
    """
    state_maps = state_maps or {"protagonist": {}, "counterpart": {}}
    try:
        # 1. Gather input
        input_bundle = gather_composer_input(
            relationship=relationship,
            protagonist_name=protagonist_name,
            counterpart_name=counterpart_name,
            out_dir=out_dir,
            session_turns=session_turns,
            new_raw_ids=new_raw_ids,
            candidate_states=candidate_states,
        )

        # 2. Build prompt
        system, user = build_composer_prompt(input_bundle)

        # 3. Pro bake
        resp = llm_client.generate(system=system, user=user, model=COMPOSER_MODEL)

        # 4. Parse output
        p_drafts, c_drafts, parse_err = parse_composer_output(
            resp.content,
            protagonist_name=protagonist_name,
            counterpart_name=counterpart_name,
        )

        # 4b. Enrich drafts with source_states from state_maps
        p_drafts = _enrich_drafts_with_states(p_drafts, state_maps.get("protagonist", {}))
        c_drafts = _enrich_drafts_with_states(c_drafts, state_maps.get("counterpart", {}))

        # 5. Append to two ledgers
        append_refined_impressions(
            relationship=relationship,
            speaker_role="protagonist",
            persona_name=protagonist_name,
            drafts=p_drafts,
            source_run=source_run,
        )
        append_refined_impressions(
            relationship=relationship,
            speaker_role="counterpart",
            persona_name=counterpart_name,
            drafts=c_drafts,
            source_run=source_run,
        )

        return ComposerSessionResult(
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            latency_ms=resp.latency_ms,
            protagonist_refined_added=len(p_drafts),
            counterpart_refined_added=len(c_drafts),
            parse_error=parse_err,
        )

    except Exception as e:
        return ComposerSessionResult(
            tokens_in=0,
            tokens_out=0,
            latency_ms=0,
            protagonist_refined_added=0,
            counterpart_refined_added=0,
            parse_error=f"composer exception: {type(e).__name__}: {e}",
        )
