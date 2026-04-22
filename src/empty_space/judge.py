"""Level 4 Judge: per-speaker (stage, mode) state machine driven by Gemini Flash.

Design:
- STAGE moves along 7-cell序列 via advance/stay/regress.
- MODE is free switching among 收/放/在.
- fire_release allows STAGE +2 jump; basin_lock forces stay.
- Judge never modifies dialogue; it only observes and labels.
"""
from __future__ import annotations

from empty_space.schemas import JudgeResult, JudgeState

STAGE_ORDER: list[str] = [
    "前置積累",
    "初感訊號",
    "半意識浮現",
    "明確切換",
    "穩定期",
    "回溫期",
    "基線",
]

MODES: list[str] = ["收", "放", "在"]

JUDGE_MODEL = "gemini-2.5-flash"


# --- YAML parsers for persona v3 files ---

def parse_judge_principles(text: str) -> str:
    """Identity — judge_principles_v3.yaml content is embedded verbatim
    in the Judge user prompt. Kept as a function so callers have a named
    seam if we later want to template-render it.
    """
    return text


# --- ratchet gate ---

def apply_stage_target(
    *,
    last_state: JudgeState,
    proposed_stage: str,
    proposed_mode: str,
    proposed_verdict: str,
    proposed_verb: str = "",
    why: str = "",
    hits: list[str] | None = None,
) -> tuple[JudgeState, str]:
    """Gate Judge's proposal through ratchet rules.

    Returns (new_state, move) where move ∈ {
        "advance", "stay", "regress",
        "illegal_stay",            # Judge named unknown stage OR tried illegal jump
        "fire_advance",            # fire_release allowed +2 jump
        "basin_stay",              # basin_lock forced stay despite proposed move
    }.
    """
    old_idx = STAGE_ORDER.index(last_state.stage)

    # 1. Resolve proposed stage index (unknown name → treat as stay attempt)
    if proposed_stage in STAGE_ORDER:
        new_idx = STAGE_ORDER.index(proposed_stage)
        diff = new_idx - old_idx
    else:
        new_idx = old_idx
        diff = 0
        move = "illegal_stay"
        return _build_new_state(
            last_state, new_idx, proposed_mode,
            proposed_verdict, proposed_verb, why, hits, move,
        ), move

    # 2. basin_lock overrides — force stay
    if proposed_verdict == "basin_lock":
        new_idx = old_idx
        move = "basin_stay"
        return _build_new_state(
            last_state, new_idx, proposed_mode,
            proposed_verdict, proposed_verb, why, hits, move,
        ), move

    # 3. fire_release exception — allow +2 jump (but not +3 or more)
    if proposed_verdict == "fire_release" and diff == 2:
        move = "fire_advance"
        return _build_new_state(
            last_state, new_idx, proposed_mode,
            proposed_verdict, proposed_verb, why, hits, move,
        ), move

    # 4. Normal ratchet: only -1, 0, +1 allowed
    if diff == 1:
        move = "advance"
    elif diff == 0:
        move = "stay"
    elif diff == -1:
        move = "regress"
    else:
        # Illegal jump (|diff| >= 2 without fire exception)
        new_idx = old_idx
        move = "illegal_stay"

    return _build_new_state(
        last_state, new_idx, proposed_mode,
        proposed_verdict, proposed_verb, why, hits, move,
    ), move


def _build_new_state(
    last_state: JudgeState,
    new_stage_idx: int,
    proposed_mode: str,
    verdict: str,
    proposed_verb: str,
    why: str,
    hits: list[str] | None,
    move: str,
) -> JudgeState:
    """Construct the new JudgeState with mode/verb fallback + history appends."""
    new_mode = proposed_mode if proposed_mode in MODES else last_state.mode
    new_verb = proposed_verb or last_state.current_verb
    new_hits = list(hits) if hits else []
    return JudgeState(
        speaker_role=last_state.speaker_role,
        stage=STAGE_ORDER[new_stage_idx],
        mode=new_mode,
        current_verb=new_verb,
        last_why=why,
        last_verdict=verdict,
        move_history=last_state.move_history + [move],
        verdict_history=last_state.verdict_history + [verdict],
        hits_history=last_state.hits_history + [new_hits],
    )


# --- tolerant output parser ---

_VERDICT_VALUES = {"fire_release", "basin_lock", "N/A"}


def parse_judge_output(text: str, *, last_state: JudgeState) -> JudgeResult:
    """Parse Flash's 6-line output tolerantly.

    Expected format:
        STAGE: <stage 名>
        MODE: <mode 名>
        VERB: <動詞>
        WHY: <一句>
        VERDICT: <fire_release | basin_lock | N/A>
        HITS: <line1; line2; line3>

    Tolerances:
    - Full-width colons (：) accepted
    - Preamble lines (not matching any field) are skipped
    - Stage/mode name substring/superstring of a canonical name → fuzzy match
    - Missing field → fallback to last_state (and mark parse_status)
    - Completely unparseable → fallback_used for all fields
    - VERB missing → silently falls back to last_state.current_verb (does not affect parse_status)
    - parse_status values:
        "ok": all three critical fields (STAGE/MODE/VERDICT) extracted (WHY/HITS/VERB don't affect status)
        "partial": at least one critical field missing but some found
        "fallback_used": no fields parsed at all
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    raw_stage = _extract_field(lines, ["STAGE:", "STAGE：", "階段:", "階段："])
    raw_mode = _extract_field(lines, ["MODE:", "MODE：", "模式:", "模式："])
    raw_verb = _extract_field(lines, ["VERB:", "VERB：", "動詞:", "動詞："])
    raw_verdict = _extract_field(lines, ["VERDICT:", "VERDICT："])
    raw_why = _extract_field(lines, ["WHY:", "WHY：", "為什麼:", "為什麼："])
    raw_hits = _extract_field(lines, ["HITS:", "HITS："])

    found_any = any(v is not None for v in (raw_stage, raw_mode, raw_verdict, raw_why, raw_hits))
    parse_status = "ok" if found_any else "fallback_used"

    stage = _normalise_stage(raw_stage or "", last_state.stage)
    mode = _normalise_mode(raw_mode or "", last_state.mode)
    # VERB: use extracted value if present, else fall back to last_state.current_verb
    verb = (raw_verb or "").strip() or last_state.current_verb
    verdict = _normalise_verdict(raw_verdict or "")
    why = (raw_why or "").strip()
    hits = _parse_hits(raw_hits or "")

    # If we got some fields but not all critical ones, mark partial
    if found_any and (raw_stage is None or raw_mode is None or raw_verdict is None):
        parse_status = "partial"

    return JudgeResult(
        proposed_stage=stage,
        proposed_mode=mode,
        proposed_verb=verb,
        proposed_verdict=verdict,
        why=why,
        hits=hits,
        meta={"parse_status": parse_status},
    )


def _extract_field(lines: list[str], prefixes: list[str]) -> str | None:
    """Return the first line's content after any matching prefix, or None."""
    for line in lines:
        for p in prefixes:
            if line.startswith(p):
                return line[len(p):].strip()
    return None


def _normalise_stage(raw: str, fallback: str) -> str:
    """Fuzzy match to STAGE_ORDER.

    Priority:
    1. Exact match.
    2. Canonical stage name is a substring of raw (e.g., "明確切換期" → "明確切換").
    3. Raw is substring of canonical, but ONLY if raw is at least 2 chars and
       within 1 char of canonical length (prevents "前" → "前置積累").
    4. Fallback to last_state.stage.
    """
    if raw in STAGE_ORDER:
        return raw
    for s in STAGE_ORDER:
        if s in raw:
            return s
    for s in STAGE_ORDER:
        if raw and len(raw) >= 2 and len(raw) >= len(s) - 1 and raw in s:
            return s
    return fallback


def _normalise_mode(raw: str, fallback: str) -> str:
    """Exact match to MODES (收/放/在). If the raw contains one of them as a char, accept."""
    if raw in MODES:
        return raw
    for m in MODES:
        if m in raw:
            return m
    return fallback


def _normalise_verdict(raw: str) -> str:
    """Only accept one of the three canonical verdicts; else N/A."""
    for v in _VERDICT_VALUES:
        if v in raw:
            return v
    return "N/A"


def _parse_hits(raw: str) -> list[str]:
    """Split HITS by ';' (半形) or '；' (全形). Empty and placeholder '-' filtered."""
    if not raw:
        return []
    parts = [p.strip() for chunk in raw.split(";") for p in chunk.split("；")]
    return [p for p in parts if p and p != "-"]


# --- YAML parsers for persona v3 files ---

# --- Judge prompt + run ---

_JUDGE_SYSTEM_PROMPT = """\
你是戲劇裡的「隱性量測者」。你不介入對話、不評分、不給建議。
你只做一件事：根據這個角色最近說的話、做的動作、身體狀態，
判斷他在 stage × mode × verb 三維空間裡「下一刻」會落在哪一格。

規則：
- STAGE 只能沿序列相鄰移動：前置積累 → 初感訊號 → 半意識浮現 → 明確切換 → 穩定期 → 回溫期 → 基線
  - 可以 advance（往下一格）、stay（同格）、regress（退回前一格）
  - 不能跳格
- MODE 是當下的身體傾向：收 / 放 / 在
  - 收：往內收斂、壓住、沉默、身體變小
  - 放：往外釋放、爆發、哭、笑、吼
  - 在：既不收也不放，只是存在、觀察、呼吸
- VERB 是角色此刻實際在做什麼的動詞（不是他此刻的狀態位置，是他此刻的意圖）：
  - 動詞通常是 persona 情緒動詞主軸的場景化變形（例：「承受」→「承受（靠近）」「承受（退回）」「承受（等待）」）
  - 如果你判斷角色此刻的意圖沒變，重複上一輪的 VERB
  - 如果角色的意圖變了（例如從靠近轉為退回、從承受轉為逃避），用新的動詞
  - 優先用 persona 的情緒動詞主軸 + 場景化修飾，不要自由發揮成無關的動詞
- VERDICT 標記特殊事件：
  - **若 persona 原則裡有 verdict_calibration 區塊，以該區塊的 signature / counterexamples 為最高準則**，而不是你的通用直覺。每個角色的 fire_release / basin_lock 門檻是他個性的一環，不要套絕對標準。
  - fire_release：對這個角色的個人尺度上剛剛發生了質變的情緒釋放
  - basin_lock：對這個角色來說真正進入了無動靜的穩態盆地（身體和語言都沒新信號）
  - N/A：其他情況
- HITS 是你觀察到的具體線索

輸出格式（6 行，嚴格）：
STAGE: <stage 名>
MODE: <mode 名>
VERB: <動詞>
WHY: <一句話>
VERDICT: <fire_release | basin_lock | N/A>
HITS: <line1; line2; line3>
"""


def build_judge_prompt(
    *,
    last_state: JudgeState,
    principles_text: str,
    stage_mode_contexts_text: str,
    recent_turns_text: str,
    speaker_role: str,
    persona_name: str,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for one Judge call."""
    user = f"""\
# 角色原則
{principles_text}

# Stage × Mode 脈絡
{stage_mode_contexts_text}

# 上一輪狀態
STAGE: {last_state.stage}
MODE: {last_state.mode}
VERB: {last_state.current_verb}
LAST_WHY: {last_state.last_why}

# 最近對話（最多 3 輪）
{recent_turns_text}

# 任務
根據以上，只判斷 {speaker_role}（{persona_name}）這個角色，
輸出他「剛說完這輪話之後」的 stage/mode/verb/why/verdict/hits。
"""
    return _JUDGE_SYSTEM_PROMPT, user


def run_judge(
    *,
    last_state: JudgeState,
    principles_text: str,
    stage_mode_contexts_text: str,
    recent_turns_text: str,
    speaker_role: str,
    persona_name: str,
    llm_client,
) -> JudgeResult:
    """One full Judge call. Returns JudgeResult; never raises.

    On LLM exception: returns a fallback result mirroring last_state with
    verdict='N/A' and meta.error set. Caller should pass this through
    apply_stage_target (which will then stay).
    """
    system, user = build_judge_prompt(
        last_state=last_state,
        principles_text=principles_text,
        stage_mode_contexts_text=stage_mode_contexts_text,
        recent_turns_text=recent_turns_text,
        speaker_role=speaker_role,
        persona_name=persona_name,
    )
    try:
        resp = llm_client.generate(system=system, user=user, model=JUDGE_MODEL)
    except Exception as e:
        return JudgeResult(
            proposed_stage=last_state.stage,
            proposed_mode=last_state.mode,
            proposed_verb=last_state.current_verb,
            proposed_verdict="N/A",
            why=f"[judge_error] {type(e).__name__}",
            hits=[],
            meta={
                "tokens_in": 0,
                "tokens_out": 0,
                "latency_ms": 0,
                "model": JUDGE_MODEL,
                "parse_status": "llm_error",
                "error": f"{type(e).__name__}: {e}",
            },
        )

    parsed = parse_judge_output(resp.content, last_state=last_state)
    parsed.meta.update({
        "tokens_in": resp.tokens_in,
        "tokens_out": resp.tokens_out,
        "latency_ms": resp.latency_ms,
        "model": resp.model,
    })
    return parsed


def is_fire_release(state: JudgeState) -> bool:
    return state.last_verdict == "fire_release"


def is_basin_lock(state: JudgeState) -> bool:
    return state.last_verdict == "basin_lock"


# --- turn-0 initial state inference from scene premise ---

_INFER_INITIAL_SYSTEM_PROMPT = """\
你是戲劇裡的「隱性量測者」。現在場景還沒開演，你只拿到導演寫的場景前提和角色的 prelude。
你的任務：判斷這個角色在戲**開始的第一秒**，會落在 stage × mode × verb 的哪一格。

規則：
- STAGE 序列：前置積累 → 初感訊號 → 半意識浮現 → 明確切換 → 穩定期 → 回溫期 → 基線
  - 戲劇開場通常落在「前置積累」或「穩定期」（低強度起手）或「回溫期」（承接上一場的餘韻）
  - 除非前提指明劇烈事件，否則不從「明確切換」或「半意識浮現」開始
- MODE：收 / 放 / 在
  - 這個角色**進場**時的身體傾向，從 persona 關係層的情緒動詞和反向記憶推導
- VERB：用 persona 關係層的情緒動詞主軸，加上場景具體修飾
  - 例：母親對兒子的主動詞是「承受」→ 這場戲她在「承受（靠近）」或「承受（等待）」或「承受（傾聽）」
  - 每個角色的 base verb 不同，不要套同一個

輸出格式（4 行，嚴格）：
STAGE: <stage 名>
MODE: <mode 名>
VERB: <動詞>
WHY: <一句為什麼這個組合>
"""


def _build_infer_prompt(
    *,
    speaker_role: str,
    persona_name: str,
    persona_core_text: str,
    persona_relationship_text: str,
    principles_text: str,
    scene_premise: str,
    prelude: str,
    other_persona_name: str,
) -> tuple[str, str]:
    user = f"""\
# 角色：{persona_name}（作為 {speaker_role}）
# 對手：{other_persona_name}

## 貫通軸
{persona_core_text}

## 關係層：對{other_persona_name}
{persona_relationship_text}

## 角色原則
{principles_text}

## 場景前提
{scene_premise}

## 這個角色的 prelude（導演給他的入場感）
{prelude or "（無 prelude）"}

# 任務
這個角色**剛踏進這場戲**的第一秒，stage / mode / verb 在哪裡？
"""
    return _INFER_INITIAL_SYSTEM_PROMPT, user


def _parse_infer_output(text: str) -> tuple[str, str, str, str]:
    """Parse 4-line inference output (STAGE/MODE/VERB/WHY). Returns tuple;
    fields default to empty if missing (caller provides fallback)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    stage = _extract_field(lines, ["STAGE:", "STAGE：", "階段:", "階段："]) or ""
    mode = _extract_field(lines, ["MODE:", "MODE：", "模式:", "模式："]) or ""
    verb = _extract_field(lines, ["VERB:", "VERB：", "動詞:", "動詞："]) or ""
    why = _extract_field(lines, ["WHY:", "WHY：", "為什麼:", "為什麼："]) or ""
    return stage, mode, verb, why


def infer_initial_state(
    *,
    speaker_role: str,
    persona_name: str,
    persona_core_text: str,
    persona_relationship_text: str,
    principles_text: str,
    scene_premise: str,
    prelude: str,
    other_persona_name: str,
    fallback_stage: str,
    fallback_mode: str,
    fallback_verb: str,
    llm_client,
) -> JudgeState:
    """Run a turn-0 Judge inference to place the character into stage×mode×verb
    based on scene premise + prelude + persona.

    On any failure (LLM error, parse failure), returns a JudgeState with the
    fallback values — so session can still start.
    """
    system, user = _build_infer_prompt(
        speaker_role=speaker_role,
        persona_name=persona_name,
        persona_core_text=persona_core_text,
        persona_relationship_text=persona_relationship_text,
        principles_text=principles_text,
        scene_premise=scene_premise,
        prelude=prelude,
        other_persona_name=other_persona_name,
    )

    try:
        resp = llm_client.generate(system=system, user=user, model=JUDGE_MODEL)
    except Exception:
        return JudgeState(
            speaker_role=speaker_role,  # type: ignore[arg-type]
            stage=fallback_stage,
            mode=fallback_mode,
            current_verb=fallback_verb,
        )

    raw_stage, raw_mode, raw_verb, _why = _parse_infer_output(resp.content)
    stage = _normalise_stage(raw_stage, fallback_stage)
    mode = _normalise_mode(raw_mode, fallback_mode)
    verb = raw_verb.strip() or fallback_verb

    return JudgeState(
        speaker_role=speaker_role,  # type: ignore[arg-type]
        stage=stage,
        mode=mode,
        current_verb=verb,
    )


# --- YAML parsers for persona v3 files ---

def parse_stage_mode_contexts(raw: dict | None) -> dict[str, dict[str, str]]:
    """Extract {"stage_mode": {身體傾向, 語聲傾向, 注意力}} from yaml dict.

    Input shape (stage_mode_contexts_v3.yaml): {<stage>_<mode>: {身體, 語言形態, 張力狀態, ...}}.
    Any top-level key whose value isn't a dict with at least one of those inner keys is skipped
    (e.g., top-level 'comment' or 'metadata').

    Output shape (for prompt_assembler):
        {"前置積累_收": {"身體傾向": "...", "語聲傾向": "...", "注意力": "..."}}

    Inner key mapping:
        身體傾向 ← 身體
        語聲傾向 ← 語言形態
        注意力   ← 張力狀態

    Missing inner fields default to "".
    """
    if not raw:
        return {}
    result: dict[str, dict[str, str]] = {}
    expected_inner = {"身體", "語言形態", "張力狀態"}
    for key, val in raw.items():
        if not isinstance(val, dict):
            continue
        if not (set(val.keys()) & expected_inner):
            continue
        result[key] = {
            "身體傾向": str(val.get("身體", "")),
            "語聲傾向": str(val.get("語言形態", "")),
            "注意力": str(val.get("張力狀態", "")),
        }
    return result
