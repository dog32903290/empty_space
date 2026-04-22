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
            proposed_verdict, why, hits, move,
        ), move

    # 2. basin_lock overrides — force stay
    if proposed_verdict == "basin_lock":
        new_idx = old_idx
        move = "basin_stay"
        return _build_new_state(
            last_state, new_idx, proposed_mode,
            proposed_verdict, why, hits, move,
        ), move

    # 3. fire_release exception — allow +2 jump (but not +3 or more)
    if proposed_verdict == "fire_release" and diff == 2:
        move = "fire_advance"
        return _build_new_state(
            last_state, new_idx, proposed_mode,
            proposed_verdict, why, hits, move,
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
        proposed_verdict, why, hits, move,
    ), move


def _build_new_state(
    last_state: JudgeState,
    new_stage_idx: int,
    proposed_mode: str,
    verdict: str,
    why: str,
    hits: list[str] | None,
    move: str,
) -> JudgeState:
    """Construct the new JudgeState with mode fallback + history appends."""
    new_mode = proposed_mode if proposed_mode in MODES else last_state.mode
    new_hits = list(hits) if hits else []
    return JudgeState(
        speaker_role=last_state.speaker_role,
        stage=STAGE_ORDER[new_stage_idx],
        mode=new_mode,
        last_why=why,
        last_verdict=verdict,
        move_history=last_state.move_history + [move],
        verdict_history=last_state.verdict_history + [verdict],
        hits_history=last_state.hits_history + [new_hits],
    )


# --- YAML parsers for persona v3 files ---

# --- tolerant output parser ---

_VERDICT_VALUES = {"fire_release", "basin_lock", "N/A"}


def parse_judge_output(text: str, *, last_state: JudgeState) -> JudgeResult:
    """Parse Flash's 5-line output tolerantly.

    Expected format:
        STAGE: <stage 名>
        MODE: <mode 名>
        WHY: <一句>
        VERDICT: <fire_release | basin_lock | N/A>
        HITS: <line1; line2; line3>

    Tolerances:
    - Full-width colons (：) accepted
    - Preamble lines (not matching any field) are skipped
    - Stage/mode name substring/superstring of a canonical name → fuzzy match
    - Missing field → fallback to last_state (and mark parse_status)
    - Completely unparseable → fallback_used for all fields
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    raw_stage = _extract_field(lines, ["STAGE:", "STAGE：", "階段:", "階段："])
    raw_mode = _extract_field(lines, ["MODE:", "MODE：", "模式:", "模式："])
    raw_verdict = _extract_field(lines, ["VERDICT:", "VERDICT："])
    raw_why = _extract_field(lines, ["WHY:", "WHY：", "為什麼:", "為什麼："])
    raw_hits = _extract_field(lines, ["HITS:", "HITS："])

    found_any = any(v is not None for v in (raw_stage, raw_mode, raw_verdict, raw_why, raw_hits))
    parse_status = "ok" if found_any else "fallback_used"

    stage = _normalise_stage(raw_stage or "", last_state.stage)
    mode = _normalise_mode(raw_mode or "", last_state.mode)
    verdict = _normalise_verdict(raw_verdict or "")
    why = (raw_why or "").strip()
    hits = _parse_hits(raw_hits or "")

    # If we got some fields but not all, mark partial
    if found_any and (raw_stage is None or raw_mode is None or raw_verdict is None):
        parse_status = "partial"

    return JudgeResult(
        proposed_stage=stage,
        proposed_mode=mode,
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
    """Fuzzy match to STAGE_ORDER. Exact match preferred; then substring both ways."""
    if raw in STAGE_ORDER:
        return raw
    for s in STAGE_ORDER:
        if s in raw or (raw and raw in s):
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
