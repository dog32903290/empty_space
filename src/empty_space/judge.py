"""Level 4 Judge: per-speaker (stage, mode) state machine driven by Gemini Flash.

Design:
- STAGE moves along 7-cell序列 via advance/stay/regress.
- MODE is free switching among 收/放/在.
- fire_release allows STAGE +2 jump; basin_lock forces stay.
- Judge never modifies dialogue; it only observes and labels.
"""
from __future__ import annotations

from empty_space.schemas import JudgeState

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
