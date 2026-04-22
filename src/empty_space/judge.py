"""Level 4 Judge: per-speaker (stage, mode) state machine driven by Gemini Flash.

Design:
- STAGE moves along 7-cell序列 via advance/stay/regress.
- MODE is free switching among 收/放/在.
- fire_release allows STAGE +2 jump; basin_lock forces stay.
- Judge never modifies dialogue; it only observes and labels.
"""
from __future__ import annotations

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
