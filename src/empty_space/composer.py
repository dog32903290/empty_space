"""Composer: session-end Pro bake that consolidates raw candidates into
refined impressions (short, atomic, first-person, register-aligned).

Output goes to two refined ledgers (by speaker). Called by runner at session end.
"""
import yaml

from empty_space.schemas import RefinedImpressionDraft


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
