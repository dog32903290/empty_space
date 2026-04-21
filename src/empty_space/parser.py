"""Parse Gemini Flash role responses into (main_content, impressions, parse_error).

Main response is always recovered. Impressions are optional — any parse failure
degrades gracefully to (main, [], error_message).
"""
import yaml

from empty_space.schemas import CandidateImpression

MARKER = "---IMPRESSIONS---"


def parse_response(
    raw: str,
) -> tuple[str, list[CandidateImpression], str | None]:
    """Split raw Gemini response into main response + list of impressions.

    Returns:
        (main_content, impressions, parse_error)
        - main_content: always non-None
        - impressions: parsed list, or [] on any error
        - parse_error: None if clean, else a short error string for turn yaml
    """
    if MARKER not in raw:
        return raw.strip(), [], None

    main_raw, _, impressions_block = raw.partition(MARKER)
    main = main_raw.strip()

    try:
        parsed = yaml.safe_load(impressions_block)
    except yaml.YAMLError as e:
        return main, [], f"YAML parse error: {e}"

    if parsed is None:
        # Empty / comment-only YAML block — treat as no impressions, not an error
        return main, [], None

    if not isinstance(parsed, list):
        return main, [], f"impressions block is not a list: {type(parsed).__name__}"

    impressions: list[CandidateImpression] = []
    for item in parsed:
        if not isinstance(item, dict) or "text" not in item:
            continue  # silently skip malformed items
        symbols_raw = item.get("symbols") or []
        symbols = [str(s) for s in symbols_raw] if isinstance(symbols_raw, list) else []
        impressions.append(
            CandidateImpression(
                text=str(item["text"]),
                symbols=symbols,
            )
        )
    return main, impressions, None
