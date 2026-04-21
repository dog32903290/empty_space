"""Tests for parse_response — structured output extraction.

Covers the tolerance table in spec §6.2.
"""
from empty_space.parser import parse_response
from empty_space.schemas import CandidateImpression


def test_clean_format_main_and_impressions():
    raw = """她低著頭，沒有回答。

---IMPRESSIONS---
- text: "她的沉默在這一刻比任何辯解都沉"
  symbols: [沉默, 辯解, 愧疚]
- text: "她的手在膝上動了一下"
  symbols: [遲疑]
"""
    main, impressions, err = parse_response(raw)
    assert main == "她低著頭，沒有回答。"
    assert err is None
    assert len(impressions) == 2
    assert impressions[0] == CandidateImpression(
        text="她的沉默在這一刻比任何辯解都沉",
        symbols=["沉默", "辯解", "愧疚"],
    )
    assert impressions[1].symbols == ["遲疑"]


def test_no_marker_main_only():
    raw = "嗯。"
    main, impressions, err = parse_response(raw)
    assert main == "嗯。"
    assert impressions == []
    assert err is None


def test_marker_with_broken_yaml():
    raw = """她低著頭。

---IMPRESSIONS---
- text: "unclosed
  symbols: [沒關
"""
    main, impressions, err = parse_response(raw)
    assert main == "她低著頭。"
    assert impressions == []
    assert err is not None
    assert "YAML" in err


def test_marker_with_non_list_root():
    raw = """她低著頭。

---IMPRESSIONS---
text: 這不是 list
"""
    main, impressions, err = parse_response(raw)
    assert main == "她低著頭。"
    assert impressions == []
    assert err is not None
    assert "list" in err


def test_list_item_missing_text_is_skipped():
    raw = """她低著頭。

---IMPRESSIONS---
- symbols: [只有 symbols 沒有 text]
- text: "這個有 text"
  symbols: [good]
"""
    main, impressions, err = parse_response(raw)
    assert main == "她低著頭。"
    assert err is None                       # bad item is silently skipped
    assert len(impressions) == 1
    assert impressions[0].text == "這個有 text"


def test_symbols_default_to_empty_list():
    raw = """她低著頭。

---IMPRESSIONS---
- text: "沒有 symbols 欄"
"""
    main, impressions, err = parse_response(raw)
    assert main == "她低著頭。"
    assert err is None
    assert len(impressions) == 1
    assert impressions[0].text == "沒有 symbols 欄"
    assert impressions[0].symbols == []


def test_leading_and_trailing_whitespace_in_main():
    raw = """

她低著頭。


---IMPRESSIONS---
- text: "x"
"""
    main, impressions, err = parse_response(raw)
    assert main == "她低著頭。"
    assert impressions[0].text == "x"


def test_marker_only_no_impressions_block():
    """Marker followed by empty block: yaml.safe_load returns None → treated as no impressions."""
    raw = """她低著頭。
---IMPRESSIONS---
"""
    main, impressions, err = parse_response(raw)
    assert main == "她低著頭。"
    assert impressions == []
    assert err is None


def test_impressions_is_none_after_marker():
    """If YAML under marker parses to None, treat as empty list (not error)."""
    raw = """話。

---IMPRESSIONS---
# just a comment
"""
    main, impressions, err = parse_response(raw)
    assert main == "話。"
    assert impressions == []
