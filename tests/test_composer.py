"""Tests for Composer module — Pro bake for refined consolidation."""
import pytest

from empty_space.composer import parse_composer_output
from empty_space.schemas import RefinedImpressionDraft


def test_parse_clean_yaml_both_sections():
    raw = """母親:
  - text: "沉默時喉嚨收緊"
    symbols: [沉默, 喉嚨, 收緊]
    source_raw_ids: [imp_003, imp_007]
  - text: "手指在膝上按壓"
    symbols: [手指, 膝]
    source_raw_ids: [imp_004]

兒子:
  - text: "背靠著牆坐"
    symbols: [背, 牆, 坐]
    source_raw_ids: [imp_002]
"""
    p_drafts, c_drafts, err = parse_composer_output(raw, protagonist_name="母親", counterpart_name="兒子")
    assert err is None
    assert len(p_drafts) == 2
    assert len(c_drafts) == 1
    assert p_drafts[0].text == "沉默時喉嚨收緊"
    assert p_drafts[0].symbols == ["沉默", "喉嚨", "收緊"]
    assert p_drafts[0].source_raw_ids == ["imp_003", "imp_007"]
    assert p_drafts[1].text == "手指在膝上按壓"
    assert c_drafts[0].text == "背靠著牆坐"


def test_parse_only_protagonist_section():
    raw = """母親:
  - text: "x"
    symbols: [a]
    source_raw_ids: []
"""
    p_drafts, c_drafts, err = parse_composer_output(raw, protagonist_name="母親", counterpart_name="兒子")
    assert err is None
    assert len(p_drafts) == 1
    assert c_drafts == []


def test_parse_bad_yaml_returns_empty_with_error():
    raw = "母親:\n  - unclosed [\n  bad"
    p_drafts, c_drafts, err = parse_composer_output(raw, protagonist_name="母親", counterpart_name="兒子")
    assert p_drafts == []
    assert c_drafts == []
    assert err is not None
    assert "YAML" in err or "parse" in err.lower()


def test_parse_non_dict_root_returns_empty_with_error():
    raw = "- just a list\n- not a dict"
    p_drafts, c_drafts, err = parse_composer_output(raw, protagonist_name="母親", counterpart_name="兒子")
    assert p_drafts == []
    assert c_drafts == []
    assert err is not None


def test_parse_missing_text_skips_item():
    raw = """母親:
  - symbols: [no_text]
    source_raw_ids: []
  - text: "valid"
    symbols: [ok]
    source_raw_ids: []
"""
    p_drafts, c_drafts, err = parse_composer_output(raw, protagonist_name="母親", counterpart_name="兒子")
    assert err is None
    assert len(p_drafts) == 1
    assert p_drafts[0].text == "valid"


def test_parse_missing_symbols_defaults_empty():
    raw = """母親:
  - text: "no symbols"
    source_raw_ids: []
"""
    p_drafts, _, err = parse_composer_output(raw, protagonist_name="母親", counterpart_name="兒子")
    assert err is None
    assert p_drafts[0].symbols == []


def test_parse_missing_source_raw_ids_defaults_empty():
    raw = """母親:
  - text: "x"
    symbols: [a]
"""
    p_drafts, _, err = parse_composer_output(raw, protagonist_name="母親", counterpart_name="兒子")
    assert err is None
    assert p_drafts[0].source_raw_ids == []


def test_parse_empty_string_returns_empty_with_error():
    p_drafts, c_drafts, err = parse_composer_output("", protagonist_name="母親", counterpart_name="兒子")
    assert p_drafts == []
    assert c_drafts == []
    # yaml.safe_load("") returns None → non-dict root → error
    assert err is not None


def test_parse_both_sections_empty_lists():
    raw = """母親: []
兒子: []
"""
    p_drafts, c_drafts, err = parse_composer_output(raw, protagonist_name="母親", counterpart_name="兒子")
    assert err is None
    assert p_drafts == []
    assert c_drafts == []


def test_parse_speaker_key_fuzzy_match():
    """Pro may use '媽媽' instead of '母親' — should still route to protagonist (or drop gracefully)."""
    raw = """媽媽:
  - text: "fuzzy match"
    symbols: [a]
    source_raw_ids: []
"""
    p_drafts, c_drafts, err = parse_composer_output(raw, protagonist_name="母親", counterpart_name="兒子")
    # Fuzzy: if no exact match and '媽媽' starts with '媽' which doesn't start with '母',
    # fuzzy by first char won't match here. Both acceptable: routing (if algorithm handles it)
    # or drop. Just don't crash.
    assert err is None
    # Lenient: accept either route-to-protagonist OR drop entirely
    assert len(p_drafts) + len(c_drafts) <= 1
