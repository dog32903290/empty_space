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


# --- gather_composer_input + build_composer_prompt ---

from pathlib import Path
from empty_space.composer import build_composer_prompt, gather_composer_input
from empty_space.schemas import (
    CandidateImpression, ComposerInput, RefinedImpressionDraft, Turn,
)
from empty_space.ledger import append_refined_impressions


@pytest.fixture(autouse=True)
def redirect_ledgers_for_composer(tmp_path, monkeypatch):
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")


def _make_turn(turn_number: int, speaker: str, persona_name: str, content: str, candidates=None) -> Turn:
    return Turn(
        turn_number=turn_number,
        speaker=speaker,  # type: ignore[arg-type]
        persona_name=persona_name,
        content=content,
        candidate_impressions=candidates or [],
        prompt_system="",
        prompt_user="",
        raw_response="",
        tokens_in=0,
        tokens_out=0,
        model="gemini-2.5-flash",
        latency_ms=0,
        timestamp="2026-04-22T10:00:00Z",
        director_events_active=[],
        parse_error=None,
        retrieved_impressions=[],
    )


def test_gather_composer_input_reads_conversation_and_buckets_candidates(tmp_path):
    out_dir = tmp_path / "run_out"
    out_dir.mkdir()
    (out_dir / "conversation.md").write_text(
        "# test_exp @ 2026-04-22\n\n**Turn 1 · 母親**\n你回來了。\n",
        encoding="utf-8",
    )

    turns = [
        _make_turn(1, "protagonist", "母親", "你回來了。",
                   candidates=[CandidateImpression(text="她的手動了", symbols=["手"])]),
        _make_turn(2, "counterpart", "兒子", "嗯。",
                   candidates=[CandidateImpression(text="他沒看她", symbols=["目光"])]),
    ]
    new_raw_ids = {"protagonist": ["imp_001"], "counterpart": ["imp_001"]}

    input_bundle = gather_composer_input(
        relationship="母親_x_兒子",
        protagonist_name="母親",
        counterpart_name="兒子",
        out_dir=out_dir,
        session_turns=turns,
        new_raw_ids=new_raw_ids,
    )

    assert "**Turn 1 · 母親**" in input_bundle.conversation_text
    assert input_bundle.new_candidates["protagonist"][0].text == "她的手動了"
    assert input_bundle.new_candidates["counterpart"][0].text == "他沒看她"
    assert input_bundle.new_candidate_ids == new_raw_ids
    # Empty refined since no existing
    assert input_bundle.existing_refined["protagonist"] == []
    assert input_bundle.existing_refined["counterpart"] == []


def test_gather_composer_input_loads_existing_refined(tmp_path):
    out_dir = tmp_path / "run_out"
    out_dir.mkdir()
    (out_dir / "conversation.md").write_text("", encoding="utf-8")

    # Pre-seed refined ledger for protagonist
    append_refined_impressions(
        relationship="母親_x_兒子",
        speaker_role="protagonist",
        persona_name="母親",
        drafts=[RefinedImpressionDraft(text="previous refined", symbols=["a"], source_raw_ids=["imp_001"])],
        source_run="prev_exp/t",
    )

    input_bundle = gather_composer_input(
        relationship="母親_x_兒子",
        protagonist_name="母親",
        counterpart_name="兒子",
        out_dir=out_dir,
        session_turns=[],
        new_raw_ids={"protagonist": [], "counterpart": []},
    )

    assert len(input_bundle.existing_refined["protagonist"]) == 1
    assert input_bundle.existing_refined["protagonist"][0].text == "previous refined"
    assert input_bundle.existing_refined["counterpart"] == []


def test_gather_composer_input_takes_last_30_of_existing_refined(tmp_path):
    out_dir = tmp_path / "run_out"
    out_dir.mkdir()
    (out_dir / "conversation.md").write_text("", encoding="utf-8")

    # Pre-seed with 40 refined
    many = [
        RefinedImpressionDraft(text=f"ref text {i}", symbols=[f"s{i}"], source_raw_ids=[])
        for i in range(40)
    ]
    append_refined_impressions(
        relationship="母親_x_兒子",
        speaker_role="protagonist",
        persona_name="母親",
        drafts=many,
        source_run="prev/t",
    )

    input_bundle = gather_composer_input(
        relationship="母親_x_兒子",
        protagonist_name="母親",
        counterpart_name="兒子",
        out_dir=out_dir,
        session_turns=[],
        new_raw_ids={"protagonist": [], "counterpart": []},
    )
    assert len(input_bundle.existing_refined["protagonist"]) == 30
    # Should be last 30 (texts "ref text 10" through "ref text 39")
    assert input_bundle.existing_refined["protagonist"][0].text == "ref text 10"
    assert input_bundle.existing_refined["protagonist"][-1].text == "ref text 39"


# --- build_composer_prompt ---

def _minimal_input() -> ComposerInput:
    return ComposerInput(
        relationship="R",
        protagonist_name="母親",
        counterpart_name="兒子",
        conversation_text="**Turn 1 · 母親**\n你回來了。\n",
        new_candidates={
            "protagonist": [CandidateImpression(text="手動", symbols=["手"])],
            "counterpart": [CandidateImpression(text="眼神閃", symbols=["眼"])],
        },
        new_candidate_ids={
            "protagonist": ["imp_003"],
            "counterpart": ["imp_002"],
        },
        existing_refined={"protagonist": [], "counterpart": []},
    )


def test_build_composer_prompt_returns_system_and_user():
    system, user = build_composer_prompt(_minimal_input())
    assert isinstance(system, str)
    assert isinstance(user, str)
    assert len(system) > 100  # non-trivial
    assert len(user) > 50


def test_build_composer_prompt_user_contains_conversation():
    system, user = build_composer_prompt(_minimal_input())
    assert "你回來了。" in user


def test_build_composer_prompt_user_contains_raw_ids():
    system, user = build_composer_prompt(_minimal_input())
    # Raw candidates should appear with their ids
    assert "imp_003" in user
    assert "imp_002" in user


def test_build_composer_prompt_system_mentions_narrative_and_first_person():
    system, _ = build_composer_prompt(_minimal_input())
    # New prompt emphasizes narrative anchors + first-person
    assert "敘事" in system or "事件" in system
    assert "第一人稱" in system


def test_build_composer_prompt_user_contains_persona_names():
    system, user = build_composer_prompt(_minimal_input())
    # User message should have the persona names as section labels
    assert "母親" in user
    assert "兒子" in user


# --- run_composer orchestrator ---

from empty_space.composer import run_composer
from empty_space.llm import GeminiResponse
from empty_space.ledger import read_refined_ledger


class _MockLLMForComposer:
    def __init__(self, response_text: str, tokens_in: int = 2000, tokens_out: int = 500, latency_ms: int = 15000):
        self.response_text = response_text
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.latency_ms = latency_ms
        self.calls = []

    def generate(self, *, system, user, model="gemini-2.5-pro"):
        self.calls.append({"system": system, "user": user, "model": model})
        return GeminiResponse(
            content=self.response_text, raw=None,
            tokens_in=self.tokens_in, tokens_out=self.tokens_out,
            model=model, latency_ms=self.latency_ms,
        )


def test_run_composer_happy_path(tmp_path, monkeypatch):
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")
    out_dir = tmp_path / "run_out"
    out_dir.mkdir()
    (out_dir / "conversation.md").write_text(
        "**Turn 1 · 母親**\n你回來了。\n", encoding="utf-8",
    )

    turns = [
        _make_turn(1, "protagonist", "母親", "你回來了。",
                   candidates=[CandidateImpression(text="她的手動了", symbols=["手"])]),
        _make_turn(2, "counterpart", "兒子", "嗯。",
                   candidates=[CandidateImpression(text="他沒看她", symbols=["目光"])]),
    ]

    client = _MockLLMForComposer(response_text="""母親:
  - text: "手動了"
    symbols: [手]
    source_raw_ids: [imp_001]

兒子:
  - text: "沒看她"
    symbols: [目光]
    source_raw_ids: [imp_002]
""")

    result = run_composer(
        relationship="母親_x_兒子",
        protagonist_name="母親",
        counterpart_name="兒子",
        out_dir=out_dir,
        session_turns=turns,
        new_raw_ids={"protagonist": ["imp_001"], "counterpart": ["imp_002"]},
        source_run="exp/2026-04-22T10-00-00",
        llm_client=client,
    )

    assert result.parse_error is None
    assert result.protagonist_refined_added == 1
    assert result.counterpart_refined_added == 1
    assert result.tokens_in > 0
    assert result.tokens_out > 0
    assert len(client.calls) == 1
    assert client.calls[0]["model"] == "gemini-2.5-pro"

    # Verify refined ledgers written
    p_ledger = read_refined_ledger(relationship="母親_x_兒子", persona_name="母親")
    c_ledger = read_refined_ledger(relationship="母親_x_兒子", persona_name="兒子")
    assert p_ledger.ledger_version == 1
    assert c_ledger.ledger_version == 1
    assert p_ledger.impressions[0].text == "手動了"
    assert c_ledger.impressions[0].text == "沒看她"


def test_run_composer_pro_exception_caught(tmp_path, monkeypatch):
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")
    out_dir = tmp_path / "run_out"
    out_dir.mkdir()
    (out_dir / "conversation.md").write_text("", encoding="utf-8")

    class ExplodingClient:
        def generate(self, *, system, user, model="gemini-2.5-pro"):
            raise RuntimeError("pro api down")

    result = run_composer(
        relationship="R", protagonist_name="母親", counterpart_name="兒子",
        out_dir=out_dir, session_turns=[],
        new_raw_ids={"protagonist": [], "counterpart": []},
        source_run="exp/t", llm_client=ExplodingClient(),
    )

    assert result.parse_error is not None
    assert "pro api down" in result.parse_error or "RuntimeError" in result.parse_error
    assert result.protagonist_refined_added == 0
    assert result.counterpart_refined_added == 0
    # No ledgers written
    assert not (tmp_path / "ledgers" / "R.refined.from_母親.yaml").exists()


def test_run_composer_bad_yaml_returns_zero_appends(tmp_path, monkeypatch):
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")
    out_dir = tmp_path / "run_out"
    out_dir.mkdir()
    (out_dir / "conversation.md").write_text("", encoding="utf-8")

    client = _MockLLMForComposer(response_text="garbage [[[ not yaml")
    result = run_composer(
        relationship="R", protagonist_name="母親", counterpart_name="兒子",
        out_dir=out_dir, session_turns=[],
        new_raw_ids={"protagonist": [], "counterpart": []},
        source_run="exp/t", llm_client=client,
    )

    assert result.parse_error is not None
    assert result.protagonist_refined_added == 0
    assert result.counterpart_refined_added == 0


def test_run_composer_partial_success_only_protagonist(tmp_path, monkeypatch):
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")
    out_dir = tmp_path / "run_out"
    out_dir.mkdir()
    (out_dir / "conversation.md").write_text("", encoding="utf-8")

    client = _MockLLMForComposer(response_text="""母親:
  - text: "only mother"
    symbols: [a]
    source_raw_ids: []

兒子: []
""")
    result = run_composer(
        relationship="R", protagonist_name="母親", counterpart_name="兒子",
        out_dir=out_dir, session_turns=[],
        new_raw_ids={"protagonist": [], "counterpart": []},
        source_run="exp/t", llm_client=client,
    )

    assert result.parse_error is None
    assert result.protagonist_refined_added == 1
    assert result.counterpart_refined_added == 0

    # Protagonist ledger written, counterpart not
    p_path = tmp_path / "ledgers" / "R.refined.from_母親.yaml"
    c_path = tmp_path / "ledgers" / "R.refined.from_兒子.yaml"
    assert p_path.exists()
    assert not c_path.exists()  # empty drafts → no file created (per append_refined_impressions)


def test_parse_strips_yaml_code_fence():
    """Pro often wraps output in ```yaml ... ``` fence."""
    raw = """```yaml
母親:
  - text: "被 fence 包住的輸出"
    symbols: [a]
    source_raw_ids: []
```"""
    p_drafts, c_drafts, err = parse_composer_output(raw, protagonist_name="母親", counterpart_name="兒子")
    assert err is None
    assert len(p_drafts) == 1
    assert p_drafts[0].text == "被 fence 包住的輸出"


def test_parse_strips_plain_code_fence():
    """Fence without language tag."""
    raw = """```
母親:
  - text: "無 lang tag"
    symbols: [a]
    source_raw_ids: []
```"""
    p_drafts, _, err = parse_composer_output(raw, protagonist_name="母親", counterpart_name="兒子")
    assert err is None
    assert p_drafts[0].text == "無 lang tag"


def test_parse_no_fence_unchanged():
    """Without fence, parsing should work as before."""
    raw = """母親:
  - text: "no fence"
    symbols: [a]
    source_raw_ids: []
"""
    p_drafts, _, err = parse_composer_output(raw, protagonist_name="母親", counterpart_name="兒子")
    assert err is None
    assert p_drafts[0].text == "no fence"


# --- Level 4.2: state_maps enrichment ---

from empty_space.composer import _enrich_drafts_with_states


def test_enrich_drafts_with_states_attaches_matching_states():
    """_enrich_drafts_with_states maps source_raw_ids → state_dicts from state_map."""
    state_map = {
        "imp_001": {"turn": 1, "stage": "前置積累", "mode": "收", "verb": "承受", "verdict": "N/A"},
        "imp_002": {"turn": 2, "stage": "初感訊號", "mode": "在", "verb": "迴避", "verdict": "N/A"},
    }
    drafts = [
        RefinedImpressionDraft(text="x", symbols=["a"], source_raw_ids=["imp_001", "imp_002"]),
        RefinedImpressionDraft(text="y", symbols=["b"], source_raw_ids=["imp_999"]),  # not in map
    ]
    enriched = _enrich_drafts_with_states(drafts, state_map)

    assert len(enriched) == 2
    # First draft: both ids in map
    assert len(enriched[0].source_states) == 2
    assert enriched[0].source_states[0]["turn"] == 1
    assert enriched[0].source_states[1]["stage"] == "初感訊號"
    # Second draft: id not in map → empty
    assert enriched[1].source_states == []


def test_enrich_drafts_empty_state_map():
    """Empty state_map → all drafts get empty source_states."""
    drafts = [
        RefinedImpressionDraft(text="x", symbols=["a"], source_raw_ids=["imp_001"]),
    ]
    enriched = _enrich_drafts_with_states(drafts, {})
    assert enriched[0].source_states == []


def test_run_composer_enriches_drafts_with_source_states(tmp_path, monkeypatch):
    """state_maps passed to run_composer → refined drafts carry source_states in ledger."""
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")
    out_dir = tmp_path / "run_out"
    out_dir.mkdir()
    (out_dir / "conversation.md").write_text(
        "**Turn 1 · 母親**\n你回來了。\n", encoding="utf-8",
    )

    state_maps = {
        "protagonist": {
            "imp_001": {"turn": 1, "stage": "初感訊號", "mode": "收", "verb": "承受", "verdict": "N/A"},
        },
        "counterpart": {},
    }

    client = _MockLLMForComposer(response_text="""母親:
  - text: "state attached"
    symbols: [手]
    source_raw_ids: [imp_001]

兒子: []
""")

    result = run_composer(
        relationship="母親_x_兒子",
        protagonist_name="母親",
        counterpart_name="兒子",
        out_dir=out_dir,
        session_turns=[],
        new_raw_ids={"protagonist": ["imp_001"], "counterpart": []},
        source_run="exp/t",
        llm_client=client,
        state_maps=state_maps,
    )

    assert result.parse_error is None
    assert result.protagonist_refined_added == 1

    # Verify source_states written to ledger
    p_ledger = read_refined_ledger(relationship="母親_x_兒子", persona_name="母親")
    assert len(p_ledger.impressions) == 1
    imp = p_ledger.impressions[0]
    assert len(imp.source_states) == 1
    ss = imp.source_states[0]
    assert ss["stage"] == "初感訊號"
    assert ss["turn"] == 1
