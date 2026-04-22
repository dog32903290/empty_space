"""Cross-session integration tests for Level 2 ledger + retrieval flow.

Uses MockLLMClient (no real API). Each test spans one or two session runs.
"""
from pathlib import Path

import pytest
import yaml

from empty_space.ledger import append_session_candidates, read_ledger
from empty_space.llm import GeminiResponse
from empty_space.runner import run_session
from empty_space.schemas import (
    CandidateImpression,
    ExperimentConfig,
    InitialState,
    PersonaRef,
    SettingRef,
    Termination,
)


class MockLLMClient:
    """Pre-scheduled responses. Flash extract_symbols comes first (per role)."""
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = []

    def generate(self, *, system, user, model="gemini-2.5-flash"):
        self.calls.append({"system": system, "user": user, "model": model})
        if not self.responses:
            raise RuntimeError(f"out of responses on call {len(self.calls)}")
        content = self.responses.pop(0)
        return GeminiResponse(
            content=content, raw=None,
            tokens_in=len(system) // 4, tokens_out=len(content) // 4,
            model=model, latency_ms=10,
        )


@pytest.fixture(autouse=True)
def redirect_all_dirs(tmp_path, monkeypatch):
    """Redirect RUNS_DIR and LEDGERS_DIR for isolation."""
    runs_dir = tmp_path / "runs"
    ledgers_dir = tmp_path / "ledgers"
    runs_dir.mkdir()
    ledgers_dir.mkdir()
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", runs_dir)
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", ledgers_dir)
    # Also redirect synonym path so tests don't accidentally pick up real config
    monkeypatch.setattr(
        "empty_space.retrieval.DEFAULT_SYNONYMS_PATH",
        tmp_path / "nonexistent_synonyms.yaml",
    )
    return {"runs_dir": runs_dir, "ledgers_dir": ledgers_dir}


def _base_config(max_turns: int = 4) -> ExperimentConfig:
    return ExperimentConfig(
        exp_id="mother_x_son_hospital_v3_001",
        protagonist=PersonaRef(path="六個劇中人/母親", version="v3_tension"),
        counterpart=PersonaRef(path="六個劇中人/兒子", version="v3_tension"),
        setting=SettingRef(path="六個劇中人/環境_醫院.yaml"),
        scene_premise="醫院裡，父親在 ICU。",
        protagonist_prelude=None,
        counterpart_prelude=None,
        initial_state=InitialState(verb="承受", stage="前置積累", mode="基線"),
        director_events={},
        max_turns=max_turns,
        termination=Termination(),
    )


def test_first_session_empty_ledgers_written_after(redirect_all_dirs):
    """第一場帳本空；session 結束後 raw + refined 帳本都被建立。"""
    config = _base_config(max_turns=2)
    responses = [
        "- 醫院\n- 父親\n",
        "- 醫院\n- 父親\n",
        "話一\n\n---IMPRESSIONS---\n- text: \"母親印象一\"\n  symbols: [A, B]\n",
        "話二\n\n---IMPRESSIONS---\n- text: \"兒子印象一\"\n  symbols: [C, D]\n",
        # Composer with actual drafts to verify refined creation
        "母親:\n  - text: \"母親精煉\"\n    symbols: [A]\n    source_raw_ids: [imp_001]\n\n兒子:\n  - text: \"兒子精煉\"\n    symbols: [C]\n    source_raw_ids: [imp_001]\n",
    ]
    client = MockLLMClient(responses)
    result = run_session(config=config, llm_client=client)

    # retrieval.yaml exists, impressions empty (refined ledger was empty at session start)
    ret = yaml.safe_load((result.out_dir / "retrieval.yaml").read_text(encoding="utf-8"))
    assert ret["protagonist"]["impressions"] == []
    assert ret["counterpart"]["impressions"] == []

    ledgers_dir = redirect_all_dirs["ledgers_dir"]
    # Raw ledgers exist
    assert (ledgers_dir / "母親_x_兒子.from_母親.yaml").is_file()
    assert (ledgers_dir / "母親_x_兒子.from_兒子.yaml").is_file()
    # Refined ledgers also exist (Composer produced drafts)
    assert (ledgers_dir / "母親_x_兒子.refined.from_母親.yaml").is_file()
    assert (ledgers_dir / "母親_x_兒子.refined.from_兒子.yaml").is_file()

    l_p = read_ledger(relationship="母親_x_兒子", persona_name="母親")
    l_c = read_ledger(relationship="母親_x_兒子", persona_name="兒子")
    assert l_p.ledger_version == 1
    assert l_c.ledger_version == 1
    assert len(l_p.candidates) == 1
    assert len(l_c.candidates) == 1
    assert l_p.candidates[0].text == "母親印象一"
    assert l_c.candidates[0].text == "兒子印象一"


def test_second_session_retrieval_hits_first_session_impressions(redirect_all_dirs):
    """第二場 session 的 retrieval 能撈到第一場 Composer 精煉的 refined。"""
    config = _base_config(max_turns=2)

    # Session 1: composer produces refined with 醫院 / 父親 symbols
    responses_1 = [
        "- 醫院\n",
        "- 醫院\n",
        "話一\n\n---IMPRESSIONS---\n- text: \"母親印象\"\n  symbols: [醫院, 父親]\n",
        "話二\n\n---IMPRESSIONS---\n- text: \"兒子印象\"\n  symbols: [醫院, 父親]\n",
        # Composer produces refined with 醫院 symbol (matches session 2 query)
        "母親:\n  - text: \"醫院走廊長\"\n    symbols: [醫院]\n    source_raw_ids: [imp_001]\n\n兒子:\n  - text: \"父親的門\"\n    symbols: [父親]\n    source_raw_ids: [imp_001]\n",
    ]
    run_session(config=config, llm_client=MockLLMClient(responses_1))

    # Session 2: retrieval should hit session 1's refined
    responses_2 = [
        "- 醫院\n- 父親\n",
        "- 醫院\n- 父親\n",
        "話三",
        "話四",
        "母親: []\n兒子: []\n",   # composer noop
    ]
    client = MockLLMClient(responses_2)
    result_2 = run_session(config=config, llm_client=client)

    ret_2 = yaml.safe_load((result_2.out_dir / "retrieval.yaml").read_text(encoding="utf-8"))
    # Should hit session 1's refined
    p_texts = [imp["text"] for imp in ret_2["protagonist"]["impressions"]]
    c_texts = [imp["text"] for imp in ret_2["counterpart"]["impressions"]]
    # Protagonist should see both (A strategy: shared memory across speaker ledgers)
    assert "醫院走廊長" in p_texts or "父親的門" in p_texts
    # All impression ids should start with ref_ (refined, not raw)
    for imp in ret_2["protagonist"]["impressions"]:
        assert imp["id"].startswith("ref_")


def test_llm_exception_aborts_session_no_ledger_written(redirect_all_dirs):
    """Session 中斷時不寫帳本。"""
    config = _base_config(max_turns=4)

    class ExplodingClient:
        def __init__(self):
            self.count = 0
        def generate(self, *, system, user, model="gemini-2.5-flash"):
            self.count += 1
            # Let the 2 extract calls + 2 turns succeed, blow up on turn 3 (call 5)
            if self.count == 5:
                raise RuntimeError("boom")
            if self.count <= 2:
                return GeminiResponse(
                    content="- X\n", raw=None,
                    tokens_in=10, tokens_out=5, model=model, latency_ms=10,
                )
            return GeminiResponse(
                content="話",
                raw=None, tokens_in=10, tokens_out=5, model=model, latency_ms=10,
            )

    with pytest.raises(RuntimeError, match="boom"):
        run_session(config=config, llm_client=ExplodingClient())

    ledgers_dir = redirect_all_dirs["ledgers_dir"]
    # No ledger files should have been created
    assert list(ledgers_dir.iterdir()) == []


def test_pre_seeded_ledger_hits_system_prompt(redirect_all_dirs):
    """預先 seed refined ledger (not raw, since Level 3 retrieval reads refined)。"""
    from empty_space.ledger import append_refined_impressions
    from empty_space.schemas import RefinedImpressionDraft

    append_refined_impressions(
        relationship="母親_x_兒子",
        speaker_role="counterpart",
        persona_name="兒子",
        drafts=[
            RefinedImpressionDraft(
                text="她的手不動，像假的",
                symbols=["手", "不動", "假"],
                source_raw_ids=["imp_045"],
            ),
        ],
        source_run="prev_exp/2026-04-20T09-00-00",
    )

    config = _base_config(max_turns=2)
    responses = [
        "- 手\n- 不動\n",
        "- 手\n- 不動\n",
        "話一",
        "話二",
        "母親: []\n兒子: []\n",   # composer
    ]
    client = MockLLMClient(responses)
    result = run_session(config=config, llm_client=client)

    turn_1 = yaml.safe_load((result.out_dir / "turns" / "turn_001.yaml").read_text(encoding="utf-8"))
    assert "她的手不動，像假的" in turn_1["prompt_assembled"]["system"]
    assert any(
        r["text"] == "她的手不動，像假的"
        for r in turn_1["retrieved_impressions"]
    )
    # Refined-sourced retrieval: id should be ref_XXX
    assert turn_1["retrieved_impressions"][0]["id"].startswith("ref_")


def test_synonym_map_enables_variant_matching(redirect_all_dirs, tmp_path, monkeypatch):
    """同義詞字典 → query 的「愧疚」命中帳本的「愧疚感」。"""
    syn_path = tmp_path / "synonyms.yaml"
    syn_path.write_text(
        "groups:\n  - [愧疚, 愧疚感]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("empty_space.retrieval.DEFAULT_SYNONYMS_PATH", syn_path)

    from empty_space.ledger import append_refined_impressions
    from empty_space.schemas import RefinedImpressionDraft
    append_refined_impressions(
        relationship="母親_x_兒子",
        speaker_role="counterpart",
        persona_name="兒子",
        drafts=[
            RefinedImpressionDraft(
                text="她看著地板，沒看我",
                symbols=["愧疚感"],
                source_raw_ids=[],
            ),
        ],
        source_run="prev/2026-04-20T09-00-00",
    )

    config = _base_config(max_turns=2)
    responses = [
        "- 愧疚\n",
        "- 愧疚\n",
        "話一",
        "話二",
        "母親: []\n兒子: []\n",   # composer
    ]
    client = MockLLMClient(responses)
    result = run_session(config=config, llm_client=client)

    ret = yaml.safe_load((result.out_dir / "retrieval.yaml").read_text(encoding="utf-8"))
    p_texts = [imp["text"] for imp in ret["protagonist"]["impressions"]]
    assert "她看著地板，沒看我" in p_texts
