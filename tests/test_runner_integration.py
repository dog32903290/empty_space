"""Integration tests for runner.run_session with a MockLLMClient.

No real Gemini API calls. These tests verify the whole pipeline
(prompt assembly → LLM → parse → write) without touching the network.
"""
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from empty_space.llm import GeminiResponse
from empty_space.runner import run_session
from empty_space.schemas import (
    ExperimentConfig,
    PersonaRef,
    SettingRef,
    InitialState,
    Termination,
)


class MockLLMClient:
    """Pre-scheduled responses keyed by call order. Records all calls."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(self, *, system: str, user: str, model: str = "gemini-2.5-flash") -> GeminiResponse:
        self.calls.append({"system": system, "user": user, "model": model})
        if not self.responses:
            raise RuntimeError(f"MockLLMClient ran out of responses on call {len(self.calls)}")
        content = self.responses.pop(0)
        return GeminiResponse(
            content=content,
            raw=None,
            tokens_in=len(system) // 4,
            tokens_out=len(content) // 4,
            model=model,
            latency_ms=10,
        )


@pytest.fixture
def minimal_config(tmp_path) -> ExperimentConfig:
    return ExperimentConfig(
        exp_id="mother_x_son_hospital_v3_001",  # uses real persona/setting
        protagonist=PersonaRef(path="六個劇中人/母親", version="v3_tension"),
        counterpart=PersonaRef(path="六個劇中人/兒子", version="v3_tension"),
        setting=SettingRef(path="六個劇中人/環境_醫院.yaml"),
        scene_premise="父親在 ICU。",
        initial_state=InitialState(verb="承受", stage="前置積累", mode="基線"),
        director_events={},
        max_turns=4,
        termination=Termination(),
    )


def test_happy_path_runs_all_turns(minimal_config, tmp_path, monkeypatch):
    # Redirect RUNS_DIR and LEDGERS_DIR to tmp_path
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", tmp_path)
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")

    responses = [
        "- 醫院\n- 父親\n",   # NEW: protagonist extract_symbols
        "- 醫院\n- 父親\n",   # NEW: counterpart extract_symbols
        "你回來了。",
        "嗯。",
        "⋯⋯",
        "不關我的事。",
        "母親: []\n兒子: []\n",  # composer noop
    ]
    client = MockLLMClient(responses)

    result = run_session(config=minimal_config, llm_client=client)

    assert result.total_turns == 4
    assert result.termination_reason == "max_turns"
    assert result.out_dir.is_dir()
    assert (result.out_dir / "turns" / "turn_001.yaml").is_file()
    assert (result.out_dir / "turns" / "turn_004.yaml").is_file()
    assert (result.out_dir / "meta.yaml").is_file()

    meta = yaml.safe_load((result.out_dir / "meta.yaml").read_text(encoding="utf-8"))
    assert meta["total_turns"] == 4
    assert meta["termination_reason"] == "max_turns"


def test_speaker_alternation_mother_starts(minimal_config, tmp_path, monkeypatch):
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", tmp_path)
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")
    client = MockLLMClient([
        "- 醫院\n- 父親\n",   # protagonist extract_symbols
        "- 醫院\n- 父親\n",   # counterpart extract_symbols
        "你回來了。", "嗯。", "⋯⋯", "不關我的事。",
        "母親: []\n兒子: []\n",  # composer noop
    ])
    result = run_session(config=minimal_config, llm_client=client)

    t1 = yaml.safe_load((result.out_dir / "turns" / "turn_001.yaml").read_text(encoding="utf-8"))
    t2 = yaml.safe_load((result.out_dir / "turns" / "turn_002.yaml").read_text(encoding="utf-8"))
    t3 = yaml.safe_load((result.out_dir / "turns" / "turn_003.yaml").read_text(encoding="utf-8"))

    assert t1["speaker"] == "protagonist"
    assert t1["persona_name"] == "母親"
    assert t2["speaker"] == "counterpart"
    assert t2["persona_name"] == "兒子"
    assert t3["speaker"] == "protagonist"


def test_system_prompt_contains_correct_persona_per_turn(minimal_config, tmp_path, monkeypatch):
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", tmp_path)
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")
    client = MockLLMClient([
        "- 醫院\n- 父親\n",   # protagonist extract_symbols (call 0)
        "- 醫院\n- 父親\n",   # counterpart extract_symbols (call 1)
        "a", "b", "c", "d",  # turn calls (calls 2-5)
        "母親: []\n兒子: []\n",  # composer noop (call 6)
    ])
    run_session(config=minimal_config, llm_client=client)

    # Calls 0-1 are extract_symbols; Turn 1 is call 2 (母親/protagonist)
    assert "## 關係層：對兒子" in client.calls[2]["system"]
    # Turn 2 is call 3 (兒子/counterpart)
    assert "## 關係層：對母親" in client.calls[3]["system"]


def test_candidate_impressions_extracted_and_stored(minimal_config, tmp_path, monkeypatch):
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", tmp_path)
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")
    responses = [
        "- 醫院\n- 父親\n",   # protagonist extract_symbols
        "- 醫院\n- 父親\n",   # counterpart extract_symbols
        """你回來了。

---IMPRESSIONS---
- text: "她的聲音放輕了一層"
  symbols: [克制, 靠近]
""",
        "嗯。",
        "⋯⋯",
        "不關我的事。",
        "母親: []\n兒子: []\n",  # composer noop
    ]
    client = MockLLMClient(responses)
    result = run_session(config=minimal_config, llm_client=client)

    t1 = yaml.safe_load((result.out_dir / "turns" / "turn_001.yaml").read_text(encoding="utf-8"))
    assert t1["response"]["content"] == "你回來了。"
    assert len(t1["candidate_impressions"]) == 1
    assert t1["candidate_impressions"][0]["text"] == "她的聲音放輕了一層"


def test_director_event_injected_into_system_from_trigger_turn(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", tmp_path)
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")
    config = ExperimentConfig(
        exp_id="mother_x_son_hospital_v3_001",
        protagonist=PersonaRef(path="六個劇中人/母親", version="v3_tension"),
        counterpart=PersonaRef(path="六個劇中人/兒子", version="v3_tension"),
        setting=SettingRef(path="六個劇中人/環境_醫院.yaml"),
        initial_state=InitialState(verb="承受", stage="前置積累", mode="基線"),
        director_events={3: "護士推一張空床進病房"},
        max_turns=5,
        # No scene_premise/preludes → _compose_query returns "" → extract_symbols
        # skips the LLM call → no prepended extract responses needed.
    )
    client = MockLLMClient(["a", "b", "c", "d", "e", "母親: []\n兒子: []\n"])
    run_session(config=config, llm_client=client)

    # No extract_symbols calls (empty query). Turn 1 is call 0, Turn 2 is call 1, etc.
    # Turn 1-2 system prompts should NOT contain the event (it triggers at Turn 3)
    assert "護士推一張空床進病房" not in client.calls[0]["system"]
    assert "護士推一張空床進病房" not in client.calls[1]["system"]
    # Turn 3 onwards SHOULD contain it
    assert "Turn 3：護士推一張空床進病房" in client.calls[2]["system"]
    assert "Turn 3：護士推一張空床進病房" in client.calls[3]["system"]
    assert "Turn 3：護士推一張空床進病房" in client.calls[4]["system"]


def test_director_events_accumulate_across_turns(tmp_path, monkeypatch):
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", tmp_path)
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")
    config = ExperimentConfig(
        exp_id="mother_x_son_hospital_v3_001",
        protagonist=PersonaRef(path="六個劇中人/母親", version="v3_tension"),
        counterpart=PersonaRef(path="六個劇中人/兒子", version="v3_tension"),
        setting=SettingRef(path="六個劇中人/環境_醫院.yaml"),
        initial_state=InitialState(verb="承受", stage="前置積累", mode="基線"),
        director_events={2: "event A", 4: "event B"},
        max_turns=5,
        # No scene_premise/preludes → no extract_symbols LLM calls.
    )
    client = MockLLMClient(["a", "b", "c", "d", "e", "母親: []\n兒子: []\n"])
    run_session(config=config, llm_client=client)

    # No extract_symbols calls. Turn 5 is call index 4.
    sys5 = client.calls[4]["system"]
    assert "Turn 2：event A" in sys5
    assert "Turn 4：event B" in sys5
    assert sys5.index("Turn 2：event A") < sys5.index("Turn 4：event B")


def test_parse_error_recorded_but_session_continues(tmp_path, monkeypatch):
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", tmp_path)
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")
    config = ExperimentConfig(
        exp_id="mother_x_son_hospital_v3_001",
        protagonist=PersonaRef(path="六個劇中人/母親", version="v3_tension"),
        counterpart=PersonaRef(path="六個劇中人/兒子", version="v3_tension"),
        setting=SettingRef(path="六個劇中人/環境_醫院.yaml"),
        initial_state=InitialState(verb="承受", stage="前置積累", mode="基線"),
        max_turns=3,
        # No scene_premise/preludes → no extract_symbols LLM calls.
    )
    broken_yaml_response = """她低著頭。

---IMPRESSIONS---
- text: "unclosed
"""
    client = MockLLMClient([broken_yaml_response, "嗯。", "⋯⋯", "母親: []\n兒子: []\n"])
    result = run_session(config=config, llm_client=client)

    assert result.total_turns == 3
    t1 = yaml.safe_load((result.out_dir / "turns" / "turn_001.yaml").read_text(encoding="utf-8"))
    assert t1["parse_error"] is not None
    assert "YAML" in t1["parse_error"]
    assert t1["response"]["content"] == "她低著頭。"   # main recovered

    meta = yaml.safe_load((result.out_dir / "meta.yaml").read_text(encoding="utf-8"))
    assert meta["turns_with_parse_error"] == 1


def test_max_turns_terminates_session(tmp_path, monkeypatch, minimal_config):
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", tmp_path)
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")
    config = minimal_config.model_copy(update={"max_turns": 2})
    client = MockLLMClient([
        "- 醫院\n- 父親\n",   # protagonist extract_symbols
        "- 醫院\n- 父親\n",   # counterpart extract_symbols
        "a", "b",
        "母親: []\n兒子: []\n",  # composer noop
    ])
    result = run_session(config=config, llm_client=client)
    assert result.total_turns == 2
    assert not (result.out_dir / "turns" / "turn_003.yaml").exists()
    assert result.termination_reason == "max_turns"


def test_llm_exception_aborts_session_partial_turns_kept(
    tmp_path, monkeypatch, minimal_config
):
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", tmp_path)
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")

    class ExplodingClient:
        def __init__(self):
            self.call_count = 0

        def generate(self, *, system, user, model="gemini-2.5-flash"):
            self.call_count += 1
            # Calls 1-2 are extract_symbols (protagonist + counterpart).
            # Calls 3-4 are turn 1 and turn 2. Call 5 = turn 3 → BOOM.
            if self.call_count == 5:
                raise RuntimeError("network boom")
            return GeminiResponse(
                content="x",
                raw=None,
                tokens_in=1,
                tokens_out=1,
                model=model,
                latency_ms=1,
            )

    client = ExplodingClient()
    with pytest.raises(RuntimeError, match="network boom"):
        run_session(config=minimal_config, llm_client=client)

    # Find the run_dir that was created (only one, under exp_id)
    exp_dirs = list((tmp_path / minimal_config.exp_id).iterdir())
    assert len(exp_dirs) == 1
    run_dir = exp_dirs[0]
    assert (run_dir / "turns" / "turn_001.yaml").exists()
    assert (run_dir / "turns" / "turn_002.yaml").exists()
    assert not (run_dir / "turns" / "turn_003.yaml").exists()
    assert not (run_dir / "meta.yaml").exists()  # meta not written on abort


def test_two_runs_of_same_exp_create_distinct_timestamp_dirs(
    tmp_path, monkeypatch, minimal_config
):
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", tmp_path)
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")

    # First run
    client1 = MockLLMClient([
        "- 醫院\n- 父親\n",   # protagonist extract_symbols
        "- 醫院\n- 父親\n",   # counterpart extract_symbols
        "a", "b", "c", "d",
        "母親: []\n兒子: []\n",  # composer noop
    ])
    result1 = run_session(config=minimal_config, llm_client=client1)

    # Sleep a second to ensure timestamp differs
    import time as _time
    _time.sleep(1.1)

    client2 = MockLLMClient([
        "- 醫院\n- 父親\n",   # protagonist extract_symbols
        "- 醫院\n- 父親\n",   # counterpart extract_symbols
        "x", "y", "z", "w",
        "母親: []\n兒子: []\n",  # composer noop
    ])
    result2 = run_session(config=minimal_config, llm_client=client2)

    assert result1.out_dir != result2.out_dir
    assert result1.out_dir.is_dir()
    assert result2.out_dir.is_dir()


# --- Level 3: Composer integration tests ---

def test_composer_runs_at_session_end(minimal_config, tmp_path, monkeypatch):
    """Normal session: composer runs, produces refined, updates meta.yaml."""
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", tmp_path)
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")

    responses = [
        "- 醫院\n", "- 醫院\n",
        "你回來了。", "嗯。", "⋯⋯", "不關我的事。",
        # Composer with actual drafts
        "母親:\n  - text: \"沉默時收縮\"\n    symbols: [沉默]\n    source_raw_ids: [imp_001]\n\n兒子:\n  - text: \"背靠冰冷\"\n    symbols: [背]\n    source_raw_ids: [imp_001]\n",
    ]
    client = MockLLMClient(responses)
    result = run_session(config=minimal_config, llm_client=client)

    # Both refined ledgers created
    assert (tmp_path / "ledgers" / "母親_x_兒子.refined.from_母親.yaml").is_file()
    assert (tmp_path / "ledgers" / "母親_x_兒子.refined.from_兒子.yaml").is_file()

    # meta.yaml has composer fields
    meta = yaml.safe_load((result.out_dir / "meta.yaml").read_text(encoding="utf-8"))
    assert meta["composer_tokens_in"] > 0
    assert meta["protagonist_refined_added"] == 1
    assert meta["counterpart_refined_added"] == 1
    assert meta["composer_parse_error"] is None
    assert "gemini-2.5-pro" in meta["models_used"]


def test_composer_failure_doesnt_break_session(minimal_config, tmp_path, monkeypatch):
    """If Pro returns garbage, refined ledgers not touched; raw ledgers intact; session completes."""
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", tmp_path)
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")

    responses = [
        "- 醫院\n", "- 醫院\n",
        # Include impression in turn 1 so raw ledger gets written
        "a\n\n---IMPRESSIONS---\n- text: \"raw impression\"\n  symbols: [醫院]\n",
        "b", "c", "d",
        "garbage [[[ not yaml",   # composer bad YAML
    ]
    client = MockLLMClient(responses)
    result = run_session(config=minimal_config, llm_client=client)

    # Session completes
    assert result.total_turns == 4
    assert (result.out_dir / "meta.yaml").is_file()

    # meta has composer parse_error
    meta = yaml.safe_load((result.out_dir / "meta.yaml").read_text(encoding="utf-8"))
    assert meta["composer_parse_error"] is not None
    assert meta["protagonist_refined_added"] == 0

    # Raw ledger exists (turn 1 produced a candidate)
    assert (tmp_path / "ledgers" / "母親_x_兒子.from_母親.yaml").is_file()
    # Refined ledgers NOT created (parse failed → 0 drafts appended)
    assert not (tmp_path / "ledgers" / "母親_x_兒子.refined.from_母親.yaml").is_file()


def test_second_session_retrieval_reads_refined(minimal_config, tmp_path, monkeypatch):
    """Two sessions: session 2's retrieval撈到 session 1's refined (ref_XXX id)."""
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", tmp_path)
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")

    # Session 1
    responses_1 = [
        "- 醫院\n- 父親\n", "- 醫院\n- 父親\n",
        "a", "b", "c", "d",
        "母親:\n  - text: \"醫院走廊長\"\n    symbols: [醫院, 走廊]\n    source_raw_ids: [imp_001]\n\n兒子:\n  - text: \"父親的門關著\"\n    symbols: [父親, 門]\n    source_raw_ids: [imp_001]\n",
    ]
    run_session(config=minimal_config, llm_client=MockLLMClient(responses_1))

    # Session 2
    responses_2 = [
        "- 醫院\n", "- 父親\n",
        "e", "f", "g", "h",
        "母親: []\n兒子: []\n",
    ]
    client = MockLLMClient(responses_2)
    result = run_session(config=minimal_config, llm_client=client)

    retrieval = yaml.safe_load((result.out_dir / "retrieval.yaml").read_text(encoding="utf-8"))
    p_impressions = retrieval["protagonist"]["impressions"]
    assert len(p_impressions) >= 1
    # Should be refined (ref_XXX) and from_turn=null
    assert all(imp["id"].startswith("ref_") for imp in p_impressions)
    assert p_impressions[0]["from_turn"] is None
