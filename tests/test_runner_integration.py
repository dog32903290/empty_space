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
    # Redirect RUNS_DIR to tmp_path
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", tmp_path)

    responses = [
        "你回來了。",
        "嗯。",
        "⋯⋯",
        "不關我的事。",
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
    client = MockLLMClient(["你回來了。", "嗯。", "⋯⋯", "不關我的事。"])
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
    client = MockLLMClient(["a", "b", "c", "d"])
    run_session(config=minimal_config, llm_client=client)

    # Turn 1 is 母親 (protagonist)
    assert "## 關係層：對兒子" in client.calls[0]["system"]
    # Turn 2 is 兒子 (counterpart)
    assert "## 關係層：對母親" in client.calls[1]["system"]


def test_candidate_impressions_extracted_and_stored(minimal_config, tmp_path, monkeypatch):
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", tmp_path)
    responses = [
        """你回來了。

---IMPRESSIONS---
- text: "她的聲音放輕了一層"
  symbols: [克制, 靠近]
""",
        "嗯。",
        "⋯⋯",
        "不關我的事。",
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
    config = ExperimentConfig(
        exp_id="mother_x_son_hospital_v3_001",
        protagonist=PersonaRef(path="六個劇中人/母親", version="v3_tension"),
        counterpart=PersonaRef(path="六個劇中人/兒子", version="v3_tension"),
        setting=SettingRef(path="六個劇中人/環境_醫院.yaml"),
        initial_state=InitialState(verb="承受", stage="前置積累", mode="基線"),
        director_events={3: "護士推一張空床進病房"},
        max_turns=5,
    )
    client = MockLLMClient(["a", "b", "c", "d", "e"])
    run_session(config=config, llm_client=client)

    # Turn 1-2 system prompts should NOT contain the event (it triggers at Turn 3)
    assert "護士推一張空床進病房" not in client.calls[0]["system"]
    assert "護士推一張空床進病房" not in client.calls[1]["system"]
    # Turn 3 onwards SHOULD contain it
    assert "Turn 3：護士推一張空床進病房" in client.calls[2]["system"]
    assert "Turn 3：護士推一張空床進病房" in client.calls[3]["system"]
    assert "Turn 3：護士推一張空床進病房" in client.calls[4]["system"]


def test_director_events_accumulate_across_turns(tmp_path, monkeypatch):
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", tmp_path)
    config = ExperimentConfig(
        exp_id="mother_x_son_hospital_v3_001",
        protagonist=PersonaRef(path="六個劇中人/母親", version="v3_tension"),
        counterpart=PersonaRef(path="六個劇中人/兒子", version="v3_tension"),
        setting=SettingRef(path="六個劇中人/環境_醫院.yaml"),
        initial_state=InitialState(verb="承受", stage="前置積累", mode="基線"),
        director_events={2: "event A", 4: "event B"},
        max_turns=5,
    )
    client = MockLLMClient(["a", "b", "c", "d", "e"])
    run_session(config=config, llm_client=client)

    # Turn 5 system should contain both events in turn order
    sys5 = client.calls[4]["system"]
    assert "Turn 2：event A" in sys5
    assert "Turn 4：event B" in sys5
    assert sys5.index("Turn 2：event A") < sys5.index("Turn 4：event B")


def test_parse_error_recorded_but_session_continues(tmp_path, monkeypatch):
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", tmp_path)
    config = ExperimentConfig(
        exp_id="mother_x_son_hospital_v3_001",
        protagonist=PersonaRef(path="六個劇中人/母親", version="v3_tension"),
        counterpart=PersonaRef(path="六個劇中人/兒子", version="v3_tension"),
        setting=SettingRef(path="六個劇中人/環境_醫院.yaml"),
        initial_state=InitialState(verb="承受", stage="前置積累", mode="基線"),
        max_turns=3,
    )
    broken_yaml_response = """她低著頭。

---IMPRESSIONS---
- text: "unclosed
"""
    client = MockLLMClient([broken_yaml_response, "嗯。", "⋯⋯"])
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
    config = minimal_config.model_copy(update={"max_turns": 2})
    client = MockLLMClient(["a", "b"])
    result = run_session(config=config, llm_client=client)
    assert result.total_turns == 2
    assert not (result.out_dir / "turns" / "turn_003.yaml").exists()
    assert result.termination_reason == "max_turns"


def test_llm_exception_aborts_session_partial_turns_kept(
    tmp_path, monkeypatch, minimal_config
):
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", tmp_path)

    class ExplodingClient:
        def __init__(self):
            self.call_count = 0

        def generate(self, *, system, user, model="gemini-2.5-flash"):
            self.call_count += 1
            if self.call_count == 3:
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

    # First run
    client1 = MockLLMClient(["a", "b", "c", "d"])
    result1 = run_session(config=minimal_config, llm_client=client1)

    # Sleep a second to ensure timestamp differs
    import time as _time
    _time.sleep(1.1)

    client2 = MockLLMClient(["x", "y", "z", "w"])
    result2 = run_session(config=minimal_config, llm_client=client2)

    assert result1.out_dir != result2.out_dir
    assert result1.out_dir.is_dir()
    assert result2.out_dir.is_dir()
