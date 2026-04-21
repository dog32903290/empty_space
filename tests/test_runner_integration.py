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
