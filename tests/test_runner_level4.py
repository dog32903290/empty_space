"""Level 4 runner integration tests — per-speaker Judge state machine.

All tests use MockLLMClient (no real API).

Call order per turn (with Judge enabled):
  1. Flash extract_symbols(protagonist prelude)  [session start only]
  2. Flash extract_symbols(counterpart prelude)  [session start only]
  3. Role LLM (dialogue turn 1 - protagonist)
  4. Flash Judge (protagonist, turn 1)
  5. Flash Judge (counterpart, turn 1)
  ... repeat per turn
  finally: Composer (Gemini Pro, 1 call)
"""
from pathlib import Path

import pytest
import yaml

from empty_space.llm import GeminiResponse
from empty_space.runner import run_session
from empty_space.schemas import (
    ExperimentConfig,
    InitialState,
    PersonaRef,
    SettingRef,
    Termination,
)


class MockLLMClient:
    def __init__(self, responses):
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
    runs_dir = tmp_path / "runs"
    ledgers_dir = tmp_path / "ledgers"
    runs_dir.mkdir()
    ledgers_dir.mkdir()
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", runs_dir)
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", ledgers_dir)
    monkeypatch.setattr(
        "empty_space.retrieval.DEFAULT_SYNONYMS_PATH",
        tmp_path / "nonexistent_synonyms.yaml",
    )
    return {"runs_dir": runs_dir, "ledgers_dir": ledgers_dir}


def _base_config(max_turns: int = 2) -> ExperimentConfig:
    return ExperimentConfig(
        exp_id="l4_test_001",
        protagonist=PersonaRef(path="六個劇中人/母親", version="v3_tension"),
        counterpart=PersonaRef(path="六個劇中人/兒子", version="v3_tension"),
        setting=SettingRef(path="六個劇中人/環境_醫院.yaml"),
        scene_premise="醫院裡，父親在 ICU。",
        protagonist_prelude=None,
        counterpart_prelude=None,
        initial_state=InitialState(verb="承受", stage="前置積累", mode="在"),
        director_events={},
        max_turns=max_turns,
        termination=Termination(),
    )


def test_initial_state_legacy_mode_baseline_migrated():
    """ExperimentConfig with mode='基線' should be migrated to '在' at validation."""
    # manually construct with legacy value — validator should normalise
    legacy = InitialState(verb="承受", stage="前置積累", mode="基線")
    assert legacy.mode == "在"
