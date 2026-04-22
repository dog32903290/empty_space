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


def test_judge_runs_twice_per_turn_for_both_speakers():
    """4-turn session → 8 Judge calls (2 per turn)."""
    # Call sequence: 2 extract + 4 dialogue + 4*2 judge + 1 composer = 15
    responses = [
        "- 醫院\n", "- 醫院\n",
        "話1", "STAGE: 前置積累\nMODE: 在\nWHY: x\nVERDICT: N/A\nHITS: -\n",
        "STAGE: 前置積累\nMODE: 在\nWHY: x\nVERDICT: N/A\nHITS: -\n",
        "話2", "STAGE: 前置積累\nMODE: 在\nWHY: x\nVERDICT: N/A\nHITS: -\n",
        "STAGE: 前置積累\nMODE: 在\nWHY: x\nVERDICT: N/A\nHITS: -\n",
        "話3", "STAGE: 前置積累\nMODE: 在\nWHY: x\nVERDICT: N/A\nHITS: -\n",
        "STAGE: 前置積累\nMODE: 在\nWHY: x\nVERDICT: N/A\nHITS: -\n",
        "話4", "STAGE: 前置積累\nMODE: 在\nWHY: x\nVERDICT: N/A\nHITS: -\n",
        "STAGE: 前置積累\nMODE: 在\nWHY: x\nVERDICT: N/A\nHITS: -\n",
        "母親: []\n兒子: []\n",
    ]
    config = _base_config(max_turns=4)
    client = MockLLMClient(responses)
    run_session(config=config, llm_client=client)

    # Count Judge calls by matching system prompt containing 隱性量測者
    judge_calls = [c for c in client.calls if "隱性量測者" in c["system"]]
    assert len(judge_calls) == 8


def test_judge_state_evolves_across_turns():
    """State updates between turns — turn 3 sees turn 1's Judge output."""
    # Turn 1 judge P: advance to 初感訊號/收
    # Turn 2 judge C: stay at 前置積累/在
    # In turn 3 P's system prompt 此刻 should now read 初感訊號/收
    responses = [
        "- 醫院\n", "- 醫院\n",
        # turn 1 — protagonist speaks
        "話1_P",
        "STAGE: 初感訊號\nMODE: 收\nWHY: 縮肩\nVERDICT: N/A\nHITS: 肩下沉\n",
        "STAGE: 前置積累\nMODE: 在\nWHY: 沒反應\nVERDICT: N/A\nHITS: -\n",
        # turn 2 — counterpart speaks; P's 此刻 still reads its own state in ITS next turn
        "話2_C",
        "STAGE: 初感訊號\nMODE: 收\nWHY: 穩\nVERDICT: N/A\nHITS: -\n",
        "STAGE: 初感訊號\nMODE: 收\nWHY: 動\nVERDICT: N/A\nHITS: -\n",
        # turn 3 — protagonist again; its 此刻 should be 初感訊號/收 (not initial 前置積累/在)
        "話3_P",
        "STAGE: 初感訊號\nMODE: 收\nWHY: x\nVERDICT: N/A\nHITS: -\n",
        "STAGE: 初感訊號\nMODE: 收\nWHY: x\nVERDICT: N/A\nHITS: -\n",
        # turn 4 counterpart
        "話4_C",
        "STAGE: 初感訊號\nMODE: 收\nWHY: x\nVERDICT: N/A\nHITS: -\n",
        "STAGE: 初感訊號\nMODE: 收\nWHY: x\nVERDICT: N/A\nHITS: -\n",
        "母親: []\n兒子: []\n",
    ]
    config = _base_config(max_turns=4)
    client = MockLLMClient(responses)
    result = run_session(config=config, llm_client=client)

    # Turn 3 is protagonist's second turn — its system prompt should show 初感訊號/收
    turn_3 = yaml.safe_load(
        (result.out_dir / "turns" / "turn_003.yaml").read_text(encoding="utf-8")
    )
    assert "階段：初感訊號" in turn_3["prompt_assembled"]["system"]
    assert "模式：收" in turn_3["prompt_assembled"]["system"]


def test_judge_skipped_when_persona_lacks_v3(monkeypatch):
    """Persona without v3 files — Judge skipped; state unchanged; no Judge calls."""
    import empty_space.runner as runner_mod

    original_load = runner_mod.load_persona

    def stub_load(path, version):
        p = original_load(path, version)
        # Wipe v3 fields as if persona has no v3 files
        p = p.model_copy(update={
            "judge_principles_text": "",
            "stage_mode_contexts_parsed": {},
        })
        return p

    monkeypatch.setattr(runner_mod, "load_persona", stub_load)

    responses = [
        "- 醫院\n", "- 醫院\n",
        "話1", "話2",
        "母親: []\n兒子: []\n",
    ]
    config = _base_config(max_turns=2)
    client = MockLLMClient(responses)
    run_session(config=config, llm_client=client)

    judge_calls = [c for c in client.calls if "隱性量測者" in c["system"]]
    assert len(judge_calls) == 0


def test_judge_llm_error_does_not_crash_session(monkeypatch):
    """When Judge LLM raises, run_judge catches it; session completes."""
    class PartiallyExplodingClient(MockLLMClient):
        def generate(self, *, system, user, model="gemini-2.5-flash"):
            # Explode on Judge calls only
            if "隱性量測者" in system:
                raise RuntimeError("flash down")
            return super().generate(system=system, user=user, model=model)

    responses = [
        "- 醫院\n", "- 醫院\n",
        "話1", "話2",
        "母親: []\n兒子: []\n",
    ]
    config = _base_config(max_turns=2)
    client = PartiallyExplodingClient(responses)
    result = run_session(config=config, llm_client=client)
    assert result.total_turns == 2
    # meta.yaml should record error
    meta = yaml.safe_load((result.out_dir / "meta.yaml").read_text(encoding="utf-8"))
    assert meta["judge_health"]["protagonist"]["llm_error"] >= 1


def test_meta_yaml_includes_judge_trajectories_and_health():
    """After 2-turn session, meta should have both trajectories and health stats."""
    responses = [
        "- 醫院\n", "- 醫院\n",
        "話1", "STAGE: 初感訊號\nMODE: 收\nWHY: x\nVERDICT: N/A\nHITS: -\n",
        "STAGE: 前置積累\nMODE: 在\nWHY: x\nVERDICT: N/A\nHITS: -\n",
        "話2", "STAGE: 初感訊號\nMODE: 收\nWHY: x\nVERDICT: N/A\nHITS: -\n",
        "STAGE: 前置積累\nMODE: 在\nWHY: x\nVERDICT: N/A\nHITS: -\n",
        "母親: []\n兒子: []\n",
    ]
    config = _base_config(max_turns=2)
    client = MockLLMClient(responses)
    result = run_session(config=config, llm_client=client)

    meta = yaml.safe_load((result.out_dir / "meta.yaml").read_text(encoding="utf-8"))
    assert meta["judge_trajectories"]["protagonist"]["stages"] == ["初感訊號", "初感訊號"]
    assert meta["judge_trajectories"]["counterpart"]["stages"] == ["前置積累", "前置積累"]
    assert meta["judge_trajectories"]["protagonist"]["moves"] == ["advance", "stay"]
    assert meta["judge_health"]["protagonist"]["total_calls"] == 2
    assert meta["judge_health"]["protagonist"]["ok"] == 2
    assert meta["termination"]["reason"] == "max_turns"
    assert meta["termination"]["turn"] == 2
    assert meta["interactive_mode"] is False


def test_dual_basin_lock_terminates_session_early():
    """Both speakers verdict=basin_lock for 2 consecutive turns → session stops."""
    # max_turns=10 but both speakers basin_lock from turn 1 → should stop at turn 2 (need 2 consecutive)
    responses = [
        "- 醫院\n", "- 醫院\n",
        "話1",
        "STAGE: 穩定期\nMODE: 在\nWHY: x\nVERDICT: basin_lock\nHITS: -\n",
        "STAGE: 穩定期\nMODE: 在\nWHY: x\nVERDICT: basin_lock\nHITS: -\n",
        "話2",
        "STAGE: 穩定期\nMODE: 在\nWHY: x\nVERDICT: basin_lock\nHITS: -\n",
        "STAGE: 穩定期\nMODE: 在\nWHY: x\nVERDICT: basin_lock\nHITS: -\n",
        "母親: []\n兒子: []\n",
    ]
    config = _base_config(max_turns=10)
    client = MockLLMClient(responses)
    result = run_session(config=config, llm_client=client)

    assert result.total_turns == 2
    assert result.termination_reason == "dual_basin_lock"


def test_single_basin_lock_does_not_terminate():
    """Only protagonist basin_lock — session continues until max_turns."""
    # max_turns=2 to keep small; counterpart always N/A
    responses = [
        "- 醫院\n", "- 醫院\n",
        "話1",
        "STAGE: 穩定期\nMODE: 在\nWHY: x\nVERDICT: basin_lock\nHITS: -\n",
        "STAGE: 前置積累\nMODE: 在\nWHY: x\nVERDICT: N/A\nHITS: -\n",
        "話2",
        "STAGE: 穩定期\nMODE: 在\nWHY: x\nVERDICT: basin_lock\nHITS: -\n",
        "STAGE: 前置積累\nMODE: 在\nWHY: x\nVERDICT: N/A\nHITS: -\n",
        "母親: []\n兒子: []\n",
    ]
    config = _base_config(max_turns=2)
    client = MockLLMClient(responses)
    result = run_session(config=config, llm_client=client)
    assert result.total_turns == 2
    assert result.termination_reason == "max_turns"


def test_interactive_peak_injects_director_event(monkeypatch):
    """In interactive mode, fire_release triggers stdin prompt; event injected to next turn."""
    import empty_space.runner as runner_mod
    monkeypatch.setattr(runner_mod, "_prompt_for_director_event",
                        lambda **kw: "護士開門")

    responses = [
        "- 醫院\n", "- 醫院\n",
        "話1_P",
        "STAGE: 半意識浮現\nMODE: 放\nWHY: 爆發\nVERDICT: fire_release\nHITS: -\n",
        "STAGE: 前置積累\nMODE: 在\nWHY: -\nVERDICT: N/A\nHITS: -\n",
        "話2_C",
        "STAGE: 半意識浮現\nMODE: 放\nWHY: x\nVERDICT: N/A\nHITS: -\n",
        "STAGE: 前置積累\nMODE: 在\nWHY: -\nVERDICT: N/A\nHITS: -\n",
        "母親: []\n兒子: []\n",
    ]
    config = _base_config(max_turns=2)
    client = MockLLMClient(responses)
    result = run_session(config=config, llm_client=client, interactive=True)

    # Turn 2 system prompt should contain the injected event under 現場/已發生的事
    turn_2 = yaml.safe_load(
        (result.out_dir / "turns" / "turn_002.yaml").read_text(encoding="utf-8")
    )
    assert "護士開門" in turn_2["prompt_assembled"]["system"]
    # Turn 1 yaml should have director_injection recorded
    turn_1 = yaml.safe_load(
        (result.out_dir / "turns" / "turn_001.yaml").read_text(encoding="utf-8")
    )
    assert turn_1["director_injection"]["event"] == "護士開門"
    # meta.interactive_mode should be True
    meta = yaml.safe_load((result.out_dir / "meta.yaml").read_text(encoding="utf-8"))
    assert meta["interactive_mode"] is True
    assert len(meta["director_injections"]) == 1


def test_non_interactive_peak_does_not_prompt(monkeypatch):
    """Without --interactive flag, fire_release does NOT block stdin."""
    import empty_space.runner as runner_mod
    called = {"count": 0}

    def boom(**kw):
        called["count"] += 1
        return None

    monkeypatch.setattr(runner_mod, "_prompt_for_director_event", boom)

    responses = [
        "- 醫院\n", "- 醫院\n",
        "話1",
        "STAGE: 半意識浮現\nMODE: 放\nWHY: 爆\nVERDICT: fire_release\nHITS: -\n",
        "STAGE: 前置積累\nMODE: 在\nWHY: -\nVERDICT: N/A\nHITS: -\n",
        "話2",
        "STAGE: 半意識浮現\nMODE: 放\nWHY: -\nVERDICT: N/A\nHITS: -\n",
        "STAGE: 前置積累\nMODE: 在\nWHY: -\nVERDICT: N/A\nHITS: -\n",
        "母親: []\n兒子: []\n",
    ]
    config = _base_config(max_turns=2)
    client = MockLLMClient(responses)
    run_session(config=config, llm_client=client, interactive=False)
    assert called["count"] == 0
