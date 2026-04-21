# 空的空間 — Phase 2: Prompt Assembler + Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 2 runtime — prompt assembler, structured-output parser, file writer, and turn-loop runner — so that `uv run python scripts/run_experiment.py <exp_id>` produces a complete 20-turn two-character dialogue landed at `runs/<exp_id>/<timestamp>/`.

**Architecture:** Function-oriented modules (pure functions + dataclasses, no class hierarchies). `prompt_assembler.py` organizes the system prompt into five blocks (貫通軸 / 關係層 / 此刻 / 現場 / 輸出格式) and builds the user message as verbatim dialogue history. `parser.py` extracts the main response plus optional `---IMPRESSIONS---` YAML block with graceful degradation. `writer.py` handles atomic disk persistence per turn. `runner.py` orchestrates the turn loop, director-event injection, speaker alternation, and run-wide state. Schema changes drop `scripted_turns` and actor-level openers; add `scene_premise` and `director_events`.

**Tech Stack:** Python 3.11+, pydantic v2, pyyaml, pytest (+pytest-mock), google-genai (Phase 1 client reused). No new dependencies.

**Spec reference:** `docs/superpowers/specs/2026-04-21-phase2-prompt-assembler-runner.md`

---

## File Structure Overview

**Modify:**
- `src/empty_space/schemas.py` — drop `ScriptedTurn`, `protagonist_opener`, `counterpart_system`, `scripted_turns`; add `scene_premise`, `director_events`; add `CandidateImpression`, `Turn`, `SessionResult` dataclasses
- `experiments/mother_x_son_hospital_v3_001.yaml` — drop opener/system/scripted_turns; add scene_premise + director_events (empty)
- `tests/test_schemas_experiment.py` — update for new fields
- `tests/test_loaders_experiment.py` — update for new fields
- `tests/test_integration_phase1.py` — drop scripted_turns assertions

**Create:**
- `src/empty_space/parser.py` — `parse_response(raw) -> (main, impressions, parse_error)`
- `src/empty_space/prompt_assembler.py` — `build_system_prompt`, `build_user_message`
- `src/empty_space/writer.py` — `init_run`, `append_turn`, `write_meta`
- `src/empty_space/runner.py` — `run_session`, local `SessionState` dataclass
- `scripts/run_experiment.py` — CLI entry
- `tests/test_parser.py`
- `tests/test_prompt_assembler.py`
- `tests/test_writer.py`
- `tests/test_runner_integration.py`

---

## Tasks

### Task 1: Schema and experiment-yaml migration

**Files:**
- Modify: `src/empty_space/schemas.py`
- Modify: `experiments/mother_x_son_hospital_v3_001.yaml`
- Modify: `tests/test_schemas_experiment.py`
- Modify: `tests/test_loaders_experiment.py`
- Modify: `tests/test_integration_phase1.py`

- [ ] **Step 1: Rewrite `src/empty_space/schemas.py`**

Replace the full contents with:

```python
"""Pydantic schemas + dataclasses for personas, settings, experiment configs,
and Phase 2 runtime values (CandidateImpression, Turn, SessionResult).

Design note: Persona/Setting YAMLs vary in structure (v3_tension uses
lists of strings under named fields; baseline uses prose narrative).
Rather than force a rigid schema, store raw YAML content as text and
let the prompt assembler inject it verbatim.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


# --- Pydantic models: loaded from YAML ---

class Persona(BaseModel):
    """A character's identity: 貫通軸 + N 關係層."""
    name: str
    version: str
    core_text: str
    relationship_texts: dict[str, str] = Field(default_factory=dict)


class Setting(BaseModel):
    """A location/environment acting as a third character.

    Contains 既定事實 / 情緒動詞 / 反向記憶 / 印象 — stored as raw YAML content.
    """
    name: str
    content: str


class PersonaRef(BaseModel):
    """Reference to a Persona by path + version (resolved by loader)."""
    path: str
    version: str


class SettingRef(BaseModel):
    """Reference to a Setting YAML file (resolved by loader)."""
    path: str


class InitialState(BaseModel):
    """Opening verb / stage / mode — feeds the initial Judge state."""
    verb: str
    stage: str
    mode: str


class Termination(BaseModel):
    """When to stop the experiment (in addition to max_turns)."""
    on_fire_release: bool = True
    on_basin_lock: bool = True


class ExperimentConfig(BaseModel):
    """Top-level config for a single experiment run.

    Phase 2 removes: protagonist_opener, counterpart_system, scripted_turns.
    Phase 2 adds: scene_premise, director_events.
    Rationale: see spec §2. Director controls the world, not the mouths.
    """
    exp_id: str
    protagonist: PersonaRef
    counterpart: PersonaRef
    setting: SettingRef
    scene_premise: str | None = None
    initial_state: InitialState
    director_events: dict[int, str] = Field(default_factory=dict)
    max_turns: int = 20
    termination: Termination = Field(default_factory=Termination)


# --- Runtime dataclasses (Phase 2) ---

@dataclass(frozen=True)
class CandidateImpression:
    """A single impression line emitted by the role LLM (unvetted until Phase 4 rubric)."""
    text: str
    symbols: list[str]


@dataclass
class Turn:
    """One turn's full record — prompt, response, parse result, timing."""
    turn_number: int
    speaker: Literal["protagonist", "counterpart"]
    persona_name: str                              # e.g., "母親" — for display
    content: str                                   # main response text
    candidate_impressions: list[CandidateImpression]
    prompt_system: str
    prompt_user: str
    raw_response: str
    tokens_in: int
    tokens_out: int
    model: str
    latency_ms: int
    timestamp: str                                 # ISO 8601 "Z" form, e.g. "2026-04-21T11:30:15Z"
    director_events_active: list[tuple[int, str]] # all events triggered so far (including this turn's)
    parse_error: str | None = None


@dataclass
class SessionResult:
    """What run_session returns."""
    exp_id: str
    out_dir: Path
    total_turns: int
    termination_reason: Literal["max_turns"]  # Phase 3 will extend
    total_tokens_in: int
    total_tokens_out: int
    duration_seconds: float
```

- [ ] **Step 2: Rewrite `experiments/mother_x_son_hospital_v3_001.yaml`**

Replace the full contents with:

```yaml
exp_id: mother_x_son_hospital_v3_001
protagonist:
  path: 六個劇中人/母親
  version: v3_tension
counterpart:
  path: 六個劇中人/兒子
  version: v3_tension
setting:
  path: 六個劇中人/環境_醫院.yaml

scene_premise: |
  他們在同一家醫院。父親在 ICU，剛被告知可能撐不過今晚。
  母親和兒子在病房外走廊的長椅上。這是他們十幾年來第一次在同一個空間。

initial_state:
  verb: 承受（靠近）
  stage: 前置積累
  mode: 基線

director_events: {}

max_turns: 20
termination:
  on_fire_release: true
  on_basin_lock: true
```

- [ ] **Step 3: Rewrite `tests/test_schemas_experiment.py`**

Replace the full contents with:

```python
import pytest
from pydantic import ValidationError

from empty_space.schemas import (
    ExperimentConfig,
    PersonaRef,
    SettingRef,
    InitialState,
    Termination,
)


def test_experiment_config_full_construction():
    config = ExperimentConfig(
        exp_id="mother_x_son_hospital_v3_001",
        protagonist=PersonaRef(path="六個劇中人/母親", version="v3_tension"),
        counterpart=PersonaRef(path="六個劇中人/兒子", version="v3_tension"),
        setting=SettingRef(path="六個劇中人/環境_醫院.yaml"),
        scene_premise="他們在醫院。父親在 ICU。",
        initial_state=InitialState(
            verb="承受（靠近）", stage="前置積累", mode="基線"
        ),
        director_events={3: "護士推一張空床進病房"},
        max_turns=20,
        termination=Termination(on_fire_release=True, on_basin_lock=True),
    )
    assert config.exp_id == "mother_x_son_hospital_v3_001"
    assert config.protagonist.version == "v3_tension"
    assert config.scene_premise == "他們在醫院。父親在 ICU。"
    assert config.director_events == {3: "護士推一張空床進病房"}
    assert config.max_turns == 20


def test_termination_has_defaults():
    t = Termination()
    assert t.on_fire_release is True
    assert t.on_basin_lock is True


def test_experiment_defaults_work_with_minimal_input():
    config = ExperimentConfig(
        exp_id="x",
        protagonist=PersonaRef(path="p", version="v"),
        counterpart=PersonaRef(path="q", version="v"),
        setting=SettingRef(path="s.yaml"),
        initial_state=InitialState(verb="v", stage="s", mode="m"),
    )
    assert config.max_turns == 20
    assert config.scene_premise is None
    assert config.director_events == {}
    assert config.termination.on_fire_release is True


def test_experiment_rejects_unknown_fields_not_required():
    # scene_premise is optional; missing is fine
    config = ExperimentConfig(
        exp_id="x",
        protagonist=PersonaRef(path="p", version="v"),
        counterpart=PersonaRef(path="q", version="v"),
        setting=SettingRef(path="s.yaml"),
        initial_state=InitialState(verb="v", stage="s", mode="m"),
    )
    assert config.scene_premise is None


def test_director_events_accepts_int_keys():
    config = ExperimentConfig(
        exp_id="x",
        protagonist=PersonaRef(path="p", version="v"),
        counterpart=PersonaRef(path="q", version="v"),
        setting=SettingRef(path="s.yaml"),
        initial_state=InitialState(verb="v", stage="s", mode="m"),
        director_events={3: "event A", 10: "event B"},
    )
    assert 3 in config.director_events
    assert config.director_events[10] == "event B"
```

- [ ] **Step 4: Rewrite `tests/test_loaders_experiment.py`**

Replace the full contents with:

```python
import pytest
from empty_space.loaders import load_experiment


def test_load_first_experiment():
    config = load_experiment("mother_x_son_hospital_v3_001")
    assert config.exp_id == "mother_x_son_hospital_v3_001"
    assert config.protagonist.path == "六個劇中人/母親"
    assert config.protagonist.version == "v3_tension"
    assert config.counterpart.path == "六個劇中人/兒子"
    assert config.setting.path == "六個劇中人/環境_醫院.yaml"
    assert config.initial_state.verb == "承受（靠近）"
    assert config.scene_premise is not None
    assert "ICU" in config.scene_premise
    assert config.director_events == {}
    assert config.max_turns == 20
    assert config.termination.on_fire_release is True


def test_load_missing_experiment_raises():
    with pytest.raises(FileNotFoundError):
        load_experiment("nonexistent_experiment")
```

- [ ] **Step 5: Rewrite `tests/test_integration_phase1.py`**

Replace the full contents with:

```python
"""Phase 1 integration test: load a real experiment with all its
persona/setting dependencies, verify everything stitches together.
"""
from empty_space.loaders import load_experiment, load_persona, load_setting


def test_full_experiment_dependency_chain():
    config = load_experiment("mother_x_son_hospital_v3_001")
    assert config.exp_id == "mother_x_son_hospital_v3_001"

    protagonist = load_persona(
        config.protagonist.path,
        version=config.protagonist.version,
    )
    assert protagonist.name == "母親"
    assert "兒子" in protagonist.relationship_texts

    counterpart = load_persona(
        config.counterpart.path,
        version=config.counterpart.version,
    )
    assert counterpart.name == "兒子"
    assert "母親" in counterpart.relationship_texts

    setting = load_setting(config.setting.path)
    assert setting.name == "環境_醫院"

    assert "關係語境" in protagonist.relationship_texts["兒子"]
    assert "關係語境" in counterpart.relationship_texts["母親"]

    # Phase 2: scene_premise is present, director_events is empty by default
    assert config.scene_premise is not None
    assert config.director_events == {}
```

- [ ] **Step 6: Run tests to verify schema migration**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_schemas_experiment.py tests/test_loaders_experiment.py tests/test_integration_phase1.py -v`

Expected: all tests PASS. Any import errors referring to `ScriptedTurn`, `protagonist_opener`, or `counterpart_system` mean something was missed.

- [ ] **Step 7: Run full Phase 1 test suite to confirm no regression elsewhere**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest -v`

Expected: all existing tests PASS (the 26-test Phase 1 suite minus any scripted-related assertions now removed).

- [ ] **Step 8: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/schemas.py experiments/mother_x_son_hospital_v3_001.yaml tests/test_schemas_experiment.py tests/test_loaders_experiment.py tests/test_integration_phase1.py
git commit -m "feat(schemas): Phase 2 config migration — scene_premise + director_events

Drop scripted_turns, protagonist_opener, counterpart_system.
Add scene_premise, director_events, CandidateImpression, Turn, SessionResult.
Experiment yaml migrated; existing tests updated."
```

---

### Task 2: `parser.py` — structured output parser

**Files:**
- Create: `src/empty_space/parser.py`
- Create: `tests/test_parser.py`

- [ ] **Step 1: Write the failing tests in `tests/test_parser.py`**

```python
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
    raw = """她低著頭。
---IMPRESSIONS---
"""
    main, impressions, err = parse_response(raw)
    # impressions_block is empty string after marker → yaml.safe_load returns None → not a list
    assert main == "她低著頭."[:-1] + "。" or main == "她低著頭。"
    assert impressions == []
    # None root triggers "not a list" error — acceptable
    assert err is None or "list" in err


def test_impressions_is_none_after_marker():
    """If YAML under marker parses to None, treat as empty list (not error)."""
    raw = """話。

---IMPRESSIONS---
# just a comment
"""
    main, impressions, err = parse_response(raw)
    assert main == "話。"
    assert impressions == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_parser.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'empty_space.parser'` or `ImportError`.

- [ ] **Step 3: Create `src/empty_space/parser.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_parser.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/parser.py tests/test_parser.py
git commit -m "feat(parser): structured output parser with graceful degradation

Parses Gemini role responses into (main, impressions, parse_error).
Main response always recovered; impressions optional.
Covers all tolerance cases from spec §6.2."
```

---

### Task 3: `prompt_assembler.build_system_prompt`

**Files:**
- Create: `src/empty_space/prompt_assembler.py` (partial — this task only)
- Create: `tests/test_prompt_assembler.py` (partial — this task only)

- [ ] **Step 1: Write failing tests in `tests/test_prompt_assembler.py`**

```python
"""Tests for prompt_assembler.build_system_prompt + build_user_message.

Covers spec §4 structure and §8.2 assertions.
"""
import pytest

from empty_space.prompt_assembler import build_system_prompt, build_user_message
from empty_space.schemas import (
    Persona,
    Setting,
    InitialState,
)


@pytest.fixture
def mother_persona() -> Persona:
    return Persona(
        name="母親",
        version="v3_tension",
        core_text="## 貫通軸\n動作詞: 承受\n",
        relationship_texts={"兒子": "## 關係語境\n母愛的退讓\n"},
    )


@pytest.fixture
def son_persona() -> Persona:
    return Persona(
        name="兒子",
        version="v3_tension",
        core_text="## 貫通軸\n動作詞: 迴避\n",
        relationship_texts={"母親": "## 關係語境\n不參與\n"},
    )


@pytest.fixture
def hospital_setting() -> Setting:
    return Setting(
        name="環境_醫院",
        content="## 設定\n壓低、懸吊、剝除\n",
    )


@pytest.fixture
def initial_state() -> InitialState:
    return InitialState(verb="承受（靠近）", stage="前置積累", mode="基線")


# --- build_system_prompt ---

def test_system_prompt_has_five_blocks_in_order(
    mother_persona, son_persona, hospital_setting, initial_state
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
    )
    # All five block headers present
    assert "## 貫通軸" in prompt
    assert "## 關係層：對兒子" in prompt
    assert "## 此刻" in prompt
    assert "## 現場" in prompt
    assert "## 輸出格式" in prompt
    # In correct order
    assert prompt.index("## 貫通軸") < prompt.index("## 關係層：對兒子")
    assert prompt.index("## 關係層：對兒子") < prompt.index("## 此刻")
    assert prompt.index("## 此刻") < prompt.index("## 現場")
    assert prompt.index("## 現場") < prompt.index("## 輸出格式")


def test_system_prompt_embeds_persona_core_text_verbatim(
    mother_persona, son_persona, hospital_setting, initial_state
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
    )
    assert mother_persona.core_text in prompt


def test_system_prompt_embeds_correct_relationship_text(
    mother_persona, son_persona, hospital_setting, initial_state
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
    )
    assert mother_persona.relationship_texts["兒子"] in prompt


def test_system_prompt_this_moment_block_formatting(
    mother_persona, son_persona, hospital_setting, initial_state
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
    )
    assert "動作詞：承受（靠近）" in prompt
    assert "階段：前置積累" in prompt
    assert "模式：基線" in prompt


def test_system_prompt_scene_premise_sub_block_omitted_when_none(
    mother_persona, son_persona, hospital_setting, initial_state
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
    )
    assert "### 場景前提" not in prompt


def test_system_prompt_scene_premise_appears_when_given(
    mother_persona, son_persona, hospital_setting, initial_state
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise="父親在 ICU。",
        initial_state=initial_state,
        active_events=[],
    )
    assert "### 場景前提" in prompt
    assert "父親在 ICU。" in prompt


def test_system_prompt_events_sub_block_omitted_when_empty(
    mother_persona, son_persona, hospital_setting, initial_state
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
    )
    assert "### 已發生的事" not in prompt


def test_system_prompt_events_listed_in_turn_order(
    mother_persona, son_persona, hospital_setting, initial_state
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[(3, "護士推空床進病房"), (10, "走廊傳來長音")],
    )
    assert "### 已發生的事" in prompt
    # Events appear in order, each with its turn prefix
    event3_pos = prompt.find("Turn 3：護士推空床進病房")
    event10_pos = prompt.find("Turn 10：走廊傳來長音")
    assert event3_pos != -1
    assert event10_pos != -1
    assert event3_pos < event10_pos


def test_system_prompt_output_format_contains_marker(
    mother_persona, son_persona, hospital_setting, initial_state
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
    )
    assert "---IMPRESSIONS---" in prompt


def test_system_prompt_setting_content_embedded(
    mother_persona, son_persona, hospital_setting, initial_state
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
    )
    assert hospital_setting.content in prompt
```

- [ ] **Step 2: Run to verify tests fail**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_prompt_assembler.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'empty_space.prompt_assembler'`.

- [ ] **Step 3: Create `src/empty_space/prompt_assembler.py` (build_system_prompt only)**

```python
"""Pure functions that build the system prompt and user message for each turn.

Spec §4: five-block system prompt (貫通軸 / 關係層 / 此刻 / 現場 / 輸出格式),
user message is verbatim dialogue history with no tail anchor.
"""
from empty_space.schemas import (
    InitialState,
    Persona,
    Setting,
    Turn,
)


_OUTPUT_FORMAT_INSTRUCTION = """\
先寫你要說的話。說完之後，另起一行寫 "---IMPRESSIONS---"，然後以 YAML list 格式
列出你這輪浮現的印象句（若無，省略整段 ---IMPRESSIONS--- 區塊）。

範例：
---
她低著頭，沒有回答。

---IMPRESSIONS---
- text: "她的沉默在這一刻比任何辯解都沉"
  symbols: [沉默, 辯解, 愧疚]
- text: "她的手在膝上動了一下，又停了"
  symbols: [遲疑, 克制]
---"""


def build_system_prompt(
    persona: Persona,
    counterpart_name: str,
    setting: Setting,
    scene_premise: str | None,
    initial_state: InitialState,
    active_events: list[tuple[int, str]],
    ambient_echo: list[str] | None = None,
) -> str:
    """Assemble the system prompt for one role's turn.

    Block order (spec §4.1): 貫通軸 → 關係層 → 此刻 → 現場 → 輸出格式.
    `ambient_echo` is a Phase 4 hook; Phase 2 callers pass None or [].
    """
    _ = ambient_echo  # reserved for Phase 4
    relationship_text = persona.relationship_texts.get(counterpart_name, "")

    blocks: list[str] = []

    blocks.append(f"## 貫通軸\n{persona.core_text.rstrip()}")

    blocks.append(f"## 關係層：對{counterpart_name}\n{relationship_text.rstrip()}")

    blocks.append(
        "## 此刻\n"
        f"動作詞：{initial_state.verb}\n"
        f"階段：{initial_state.stage}\n"
        f"模式：{initial_state.mode}"
    )

    scene_parts: list[str] = [setting.content.rstrip()]
    if scene_premise:
        scene_parts.append(f"### 場景前提\n{scene_premise.rstrip()}")
    if active_events:
        event_lines = "\n".join(
            f"Turn {turn}：{content}" for turn, content in active_events
        )
        scene_parts.append(f"### 已發生的事\n{event_lines}")
    blocks.append("## 現場\n" + "\n\n".join(scene_parts))

    blocks.append(f"## 輸出格式\n{_OUTPUT_FORMAT_INSTRUCTION}")

    return "\n\n".join(blocks)
```

Note: all colons in the `## 此刻` block are full-width `：` (U+FF1A). This matches the convention used elsewhere in the Chinese-language prompts and tests.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_prompt_assembler.py -v`

Expected: all 10 system-prompt tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/prompt_assembler.py tests/test_prompt_assembler.py
git commit -m "feat(prompt_assembler): build_system_prompt with five-block structure

Blocks in order: 貫通軸 → 關係層 → 此刻 → 現場 → 輸出格式.
Scene premise and director events appear as sub-blocks under 現場.
YAML content embedded verbatim from Persona/Setting."
```

---

### Task 4: `prompt_assembler.build_user_message`

**Files:**
- Modify: `src/empty_space/prompt_assembler.py` (add function)
- Modify: `tests/test_prompt_assembler.py` (add tests)

- [ ] **Step 1: Add failing tests at the bottom of `tests/test_prompt_assembler.py`**

```python
# --- build_user_message ---

def _make_turn(n: int, speaker: str, name: str, content: str) -> Turn:
    from empty_space.schemas import Turn
    return Turn(
        turn_number=n,
        speaker=speaker,  # type: ignore[arg-type]
        persona_name=name,
        content=content,
        candidate_impressions=[],
        prompt_system="",
        prompt_user="",
        raw_response="",
        tokens_in=0,
        tokens_out=0,
        model="gemini-2.5-flash",
        latency_ms=0,
        timestamp="2026-04-21T11:30:00Z",
        director_events_active=[],
        parse_error=None,
    )


def test_user_message_turn_1_is_scene_opening():
    msg = build_user_message(history=[])
    assert msg == "（場景開始。）"


def test_user_message_turn_2_single_history_line():
    history = [_make_turn(1, "protagonist", "母親", "你回來了。")]
    msg = build_user_message(history=history)
    assert msg == "[Turn 1 母親] 你回來了。"


def test_user_message_turn_3_two_history_lines_in_order():
    history = [
        _make_turn(1, "protagonist", "母親", "你回來了。"),
        _make_turn(2, "counterpart", "兒子", "嗯。"),
    ]
    msg = build_user_message(history=history)
    assert msg == "[Turn 1 母親] 你回來了。\n[Turn 2 兒子] 嗯。"


def test_user_message_uses_persona_name_not_role_code():
    history = [_make_turn(1, "protagonist", "母親", "x")]
    msg = build_user_message(history=history)
    assert "母親" in msg
    assert "protagonist" not in msg


def test_user_message_has_no_tail_anchor_or_instruction():
    history = [
        _make_turn(1, "protagonist", "母親", "你回來了。"),
        _make_turn(2, "counterpart", "兒子", "嗯。"),
    ]
    msg = build_user_message(history=history)
    # No directive like "說第 N 句" or "你是 母親" appended
    assert "說第" not in msg
    assert "你是" not in msg
    assert msg.endswith("嗯。")
```

The `Turn` import inside the helper avoids polluting the top-level imports; you can also move it up for clarity.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_prompt_assembler.py::test_user_message_turn_1_is_scene_opening -v`

Expected: FAIL — `AttributeError: module 'empty_space.prompt_assembler' has no attribute 'build_user_message'` or an ImportError in the import line at the top of the test file.

- [ ] **Step 3: Add `build_user_message` to `src/empty_space/prompt_assembler.py`**

Append the following to the bottom of the file:

```python
def build_user_message(history: list[Turn]) -> str:
    """Assemble the user message from accumulated turn history.

    Turn 1 (empty history) → "（場景開始。）" (minimal mechanical trigger).
    Turn N ≥ 2 → lines of "[Turn K <persona_name>] <content>", one per turn.

    No tail anchor, no directive — role shaping is the system prompt's job.
    """
    if not history:
        return "（場景開始。）"
    return "\n".join(
        f"[Turn {t.turn_number} {t.persona_name}] {t.content}"
        for t in history
    )
```

Also add `Turn` to the imports at the top (it was already declared in Task 3's code).

- [ ] **Step 4: Run all prompt_assembler tests**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_prompt_assembler.py -v`

Expected: all tests PASS (system prompt tests + user message tests).

- [ ] **Step 5: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/prompt_assembler.py tests/test_prompt_assembler.py
git commit -m "feat(prompt_assembler): build_user_message with verbatim dialogue history

Turn 1 emits minimal '（場景開始。）' trigger.
Turn ≥ 2 emits [Turn N <name>] <content> lines, no tail anchor.
Role shaping entirely in system prompt."
```

---

### Task 5: `writer.init_run`

**Files:**
- Create: `src/empty_space/writer.py` (partial — this task only)
- Create: `tests/test_writer.py` (partial — this task only)

- [ ] **Step 1: Write failing tests in `tests/test_writer.py`**

```python
"""Tests for writer — run directory init, per-turn append, meta writeout."""
import json
from pathlib import Path

import pytest
import yaml

from empty_space.schemas import (
    ExperimentConfig,
    PersonaRef,
    SettingRef,
    InitialState,
    Termination,
)
from empty_space.writer import init_run


@pytest.fixture
def sample_config() -> ExperimentConfig:
    return ExperimentConfig(
        exp_id="test_exp_001",
        protagonist=PersonaRef(path="六個劇中人/母親", version="v3_tension"),
        counterpart=PersonaRef(path="六個劇中人/兒子", version="v3_tension"),
        setting=SettingRef(path="六個劇中人/環境_醫院.yaml"),
        scene_premise="父親在 ICU。",
        initial_state=InitialState(verb="承受", stage="前置積累", mode="基線"),
        director_events={3: "護士推床進來"},
        max_turns=5,
        termination=Termination(),
    )


def test_init_run_creates_directory_structure(tmp_path, sample_config):
    out_dir = tmp_path / "runs" / "test_exp_001" / "2026-04-21T11-30-15"
    init_run(out_dir, sample_config)
    assert out_dir.is_dir()
    assert (out_dir / "turns").is_dir()


def test_init_run_writes_config_yaml_deep_copy(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    config_file = out_dir / "config.yaml"
    assert config_file.is_file()
    loaded = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert loaded["exp_id"] == "test_exp_001"
    assert loaded["scene_premise"] == "父親在 ICU。"
    assert loaded["director_events"] == {3: "護士推床進來"}
    assert loaded["max_turns"] == 5


def test_init_run_initializes_conversation_md_with_scene(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    conv_md = (out_dir / "conversation.md").read_text(encoding="utf-8")
    assert "test_exp_001" in conv_md
    assert "父親在 ICU。" in conv_md


def test_init_run_creates_empty_conversation_jsonl(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    conv_jsonl = out_dir / "conversation.jsonl"
    assert conv_jsonl.is_file()
    assert conv_jsonl.read_text(encoding="utf-8") == ""
```

- [ ] **Step 2: Run to verify tests fail**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_writer.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'empty_space.writer'`.

- [ ] **Step 3: Create `src/empty_space/writer.py` with init_run**

```python
"""Atomic per-turn persistence for Phase 2 runs.

Writes to runs/<exp_id>/<timestamp>/:
  - config.yaml (deep copy of ExperimentConfig)
  - turns/turn_NNN.yaml (one file per turn, atomic rename)
  - conversation.md (append per turn, markdown)
  - conversation.jsonl (append per turn, one JSON object per line)
  - meta.yaml (written once at the end)

Atomicity for yaml files: write to .tmp, os.replace → final.
"""
import json
import os
from pathlib import Path

import yaml

from empty_space.schemas import ExperimentConfig


def init_run(out_dir: Path, config: ExperimentConfig) -> None:
    """Create the run directory skeleton and write config.yaml + conversation init."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "turns").mkdir(exist_ok=True)

    # config.yaml — deep copy via pydantic's model_dump
    config_dump = config.model_dump()
    _atomic_write_yaml(out_dir / "config.yaml", config_dump)

    # conversation.md header
    scene = config.scene_premise or ""
    header_lines = [
        f"# {config.exp_id} @ {out_dir.name}",
        "",
    ]
    if scene:
        header_lines.extend([f"**場景**：{scene.rstrip()}", "", "---", ""])
    else:
        header_lines.extend(["---", ""])
    (out_dir / "conversation.md").write_text(
        "\n".join(header_lines), encoding="utf-8"
    )

    # conversation.jsonl — empty file, appended per turn
    (out_dir / "conversation.jsonl").write_text("", encoding="utf-8")


def _atomic_write_yaml(path: Path, data: object) -> None:
    """Write YAML via .tmp + os.replace for atomicity."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)
```

- [ ] **Step 4: Run tests**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_writer.py -v`

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/writer.py tests/test_writer.py
git commit -m "feat(writer): init_run — create run dir, config.yaml, conversation files

Creates runs/<exp_id>/<timestamp>/ with turns/ subdir, copies
ExperimentConfig to config.yaml, initializes conversation.md header
with scene_premise, creates empty conversation.jsonl."
```

---

### Task 6: `writer.append_turn`

**Files:**
- Modify: `src/empty_space/writer.py` (add function)
- Modify: `tests/test_writer.py` (add tests)

- [ ] **Step 1: Add failing tests to `tests/test_writer.py`**

Append the following:

```python
from empty_space.schemas import CandidateImpression, Turn
from empty_space.writer import append_turn


def _make_turn(
    *,
    turn_number: int,
    speaker: str,
    name: str,
    content: str,
    impressions: list[CandidateImpression] | None = None,
    events_active: list[tuple[int, str]] | None = None,
    parse_error: str | None = None,
) -> Turn:
    return Turn(
        turn_number=turn_number,
        speaker=speaker,  # type: ignore[arg-type]
        persona_name=name,
        content=content,
        candidate_impressions=impressions or [],
        prompt_system="(system prompt verbatim)",
        prompt_user="(user message verbatim)",
        raw_response=content + "\n",
        tokens_in=100,
        tokens_out=20,
        model="gemini-2.5-flash",
        latency_ms=300,
        timestamp="2026-04-21T11:30:01Z",
        director_events_active=events_active or [],
        parse_error=parse_error,
    )


def test_append_turn_writes_turn_yaml(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    turn = _make_turn(turn_number=1, speaker="protagonist", name="母親", content="你回來了。")
    append_turn(out_dir, turn)

    turn_file = out_dir / "turns" / "turn_001.yaml"
    assert turn_file.is_file()
    loaded = yaml.safe_load(turn_file.read_text(encoding="utf-8"))
    assert loaded["turn"] == 1
    assert loaded["speaker"] == "protagonist"
    assert loaded["response"]["content"] == "你回來了。"
    assert loaded["response"]["model"] == "gemini-2.5-flash"
    assert loaded["prompt_assembled"]["system"] == "(system prompt verbatim)"
    assert loaded["parse_error"] is None


def test_append_turn_numbers_are_zero_padded_to_three(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    turn = _make_turn(turn_number=12, speaker="counterpart", name="兒子", content="嗯。")
    append_turn(out_dir, turn)
    assert (out_dir / "turns" / "turn_012.yaml").is_file()


def test_append_turn_records_candidate_impressions(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    turn = _make_turn(
        turn_number=1,
        speaker="protagonist",
        name="母親",
        content="話。",
        impressions=[
            CandidateImpression(text="她的沉默很沉", symbols=["沉默", "愧疚"]),
        ],
    )
    append_turn(out_dir, turn)
    loaded = yaml.safe_load((out_dir / "turns" / "turn_001.yaml").read_text(encoding="utf-8"))
    assert len(loaded["candidate_impressions"]) == 1
    assert loaded["candidate_impressions"][0]["text"] == "她的沉默很沉"
    assert loaded["candidate_impressions"][0]["symbols"] == ["沉默", "愧疚"]


def test_append_turn_records_parse_error(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    turn = _make_turn(
        turn_number=1, speaker="protagonist", name="母親",
        content="話。", parse_error="YAML parse error: bad",
    )
    append_turn(out_dir, turn)
    loaded = yaml.safe_load((out_dir / "turns" / "turn_001.yaml").read_text(encoding="utf-8"))
    assert loaded["parse_error"] == "YAML parse error: bad"


def test_append_turn_records_director_events_active(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    turn = _make_turn(
        turn_number=4, speaker="counterpart", name="兒子", content="嗯。",
        events_active=[(3, "護士推床進來")],
    )
    append_turn(out_dir, turn)
    loaded = yaml.safe_load((out_dir / "turns" / "turn_004.yaml").read_text(encoding="utf-8"))
    assert loaded["director_events_active"] == [{"turn": 3, "content": "護士推床進來"}]


def test_append_turn_appends_to_conversation_md(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    append_turn(out_dir, _make_turn(
        turn_number=1, speaker="protagonist", name="母親", content="你回來了。",
    ))
    append_turn(out_dir, _make_turn(
        turn_number=2, speaker="counterpart", name="兒子", content="嗯。",
    ))
    md = (out_dir / "conversation.md").read_text(encoding="utf-8")
    assert "**Turn 1 · 母親**\n你回來了。" in md
    assert "**Turn 2 · 兒子**\n嗯。" in md


def test_append_turn_inserts_director_event_marker_before_triggering_turn(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    # Turn 3 triggered a new event at turn 3 — last element of events_active has turn == turn_number
    append_turn(out_dir, _make_turn(
        turn_number=3, speaker="protagonist", name="母親", content="⋯⋯",
        events_active=[(3, "護士推床進來")],
    ))
    md = (out_dir / "conversation.md").read_text(encoding="utf-8")
    marker_pos = md.find("**[世界] Turn 3：護士推床進來**")
    turn_header_pos = md.find("**Turn 3 · 母親**")
    assert marker_pos != -1
    assert turn_header_pos != -1
    assert marker_pos < turn_header_pos


def test_append_turn_no_event_marker_when_no_new_event_this_turn(tmp_path, sample_config):
    """Turn 5 sees event from Turn 3 in its active list but must not re-print marker."""
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    append_turn(out_dir, _make_turn(
        turn_number=5, speaker="protagonist", name="母親", content="話。",
        events_active=[(3, "護士推床進來")],  # old event, not triggered this turn
    ))
    md = (out_dir / "conversation.md").read_text(encoding="utf-8")
    assert "**[世界] Turn 3：護士推床進來**" not in md
    assert "**Turn 5 · 母親**" in md


def test_append_turn_appends_to_conversation_jsonl(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    append_turn(out_dir, _make_turn(
        turn_number=1, speaker="protagonist", name="母親", content="你回來了。",
    ))
    append_turn(out_dir, _make_turn(
        turn_number=3, speaker="protagonist", name="母親", content="⋯⋯",
        events_active=[(3, "護士推床進來")],
    ))
    lines = (out_dir / "conversation.jsonl").read_text(encoding="utf-8").strip().split("\n")
    # First turn entry, then director_event entry + turn entry for Turn 3
    entries = [json.loads(ln) for ln in lines]
    assert entries[0] == {
        "turn": 1, "speaker": "protagonist", "name": "母親",
        "content": "你回來了。", "timestamp": "2026-04-21T11:30:01Z",
    }
    assert entries[1] == {
        "type": "director_event", "turn": 3, "content": "護士推床進來",
    }
    assert entries[2]["turn"] == 3
    assert entries[2]["content"] == "⋯⋯"
```

- [ ] **Step 2: Run to verify tests fail**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_writer.py -v`

Expected: FAIL — `ImportError: cannot import name 'append_turn' from 'empty_space.writer'`.

- [ ] **Step 3: Add `append_turn` to `src/empty_space/writer.py`**

Append to the bottom of the file:

```python
def append_turn(out_dir: Path, turn: "Turn") -> None:
    """Write turn_NNN.yaml atomically; append director_event marker (if new) and
    turn entry to conversation.md + conversation.jsonl.
    """
    from empty_space.schemas import Turn as _Turn  # local import avoids circularity

    turn_file = out_dir / "turns" / f"turn_{turn.turn_number:03d}.yaml"
    _atomic_write_yaml(turn_file, _turn_to_yaml_dict(turn))

    new_event = _new_event_this_turn(turn)
    _append_conversation_md(out_dir, turn, new_event)
    _append_conversation_jsonl(out_dir, turn, new_event)


def _new_event_this_turn(turn: "Turn") -> tuple[int, str] | None:
    """Return the event triggered AT this turn, if any.

    Convention: runner appends director_events[turn_number] to active_events
    just before the LLM call, so if the last active event's turn equals
    turn.turn_number, that event was triggered this turn.
    """
    if turn.director_events_active and turn.director_events_active[-1][0] == turn.turn_number:
        return turn.director_events_active[-1]
    return None


def _turn_to_yaml_dict(turn: "Turn") -> dict:
    return {
        "turn": turn.turn_number,
        "speaker": turn.speaker,
        "persona_name": turn.persona_name,
        "timestamp": turn.timestamp,
        "prompt_assembled": {
            "system": turn.prompt_system,
            "user": turn.prompt_user,
            "tokens": {
                "system": turn.tokens_in,
                "user": 0,  # Phase 2 doesn't split system/user tokens separately; leave 0
            },
        },
        "response": {
            "content": turn.content,
            "raw": turn.raw_response,
            "tokens_out": turn.tokens_out,
            "model": turn.model,
            "latency_ms": turn.latency_ms,
        },
        "candidate_impressions": [
            {"text": imp.text, "symbols": list(imp.symbols)}
            for imp in turn.candidate_impressions
        ],
        "director_events_active": [
            {"turn": t, "content": c} for t, c in turn.director_events_active
        ],
        "parse_error": turn.parse_error,
    }


def _append_conversation_md(
    out_dir: Path, turn: "Turn", new_event: tuple[int, str] | None
) -> None:
    parts: list[str] = []
    if new_event is not None:
        parts.append(f"**[世界] Turn {new_event[0]}：{new_event[1]}**\n")
    parts.append(f"**Turn {turn.turn_number} · {turn.persona_name}**\n{turn.content}\n")
    with (out_dir / "conversation.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")


def _append_conversation_jsonl(
    out_dir: Path, turn: "Turn", new_event: tuple[int, str] | None
) -> None:
    lines: list[str] = []
    if new_event is not None:
        lines.append(json.dumps(
            {"type": "director_event", "turn": new_event[0], "content": new_event[1]},
            ensure_ascii=False,
        ))
    lines.append(json.dumps(
        {
            "turn": turn.turn_number,
            "speaker": turn.speaker,
            "name": turn.persona_name,
            "content": turn.content,
            "timestamp": turn.timestamp,
        },
        ensure_ascii=False,
    ))
    with (out_dir / "conversation.jsonl").open("a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")
```

Also add the `TYPE_CHECKING` import at top if you want proper type hints without runtime overhead:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from empty_space.schemas import Turn
```

- [ ] **Step 4: Run tests**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_writer.py -v`

Expected: all writer tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/writer.py tests/test_writer.py
git commit -m "feat(writer): append_turn — atomic turn yaml + conversation md/jsonl append

Writes turns/turn_NNN.yaml via .tmp + os.replace.
Appends **Turn N · name** block to conversation.md (with [世界] marker
when a new director event triggered this turn).
Appends JSON line to conversation.jsonl (director_event line first if
new event, then turn entry)."
```

---

### Task 7: `writer.write_meta`

**Files:**
- Modify: `src/empty_space/writer.py` (add function)
- Modify: `tests/test_writer.py` (add tests)

- [ ] **Step 1: Add failing tests to `tests/test_writer.py`**

Append:

```python
from empty_space.writer import write_meta


def test_write_meta_records_all_summary_fields(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    # Simulate two turns
    append_turn(out_dir, _make_turn(
        turn_number=1, speaker="protagonist", name="母親", content="你回來了。",
        impressions=[CandidateImpression(text="x", symbols=[])],
    ))
    append_turn(out_dir, _make_turn(
        turn_number=2, speaker="counterpart", name="兒子", content="嗯。",
        parse_error="YAML parse error: something",
    ))
    write_meta(
        out_dir=out_dir,
        config=sample_config,
        total_turns=2,
        termination_reason="max_turns",
        total_tokens_in=200,
        total_tokens_out=40,
        total_candidate_impressions=1,
        turns_with_parse_error=1,
        director_events_triggered=[(3, "護士推床進來")],
        models_used=["gemini-2.5-flash"],
        duration_seconds=12.5,
    )
    meta = yaml.safe_load((out_dir / "meta.yaml").read_text(encoding="utf-8"))
    assert meta["exp_id"] == "test_exp_001"
    assert meta["total_turns"] == 2
    assert meta["termination_reason"] == "max_turns"
    assert meta["total_tokens_in"] == 200
    assert meta["total_tokens_out"] == 40
    assert meta["duration_seconds"] == 12.5
    assert meta["total_candidate_impressions"] == 1
    assert meta["turns_with_parse_error"] == 1
    assert meta["director_events_triggered"] == [{"turn": 3, "content": "護士推床進來"}]
    assert meta["models_used"] == ["gemini-2.5-flash"]
    assert "run_timestamp" in meta
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_writer.py::test_write_meta_records_all_summary_fields -v`

Expected: FAIL — `ImportError: cannot import name 'write_meta'`.

- [ ] **Step 3: Add `write_meta` to `src/empty_space/writer.py`**

Append:

```python
def write_meta(
    *,
    out_dir: Path,
    config: ExperimentConfig,
    total_turns: int,
    termination_reason: str,
    total_tokens_in: int,
    total_tokens_out: int,
    total_candidate_impressions: int,
    turns_with_parse_error: int,
    director_events_triggered: list[tuple[int, str]],
    models_used: list[str],
    duration_seconds: float,
) -> None:
    """Write meta.yaml with session-level summary."""
    meta = {
        "exp_id": config.exp_id,
        "run_timestamp": out_dir.name,
        "total_turns": total_turns,
        "termination_reason": termination_reason,
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "duration_seconds": duration_seconds,
        "total_candidate_impressions": total_candidate_impressions,
        "turns_with_parse_error": turns_with_parse_error,
        "director_events_triggered": [
            {"turn": t, "content": c} for t, c in director_events_triggered
        ],
        "models_used": models_used,
    }
    _atomic_write_yaml(out_dir / "meta.yaml", meta)
```

- [ ] **Step 4: Run tests**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_writer.py -v`

Expected: all writer tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/writer.py tests/test_writer.py
git commit -m "feat(writer): write_meta — session summary with tokens, parse errors, events"
```

---

### Task 8: `runner.run_session` — happy path with mock LLM

**Files:**
- Create: `src/empty_space/runner.py`
- Create: `tests/test_runner_integration.py`

- [ ] **Step 1: Write failing tests in `tests/test_runner_integration.py`**

```python
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
```

- [ ] **Step 2: Run to verify tests fail**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_runner_integration.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'empty_space.runner'`.

- [ ] **Step 3: Create `src/empty_space/runner.py`**

```python
"""Session runner — orchestrates the Phase 2 turn loop.

run_session(config, llm_client) -> SessionResult

For each turn:
  1. Determine speaker (odd=protagonist / even=counterpart).
  2. Trigger director event if scheduled.
  3. Build system + user prompts.
  4. Call LLM.
  5. Parse response.
  6. Append Turn to in-memory state.
  7. Write turn_NNN.yaml + conversation append.

Termination (Phase 2): max_turns only.
Errors: LLM exceptions propagate; partial turn files are kept; meta.yaml is not written.
"""
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from empty_space.loaders import load_persona, load_setting
from empty_space.llm import GeminiResponse
from empty_space.parser import parse_response
from empty_space.paths import RUNS_DIR
from empty_space.prompt_assembler import build_system_prompt, build_user_message
from empty_space.schemas import (
    ExperimentConfig,
    Persona,
    SessionResult,
    Setting,
    Turn,
)
from empty_space.writer import append_turn, init_run, write_meta


class LLMClient(Protocol):
    """Duck-typed interface that both GeminiClient and MockLLMClient satisfy."""
    def generate(self, *, system: str, user: str, model: str = ...) -> GeminiResponse: ...


@dataclass
class SessionState:
    """Runner-internal state. Not persisted to schemas.py."""
    config: ExperimentConfig
    protagonist: Persona
    counterpart: Persona
    setting: Setting
    turns: list[Turn] = field(default_factory=list)
    active_events: list[tuple[int, str]] = field(default_factory=list)
    out_dir: Path | None = None


def run_session(
    *, config: ExperimentConfig, llm_client: LLMClient
) -> SessionResult:
    """Run one experiment session end-to-end."""
    protagonist = load_persona(config.protagonist.path, version=config.protagonist.version)
    counterpart = load_persona(config.counterpart.path, version=config.counterpart.version)
    setting = load_setting(config.setting.path)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = RUNS_DIR / config.exp_id / timestamp
    init_run(out_dir, config)

    state = SessionState(
        config=config,
        protagonist=protagonist,
        counterpart=counterpart,
        setting=setting,
        out_dir=out_dir,
    )

    start_time = time.monotonic()
    events_triggered: list[tuple[int, str]] = []

    for n in range(1, config.max_turns + 1):
        # 1. speaker
        speaker_role = "protagonist" if n % 2 == 1 else "counterpart"
        speaker_persona = protagonist if speaker_role == "protagonist" else counterpart
        counterpart_name = counterpart.name if speaker_role == "protagonist" else protagonist.name

        # 2. director event trigger
        if n in config.director_events:
            new_event = (n, config.director_events[n])
            state.active_events.append(new_event)
            events_triggered.append(new_event)

        # 3. prompts
        system_prompt = build_system_prompt(
            persona=speaker_persona,
            counterpart_name=counterpart_name,
            setting=setting,
            scene_premise=config.scene_premise,
            initial_state=config.initial_state,
            active_events=state.active_events,
        )
        user_message = build_user_message(history=state.turns)

        # 4. LLM call
        resp = llm_client.generate(
            system=system_prompt,
            user=user_message,
            model="gemini-2.5-flash",
        )

        # 5. parse
        main_content, impressions, parse_err = parse_response(resp.content)

        # 6. turn record
        turn = Turn(
            turn_number=n,
            speaker=speaker_role,  # type: ignore[arg-type]
            persona_name=speaker_persona.name,
            content=main_content,
            candidate_impressions=impressions,
            prompt_system=system_prompt,
            prompt_user=user_message,
            raw_response=resp.content,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            model=resp.model,
            latency_ms=resp.latency_ms,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            director_events_active=list(state.active_events),
            parse_error=parse_err,
        )
        state.turns.append(turn)

        # 7. append
        append_turn(out_dir, turn)

    duration = time.monotonic() - start_time
    termination_reason = "max_turns"

    total_tokens_in = sum(t.tokens_in for t in state.turns)
    total_tokens_out = sum(t.tokens_out for t in state.turns)
    total_candidate_impressions = sum(len(t.candidate_impressions) for t in state.turns)
    turns_with_parse_error = sum(1 for t in state.turns if t.parse_error is not None)
    models_used = sorted({t.model for t in state.turns})

    write_meta(
        out_dir=out_dir,
        config=config,
        total_turns=len(state.turns),
        termination_reason=termination_reason,
        total_tokens_in=total_tokens_in,
        total_tokens_out=total_tokens_out,
        total_candidate_impressions=total_candidate_impressions,
        turns_with_parse_error=turns_with_parse_error,
        director_events_triggered=events_triggered,
        models_used=models_used,
        duration_seconds=duration,
    )

    return SessionResult(
        exp_id=config.exp_id,
        out_dir=out_dir,
        total_turns=len(state.turns),
        termination_reason=termination_reason,
        total_tokens_in=total_tokens_in,
        total_tokens_out=total_tokens_out,
        duration_seconds=duration,
    )
```

- [ ] **Step 4: Run tests**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_runner_integration.py -v`

Expected: all 4 happy-path tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/runner.py tests/test_runner_integration.py
git commit -m "feat(runner): run_session — Phase 2 turn loop orchestrator

Drives the full pipeline: load personas/setting, init run dir,
loop through turns (speaker alternation, director event trigger,
prompt assembly, LLM call, parse, write turn yaml/md/jsonl),
finalize with meta.yaml. Termination: max_turns only (Phase 2)."
```

---

### Task 9: Runner integration tests — director events + edge cases

**Files:**
- Modify: `tests/test_runner_integration.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_runner_integration.py`:

```python
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
```

- [ ] **Step 2: Run to verify new tests fail or (for some) pass**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_runner_integration.py -v`

Expected: Most of the new tests should PASS because run_session was written correctly in Task 8. If any fail, fix the runner accordingly. The test for `test_llm_exception_aborts_session_partial_turns_kept` might need verification that `meta.yaml` is not written on exception — which it shouldn't be, since the call to `write_meta` is after the loop.

- [ ] **Step 3: Fix any test failures**

If any test fails, inspect the output and adjust `src/empty_space/runner.py`. The most likely area: event ordering (ensure `state.active_events` is appended before `build_system_prompt` is called; already done in Task 8).

- [ ] **Step 4: Run full test suite**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add tests/test_runner_integration.py src/empty_space/runner.py
git commit -m "test(runner): director events, parse errors, termination, exceptions

Covers: event injection at trigger turn, event accumulation,
parse-error graceful handling, max_turns termination,
LLM exception propagation with partial turns kept,
distinct timestamp dirs for repeated runs."
```

---

### Task 10: CLI entry — `scripts/run_experiment.py`

**Files:**
- Create: `scripts/run_experiment.py`

- [ ] **Step 1: Create `scripts/run_experiment.py`**

```python
"""Run a single experiment session.

Usage:
    uv run python scripts/run_experiment.py <exp_id>

Example:
    uv run python scripts/run_experiment.py mother_x_son_hospital_v3_001
"""
import sys

from empty_space.llm import GeminiClient
from empty_space.loaders import load_experiment
from empty_space.runner import run_session


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: run_experiment.py <exp_id>", file=sys.stderr)
        return 2

    exp_id = sys.argv[1]
    config = load_experiment(exp_id)
    client = GeminiClient()

    result = run_session(config=config, llm_client=client)

    print(f"✓ Completed {result.exp_id}")
    print(f"  Output: {result.out_dir}")
    print(f"  Turns: {result.total_turns}")
    print(f"  Termination: {result.termination_reason}")
    print(f"  Tokens in/out: {result.total_tokens_in} / {result.total_tokens_out}")
    print(f"  Duration: {result.duration_seconds:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify script imports cleanly (no syntax / import errors)**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run python -c "import scripts.run_experiment"`

Expected: no output, no traceback. (This confirms the module loads.)

- [ ] **Step 3: Verify usage message on missing argument**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run python scripts/run_experiment.py ; echo "exit=$?"`

Expected: stderr shows `Usage: run_experiment.py <exp_id>`; exit code is 2.

- [ ] **Step 4: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add scripts/run_experiment.py
git commit -m "feat(cli): run_experiment.py CLI entry for Phase 2

Usage: uv run python scripts/run_experiment.py <exp_id>
Loads experiment, creates GeminiClient, runs session, prints summary.
No argparse; single positional arg is enough for Phase 2."
```

---

### Task 11: End-to-end smoke test with real Gemini API

**Files:**
- None created (manual verification only)

This task produces no code artifacts; it validates the pipeline against real Gemini.

- [ ] **Step 1: Verify `.env` has `GEMINI_API_KEY`**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && grep -q "^GEMINI_API_KEY=" .env && echo "key present" || echo "MISSING"`

Expected: `key present`. If missing, stop and tell 柏為 to fill it in.

- [ ] **Step 2: Temporarily reduce max_turns to 4 for a quick smoke run**

Edit `experiments/mother_x_son_hospital_v3_001.yaml` → change `max_turns: 20` to `max_turns: 4`.

- [ ] **Step 3: Run the experiment against real Gemini**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run python scripts/run_experiment.py mother_x_son_hospital_v3_001`

Expected output (something like):

```
✓ Completed mother_x_son_hospital_v3_001
  Output: /Users/chenbaiwei/.../runs/mother_x_son_hospital_v3_001/2026-04-21T12-00-00
  Turns: 4
  Termination: max_turns
  Tokens in/out: ~8000 / ~200
  Duration: ~15-40s
```

- [ ] **Step 4: Inspect the outputs**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && ls runs/mother_x_son_hospital_v3_001/*/`

Expected: see `config.yaml`, `conversation.md`, `conversation.jsonl`, `meta.yaml`, `turns/turn_001.yaml..turn_004.yaml`.

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && cat runs/mother_x_son_hospital_v3_001/*/conversation.md`

Expected: readable two-character dialogue with 母親 / 兒子 labels.

- [ ] **Step 5: Restore max_turns=20**

Edit `experiments/mother_x_son_hospital_v3_001.yaml` → revert `max_turns` to `20`.

- [ ] **Step 6: Manually read one `turn_NNN.yaml`**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && cat runs/mother_x_son_hospital_v3_001/*/turns/turn_001.yaml`

Expected: full schema with prompt_assembled (system + user verbatim), response content, candidate_impressions (may be empty), director_events_active ([]), parse_error (null).

- [ ] **Step 7: Commit the max_turns restoration (if modified on disk)**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git diff experiments/mother_x_son_hospital_v3_001.yaml
# If the only change is max_turns back to 20, no commit needed.
# If the smoke test created runs/ output, that's gitignored.
```

Expected: no changes to commit (the smoke run's max_turns=4 was temporary).

- [ ] **Step 8: Write Phase 2 summary doc**

Create `docs/phase2-summary.md` mirroring `docs/phase1-summary.md`'s structure. Use the smoke run's output (tokens, duration, turn count) as concrete data. Include:

- What was built
- Key decisions (director events over scripted injection; no演出指示 block; no tail anchor; stateless history)
- Module inventory + test count
- Command to run the full experiment
- Known issues / follow-ups (rubric quality risk, etc. — cross-reference spec §8)

Keep under 150 lines; model after Phase 1 summary.

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add docs/phase2-summary.md
git commit -m "docs: Phase 2 summary — runner + prompt assembler shipped

Records smoke-run results and key decisions for future session handoff."
```

---

## Self-Review Complete

**Spec coverage check:**

| Spec section | Implemented in task |
|---|---|
| §1 Scope | Covered by Task 10 (CLI) + Task 11 (smoke test) |
| §2 Schema changes | Task 1 |
| §3 Module skeleton | Tasks 2-10 each create one module |
| §4 Prompt assembly rules (5 blocks, no tail anchor) | Tasks 3, 4 |
| §5 Turn loop state machine | Task 8 |
| §6 Candidate impression parser | Task 2 |
| §7 Disk schema (turn yaml, conversation.md/jsonl, meta.yaml) | Tasks 5, 6, 7 |
| §8 Testing strategy | Distributed across all tasks + Task 9 for runner edge cases |
| §9 CLI entry | Task 10 |
| §10 Completion conditions | Task 11 verification |

**Placeholder scan:** No TBD / TODO / "similar to" / "add appropriate error handling" entries. Every step shows the actual code.

**Type consistency check:**
- `Turn.speaker` is `Literal["protagonist", "counterpart"]` in schemas.py; runner sets it to `"protagonist" if n % 2 == 1 else "counterpart"`. ✓
- `build_system_prompt` signature in Task 3 matches call site in Task 8 runner. ✓
- `build_user_message(history)` in Task 4 (single arg) matches call site in Task 8. ✓
- `append_turn(out_dir, turn)` in Task 6 matches call site in Task 8. ✓
- `write_meta` kwargs in Task 7 match call site in Task 8. ✓
- `GeminiResponse` field `content` used by `parse_response`; confirmed in Phase 1's `llm.py`. ✓
