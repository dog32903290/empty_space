# 空的空間 — Level 2: Ledger + Session-Start Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build cross-session impression ledger + session-start retrieval, enabling演員 to carry impressions from prior sessions into new sessions via director-written `prelude` + scene_premise as the query source.

**Architecture:** Two new function-oriented modules — `ledger.py` (append-only per-speaker YAML files with incremental symbol_index and cooccurrence maintenance) and `retrieval.py` (Flash-based symbol extraction + 1-hop co-occurrence expansion + synonym-canonical matching across two ledgers). Runner orchestrates: session-start retrieval (once per role), injects results into system prompt's new `## 你的內在` block, appends all candidates to ledgers at session end.

**Tech Stack:** Python 3.11+, pydantic v2, pyyaml, pytest (+pytest-mock), google-genai (Phase 1 client reused). No new dependencies.

**Spec reference:** `docs/superpowers/specs/2026-04-21-level-2-ledger-retrieval.md`

---

## File Structure Overview

**Modify:**
- `src/empty_space/schemas.py` — add `protagonist_prelude` + `counterpart_prelude` to `ExperimentConfig`; add `Ledger`, `LedgerEntry`, `RetrievedImpression`, `RetrievalResult` dataclasses
- `src/empty_space/prompt_assembler.py` — extend `build_system_prompt` with `prelude` + `retrieved_impressions` kwargs, emit `## 你的內在` block
- `src/empty_space/writer.py` — add `write_retrieval` function, extend turn yaml with `retrieved_impressions` field, extend meta yaml with `retrieval_total_tokens_*` and `ledger_appends`
- `src/empty_space/runner.py` — extend `SessionState` with retrieval fields, add session-start retrieval block, add session-end ledger append helper
- `tests/test_schemas_experiment.py` — prelude field tests
- `tests/test_loaders_experiment.py` — loader reads prelude fields
- `tests/test_prompt_assembler.py` — `## 你的內在` block tests
- `tests/test_writer.py` — `write_retrieval` tests + turn yaml + meta field tests
- `tests/test_runner_integration.py` — 2 additions (retrieval.yaml exists, turn yaml has retrieved_impressions)

**Create:**
- `src/empty_space/ledger.py` — persistent ledger I/O with incremental index
- `src/empty_space/retrieval.py` — symbol extraction + expansion + canonicalization + top-N
- `config/symbol_synonyms.yaml` — synonym canonical map (empty initial)
- `tests/test_ledger.py` — ~10 tests
- `tests/test_retrieval.py` — ~15 tests
- `tests/test_runner_level2.py` — ~5 cross-session integration tests

---

## Tasks

### Task 1: Schema migration — prelude fields + Level 2 dataclasses

**Files:**
- Modify: `src/empty_space/schemas.py`
- Modify: `tests/test_schemas_experiment.py`
- Modify: `tests/test_loaders_experiment.py`

- [ ] **Step 1: Modify `src/empty_space/schemas.py` — add prelude to ExperimentConfig**

Find the `ExperimentConfig` class. After the existing `scene_premise: str | None = None` line (before `initial_state`), add:

```python
    protagonist_prelude: str | None = None
    counterpart_prelude: str | None = None
```

The full `ExperimentConfig` becomes:

```python
class ExperimentConfig(BaseModel):
    """Top-level config for a single experiment run.

    Phase 2 removes: protagonist_opener, counterpart_system, scripted_turns.
    Phase 2 adds: scene_premise, director_events.
    Level 2 adds: protagonist_prelude, counterpart_prelude.
    Rationale: see spec §2. Director controls the world, not the mouths.
    """
    exp_id: str
    protagonist: PersonaRef
    counterpart: PersonaRef
    setting: SettingRef
    scene_premise: str | None = None
    protagonist_prelude: str | None = None
    counterpart_prelude: str | None = None
    initial_state: InitialState
    director_events: dict[int, str] = Field(default_factory=dict)
    max_turns: int = 20
    termination: Termination = Field(default_factory=Termination)
```

- [ ] **Step 2: Modify `src/empty_space/schemas.py` — add Level 2 dataclasses**

At the end of the file (after existing dataclasses), append:

```python
# --- Level 2 runtime dataclasses ---

@dataclass
class LedgerEntry:
    """One candidate impression persisted in a ledger."""
    id: str                              # imp_001, imp_002, ...
    text: str
    symbols: list[str]
    from_run: str                        # e.g. mother_x_son_hospital_v3_001/2026-04-21T10-24-12
    from_turn: int
    created: str                         # ISO 8601


@dataclass
class Ledger:
    """In-memory representation of a single <relationship>.from_<persona>.yaml."""
    relationship: str
    speaker: Literal["protagonist", "counterpart"]
    persona_name: str
    ledger_version: int
    candidates: list[LedgerEntry]
    symbol_index: dict[str, list[str]]         # symbol → [imp_id, ...]
    cooccurrence: dict[str, dict[str, int]]    # symbol_a → symbol_b → count


@dataclass(frozen=True)
class RetrievedImpression:
    """Read from ledger; what went into the '你的內在' block."""
    id: str
    text: str
    symbols: tuple[str, ...]             # entry.symbols 原樣 (tuple for frozen hashability)
    speaker: Literal["protagonist", "counterpart"]
    persona_name: str
    from_run: str
    from_turn: int
    score: int                           # len(matched_symbols)
    matched_symbols: tuple[str, ...]     # canonical 形式的交集


@dataclass
class RetrievalResult:
    """Session-start retrieval outcome for one role."""
    speaker_role: Literal["protagonist", "counterpart"]
    persona_name: str
    query_text: str                      # scene_premise + prelude joined
    query_symbols: list[str]             # Flash extract 原始輸出
    expanded_symbols: list[str]          # + co-occurrence 鄰居
    impressions: list[RetrievedImpression]
    flash_latency_ms: int
    flash_tokens_in: int
    flash_tokens_out: int
```

Note: `RetrievedImpression` uses `tuple` for `symbols` and `matched_symbols` because `frozen=True` dataclass requires hashable fields. Unlike `CandidateImpression` (which was unfrozen in Phase 2 Task 1 cleanup for the same reason), `RetrievedImpression` really benefits from being hashable (it may end up in sets/dedup keys).

- [ ] **Step 3: Write failing tests in `tests/test_schemas_experiment.py`**

Append these tests at the end of the file:

```python
def test_experiment_accepts_protagonist_prelude():
    config = ExperimentConfig(
        exp_id="x",
        protagonist=PersonaRef(path="p", version="v"),
        counterpart=PersonaRef(path="q", version="v"),
        setting=SettingRef(path="s.yaml"),
        protagonist_prelude="你昨夜夢到他小時候被帶走。",
        initial_state=InitialState(verb="v", stage="s", mode="m"),
    )
    assert config.protagonist_prelude == "你昨夜夢到他小時候被帶走。"


def test_experiment_accepts_counterpart_prelude():
    config = ExperimentConfig(
        exp_id="x",
        protagonist=PersonaRef(path="p", version="v"),
        counterpart=PersonaRef(path="q", version="v"),
        setting=SettingRef(path="s.yaml"),
        counterpart_prelude="你昨晚和女朋友分手。",
        initial_state=InitialState(verb="v", stage="s", mode="m"),
    )
    assert config.counterpart_prelude == "你昨晚和女朋友分手。"


def test_experiment_preludes_default_to_None():
    config = ExperimentConfig(
        exp_id="x",
        protagonist=PersonaRef(path="p", version="v"),
        counterpart=PersonaRef(path="q", version="v"),
        setting=SettingRef(path="s.yaml"),
        initial_state=InitialState(verb="v", stage="s", mode="m"),
    )
    assert config.protagonist_prelude is None
    assert config.counterpart_prelude is None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_schemas_experiment.py -v`

Expected: all tests PASS (new 3 + existing).

- [ ] **Step 5: Run full test suite to confirm no regressions**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest -v`

Expected: all 75 existing tests pass + 3 new = 78 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/schemas.py tests/test_schemas_experiment.py
git commit -m "feat(schemas): Level 2 migration — prelude + ledger/retrieval dataclasses

Adds ExperimentConfig.protagonist_prelude + counterpart_prelude (optional).
Adds Ledger, LedgerEntry, RetrievedImpression, RetrievalResult dataclasses
for cross-session ledger and session-start retrieval."
```

---

### Task 2: `ledger.py` — full module with incremental indices

**Files:**
- Create: `src/empty_space/ledger.py`
- Create: `tests/test_ledger.py`

- [ ] **Step 1: Write failing tests in `tests/test_ledger.py`**

```python
"""Tests for ledger.py — append-only per-speaker impression ledger."""
from pathlib import Path

import pytest
import yaml

from empty_space.schemas import CandidateImpression, Ledger
from empty_space.ledger import (
    append_session_candidates,
    ledger_path,
    read_ledger,
)


@pytest.fixture(autouse=True)
def redirect_ledgers_dir(tmp_path, monkeypatch):
    """Redirect LEDGERS_DIR to a tmp path for all tests."""
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path)


def test_read_ledger_missing_file_returns_empty():
    ledger = read_ledger(relationship="母親_x_兒子", persona_name="母親")
    assert ledger.relationship == "母親_x_兒子"
    assert ledger.persona_name == "母親"
    assert ledger.ledger_version == 0
    assert ledger.candidates == []
    assert ledger.symbol_index == {}
    assert ledger.cooccurrence == {}


def test_ledger_path_uses_from_speaker_convention(tmp_path):
    path = ledger_path(relationship="母親_x_兒子", persona_name="母親")
    assert path.name == "母親_x_兒子.from_母親.yaml"
    assert path.parent == tmp_path


def test_append_creates_file_with_candidates():
    candidates = [
        (1, CandidateImpression(text="消毒水的味道很淡", symbols=["消毒水", "淡"])),
        (1, CandidateImpression(text="她的手沒有動", symbols=["手", "不動"])),
    ]
    append_session_candidates(
        relationship="母親_x_兒子",
        speaker_role="protagonist",
        persona_name="母親",
        candidates=candidates,
        source_run="mother_x_son_hospital_v3_001/2026-04-21T10-24-12",
    )

    ledger = read_ledger(relationship="母親_x_兒子", persona_name="母親")
    assert ledger.ledger_version == 1
    assert len(ledger.candidates) == 2
    assert ledger.candidates[0].id == "imp_001"
    assert ledger.candidates[0].text == "消毒水的味道很淡"
    assert ledger.candidates[0].symbols == ["消毒水", "淡"]
    assert ledger.candidates[0].from_turn == 1
    assert ledger.candidates[0].from_run == "mother_x_son_hospital_v3_001/2026-04-21T10-24-12"
    assert ledger.candidates[1].id == "imp_002"


def test_append_updates_symbol_index():
    candidates = [
        (1, CandidateImpression(text="x", symbols=["A", "B"])),
        (2, CandidateImpression(text="y", symbols=["B", "C"])),
    ]
    append_session_candidates(
        relationship="R",
        speaker_role="protagonist",
        persona_name="母親",
        candidates=candidates,
        source_run="r/t",
    )
    ledger = read_ledger(relationship="R", persona_name="母親")
    assert ledger.symbol_index == {
        "A": ["imp_001"],
        "B": ["imp_001", "imp_002"],
        "C": ["imp_002"],
    }


def test_append_updates_cooccurrence_symmetric():
    candidates = [
        (1, CandidateImpression(text="x", symbols=["A", "B", "C"])),
    ]
    append_session_candidates(
        relationship="R",
        speaker_role="protagonist",
        persona_name="母親",
        candidates=candidates,
        source_run="r/t",
    )
    ledger = read_ledger(relationship="R", persona_name="母親")
    # All pairs: (A,B), (A,C), (B,C). Symmetric.
    assert ledger.cooccurrence["A"] == {"B": 1, "C": 1}
    assert ledger.cooccurrence["B"] == {"A": 1, "C": 1}
    assert ledger.cooccurrence["C"] == {"A": 1, "B": 1}


def test_append_single_symbol_candidate_no_cooccurrence():
    candidates = [
        (1, CandidateImpression(text="x", symbols=["ONLY"])),
    ]
    append_session_candidates(
        relationship="R",
        speaker_role="protagonist",
        persona_name="母親",
        candidates=candidates,
        source_run="r/t",
    )
    ledger = read_ledger(relationship="R", persona_name="母親")
    assert ledger.symbol_index == {"ONLY": ["imp_001"]}
    # No cooccurrence entry for ONLY (arity < 2)
    assert "ONLY" not in ledger.cooccurrence


def test_second_append_accumulates_correctly():
    # First append
    append_session_candidates(
        relationship="R",
        speaker_role="protagonist",
        persona_name="母親",
        candidates=[(1, CandidateImpression(text="first", symbols=["A", "B"]))],
        source_run="r/t1",
    )
    # Second append
    append_session_candidates(
        relationship="R",
        speaker_role="protagonist",
        persona_name="母親",
        candidates=[(2, CandidateImpression(text="second", symbols=["B", "C"]))],
        source_run="r/t2",
    )
    ledger = read_ledger(relationship="R", persona_name="母親")
    assert ledger.ledger_version == 2
    assert len(ledger.candidates) == 2
    assert ledger.candidates[0].id == "imp_001"
    assert ledger.candidates[1].id == "imp_002"
    assert ledger.candidates[1].from_run == "r/t2"
    # Cooccurrence accumulates
    assert ledger.cooccurrence["B"] == {"A": 1, "C": 1}


def test_append_empty_candidates_list_still_increments_version():
    append_session_candidates(
        relationship="R",
        speaker_role="protagonist",
        persona_name="母親",
        candidates=[],
        source_run="r/t",
    )
    ledger = read_ledger(relationship="R", persona_name="母親")
    # Empty list: file created, version = 1, no candidates
    assert ledger.ledger_version == 1
    assert ledger.candidates == []


def test_atomic_write_no_corrupt_on_replace_failure(monkeypatch):
    """If os.replace raises, the final file must not exist (tmp left is OK)."""
    def failing_replace(*args, **kwargs):
        raise OSError("simulated disk failure")
    monkeypatch.setattr("os.replace", failing_replace)

    with pytest.raises(OSError, match="simulated disk failure"):
        append_session_candidates(
            relationship="R",
            speaker_role="protagonist",
            persona_name="母親",
            candidates=[(1, CandidateImpression(text="x", symbols=["A"]))],
            source_run="r/t",
        )

    # Final file should not exist; tmp may exist but is not the canonical
    final = ledger_path(relationship="R", persona_name="母親")
    assert not final.exists()


def test_schema_round_trip_preserves_everything():
    """Write a ledger via append, read it back, all fields match."""
    candidates = [
        (5, CandidateImpression(text="原始文字", symbols=["符號1", "符號2"])),
    ]
    append_session_candidates(
        relationship="關係",
        speaker_role="counterpart",
        persona_name="兒子",
        candidates=candidates,
        source_run="exp_x/2026-04-21T12-00-00",
    )
    ledger = read_ledger(relationship="關係", persona_name="兒子")
    entry = ledger.candidates[0]
    assert entry.id == "imp_001"
    assert entry.text == "原始文字"
    assert entry.symbols == ["符號1", "符號2"]
    assert entry.from_run == "exp_x/2026-04-21T12-00-00"
    assert entry.from_turn == 5
    # created is ISO 8601 — just check it exists and looks like ISO
    assert entry.created  # non-empty
    assert "T" in entry.created
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_ledger.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'empty_space.ledger'`.

- [ ] **Step 3: Create `src/empty_space/ledger.py`**

```python
"""Cross-session impression ledger — append-only, per speaker.

Files land at ledgers/<relationship>.from_<persona_name>.yaml.
Maintains symbol_index (reverse lookup) and cooccurrence (1-hop graph edges)
incrementally on each append.

Atomic write via .tmp + os.replace.
"""
import os
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import yaml

from empty_space.paths import LEDGERS_DIR
from empty_space.schemas import (
    CandidateImpression,
    Ledger,
    LedgerEntry,
)


def ledger_path(*, relationship: str, persona_name: str) -> Path:
    """Returns <LEDGERS_DIR>/<relationship>.from_<persona_name>.yaml"""
    return LEDGERS_DIR / f"{relationship}.from_{persona_name}.yaml"


def read_ledger(*, relationship: str, persona_name: str) -> Ledger:
    """Read ledger file; if absent, return empty Ledger (do not raise).

    Note: when the file is absent, speaker is set to 'protagonist' as a
    placeholder. Callers should not rely on the .speaker field of an empty
    ledger since the speaker_role isn't knowable from the file path alone.
    """
    path = ledger_path(relationship=relationship, persona_name=persona_name)
    if not path.exists():
        return Ledger(
            relationship=relationship,
            speaker="protagonist",  # placeholder; overwritten on first append
            persona_name=persona_name,
            ledger_version=0,
            candidates=[],
            symbol_index={},
            cooccurrence={},
        )

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Ledger(
        relationship=data["relationship"],
        speaker=data["speaker"],
        persona_name=data["persona_name"],
        ledger_version=data["ledger_version"],
        candidates=[
            LedgerEntry(
                id=c["id"],
                text=c["text"],
                symbols=list(c["symbols"]),
                from_run=c["from_run"],
                from_turn=c["from_turn"],
                created=c["created"],
            )
            for c in (data.get("candidates") or [])
        ],
        symbol_index={k: list(v) for k, v in (data.get("symbol_index") or {}).items()},
        cooccurrence={
            k: dict(v) for k, v in (data.get("cooccurrence") or {}).items()
        },
    )


def append_session_candidates(
    *,
    relationship: str,
    speaker_role: str,
    persona_name: str,
    candidates: list[tuple[int, CandidateImpression]],
    source_run: str,
) -> None:
    """Append one session's worth of candidates to a ledger. Atomic write.

    candidates: list of (turn_number, CandidateImpression) tuples.
    Updates symbol_index (reverse) and cooccurrence (symmetric pair counts).
    Increments ledger_version.

    Empty candidates list still creates/updates the file and bumps version.
    """
    # Read existing (may be empty)
    existing = read_ledger(relationship=relationship, persona_name=persona_name)

    # Determine next id
    next_id_num = len(existing.candidates) + 1
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Build new entries and merge
    new_entries: list[LedgerEntry] = []
    for turn_number, imp in candidates:
        entry = LedgerEntry(
            id=f"imp_{next_id_num:03d}",
            text=imp.text,
            symbols=list(imp.symbols),
            from_run=source_run,
            from_turn=turn_number,
            created=now_iso,
        )
        new_entries.append(entry)
        next_id_num += 1

    all_candidates = existing.candidates + new_entries

    # Update symbol_index incrementally
    symbol_index = {k: list(v) for k, v in existing.symbol_index.items()}
    for entry in new_entries:
        for sym in entry.symbols:
            symbol_index.setdefault(sym, []).append(entry.id)

    # Update cooccurrence (symmetric)
    cooccurrence = {k: dict(v) for k, v in existing.cooccurrence.items()}
    for entry in new_entries:
        for sym_a, sym_b in combinations(entry.symbols, 2):
            cooccurrence.setdefault(sym_a, {})
            cooccurrence[sym_a][sym_b] = cooccurrence[sym_a].get(sym_b, 0) + 1
            cooccurrence.setdefault(sym_b, {})
            cooccurrence[sym_b][sym_a] = cooccurrence[sym_b].get(sym_a, 0) + 1

    # Construct final Ledger
    new_ledger = Ledger(
        relationship=relationship,
        speaker=speaker_role,
        persona_name=persona_name,
        ledger_version=existing.ledger_version + 1,
        candidates=all_candidates,
        symbol_index=symbol_index,
        cooccurrence=cooccurrence,
    )

    _atomic_write_ledger(new_ledger)


def _atomic_write_ledger(ledger: Ledger) -> None:
    """Serialize ledger to YAML via .tmp + os.replace."""
    path = ledger_path(relationship=ledger.relationship, persona_name=ledger.persona_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "relationship": ledger.relationship,
        "speaker": ledger.speaker,
        "persona_name": ledger.persona_name,
        "ledger_version": ledger.ledger_version,
        "candidates": [
            {
                "id": e.id,
                "text": e.text,
                "symbols": list(e.symbols),
                "from_run": e.from_run,
                "from_turn": e.from_turn,
                "created": e.created,
            }
            for e in ledger.candidates
        ],
        "symbol_index": ledger.symbol_index,
        "cooccurrence": ledger.cooccurrence,
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_ledger.py -v`

Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/ledger.py tests/test_ledger.py
git commit -m "feat(ledger): append-only per-speaker impression ledger

Two-file-per-relationship layout (<A>_x_<B>.from_<A>.yaml + from_<B>).
Maintains symbol_index (reverse lookup) and cooccurrence (symmetric 1-hop
graph edges) incrementally on each append. Atomic write via .tmp + os.replace.
Empty-list append still bumps ledger_version."
```

---

### Task 3: `retrieval.py` — pure utilities (canonicalize + synonym + expand + merge)

**Files:**
- Create: `src/empty_space/retrieval.py` (partial — this task)
- Create: `tests/test_retrieval.py` (partial — this task)

- [ ] **Step 1: Write failing tests in `tests/test_retrieval.py`**

```python
"""Tests for retrieval.py — symbol extraction, canonicalization, expansion, scoring."""
from pathlib import Path

import pytest
import yaml

from empty_space.retrieval import (
    canonicalize,
    expand_with_cooccurrence,
    load_synonym_map,
    merge_cooccurrence,
)


# --- canonicalize ---

def test_canonicalize_returns_self_when_not_in_map():
    assert canonicalize("愧疚", {}) == "愧疚"


def test_canonicalize_returns_canonical_when_mapped():
    synonym_map = {"愧疚感": "愧疚", "罪惡感": "愧疚"}
    assert canonicalize("愧疚感", synonym_map) == "愧疚"
    assert canonicalize("罪惡感", synonym_map) == "愧疚"


def test_canonicalize_canonical_points_to_itself():
    synonym_map = {"愧疚": "愧疚", "愧疚感": "愧疚"}
    assert canonicalize("愧疚", synonym_map) == "愧疚"


# --- load_synonym_map ---

def test_load_synonym_map_missing_file_returns_empty(tmp_path):
    result = load_synonym_map(tmp_path / "nonexistent.yaml")
    assert result == {}


def test_load_synonym_map_empty_groups_returns_empty(tmp_path):
    path = tmp_path / "syn.yaml"
    path.write_text("groups: []\n", encoding="utf-8")
    result = load_synonym_map(path)
    assert result == {}


def test_load_synonym_map_expands_groups_correctly(tmp_path):
    path = tmp_path / "syn.yaml"
    path.write_text(
        "groups:\n"
        "  - [愧疚, 愧疚感, 罪惡感]\n"
        "  - [沉默, 不說話]\n",
        encoding="utf-8",
    )
    result = load_synonym_map(path)
    assert result == {
        "愧疚": "愧疚",
        "愧疚感": "愧疚",
        "罪惡感": "愧疚",
        "沉默": "沉默",
        "不說話": "沉默",
    }


def test_load_synonym_map_default_path_uses_config_dir(tmp_path, monkeypatch):
    """When no path arg given, default resolves to config/symbol_synonyms.yaml."""
    # Create a fake project root
    project_root = tmp_path / "proj"
    (project_root / "config").mkdir(parents=True)
    (project_root / "config" / "symbol_synonyms.yaml").write_text(
        "groups:\n  - [A, B]\n", encoding="utf-8",
    )
    monkeypatch.setattr("empty_space.retrieval.DEFAULT_SYNONYMS_PATH",
                        project_root / "config" / "symbol_synonyms.yaml")
    result = load_synonym_map()
    assert result == {"A": "A", "B": "A"}


# --- expand_with_cooccurrence ---

def test_expand_empty_cooccurrence_returns_seeds():
    result = expand_with_cooccurrence(
        seed_symbols=["A", "B"],
        cooccurrence={},
        top_neighbors_per_seed=2,
    )
    assert result == ["A", "B"]


def test_expand_seed_without_neighbors_no_addition():
    cooc = {"B": {"X": 1}}  # A has no neighbors
    result = expand_with_cooccurrence(
        seed_symbols=["A"],
        cooccurrence=cooc,
        top_neighbors_per_seed=2,
    )
    assert result == ["A"]


def test_expand_takes_top_k_by_count():
    cooc = {
        "A": {"X": 5, "Y": 3, "Z": 1},
    }
    result = expand_with_cooccurrence(
        seed_symbols=["A"],
        cooccurrence=cooc,
        top_neighbors_per_seed=2,
    )
    assert result == ["A", "X", "Y"]  # Z excluded


def test_expand_tiebreak_alphabetical():
    cooc = {
        "A": {"Y": 2, "X": 2},  # Same count
    }
    result = expand_with_cooccurrence(
        seed_symbols=["A"],
        cooccurrence=cooc,
        top_neighbors_per_seed=2,
    )
    assert result == ["A", "X", "Y"]  # Alphabetical tiebreak


def test_expand_dedups_across_seeds():
    cooc = {
        "A": {"X": 3},
        "B": {"X": 2},
    }
    result = expand_with_cooccurrence(
        seed_symbols=["A", "B"],
        cooccurrence=cooc,
        top_neighbors_per_seed=1,
    )
    assert result == ["A", "B", "X"]  # X only once


def test_expand_preserves_seed_order():
    cooc = {"A": {"X": 1}}
    result = expand_with_cooccurrence(
        seed_symbols=["Z", "A", "M"],
        cooccurrence=cooc,
        top_neighbors_per_seed=1,
    )
    assert result[:3] == ["Z", "A", "M"]  # seeds first
    assert "X" in result[3:]


# --- merge_cooccurrence ---

def test_merge_cooccurrence_disjoint():
    a = {"X": {"Y": 1}}
    b = {"Z": {"W": 2}}
    result = merge_cooccurrence(a, b)
    assert result == {"X": {"Y": 1}, "Z": {"W": 2}}


def test_merge_cooccurrence_overlapping_keys_sums_counts():
    a = {"X": {"Y": 1, "Z": 2}}
    b = {"X": {"Y": 3, "W": 1}}
    result = merge_cooccurrence(a, b)
    assert result == {"X": {"Y": 4, "Z": 2, "W": 1}}


def test_merge_cooccurrence_does_not_mutate_inputs():
    a = {"X": {"Y": 1}}
    b = {"X": {"Y": 2}}
    result = merge_cooccurrence(a, b)
    assert a == {"X": {"Y": 1}}  # unchanged
    assert b == {"X": {"Y": 2}}  # unchanged
    assert result["X"]["Y"] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_retrieval.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'empty_space.retrieval'`.

- [ ] **Step 3: Create `src/empty_space/retrieval.py` with pure utilities**

```python
"""Session-start retrieval: extract symbols → expand via co-occurrence →
score candidates in both ledgers → return top-3 per role.
"""
from pathlib import Path

import yaml

from empty_space.paths import PROJECT_ROOT

DEFAULT_SYNONYMS_PATH = PROJECT_ROOT / "config" / "symbol_synonyms.yaml"


# --- canonicalization ---

def canonicalize(symbol: str, synonym_map: dict[str, str]) -> str:
    """Return synonym_map[symbol] if mapped, else symbol."""
    return synonym_map.get(symbol, symbol)


def load_synonym_map(path: Path | None = None) -> dict[str, str]:
    """Load config/symbol_synonyms.yaml. Returns symbol→canonical dict.

    If file missing, returns {}.
    If file present but groups is empty or absent, returns {}.
    Within each group, the first element is canonical; all elements map to it.
    """
    if path is None:
        path = DEFAULT_SYNONYMS_PATH
    if not Path(path).exists():
        return {}

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    groups = data.get("groups") or []

    result: dict[str, str] = {}
    for group in groups:
        if not group:
            continue
        canonical = group[0]
        for sym in group:
            result[sym] = canonical
    return result


# --- co-occurrence expansion ---

def expand_with_cooccurrence(
    *,
    seed_symbols: list[str],
    cooccurrence: dict[str, dict[str, int]],
    top_neighbors_per_seed: int = 2,
) -> list[str]:
    """For each seed, add its top-K most-cooccurring neighbors.

    Preserve seed order; append neighbors after seeds. Dedup globally.
    Tiebreak: count desc, then alphabetical asc.
    """
    seen = set(seed_symbols)
    result = list(seed_symbols)
    for seed in seed_symbols:
        neighbors = cooccurrence.get(seed, {})
        # Sort by count desc, then key asc
        sorted_neighbors = sorted(
            neighbors.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )[:top_neighbors_per_seed]
        for sym, _ in sorted_neighbors:
            if sym not in seen:
                result.append(sym)
                seen.add(sym)
    return result


def merge_cooccurrence(
    a: dict[str, dict[str, int]],
    b: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    """Sum two cooccurrence maps. Does not mutate inputs."""
    result = {k: dict(v) for k, v in a.items()}
    for sym_a, neighbors in b.items():
        if sym_a not in result:
            result[sym_a] = dict(neighbors)
        else:
            for sym_b, count in neighbors.items():
                result[sym_a][sym_b] = result[sym_a].get(sym_b, 0) + count
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_retrieval.py -v`

Expected: all ~15 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/retrieval.py tests/test_retrieval.py
git commit -m "feat(retrieval): pure utilities — canonicalize + synonym loader + expand

canonicalize: symbol→canonical via synonym dict (passthrough if unmapped).
load_synonym_map: parse config/symbol_synonyms.yaml, flatten groups to map.
expand_with_cooccurrence: seed symbols + top-K 1-hop neighbors (per seed).
merge_cooccurrence: sum two cooccurrence maps without mutation."
```

---

### Task 4: `retrieval.py` — extract_symbols (Flash call with mock)

**Files:**
- Modify: `src/empty_space/retrieval.py` (add function)
- Modify: `tests/test_retrieval.py` (add tests)

- [ ] **Step 1: Add failing tests to `tests/test_retrieval.py`**

Append to the end of the test file:

```python
# --- extract_symbols ---

from empty_space.llm import GeminiResponse
from empty_space.retrieval import extract_symbols


class _MockLLMClient:
    """Minimal mock for Flash extract_symbols tests."""
    def __init__(self, response_text: str, tokens_in: int = 100, tokens_out: int = 20, latency_ms: int = 150):
        self.response_text = response_text
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.latency_ms = latency_ms
        self.calls = []

    def generate(self, *, system: str, user: str, model: str = "gemini-2.5-flash") -> GeminiResponse:
        self.calls.append({"system": system, "user": user, "model": model})
        return GeminiResponse(
            content=self.response_text,
            raw=None,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            model=model,
            latency_ms=self.latency_ms,
        )


def test_extract_symbols_parses_clean_yaml_list():
    client = _MockLLMClient(response_text="- 分手\n- 女朋友\n- 拒絕\n")
    symbols, tokens_in, tokens_out, latency = extract_symbols(
        text="你昨晚和女朋友分手。",
        llm_client=client,
    )
    assert symbols == ["分手", "女朋友", "拒絕"]
    assert tokens_in == 100
    assert tokens_out == 20
    assert latency == 150
    assert len(client.calls) == 1


def test_extract_symbols_empty_text_skips_llm_call():
    client = _MockLLMClient(response_text="- x\n")
    symbols, tokens_in, tokens_out, latency = extract_symbols(
        text="",
        llm_client=client,
    )
    assert symbols == []
    assert tokens_in == 0
    assert tokens_out == 0
    assert latency == 0
    assert client.calls == []  # LLM was not called


def test_extract_symbols_whitespace_only_text_skips_llm():
    client = _MockLLMClient(response_text="- x\n")
    symbols, *_ = extract_symbols(
        text="   \n\n  ",
        llm_client=client,
    )
    assert symbols == []
    assert client.calls == []


def test_extract_symbols_invalid_yaml_returns_empty():
    client = _MockLLMClient(response_text="- unclosed [\n  bad")
    symbols, _, _, _ = extract_symbols(
        text="something",
        llm_client=client,
    )
    assert symbols == []


def test_extract_symbols_non_list_yaml_returns_empty():
    client = _MockLLMClient(response_text="key: value\n")
    symbols, _, _, _ = extract_symbols(
        text="something",
        llm_client=client,
    )
    assert symbols == []


def test_extract_symbols_strips_whitespace_and_filters_empty():
    client = _MockLLMClient(response_text="-   愧疚  \n- \n- 沉默\n")
    symbols, _, _, _ = extract_symbols(
        text="something",
        llm_client=client,
    )
    assert symbols == ["愧疚", "沉默"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_retrieval.py -v -k "extract_symbols"`

Expected: FAIL with `ImportError: cannot import name 'extract_symbols' from 'empty_space.retrieval'`.

- [ ] **Step 3: Add `extract_symbols` to `src/empty_space/retrieval.py`**

Append to the bottom of `retrieval.py`:

```python
# --- Flash-based symbol extraction ---

_EXTRACT_SYMBOLS_SYSTEM_PROMPT = """\
你負責從一段中文敘述中提取「感受符號」——能作為記憶檢索 key 的關鍵詞。
規則：
- 每個 symbol 是一個具體的名詞、動詞或感官詞
- 不要抽象名詞（「痛苦」「關係」這類太大）
- 不要連詞、助詞、時間副詞
- 偏好單字或兩字詞，不用長片語
- 輸出 3-10 個 symbols，YAML list 格式
- 只輸出 YAML，不加解釋

範例：
輸入：
「你昨晚和女朋友分手。她說『我不能等你。』」

輸出：
- 分手
- 女朋友
- 等
- 拒絕
"""


def extract_symbols(
    *,
    text: str,
    llm_client,
) -> tuple[list[str], int, int, int]:
    """Ask Flash to extract semantic symbols from a query text.

    Returns:
        (symbols, tokens_in, tokens_out, latency_ms)

    If text is empty/whitespace, skips the LLM call and returns empty.
    On YAML parse failure or non-list response, returns empty symbols list
    (tokens and latency still report the actual call).
    """
    if not text or not text.strip():
        return [], 0, 0, 0

    resp = llm_client.generate(
        system=_EXTRACT_SYMBOLS_SYSTEM_PROMPT,
        user=text,
        model="gemini-2.5-flash",
    )

    try:
        parsed = yaml.safe_load(resp.content)
    except yaml.YAMLError:
        return [], resp.tokens_in, resp.tokens_out, resp.latency_ms

    if not isinstance(parsed, list):
        return [], resp.tokens_in, resp.tokens_out, resp.latency_ms

    symbols = [str(s).strip() for s in parsed if str(s).strip()]
    return symbols, resp.tokens_in, resp.tokens_out, resp.latency_ms
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_retrieval.py -v`

Expected: all tests from this task + Task 3 pass.

- [ ] **Step 5: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/retrieval.py tests/test_retrieval.py
git commit -m "feat(retrieval): extract_symbols — Flash-based symbol extraction

Calls Gemini Flash with a constrained prompt to extract 3-10 感受符號
from a query text. Skips LLM call on empty input. Graceful degradation
on YAML parse failure or non-list response (returns [] but reports
actual tokens/latency)."
```

---

### Task 5: `retrieval.py` — retrieve_top_n + run_session_start_retrieval

**Files:**
- Modify: `src/empty_space/retrieval.py` (add functions)
- Modify: `tests/test_retrieval.py` (add tests)

- [ ] **Step 1: Add failing tests to `tests/test_retrieval.py`**

Append to the end:

```python
# --- retrieve_top_n ---

from empty_space.schemas import Ledger, LedgerEntry
from empty_space.retrieval import retrieve_top_n, run_session_start_retrieval


def _make_ledger(
    *,
    speaker: str = "protagonist",
    persona_name: str = "母親",
    entries: list[LedgerEntry] | None = None,
    cooccurrence: dict | None = None,
) -> Ledger:
    entries = entries or []
    cooc = cooccurrence or {}
    # Build symbol_index from entries
    symbol_index: dict[str, list[str]] = {}
    for e in entries:
        for s in e.symbols:
            symbol_index.setdefault(s, []).append(e.id)
    return Ledger(
        relationship="R",
        speaker=speaker,
        persona_name=persona_name,
        ledger_version=len(entries),
        candidates=entries,
        symbol_index=symbol_index,
        cooccurrence=cooc,
    )


def _make_entry(id: str, text: str, symbols: list[str], created: str = "2026-04-21T10:00:00Z") -> LedgerEntry:
    return LedgerEntry(
        id=id, text=text, symbols=symbols,
        from_run="r/t", from_turn=1, created=created,
    )


def test_retrieve_empty_ledgers_returns_empty():
    result = retrieve_top_n(
        query_symbols=["A", "B"],
        ledger_a=_make_ledger(),
        ledger_b=_make_ledger(persona_name="兒子", speaker="counterpart"),
        synonym_map={},
        top_n=3,
    )
    assert result == []


def test_retrieve_single_ledger_hit():
    e1 = _make_entry("imp_001", "text 1", ["A", "B"])
    e2 = _make_entry("imp_002", "text 2", ["C"])
    ledger_a = _make_ledger(entries=[e1, e2])
    ledger_b = _make_ledger(persona_name="兒子", speaker="counterpart")

    result = retrieve_top_n(
        query_symbols=["A"],
        ledger_a=ledger_a,
        ledger_b=ledger_b,
        synonym_map={},
        top_n=3,
    )
    assert len(result) == 1
    assert result[0].id == "imp_001"
    assert result[0].score == 1
    assert result[0].matched_symbols == ("A",)
    assert result[0].speaker == "protagonist"
    assert result[0].persona_name == "母親"


def test_retrieve_cross_ledger_dedup_by_speaker_and_id():
    # Same id in both ledgers; dedup by (speaker, id)
    e_a = _make_entry("imp_001", "a text", ["A"])
    e_b = _make_entry("imp_001", "b text", ["A"])
    ledger_a = _make_ledger(entries=[e_a])
    ledger_b = _make_ledger(persona_name="兒子", speaker="counterpart", entries=[e_b])

    result = retrieve_top_n(
        query_symbols=["A"],
        ledger_a=ledger_a,
        ledger_b=ledger_b,
        synonym_map={},
        top_n=5,
    )
    # Both kept because (speaker, id) differs
    assert len(result) == 2
    speakers = {r.speaker for r in result}
    assert speakers == {"protagonist", "counterpart"}


def test_retrieve_sorts_by_score_desc():
    e_high = _make_entry("imp_001", "high", ["A", "B", "C"])
    e_low = _make_entry("imp_002", "low", ["A"])
    ledger_a = _make_ledger(entries=[e_high, e_low])
    ledger_b = _make_ledger(persona_name="兒子", speaker="counterpart")

    result = retrieve_top_n(
        query_symbols=["A", "B", "C"],
        ledger_a=ledger_a,
        ledger_b=ledger_b,
        synonym_map={},
        top_n=3,
    )
    assert [r.id for r in result] == ["imp_001", "imp_002"]
    assert result[0].score == 3
    assert result[1].score == 1


def test_retrieve_tiebreak_by_created_desc():
    e_old = _make_entry("imp_001", "old", ["A"], created="2026-04-21T10:00:00Z")
    e_new = _make_entry("imp_002", "new", ["A"], created="2026-04-22T10:00:00Z")
    ledger_a = _make_ledger(entries=[e_old, e_new])
    ledger_b = _make_ledger(persona_name="兒子", speaker="counterpart")

    result = retrieve_top_n(
        query_symbols=["A"],
        ledger_a=ledger_a,
        ledger_b=ledger_b,
        synonym_map={},
        top_n=3,
    )
    # Same score, newer first
    assert result[0].id == "imp_002"
    assert result[1].id == "imp_001"


def test_retrieve_synonym_map_matches_variants():
    synonym_map = {"愧疚": "愧疚", "愧疚感": "愧疚"}
    e = _make_entry("imp_001", "mother's regret", ["愧疚感"])
    ledger_a = _make_ledger(entries=[e])
    ledger_b = _make_ledger(persona_name="兒子", speaker="counterpart")

    result = retrieve_top_n(
        query_symbols=["愧疚"],
        ledger_a=ledger_a,
        ledger_b=ledger_b,
        synonym_map=synonym_map,
        top_n=3,
    )
    assert len(result) == 1
    assert result[0].matched_symbols == ("愧疚",)


def test_retrieve_top_n_truncates():
    entries = [_make_entry(f"imp_{i:03d}", f"t{i}", ["A"]) for i in range(1, 6)]
    ledger_a = _make_ledger(entries=entries)
    ledger_b = _make_ledger(persona_name="兒子", speaker="counterpart")

    result = retrieve_top_n(
        query_symbols=["A"],
        ledger_a=ledger_a,
        ledger_b=ledger_b,
        synonym_map={},
        top_n=3,
    )
    assert len(result) == 3


# --- run_session_start_retrieval ---

def test_run_session_start_retrieval_empty_query_returns_empty(tmp_path, monkeypatch):
    # Redirect LEDGERS_DIR
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path)

    client = _MockLLMClient(response_text="- x\n")
    result = run_session_start_retrieval(
        speaker_role="protagonist",
        persona_name="母親",
        query_text="",
        relationship="母親_x_兒子",
        other_persona_name="兒子",
        synonym_map={},
        llm_client=client,
        top_n=3,
    )
    assert result.query_symbols == []
    assert result.expanded_symbols == []
    assert result.impressions == []
    assert result.flash_tokens_in == 0
    assert client.calls == []


def test_run_session_start_retrieval_full_flow(tmp_path, monkeypatch):
    # Pre-populate a ledger via ledger.append_session_candidates
    from empty_space.ledger import append_session_candidates
    from empty_space.schemas import CandidateImpression

    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path)

    append_session_candidates(
        relationship="母親_x_兒子",
        speaker_role="counterpart",
        persona_name="兒子",
        candidates=[
            (8, CandidateImpression(
                text="她的沉默在這一刻比任何辯解都沉",
                symbols=["沉默", "辯解", "愧疚"],
            )),
        ],
        source_run="prev_exp/2026-04-21T09-00-00",
    )

    # Flash returns "愧疚" among extracted symbols
    client = _MockLLMClient(response_text="- 愧疚\n- 母愛\n")
    result = run_session_start_retrieval(
        speaker_role="protagonist",
        persona_name="母親",
        query_text="你昨夜夢到他小時候被帶走。",
        relationship="母親_x_兒子",
        other_persona_name="兒子",
        synonym_map={},
        llm_client=client,
        top_n=3,
    )

    assert result.query_symbols == ["愧疚", "母愛"]
    assert "愧疚" in result.expanded_symbols
    assert len(result.impressions) == 1
    assert result.impressions[0].text == "她的沉默在這一刻比任何辯解都沉"
    assert result.impressions[0].matched_symbols == ("愧疚",)
    assert result.impressions[0].speaker == "counterpart"
    assert result.flash_tokens_in > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_retrieval.py -v -k "retrieve or run_session_start"`

Expected: FAIL with `ImportError: cannot import name 'retrieve_top_n' from 'empty_space.retrieval'`.

- [ ] **Step 3: Add `retrieve_top_n` and `run_session_start_retrieval` to `src/empty_space/retrieval.py`**

Append to the bottom:

```python
# --- scoring & top-N retrieval ---

from empty_space.schemas import (
    Ledger,
    LedgerEntry,
    RetrievedImpression,
    RetrievalResult,
)
from empty_space.ledger import read_ledger


def retrieve_top_n(
    *,
    query_symbols: list[str],
    ledger_a: Ledger,
    ledger_b: Ledger,
    synonym_map: dict[str, str],
    top_n: int = 3,
) -> list[RetrievedImpression]:
    """Score candidates in both ledgers via symbol hit count under canonical
    equivalence. Return top N by (score desc, created desc). Dedup by
    (speaker, id).
    """
    canon_q = {canonicalize(s, synonym_map) for s in query_symbols}
    if not canon_q:
        return []

    # Score every entry in both ledgers that has at least one match
    scored: list[tuple[int, str, LedgerEntry, Ledger, list[str]]] = []
    # (score, created_desc_key, entry, ledger, matched_canonicals_sorted)
    for ledger in (ledger_a, ledger_b):
        for entry in ledger.candidates:
            canon_e = {canonicalize(s, synonym_map) for s in entry.symbols}
            matched = canon_q & canon_e
            if matched:
                scored.append((
                    len(matched),
                    entry.created,  # ISO 8601 string sorts lexicographically
                    entry,
                    ledger,
                    sorted(matched),
                ))

    # Sort: score desc, then created desc (later = more recent = lexicographically larger)
    scored.sort(key=lambda t: (-t[0], t[1]), reverse=False)
    # The above sorts by (-score asc, created asc). We want score desc + created desc.
    # Re-sort cleanly:
    scored.sort(key=lambda t: (-t[0], _neg_iso(t[1])))

    # Dedup by (speaker, id)
    seen_keys: set[tuple[str, str]] = set()
    result: list[RetrievedImpression] = []
    for score, _, entry, ledger, matched in scored:
        key = (ledger.speaker, entry.id)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        result.append(RetrievedImpression(
            id=entry.id,
            text=entry.text,
            symbols=tuple(entry.symbols),
            speaker=ledger.speaker,
            persona_name=ledger.persona_name,
            from_run=entry.from_run,
            from_turn=entry.from_turn,
            score=score,
            matched_symbols=tuple(matched),
        ))
        if len(result) >= top_n:
            break

    return result


def _neg_iso(iso: str) -> str:
    """Trick to sort ISO 8601 strings in descending order inside tuple-key sort.

    Reverses the string so that later ISO times sort first.
    (tuple sort is lexicographic on this reversed string.)
    """
    # Simpler: just use negative-of-sortable. Since we use sort with key,
    # and ISO strings sort lexicographically, we return a string that
    # inverts the order. Easiest: sort twice — done above. Use max-string trick:
    return chr(0x10FFFF) * (40 - len(iso)) + iso[::-1]  # heuristic; ISO fits in 40 chars


# --- session-start orchestrator ---

def run_session_start_retrieval(
    *,
    speaker_role: str,
    persona_name: str,
    query_text: str,
    relationship: str,
    other_persona_name: str,
    synonym_map: dict[str, str],
    llm_client,
    top_n: int = 3,
) -> RetrievalResult:
    """Full session-start retrieval pipeline for one role.

    1. Flash extract symbols from query_text.
    2. Load both ledgers (self + other).
    3. Expand with merged cooccurrence (1-hop).
    4. Score candidates in both ledgers, return top N.
    5. Package as RetrievalResult (with debug info).
    """
    # Step 1: extract
    query_symbols, tokens_in, tokens_out, latency_ms = extract_symbols(
        text=query_text, llm_client=llm_client,
    )

    # Step 2: load ledgers
    ledger_self = read_ledger(relationship=relationship, persona_name=persona_name)
    ledger_other = read_ledger(relationship=relationship, persona_name=other_persona_name)

    # Set correct speaker on self (read_ledger uses placeholder when file missing)
    # Both ledgers' `speaker` must reflect whose ledger they are — set from args.
    ledger_self = _with_speaker(ledger_self, speaker_role)
    other_role = "counterpart" if speaker_role == "protagonist" else "protagonist"
    ledger_other = _with_speaker(ledger_other, other_role)

    # Step 3: expand
    merged_cooc = merge_cooccurrence(ledger_self.cooccurrence, ledger_other.cooccurrence)
    expanded_symbols = expand_with_cooccurrence(
        seed_symbols=query_symbols,
        cooccurrence=merged_cooc,
        top_neighbors_per_seed=2,
    )

    # Step 4: retrieve top N
    impressions = retrieve_top_n(
        query_symbols=expanded_symbols,
        ledger_a=ledger_self,
        ledger_b=ledger_other,
        synonym_map=synonym_map,
        top_n=top_n,
    )

    return RetrievalResult(
        speaker_role=speaker_role,
        persona_name=persona_name,
        query_text=query_text,
        query_symbols=query_symbols,
        expanded_symbols=expanded_symbols,
        impressions=impressions,
        flash_latency_ms=latency_ms,
        flash_tokens_in=tokens_in,
        flash_tokens_out=tokens_out,
    )


def _with_speaker(ledger: Ledger, speaker: str) -> Ledger:
    """Return a new Ledger with .speaker overridden (for empty ledgers read
    from missing files). No-op if already correct."""
    if ledger.speaker == speaker:
        return ledger
    return Ledger(
        relationship=ledger.relationship,
        speaker=speaker,
        persona_name=ledger.persona_name,
        ledger_version=ledger.ledger_version,
        candidates=ledger.candidates,
        symbol_index=ledger.symbol_index,
        cooccurrence=ledger.cooccurrence,
    )
```

Note: the `_neg_iso` trick for tuple-key sort is hacky. A cleaner implementation (use a single `sort()` with custom key, no double-sort, no reverse trick):

Actually, replace the scoring/sorting block inside `retrieve_top_n` with this cleaner version:

```python
    # Sort: score desc, then created desc
    # Tuple (score, created) sorts naturally; we want both desc:
    # Use negative score + reverse=True for created comparison via tuple tricks
    def sort_key(t):
        score, created, *_ = t
        return (-score, _sortkey_iso_desc(created))
    scored.sort(key=sort_key)
```

With:

```python
def _sortkey_iso_desc(iso: str) -> str:
    """Return a string that sorts reverse of `iso` lexicographically.

    ISO 8601 strings already sort lexicographically = chronologically.
    To sort desc (newer first), use the reverse-lexicographic key.
    Trick: all ISO 8601 strings use chars in [0-9T:-Z], so we can xor them.
    Simpler: use the sort key '~' + iso, no wait -- 

    Pragmatic: pad with a fixed high-char so shorter ISOs sort as expected,
    then invert each char against '~' (0x7E).
    """
    high = "~" * max(0, 32 - len(iso))
    # Invert each character within the printable range so larger chars sort smaller
    inverted = "".join(chr(max(0, 126 - ord(c))) for c in iso)
    return high + inverted
```

Actually, drop the fancy math. **The simplest correct approach**: sort twice (Python's sort is stable), with the minor key first, then the major key:

Replace the scored.sort blocks with:

```python
    # Stable sort: first by tiebreaker (created desc), then by primary (score desc)
    scored.sort(key=lambda t: t[1], reverse=True)  # created desc
    scored.sort(key=lambda t: t[0], reverse=True)  # score desc (stable: ties keep created order)
```

This is the standard Python idiom for multi-key sort when some keys want reverse. Use this version in the final code.

**Action:** In the code above, replace the complex `_neg_iso` and the two `scored.sort` lines with just:

```python
    # Multi-key sort using Python's stable sort: apply tiebreaker first, primary last
    scored.sort(key=lambda t: t[1], reverse=True)  # created desc (tiebreaker)
    scored.sort(key=lambda t: t[0], reverse=True)  # score desc (primary)
```

And delete the `_neg_iso` helper function entirely.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_retrieval.py -v`

Expected: all retrieval tests PASS (Task 3 + 4 + 5 combined, ~25 tests).

- [ ] **Step 5: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/retrieval.py tests/test_retrieval.py
git commit -m "feat(retrieval): scoring + session-start orchestrator

retrieve_top_n: score entries across two ledgers via symbol hit under
canonical equivalence. Dedup by (speaker, id). Sort score desc, tiebreak
created desc. Return top N as RetrievedImpression.

run_session_start_retrieval: full pipeline — extract symbols (Flash) →
load both ledgers → 1-hop co-occurrence expansion → score → top N →
package as RetrievalResult with debug info."
```

---

### Task 6: `prompt_assembler` extension — `## 你的內在` block

**Files:**
- Modify: `src/empty_space/prompt_assembler.py`
- Modify: `tests/test_prompt_assembler.py`

- [ ] **Step 1: Add failing tests to `tests/test_prompt_assembler.py`**

Append to the end of the test file:

```python
# --- Level 2: 你的內在 block ---

from empty_space.schemas import RetrievedImpression


def _make_retrieved(text: str, speaker: str = "counterpart") -> RetrievedImpression:
    return RetrievedImpression(
        id="imp_001",
        text=text,
        symbols=("x",),
        speaker=speaker,  # type: ignore[arg-type]
        persona_name="兒子",
        from_run="r/t",
        from_turn=1,
        score=1,
        matched_symbols=("x",),
    )


def test_system_prompt_omits_inner_block_when_no_prelude_and_no_retrieved(
    mother_persona, son_persona, hospital_setting, initial_state,
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
        prelude=None,
        retrieved_impressions=[],
    )
    assert "## 你的內在" not in prompt


def test_system_prompt_inner_block_with_only_prelude(
    mother_persona, son_persona, hospital_setting, initial_state,
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
        prelude="你昨夜夢到他小時候被帶走。",
        retrieved_impressions=[],
    )
    assert "## 你的內在" in prompt
    assert "你昨夜夢到他小時候被帶走。" in prompt
    assert "你可能想起的：" not in prompt


def test_system_prompt_inner_block_with_only_retrieved(
    mother_persona, son_persona, hospital_setting, initial_state,
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
        prelude=None,
        retrieved_impressions=[
            _make_retrieved("她的沉默在這一刻比任何辯解都沉"),
            _make_retrieved("他看著鞋帶不看她的眼"),
        ],
    )
    assert "## 你的內在" in prompt
    assert "你可能想起的：" in prompt
    assert "- 她的沉默在這一刻比任何辯解都沉" in prompt
    assert "- 他看著鞋帶不看她的眼" in prompt


def test_system_prompt_inner_block_with_both_prelude_and_retrieved(
    mother_persona, son_persona, hospital_setting, initial_state,
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
        prelude="你昨夜夢到他小時候被帶走。",
        retrieved_impressions=[_make_retrieved("她的沉默比辯解都沉")],
    )
    assert "## 你的內在" in prompt
    assert "你昨夜夢到他小時候被帶走。" in prompt
    assert "你可能想起的：" in prompt
    assert "- 她的沉默比辯解都沉" in prompt
    # prelude should come before retrieved within the block
    prelude_pos = prompt.find("你昨夜夢到他小時候被帶走。")
    recall_pos = prompt.find("你可能想起的：")
    assert 0 <= prelude_pos < recall_pos


def test_system_prompt_inner_block_position_after_xianchang(
    mother_persona, son_persona, hospital_setting, initial_state,
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise="父親 ICU。",
        initial_state=initial_state,
        active_events=[],
        prelude="你在想他。",
        retrieved_impressions=[],
    )
    # Order: 貫通軸 → 關係層 → 此刻 → 現場 → 你的內在 → 輸出格式
    assert prompt.index("## 現場") < prompt.index("## 你的內在")
    assert prompt.index("## 你的內在") < prompt.index("## 輸出格式")


def test_system_prompt_retrieved_impressions_rendered_as_bullet_list(
    mother_persona, son_persona, hospital_setting, initial_state,
):
    prompt = build_system_prompt(
        persona=mother_persona,
        counterpart_name=son_persona.name,
        setting=hospital_setting,
        scene_premise=None,
        initial_state=initial_state,
        active_events=[],
        prelude=None,
        retrieved_impressions=[
            _make_retrieved("印象一"),
            _make_retrieved("印象二"),
            _make_retrieved("印象三"),
        ],
    )
    # Exact bullet format
    assert "你可能想起的：\n- 印象一\n- 印象二\n- 印象三" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_prompt_assembler.py -v -k "inner or 你的內在 or retrieved or prelude"`

Expected: FAIL — either `TypeError: build_system_prompt() got an unexpected keyword argument 'prelude'` or missing kwargs.

- [ ] **Step 3: Extend `build_system_prompt` in `src/empty_space/prompt_assembler.py`**

Find the current signature:

```python
def build_system_prompt(
    persona: Persona,
    counterpart_name: str,
    setting: Setting,
    scene_premise: str | None,
    initial_state: InitialState,
    active_events: list[tuple[int, str]],
    ambient_echo: list[str] | None = None,
) -> str:
```

Change to:

```python
def build_system_prompt(
    persona: Persona,
    counterpart_name: str,
    setting: Setting,
    scene_premise: str | None,
    initial_state: InitialState,
    active_events: list[tuple[int, str]],
    prelude: str | None = None,
    retrieved_impressions: list["RetrievedImpression"] | None = None,
    ambient_echo: list[str] | None = None,
) -> str:
```

Add the import at the top (keep the TYPE_CHECKING pattern if used, else just import directly):

```python
from empty_space.schemas import (
    InitialState,
    Persona,
    RetrievedImpression,
    Setting,
    Turn,
)
```

Then find the line where blocks are assembled — it should look like:

```python
    blocks.append(f"## 輸出格式\n{_OUTPUT_FORMAT_INSTRUCTION}")

    return "\n\n".join(blocks)
```

Before the `## 輸出格式` append, insert the `## 你的內在` block construction:

```python
    # Level 2: 你的內在 block — conditionally added
    inner_parts: list[str] = []
    if prelude:
        inner_parts.append(prelude.rstrip())
    if retrieved_impressions:
        recall_lines = ["你可能想起的："] + [f"- {imp.text}" for imp in retrieved_impressions]
        inner_parts.append("\n".join(recall_lines))
    if inner_parts:
        blocks.append("## 你的內在\n" + "\n\n".join(inner_parts))

    blocks.append(f"## 輸出格式\n{_OUTPUT_FORMAT_INSTRUCTION}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_prompt_assembler.py -v`

Expected: all prompt_assembler tests PASS (Phase 2 existing 15 + 6 new = 21).

- [ ] **Step 5: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/prompt_assembler.py tests/test_prompt_assembler.py
git commit -m "feat(prompt_assembler): 你的內在 block for Level 2 memory injection

build_system_prompt gains prelude + retrieved_impressions kwargs.
Emits ## 你的內在 block positioned between 現場 and 輸出格式 (closest
to user message, to resist attention dilution over a long session).
Block conditionally omitted if both prelude and retrieved are empty.
Internal order: prelude first, then '你可能想起的：' bullet list."
```

---

### Task 7: `writer` extensions — write_retrieval + turn yaml field + meta fields

**Files:**
- Modify: `src/empty_space/writer.py`
- Modify: `tests/test_writer.py`

- [ ] **Step 1: Add failing tests to `tests/test_writer.py`**

Append to the end of the test file:

```python
# --- Level 2 additions ---

from empty_space.schemas import RetrievalResult, RetrievedImpression
from empty_space.writer import write_retrieval


def _make_retrieval_result(
    role: str = "protagonist",
    name: str = "母親",
    impressions: list[RetrievedImpression] | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        speaker_role=role,  # type: ignore[arg-type]
        persona_name=name,
        query_text="父親在 ICU。\n\n你昨夜夢到他。",
        query_symbols=["父親", "ICU", "夢"],
        expanded_symbols=["父親", "ICU", "夢", "走廊"],
        impressions=impressions or [],
        flash_latency_ms=250,
        flash_tokens_in=130,
        flash_tokens_out=28,
    )


def test_write_retrieval_records_both_roles(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)

    p = _make_retrieval_result(role="protagonist", name="母親")
    c = _make_retrieval_result(role="counterpart", name="兒子")
    write_retrieval(out_dir, protagonist=p, counterpart=c)

    retrieval_file = out_dir / "retrieval.yaml"
    assert retrieval_file.is_file()
    loaded = yaml.safe_load(retrieval_file.read_text(encoding="utf-8"))
    assert loaded["protagonist"]["persona_name"] == "母親"
    assert loaded["protagonist"]["query_symbols"] == ["父親", "ICU", "夢"]
    assert loaded["protagonist"]["flash_tokens_in"] == 130
    assert loaded["counterpart"]["persona_name"] == "兒子"


def test_write_retrieval_empty_impressions(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    p = _make_retrieval_result(impressions=[])
    c = _make_retrieval_result(impressions=[])
    write_retrieval(out_dir, protagonist=p, counterpart=c)

    loaded = yaml.safe_load((out_dir / "retrieval.yaml").read_text(encoding="utf-8"))
    assert loaded["protagonist"]["impressions"] == []
    assert loaded["counterpart"]["impressions"] == []


def test_write_retrieval_records_impression_details(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    imp = RetrievedImpression(
        id="imp_042",
        text="她的沉默比辯解都沉",
        symbols=("沉默", "辯解"),
        speaker="counterpart",
        persona_name="兒子",
        from_run="prev/2026-04-20T09-00-00",
        from_turn=8,
        score=1,
        matched_symbols=("沉默",),
    )
    p = _make_retrieval_result(impressions=[imp])
    c = _make_retrieval_result(impressions=[])
    write_retrieval(out_dir, protagonist=p, counterpart=c)

    loaded = yaml.safe_load((out_dir / "retrieval.yaml").read_text(encoding="utf-8"))
    imp_data = loaded["protagonist"]["impressions"][0]
    assert imp_data["id"] == "imp_042"
    assert imp_data["text"] == "她的沉默比辯解都沉"
    assert imp_data["symbols"] == ["沉默", "辯解"]
    assert imp_data["matched_symbols"] == ["沉默"]
    assert imp_data["from_turn"] == 8
    assert imp_data["score"] == 1


def test_append_turn_records_retrieved_impressions_field(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    imp = RetrievedImpression(
        id="imp_007",
        text="她走路時肩膀稍微往前",
        symbols=("肩膀", "前傾"),
        speaker="counterpart",
        persona_name="兒子",
        from_run="prev/x",
        from_turn=3,
        score=1,
        matched_symbols=("肩膀",),
    )
    turn = _make_turn(
        turn_number=1,
        speaker="protagonist",
        name="母親",
        content="話。",
    )
    # Augment with retrieved_impressions by replacing (dataclasses are mutable by default)
    from empty_space.schemas import Turn as _Turn
    # Rebuild turn with retrieved field
    turn = _Turn(
        turn_number=1,
        speaker="protagonist",
        persona_name="母親",
        content="話。",
        candidate_impressions=[],
        prompt_system="sys",
        prompt_user="user",
        raw_response="話。",
        tokens_in=10,
        tokens_out=5,
        model="gemini-2.5-flash",
        latency_ms=100,
        timestamp="2026-04-21T11:30:00Z",
        director_events_active=[],
        parse_error=None,
        retrieved_impressions=[imp],
    )
    append_turn(out_dir, turn)

    loaded = yaml.safe_load((out_dir / "turns" / "turn_001.yaml").read_text(encoding="utf-8"))
    assert "retrieved_impressions" in loaded
    assert len(loaded["retrieved_impressions"]) == 1
    assert loaded["retrieved_impressions"][0]["id"] == "imp_007"
    assert loaded["retrieved_impressions"][0]["text"] == "她走路時肩膀稍微往前"


def test_write_meta_records_retrieval_and_ledger_fields(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)

    write_meta(
        out_dir=out_dir,
        config=sample_config,
        total_turns=2,
        termination_reason="max_turns",
        total_tokens_in=100,
        total_tokens_out=20,
        total_candidate_impressions=3,
        turns_with_parse_error=0,
        director_events_triggered=[],
        models_used=["gemini-2.5-flash"],
        duration_seconds=5.0,
        # Level 2 additions
        retrieval_total_tokens_in=250,
        retrieval_total_tokens_out=55,
        ledger_appends=[
            {"relationship": "母親_x_兒子", "speaker": "protagonist",
             "persona_name": "母親", "candidates_added": 2, "new_ledger_version": 3},
            {"relationship": "母親_x_兒子", "speaker": "counterpart",
             "persona_name": "兒子", "candidates_added": 1, "new_ledger_version": 3},
        ],
    )
    meta = yaml.safe_load((out_dir / "meta.yaml").read_text(encoding="utf-8"))
    assert meta["retrieval_total_tokens_in"] == 250
    assert meta["retrieval_total_tokens_out"] == 55
    assert len(meta["ledger_appends"]) == 2
    assert meta["ledger_appends"][0]["candidates_added"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_writer.py -v -k "retrieval or retrieved or ledger_appends"`

Expected: FAIL — missing imports, missing kwargs, or missing Turn field.

- [ ] **Step 3: Modify `src/empty_space/schemas.py` — add `retrieved_impressions` field to Turn**

Find the `Turn` dataclass. Add `retrieved_impressions` field at the end (after `parse_error`):

```python
@dataclass
class Turn:
    """One turn's full record — prompt, response, parse result, timing."""
    turn_number: int
    speaker: Literal["protagonist", "counterpart"]
    persona_name: str
    content: str
    candidate_impressions: list[CandidateImpression]
    prompt_system: str
    prompt_user: str
    raw_response: str
    tokens_in: int
    tokens_out: int
    model: str
    latency_ms: int
    timestamp: str
    director_events_active: list[tuple[int, str]]
    parse_error: str | None = None
    retrieved_impressions: list[RetrievedImpression] = field(default_factory=list)
```

Also ensure `RetrievedImpression` is declared **before** `Turn` in the file (reorder if needed). Since we added `RetrievedImpression` at the end in Task 1, move it above `Turn`.

**Reorder note:** Structure should be:
1. Persona / Setting / PersonaRef / SettingRef / InitialState / Termination
2. ExperimentConfig
3. CandidateImpression
4. LedgerEntry / Ledger (already added in Task 1)
5. RetrievedImpression (already added in Task 1 — but need it before Turn)
6. Turn (needs update)
7. SessionResult
8. RetrievalResult (already added)

Since `Turn` already exists from Phase 2 and `RetrievedImpression` was appended at the end in Task 1, move the `RetrievedImpression` declaration to before `Turn`.

- [ ] **Step 4: Extend `src/empty_space/writer.py` — write_retrieval function**

Add at the bottom of `writer.py`:

```python
def write_retrieval(
    out_dir: Path,
    *,
    protagonist: "RetrievalResult",
    counterpart: "RetrievalResult",
) -> None:
    """Write retrieval.yaml with both roles' session-start retrieval outcomes."""
    from empty_space.schemas import RetrievalResult  # local import; avoid circular
    data = {
        "protagonist": _retrieval_to_yaml_dict(protagonist),
        "counterpart": _retrieval_to_yaml_dict(counterpart),
    }
    _atomic_write_yaml(out_dir / "retrieval.yaml", data)


def _retrieval_to_yaml_dict(r) -> dict:
    return {
        "speaker_role": r.speaker_role,
        "persona_name": r.persona_name,
        "query_text": r.query_text,
        "query_symbols": list(r.query_symbols),
        "expanded_symbols": list(r.expanded_symbols),
        "impressions": [
            {
                "id": imp.id,
                "text": imp.text,
                "symbols": list(imp.symbols),
                "speaker": imp.speaker,
                "persona_name": imp.persona_name,
                "from_run": imp.from_run,
                "from_turn": imp.from_turn,
                "score": imp.score,
                "matched_symbols": list(imp.matched_symbols),
            }
            for imp in r.impressions
        ],
        "flash_latency_ms": r.flash_latency_ms,
        "flash_tokens_in": r.flash_tokens_in,
        "flash_tokens_out": r.flash_tokens_out,
    }
```

- [ ] **Step 5: Extend `_turn_to_yaml_dict` in `writer.py` — add retrieved_impressions field**

Find the existing `_turn_to_yaml_dict` function. Add at the end of the returned dict (after `parse_error`):

```python
def _turn_to_yaml_dict(turn: "Turn") -> dict:
    return {
        "turn": turn.turn_number,
        "speaker": turn.speaker,
        "persona_name": turn.persona_name,
        "timestamp": turn.timestamp,
        "prompt_assembled": {
            "system": turn.prompt_system,
            "user": turn.prompt_user,
            "prompt_tokens": turn.tokens_in,
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
        "retrieved_impressions": [                                # NEW
            {
                "id": imp.id,
                "text": imp.text,
                "symbols": list(imp.symbols),
                "speaker": imp.speaker,
                "persona_name": imp.persona_name,
                "from_run": imp.from_run,
                "from_turn": imp.from_turn,
                "score": imp.score,
                "matched_symbols": list(imp.matched_symbols),
            }
            for imp in turn.retrieved_impressions
        ],
    }
```

- [ ] **Step 6: Extend `write_meta` in `writer.py` — retrieval + ledger_appends kwargs**

Find the existing `write_meta`. Add two new kwargs with defaults:

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
    retrieval_total_tokens_in: int = 0,             # NEW
    retrieval_total_tokens_out: int = 0,            # NEW
    ledger_appends: list[dict] | None = None,       # NEW
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
        "retrieval_total_tokens_in": retrieval_total_tokens_in,
        "retrieval_total_tokens_out": retrieval_total_tokens_out,
        "ledger_appends": ledger_appends or [],
    }
    _atomic_write_yaml(out_dir / "meta.yaml", meta)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_writer.py -v`

Expected: all writer tests PASS (Phase 2 existing 14 + new ~4 = ~18).

- [ ] **Step 8: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/schemas.py src/empty_space/writer.py tests/test_writer.py
git commit -m "feat(writer): Level 2 — write_retrieval + turn yaml + meta extensions

Adds write_retrieval function producing retrieval.yaml (session-start
retrieval audit trail per role).
Extends _turn_to_yaml_dict to include retrieved_impressions field.
Extends write_meta with retrieval_total_tokens_in/out + ledger_appends
(all defaulting to zero-valued, backward compatible)."
```

---

### Task 8: Runner integration + `config/symbol_synonyms.yaml`

**Files:**
- Modify: `src/empty_space/runner.py`
- Create: `config/symbol_synonyms.yaml`

- [ ] **Step 1: Create `config/symbol_synonyms.yaml`**

```yaml
# Synonym canonical map for cross-session ledger retrieval.
# Format: each group is a list; first element is the canonical form.
#
# Example:
#   groups:
#     - [愧疚, 愧疚感, 罪惡感]       # three variants normalize to 愧疚
#     - [沉默, 不說話, 沉默不語]
#
# Effect (spec §7.7): only applies at retrieval matching step. Cooccurrence
# and candidate impression ledger entries are stored verbatim (Flash's
# original output), preserving the raw evidence trail.

groups: []
```

- [ ] **Step 2: Modify `src/empty_space/runner.py` — extend imports**

Find the existing imports at the top of `runner.py`. Add:

```python
from empty_space import ledger
from empty_space.retrieval import load_synonym_map, run_session_start_retrieval
from empty_space.schemas import (
    ExperimentConfig,
    Persona,
    RetrievalResult,
    SessionResult,
    Setting,
    Turn,
)
from empty_space.writer import append_turn, init_run, write_meta, write_retrieval
```

- [ ] **Step 3: Extend `SessionState` in `runner.py`**

Find the `SessionState` dataclass. Add two new fields:

```python
@dataclass
class SessionState:
    """Runner-internal state. Not persisted to schemas.py."""
    config: ExperimentConfig
    protagonist: Persona
    counterpart: Persona
    setting: Setting
    turns: list[Turn] = field(default_factory=list)
    active_events: list[tuple[int, str]] = field(default_factory=list)
    retrieval_protagonist: RetrievalResult | None = None   # NEW
    retrieval_counterpart: RetrievalResult | None = None   # NEW
```

- [ ] **Step 4: Add helpers at bottom of `runner.py`**

```python
def _compose_query(scene_premise: str | None, prelude: str | None) -> str:
    """Join scene_premise + prelude for retrieval query (both optional)."""
    parts = [scene_premise, prelude]
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def _append_session_ledgers(
    *,
    relationship: str,
    protagonist_persona: Persona,
    counterpart_persona: Persona,
    turns: list[Turn],
    source_run: str,
) -> list[dict]:
    """Bucket turns' candidates by speaker, append each bucket to its ledger.

    Returns a list of dicts describing each append, for meta.yaml.
    Empty buckets are skipped (no ledger file created for that side).
    """
    p_candidates = [
        (t.turn_number, imp) for t in turns
        if t.speaker == "protagonist"
        for imp in t.candidate_impressions
    ]
    c_candidates = [
        (t.turn_number, imp) for t in turns
        if t.speaker == "counterpart"
        for imp in t.candidate_impressions
    ]

    appends: list[dict] = []

    if p_candidates:
        ledger.append_session_candidates(
            relationship=relationship,
            speaker_role="protagonist",
            persona_name=protagonist_persona.name,
            candidates=p_candidates,
            source_run=source_run,
        )
        new_ledger = ledger.read_ledger(
            relationship=relationship, persona_name=protagonist_persona.name,
        )
        appends.append({
            "relationship": relationship,
            "speaker": "protagonist",
            "persona_name": protagonist_persona.name,
            "candidates_added": len(p_candidates),
            "new_ledger_version": new_ledger.ledger_version,
        })

    if c_candidates:
        ledger.append_session_candidates(
            relationship=relationship,
            speaker_role="counterpart",
            persona_name=counterpart_persona.name,
            candidates=c_candidates,
            source_run=source_run,
        )
        new_ledger = ledger.read_ledger(
            relationship=relationship, persona_name=counterpart_persona.name,
        )
        appends.append({
            "relationship": relationship,
            "speaker": "counterpart",
            "persona_name": counterpart_persona.name,
            "candidates_added": len(c_candidates),
            "new_ledger_version": new_ledger.ledger_version,
        })

    return appends
```

- [ ] **Step 5: Modify `run_session` to add session-start retrieval + session-end append**

Find the existing `run_session` function. Key changes (apply in order):

**a. After `init_run(out_dir, config)` and before building `SessionState`**, add:

```python
    # Level 2: Session-start retrieval (once per role)
    synonym_map = load_synonym_map()
    relationship = f"{protagonist.name}_x_{counterpart.name}"

    retrieval_protagonist = run_session_start_retrieval(
        speaker_role="protagonist",
        persona_name=protagonist.name,
        query_text=_compose_query(config.scene_premise, config.protagonist_prelude),
        relationship=relationship,
        other_persona_name=counterpart.name,
        synonym_map=synonym_map,
        llm_client=llm_client,
        top_n=3,
    )
    retrieval_counterpart = run_session_start_retrieval(
        speaker_role="counterpart",
        persona_name=counterpart.name,
        query_text=_compose_query(config.scene_premise, config.counterpart_prelude),
        relationship=relationship,
        other_persona_name=protagonist.name,
        synonym_map=synonym_map,
        llm_client=llm_client,
        top_n=3,
    )
    write_retrieval(
        out_dir,
        protagonist=retrieval_protagonist,
        counterpart=retrieval_counterpart,
    )
```

**b. When constructing `SessionState`**, add the retrieval fields:

```python
    state = SessionState(
        config=config,
        protagonist=protagonist,
        counterpart=counterpart,
        setting=setting,
        retrieval_protagonist=retrieval_protagonist,  # NEW
        retrieval_counterpart=retrieval_counterpart,  # NEW
    )
```

**c. Inside the `for n in range(1, config.max_turns + 1):` loop**, after the speaker role judgment:

Change the `build_system_prompt` call to pass `prelude` + `retrieved_impressions`:

```python
        # Select this role's retrieval + prelude
        role_retrieval = (
            state.retrieval_protagonist if speaker_role == "protagonist"
            else state.retrieval_counterpart
        )
        role_prelude = (
            config.protagonist_prelude if speaker_role == "protagonist"
            else config.counterpart_prelude
        )

        system_prompt = build_system_prompt(
            persona=speaker_persona,
            counterpart_name=other_party_name,
            setting=setting,
            scene_premise=config.scene_premise,
            initial_state=config.initial_state,
            active_events=state.active_events,
            prelude=role_prelude,
            retrieved_impressions=role_retrieval.impressions,
        )
```

**d. When constructing `Turn` inside the loop**, add the `retrieved_impressions` field:

```python
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
            retrieved_impressions=list(role_retrieval.impressions),  # NEW
        )
```

**e. After the turn loop ends and before `write_meta`**, add:

```python
    # Level 2: Session-end ledger append
    source_run = f"{config.exp_id}/{timestamp}"
    ledger_appends = _append_session_ledgers(
        relationship=relationship,
        protagonist_persona=protagonist,
        counterpart_persona=counterpart,
        turns=state.turns,
        source_run=source_run,
    )
```

**f. Modify the `write_meta` call to pass new kwargs**:

```python
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
        # Level 2 additions
        retrieval_total_tokens_in=(
            retrieval_protagonist.flash_tokens_in + retrieval_counterpart.flash_tokens_in
        ),
        retrieval_total_tokens_out=(
            retrieval_protagonist.flash_tokens_out + retrieval_counterpart.flash_tokens_out
        ),
        ledger_appends=ledger_appends,
    )
```

- [ ] **Step 6: Run all existing tests — no regressions**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_runner_integration.py -v`

Expected: existing 10 runner_integration tests should STILL PASS (they use MockLLMClient; the mock now gets called for extract_symbols too, so they'll need enough responses).

**IF any existing test fails**, inspect the output. Common causes:
- MockLLMClient ran out of responses: `test_happy_path_runs_all_turns` provides 4 responses for 4 turns but now the runner makes 2 extra Flash calls for extract_symbols first. **Fix**: update existing tests to add 2 more leading responses, OR modify the approach: return empty symbols for empty ledgers (MockLLMClient returns something like `- x` that gets extracted; empty ledger means no impressions retrieved anyway).

The simplest fix for existing tests: since `minimal_config` in `test_runner_integration.py` has `scene_premise` but no preludes, extract_symbols WILL be called (scene_premise is non-empty). **Update existing integration tests to prepend 2 extract responses to the MockLLMClient responses list**.

Example patch for `test_happy_path_runs_all_turns`:

```python
    responses = [
        "- 醫院\n- 父親\n",   # NEW: protagonist extract_symbols
        "- 醫院\n- 父親\n",   # NEW: counterpart extract_symbols
        "你回來了。",
        "嗯。",
        "⋯⋯",
        "不關我的事。",
    ]
```

Apply this fix to every affected test in `test_runner_integration.py` — prepend 2 dummy symbol responses.

- [ ] **Step 7: Run full test suite**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest -v`

Expected: all existing 75 + new (Level 2 tasks so far) tests PASS.

- [ ] **Step 8: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/runner.py config/symbol_synonyms.yaml tests/test_runner_integration.py
git commit -m "feat(runner): Level 2 integration — session-start retrieval + session-end ledger

run_session now:
- Before turn loop: runs session-start retrieval for both roles, writes retrieval.yaml
- Turn loop: passes prelude + retrieved_impressions to build_system_prompt,
  includes retrieved in Turn record
- After turn loop (before write_meta): appends session candidates to both ledgers

New config/symbol_synonyms.yaml (initial: empty groups).

Existing integration tests updated to prepend 2 extract_symbols
responses to MockLLMClient (for protagonist + counterpart)."
```

---

### Task 9: Cross-session integration tests

**Files:**
- Create: `tests/test_runner_level2.py`

- [ ] **Step 1: Create `tests/test_runner_level2.py`**

```python
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
    """第一場帳本空；session 結束後帳本被建立。"""
    config = _base_config(max_turns=2)
    responses = [
        "- 醫院\n- 父親\n",     # protagonist extract
        "- 醫院\n- 父親\n",     # counterpart extract
        "話一\n\n---IMPRESSIONS---\n- text: \"母親印象一\"\n  symbols: [A, B]\n",
        "話二\n\n---IMPRESSIONS---\n- text: \"兒子印象一\"\n  symbols: [C, D]\n",
    ]
    client = MockLLMClient(responses)
    result = run_session(config=config, llm_client=client)

    # retrieval.yaml exists, impressions empty (ledger was empty)
    ret = yaml.safe_load((result.out_dir / "retrieval.yaml").read_text(encoding="utf-8"))
    assert ret["protagonist"]["impressions"] == []
    assert ret["counterpart"]["impressions"] == []

    # After session: both ledger files exist with 1 candidate each
    ledgers_dir = redirect_all_dirs["ledgers_dir"]
    assert (ledgers_dir / "母親_x_兒子.from_母親.yaml").is_file()
    assert (ledgers_dir / "母親_x_兒子.from_兒子.yaml").is_file()

    l_p = read_ledger(relationship="母親_x_兒子", persona_name="母親")
    l_c = read_ledger(relationship="母親_x_兒子", persona_name="兒子")
    assert l_p.ledger_version == 1
    assert l_c.ledger_version == 1
    assert len(l_p.candidates) == 1
    assert len(l_c.candidates) == 1
    assert l_p.candidates[0].text == "母親印象一"
    assert l_c.candidates[0].text == "兒子印象一"


def test_second_session_retrieval_hits_first_session_impressions(redirect_all_dirs):
    """第二場 session 的 retrieval 能撈到第一場的印象。"""
    config = _base_config(max_turns=2)

    # Session 1: creates ledger with ["醫院", "父親"] matching symbols
    responses_1 = [
        "- 醫院\n",
        "- 醫院\n",
        "話一\n\n---IMPRESSIONS---\n- text: \"母親印象\"\n  symbols: [醫院, 父親]\n",
        "話二\n\n---IMPRESSIONS---\n- text: \"兒子印象\"\n  symbols: [醫院, 父親]\n",
    ]
    run_session(config=config, llm_client=MockLLMClient(responses_1))

    # Session 2: same config — retrieval should find both prior impressions
    responses_2 = [
        "- 醫院\n- 父親\n",    # protagonist extract — will match both
        "- 醫院\n- 父親\n",    # counterpart extract — will match both
        "話三",
        "話四",
    ]
    client = MockLLMClient(responses_2)
    result_2 = run_session(config=config, llm_client=client)

    ret_2 = yaml.safe_load((result_2.out_dir / "retrieval.yaml").read_text(encoding="utf-8"))
    # Both roles should see both prior impressions (共同記憶 A strategy)
    p_texts = [imp["text"] for imp in ret_2["protagonist"]["impressions"]]
    c_texts = [imp["text"] for imp in ret_2["counterpart"]["impressions"]]
    assert "母親印象" in p_texts
    assert "兒子印象" in p_texts
    assert "母親印象" in c_texts
    assert "兒子印象" in c_texts


def test_llm_exception_aborts_session_no_ledger_written(redirect_all_dirs):
    """Session 中斷時不寫帳本。"""
    config = _base_config(max_turns=4)

    class ExplodingClient:
        def __init__(self):
            self.count = 0
        def generate(self, *, system, user, model="gemini-2.5-flash"):
            self.count += 1
            # Let the 2 extract calls + 2 turns succeed, blow up on turn 3
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
    """預先 seed ledger，新 session 的 system prompt 含命中印象。"""
    # Pre-seed a ledger via direct append
    append_session_candidates(
        relationship="母親_x_兒子",
        speaker_role="counterpart",
        persona_name="兒子",
        candidates=[
            (5, CandidateImpression(
                text="她的手不動，像假的",
                symbols=["手", "不動", "假"],
            )),
        ],
        source_run="prev_exp/2026-04-20T09-00-00",
    )

    config = _base_config(max_turns=2)
    responses = [
        "- 手\n- 不動\n",     # protagonist extract → will match the seeded impression
        "- 手\n- 不動\n",     # counterpart extract → will also match
        "話一",
        "話二",
    ]
    client = MockLLMClient(responses)
    result = run_session(config=config, llm_client=client)

    # Turn 1's system prompt should contain the retrieved text
    turn_1 = yaml.safe_load((result.out_dir / "turns" / "turn_001.yaml").read_text(encoding="utf-8"))
    assert "她的手不動，像假的" in turn_1["prompt_assembled"]["system"]
    # Retrieved field in turn yaml matches
    assert any(
        r["text"] == "她的手不動，像假的"
        for r in turn_1["retrieved_impressions"]
    )


def test_synonym_map_enables_variant_matching(redirect_all_dirs, tmp_path, monkeypatch):
    """同義詞字典 → query 的「愧疚」命中帳本的「愧疚感」。"""
    # Create a synonym map at a tmp path
    syn_path = tmp_path / "synonyms.yaml"
    syn_path.write_text(
        "groups:\n  - [愧疚, 愧疚感]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("empty_space.retrieval.DEFAULT_SYNONYMS_PATH", syn_path)

    # Seed ledger with 愧疚感 (variant)
    append_session_candidates(
        relationship="母親_x_兒子",
        speaker_role="counterpart",
        persona_name="兒子",
        candidates=[
            (3, CandidateImpression(
                text="她看著地板，沒看我",
                symbols=["愧疚感"],  # variant form
            )),
        ],
        source_run="prev/2026-04-20T09-00-00",
    )

    config = _base_config(max_turns=2)
    responses = [
        "- 愧疚\n",     # protagonist extract — canonical form
        "- 愧疚\n",     # counterpart extract
        "話一",
        "話二",
    ]
    client = MockLLMClient(responses)
    result = run_session(config=config, llm_client=client)

    ret = yaml.safe_load((result.out_dir / "retrieval.yaml").read_text(encoding="utf-8"))
    p_texts = [imp["text"] for imp in ret["protagonist"]["impressions"]]
    assert "她看著地板，沒看我" in p_texts
```

- [ ] **Step 2: Run tests**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_runner_level2.py -v`

Expected: all 5 cross-session tests PASS.

- [ ] **Step 3: Run full suite — no regressions**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest -v`

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add tests/test_runner_level2.py
git commit -m "test(runner_level2): cross-session ledger + retrieval integration

Covers: first-session empty ledger + append, second-session retrieval
hit, LLM exception aborts with no ledger write, pre-seeded ledger
reaches system prompt, synonym map enables variant matching."
```

---

### Task 10: End-to-end smoke test + Level 2 summary doc

**Files:**
- Create: `docs/level-2-summary.md`
- Temporary edit: `experiments/mother_x_son_hospital_v3_001.yaml` (max_turns → 4 for smoke, then revert)

- [ ] **Step 1: Verify `.env` has `GEMINI_API_KEY`**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && grep -q "^GEMINI_API_KEY=" .env && echo "key present" || echo "MISSING"`

Expected: `key present`. If MISSING, stop and report BLOCKED.

- [ ] **Step 2: Add prelude content to experiment yaml (keep it for real demo)**

Edit `experiments/mother_x_son_hospital_v3_001.yaml`. Add these after `scene_premise` and before `initial_state`:

```yaml
protagonist_prelude: |
  你昨夜夢到他小時候被帶走。醒來時枕頭是濕的。

counterpart_prelude: |
  你昨晚和女朋友分手。她說「我不能等你。」
  你還沒跟任何人說。
```

Also reduce `max_turns: 20` to `max_turns: 4` for the first smoke test.

- [ ] **Step 3: Run first session (against real Gemini)**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run python scripts/run_experiment.py mother_x_son_hospital_v3_001`

Expected output pattern:
```
✓ Completed mother_x_son_hospital_v3_001
  Output: /.../runs/mother_x_son_hospital_v3_001/2026-04-21T...
  Turns: 4
  Termination: max_turns
  Tokens in/out: ~8000-12000 / ~200-400
  Duration: ~20-80s
```

Capture the exact output. Note the tokens reported include Flash extract calls.

- [ ] **Step 4: Inspect first-session outputs**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
ls runs/mother_x_son_hospital_v3_001/*/
```

Expected files: `config.yaml`, `conversation.md`, `conversation.jsonl`, `meta.yaml`, `retrieval.yaml`, `turns/turn_001.yaml` through `turn_004.yaml`.

```bash
cat runs/mother_x_son_hospital_v3_001/*/retrieval.yaml
```

Expected: `impressions: []` for both roles (first session, empty ledgers).

Confirm the ledger files exist:

```bash
ls ledgers/
```

Expected: `母親_x_兒子.from_母親.yaml` + `母親_x_兒子.from_兒子.yaml`.

- [ ] **Step 5: Run second session (same config → retrieval should hit)**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
uv run python scripts/run_experiment.py mother_x_son_hospital_v3_001
```

Expected: new timestamp directory created.

- [ ] **Step 6: Inspect second-session retrieval**

```bash
cat runs/mother_x_son_hospital_v3_001/<second-timestamp>/retrieval.yaml
```

Expected: `impressions` should NOT be empty — should contain 0-3 retrieved impressions per role (depends on whether Flash's extracted symbols from scene_premise + prelude happen to match the first session's candidate impression symbols).

If second-session `impressions` is still empty, it means symbol overlap was zero — this is possible but worth noting. Not a bug, just indicates Flash chose different-flavored symbols in the two sessions. Still a valid smoke test result.

- [ ] **Step 7: Restore experiment yaml**

Revert `experiments/mother_x_son_hospital_v3_001.yaml` to `max_turns: 20` (keep the preludes — they're part of Level 2's ongoing state).

- [ ] **Step 8: Write `docs/level-2-summary.md`**

Mirror `docs/phase2-summary.md`'s structure. Include:

- Status, commit range (from Task 1 start to last Level 2 commit), test count, branch
- What this level shipped
- Key decisions (two-ledger per relationship; rubric砍掉; Composer 延後到 Level 4; session-start retrieval instead of per-turn; 1-hop co-occurrence; synonym dict manual)
- Module inventory: ledger.py + retrieval.py + changes to schemas/writer/runner/prompt_assembler
- Test coverage breakdown (ledger: 10, retrieval: 25, runner_level2: 5, schema: +3, prompt_assembler: +6, writer: +4, runner_integration: updated)
- Smoke-run results — concrete tokens/duration from both sessions; whether second session retrieved anything
- Known issues / follow-ups — reference spec §11 items: attention dilution unresolved (Level 3), symbol string inconsistency (synonym dict is bandage), ledger bloat (Level 4 Composer), semantic gap (future embedding)
- Pointer to Level 3 starting point: Judge (fire_release / basin_lock termination + stage/mode tracking + 可能 agentic pull retrieval)

Keep under 200 lines.

- [ ] **Step 9: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add docs/level-2-summary.md experiments/mother_x_son_hospital_v3_001.yaml
git commit -m "docs: Level 2 summary — ledger + session-start retrieval shipped

Records smoke-run results and key decisions. Experiment yaml retains
the new preludes as demo of導演手寫 internal prior state."
```

- [ ] **Step 10: Tag the milestone**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git tag -a level-2 -m "$(cat <<'EOF'
Level 2 成功 — 兩個演員跨場累積、帶著前場的印象進入下一場

Peter Brook 命題的時間維度打開：母親 × 兒子 @ 醫院 同一對角色
能跑多場 session，每場結束後 candidate impressions 沉進兩本帳本；
下場 session 開始時，Flash 從導演手寫的 prelude + scene_premise
拆 symbols，撈出帳本中符合的印象，塞進演員的 system prompt。

導演的控制哲學再深一步：
- Level 1: 導演塞世界（scene_premise + director_events）不塞嘴
- Level 2: 導演塞心（prelude 當作記憶檢索的 query 源），engine
  從跨場帳本提取命中的印象。演員 agency + 導演 agency 兩者皆在場。

技術決策：
- 兩本帳本 per 關係（by speaker），A 策略（共同記憶）retrieval
- append-only ledger + symbol_index + cooccurrence（1-hop）
- 同義詞字典手工維護，matching step 生效
- rubric 砍掉（Flash 評 Flash 自己打轉，Level 4 Composer 做
  holistic 整合）
- Composer bake 延後到 Level 4 的「一天後」
- 所有 LLM 成本仍走 Flash，Pro 保留給 Level 4

Phase 3 Judge 入口：spec §3 的 stage/mode/張力追蹤 +
fire_release/basin_lock 終止條件。
EOF
)"
git push origin level-2
```

---

## Self-Review

**1. Spec coverage:**

| Spec § | Requirement | Task |
|---|---|---|
| §1 | In scope items | Tasks 1-10 cumulatively |
| §2 | ExperimentConfig + dataclasses | Task 1 |
| §3 | Ledger schema | Task 2 |
| §4 | Module skeleton | Tasks 2, 3-5 |
| §5 | Session lifecycle | Task 8 |
| §6 | Symbol extraction Flash prompt | Task 4 |
| §7 | Retrieval algorithm (expand / score / canonicalize) | Tasks 3, 5 |
| §7.7 | Synonym dictionary | Tasks 3 (functions) + 8 (config file) |
| §8 | Runner integration | Task 8 |
| §9 | Disk schema (retrieval.yaml + turn yaml field + meta fields) | Task 7 |
| §10 | Test strategy | Tasks 1-9 cover per-module; Task 9 cross-session |
| §11 | Risks | Documented in spec; not implemented (they're acknowledged unknowns) |

All spec requirements covered. ✓

**2. Placeholder scan:** No TBD / TODO / "similar to" / "add validation" anywhere. All steps contain exact code.

**3. Type consistency:**
- `Ledger.speaker` is `Literal["protagonist", "counterpart"]` everywhere
- `RetrievedImpression.symbols` and `matched_symbols` are `tuple[str, ...]` consistently (frozen dataclass needs hashable)
- `CandidateImpression.symbols` remains `list[str]` (not frozen) — consistent with Phase 2 Task 1 cleanup
- `build_system_prompt` signature in Task 6 matches call site in Task 8 ✓
- `write_retrieval(out_dir, protagonist=..., counterpart=...)` signature in Task 7 matches call site in Task 8 ✓
- `_append_session_ledgers` return type (`list[dict]`) matches `write_meta`'s `ledger_appends` param ✓
- `_neg_iso` helper was replaced with stable sort — code in Task 5 Step 3 note reflects this ✓

All type / signature consistency verified. ✓

Plan complete.
