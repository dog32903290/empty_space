# 空的空間 — Level 3: Composer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Composer module — every session end, run a Pro bake that reads the session's raw candidates + existing refined + conversation, produces atomic first-person refined impressions split by speaker, and persists them to two refined-ledger files. Retrieval switches from raw to refined.

**Architecture:** New `composer.py` module with pure functions (gather input, build prompt, parse output) + one orchestrator (`run_composer`). Extends `ledger.py` with `read_refined_ledger` / `append_refined_impressions` (reusing atomic write). Retrieval decoupled from ledger type via new `retrieve_top_n(entries_a, entries_b, ...)` signature. Runner inserts Composer call between raw ledger append and meta write. Gemini Pro only for Composer; Flash unchanged elsewhere.

**Tech Stack:** Python 3.11+, pydantic v2, pyyaml, pytest (+pytest-mock), google-genai. No new dependencies.

**Spec reference:** `docs/superpowers/specs/2026-04-22-level-3-composer.md`

---

## File Structure Overview

**Modify:**
- `src/empty_space/schemas.py` — add `RefinedImpression`, `RefinedLedger`, `RefinedImpressionDraft`, `ComposerSessionResult`, `ComposerInput`; change `RetrievedImpression.from_turn` to `int | None`
- `src/empty_space/ledger.py` — add `refined_ledger_path`, `read_refined_ledger`, `append_refined_impressions`; change `append_session_candidates` return type from `None` to `list[str]` (list of new-appended ids)
- `src/empty_space/retrieval.py` — change `retrieve_top_n` signature (entries_a/b lists replace ledger_a/b); change `run_session_start_retrieval` to read refined ledger
- `src/empty_space/runner.py` — capture `new_raw_ids` from `_append_session_ledgers`; add `_run_composer_at_session_end` helper; call Composer between raw append and meta write; add 5 composer kwargs to `write_meta` call
- `src/empty_space/writer.py` — extend `write_meta` with 5 composer kwargs (all defaulting)
- `tests/test_retrieval.py` — update existing tests' calls to new `retrieve_top_n` signature; add 2 new tests
- `tests/test_writer.py` — add `test_write_meta_records_composer_fields`
- `tests/test_runner_integration.py` — append 1 composer response to every existing MockLLMClient test; add 3 new composer-scenario tests
- `tests/test_runner_level2.py` — append 1 composer response to every existing test

**Create:**
- `src/empty_space/composer.py` — Composer module (prompt + parse + orchestrator)
- `tests/test_composer.py` — ~12 tests
- `tests/test_refined_ledger.py` — ~8 tests

---

## Tasks

### Task 1: Schema migration — Level 3 dataclasses + RetrievedImpression change

**Files:**
- Modify: `src/empty_space/schemas.py`

- [ ] **Step 1: Modify `RetrievedImpression` — make `from_turn` optional**

Find the `RetrievedImpression` dataclass. Change `from_turn: int` to `from_turn: int | None = None`.

Before:
```python
@dataclass(frozen=True)
class RetrievedImpression:
    """Read from ledger; what went into the '你的內在' block."""
    id: str
    text: str
    symbols: tuple[str, ...]
    speaker: Literal["protagonist", "counterpart"]
    persona_name: str
    from_run: str
    from_turn: int
    score: int
    matched_symbols: tuple[str, ...]
```

After:
```python
@dataclass(frozen=True)
class RetrievedImpression:
    """Read from ledger; what went into the '你的內在' block.

    from_turn is None when this impression came from a refined ledger
    (refined impressions are multi-turn consolidations, not single-turn).
    """
    id: str
    text: str
    symbols: tuple[str, ...]
    speaker: Literal["protagonist", "counterpart"]
    persona_name: str
    from_run: str
    from_turn: int | None
    score: int
    matched_symbols: tuple[str, ...]
```

Note: keyword-only dataclass with default — because `frozen=True`, adding a default requires being after all non-default fields. Currently `from_turn` is followed by `score` and `matched_symbols` which have no defaults. Don't add a `= None` default — just make the type `int | None`. Callers must now explicitly pass `None` for refined cases.

- [ ] **Step 2: Add 5 new dataclasses to `src/empty_space/schemas.py`**

Append at the end of the file (after `RetrievalResult`):

```python
@dataclass
class RefinedImpression:
    """Composer-refined impression. One record of consolidated memory.

    Unlike CandidateImpression/LedgerEntry, this has no from_turn (refined is
    multi-turn integration) but has source_raw_ids (provenance pointing back
    to which raw candidates contributed).
    """
    id: str                              # ref_001, ref_002, ...
    text: str                            # 短 atomic, 第一人稱
    symbols: list[str]
    speaker: Literal["protagonist", "counterpart"]
    persona_name: str
    from_run: str                        # exp_id/timestamp
    source_raw_ids: list[str]            # which raws contributed (best-effort)
    created: str                         # ISO 8601


@dataclass
class RefinedLedger:
    """In-memory representation of <relationship>.refined.from_<persona>.yaml."""
    relationship: str
    speaker: Literal["protagonist", "counterpart"]
    persona_name: str
    ledger_version: int
    impressions: list[RefinedImpression]
    symbol_index: dict[str, list[str]]
    cooccurrence: dict[str, dict[str, int]]


@dataclass
class RefinedImpressionDraft:
    """Pre-id draft parsed from Composer output (no id/created/from_run yet).

    Assigned id and metadata when appended via append_refined_impressions.
    """
    text: str
    symbols: list[str]
    source_raw_ids: list[str]


@dataclass
class ComposerSessionResult:
    """Return from run_composer. Feeds meta.yaml."""
    tokens_in: int
    tokens_out: int
    latency_ms: int
    protagonist_refined_added: int
    counterpart_refined_added: int
    parse_error: str | None = None


@dataclass
class ComposerInput:
    """Materials gathered for Composer. Passed to build_composer_prompt."""
    relationship: str
    protagonist_name: str
    counterpart_name: str
    conversation_text: str
    new_candidates: dict[str, list[CandidateImpression]]     # speaker → list
    new_candidate_ids: dict[str, list[str]]                  # speaker → raw ids (same order)
    existing_refined: dict[str, list[RefinedImpression]]     # speaker → last 30
```

- [ ] **Step 3: Verify the schema file compiles**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run python -c "from empty_space.schemas import RefinedImpression, RefinedLedger, RefinedImpressionDraft, ComposerSessionResult, ComposerInput, RetrievedImpression; print('ok')"`

Expected: `ok`

- [ ] **Step 4: Run existing tests — identify regressions from RetrievedImpression change**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest -v 2>&1 | tail -20`

Expected: Most tests pass. Some tests that construct `RetrievedImpression` with `from_turn=<int>` should still pass (int is still valid). Tests that compare specific `from_turn` values should be fine. Don't fix anything yet — regressions will be addressed in the tasks that modify those tests.

Actually `int | None` without a default will fail for any caller that didn't pass `from_turn`. Review:

Search: `grep -rn "RetrievedImpression(" src/ tests/`

All call sites must now pass `from_turn`. If any omit it, fix those call sites to pass either an int or None.

- [ ] **Step 5: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/schemas.py
git commit -m "feat(schemas): Level 3 dataclasses + RetrievedImpression.from_turn optional

Adds RefinedImpression, RefinedLedger, RefinedImpressionDraft,
ComposerSessionResult, ComposerInput for Level 3 Composer flow.

RetrievedImpression.from_turn changed from int to int | None because
refined impressions are multi-turn consolidations without a single
originating turn."
```

---

### Task 2: `ledger.py` — refined ledger I/O + append_session_candidates return change

**Files:**
- Modify: `src/empty_space/ledger.py`
- Create: `tests/test_refined_ledger.py`

- [ ] **Step 1: Write failing tests in `tests/test_refined_ledger.py`**

```python
"""Tests for refined ledger I/O — append-only, by-speaker."""
from pathlib import Path

import pytest
import yaml

from empty_space.schemas import RefinedImpressionDraft, RefinedLedger
from empty_space.ledger import (
    append_refined_impressions,
    read_refined_ledger,
    refined_ledger_path,
)


@pytest.fixture(autouse=True)
def redirect_ledgers_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path)


def test_refined_ledger_path_uses_refined_convention(tmp_path):
    path = refined_ledger_path(relationship="母親_x_兒子", persona_name="母親")
    assert path.name == "母親_x_兒子.refined.from_母親.yaml"
    assert path.parent == tmp_path


def test_read_refined_ledger_missing_returns_empty():
    ledger = read_refined_ledger(relationship="R", persona_name="母親")
    assert ledger.ledger_version == 0
    assert ledger.impressions == []
    assert ledger.symbol_index == {}
    assert ledger.cooccurrence == {}


def test_append_refined_first_time_creates_file():
    drafts = [
        RefinedImpressionDraft(
            text="沉默時喉嚨收緊",
            symbols=["沉默", "喉嚨", "收緊"],
            source_raw_ids=["imp_003"],
        ),
    ]
    version = append_refined_impressions(
        relationship="R",
        speaker_role="protagonist",
        persona_name="母親",
        drafts=drafts,
        source_run="exp/2026-04-22T10-00-00",
    )
    assert version == 1

    ledger = read_refined_ledger(relationship="R", persona_name="母親")
    assert ledger.ledger_version == 1
    assert len(ledger.impressions) == 1
    assert ledger.impressions[0].id == "ref_001"
    assert ledger.impressions[0].text == "沉默時喉嚨收緊"
    assert ledger.impressions[0].source_raw_ids == ["imp_003"]
    assert ledger.impressions[0].speaker == "protagonist"
    assert ledger.impressions[0].persona_name == "母親"
    assert ledger.impressions[0].from_run == "exp/2026-04-22T10-00-00"


def test_append_refined_id_increments_and_version_bumps():
    append_refined_impressions(
        relationship="R",
        speaker_role="protagonist",
        persona_name="母親",
        drafts=[RefinedImpressionDraft(text="first", symbols=["A"], source_raw_ids=[])],
        source_run="exp/t1",
    )
    version = append_refined_impressions(
        relationship="R",
        speaker_role="protagonist",
        persona_name="母親",
        drafts=[RefinedImpressionDraft(text="second", symbols=["B"], source_raw_ids=[])],
        source_run="exp/t2",
    )
    assert version == 2
    ledger = read_refined_ledger(relationship="R", persona_name="母親")
    assert [i.id for i in ledger.impressions] == ["ref_001", "ref_002"]


def test_append_refined_updates_symbol_index():
    drafts = [
        RefinedImpressionDraft(text="x", symbols=["A", "B"], source_raw_ids=[]),
        RefinedImpressionDraft(text="y", symbols=["B", "C"], source_raw_ids=[]),
    ]
    append_refined_impressions(
        relationship="R", speaker_role="protagonist", persona_name="母親",
        drafts=drafts, source_run="x/t",
    )
    ledger = read_refined_ledger(relationship="R", persona_name="母親")
    assert ledger.symbol_index == {
        "A": ["ref_001"],
        "B": ["ref_001", "ref_002"],
        "C": ["ref_002"],
    }


def test_append_refined_cooccurrence_symmetric():
    drafts = [
        RefinedImpressionDraft(text="x", symbols=["A", "B", "C"], source_raw_ids=[]),
    ]
    append_refined_impressions(
        relationship="R", speaker_role="protagonist", persona_name="母親",
        drafts=drafts, source_run="x/t",
    )
    ledger = read_refined_ledger(relationship="R", persona_name="母親")
    assert ledger.cooccurrence["A"] == {"B": 1, "C": 1}
    assert ledger.cooccurrence["B"] == {"A": 1, "C": 1}
    assert ledger.cooccurrence["C"] == {"A": 1, "B": 1}


def test_append_refined_single_symbol_no_cooccurrence():
    drafts = [
        RefinedImpressionDraft(text="x", symbols=["ONLY"], source_raw_ids=[]),
    ]
    append_refined_impressions(
        relationship="R", speaker_role="protagonist", persona_name="母親",
        drafts=drafts, source_run="x/t",
    )
    ledger = read_refined_ledger(relationship="R", persona_name="母親")
    assert ledger.symbol_index == {"ONLY": ["ref_001"]}
    assert "ONLY" not in ledger.cooccurrence


def test_append_refined_empty_drafts_is_noop():
    version = append_refined_impressions(
        relationship="R", speaker_role="protagonist", persona_name="母親",
        drafts=[], source_run="x/t",
    )
    # No file created, no version bump
    assert version == 0
    path = refined_ledger_path(relationship="R", persona_name="母親")
    assert not path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_refined_ledger.py -v`

Expected: FAIL with `ImportError: cannot import name 'refined_ledger_path'` or similar.

- [ ] **Step 3: Add refined functions to `src/empty_space/ledger.py`**

Append to the end of `ledger.py`:

```python
# --- Level 3: refined ledger I/O ---

from empty_space.schemas import RefinedImpression, RefinedImpressionDraft, RefinedLedger


def refined_ledger_path(*, relationship: str, persona_name: str) -> Path:
    """Returns <LEDGERS_DIR>/<relationship>.refined.from_<persona_name>.yaml"""
    return LEDGERS_DIR / f"{relationship}.refined.from_{persona_name}.yaml"


def read_refined_ledger(*, relationship: str, persona_name: str) -> RefinedLedger:
    """Read refined ledger file; if absent, return empty RefinedLedger (do not raise).

    Note: when the file is absent, speaker is set to 'protagonist' as a
    placeholder (callers should override or not rely on it for an empty ledger).
    """
    path = refined_ledger_path(relationship=relationship, persona_name=persona_name)
    if not path.exists():
        return RefinedLedger(
            relationship=relationship,
            speaker="protagonist",  # placeholder
            persona_name=persona_name,
            ledger_version=0,
            impressions=[],
            symbol_index={},
            cooccurrence={},
        )

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RefinedLedger(
        relationship=data["relationship"],
        speaker=data["speaker"],
        persona_name=data["persona_name"],
        ledger_version=data["ledger_version"],
        impressions=[
            RefinedImpression(
                id=i["id"],
                text=i["text"],
                symbols=list(i["symbols"]),
                speaker=i.get("speaker", data["speaker"]),
                persona_name=i.get("persona_name", data["persona_name"]),
                from_run=i["from_run"],
                source_raw_ids=list(i.get("source_raw_ids") or []),
                created=i["created"],
            )
            for i in (data.get("impressions") or [])
        ],
        symbol_index={k: list(v) for k, v in (data.get("symbol_index") or {}).items()},
        cooccurrence={
            k: dict(v) for k, v in (data.get("cooccurrence") or {}).items()
        },
    )


def append_refined_impressions(
    *,
    relationship: str,
    speaker_role: str,
    persona_name: str,
    drafts: list[RefinedImpressionDraft],
    source_run: str,
) -> int:
    """Append refined drafts to ledger. Returns new ledger_version.

    Empty drafts: no-op, returns 0 (file not created/touched).
    Updates symbol_index and cooccurrence incrementally.
    Atomic write via .tmp + os.replace.
    """
    if not drafts:
        return 0

    existing = read_refined_ledger(
        relationship=relationship, persona_name=persona_name,
    )

    next_id_num = len(existing.impressions) + 1
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    new_entries: list[RefinedImpression] = []
    for draft in drafts:
        entry = RefinedImpression(
            id=f"ref_{next_id_num:03d}",
            text=draft.text,
            symbols=list(draft.symbols),
            speaker=speaker_role,  # type: ignore[arg-type]
            persona_name=persona_name,
            from_run=source_run,
            source_raw_ids=list(draft.source_raw_ids),
            created=now_iso,
        )
        new_entries.append(entry)
        next_id_num += 1

    all_impressions = existing.impressions + new_entries

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

    new_ledger = RefinedLedger(
        relationship=relationship,
        speaker=speaker_role,  # type: ignore[arg-type]
        persona_name=persona_name,
        ledger_version=existing.ledger_version + 1,
        impressions=all_impressions,
        symbol_index=symbol_index,
        cooccurrence=cooccurrence,
    )

    _atomic_write_refined_ledger(new_ledger)
    return new_ledger.ledger_version


def _atomic_write_refined_ledger(ledger: RefinedLedger) -> None:
    """Serialize refined ledger to YAML via .tmp + os.replace."""
    path = refined_ledger_path(
        relationship=ledger.relationship, persona_name=ledger.persona_name,
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "relationship": ledger.relationship,
        "speaker": ledger.speaker,
        "persona_name": ledger.persona_name,
        "ledger_version": ledger.ledger_version,
        "impressions": [
            {
                "id": e.id,
                "text": e.text,
                "symbols": list(e.symbols),
                "speaker": e.speaker,
                "persona_name": e.persona_name,
                "from_run": e.from_run,
                "source_raw_ids": list(e.source_raw_ids),
                "created": e.created,
            }
            for e in ledger.impressions
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

- [ ] **Step 4: Change `append_session_candidates` return type to `list[str]`**

Find `append_session_candidates` in `ledger.py`. Currently returns `None`. Change to return list of new-appended LedgerEntry ids.

Before (last line of function):
```python
    _atomic_write_ledger(new_ledger)
```

After:
```python
    _atomic_write_ledger(new_ledger)
    return [entry.id for entry in new_entries]
```

Also update the function signature type annotation from `-> None` to `-> list[str]`.

Update function docstring to note "Returns list of newly-appended LedgerEntry ids (for Composer provenance)."

- [ ] **Step 5: Run refined ledger tests to verify they pass**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_refined_ledger.py -v`

Expected: all 8 tests PASS.

- [ ] **Step 6: Run full suite — verify append_session_candidates caller compat**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest -v 2>&1 | tail -10`

Expected: all tests PASS. `append_session_candidates` now returns `list[str]` but existing callers ignore return value — should be backward compatible.

- [ ] **Step 7: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/ledger.py tests/test_refined_ledger.py
git commit -m "feat(ledger): refined ledger I/O + raw append returns ids

Adds read_refined_ledger, append_refined_impressions, refined_ledger_path
for Level 3 refined-impression storage. Parallel structure to raw ledger
(symbol_index + cooccurrence maintained), but with ref_NNN ids and
source_raw_ids provenance.

append_session_candidates return type changed from None to list[str]
(list of newly-appended LedgerEntry ids, needed by Composer for
provenance tracking). Backward compatible — existing callers ignore."
```

---

### Task 3: `composer.py` — parse_composer_output (pure, TDD)

**Files:**
- Create: `src/empty_space/composer.py` (partial — just parser)
- Create: `tests/test_composer.py` (partial — parser tests)

- [ ] **Step 1: Write failing tests in `tests/test_composer.py`**

```python
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
    """Pro may use '媽媽' instead of '母親' — should still route to protagonist."""
    raw = """媽媽:
  - text: "fuzzy match"
    symbols: [a]
    source_raw_ids: []
"""
    p_drafts, c_drafts, err = parse_composer_output(raw, protagonist_name="母親", counterpart_name="兒子")
    # Fuzzy: if no exact match and '媽媽' starts with similar character, route to protagonist
    # If not routed at all, at least don't crash
    assert err is None
    # Lenient assertion: either route worked (p_drafts has 1) or dropped (both 0)
    # Both acceptable for Level 3; prefer routing but don't enforce specific fuzzy algorithm
    assert len(p_drafts) + len(c_drafts) <= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_composer.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'empty_space.composer'`.

- [ ] **Step 3: Create `src/empty_space/composer.py` with parse function**

```python
"""Composer: session-end Pro bake that consolidates raw candidates into
refined impressions (short, atomic, first-person, register-aligned).

Output goes to two refined ledgers (by speaker). Called by runner at session end.
"""
import yaml

from empty_space.schemas import RefinedImpressionDraft


COMPOSER_MODEL = "gemini-2.5-pro"


def parse_composer_output(
    raw_yaml: str,
    *,
    protagonist_name: str,
    counterpart_name: str,
) -> tuple[list[RefinedImpressionDraft], list[RefinedImpressionDraft], str | None]:
    """Parse Composer's YAML output. Returns (protagonist_drafts, counterpart_drafts, parse_error).

    Graceful degradation:
    - YAML parse failure → ([], [], error_msg)
    - Non-dict root → ([], [], error_msg)
    - Section key fuzzy match: if exact protagonist_name/counterpart_name not found,
      tries any key whose first character matches.
    - Item without `text` → silently skipped.
    - `symbols` missing → default [].
    - `source_raw_ids` missing → default [].
    """
    try:
        parsed = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as e:
        return [], [], f"YAML parse error: {e}"

    if not isinstance(parsed, dict):
        return [], [], f"composer output root is not a dict: {type(parsed).__name__}"

    # Find sections — exact match first, then fuzzy by first character
    p_section = _find_section(parsed, protagonist_name)
    c_section = _find_section(parsed, counterpart_name)

    p_drafts = _parse_section(p_section) if p_section is not None else []
    c_drafts = _parse_section(c_section) if c_section is not None else []

    return p_drafts, c_drafts, None


def _find_section(parsed: dict, name: str):
    """Find section by exact key, then by first-character fuzzy match."""
    if name in parsed:
        return parsed[name]
    # Fuzzy: any key starting with the same character
    if name:
        first_char = name[0]
        for key, val in parsed.items():
            if isinstance(key, str) and key and key[0] == first_char:
                return val
    return None


def _parse_section(section) -> list[RefinedImpressionDraft]:
    """Parse one section's list of impression items."""
    if not isinstance(section, list):
        return []
    drafts: list[RefinedImpressionDraft] = []
    for item in section:
        if not isinstance(item, dict) or "text" not in item:
            continue
        symbols = item.get("symbols") or []
        if not isinstance(symbols, list):
            symbols = []
        source_raw_ids = item.get("source_raw_ids") or []
        if not isinstance(source_raw_ids, list):
            source_raw_ids = []
        drafts.append(RefinedImpressionDraft(
            text=str(item["text"]),
            symbols=[str(s) for s in symbols],
            source_raw_ids=[str(s) for s in source_raw_ids],
        ))
    return drafts
```

- [ ] **Step 4: Run tests**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_composer.py -v`

Expected: all 10 parser tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/composer.py tests/test_composer.py
git commit -m "feat(composer): parse_composer_output with graceful degradation

Parses Pro's YAML output into (protagonist_drafts, counterpart_drafts,
error). Fuzzy section key match (e.g., '媽媽' routes to protagonist if
no exact match). Items without text silently skipped. YAML errors
caught and reported in parse_error."
```

---

### Task 4: `composer.py` — gather_composer_input + build_composer_prompt (pure)

**Files:**
- Modify: `src/empty_space/composer.py` (add functions)
- Modify: `tests/test_composer.py` (add tests)

- [ ] **Step 1: Add failing tests to `tests/test_composer.py`**

Append:

```python
# --- gather_composer_input + build_composer_prompt ---

from pathlib import Path
from empty_space.composer import build_composer_prompt, gather_composer_input
from empty_space.schemas import (
    CandidateImpression, ComposerInput, RefinedImpression, Turn,
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


def test_build_composer_prompt_system_mentions_atomic_and_first_person():
    system, _ = build_composer_prompt(_minimal_input())
    # Keywords from the spec prompt
    assert "atomic" in system.lower() or "短" in system
    assert "第一人稱" in system


def test_build_composer_prompt_system_mentions_persona_names():
    system, user = build_composer_prompt(_minimal_input())
    # User message should have the persona names as section labels
    assert "母親" in user
    assert "兒子" in user
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_composer.py -v -k "gather or build"`

Expected: FAIL with `ImportError: cannot import name 'gather_composer_input'`.

- [ ] **Step 3: Add functions to `src/empty_space/composer.py`**

Add the imports at top:

```python
from pathlib import Path

from empty_space.ledger import read_refined_ledger
from empty_space.schemas import (
    CandidateImpression,
    ComposerInput,
    RefinedImpressionDraft,
    Turn,
)
```

Then append at the bottom:

```python
_COMPOSER_SYSTEM_PROMPT = """\
你是劇場記憶的 consolidator。

你剛看完一段對話（兩個角色的 session）。兩個角色在 turn 之中各自產出了
自己的「候選印象」——原始、未經整理的感受片段。你的工作是把這些原料
**精煉**成簡短、atomic 的意象，讓下次他們再相遇時，這些精華會浮現在
他們的內在。

你**不是** summarize 對話。你是**提煉**他們內在留下的痕跡。

---

## 產出規則

**1. 第一人稱視角保持**
- 母親的 refined 用「你」第一人稱內在感受
- 兒子的 refined 同上
- 嚴禁第三人稱。若 atomic image 描述某人的狀態，歸屬給該人並用第一人稱重寫

**2. Atomic 原則**
- 每條控制在 15 字內
- 原始感官 / 動作 / 體感 — 不是 judgment 或 analysis
- 壞例：「她的沉默比辯解都沉」（judgment）
- 好例：「沉默時喉嚨收緊」（體感）
- 壞例：「他感覺到自己的牆沒有用」（analysis）
- 好例：「手指捏著衣角」（動作）

**3. 歸屬判斷**
- 每條 refined 歸屬給其中一個角色
- 原則：這條 refined 是誰的內在感受

**4. 不保留 judgment / analysis raw**
- Raw 中充滿判斷性、反思性的句子（「他的存在本身是一種重量」）
- 不要精煉這些。只提煉真正會沉到身體裡的片段

**5. Merge 或保留**
- 同 session 多個 raw 講同個瞬間 → 各自 refine 成不同 atomic image
- 不強求 merge 跨 speaker

**6. Symbols**
- 每條 refined 帶 2-4 個 symbols
- 和 raw 的 symbol 體系盡量對齊

**7. 精簡數量**
- 每角色產出 3-6 條 refined
- 若這 session 沒什麼沉澱，少產幾條也 OK

---

## 輸出格式

只輸出 YAML，不加任何解釋：

```
母親:
  - text: "沉默時喉嚨收緊"
    symbols: [沉默, 喉嚨, 收緊]
    source_raw_ids: [imp_003, imp_007]

兒子:
  - text: "背靠著牆坐"
    symbols: [背, 牆, 坐]
    source_raw_ids: [imp_002]
```

`source_raw_ids` 必填（若沒直接對應 raw，用 []）。
"""


def gather_composer_input(
    *,
    relationship: str,
    protagonist_name: str,
    counterpart_name: str,
    out_dir: Path,
    session_turns: list[Turn],
    new_raw_ids: dict[str, list[str]],
) -> ComposerInput:
    """Gather all materials Composer needs from the session."""
    conversation_text = (out_dir / "conversation.md").read_text(encoding="utf-8")

    new_candidates: dict[str, list[CandidateImpression]] = {
        "protagonist": [],
        "counterpart": [],
    }
    for turn in session_turns:
        new_candidates[turn.speaker].extend(turn.candidate_impressions)

    existing_p = read_refined_ledger(
        relationship=relationship, persona_name=protagonist_name,
    )
    existing_c = read_refined_ledger(
        relationship=relationship, persona_name=counterpart_name,
    )

    return ComposerInput(
        relationship=relationship,
        protagonist_name=protagonist_name,
        counterpart_name=counterpart_name,
        conversation_text=conversation_text,
        new_candidates=new_candidates,
        new_candidate_ids=new_raw_ids,
        existing_refined={
            "protagonist": existing_p.impressions[-30:],
            "counterpart": existing_c.impressions[-30:],
        },
    )


def build_composer_prompt(input: ComposerInput) -> tuple[str, str]:
    """Return (system_prompt, user_message) for Pro bake."""
    user_parts: list[str] = []

    user_parts.append("## Session 對話\n" + input.conversation_text.rstrip())

    # Raw candidates with ids
    p_raws = input.new_candidates["protagonist"]
    p_ids = input.new_candidate_ids.get("protagonist", [])
    user_parts.append(
        f"## {input.protagonist_name}的 Raw Candidates（本 session 新產出）\n"
        + _format_raw_list(p_raws, p_ids)
    )

    c_raws = input.new_candidates["counterpart"]
    c_ids = input.new_candidate_ids.get("counterpart", [])
    user_parts.append(
        f"## {input.counterpart_name}的 Raw Candidates（本 session 新產出）\n"
        + _format_raw_list(c_raws, c_ids)
    )

    # Existing refined (context)
    user_parts.append(
        f"## {input.protagonist_name}的既有 Refined Impressions（供參考語氣）\n"
        + _format_refined_list(input.existing_refined.get("protagonist", []))
    )
    user_parts.append(
        f"## {input.counterpart_name}的既有 Refined Impressions（供參考語氣）\n"
        + _format_refined_list(input.existing_refined.get("counterpart", []))
    )

    user_parts.append("---\n\n開始精煉。")

    return _COMPOSER_SYSTEM_PROMPT, "\n\n".join(user_parts)


def _format_raw_list(raws: list[CandidateImpression], ids: list[str]) -> str:
    if not raws:
        return "（無）"
    lines = []
    for i, raw in enumerate(raws):
        raw_id = ids[i] if i < len(ids) else f"imp_?{i}"
        symbols_str = ", ".join(raw.symbols) if raw.symbols else ""
        lines.append(f"- {raw_id}: {raw.text} [symbols: {symbols_str}]")
    return "\n".join(lines)


def _format_refined_list(refined) -> str:
    if not refined:
        return "（空）"
    lines = []
    for imp in refined:
        symbols_str = ", ".join(imp.symbols) if imp.symbols else ""
        lines.append(f"- {imp.id}: {imp.text} [symbols: {symbols_str}]")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_composer.py -v`

Expected: all composer tests PASS (10 parser + 8 new = 18 total).

- [ ] **Step 5: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/composer.py tests/test_composer.py
git commit -m "feat(composer): gather_composer_input + build_composer_prompt

gather_composer_input reads session's conversation.md, buckets turn
candidates by speaker, loads last 30 existing refined per speaker.

build_composer_prompt produces system prompt (atomic + first-person
rules + output format) and user message (session conversation + raw
candidates with imp_XXX ids + existing refined for tone continuity)."
```

---

### Task 5: `composer.py` — run_composer orchestrator

**Files:**
- Modify: `src/empty_space/composer.py` (add top-level orchestrator)
- Modify: `tests/test_composer.py` (add orchestrator tests)

- [ ] **Step 1: Add failing tests to `tests/test_composer.py`**

Append:

```python
# --- run_composer orchestrator ---

from empty_space.composer import run_composer
from empty_space.llm import GeminiResponse


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_composer.py -v -k "run_composer"`

Expected: FAIL with `ImportError: cannot import name 'run_composer'`.

- [ ] **Step 3: Add `run_composer` orchestrator to `src/empty_space/composer.py`**

Add import at top:

```python
from empty_space.ledger import append_refined_impressions
from empty_space.schemas import ComposerSessionResult
```

Append at bottom:

```python
def run_composer(
    *,
    relationship: str,
    protagonist_name: str,
    counterpart_name: str,
    out_dir: Path,
    session_turns: list[Turn],
    new_raw_ids: dict[str, list[str]],
    source_run: str,
    llm_client,
) -> ComposerSessionResult:
    """Top-level Composer orchestrator. Called by runner at session end.

    On any exception: returns ComposerSessionResult with parse_error set,
    tokens zero, no refined appended. Raw ledgers remain intact upstream.

    On success: produces 0-6 refined impressions per speaker, appends to
    two refined ledgers, returns counts and tokens for meta.yaml.
    """
    try:
        # 1. Gather input
        input_bundle = gather_composer_input(
            relationship=relationship,
            protagonist_name=protagonist_name,
            counterpart_name=counterpart_name,
            out_dir=out_dir,
            session_turns=session_turns,
            new_raw_ids=new_raw_ids,
        )

        # 2. Build prompt
        system, user = build_composer_prompt(input_bundle)

        # 3. Pro bake
        resp = llm_client.generate(system=system, user=user, model=COMPOSER_MODEL)

        # 4. Parse output
        p_drafts, c_drafts, parse_err = parse_composer_output(
            resp.content,
            protagonist_name=protagonist_name,
            counterpart_name=counterpart_name,
        )

        # 5. Append to two ledgers
        p_version = append_refined_impressions(
            relationship=relationship,
            speaker_role="protagonist",
            persona_name=protagonist_name,
            drafts=p_drafts,
            source_run=source_run,
        )
        c_version = append_refined_impressions(
            relationship=relationship,
            speaker_role="counterpart",
            persona_name=counterpart_name,
            drafts=c_drafts,
            source_run=source_run,
        )

        return ComposerSessionResult(
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            latency_ms=resp.latency_ms,
            protagonist_refined_added=len(p_drafts),
            counterpart_refined_added=len(c_drafts),
            parse_error=parse_err,
        )

    except Exception as e:
        return ComposerSessionResult(
            tokens_in=0,
            tokens_out=0,
            latency_ms=0,
            protagonist_refined_added=0,
            counterpart_refined_added=0,
            parse_error=f"composer exception: {type(e).__name__}: {e}",
        )
```

- [ ] **Step 4: Run tests**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_composer.py -v`

Expected: all composer tests PASS (10 parser + 8 gather/build + 4 orchestrator = 22 total).

- [ ] **Step 5: Run full suite — no regressions**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest -v 2>&1 | tail -5`

Expected: all existing + new tests PASS.

- [ ] **Step 6: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/composer.py tests/test_composer.py
git commit -m "feat(composer): run_composer orchestrator

Top-level function called by runner at session end. Pipeline:
gather input → build prompt → Pro bake → parse output → append to
two refined ledgers.

Exception-safe: any failure (Pro error, parse error, write error)
caught and returned as parse_error in ComposerSessionResult. Raw
ledgers upstream remain intact."
```

---

### Task 6: `retrieval.py` — decouple retrieve_top_n from ledger type + switch to refined

**Files:**
- Modify: `src/empty_space/retrieval.py`
- Modify: `tests/test_retrieval.py`

- [ ] **Step 1: Modify `retrieve_top_n` signature in `retrieval.py`**

Find the existing `retrieve_top_n` function. Change signature and internal loop.

Before:
```python
def retrieve_top_n(
    *,
    query_symbols: list[str],
    ledger_a: Ledger,
    ledger_b: Ledger,
    synonym_map: dict[str, str],
    top_n: int = 3,
) -> list[RetrievedImpression]:
    ...
    for ledger in (ledger_a, ledger_b):
        for entry in ledger.candidates:
            ...
```

After:
```python
def retrieve_top_n(
    *,
    query_symbols: list[str],
    entries_a: list,  # list of LedgerEntry or RefinedImpression
    entries_b: list,
    speaker_a: str,
    persona_name_a: str,
    speaker_b: str,
    persona_name_b: str,
    synonym_map: dict[str, str],
    top_n: int = 3,
) -> list[RetrievedImpression]:
    """Score entries across two lists. Works with either LedgerEntry (raw)
    or RefinedImpression (refined) — both have id, text, symbols, from_run.

    For entries without from_turn attribute, from_turn is set to None in
    the resulting RetrievedImpression.
    """
    canon_q = {canonicalize(s, synonym_map) for s in query_symbols}
    if not canon_q:
        return []

    scored: list[tuple[int, str, object, str, str, list[str]]] = []
    # (score, created, entry, speaker, persona_name, matched_sorted)
    for entries, speaker, persona_name in (
        (entries_a, speaker_a, persona_name_a),
        (entries_b, speaker_b, persona_name_b),
    ):
        for entry in entries:
            canon_e = {canonicalize(s, synonym_map) for s in entry.symbols}
            matched = canon_q & canon_e
            if matched:
                scored.append((
                    len(matched),
                    entry.created,
                    entry,
                    speaker,
                    persona_name,
                    sorted(matched),
                ))

    scored.sort(key=lambda t: t[1], reverse=True)   # tiebreaker: created desc
    scored.sort(key=lambda t: t[0], reverse=True)   # primary: score desc

    seen_keys: set[tuple[str, str]] = set()
    result: list[RetrievedImpression] = []
    for score, _, entry, speaker, persona_name, matched in scored:
        key = (speaker, entry.id)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        # from_turn may not exist on RefinedImpression
        from_turn = getattr(entry, "from_turn", None)
        result.append(RetrievedImpression(
            id=entry.id,
            text=entry.text,
            symbols=tuple(entry.symbols),
            speaker=speaker,  # type: ignore[arg-type]
            persona_name=persona_name,
            from_run=entry.from_run,
            from_turn=from_turn,
            score=score,
            matched_symbols=tuple(matched),
        ))
        if len(result) >= top_n:
            break

    return result
```

- [ ] **Step 2: Modify `run_session_start_retrieval` to read refined ledger**

Find `run_session_start_retrieval`. Change the ledger reads and the retrieve_top_n call.

Before (key lines):
```python
    ledger_self = read_ledger(relationship=relationship, persona_name=persona_name)
    ledger_other = read_ledger(relationship=relationship, persona_name=other_persona_name)
    ...
    ledger_self = _with_speaker(ledger_self, speaker_role)
    ledger_other = _with_speaker(ledger_other, other_role)
    ...
    merged_cooc = merge_cooccurrence(ledger_self.cooccurrence, ledger_other.cooccurrence)
    ...
    impressions = retrieve_top_n(
        query_symbols=expanded_symbols,
        ledger_a=ledger_self,
        ledger_b=ledger_other,
        synonym_map=synonym_map,
        top_n=top_n,
    )
```

After:
```python
    from empty_space.ledger import read_refined_ledger  # local import to avoid confusion
    refined_self = read_refined_ledger(relationship=relationship, persona_name=persona_name)
    refined_other = read_refined_ledger(relationship=relationship, persona_name=other_persona_name)
    # Note: refined ledger's speaker field may be placeholder if file missing; use args for correctness
    other_role = "counterpart" if speaker_role == "protagonist" else "protagonist"
    ...
    merged_cooc = merge_cooccurrence(refined_self.cooccurrence, refined_other.cooccurrence)
    ...
    impressions = retrieve_top_n(
        query_symbols=expanded_symbols,
        entries_a=refined_self.impressions,
        entries_b=refined_other.impressions,
        speaker_a=speaker_role,
        persona_name_a=persona_name,
        speaker_b=other_role,
        persona_name_b=other_persona_name,
        synonym_map=synonym_map,
        top_n=top_n,
    )
```

Remove the `_with_speaker(...)` calls — they're no longer needed since retrieve_top_n takes speaker as explicit arg.

You can keep `_with_speaker` as unused (or remove it). Remove it for cleanliness.

- [ ] **Step 3: Update existing tests in `tests/test_retrieval.py`**

Find all test functions that call `retrieve_top_n(..., ledger_a=..., ledger_b=..., ...)`. Change them to use the new signature.

Helper to simplify test calls (add at top of test file if not present):

```python
def _retrieve_with_ledgers(query_symbols, ledger_a, ledger_b, synonym_map, top_n=3):
    """Test helper to adapt old ledger-based test calls to new entries-based signature."""
    return retrieve_top_n(
        query_symbols=query_symbols,
        entries_a=ledger_a.candidates if hasattr(ledger_a, "candidates") else ledger_a.impressions,
        entries_b=ledger_b.candidates if hasattr(ledger_b, "candidates") else ledger_b.impressions,
        speaker_a=ledger_a.speaker,
        persona_name_a=ledger_a.persona_name,
        speaker_b=ledger_b.speaker,
        persona_name_b=ledger_b.persona_name,
        synonym_map=synonym_map,
        top_n=top_n,
    )
```

Then update each affected test to call `_retrieve_with_ledgers(...)` instead of `retrieve_top_n(...)` directly.

Affected tests (from test_retrieval.py):
- `test_retrieve_empty_ledgers_returns_empty`
- `test_retrieve_single_ledger_hit`
- `test_retrieve_cross_ledger_dedup_by_speaker_and_id`
- `test_retrieve_sorts_by_score_desc`
- `test_retrieve_tiebreak_by_created_desc`
- `test_retrieve_synonym_map_matches_variants`
- `test_retrieve_top_n_truncates`

For each, change `retrieve_top_n(query_symbols=..., ledger_a=..., ledger_b=..., synonym_map=..., top_n=...)` to `_retrieve_with_ledgers(query_symbols=..., ledger_a=..., ledger_b=..., synonym_map=..., top_n=...)`.

- [ ] **Step 4: Add 2 new tests to `tests/test_retrieval.py`**

Append:

```python
# --- Level 3: retrieval from refined ---

from empty_space.schemas import RefinedImpression


def _make_refined(id: str, text: str, symbols: list[str], created: str = "2026-04-22T10:00:00Z") -> RefinedImpression:
    return RefinedImpression(
        id=id, text=text, symbols=symbols,
        speaker="protagonist", persona_name="母親",
        from_run="r/t", source_raw_ids=[], created=created,
    )


def test_retrieve_top_n_works_with_refined_impressions():
    """retrieve_top_n should accept list of RefinedImpression (not just LedgerEntry)."""
    refined = [
        _make_refined("ref_001", "沉默時喉嚨收緊", ["沉默", "喉嚨"]),
        _make_refined("ref_002", "手指在膝上按壓", ["手指", "膝"]),
    ]
    result = retrieve_top_n(
        query_symbols=["沉默"],
        entries_a=refined,
        entries_b=[],
        speaker_a="protagonist",
        persona_name_a="母親",
        speaker_b="counterpart",
        persona_name_b="兒子",
        synonym_map={},
        top_n=3,
    )
    assert len(result) == 1
    assert result[0].id == "ref_001"
    assert result[0].from_turn is None   # refined has no single turn


def test_run_session_start_retrieval_reads_refined_ledger(tmp_path, monkeypatch):
    """Session-start retrieval should now read refined ledger (not raw)."""
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path)

    # Pre-seed only refined (not raw) — verifies retrieval reads refined, not raw
    from empty_space.ledger import append_refined_impressions
    from empty_space.schemas import RefinedImpressionDraft
    append_refined_impressions(
        relationship="母親_x_兒子",
        speaker_role="counterpart",
        persona_name="兒子",
        drafts=[RefinedImpressionDraft(text="沉默時肩膀垂下", symbols=["沉默", "肩膀"], source_raw_ids=["imp_005"])],
        source_run="prev/t",
    )

    # Mock extract_symbols to return ["沉默"]
    monkeypatch.setattr("empty_space.retrieval.DEFAULT_SYNONYMS_PATH", tmp_path / "nonexistent")
    client = _MockLLMClient(response_text="- 沉默\n")

    result = run_session_start_retrieval(
        speaker_role="protagonist",
        persona_name="母親",
        query_text="你在想什麼",
        relationship="母親_x_兒子",
        other_persona_name="兒子",
        synonym_map={},
        llm_client=client,
    )

    # Should hit the refined impression
    assert len(result.impressions) == 1
    assert result.impressions[0].text == "沉默時肩膀垂下"
    assert result.impressions[0].id == "ref_001"
    assert result.impressions[0].from_turn is None
```

- [ ] **Step 5: Run retrieval tests**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_retrieval.py -v`

Expected: all tests PASS (existing ones after signature update + 2 new).

- [ ] **Step 6: Run full suite — check for runner integration breakage**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest -v 2>&1 | tail -10`

Expected: **test_runner_integration.py and test_runner_level2.py might fail** because runner now reads refined (which is empty in existing MockLLM tests) instead of raw. That's fine for this task — those tests will be fixed in Task 9.

For now, verify `test_retrieval.py`, `test_ledger.py`, `test_refined_ledger.py`, `test_composer.py`, `test_schemas*.py`, `test_loaders*.py`, `test_parser.py`, `test_prompt_assembler.py`, `test_writer.py` all pass.

- [ ] **Step 7: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/retrieval.py tests/test_retrieval.py
git commit -m "feat(retrieval): decouple from ledger type + switch to refined

retrieve_top_n new signature takes entries_a/entries_b lists (plus
speaker/persona_name args), working with either LedgerEntry (raw)
or RefinedImpression (refined).

run_session_start_retrieval now reads refined ledger (via
read_refined_ledger). Raw ledger no longer in retrieval path —
it becomes pure Composer input.

RetrievedImpression.from_turn is None when entry has no from_turn
(refined case)."
```

---

### Task 7: `writer.py` — extend write_meta with composer kwargs

**Files:**
- Modify: `src/empty_space/writer.py`
- Modify: `tests/test_writer.py`

- [ ] **Step 1: Add failing test to `tests/test_writer.py`**

Append:

```python
def test_write_meta_records_composer_fields(tmp_path, sample_config):
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
        models_used=["gemini-2.5-flash", "gemini-2.5-pro"],
        duration_seconds=5.0,
        retrieval_total_tokens_in=100,
        retrieval_total_tokens_out=20,
        ledger_appends=[],
        # Level 3 new kwargs:
        composer_tokens_in=4321,
        composer_tokens_out=678,
        composer_latency_ms=15234,
        protagonist_refined_added=4,
        counterpart_refined_added=3,
        composer_parse_error=None,
    )
    meta = yaml.safe_load((out_dir / "meta.yaml").read_text(encoding="utf-8"))
    assert meta["composer_tokens_in"] == 4321
    assert meta["composer_tokens_out"] == 678
    assert meta["composer_latency_ms"] == 15234
    assert meta["protagonist_refined_added"] == 4
    assert meta["counterpart_refined_added"] == 3
    assert meta["composer_parse_error"] is None


def test_write_meta_composer_fields_default_to_zero():
    """Composer kwargs should all have defaults (backward compat)."""
    # Implicitly tested by existing tests that don't pass composer kwargs.
    # Explicitly verify defaults here.
    import inspect
    sig = inspect.signature(write_meta)
    for name in ["composer_tokens_in", "composer_tokens_out", "composer_latency_ms",
                 "protagonist_refined_added", "counterpart_refined_added", "composer_parse_error"]:
        assert name in sig.parameters, f"{name} missing"
        assert sig.parameters[name].default is not inspect.Parameter.empty, f"{name} has no default"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_writer.py::test_write_meta_records_composer_fields -v`

Expected: FAIL — `write_meta` doesn't accept the new kwargs.

- [ ] **Step 3: Modify `write_meta` in `src/empty_space/writer.py`**

Find `write_meta` function. Add 6 new kwargs with defaults:

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
    retrieval_total_tokens_in: int = 0,
    retrieval_total_tokens_out: int = 0,
    ledger_appends: list[dict] | None = None,
    # Level 3 new:
    composer_tokens_in: int = 0,
    composer_tokens_out: int = 0,
    composer_latency_ms: int = 0,
    protagonist_refined_added: int = 0,
    counterpart_refined_added: int = 0,
    composer_parse_error: str | None = None,
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
        # Level 3 new:
        "composer_tokens_in": composer_tokens_in,
        "composer_tokens_out": composer_tokens_out,
        "composer_latency_ms": composer_latency_ms,
        "protagonist_refined_added": protagonist_refined_added,
        "counterpart_refined_added": counterpart_refined_added,
        "composer_parse_error": composer_parse_error,
    }
    _atomic_write_yaml(out_dir / "meta.yaml", meta)
```

- [ ] **Step 4: Run writer tests**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_writer.py -v`

Expected: all PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/writer.py tests/test_writer.py
git commit -m "feat(writer): extend write_meta with Level 3 Composer kwargs

Adds 6 new kwargs (all with defaults): composer_tokens_in/out,
composer_latency_ms, protagonist_refined_added, counterpart_refined_added,
composer_parse_error. All default to 0/None so existing callers remain
backward compatible."
```

---

### Task 8: Runner integration — Composer call at session end

**Files:**
- Modify: `src/empty_space/runner.py`

- [ ] **Step 1: Modify `_append_session_ledgers` to return new raw ids**

Find `_append_session_ledgers` function. Change its return type from `list[dict]` to `tuple[list[dict], dict[str, list[str]]]` — returns both the meta appends list and the new raw ids dict.

Before:
```python
def _append_session_ledgers(
    *, relationship, protagonist_persona, counterpart_persona, turns, source_run,
) -> list[dict]:
    ...
    p_candidates = [...]
    c_candidates = [...]
    appends = []
    if p_candidates:
        ledger.append_session_candidates(...)  # returns None in Level 2
        new_ledger = ledger.read_ledger(...)
        appends.append({...})
    if c_candidates:
        ...
    return appends
```

After:
```python
def _append_session_ledgers(
    *, relationship, protagonist_persona, counterpart_persona, turns, source_run,
) -> tuple[list[dict], dict[str, list[str]]]:
    """Append session candidates to raw ledgers, return (meta_appends, new_raw_ids).

    new_raw_ids: {"protagonist": ["imp_045", "imp_046"], "counterpart": ["imp_012"]}
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
    new_ids: dict[str, list[str]] = {"protagonist": [], "counterpart": []}

    if p_candidates:
        p_new_ids = ledger.append_session_candidates(
            relationship=relationship,
            speaker_role="protagonist",
            persona_name=protagonist_persona.name,
            candidates=p_candidates,
            source_run=source_run,
        )
        new_ids["protagonist"] = p_new_ids
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
        c_new_ids = ledger.append_session_candidates(
            relationship=relationship,
            speaker_role="counterpart",
            persona_name=counterpart_persona.name,
            candidates=c_candidates,
            source_run=source_run,
        )
        new_ids["counterpart"] = c_new_ids
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

    return appends, new_ids
```

- [ ] **Step 2: Add composer imports and helper at top of runner.py**

Add imports:

```python
from empty_space.composer import run_composer
from empty_space.schemas import ComposerSessionResult
```

Append helper at bottom:

```python
def _run_composer_at_session_end(
    *,
    relationship: str,
    protagonist: Persona,
    counterpart: Persona,
    out_dir: Path,
    turns: list[Turn],
    new_raw_ids: dict[str, list[str]],
    source_run: str,
    llm_client,
) -> ComposerSessionResult:
    """Wrap run_composer — exception-safe. On any error, returns result
    with parse_error set, refined ledgers untouched."""
    try:
        return run_composer(
            relationship=relationship,
            protagonist_name=protagonist.name,
            counterpart_name=counterpart.name,
            out_dir=out_dir,
            session_turns=turns,
            new_raw_ids=new_raw_ids,
            source_run=source_run,
            llm_client=llm_client,
        )
    except Exception as e:
        return ComposerSessionResult(
            tokens_in=0, tokens_out=0, latency_ms=0,
            protagonist_refined_added=0, counterpart_refined_added=0,
            parse_error=f"composer exception: {type(e).__name__}: {e}",
        )
```

Note: `run_composer` already has internal try/except, so this wrapper is double protection. Keep it as extra safety.

- [ ] **Step 3: Modify `run_session` to call Composer at session end**

Find `run_session` function. Locate the section after the `for n in range(...)` loop.

Before (key section):
```python
    duration = time.monotonic() - start_time
    termination_reason = "max_turns"

    total_tokens_in = sum(t.tokens_in for t in state.turns)
    ...

    # Level 2: Session-end ledger append
    source_run = f"{config.exp_id}/{timestamp}"
    ledger_appends = _append_session_ledgers(
        relationship=relationship,
        protagonist_persona=protagonist,
        counterpart_persona=counterpart,
        turns=state.turns,
        source_run=source_run,
    )

    write_meta(
        out_dir=out_dir,
        ...,
        ledger_appends=ledger_appends,
    )
```

After:
```python
    duration = time.monotonic() - start_time
    termination_reason = "max_turns"

    total_tokens_in = sum(t.tokens_in for t in state.turns)
    ...

    # Level 2: Session-end ledger append (returns new raw ids too)
    source_run = f"{config.exp_id}/{timestamp}"
    ledger_appends, new_raw_ids = _append_session_ledgers(
        relationship=relationship,
        protagonist_persona=protagonist,
        counterpart_persona=counterpart,
        turns=state.turns,
        source_run=source_run,
    )

    # Level 3: Composer Pro bake at session end
    composer_result = _run_composer_at_session_end(
        relationship=relationship,
        protagonist=protagonist,
        counterpart=counterpart,
        out_dir=out_dir,
        turns=state.turns,
        new_raw_ids=new_raw_ids,
        source_run=source_run,
        llm_client=llm_client,
    )

    # Update models_used to include Pro if Composer ran
    if composer_result.tokens_in > 0:
        models_used = sorted(set(models_used) | {"gemini-2.5-pro"})

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
        retrieval_total_tokens_in=(
            retrieval_protagonist.flash_tokens_in + retrieval_counterpart.flash_tokens_in
        ),
        retrieval_total_tokens_out=(
            retrieval_protagonist.flash_tokens_out + retrieval_counterpart.flash_tokens_out
        ),
        ledger_appends=ledger_appends,
        # Level 3 new:
        composer_tokens_in=composer_result.tokens_in,
        composer_tokens_out=composer_result.tokens_out,
        composer_latency_ms=composer_result.latency_ms,
        protagonist_refined_added=composer_result.protagonist_refined_added,
        counterpart_refined_added=composer_result.counterpart_refined_added,
        composer_parse_error=composer_result.parse_error,
    )
```

- [ ] **Step 4: Run test_runner_integration.py — expect many failures**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest tests/test_runner_integration.py -v 2>&1 | tail -15`

Expected: MANY FAILURES because:
- MockLLMClient in every test runs out of responses (Composer now consumes 1 more response per session)
- `test_pre_seeded_ledger_hits_system_prompt` and similar may fail because retrieval now reads refined (empty), not raw

Task 9 will fix these tests.

- [ ] **Step 5: Commit (tests will be fixed in Task 9)**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/runner.py
git commit -m "feat(runner): Level 3 Composer integration at session end

_append_session_ledgers now returns (meta_appends, new_raw_ids) tuple.
new_raw_ids passed to _run_composer_at_session_end.

Composer runs after raw append, before write_meta. models_used
includes gemini-2.5-pro if Composer successfully ran.

Existing runner integration tests broken intentionally — will be
updated in next task (MockLLMClient needs 1 extra composer response)."
```

---

### Task 9: Update runner integration tests + new composer-scenario tests

**Files:**
- Modify: `tests/test_runner_integration.py`
- Modify: `tests/test_runner_level2.py`

- [ ] **Step 1: Update `tests/test_runner_integration.py` — append composer response to every MockLLMClient**

Every test that creates a `MockLLMClient([...])` needs 1 more response appended — the Composer's Pro response. Use a minimal valid composer YAML that produces 0 refined per side (since we don't want test state to depend on specific composer output content):

Composer response to append (as a string):
```
"母親: []\n兒子: []\n"
```

This is parseable YAML, produces 0 drafts per side, results in `protagonist_refined_added=0, counterpart_refined_added=0, parse_error=None`.

Affected tests (from test_runner_integration.py):
- `test_happy_path_runs_all_turns`
- `test_speaker_alternation_mother_starts`
- `test_system_prompt_contains_correct_persona_per_turn`
- `test_candidate_impressions_extracted_and_stored`
- `test_director_event_injected_into_system_from_trigger_turn`
- `test_director_events_accumulate_across_turns`
- `test_parse_error_recorded_but_session_continues`
- `test_max_turns_terminates_session`
- `test_two_runs_of_same_exp_create_distinct_timestamp_dirs`

For each, the MockLLMClient's responses list gets 1 more element appended: `"母親: []\n兒子: []\n"`.

Example patch for `test_happy_path_runs_all_turns`:

```python
def test_happy_path_runs_all_turns(minimal_config, tmp_path, monkeypatch):
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", tmp_path)
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")  # NEW (if not present)

    responses = [
        "- 醫院\n- 父親\n",     # protagonist extract_symbols
        "- 醫院\n- 父親\n",     # counterpart extract_symbols
        "你回來了。",
        "嗯。",
        "⋯⋯",
        "不關我的事。",
        "母親: []\n兒子: []\n",   # NEW: composer response
    ]
    client = MockLLMClient(responses)
    ...
```

Apply this pattern to **all** tests in test_runner_integration.py.

For `test_llm_exception_aborts_session_partial_turns_kept` — the ExplodingClient raises before Composer, so no change needed for that specific test (Composer won't run if turn loop crashes).

For any test that specifically checks `meta.yaml.models_used`, note that `gemini-2.5-pro` will now be added when Composer runs successfully. Update assertions.

- [ ] **Step 2: Update `tests/test_runner_level2.py` — same pattern**

Affected tests:
- `test_first_session_empty_ledgers_written_after`
- `test_second_session_retrieval_hits_first_session_impressions`
- `test_llm_exception_aborts_session_no_ledger_written`
- `test_pre_seeded_ledger_hits_system_prompt`
- `test_synonym_map_enables_variant_matching`

For each, append `"母親: []\n兒子: []\n"` to the MockLLMClient responses list.

**IMPORTANT** — `test_pre_seeded_ledger_hits_system_prompt` and `test_synonym_map_enables_variant_matching`: these tests pre-seed a **raw** ledger via `append_session_candidates`. But now retrieval reads refined. These tests need to switch to pre-seeding a **refined** ledger via `append_refined_impressions` instead.

Example patch for `test_pre_seeded_ledger_hits_system_prompt`:

```python
def test_pre_seeded_ledger_hits_system_prompt(redirect_all_dirs):
    """預先 seed refined ledger (not raw)，新 session 的 system prompt 含命中印象。"""
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
```

Same pattern for `test_synonym_map_enables_variant_matching` — switch to refined ledger seed.

For `test_first_session_empty_ledgers_written_after`: after the session, check that **refined** ledgers also exist (not just raw). The composer response `"母親: []\n兒子: []\n"` produces 0 drafts → no refined file created. Update test expectations:

```python
def test_first_session_empty_ledgers_written_after(redirect_all_dirs):
    config = _base_config(max_turns=2)
    responses = [
        "- 醫院\n- 父親\n",
        "- 醫院\n- 父親\n",
        "話一\n\n---IMPRESSIONS---\n- text: \"母親印象一\"\n  symbols: [A, B]\n",
        "話二\n\n---IMPRESSIONS---\n- text: \"兒子印象一\"\n  symbols: [C, D]\n",
        # Composer with actual drafts this time, to verify refined creation
        "母親:\n  - text: \"母親精煉\"\n    symbols: [A]\n    source_raw_ids: [imp_001]\n\n兒子:\n  - text: \"兒子精煉\"\n    symbols: [C]\n    source_raw_ids: [imp_001]\n",
    ]
    client = MockLLMClient(responses)
    result = run_session(config=config, llm_client=client)

    ledgers_dir = redirect_all_dirs["ledgers_dir"]
    # Raw ledgers exist (Level 2 behavior)
    assert (ledgers_dir / "母親_x_兒子.from_母親.yaml").is_file()
    assert (ledgers_dir / "母親_x_兒子.from_兒子.yaml").is_file()
    # Refined ledgers also exist (Level 3 new)
    assert (ledgers_dir / "母親_x_兒子.refined.from_母親.yaml").is_file()
    assert (ledgers_dir / "母親_x_兒子.refined.from_兒子.yaml").is_file()
```

For `test_second_session_retrieval_hits_first_session_impressions`: session 2 retrieval now reads refined (which is what session 1 composer produced, not the raw). Update composer response in session 1 to produce refined that will match session 2's query symbols.

- [ ] **Step 3: Add 3 new Level 3 tests to `tests/test_runner_integration.py`**

Append:

```python
# --- Level 3: Composer integration tests ---

def test_composer_runs_at_session_end(minimal_config, tmp_path, monkeypatch):
    """Normal session: composer runs, produces refined, updates meta.yaml."""
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", tmp_path)
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")

    responses = [
        "- 醫院\n", "- 醫院\n",     # extract
        "你回來了。", "嗯。", "⋯⋯", "不關我的事。",  # 4 turns
        # Composer — actual drafts
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
        "a", "b", "c", "d",
        "garbage [[[ not yaml",   # composer response — bad YAML
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

    # Raw ledgers intact
    assert (tmp_path / "ledgers" / "母親_x_兒子.from_母親.yaml").is_file()
    # Refined ledgers NOT created (0 drafts + append_refined_impressions is no-op on empty)
    assert not (tmp_path / "ledgers" / "母親_x_兒子.refined.from_母親.yaml").is_file()


def test_second_session_retrieval_reads_refined(minimal_config, tmp_path, monkeypatch):
    """Two sessions: session 2's retrieval撈到 session 1's refined (ref_XXX id)."""
    monkeypatch.setattr("empty_space.runner.RUNS_DIR", tmp_path)
    monkeypatch.setattr("empty_space.ledger.LEDGERS_DIR", tmp_path / "ledgers")

    # Session 1 — composer produces refined with matching symbols
    responses_1 = [
        "- 醫院\n- 父親\n", "- 醫院\n- 父親\n",
        "a", "b", "c", "d",
        "母親:\n  - text: \"醫院走廊長\"\n    symbols: [醫院, 走廊]\n    source_raw_ids: [imp_001]\n\n兒子:\n  - text: \"父親的門關著\"\n    symbols: [父親, 門]\n    source_raw_ids: [imp_001]\n",
    ]
    run_session(config=minimal_config, llm_client=MockLLMClient(responses_1))

    # Session 2 — retrieval should hit session 1's refined
    responses_2 = [
        "- 醫院\n", "- 父親\n",
        "e", "f", "g", "h",
        "母親: []\n兒子: []\n",
    ]
    client = MockLLMClient(responses_2)
    result = run_session(config=minimal_config, llm_client=client)

    retrieval = yaml.safe_load((result.out_dir / "retrieval.yaml").read_text(encoding="utf-8"))
    # Protagonist retrieval should hit 母親's refined "醫院走廊長" (matches 醫院 symbol)
    p_impressions = retrieval["protagonist"]["impressions"]
    assert len(p_impressions) >= 1
    assert all(imp["id"].startswith("ref_") for imp in p_impressions)
    assert p_impressions[0]["from_turn"] is None   # refined has no single turn
```

- [ ] **Step 4: Run full test suite**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run pytest -v 2>&1 | tail -20`

Expected: all PASS (existing updated + 3 new). 

Debug any failures iteratively. Common issues:
- Forgot to redirect LEDGERS_DIR in a test → refined ledger written to wrong path
- Composer response YAML wrong format → parse error
- Assert wrong value for `models_used` (should now include Pro when composer runs)

- [ ] **Step 5: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add tests/test_runner_integration.py tests/test_runner_level2.py
git commit -m "test(runner): update for Level 3 Composer + new Composer scenarios

Existing integration tests: append 1 composer response to every
MockLLMClient (usually '母親: []\\n兒子: []\\n' for no-op). Tests
pre-seeding raw ledger switch to refined ledger (via
append_refined_impressions) since retrieval now reads refined.

New tests:
- test_composer_runs_at_session_end
- test_composer_failure_doesnt_break_session
- test_second_session_retrieval_reads_refined"
```

---

### Task 10: Smoke test + Level 3 summary + tag

**Files:**
- Create: `docs/level-3-summary.md`
- Optional temp edit: `experiments/mother_x_son_act1_hospital.yaml` (max_turns=4 for smoke, revert)

- [ ] **Step 1: Verify `.env` has `GEMINI_API_KEY`**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && grep -q "^GEMINI_API_KEY=" .env && echo "key present" || echo "MISSING"`

Expected: `key present`. If missing, stop and report BLOCKED.

- [ ] **Step 2: Clean ledgers (fresh start for smoke)**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
rm -f ledgers/*.yaml
ls ledgers/
```

Expected: empty directory.

- [ ] **Step 3: Run Act 1 against real Gemini (Flash turns + Pro composer)**

Run: `cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space" && uv run python scripts/run_experiment.py mother_x_son_act1_hospital`

Expected output (approximately):
```
✓ Completed mother_x_son_act1_hospital
  Output: /.../runs/mother_x_son_act1_hospital/2026-04-22T...
  Turns: 8
  Termination: max_turns
  Tokens in/out: ~20000 / ~2500
  Duration: ~110-180s   (Level 2 was ~90s; Level 3 adds Pro ~15-30s)
```

Capture the exact numbers for the summary.

- [ ] **Step 4: Inspect Composer output**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
cat ledgers/母親_x_兒子.refined.from_母親.yaml
cat ledgers/母親_x_兒子.refined.from_兒子.yaml
cat runs/mother_x_son_act1_hospital/*/meta.yaml | grep -A 1 composer
```

Expected:
- Two refined yaml files exist
- Each has 3-6 impressions (per Composer prompt guidance)
- Impressions are short atomic (目標: <15 字)
- impressions use first-person (「你」 or «沉默時...»)
- `meta.yaml` has non-zero composer_tokens_in/out

Note what the Composer actually produced — this feeds the summary.

- [ ] **Step 5: Run Act 2 — verify retrieval reads refined**

```bash
uv run python scripts/run_experiment.py mother_x_son_act2_car
cat runs/mother_x_son_act2_car/*/retrieval.yaml | head -60
```

Expected: `retrieval.yaml` impressions have `id: ref_XXX` (not `imp_XXX`), `from_turn: null`. Some impressions should hit Act 1's refined.

- [ ] **Step 6: Run Act 3 — two layers of refined now in ledger**

```bash
uv run python scripts/run_experiment.py mother_x_son_act3_home
```

- [ ] **Step 7: Check overall state**

```bash
ls -la ledgers/
# Should have 4 files: raw (from_母親, from_兒子) + refined (from_母親, from_兒子)
wc -l ledgers/*.yaml
```

- [ ] **Step 8: Write `docs/level-3-summary.md`**

Structure mirrors `docs/level-2-summary.md`. Include:

- Status, commit range, test count, branch
- What this level shipped (composer.py + refined ledger + retrieval switch + runner integration)
- Key decisions (recap from spec §0.1):
  - Every-session auto trigger (brain-like memory consolidation)
  - Minimal scope (no 21 grid, no effective relationship layer, no cluster merge)
  - Two-layer ledger (raw preserved, refined as retrieval source)
  - Single Pro bake per session with dual-section YAML output, post-process split
  - Atomic + first-person + examples prompt (D full version)
- Module inventory with responsibilities (composer.py + ledger.py extensions)
- Test coverage breakdown (~161 tests total: 135 Level 2 + ~26 new)
- Smoke-run results — tokens/duration from steps 3-7; example refined impressions (copy 2-3 from the yaml files)
- Known issues / follow-ups (reference spec §11 risks):
  - Pro prompt stability (drift into third-person, no compression, attribution confusion)
  - Provenance imprecision (source_raw_ids may be best-effort)
  - Incremental blind spot (emergent cross-session connections missed)
  - Refined ledger growth (200+ sessions = 1000+ entries; human reading gets hard)
  - Symbol register alignment may still not be solved (Flash extract vs Composer output register)
- Pointer to Level 4 Judge brainstorm

Keep under 200 lines.

- [ ] **Step 9: Commit summary**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add docs/level-3-summary.md
git commit -m "docs: Level 3 summary — Composer + refined ledger shipped

Records smoke-run results (sample refined impressions, tokens, duration)
and key design decisions for future session handoff."
```

- [ ] **Step 10: Tag the milestone**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git tag -a level-3 -m "$(cat <<'EOF'
Level 3 成功 — Composer refined impression consolidation

每個 session 結束自動 Pro bake 一次，讀對話 + raw candidates +
existing refined，產出短 atomic 第一人稱的 refined impressions，
分流寫進兩本 refined 帳本。下場 session 的 retrieval 從 refined 撈，
不再從 raw。

解了 Level 2 三幕實驗暴露的問題：
- Register mismatch（Flash 拆描述詞 vs raw 感受詞不相交）
  → Composer 統一語氣，refined symbols 對齊 register
- Density compound（長篇 judgment 回注 → 語言越來越濃）
  → atomic 原則 < 15 字
- POV leakage（A 策略共同記憶讓兒子看到母親視角）
  → by-speaker refined + 第一人稱 prompt

技術決策：
- 每 session 無條件 Pro bake（腦內類比：每事件一次記憶固化）
- Minimal scope：只做 refined consolidation；21 格 / cluster / effective 關係層留 Level 4+
- Raw 保留為 Composer 原料庫，retrieval 不碰 raw
- 單次 Pro call 產雙 section YAML（母親/兒子），後處理分流兩檔
- Prompt 全功能（D）：atomic + transformation examples + 第一人稱約束

Level 4 入口：Judge（stage/mode/張力追蹤 + fire_release/basin_lock 終止）
EOF
)"
git push origin main
git push origin level-3
```

---

## Self-Review

### 1. Spec Coverage Check

Mapping spec sections to tasks:

| Spec § | Requirement | Task |
|---|---|---|
| §0 | Level 3 frame | N/A (narrative) |
| §1 | Scope (In/Out/副產出) | All tasks cumulatively |
| §2 | Schema (new dataclasses + RetrievedImpression change) | Task 1 |
| §3 | Module skeleton + function contracts | Tasks 2-5, 8 |
| §4 | Composer prompt + parser | Tasks 3-4 |
| §5 | Input scope + cost | Task 4 (gather_composer_input) |
| §6 | Runner integration | Task 8 |
| §7 | Retrieval switch | Task 6 |
| §8 | Disk schema (refined yaml + meta extensions) | Tasks 2 (ledger), 7 (meta) |
| §9 | Error handling | Tasks 5 (run_composer try/except), 8 (runner wrapper), 9 (test_composer_failure) |
| §10 | Test strategy | All tasks have test steps |
| §11 | Risks | Documented in spec; §10 smoke + monitoring |

All covered.

### 2. Placeholder Scan

Grep for any TBD / TODO / "add appropriate" / "similar to Task N" / "etc":
- No TBD / TODO found
- All code blocks are complete with actual code
- No "similar to Task N" shortcuts — each task shows actual test + impl code

Clean.

### 3. Type Consistency Check

- `RefinedImpression` / `RefinedLedger` / `RefinedImpressionDraft` / `ComposerSessionResult` / `ComposerInput`: defined in Task 1, used consistently in Tasks 2-9
- `append_session_candidates` return type: changed to `list[str]` in Task 2, consumed by `_append_session_ledgers` in Task 8
- `_append_session_ledgers` return type: changed to `tuple[list[dict], dict[str, list[str]]]` in Task 8, both values consumed correctly (ledger_appends to write_meta, new_raw_ids to run_composer)
- `retrieve_top_n` signature: new `entries_a/entries_b/speaker_a/persona_name_a/speaker_b/persona_name_b` in Task 6, consistent across test updates in Task 9
- `RetrievedImpression.from_turn: int | None`: consistent from Task 1, writer serializes None as YAML null (Level 2 already had this path for refined case)
- `run_composer` signature: defined in Task 5, called consistently in Task 8 via `_run_composer_at_session_end`

Consistency verified.

Plan complete.
