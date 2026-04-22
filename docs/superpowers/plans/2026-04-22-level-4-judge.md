# Level 4 Judge 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 為每個 speaker 獨立維護 (stage, mode) 狀態機，讓 system prompt 的「此刻」隨對話演化，並在情緒峰值提供導演互動介入入口。

**Architecture:** 新增 `judge.py` 模組封裝 Flash-based Judge 的 prompt/parser/ratchet/呼叫。`runner.py` 每輪結束後對雙方各跑一次 Judge，更新各自 `JudgeState`。`prompt_assembler.py` 的 `## 此刻` 區塊從靜態 `initial_state` 改讀該 speaker 的當前 JudgeState + `stage_mode_contexts_v3` 對應格子。`--interactive` flag 開啟時，verdict 為 fire_release / basin_lock 時阻塞 stdin 讓導演注入事件（重用 Level 1 的 `director_events` 機制）。

**Tech Stack:** Python 3.11+, pydantic v2, pytest, Gemini Flash via `GeminiClient`（既有）, yaml, dataclasses。

**Spec:** `docs/superpowers/specs/2026-04-22-level-4-judge.md`

---

## 檔案結構 (File Structure)

| 檔案 | 角色 | 新/改 |
|---|---|---|
| `src/empty_space/judge.py` | Judge prompt + parser + ratchet + run_judge | NEW |
| `src/empty_space/schemas.py` | +JudgeState, +JudgeResult, +Persona v3 欄位, +InitialState validator, +SessionResult v4 欄位 | MODIFY |
| `src/empty_space/loaders.py` | Persona loader 讀 v3 檔案 | MODIFY |
| `src/empty_space/prompt_assembler.py` | `## 此刻` 改讀 judge_state + cell context | MODIFY |
| `src/empty_space/runner.py` | init states / post-turn Judge / termination / interactive hook | MODIFY |
| `src/empty_space/writer.py` | turn yaml judge_output + meta judge_trajectories/health | MODIFY |
| `scripts/run_experiment.py` | `--interactive` flag | MODIFY |
| `scripts/smoke_level4.py` | 真 Flash smoke | NEW |
| `tests/test_judge.py` | 單元測試 | NEW |
| `tests/test_runner_level4.py` | 整合測試 | NEW |
| `experiments/mother_x_son_*.yaml` | v3 詞彙遷移 mode: 基線 → 在 | MODIFY |

---

## 任務一覽 (Tasks)

1. Schemas foundation — JudgeState, JudgeResult, Persona fields, SessionResult fields
2. judge.py 常數 + `parse_judge_principles` + `parse_stage_mode_contexts`
3. judge.py `apply_stage_target` ratchet 閘門
4. judge.py `parse_judge_output` 寬容解析
5. judge.py `build_judge_prompt` + `run_judge` + 工具函式
6. loaders.py 讀取 v3 檔案
7. prompt_assembler.py 動態 `## 此刻`
8. runner.py 初始化 JudgeState + InitialState validator
9. runner.py 每輪結束跑雙 Judge
10. writer.py turn yaml judge_output 區塊
11. writer.py session meta judge_trajectories / judge_health / termination
12. runner.py dual_basin_lock termination
13. runner.py 互動 peak hook
14. `scripts/run_experiment.py` 加 `--interactive` flag
15. 既有 experiment yaml 的 v3 詞彙遷移
16. `tests/test_runner_level4.py` 整合測試
17. `scripts/smoke_level4.py` + 手動驗證

---

### Task 1: Schemas foundation

**Files:**
- Modify: `src/empty_space/schemas.py`
- Test: `tests/test_schemas_level4.py` (create new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_schemas_level4.py`:

```python
"""Level 4 schema additions: JudgeState, JudgeResult, Persona v3 fields,
SessionResult extensions, InitialState v3 vocabulary validator.
"""
import pytest

from empty_space.schemas import (
    ExperimentConfig,
    InitialState,
    JudgeResult,
    JudgeState,
    Persona,
    PersonaRef,
    SettingRef,
    Termination,
)


def test_judge_state_defaults():
    s = JudgeState(speaker_role="protagonist", stage="前置積累", mode="在")
    assert s.last_why == ""
    assert s.last_verdict == ""
    assert s.move_history == []
    assert s.verdict_history == []
    assert s.hits_history == []


def test_judge_result_defaults():
    r = JudgeResult(
        proposed_stage="前置積累",
        proposed_mode="收",
        proposed_verdict="N/A",
        why="",
        hits=[],
        meta={},
    )
    assert r.proposed_verdict == "N/A"


def test_persona_v3_fields_default_empty():
    p = Persona(name="母親", version="v3_tension", core_text="...")
    assert p.judge_principles_text == ""
    assert p.stage_mode_contexts_parsed == {}


def test_persona_v3_fields_assignable():
    p = Persona(
        name="母親",
        version="v3_tension",
        core_text="...",
        judge_principles_text="鯨的下潛...",
        stage_mode_contexts_parsed={
            "前置積累_收": {"身體傾向": "鯨的下潛", "語聲傾向": "極短", "注意力": "內收"},
        },
    )
    assert "鯨" in p.judge_principles_text
    assert p.stage_mode_contexts_parsed["前置積累_收"]["身體傾向"] == "鯨的下潛"


def test_initial_state_accepts_v3_mode():
    s = InitialState(verb="承受", stage="前置積累", mode="在")
    assert s.mode == "在"


def test_initial_state_migrates_legacy_baseline_to_在():
    """Legacy mode='基線' auto-migrated to '在' (軟遷移)."""
    s = InitialState(verb="承受", stage="前置積累", mode="基線")
    assert s.mode == "在"


def test_initial_state_legacy_migration_preserves_other_modes():
    s = InitialState(verb="承受", stage="前置積累", mode="收")
    assert s.mode == "收"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_schemas_level4.py -v`
Expected: FAIL — `ImportError: cannot import name 'JudgeState'` / `JudgeResult`

- [ ] **Step 3: Add JudgeState + JudgeResult dataclasses to schemas.py**

Append to `src/empty_space/schemas.py` (after existing dataclasses, before `ComposerInput`):

```python
# --- Level 4: Judge state machine ---

@dataclass
class JudgeState:
    """Per-speaker (stage, mode) state, updated after each turn by Judge.

    move_history / verdict_history / hits_history accumulate per turn for
    termination checks and post-session analysis.
    """
    speaker_role: Literal["protagonist", "counterpart"]
    stage: str
    mode: str
    last_why: str = ""
    last_verdict: str = ""
    move_history: list[str] = field(default_factory=list)
    verdict_history: list[str] = field(default_factory=list)
    hits_history: list[list[str]] = field(default_factory=list)


@dataclass
class JudgeResult:
    """One Judge LLM call outcome (proposed state + parse metadata).

    `proposed_*` are what Judge suggested; the runner passes these through
    `apply_stage_target` to produce the actual new JudgeState.
    """
    proposed_stage: str
    proposed_mode: str
    proposed_verdict: str  # "fire_release" | "basin_lock" | "N/A"
    why: str
    hits: list[str]
    meta: dict  # tokens_in, tokens_out, latency_ms, model, parse_status, error?
```

- [ ] **Step 4: Extend Persona with v3 fields**

Edit `src/empty_space/schemas.py`, replace the `Persona` class:

```python
class Persona(BaseModel):
    """A character's identity: 貫通軸 + N 關係層 + (optional) v3 judge context."""
    name: str
    version: str
    core_text: str
    relationship_texts: dict[str, str] = Field(default_factory=dict)
    # Level 4 (optional — empty when persona lacks v3 files):
    judge_principles_text: str = ""
    stage_mode_contexts_parsed: dict[str, dict[str, str]] = Field(default_factory=dict)
```

- [ ] **Step 5: Add InitialState validator for v3 mode migration**

Replace the `InitialState` class in `src/empty_space/schemas.py`:

```python
from pydantic import field_validator  # add to existing pydantic import line


class InitialState(BaseModel):
    """Opening verb / stage / mode — feeds the initial Judge state.

    v3 mode vocabulary: 收 / 放 / 在. Legacy '基線' auto-migrated to '在'.
    """
    verb: str
    stage: str
    mode: str

    @field_validator("mode")
    @classmethod
    def _migrate_legacy_mode(cls, v: str) -> str:
        return "在" if v == "基線" else v
```

Adjust the import at top of file (line 13):
```python
from pydantic import BaseModel, Field, field_validator
```

- [ ] **Step 6: Extend SessionResult with Level 4 fields**

Replace the `SessionResult` class in `src/empty_space/schemas.py`:

```python
@dataclass
class SessionResult:
    """What run_session returns."""
    exp_id: str
    out_dir: Path
    total_turns: int
    termination_reason: str  # "max_turns" | "dual_basin_lock"
    total_tokens_in: int
    total_tokens_out: int
    duration_seconds: float
    # Level 4 additions:
    judge_trajectories: dict = field(default_factory=dict)
    director_injections: list[dict] = field(default_factory=list)
    interactive_mode: bool = False
    judge_health: dict = field(default_factory=dict)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_schemas_level4.py -v`
Expected: PASS (6 tests)

- [ ] **Step 8: Run existing tests to catch regressions**

Run: `uv run pytest tests/test_schemas_experiment.py tests/test_schemas_persona.py -v`
Expected: All PASS (existing tests unaffected because new fields have defaults).

- [ ] **Step 9: Commit**

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"
git add src/empty_space/schemas.py tests/test_schemas_level4.py
git commit -m "feat(schemas): add JudgeState, JudgeResult, Persona v3 fields, InitialState migration"
```

---

### Task 2: judge.py 常數 + YAML 解析器

**Files:**
- Create: `src/empty_space/judge.py`
- Test: `tests/test_judge.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_judge.py`:

```python
"""Unit tests for empty_space.judge — constants, parsers, ratchet, prompt, run."""
import pytest

from empty_space.judge import (
    MODES,
    STAGE_ORDER,
    parse_judge_principles,
    parse_stage_mode_contexts,
)


def test_stage_order_length_and_first_last():
    assert len(STAGE_ORDER) == 7
    assert STAGE_ORDER[0] == "前置積累"
    assert STAGE_ORDER[-1] == "基線"


def test_modes_are_three():
    assert MODES == ["收", "放", "在"]


def test_parse_judge_principles_is_identity():
    text = "鯨的下潛——肩內收。"
    assert parse_judge_principles(text) == text


def test_parse_stage_mode_contexts_extracts_cells():
    raw = {
        "前置積累_收": {
            "張力狀態": "拉力 > 推力",
            "身體": "鯨的下潛",
            "語言形態": "極短",
            "碎裂密度": "低",
        },
        "前置積累_在": {
            "身體": "鯨的巡游",
            "語言形態": "極少",
            "碎裂密度": "最低",
        },
    }
    parsed = parse_stage_mode_contexts(raw)
    assert "前置積累_收" in parsed
    assert parsed["前置積累_收"]["身體傾向"] == "鯨的下潛"
    assert parsed["前置積累_收"]["語聲傾向"] == "極短"
    assert parsed["前置積累_收"]["注意力"] == "拉力 > 推力"
    assert parsed["前置積累_在"]["注意力"] == ""  # 張力狀態 missing → empty


def test_parse_stage_mode_contexts_ignores_non_cell_keys():
    raw = {
        "前置積累_收": {"身體": "A", "語言形態": "B", "張力狀態": "C"},
        "comment": "this is not a cell",
        "metadata": {"version": "v3"},
    }
    parsed = parse_stage_mode_contexts(raw)
    assert set(parsed.keys()) == {"前置積累_收"}


def test_parse_stage_mode_contexts_empty_input():
    assert parse_stage_mode_contexts({}) == {}
    assert parse_stage_mode_contexts(None) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_judge.py -v`
Expected: FAIL — `ModuleNotFoundError: empty_space.judge`

- [ ] **Step 3: Create judge.py with constants + parsers**

Create `src/empty_space/judge.py`:

```python
"""Level 4 Judge: per-speaker (stage, mode) state machine driven by Gemini Flash.

Design:
- STAGE moves along 7-cell序列 via advance/stay/regress.
- MODE is free switching among 收/放/在.
- fire_release allows STAGE +2 jump; basin_lock forces stay.
- Judge never modifies dialogue; it only observes and labels.
"""
from __future__ import annotations

from empty_space.schemas import JudgeResult, JudgeState


STAGE_ORDER: list[str] = [
    "前置積累",
    "初感訊號",
    "半意識浮現",
    "明確切換",
    "穩定期",
    "回溫期",
    "基線",
]

MODES: list[str] = ["收", "放", "在"]

JUDGE_MODEL = "gemini-2.5-flash"


# --- YAML parsers for persona v3 files ---

def parse_judge_principles(text: str) -> str:
    """Identity — judge_principles_v3.yaml content is embedded verbatim
    in the Judge user prompt. Kept as a function so callers have a named
    seam if we later want to template-render it.
    """
    return text


def parse_stage_mode_contexts(raw: dict | None) -> dict[str, dict[str, str]]:
    """Extract {"stage_mode": {身體傾向, 語聲傾向, 注意力}} from yaml dict.

    Input shape (stage_mode_contexts_v3.yaml): {<stage>_<mode>: {身體, 語言形態, 張力狀態, ...}}.
    Any top-level key whose value isn't a dict with at least one of those inner keys is skipped
    (e.g., top-level 'comment' or 'metadata').

    Output shape (for prompt_assembler):
        {"前置積累_收": {"身體傾向": "...", "語聲傾向": "...", "注意力": "..."}}

    Inner key mapping:
        身體傾向 ← 身體
        語聲傾向 ← 語言形態
        注意力   ← 張力狀態

    Missing inner fields default to "".
    """
    if not raw:
        return {}
    result: dict[str, dict[str, str]] = {}
    expected_inner = {"身體", "語言形態", "張力狀態", "碎裂密度"}
    for key, val in raw.items():
        if not isinstance(val, dict):
            continue
        if not (set(val.keys()) & expected_inner):
            continue
        result[key] = {
            "身體傾向": str(val.get("身體", "")),
            "語聲傾向": str(val.get("語言形態", "")),
            "注意力": str(val.get("張力狀態", "")),
        }
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_judge.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/empty_space/judge.py tests/test_judge.py
git commit -m "feat(judge): add STAGE_ORDER/MODES constants and v3 yaml parsers"
```

---

### Task 3: `apply_stage_target` ratchet 閘門

**Files:**
- Modify: `src/empty_space/judge.py`
- Test: `tests/test_judge.py` (extend)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_judge.py`:

```python
from empty_space.judge import apply_stage_target


def _state(stage="前置積累", mode="在") -> JudgeState:
    return JudgeState(speaker_role="protagonist", stage=stage, mode=mode)


from empty_space.schemas import JudgeState   # add with other imports at top if not present


def test_apply_stay():
    last = _state(stage="前置積累", mode="在")
    new, move = apply_stage_target(
        last_state=last, proposed_stage="前置積累",
        proposed_mode="在", proposed_verdict="N/A",
    )
    assert new.stage == "前置積累"
    assert move == "stay"


def test_apply_advance():
    last = _state(stage="前置積累", mode="在")
    new, move = apply_stage_target(
        last_state=last, proposed_stage="初感訊號",
        proposed_mode="收", proposed_verdict="N/A",
    )
    assert new.stage == "初感訊號"
    assert new.mode == "收"
    assert move == "advance"


def test_apply_regress():
    last = _state(stage="初感訊號", mode="收")
    new, move = apply_stage_target(
        last_state=last, proposed_stage="前置積累",
        proposed_mode="在", proposed_verdict="N/A",
    )
    assert new.stage == "前置積累"
    assert move == "regress"


def test_apply_illegal_jump_forces_stay():
    last = _state(stage="前置積累", mode="在")
    new, move = apply_stage_target(
        last_state=last, proposed_stage="明確切換",   # +3 jump, no fire_release
        proposed_mode="放", proposed_verdict="N/A",
    )
    assert new.stage == "前置積累"
    assert move == "illegal_stay"


def test_apply_fire_release_allows_plus_two():
    last = _state(stage="前置積累", mode="在")
    new, move = apply_stage_target(
        last_state=last, proposed_stage="半意識浮現",   # +2
        proposed_mode="放", proposed_verdict="fire_release",
    )
    assert new.stage == "半意識浮現"
    assert move == "fire_advance"


def test_apply_fire_release_does_not_allow_plus_three():
    last = _state(stage="前置積累", mode="在")
    new, move = apply_stage_target(
        last_state=last, proposed_stage="明確切換",   # +3 even under fire
        proposed_mode="放", proposed_verdict="fire_release",
    )
    assert new.stage == "前置積累"
    assert move == "illegal_stay"


def test_apply_basin_lock_forces_stay():
    last = _state(stage="穩定期", mode="在")
    new, move = apply_stage_target(
        last_state=last, proposed_stage="前置積累",   # Judge tried regress
        proposed_mode="收", proposed_verdict="basin_lock",
    )
    assert new.stage == "穩定期"
    assert move == "basin_stay"


def test_apply_mode_fallback_when_unknown():
    last = _state(stage="前置積累", mode="在")
    new, _ = apply_stage_target(
        last_state=last, proposed_stage="前置積累",
        proposed_mode="壓抑",   # not in MODES
        proposed_verdict="N/A",
    )
    assert new.mode == "在"   # fallback to last


def test_apply_mode_free_switch_within_legal():
    last = _state(stage="前置積累", mode="在")
    new, _ = apply_stage_target(
        last_state=last, proposed_stage="前置積累",
        proposed_mode="放", proposed_verdict="N/A",
    )
    assert new.mode == "放"


def test_apply_unknown_stage_name_forces_stay():
    last = _state(stage="前置積累", mode="在")
    new, move = apply_stage_target(
        last_state=last, proposed_stage="緩和期",   # not in STAGE_ORDER
        proposed_mode="收", proposed_verdict="N/A",
    )
    assert new.stage == "前置積累"
    assert move == "illegal_stay"


def test_apply_appends_move_history():
    last = _state()
    last.move_history = ["stay", "advance"]
    new, _ = apply_stage_target(
        last_state=last, proposed_stage="前置積累",
        proposed_mode="在", proposed_verdict="N/A",
    )
    assert new.move_history == ["stay", "advance", "stay"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_judge.py -v`
Expected: FAIL on new tests — `apply_stage_target` not found.

- [ ] **Step 3: Implement `apply_stage_target`**

Append to `src/empty_space/judge.py`:

```python
# --- ratchet gate ---

def apply_stage_target(
    *,
    last_state: JudgeState,
    proposed_stage: str,
    proposed_mode: str,
    proposed_verdict: str,
    why: str = "",
    hits: list[str] | None = None,
) -> tuple[JudgeState, str]:
    """Gate Judge's proposal through ratchet rules.

    Returns (new_state, move) where move ∈ {
        "advance", "stay", "regress",
        "illegal_stay",            # Judge named unknown stage OR tried illegal jump
        "fire_advance",            # fire_release allowed +2 jump
        "basin_stay",              # basin_lock forced stay despite proposed move
    }.
    """
    old_idx = STAGE_ORDER.index(last_state.stage)

    # 1. Resolve proposed stage index (unknown name → treat as stay attempt)
    if proposed_stage in STAGE_ORDER:
        new_idx = STAGE_ORDER.index(proposed_stage)
        diff = new_idx - old_idx
    else:
        new_idx = old_idx
        diff = 0
        move = "illegal_stay"
        return _build_new_state(
            last_state, new_idx, proposed_mode,
            proposed_verdict, why, hits, move,
        ), move

    # 2. basin_lock overrides — force stay
    if proposed_verdict == "basin_lock":
        new_idx = old_idx
        move = "basin_stay"
        return _build_new_state(
            last_state, new_idx, proposed_mode,
            proposed_verdict, why, hits, move,
        ), move

    # 3. fire_release exception — allow +2 jump (but not +3 or more)
    if proposed_verdict == "fire_release" and diff == 2:
        move = "fire_advance"
        return _build_new_state(
            last_state, new_idx, proposed_mode,
            proposed_verdict, why, hits, move,
        ), move

    # 4. Normal ratchet: only -1, 0, +1 allowed
    if diff == 1:
        move = "advance"
    elif diff == 0:
        move = "stay"
    elif diff == -1:
        move = "regress"
    else:
        # Illegal jump (|diff| >= 2 without fire exception)
        new_idx = old_idx
        move = "illegal_stay"

    return _build_new_state(
        last_state, new_idx, proposed_mode,
        proposed_verdict, why, hits, move,
    ), move


def _build_new_state(
    last_state: JudgeState,
    new_stage_idx: int,
    proposed_mode: str,
    verdict: str,
    why: str,
    hits: list[str] | None,
    move: str,
) -> JudgeState:
    """Construct the new JudgeState with mode fallback + history appends."""
    new_mode = proposed_mode if proposed_mode in MODES else last_state.mode
    new_hits = list(hits) if hits else []
    return JudgeState(
        speaker_role=last_state.speaker_role,
        stage=STAGE_ORDER[new_stage_idx],
        mode=new_mode,
        last_why=why,
        last_verdict=verdict,
        move_history=last_state.move_history + [move],
        verdict_history=last_state.verdict_history + [verdict],
        hits_history=last_state.hits_history + [new_hits],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_judge.py -v`
Expected: PASS (all tests, including 11 new ratchet tests)

- [ ] **Step 5: Commit**

```bash
git add src/empty_space/judge.py tests/test_judge.py
git commit -m "feat(judge): add apply_stage_target ratchet gate"
```

---

### Task 4: `parse_judge_output` 寬容解析

**Files:**
- Modify: `src/empty_space/judge.py`
- Test: `tests/test_judge.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_judge.py`:

```python
from empty_space.judge import parse_judge_output


def test_parse_judge_output_happy_path():
    text = """STAGE: 初感訊號
MODE: 收
WHY: 母親縮肩且答話變短
VERDICT: N/A
HITS: 肩往下; 「嗯。」; 沉默 4 秒
"""
    last = _state(stage="前置積累", mode="在")
    r = parse_judge_output(text, last_state=last)
    assert r.proposed_stage == "初感訊號"
    assert r.proposed_mode == "收"
    assert r.proposed_verdict == "N/A"
    assert r.why == "母親縮肩且答話變短"
    assert r.hits == ["肩往下", "「嗯。」", "沉默 4 秒"]
    assert r.meta["parse_status"] == "ok"


def test_parse_judge_output_full_width_colons():
    text = """STAGE：初感訊號
MODE：收
WHY：母親縮肩
VERDICT：N/A
HITS：line1
"""
    last = _state()
    r = parse_judge_output(text, last_state=last)
    assert r.proposed_stage == "初感訊號"
    assert r.proposed_mode == "收"
    assert r.why == "母親縮肩"


def test_parse_judge_output_preamble_ignored():
    text = """好的，我來判斷：

STAGE: 前置積累
MODE: 在
WHY: 對話剛開始
VERDICT: N/A
HITS: -
"""
    last = _state()
    r = parse_judge_output(text, last_state=last)
    assert r.proposed_stage == "前置積累"
    assert r.proposed_mode == "在"


def test_parse_judge_output_stage_fuzzy_match():
    text = """STAGE: 明確切換期
MODE: 放
WHY: 爆發
VERDICT: fire_release
HITS: line1
"""
    last = _state()
    r = parse_judge_output(text, last_state=last)
    # Fuzzy substring match → 明確切換
    assert r.proposed_stage == "明確切換"


def test_parse_judge_output_missing_hits_line():
    text = """STAGE: 前置積累
MODE: 收
WHY: 沒線索
VERDICT: N/A
"""
    last = _state()
    r = parse_judge_output(text, last_state=last)
    assert r.hits == []
    assert r.meta["parse_status"] in ("ok", "partial")


def test_parse_judge_output_totally_broken_falls_back():
    text = "the model said nothing useful"
    last = _state(stage="半意識浮現", mode="收")
    r = parse_judge_output(text, last_state=last)
    # Everything falls back to last_state
    assert r.proposed_stage == "半意識浮現"
    assert r.proposed_mode == "收"
    assert r.proposed_verdict == "N/A"
    assert r.meta["parse_status"] == "fallback_used"


def test_parse_judge_output_unknown_mode_falls_back():
    text = """STAGE: 前置積累
MODE: 壓抑
WHY: mode 詞彙錯
VERDICT: N/A
HITS: x
"""
    last = _state(stage="前置積累", mode="收")
    r = parse_judge_output(text, last_state=last)
    assert r.proposed_mode == "收"   # fallback


def test_parse_judge_output_unknown_verdict_becomes_na():
    text = """STAGE: 前置積累
MODE: 收
WHY: x
VERDICT: ignition
HITS: x
"""
    last = _state()
    r = parse_judge_output(text, last_state=last)
    assert r.proposed_verdict == "N/A"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_judge.py -v`
Expected: FAIL — `parse_judge_output` not defined.

- [ ] **Step 3: Implement `parse_judge_output`**

Append to `src/empty_space/judge.py`:

```python
# --- tolerant output parser ---

_VERDICT_VALUES = {"fire_release", "basin_lock", "N/A"}


def parse_judge_output(text: str, *, last_state: JudgeState) -> JudgeResult:
    """Parse Flash's 5-line output tolerantly.

    Expected format:
        STAGE: <stage 名>
        MODE: <mode 名>
        WHY: <一句>
        VERDICT: <fire_release | basin_lock | N/A>
        HITS: <line1; line2; line3>

    Tolerances:
    - Full-width colons (：) accepted
    - Preamble lines (not matching any field) are skipped
    - Stage/mode name substring/superstring of a canonical name → fuzzy match
    - Missing field → fallback to last_state (and mark parse_status)
    - Completely unparseable → fallback_used for all fields
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    raw_stage = _extract_field(lines, ["STAGE:", "STAGE：", "階段:", "階段："])
    raw_mode = _extract_field(lines, ["MODE:", "MODE：", "模式:", "模式："])
    raw_verdict = _extract_field(lines, ["VERDICT:", "VERDICT："])
    raw_why = _extract_field(lines, ["WHY:", "WHY：", "為什麼:", "為什麼："])
    raw_hits = _extract_field(lines, ["HITS:", "HITS："])

    found_any = any(v is not None for v in (raw_stage, raw_mode, raw_verdict, raw_why, raw_hits))
    parse_status = "ok" if found_any else "fallback_used"

    stage = _normalise_stage(raw_stage or "", last_state.stage)
    mode = _normalise_mode(raw_mode or "", last_state.mode)
    verdict = _normalise_verdict(raw_verdict or "")
    why = (raw_why or "").strip()
    hits = _parse_hits(raw_hits or "")

    # If we got some fields but not all, mark partial
    if found_any and (raw_stage is None or raw_mode is None or raw_verdict is None):
        parse_status = "partial"

    return JudgeResult(
        proposed_stage=stage,
        proposed_mode=mode,
        proposed_verdict=verdict,
        why=why,
        hits=hits,
        meta={"parse_status": parse_status},
    )


def _extract_field(lines: list[str], prefixes: list[str]) -> str | None:
    """Return the first line's content after any matching prefix, or None."""
    for line in lines:
        for p in prefixes:
            if line.startswith(p):
                return line[len(p):].strip()
    return None


def _normalise_stage(raw: str, fallback: str) -> str:
    """Fuzzy match to STAGE_ORDER. Exact match preferred; then substring both ways."""
    if raw in STAGE_ORDER:
        return raw
    for s in STAGE_ORDER:
        if s in raw or (raw and raw in s):
            return s
    return fallback


def _normalise_mode(raw: str, fallback: str) -> str:
    """Exact match to MODES (收/放/在). If the raw contains one of them as a char, accept."""
    if raw in MODES:
        return raw
    for m in MODES:
        if m in raw:
            return m
    return fallback


def _normalise_verdict(raw: str) -> str:
    """Only accept one of the three canonical verdicts; else N/A."""
    for v in _VERDICT_VALUES:
        if v in raw:
            return v
    return "N/A"


def _parse_hits(raw: str) -> list[str]:
    """Split HITS by ';' (半形) or '；' (全形). Empty and placeholder '-' filtered."""
    if not raw:
        return []
    parts = [p.strip() for chunk in raw.split(";") for p in chunk.split("；")]
    return [p for p in parts if p and p != "-"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_judge.py -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add src/empty_space/judge.py tests/test_judge.py
git commit -m "feat(judge): add tolerant parse_judge_output with fuzzy normalisation"
```

---

### Task 5: `build_judge_prompt` + `run_judge` + `is_*` helpers

**Files:**
- Modify: `src/empty_space/judge.py`
- Test: `tests/test_judge.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_judge.py`:

```python
from empty_space.judge import (
    build_judge_prompt,
    is_basin_lock,
    is_fire_release,
    run_judge,
)
from empty_space.llm import GeminiResponse


class _MockLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.explode = False

    def generate(self, *, system, user, model="gemini-2.5-flash"):
        self.calls.append({"system": system, "user": user, "model": model})
        if self.explode:
            raise RuntimeError("network down")
        content = self.responses.pop(0)
        return GeminiResponse(
            content=content, raw=None,
            tokens_in=100, tokens_out=30, model=model, latency_ms=80,
        )


def test_build_judge_prompt_includes_last_state_and_persona():
    last = _state(stage="前置積累", mode="在")
    last.last_why = "上一句只說嗯"
    system, user = build_judge_prompt(
        last_state=last,
        principles_text="鯨的下潛——肩內收",
        stage_mode_contexts_text="前置積累_在：巡游",
        recent_turns_text="[Turn 1 母親] 嗯。",
        speaker_role="protagonist",
        persona_name="母親",
    )
    assert "STAGE:" in system
    assert "VERDICT:" in system
    assert "鯨的下潛" in user
    assert "巡游" in user
    assert "前置積累" in user
    assert "上一句只說嗯" in user
    assert "[Turn 1 母親] 嗯。" in user
    assert "母親" in user


def test_run_judge_happy_path():
    mock = _MockLLM(responses=[
        "STAGE: 初感訊號\nMODE: 收\nWHY: ok\nVERDICT: N/A\nHITS: -\n",
    ])
    last = _state(stage="前置積累", mode="在")
    result = run_judge(
        last_state=last,
        principles_text="p",
        stage_mode_contexts_text="c",
        recent_turns_text="t",
        speaker_role="protagonist",
        persona_name="母親",
        llm_client=mock,
    )
    assert result.proposed_stage == "初感訊號"
    assert result.proposed_mode == "收"
    assert result.meta["parse_status"] == "ok"
    assert result.meta["model"] == "gemini-2.5-flash"
    assert result.meta["tokens_in"] == 100


def test_run_judge_llm_exception_returns_fallback_result():
    mock = _MockLLM(responses=[])
    mock.explode = True
    last = _state(stage="半意識浮現", mode="收")
    result = run_judge(
        last_state=last,
        principles_text="p",
        stage_mode_contexts_text="c",
        recent_turns_text="t",
        speaker_role="protagonist",
        persona_name="母親",
        llm_client=mock,
    )
    # Fallback: last state preserved, verdict N/A, error recorded
    assert result.proposed_stage == "半意識浮現"
    assert result.proposed_mode == "收"
    assert result.proposed_verdict == "N/A"
    assert "error" in result.meta
    assert "network down" in result.meta["error"]


def test_is_fire_release_and_basin_lock():
    s = _state()
    s.last_verdict = "fire_release"
    assert is_fire_release(s) is True
    assert is_basin_lock(s) is False
    s.last_verdict = "basin_lock"
    assert is_fire_release(s) is False
    assert is_basin_lock(s) is True
    s.last_verdict = "N/A"
    assert is_fire_release(s) is False
    assert is_basin_lock(s) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_judge.py -v`
Expected: FAIL — `build_judge_prompt` / `run_judge` / `is_fire_release` not defined.

- [ ] **Step 3: Implement prompt builder + run_judge**

Append to `src/empty_space/judge.py`:

```python
# --- Judge prompt + run ---

_JUDGE_SYSTEM_PROMPT = """\
你是戲劇裡的「隱性量測者」。你不介入對話、不評分、不給建議。
你只做一件事：根據這個角色最近說的話、做的動作、身體狀態，
判斷他在 stage × mode 二維空間裡「下一刻」會落在哪一格。

規則：
- STAGE 只能沿序列相鄰移動：前置積累 → 初感訊號 → 半意識浮現 → 明確切換 → 穩定期 → 回溫期 → 基線
  - 可以 advance（往下一格）、stay（同格）、regress（退回前一格）
  - 不能跳格
- MODE 是當下的身體傾向：收 / 放 / 在
  - 收：往內收斂、壓住、沉默、身體變小
  - 放：往外釋放、爆發、哭、笑、吼
  - 在：既不收也不放，只是存在、觀察、呼吸
- VERDICT 標記特殊事件：
  - fire_release：角色剛剛發生了明顯的情緒釋放
  - basin_lock：角色進入穩態盆地（stage=穩定期/回溫期 且連續 2 輪以上不動）
  - N/A：其他情況
- HITS 是你觀察到的具體線索

輸出格式（5 行，嚴格）：
STAGE: <stage 名>
MODE: <mode 名>
WHY: <一句話>
VERDICT: <fire_release | basin_lock | N/A>
HITS: <line1; line2; line3>
"""


def build_judge_prompt(
    *,
    last_state: JudgeState,
    principles_text: str,
    stage_mode_contexts_text: str,
    recent_turns_text: str,
    speaker_role: str,
    persona_name: str,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for one Judge call."""
    user = f"""\
# 角色原則
{principles_text}

# Stage × Mode 脈絡
{stage_mode_contexts_text}

# 上一輪狀態
STAGE: {last_state.stage}
MODE: {last_state.mode}
LAST_WHY: {last_state.last_why}

# 最近對話（最多 3 輪）
{recent_turns_text}

# 任務
根據以上，只判斷 {speaker_role}（{persona_name}）這個角色，
輸出他「剛說完這輪話之後」的 stage/mode/why/verdict/hits。
"""
    return _JUDGE_SYSTEM_PROMPT, user


def run_judge(
    *,
    last_state: JudgeState,
    principles_text: str,
    stage_mode_contexts_text: str,
    recent_turns_text: str,
    speaker_role: str,
    persona_name: str,
    llm_client,
) -> JudgeResult:
    """One full Judge call. Returns JudgeResult; never raises.

    On LLM exception: returns a fallback result mirroring last_state with
    verdict='N/A' and meta.error set. Caller should pass this through
    apply_stage_target (which will then stay).
    """
    system, user = build_judge_prompt(
        last_state=last_state,
        principles_text=principles_text,
        stage_mode_contexts_text=stage_mode_contexts_text,
        recent_turns_text=recent_turns_text,
        speaker_role=speaker_role,
        persona_name=persona_name,
    )
    try:
        resp = llm_client.generate(system=system, user=user, model=JUDGE_MODEL)
    except Exception as e:
        return JudgeResult(
            proposed_stage=last_state.stage,
            proposed_mode=last_state.mode,
            proposed_verdict="N/A",
            why=f"[judge_error] {type(e).__name__}",
            hits=[],
            meta={
                "tokens_in": 0,
                "tokens_out": 0,
                "latency_ms": 0,
                "model": JUDGE_MODEL,
                "parse_status": "llm_error",
                "error": f"{type(e).__name__}: {e}",
            },
        )

    parsed = parse_judge_output(resp.content, last_state=last_state)
    parsed.meta.update({
        "tokens_in": resp.tokens_in,
        "tokens_out": resp.tokens_out,
        "latency_ms": resp.latency_ms,
        "model": resp.model,
    })
    return parsed


def is_fire_release(state: JudgeState) -> bool:
    return state.last_verdict == "fire_release"


def is_basin_lock(state: JudgeState) -> bool:
    return state.last_verdict == "basin_lock"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_judge.py -v`
Expected: PASS (all ~24 tests)

- [ ] **Step 5: Commit**

```bash
git add src/empty_space/judge.py tests/test_judge.py
git commit -m "feat(judge): add build_judge_prompt, run_judge, is_fire_release/basin_lock"
```

---

### Task 6: loaders.py 讀取 v3 檔案

**Files:**
- Modify: `src/empty_space/loaders.py`
- Test: `tests/test_loaders_persona.py` (extend) or new `tests/test_loaders_v3.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_loaders_v3.py`:

```python
"""Tests for loader reading v3 judge files (optional, empty when missing)."""
from pathlib import Path

import pytest
import yaml

from empty_space.loaders import load_persona


@pytest.fixture
def persona_with_v3(tmp_path, monkeypatch):
    """Build a minimal persona dir with v3 files."""
    root = tmp_path / "persona_root"
    pdir = root / "test_group" / "母親"
    pdir.mkdir(parents=True)

    (pdir / "貫通軸_v3_tension.yaml").write_text("core text", encoding="utf-8")
    (pdir / "關係層_兒子_v3_tension.yaml").write_text("rel text", encoding="utf-8")
    (pdir / "judge_principles_v3.yaml").write_text(
        "MODE_傾向:\n  收: 0.6\n  放: 0.05\n  在: 0.35\n", encoding="utf-8",
    )
    (pdir / "stage_mode_contexts_v3.yaml").write_text(
        yaml.safe_dump({
            "前置積累_收": {
                "身體": "鯨的下潛", "語言形態": "極短", "張力狀態": "拉力 > 推力",
            },
        }, allow_unicode=True),
        encoding="utf-8",
    )

    monkeypatch.setattr("empty_space.loaders.PERSONA_ROOT", root)
    return root, pdir


def test_load_persona_with_v3_files(persona_with_v3):
    root, _ = persona_with_v3
    p = load_persona("test_group/母親", version="v3_tension")
    assert "MODE_傾向" in p.judge_principles_text
    assert "前置積累_收" in p.stage_mode_contexts_parsed
    assert p.stage_mode_contexts_parsed["前置積累_收"]["身體傾向"] == "鯨的下潛"


def test_load_persona_without_v3_files_has_empty_fields(tmp_path, monkeypatch):
    root = tmp_path / "persona_root"
    pdir = root / "g" / "父親"
    pdir.mkdir(parents=True)
    (pdir / "貫通軸_v3_tension.yaml").write_text("core", encoding="utf-8")
    monkeypatch.setattr("empty_space.loaders.PERSONA_ROOT", root)

    p = load_persona("g/父親", version="v3_tension")
    assert p.judge_principles_text == ""
    assert p.stage_mode_contexts_parsed == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_loaders_v3.py -v`
Expected: FAIL — v3 files are not being read by `load_persona`.

- [ ] **Step 3: Extend `load_persona` to read v3 files**

Edit `src/empty_space/loaders.py`. In `load_persona`, after the `relationship_texts` loop (currently ending at line 68) and before the `return Persona(...)` call, insert:

```python
    # Level 4: optional v3 judge files
    import yaml as _yaml  # local import to avoid adding yaml to module-level imports unnecessarily
    from empty_space.judge import parse_judge_principles, parse_stage_mode_contexts

    judge_principles_text = ""
    jp_file = persona_dir / "judge_principles_v3.yaml"
    if jp_file.exists():
        judge_principles_text = parse_judge_principles(
            jp_file.read_text(encoding="utf-8")
        )

    stage_mode_contexts_parsed: dict[str, dict[str, str]] = {}
    smc_file = persona_dir / "stage_mode_contexts_v3.yaml"
    if smc_file.exists():
        raw = _yaml.safe_load(smc_file.read_text(encoding="utf-8")) or {}
        stage_mode_contexts_parsed = parse_stage_mode_contexts(raw)
```

Then update the `return Persona(...)` call to pass these:

```python
    return Persona(
        name=persona_dir.name,
        version=version,
        core_text=core_text,
        relationship_texts=relationship_texts,
        judge_principles_text=judge_principles_text,
        stage_mode_contexts_parsed=stage_mode_contexts_parsed,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_loaders_v3.py tests/test_loaders_persona.py -v`
Expected: PASS (new tests + existing unchanged)

- [ ] **Step 5: Commit**

```bash
git add src/empty_space/loaders.py tests/test_loaders_v3.py
git commit -m "feat(loaders): read optional v3 judge files into Persona"
```

---

### Task 7: prompt_assembler `## 此刻` 動態化

**Files:**
- Modify: `src/empty_space/prompt_assembler.py`
- Test: `tests/test_prompt_assembler.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_prompt_assembler.py`:

```python
from empty_space.schemas import JudgeState


def _mk_persona(name="母親") -> Persona:
    return Persona(name=name, version="v3_tension", core_text="CORE")


def _mk_setting() -> Setting:
    return Setting(name="醫院", content="SETTING")


def test_此刻_block_with_judge_state_and_contexts():
    from empty_space.prompt_assembler import build_system_prompt

    js = JudgeState(speaker_role="protagonist", stage="初感訊號", mode="收")
    smc = {
        "初感訊號_收": {
            "身體傾向": "鯨偵測到聲波",
            "語聲傾向": "沉默或單音節",
            "注意力": "拉力在升",
        }
    }
    prompt = build_system_prompt(
        persona=_mk_persona(),
        counterpart_name="兒子",
        setting=_mk_setting(),
        scene_premise="醫院",
        initial_state=InitialState(verb="承受", stage="前置積累", mode="在"),
        active_events=[],
        judge_state=js,
        stage_mode_contexts=smc,
    )
    assert "階段：初感訊號" in prompt
    assert "模式：收" in prompt
    assert "鯨偵測到聲波" in prompt
    assert "沉默或單音節" in prompt
    assert "拉力在升" in prompt


def test_此刻_block_fallback_when_judge_state_none():
    from empty_space.prompt_assembler import build_system_prompt

    prompt = build_system_prompt(
        persona=_mk_persona(),
        counterpart_name="兒子",
        setting=_mk_setting(),
        scene_premise="醫院",
        initial_state=InitialState(verb="承受", stage="前置積累", mode="在"),
        active_events=[],
        judge_state=None,
        stage_mode_contexts=None,
    )
    # Fallback to initial_state (Level 3 behavior)
    assert "動作詞：承受" in prompt
    assert "階段：前置積累" in prompt
    assert "模式：在" in prompt


def test_此刻_block_fallback_when_cell_missing():
    """Judge_state present but cell not in stage_mode_contexts — minimal render."""
    from empty_space.prompt_assembler import build_system_prompt

    js = JudgeState(speaker_role="protagonist", stage="穩定期", mode="放")
    prompt = build_system_prompt(
        persona=_mk_persona(),
        counterpart_name="兒子",
        setting=_mk_setting(),
        scene_premise="醫院",
        initial_state=InitialState(verb="承受", stage="前置積累", mode="在"),
        active_events=[],
        judge_state=js,
        stage_mode_contexts={},   # empty — 穩定期_放 not present
    )
    assert "階段：穩定期" in prompt
    assert "模式：放" in prompt
    # No body/voice/attention sub-lines when cell missing
    assert "身體傾向" not in prompt
```

Ensure at the top of `tests/test_prompt_assembler.py` these imports exist (add if missing):

```python
from empty_space.schemas import InitialState, Persona, Setting
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_prompt_assembler.py -v`
Expected: FAIL — `build_system_prompt` doesn't accept `judge_state` / `stage_mode_contexts` kwargs.

- [ ] **Step 3: Rewrite `build_system_prompt`**

Edit `src/empty_space/prompt_assembler.py`. Add `JudgeState` to imports at top:

```python
from empty_space.schemas import (
    InitialState,
    JudgeState,
    Persona,
    RetrievedImpression,
    Setting,
    Turn,
)
```

Replace the entire `build_system_prompt` function signature and the `## 此刻` block. New version:

```python
def build_system_prompt(
    persona: Persona,
    counterpart_name: str,
    setting: Setting,
    scene_premise: str | None,
    initial_state: InitialState,
    active_events: list[tuple[int, str]],
    prelude: str | None = None,
    retrieved_impressions: list[RetrievedImpression] | None = None,
    ambient_echo: list[str] | None = None,
    judge_state: JudgeState | None = None,
    stage_mode_contexts: dict[str, dict[str, str]] | None = None,
) -> str:
    """Assemble the system prompt for one role's turn.

    Block order (spec §4.1): 貫通軸 → 關係層 → 此刻 → 現場 → 輸出格式.

    Level 4: 此刻 block reads from judge_state + stage_mode_contexts cell
    if provided; otherwise falls back to initial_state (Level 3 behavior).
    """
    _ = ambient_echo  # reserved for Phase 4
    relationship_text = persona.relationship_texts.get(counterpart_name, "")

    blocks: list[str] = []

    blocks.append(f"## 貫通軸\n{persona.core_text.rstrip()}")

    blocks.append(f"## 關係層：對{counterpart_name}\n{relationship_text.rstrip()}")

    blocks.append(_build_此刻_block(
        judge_state=judge_state,
        stage_mode_contexts=stage_mode_contexts,
        initial_state=initial_state,
    ))

    scene_parts: list[str] = [setting.content.rstrip()]
    if scene_premise is not None:
        scene_parts.append(f"### 場景前提\n{scene_premise.rstrip()}")
    if active_events:
        event_lines = "\n".join(
            f"Turn {turn}：{content}" for turn, content in active_events
        )
        scene_parts.append(f"### 已發生的事\n{event_lines}")
    blocks.append("## 現場\n" + "\n\n".join(scene_parts))

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

    return "\n\n".join(blocks)


def _build_此刻_block(
    *,
    judge_state: JudgeState | None,
    stage_mode_contexts: dict[str, dict[str, str]] | None,
    initial_state: InitialState,
) -> str:
    """Render the ## 此刻 block, preferring judge_state + cell context.

    Fallback order:
    1. judge_state + cell found → full render (stage/mode/body/voice/attention)
    2. judge_state + cell missing → minimal render (stage/mode only)
    3. judge_state None → legacy render (verb/stage/mode from initial_state)
    """
    if judge_state is None:
        return (
            "## 此刻\n"
            f"動作詞：{initial_state.verb}\n"
            f"階段：{initial_state.stage}\n"
            f"模式：{initial_state.mode}"
        )

    cell_key = f"{judge_state.stage}_{judge_state.mode}"
    cell = (stage_mode_contexts or {}).get(cell_key)

    lines = [
        "## 此刻",
        f"階段：{judge_state.stage}",
        f"模式：{judge_state.mode}",
    ]
    if cell:
        if cell.get("身體傾向"):
            lines.append(f"身體傾向：{cell['身體傾向']}")
        if cell.get("語聲傾向"):
            lines.append(f"語聲傾向：{cell['語聲傾向']}")
        if cell.get("注意力"):
            lines.append(f"注意力：{cell['注意力']}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_prompt_assembler.py -v`
Expected: PASS (existing + 3 new)

- [ ] **Step 5: Commit**

```bash
git add src/empty_space/prompt_assembler.py tests/test_prompt_assembler.py
git commit -m "feat(prompt_assembler): 此刻 block reads judge_state + cell context with fallback"
```

---

### Task 8: runner.py 初始化 JudgeState + InitialState migration 落地

**Files:**
- Modify: `src/empty_space/runner.py`
- Test: `tests/test_runner_level4.py` (create)

- [ ] **Step 1: Create `tests/test_runner_level4.py` with initial test**

Create `tests/test_runner_level4.py`:

```python
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


def test_initial_judge_states_created_with_v3_mode():
    """SessionState should have two JudgeStates seeded from initial_state."""
    # 2-turn session with both speakers' Judge responding stay.
    # Call sequence: 2 extract + 2 dialogue + 2*2 judge + 1 composer = 9
    responses = [
        "- 醫院\n",                                                # extract P
        "- 醫院\n",                                                # extract C
        "話一",                                                   # turn 1 protagonist
        "STAGE: 前置積累\nMODE: 在\nWHY: -\nVERDICT: N/A\nHITS: -\n",  # judge P t1
        "STAGE: 前置積累\nMODE: 在\nWHY: -\nVERDICT: N/A\nHITS: -\n",  # judge C t1
        "話二",                                                   # turn 2 counterpart
        "STAGE: 前置積累\nMODE: 在\nWHY: -\nVERDICT: N/A\nHITS: -\n",  # judge P t2
        "STAGE: 前置積累\nMODE: 在\nWHY: -\nVERDICT: N/A\nHITS: -\n",  # judge C t2
        "母親: []\n兒子: []\n",                                    # composer noop
    ]
    config = _base_config(max_turns=2)
    result = run_session(config=config, llm_client=MockLLMClient(responses))

    # Meta should have judge_trajectories (this check validates task 11 too but
    # passes once task 8+9+11 land; keep the smoke assertion broad).
    meta = yaml.safe_load((result.out_dir / "meta.yaml").read_text(encoding="utf-8"))
    assert "judge_trajectories" in meta
    assert meta["judge_trajectories"]["protagonist"]["stages"][0] == "前置積累"
    assert meta["judge_trajectories"]["counterpart"]["stages"][0] == "前置積累"


def test_initial_state_legacy_mode_baseline_migrated():
    """ExperimentConfig with mode='基線' should be migrated to '在' at validation."""
    config = _base_config()
    # manually construct with legacy value — validator should normalise
    legacy = InitialState(verb="承受", stage="前置積累", mode="基線")
    assert legacy.mode == "在"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_runner_level4.py::test_initial_state_legacy_mode_baseline_migrated -v`
Expected: PASS already (Task 1 handled it)

Run: `uv run pytest tests/test_runner_level4.py::test_initial_judge_states_created_with_v3_mode -v`
Expected: FAIL — runner doesn't wire Judge yet; MockLLMClient runs out of responses or meta.yaml lacks judge_trajectories.

- [ ] **Step 3: Extend SessionState + add `_init_judge_state`**

Edit `src/empty_space/runner.py`. Add to the top imports:

```python
from empty_space.judge import (
    STAGE_ORDER,
    apply_stage_target,
    is_basin_lock,
    is_fire_release,
    run_judge,
)
from empty_space.schemas import (
    ComposerSessionResult,
    ExperimentConfig,
    JudgeState,
    Persona,
    RetrievalResult,
    SessionResult,
    Setting,
    Turn,
)
```

Replace the `SessionState` dataclass with:

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
    retrieval_protagonist: RetrievalResult | None = None
    retrieval_counterpart: RetrievalResult | None = None
    # Level 4:
    judge_state_protagonist: JudgeState | None = None
    judge_state_counterpart: JudgeState | None = None
    director_injections: list[dict] = field(default_factory=list)
```

Add helper near end of file (before `_append_session_ledgers`):

```python
def _init_judge_state(
    speaker_role: str, initial_state
) -> JudgeState:
    """Seed JudgeState from experiment's initial_state (InitialState).

    initial_state.mode is already v3-migrated by the pydantic validator
    (legacy '基線' → '在').
    """
    return JudgeState(
        speaker_role=speaker_role,  # type: ignore[arg-type]
        stage=initial_state.stage,
        mode=initial_state.mode,
        last_why="",
        last_verdict="",
        move_history=[],
        verdict_history=[],
        hits_history=[],
    )


def _should_run_judge(persona: Persona) -> bool:
    """Judge only runs for personas with both v3 files present."""
    return bool(persona.judge_principles_text) and bool(persona.stage_mode_contexts_parsed)


def _stage_mode_contexts_text(persona: Persona) -> str:
    """Render persona.stage_mode_contexts_parsed as readable text for Judge prompt."""
    if not persona.stage_mode_contexts_parsed:
        return ""
    lines: list[str] = []
    for key, cell in persona.stage_mode_contexts_parsed.items():
        lines.append(f"{key}:")
        for field_name in ("身體傾向", "語聲傾向", "注意力"):
            if cell.get(field_name):
                lines.append(f"  {field_name}: {cell[field_name]}")
    return "\n".join(lines)
```

In `run_session`, after `state = SessionState(...)` and before the `for n in range(1, config.max_turns + 1):` loop, insert:

```python
    state.judge_state_protagonist = _init_judge_state("protagonist", config.initial_state)
    state.judge_state_counterpart = _init_judge_state("counterpart", config.initial_state)
```

- [ ] **Step 4: Confirm test for migration still passes**

Run: `uv run pytest tests/test_runner_level4.py::test_initial_state_legacy_mode_baseline_migrated -v`
Expected: PASS

(The other test still requires Tasks 9 + 11; leave it failing for now — next tasks fix it.)

- [ ] **Step 5: Commit**

```bash
git add src/empty_space/runner.py tests/test_runner_level4.py
git commit -m "feat(runner): init per-speaker JudgeState in SessionState"
```

---

### Task 9: runner.py 每輪結束跑雙 Judge

**Files:**
- Modify: `src/empty_space/runner.py`

- [ ] **Step 1: Write failing test (extend test_runner_level4.py)**

Append to `tests/test_runner_level4.py`:

```python
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
    """State updates between turns — turn 2 sees turn 1's Judge output."""
    # Turn 1 judge P: advance to 初感訊號/收
    # Turn 1 judge C: stay at 前置積累/在
    # In turn 2 P's system prompt 此刻 should now read 初感訊號/收
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_runner_level4.py -v`
Expected: FAIL — Judge not yet wired into the turn loop.

- [ ] **Step 3: Add `_run_judges_post_turn` helper to runner.py**

Append to `src/empty_space/runner.py` (before `_append_session_ledgers`):

```python
def _format_recent_turns(turns: list[Turn], n: int = 3) -> str:
    """Render last n turns as readable text for Judge prompt."""
    if not turns:
        return "（尚無對話）"
    tail = turns[-n:]
    return "\n".join(
        f"[Turn {t.turn_number} {t.persona_name}] {t.content}"
        for t in tail
    )


def _run_judges_post_turn(
    state: SessionState,
) -> tuple[dict, dict]:
    """Run Judge for both speakers (if eligible). Returns (p_output_dict, c_output_dict)
    suitable for writing to turn yaml's judge_output_protagonist / _counterpart.

    Updates state.judge_state_protagonist and state.judge_state_counterpart
    in place via apply_stage_target.
    """
    recent = _format_recent_turns(state.turns, n=3)
    outputs: dict[str, dict] = {}

    for role, persona, attr in (
        ("protagonist", state.protagonist, "judge_state_protagonist"),
        ("counterpart", state.counterpart, "judge_state_counterpart"),
    ):
        last = getattr(state, attr)
        if not _should_run_judge(persona):
            outputs[role] = {"skipped": True, "reason": "no_v3_config"}
            continue
        jr = run_judge(
            last_state=last,
            principles_text=persona.judge_principles_text,
            stage_mode_contexts_text=_stage_mode_contexts_text(persona),
            recent_turns_text=recent,
            speaker_role=role,
            persona_name=persona.name,
            llm_client=_judge_llm_client_from_state(state),
        )
        new_state, move = apply_stage_target(
            last_state=last,
            proposed_stage=jr.proposed_stage,
            proposed_mode=jr.proposed_mode,
            proposed_verdict=jr.proposed_verdict,
            why=jr.why,
            hits=jr.hits,
        )
        setattr(state, attr, new_state)
        outputs[role] = {
            "proposed": {
                "stage": jr.proposed_stage,
                "mode": jr.proposed_mode,
                "verdict": jr.proposed_verdict,
                "why": jr.why,
            },
            "applied": {
                "stage": new_state.stage,
                "mode": new_state.mode,
                "move": move,
            },
            "hits": list(jr.hits),
            "meta": dict(jr.meta),
        }
    return outputs["protagonist"], outputs["counterpart"]
```

The function `_judge_llm_client_from_state` is a small indirection we'll implement as stashing the client on SessionState. Update `SessionState` to carry the client reference:

Edit the `SessionState` dataclass:

```python
@dataclass
class SessionState:
    config: ExperimentConfig
    protagonist: Persona
    counterpart: Persona
    setting: Setting
    turns: list[Turn] = field(default_factory=list)
    active_events: list[tuple[int, str]] = field(default_factory=list)
    retrieval_protagonist: RetrievalResult | None = None
    retrieval_counterpart: RetrievalResult | None = None
    judge_state_protagonist: JudgeState | None = None
    judge_state_counterpart: JudgeState | None = None
    director_injections: list[dict] = field(default_factory=list)
    llm_client: object = None   # stash for Judge post-turn
```

Add helper:

```python
def _judge_llm_client_from_state(state: SessionState):
    assert state.llm_client is not None, "llm_client must be set on SessionState"
    return state.llm_client
```

Update `run_session` to set `llm_client` on state (right after creating `state`):

```python
    state = SessionState(
        config=config,
        protagonist=protagonist,
        counterpart=counterpart,
        setting=setting,
        retrieval_protagonist=retrieval_protagonist,
        retrieval_counterpart=retrieval_counterpart,
    )
    state.llm_client = llm_client
```

- [ ] **Step 4: Wire `## 此刻` and post-turn Judge into the turn loop**

In `run_session`, inside the `for n in range(...)` loop, replace the existing `system_prompt = build_system_prompt(...)` call with:

```python
        active_judge_state = (
            state.judge_state_protagonist if speaker_role == "protagonist"
            else state.judge_state_counterpart
        )
        active_persona_contexts = (
            speaker_persona.stage_mode_contexts_parsed
            if speaker_persona.stage_mode_contexts_parsed
            else None
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
            judge_state=active_judge_state,
            stage_mode_contexts=active_persona_contexts,
        )
```

After `state.turns.append(turn)` (currently line ~179) and before `append_turn(out_dir, turn)`, insert:

```python
        # Level 4: run Judge for both speakers after this turn
        judge_out_p, judge_out_c = _run_judges_post_turn(state)
```

Change `append_turn(out_dir, turn)` call — writer needs the judge outputs. Update to:

```python
        append_turn(out_dir, turn, judge_output_protagonist=judge_out_p, judge_output_counterpart=judge_out_c)
```

(This requires writer.py changes — Task 10 will add these kwargs. For now the test won't pass until Task 10 lands — that's OK, TDD across multi-task chain.)

- [ ] **Step 5: Commit partial wiring**

```bash
git add src/empty_space/runner.py
git commit -m "feat(runner): invoke dual Judge post-turn and pass state into prompt assembler"
```

(Tests still fail — writer.py next.)

---

### Task 10: writer.py turn yaml judge_output 區塊

**Files:**
- Modify: `src/empty_space/writer.py`
- Test: `tests/test_writer.py` (extend)

- [ ] **Step 1: Write failing test**

Append to `tests/test_writer.py`:

```python
def test_append_turn_writes_judge_outputs(tmp_path):
    """append_turn should persist judge_output_protagonist/counterpart in turn yaml."""
    from empty_space.schemas import CandidateImpression
    from empty_space.writer import append_turn, init_run

    cfg = _mk_minimal_config()  # existing helper in this test file
    out_dir = tmp_path / "run1"
    init_run(out_dir, cfg)

    turn = _mk_turn(turn_number=1, speaker="protagonist", persona_name="母親")
    jop = {
        "proposed": {"stage": "初感訊號", "mode": "收", "verdict": "N/A", "why": "x"},
        "applied":  {"stage": "初感訊號", "mode": "收", "move": "advance"},
        "hits": ["line1"],
        "meta": {"tokens_in": 100, "tokens_out": 30, "model": "gemini-2.5-flash",
                 "latency_ms": 80, "parse_status": "ok"},
    }
    joc = {"skipped": True, "reason": "no_v3_config"}

    append_turn(out_dir, turn, judge_output_protagonist=jop, judge_output_counterpart=joc)

    import yaml
    data = yaml.safe_load((out_dir / "turns" / "turn_001.yaml").read_text(encoding="utf-8"))
    assert data["judge_output_protagonist"]["applied"]["move"] == "advance"
    assert data["judge_output_counterpart"]["skipped"] is True


def test_append_turn_writes_director_injection_when_provided(tmp_path):
    from empty_space.writer import append_turn, init_run

    cfg = _mk_minimal_config()
    out_dir = tmp_path / "run2"
    init_run(out_dir, cfg)
    turn = _mk_turn(turn_number=1, speaker="protagonist", persona_name="母親")
    injection = {
        "event": "護士開門",
        "triggered_by": "fire_release on protagonist",
        "applied_to_turn": 2,
    }
    append_turn(
        out_dir, turn,
        judge_output_protagonist={"skipped": True, "reason": "none"},
        judge_output_counterpart={"skipped": True, "reason": "none"},
        director_injection=injection,
    )
    import yaml
    data = yaml.safe_load((out_dir / "turns" / "turn_001.yaml").read_text(encoding="utf-8"))
    assert data["director_injection"]["event"] == "護士開門"
```

Ensure `_mk_minimal_config` and `_mk_turn` helpers exist in `tests/test_writer.py`. If not, add them at the top of the test file:

```python
from datetime import datetime, timezone

from empty_space.schemas import (
    CandidateImpression,
    ExperimentConfig,
    InitialState,
    PersonaRef,
    SettingRef,
    Termination,
    Turn,
)


def _mk_minimal_config():
    return ExperimentConfig(
        exp_id="writer_test",
        protagonist=PersonaRef(path="x", version="v3_tension"),
        counterpart=PersonaRef(path="y", version="v3_tension"),
        setting=SettingRef(path="z.yaml"),
        scene_premise="test",
        initial_state=InitialState(verb="承受", stage="前置積累", mode="在"),
        max_turns=2,
        termination=Termination(),
    )


def _mk_turn(turn_number: int, speaker: str, persona_name: str) -> Turn:
    return Turn(
        turn_number=turn_number,
        speaker=speaker,  # type: ignore[arg-type]
        persona_name=persona_name,
        content="content",
        candidate_impressions=[],
        prompt_system="sys",
        prompt_user="usr",
        raw_response="content",
        tokens_in=10, tokens_out=5,
        model="gemini-2.5-flash",
        latency_ms=10,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        director_events_active=[],
    )
```

(If helpers already exist, skip this insertion.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_writer.py -v`
Expected: FAIL — `append_turn` doesn't accept `judge_output_*` or `director_injection`.

- [ ] **Step 3: Extend `append_turn` signature and `_turn_to_yaml_dict`**

Edit `src/empty_space/writer.py`. Change `append_turn` to:

```python
def append_turn(
    out_dir: Path,
    turn: "Turn",
    *,
    judge_output_protagonist: dict | None = None,
    judge_output_counterpart: dict | None = None,
    director_injection: dict | None = None,
) -> None:
    """Write turn_NNN.yaml atomically; append director_event marker (if new) and
    turn entry to conversation.md + conversation.jsonl.

    Level 4: optional judge_output_{protagonist,counterpart} and director_injection
    are embedded in turn yaml for later analysis.
    """
    turn_file = out_dir / "turns" / f"turn_{turn.turn_number:03d}.yaml"
    _atomic_write_yaml(
        turn_file,
        _turn_to_yaml_dict(
            turn,
            judge_output_protagonist=judge_output_protagonist,
            judge_output_counterpart=judge_output_counterpart,
            director_injection=director_injection,
        ),
    )

    new_event = _new_event_this_turn(turn)
    _append_conversation_md(out_dir, turn, new_event)
    _append_conversation_jsonl(out_dir, turn, new_event)
```

Change `_turn_to_yaml_dict` signature and body tail:

```python
def _turn_to_yaml_dict(
    turn: "Turn",
    *,
    judge_output_protagonist: dict | None = None,
    judge_output_counterpart: dict | None = None,
    director_injection: dict | None = None,
) -> dict:
    d = {
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
        "retrieved_impressions": [
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
    if judge_output_protagonist is not None:
        d["judge_output_protagonist"] = judge_output_protagonist
    if judge_output_counterpart is not None:
        d["judge_output_counterpart"] = judge_output_counterpart
    if director_injection is not None:
        d["director_injection"] = director_injection
    return d
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_writer.py -v`
Expected: PASS on new tests. Existing `append_turn` call sites (in `tests/test_writer.py` existing tests and `runner.py`) still work because new params are keyword-only with defaults.

- [ ] **Step 5: Commit**

```bash
git add src/empty_space/writer.py tests/test_writer.py
git commit -m "feat(writer): embed judge_output and director_injection in turn yaml"
```

---

### Task 11: writer.py meta.yaml judge_trajectories / judge_health / termination

**Files:**
- Modify: `src/empty_space/writer.py`
- Modify: `src/empty_space/runner.py` (aggregate and pass)

- [ ] **Step 1: Write failing test (extend test_runner_level4.py)**

Append to `tests/test_runner_level4.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_runner_level4.py::test_meta_yaml_includes_judge_trajectories_and_health -v`
Expected: FAIL — meta.yaml lacks these keys.

- [ ] **Step 3: Extend `write_meta` signature**

Edit `src/empty_space/writer.py` `write_meta`:

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
    composer_tokens_in: int = 0,
    composer_tokens_out: int = 0,
    composer_latency_ms: int = 0,
    protagonist_refined_added: int = 0,
    counterpart_refined_added: int = 0,
    composer_parse_error: str | None = None,
    # Level 4:
    judge_trajectories: dict | None = None,
    judge_health: dict | None = None,
    termination_turn: int = 0,
    director_injections: list[dict] | None = None,
    interactive_mode: bool = False,
) -> None:
    meta = {
        "exp_id": config.exp_id,
        "run_timestamp": out_dir.name,
        "total_turns": total_turns,
        "termination_reason": termination_reason,
        "termination": {
            "reason": termination_reason,
            "turn": termination_turn or total_turns,
        },
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
        "composer_tokens_in": composer_tokens_in,
        "composer_tokens_out": composer_tokens_out,
        "composer_latency_ms": composer_latency_ms,
        "protagonist_refined_added": protagonist_refined_added,
        "counterpart_refined_added": counterpart_refined_added,
        "composer_parse_error": composer_parse_error,
        # Level 4:
        "judge_trajectories": judge_trajectories or {},
        "judge_health": judge_health or {},
        "director_injections": director_injections or [],
        "interactive_mode": interactive_mode,
    }
    _atomic_write_yaml(out_dir / "meta.yaml", meta)
```

- [ ] **Step 4: Aggregate trajectories + health in runner.py**

In `src/empty_space/runner.py`, add two helpers near the other helpers:

```python
def _build_judge_trajectories(state: SessionState) -> dict:
    """Pull per-speaker (stage, mode, move, verdict) lists from final JudgeStates."""
    def pack(js: JudgeState | None) -> dict:
        if js is None:
            return {"stages": [], "modes": [], "moves": [], "verdicts": []}
        return {
            "stages": [],  # populated below
            "modes": [],
            "moves": list(js.move_history),
            "verdicts": list(js.verdict_history),
        }
    # We need per-turn stages/modes; reconstruct from move_history applied to the
    # initial state (the state we seeded from initial_state).
    # Simpler: track a parallel list on SessionState updated in _run_judges_post_turn.
    return {
        "protagonist": {
            "stages": list(state.judge_history_protagonist["stages"]),
            "modes": list(state.judge_history_protagonist["modes"]),
            "moves": list(state.judge_state_protagonist.move_history) if state.judge_state_protagonist else [],
            "verdicts": list(state.judge_state_protagonist.verdict_history) if state.judge_state_protagonist else [],
        },
        "counterpart": {
            "stages": list(state.judge_history_counterpart["stages"]),
            "modes": list(state.judge_history_counterpart["modes"]),
            "moves": list(state.judge_state_counterpart.move_history) if state.judge_state_counterpart else [],
            "verdicts": list(state.judge_state_counterpart.verdict_history) if state.judge_state_counterpart else [],
        },
    }


def _build_judge_health(state: SessionState) -> dict:
    """Aggregate from state.judge_health_events, keyed by speaker role.

    Categories: total_calls, ok, parse_fallback (parse_status in {partial,fallback_used}),
    llm_error, no_judge (skipped).
    """
    def pack(events: list[dict]) -> dict:
        total = len(events)
        ok = sum(1 for e in events if e.get("parse_status") == "ok")
        parse_fallback = sum(1 for e in events if e.get("parse_status") in ("partial", "fallback_used"))
        llm_error = sum(1 for e in events if e.get("parse_status") == "llm_error")
        no_judge = sum(1 for e in events if e.get("parse_status") == "no_judge")
        return {
            "total_calls": total,
            "ok": ok,
            "parse_fallback": parse_fallback,
            "llm_error": llm_error,
            "no_judge": no_judge,
        }
    return {
        "protagonist": pack(state.judge_health_events["protagonist"]),
        "counterpart": pack(state.judge_health_events["counterpart"]),
    }
```

Extend `SessionState` with tracking fields:

```python
@dataclass
class SessionState:
    config: ExperimentConfig
    protagonist: Persona
    counterpart: Persona
    setting: Setting
    turns: list[Turn] = field(default_factory=list)
    active_events: list[tuple[int, str]] = field(default_factory=list)
    retrieval_protagonist: RetrievalResult | None = None
    retrieval_counterpart: RetrievalResult | None = None
    judge_state_protagonist: JudgeState | None = None
    judge_state_counterpart: JudgeState | None = None
    director_injections: list[dict] = field(default_factory=list)
    llm_client: object = None
    # Level 4 bookkeeping:
    judge_history_protagonist: dict = field(
        default_factory=lambda: {"stages": [], "modes": []}
    )
    judge_history_counterpart: dict = field(
        default_factory=lambda: {"stages": [], "modes": []}
    )
    judge_health_events: dict = field(
        default_factory=lambda: {"protagonist": [], "counterpart": []}
    )
```

Update `_run_judges_post_turn` to append to bookkeeping. Replace its body to:

```python
def _run_judges_post_turn(
    state: SessionState,
) -> tuple[dict, dict]:
    recent = _format_recent_turns(state.turns, n=3)
    outputs: dict[str, dict] = {}

    for role, persona, attr, hist_attr in (
        ("protagonist", state.protagonist, "judge_state_protagonist", "judge_history_protagonist"),
        ("counterpart", state.counterpart, "judge_state_counterpart", "judge_history_counterpart"),
    ):
        last = getattr(state, attr)
        if not _should_run_judge(persona):
            outputs[role] = {"skipped": True, "reason": "no_v3_config"}
            # history still tracks last known stage/mode (unchanged)
            getattr(state, hist_attr)["stages"].append(last.stage)
            getattr(state, hist_attr)["modes"].append(last.mode)
            state.judge_health_events[role].append({"parse_status": "no_judge"})
            continue
        jr = run_judge(
            last_state=last,
            principles_text=persona.judge_principles_text,
            stage_mode_contexts_text=_stage_mode_contexts_text(persona),
            recent_turns_text=recent,
            speaker_role=role,
            persona_name=persona.name,
            llm_client=_judge_llm_client_from_state(state),
        )
        new_state, move = apply_stage_target(
            last_state=last,
            proposed_stage=jr.proposed_stage,
            proposed_mode=jr.proposed_mode,
            proposed_verdict=jr.proposed_verdict,
            why=jr.why,
            hits=jr.hits,
        )
        setattr(state, attr, new_state)
        getattr(state, hist_attr)["stages"].append(new_state.stage)
        getattr(state, hist_attr)["modes"].append(new_state.mode)
        state.judge_health_events[role].append({"parse_status": jr.meta.get("parse_status", "ok")})
        outputs[role] = {
            "proposed": {
                "stage": jr.proposed_stage,
                "mode": jr.proposed_mode,
                "verdict": jr.proposed_verdict,
                "why": jr.why,
            },
            "applied": {
                "stage": new_state.stage,
                "mode": new_state.mode,
                "move": move,
            },
            "hits": list(jr.hits),
            "meta": dict(jr.meta),
        }
    return outputs["protagonist"], outputs["counterpart"]
```

Update `run_session` — pass new kwargs to `write_meta`. At the bottom of `run_session`, in the `write_meta(...)` call, add:

```python
        judge_trajectories=_build_judge_trajectories(state),
        judge_health=_build_judge_health(state),
        termination_turn=len(state.turns),
        director_injections=list(state.director_injections),
        interactive_mode=False,   # Task 13 will flip this when --interactive is on
```

Also delete the now-wrong `_build_judge_trajectories` inner dict placeholders (the `pack` helper that only returned empty stages is dead code — remove it). The function body should just be the `return {...}` block pulling from the `judge_history_*` on state. Clean final version:

```python
def _build_judge_trajectories(state: SessionState) -> dict:
    return {
        "protagonist": {
            "stages": list(state.judge_history_protagonist["stages"]),
            "modes": list(state.judge_history_protagonist["modes"]),
            "moves": list(state.judge_state_protagonist.move_history) if state.judge_state_protagonist else [],
            "verdicts": list(state.judge_state_protagonist.verdict_history) if state.judge_state_protagonist else [],
        },
        "counterpart": {
            "stages": list(state.judge_history_counterpart["stages"]),
            "modes": list(state.judge_history_counterpart["modes"]),
            "moves": list(state.judge_state_counterpart.move_history) if state.judge_state_counterpart else [],
            "verdicts": list(state.judge_state_counterpart.verdict_history) if state.judge_state_counterpart else [],
        },
    }
```

Also, `run_judge`'s fallback meta uses `parse_status="llm_error"` — ensure `_build_judge_health` counts those. (Already handled above.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_runner_level4.py -v`
Expected: Most Level 4 tests should now pass (tasks 8+9+10+11 all land together).

- [ ] **Step 6: Run existing tests for regressions**

Run: `uv run pytest tests/ -v`
Expected: All tests pass. (The existing `test_runner_level2.py` tests continue to pass because personas in those tests also have v3 files in the real persona repo — Judge will run. That's fine; those tests will just also produce judge_trajectories but won't assert on them.)

If the Level 2 tests break because MockLLMClient runs out of responses (since Judge now consumes more calls), update the Level 2 test responses to include Judge answers. Check each test:

```bash
uv run pytest tests/test_runner_level2.py -v
```

Expected failure pattern: `RuntimeError: out of responses on call N`.

If that happens, for each failing test, pad `responses` with Judge stay-answers — add 2 Judge responses per dialogue turn:

```python
"STAGE: 前置積累\nMODE: 在\nWHY: -\nVERDICT: N/A\nHITS: -\n",
"STAGE: 前置積累\nMODE: 在\nWHY: -\nVERDICT: N/A\nHITS: -\n",
```

Inserted right after each `話一`/`話二`/etc. dialogue response.

- [ ] **Step 7: Commit**

```bash
git add src/empty_space/runner.py src/empty_space/writer.py tests/test_runner_level2.py tests/test_runner_level4.py
git commit -m "feat(runner,writer): aggregate judge_trajectories + judge_health into meta.yaml"
```

---

### Task 12: runner.py dual_basin_lock termination

**Files:**
- Modify: `src/empty_space/runner.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_runner_level4.py`:

```python
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
    # Note: we need initial_state stage=穩定期 for basin_lock to apply meaningfully,
    # but apply_stage_target will force stay anyway; just use default and the verdict still counts.
    config = _base_config(max_turns=10)
    client = MockLLMClient(responses)
    result = run_session(config=config, llm_client=client)

    assert result.total_turns == 2
    assert result.termination_reason == "dual_basin_lock"


def test_single_basin_lock_does_not_terminate():
    """Only protagonist basin_lock — session continues."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_runner_level4.py::test_dual_basin_lock_terminates_session_early -v`
Expected: FAIL — session runs to max_turns (10) instead of stopping at 2.

- [ ] **Step 3: Implement termination check**

In `src/empty_space/runner.py`, add helper near other helpers:

```python
def _should_terminate(state: SessionState, consecutive_required: int = 2) -> tuple[bool, str]:
    """Check whether session should stop BEFORE the next turn.

    Returns (should_stop, reason). Reason is "dual_basin_lock" or "" (no stop).
    """
    jp, jc = state.judge_state_protagonist, state.judge_state_counterpart
    if jp is None or jc is None:
        return False, ""

    def last_n_all_basin(state: JudgeState, n: int) -> bool:
        h = state.verdict_history
        if len(h) < n:
            return False
        return all(v == "basin_lock" for v in h[-n:])

    if last_n_all_basin(jp, consecutive_required) and last_n_all_basin(jc, consecutive_required):
        return True, "dual_basin_lock"
    return False, ""
```

In `run_session`, modify the turn loop. Add a termination check after Judge runs and before continuing to next iteration:

Find the block:
```python
        # Level 4: run Judge for both speakers after this turn
        judge_out_p, judge_out_c = _run_judges_post_turn(state)

        # 7. append
        append_turn(out_dir, turn, judge_output_protagonist=judge_out_p, judge_output_counterpart=judge_out_c)
```

Add after the `append_turn` line:

```python
        # Level 4: termination check
        should_stop, reason = _should_terminate(state)
        if should_stop:
            termination_reason = reason
            break
    else:
        termination_reason = "max_turns"
```

Replace the existing unconditional `termination_reason = "max_turns"` line (just below the `for` loop) with this `for...else` construct. The `else` clause on the `for` loop fires only when the loop completes without `break`.

Also adjust the existing `termination_reason = "max_turns"` line that was AFTER the for loop — remove it since the else now handles the no-break case.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_runner_level4.py -v`
Expected: PASS (both new tests)

- [ ] **Step 5: Commit**

```bash
git add src/empty_space/runner.py tests/test_runner_level4.py
git commit -m "feat(runner): dual_basin_lock termination (consecutive 2 turns both speakers)"
```

---

### Task 13: runner.py 互動 peak hook

**Files:**
- Modify: `src/empty_space/runner.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_runner_level4.py`:

```python
def test_interactive_peak_injects_director_event(monkeypatch):
    """In interactive mode, fire_release triggers stdin prompt; event injected to next turn."""
    # protagonist fire_release on turn 1 → prompt stdin → user types "護士開門"
    # turn 2's system prompt (counterpart speaks) should contain that event under 現場

    inputs = iter(["護士開門"])

    def fake_input(_prompt=""):
        return next(inputs)

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
    # If _prompt_for_director_event were called, the test stub below would fail.
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_runner_level4.py::test_interactive_peak_injects_director_event -v`
Expected: FAIL — `run_session` doesn't accept `interactive` kwarg.

- [ ] **Step 3: Add peak detection + prompt + interactive wiring**

In `src/empty_space/runner.py`, add helpers:

```python
def _is_peak(
    state: SessionState,
) -> tuple[bool, str]:
    """Return (is_peak, triggered_by). is_peak True iff any speaker last_verdict
    is fire_release or basin_lock this turn.
    """
    jp, jc = state.judge_state_protagonist, state.judge_state_counterpart
    if jp and is_fire_release(jp):
        return True, "fire_release on protagonist"
    if jp and is_basin_lock(jp):
        return True, "basin_lock on protagonist"
    if jc and is_fire_release(jc):
        return True, "fire_release on counterpart"
    if jc and is_basin_lock(jc):
        return True, "basin_lock on counterpart"
    return False, ""


def _prompt_for_director_event(
    *,
    turn_number: int,
    state_p: JudgeState,
    state_c: JudgeState,
    triggered_by: str,
) -> str | None:
    """Block stdin for director input. Empty line → None (skip). EOF → None."""
    print(f"\n{'='*60}")
    print(f"[PEAK @ turn {turn_number}] {triggered_by}")
    print(f"  protagonist: stage={state_p.stage}, mode={state_p.mode}, verdict={state_p.last_verdict}")
    print(f"  counterpart: stage={state_c.stage}, mode={state_c.mode}, verdict={state_c.last_verdict}")
    print(f"{'='*60}")
    print("導演介入？輸入事件（空行=跳過）：")
    try:
        line = input().strip()
    except EOFError:
        return None
    return line if line else None
```

Update `run_session` signature:

```python
def run_session(
    *, config: ExperimentConfig, llm_client: LLMClient, interactive: bool = False
) -> SessionResult:
```

Inside the turn loop, after Judge runs and AFTER `append_turn(...)` and BEFORE the termination check, add peak handling:

```python
        # Level 4: interactive peak hook
        director_injection = None
        if interactive:
            is_peak, triggered_by = _is_peak(state)
            if is_peak:
                event_text = _prompt_for_director_event(
                    turn_number=n,
                    state_p=state.judge_state_protagonist,
                    state_c=state.judge_state_counterpart,
                    triggered_by=triggered_by,
                )
                if event_text:
                    next_turn = n + 1
                    state.config.director_events[next_turn] = event_text
                    director_injection = {
                        "event": event_text,
                        "triggered_by": triggered_by,
                        "applied_to_turn": next_turn,
                    }
                    state.director_injections.append({
                        "turn": n, "event": event_text, "triggered_by": triggered_by,
                    })
```

Problem: the existing `append_turn` call happens BEFORE the peak hook, so `director_injection` isn't recorded in turn yaml. Reorder: move `append_turn` call to AFTER the peak hook. Rewrite the bottom of the turn-loop body as:

```python
        # Level 4: run Judge for both speakers after this turn
        judge_out_p, judge_out_c = _run_judges_post_turn(state)

        # Level 4: interactive peak hook (may inject director event for next turn)
        director_injection = None
        if interactive:
            is_peak, triggered_by = _is_peak(state)
            if is_peak:
                event_text = _prompt_for_director_event(
                    turn_number=n,
                    state_p=state.judge_state_protagonist,
                    state_c=state.judge_state_counterpart,
                    triggered_by=triggered_by,
                )
                if event_text:
                    next_turn = n + 1
                    state.config.director_events[next_turn] = event_text
                    director_injection = {
                        "event": event_text,
                        "triggered_by": triggered_by,
                        "applied_to_turn": next_turn,
                    }
                    state.director_injections.append({
                        "turn": n, "event": event_text, "triggered_by": triggered_by,
                    })

        # 7. append (now includes judge + injection)
        append_turn(
            out_dir, turn,
            judge_output_protagonist=judge_out_p,
            judge_output_counterpart=judge_out_c,
            director_injection=director_injection,
        )

        # Level 4: termination check
        should_stop, reason = _should_terminate(state)
        if should_stop:
            termination_reason = reason
            break
    else:
        termination_reason = "max_turns"
```

Finally, update the `write_meta` call to pass `interactive_mode=interactive`:

```python
        interactive_mode=interactive,
```

(replacing the earlier `interactive_mode=False` hard-coded value from Task 11.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_runner_level4.py -v`
Expected: PASS (new tests + all prior)

- [ ] **Step 5: Commit**

```bash
git add src/empty_space/runner.py tests/test_runner_level4.py
git commit -m "feat(runner): interactive peak hook injects director event on fire_release/basin_lock"
```

---

### Task 14: CLI `--interactive` flag

**Files:**
- Modify: `scripts/run_experiment.py`

- [ ] **Step 1: Add failing test (optional — CLI is thin)**

CLI behaviour already covered by runner tests. This is a wiring-only task; a smoke check suffices:

Run: `uv run python scripts/run_experiment.py --help`
Expected: currently shows usage string without `--interactive`.

- [ ] **Step 2: Rewrite `scripts/run_experiment.py` to use argparse**

Replace the entire file with:

```python
"""Run a single experiment session.

Usage:
    uv run python scripts/run_experiment.py <exp_id> [--interactive]

Examples:
    uv run python scripts/run_experiment.py mother_x_son_hospital_v3_001
    uv run python scripts/run_experiment.py mother_x_son_act1_hospital --interactive
"""
import argparse
import sys

from empty_space.llm import GeminiClient
from empty_space.loaders import load_experiment
from empty_space.runner import run_session


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one empty-space experiment session.")
    ap.add_argument("exp_id", help="experiment id (matches experiments/<exp_id>.yaml)")
    ap.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive director hook at fire_release/basin_lock peaks",
    )
    args = ap.parse_args()

    config = load_experiment(args.exp_id)
    client = GeminiClient()

    result = run_session(config=config, llm_client=client, interactive=args.interactive)

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

- [ ] **Step 3: Verify help works**

Run: `uv run python scripts/run_experiment.py --help`
Expected: shows `--interactive` flag.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_experiment.py
git commit -m "feat(cli): add --interactive flag to run_experiment.py"
```

---

### Task 15: 既有 experiment yaml 的 v3 詞彙遷移

**Files:**
- Modify: `experiments/mother_x_son_act1_hospital.yaml`
- Modify: `experiments/mother_x_son_act2_car.yaml`
- Modify: `experiments/mother_x_son_act3_home.yaml`
- Modify: `experiments/mother_x_son_hospital_v3_001.yaml`

- [ ] **Step 1: Verify current state**

Run: `grep -A3 "initial_state:" /Users/chenbaiwei/Desktop/vibe\ coding/empty-space/experiments/*.yaml`

Expected output shows `mode: 基線` in all four files.

- [ ] **Step 2: Rewrite `mode: 基線` → `mode: 在` in all four files**

Use Edit tool on each file (the YAML around `initial_state:` is small — direct text edit is safe):

For each of the four files, change:
```yaml
initial_state:
  verb: ...
  stage: 前置積累
  mode: 基線
```

to:
```yaml
initial_state:
  verb: ...
  stage: 前置積累
  mode: 在
```

- [ ] **Step 3: Verify**

Run: `grep -A3 "initial_state:" /Users/chenbaiwei/Desktop/vibe\ coding/empty-space/experiments/*.yaml`
Expected: all four show `mode: 在`.

- [ ] **Step 4: Run loaders experiment test**

Run: `uv run pytest tests/test_loaders_experiment.py -v`
Expected: PASS (loading still works; validator is a no-op on "在" since no migration needed).

- [ ] **Step 5: Commit**

```bash
git add experiments/*.yaml
git commit -m "migrate: initial_state.mode 基線 → 在 (v3 vocabulary)"
```

---

### Task 16: 整合測試完整跑

**Files:**
- Already created: `tests/test_runner_level4.py`
- Final verification step

- [ ] **Step 1: Run the full Level 4 test suite**

Run: `uv run pytest tests/test_runner_level4.py -v`
Expected: all tests PASS:
- test_initial_judge_states_created_with_v3_mode
- test_initial_state_legacy_mode_baseline_migrated
- test_judge_runs_twice_per_turn_for_both_speakers
- test_judge_state_evolves_across_turns
- test_judge_skipped_when_persona_lacks_v3
- test_judge_llm_error_does_not_crash_session
- test_meta_yaml_includes_judge_trajectories_and_health
- test_dual_basin_lock_terminates_session_early
- test_single_basin_lock_does_not_terminate
- test_interactive_peak_injects_director_event
- test_non_interactive_peak_does_not_prompt

- [ ] **Step 2: Run the ENTIRE test suite for regressions**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS. If `tests/test_runner_level2.py` or `tests/test_runner_integration.py` fail with "out of responses", add Judge stay-responses (2 per dialogue turn) as per Task 11 Step 6 instructions.

- [ ] **Step 3: Re-commit any test fixes**

If you had to patch older tests:

```bash
git add tests/test_runner_level2.py tests/test_runner_integration.py
git commit -m "test: pad existing runner tests with Judge stay-responses"
```

---

### Task 17: Smoke test against real Flash

**Files:**
- Create: `scripts/smoke_level4.py`

- [ ] **Step 1: Write the smoke script**

Create `scripts/smoke_level4.py`:

```python
"""Level 4 smoke test: 6-turn hospital session with real Gemini Flash.

Runs batch mode (no interactive) and prints key Level 4 artifacts for
human eyeball check.

Usage:
    uv run python scripts/smoke_level4.py

Prerequisites:
    GEMINI_API_KEY env var set.
"""
import sys
from pathlib import Path

import yaml

from empty_space.llm import GeminiClient
from empty_space.loaders import load_experiment
from empty_space.runner import run_session


def main() -> int:
    config = load_experiment("mother_x_son_act1_hospital")
    # Force short session for smoke
    config.max_turns = 6

    client = GeminiClient()
    result = run_session(config=config, llm_client=client, interactive=False)

    print(f"\n✓ Completed {result.exp_id}")
    print(f"  Out: {result.out_dir}")
    print(f"  Turns: {result.total_turns}")
    print(f"  Termination: {result.termination_reason}")
    print(f"  Duration: {result.duration_seconds:.1f}s")

    meta = yaml.safe_load((result.out_dir / "meta.yaml").read_text(encoding="utf-8"))
    print("\n--- Judge Trajectories ---")
    for role in ("protagonist", "counterpart"):
        t = meta["judge_trajectories"][role]
        print(f"\n{role}:")
        print(f"  stages:  {t['stages']}")
        print(f"  modes:   {t['modes']}")
        print(f"  moves:   {t['moves']}")
        print(f"  verdicts:{t['verdicts']}")

    print("\n--- Judge Health ---")
    for role in ("protagonist", "counterpart"):
        print(f"{role}: {meta['judge_health'][role]}")

    print("\n--- Sample turn 3 judge_output ---")
    turn_3 = yaml.safe_load(
        (result.out_dir / "turns" / "turn_003.yaml").read_text(encoding="utf-8")
    )
    print("protagonist:")
    print(yaml.safe_dump(turn_3.get("judge_output_protagonist", {}), allow_unicode=True))
    print("counterpart:")
    print(yaml.safe_dump(turn_3.get("judge_output_counterpart", {}), allow_unicode=True))

    print("\n--- Sample turn 3 system prompt (此刻 block) ---")
    sys_prompt = turn_3["prompt_assembled"]["system"]
    # Extract 此刻 block
    start = sys_prompt.find("## 此刻")
    end = sys_prompt.find("## 現場")
    if start >= 0 and end > start:
        print(sys_prompt[start:end].strip())

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run smoke (requires GEMINI_API_KEY)**

Run: `uv run python scripts/smoke_level4.py`

Expected: completes without error, prints trajectories where stages shift at least once (not all `前置積累`), modes mix 收/在, judge_health.ok ≥ 80% of total_calls.

If stages never move → investigate Judge prompt (persona principles might be too conservative or Judge is being too lazy). If llm_error > 0 → check network / API key. Do not mark Task 17 complete if stages are 100% stuck AND modes are all one value — Judge is clearly not working.

- [ ] **Step 3: Commit smoke script**

```bash
git add scripts/smoke_level4.py
git commit -m "test: add Level 4 Flash smoke script"
```

- [ ] **Step 4: Manual interactive sanity check (not automated)**

Run: `uv run python scripts/run_experiment.py mother_x_son_act1_hospital --interactive`

When prompt pauses for director input, type "父親的監視器突然響了" and press Enter. Watch whether the next role turn's response references that event.

Expected: session continues after injection; turn +1 yaml shows `## 現場` → `### 已發生的事` contains the injected text; the role's dialogue shows awareness of it.

If peak never triggers within 6 turns → lower max_turns for faster retry, or confirm one of the personas can reach fire_release at all. Not a blocker; just document in session notes.

---

## Self-Review

**Spec coverage check** — each spec section mapped to tasks:

| Spec § | Requirement | Task |
|---|---|---|
| §3.1 JudgeState | New dataclass with 7 fields | Task 1 |
| §3.2 Persona v3 fields | judge_principles_text, stage_mode_contexts_parsed | Task 1 |
| §3.3 SessionState fields | judge_state_{protagonist,counterpart} | Task 8 |
| §3.4 SessionResult fields | termination_reason, judge_trajectories, etc. | Task 1 |
| §3.5 Constants | STAGE_ORDER, MODES, JUDGE_MODEL | Task 2 |
| §4 judge.py module | parse_*, apply_stage_target, run_judge, is_* | Tasks 2-5 |
| §5 Judge prompt | System + User templates | Task 5 |
| §6 Ratchet | advance/stay/regress/illegal/fire_advance/basin_stay | Task 3 |
| §7 Assembler | dynamic 此刻 block with fallbacks | Task 7 |
| §8.1-8.2 Runner lifecycle | init + per-turn + skip | Tasks 8, 9 |
| §8.3 Interactive | --interactive + peak hook | Tasks 13, 14 |
| §9 Termination | dual_basin_lock + max_turns | Task 12 |
| §10.1 Turn yaml | judge_output_* + director_injection | Task 10 |
| §10.2 meta.yaml | trajectories/health/termination | Task 11 |
| §10.3 Persona loading | Read v3 files, skip if absent | Task 6 |
| §10.4 v3 migration | InitialState validator + yaml rewrite | Tasks 1, 15 |
| §11 Error handling | LLM/parse/jump/stdin all fallback | Tasks 3, 4, 5, 13 |
| §12 Testing Layer 1 | test_judge.py | Tasks 2-5 |
| §12 Testing Layer 2 | test_runner_level4.py | Tasks 8-13, 16 |
| §12 Testing Layer 3 | smoke_level4.py | Task 17 |
| §14 CLI | --interactive flag | Task 14 |

All spec requirements mapped. No gaps.

**Placeholder scan:** all code steps include complete code. No TODO/TBD. No "add error handling" without specifics. No references to functions not defined in earlier tasks.

**Type consistency:**
- `JudgeState` fields consistent across Tasks 1, 3, 5, 9, 11
- `JudgeResult` fields consistent across Tasks 1, 4, 5, 9
- `apply_stage_target` return `(JudgeState, str)` consistent across Tasks 3, 9
- `_run_judges_post_turn` return `(dict, dict)` consistent across Tasks 9, 10, 11, 13
- Move values `advance/stay/regress/illegal_stay/fire_advance/basin_stay/no_judge` consistent across Tasks 3, 6 (loader reports none — n/a there), 9 (emit no_judge), 11 (health)
- `append_turn` kwargs `judge_output_protagonist / counterpart / director_injection` consistent across Tasks 9, 10, 13
- `write_meta` kwargs `judge_trajectories / judge_health / termination_turn / director_injections / interactive_mode` consistent Tasks 11, 13

All consistent.

---

## Plan complete

Plan saved to `docs/superpowers/plans/2026-04-22-level-4-judge.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch fresh subagent per task, spec + code quality review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session with checkpoints for review.

Which approach?
