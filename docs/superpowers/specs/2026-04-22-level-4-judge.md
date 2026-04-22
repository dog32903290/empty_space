# Level 4: Judge 狀態機 + 導演互動鉤子 Spec

> 設計日期 2026-04-22。Level 4 在 Level 1-3（對話 runner + 帳本 + Composer）之上，加入 per-speaker Judge 狀態機與導演在情緒峰值的互動介入。

## 1. 目標 (Goals)

**為每個 speaker 獨立維護 (stage, mode) 二維狀態，讓 `## 此刻` system prompt 區塊隨對話演化；並在 fire_release / basin_lock 等情緒峰值提供導演事件注入入口。**

- **動態此刻**：system prompt 的 `## 此刻` 不再是整場 static `initial_state`，而是每輪重讀該 speaker 的當前 JudgeState 並查 `stage_mode_contexts_v3` 對應格子的身體/語聲/注意力文字
- **雙 Judge 並行**：母親與兒子各有自己的 JudgeState，獨立演化，不強迫同步
- **ratchet 語意**：stage 只能沿 7 格序列相鄰移動；fire_release 允許 +2；basin_lock 強制 stay
- **導演互動**：`--interactive` flag 開啟後，verdict 為 fire_release / basin_lock 時阻塞 stdin 讓導演選擇是否注入事件
- **提早結束**：雙方同時連續 2 輪 basin_lock → dual_basin_lock terminate

## 2. 非目標 (Non-goals)

- 不自動生成 21 格 context（留給 Level 5 的 auto-context pipeline）
- 父親/繼女不走 Judge（沒有 v3 檔案，runner skip）
- Judge 不看 retrieval / composer 輸出（狀態與記憶解耦是設計選擇）
- 兩個 Judge 不讀彼此 state history（B 策略精神延伸）
- Composer 不讀 Judge trajectories
- 不做 web UI 的導演介入（MVP 用 terminal stdin）

## 3. 核心資料結構 (Data Structures)

### 3.1 `JudgeState`（新）

```python
@dataclass
class JudgeState:
    speaker_role: Literal["protagonist", "counterpart"]
    stage: str
    mode: str
    last_why: str = ""
    last_verdict: str = ""
    move_history: list[str] = field(default_factory=list)
    verdict_history: list[str] = field(default_factory=list)
    hits_history: list[list[str]] = field(default_factory=list)
```

### 3.2 `Persona` 擴充

新增兩欄（可空，無 v3 時留空）：

```python
judge_principles_text: str = ""
stage_mode_contexts_parsed: dict[str, dict[str, str]] = field(default_factory=dict)
```

`stage_mode_contexts_parsed` 結構：
```python
{
  "前置積累_收": {"身體傾向": "...", "語聲傾向": "...", "注意力": "..."},
  "前置積累_放": {...}, "前置積累_在": {...},
  ... # 21 格
}
```

### 3.3 `SessionState` 擴充

```python
judge_state_protagonist: JudgeState | None = None
judge_state_counterpart: JudgeState | None = None
```

### 3.4 `SessionResult` 擴充

```python
termination_reason: str   # "max_turns_reached" | "dual_basin_lock"
termination_turn: int
judge_trajectories: dict  # 見 §8 落盤
director_injections: list[dict]
interactive_mode: bool
judge_health: dict
```

### 3.5 常數

```python
STAGE_ORDER = ["前置積累", "初感訊號", "半意識浮現", "明確切換",
               "穩定期", "回溫期", "基線"]
MODES = ["收", "放", "在"]
JUDGE_MODEL = "gemini-2.5-flash"
```

## 4. `judge.py` 新模組

### 4.1 函式表

| 函式 | 職責 |
|---|---|
| `parse_judge_principles(text: str) -> str` | Identity（原字串塞進 prompt） |
| `parse_stage_mode_contexts(raw: dict) -> dict` | 把 v3 yaml 轉成 `{"stage_mode": {身體傾向,語聲傾向,注意力}}` |
| `build_judge_prompt(last_state, principles, recent_turns, speaker_role) -> tuple[str, str]` | 回 (system, user) |
| `parse_judge_output(text: str, last_state: JudgeState) -> JudgeResult` | 寬容解析 5 行輸出 |
| `apply_stage_target(last_state, proposed_stage, proposed_mode, proposed_verdict) -> tuple[JudgeState, str]` | ratchet 閘門 |
| `run_judge(last_state, principles, recent_turns, llm_client, speaker_role) -> JudgeResult` | 一次完整 Judge 呼叫（含 LLM 失敗 fallback） |
| `is_fire_release(state: JudgeState) -> bool` | last_verdict == "fire_release" |
| `is_basin_lock(state: JudgeState) -> bool` | last_verdict == "basin_lock" |

### 4.2 `JudgeResult` dataclass

```python
@dataclass
class JudgeResult:
    proposed_stage: str
    proposed_mode: str
    proposed_verdict: str  # fire_release | basin_lock | N/A
    why: str
    hits: list[str]
    meta: dict  # {tokens_in, tokens_out, latency_ms, model, parse_status, error?}
```

## 5. Judge Prompt 設計

### 5.1 System prompt（固定）

```
你是戲劇裡的「隱性量測者」。你不介入對話、不評分、不給建議。
你只做一件事：根據這個角色最近說的話、做的動作、身體狀態，
判斷他在 stage × mode 二維空間裡「下一刻」會落在哪一格。

規則：
- STAGE 只能沿序列相鄰移動：前置積累 → 初感訊號 → 半意識浮現 → 明確切換 → 穩定期 → 回溫期 → 基線
  - 可以 advance、stay、regress，不能跳格
- MODE 是當下的身體傾向：收 / 放 / 在
- VERDICT 標記特殊事件：fire_release | basin_lock | N/A
- HITS 是你觀察到的具體線索

輸出格式（5 行，嚴格）：
STAGE: <stage 名>
MODE: <mode 名>
WHY: <一句話>
VERDICT: <fire_release | basin_lock | N/A>
HITS: <line1; line2; line3>
```

### 5.2 User prompt（動態）

```
# 角色原則
{persona.judge_principles_text}

# Stage × Mode 脈絡
{persona.stage_mode_contexts_text}   # 21 格白話文

# 上一輪狀態
STAGE: {last_state.stage}
MODE: {last_state.mode}
LAST_WHY: {last_state.last_why}

# 最近對話（最多 3 輪）
{recent_turns_formatted}

# 任務
根據以上，只判斷 {speaker_role}（{persona_name}）這個角色，
輸出他「剛說完這輪話之後」的 stage/mode/why/verdict/hits。
```

## 6. `apply_stage_target` ratchet 閘門

按順序檢查：

1. STAGE 合法性：只允許 diff ∈ {-1, 0, +1}，否則 `illegal_stay`
2. `fire_release` 放寬到 diff=+2 → `fire_advance`（+3 仍擋）
3. `basin_lock` 強制 stay → `basin_stay`
4. MODE 無合法性檢查（自由切換），但不在 MODES 清單則 fallback 到 last_mode

move 值：`advance | stay | regress | illegal_stay | fire_advance | basin_stay | no_judge`

## 7. `prompt_assembler` 擴充

`build_system_prompt` 加兩個參數：

```python
judge_state: JudgeState,
stage_mode_contexts: dict[str, dict[str, str]] | None,
```

`## 此刻` 區塊改為：

```
## 此刻
- 階段：{judge_state.stage}
- 模式：{judge_state.mode}
- 身體傾向：{cell_context["身體傾向"]}
- 語聲傾向：{cell_context["語聲傾向"]}
- 注意力：{cell_context["注意力"]}
```

若 `stage_mode_contexts` 為 None 或查不到 cell，fallback 為只顯示 stage/mode 兩行（完全沿用 Level 3 行為）。

其他 block（貫通軸 / 關係層 / 現場 / 你的內在 / 輸出格式）不變。

## 8. Runner 接線

### 8.1 生命週期

- **Turn 0**：從 `config.initial_state` 初始化兩個 JudgeState（`_init_judge_state`）
- **每輪結束**：對 protagonist + counterpart 各跑一次 Judge → 更新各自 state
- **下輪開始**：assembler 讀當前 speaker 的 state

### 8.2 skip Judge 條件

```python
def _should_run_judge(persona: Persona) -> bool:
    return bool(persona.judge_principles_text) and bool(persona.stage_mode_contexts_parsed)
```

skip 時：state 不變，落盤 `{skipped: true, reason: "no_v3_config"}`。

### 8.3 Interactive mode

CLI：`--interactive` flag。peak 定義為任一方 `last_verdict ∈ {fire_release, basin_lock}`。peak 時：

1. Terminal 印兩人當前 state + verdict
2. `input()` 阻塞讀導演事件文字
3. 空行 → 跳過；非空 → 寫入 `session_state.director_events[next_turn_idx]`
4. 下一輪 assembler 會把該事件塞進 `## 現場` 底部（復用 Level 1 機制）

## 9. 終止 (Termination)

三條件任一：

1. `max_turns_reached`（保底）
2. `dual_basin_lock`：雙方 verdict_history 末 N 個全為 "basin_lock"（N=2，可 config）
3. （保留）顯式 config 終止 — Level 4 不實作

單方 basin_lock 不退出。fire_release 不觸發退出。

## 10. 落盤 Schema

### 10.1 `turn_{idx:03d}.yaml` 新增

```yaml
judge_output_protagonist:
  proposed: {stage, mode, verdict, why}
  applied:  {stage, mode, move}
  hits: [...]
  meta: {tokens_in, tokens_out, latency_ms, model, parse_status, error?}
judge_output_counterpart:
  ...
director_injection:   # 可選（導演介入時才有）
  event: "..."
  triggered_by: "fire_release on protagonist"
  applied_to_turn: <next_turn_idx>
```

skip 情況：
```yaml
judge_output_protagonist:
  skipped: true
  reason: "no_v3_config"
```

### 10.2 `session.yaml` 新增

```yaml
termination:
  reason: "dual_basin_lock"
  turn: 14
judge_trajectories:
  protagonist:
    stages: [...]
    modes:  [...]
    moves:  [...]
    verdicts: [...]
  counterpart: {...}
director_injections:
  - {turn: 8, event: "..."}
interactive_mode: true | false
judge_health:
  protagonist:
    total_calls: 14
    ok: 12
    parse_fallback: 1
    llm_error: 1
    no_judge: 0
  counterpart: {...}
```

### 10.3 Persona 載入

`persona_loader.py` 嘗試讀 `judge_principles_v3.yaml` + `stage_mode_contexts_v3.yaml`，缺一就留空（Judge 整個 skip）。

### 10.4 `InitialState` v3 遷移

既有 experiment.yaml 的 `initial_state.mode: "基線"` 需改寫為 `"在"`。`InitialState` pydantic schema 加 validator：收到 `"基線"` 自動轉 `"在"`（軟遷移）。3 個既有 experiment yaml（醫院/車上/家）做一次性改寫。

## 11. 錯誤處理

| 場景 | 處理 |
|---|---|
| 無 v3 檔案 | skip Judge，state 不變，meta 記 `skipped=true` |
| LLM 失敗 | return last_state + verdict=N/A + meta.error，對話不中斷 |
| Parse 爛 | 寬容抽 STAGE/MODE/VERDICT/WHY/HITS，不中者 fallback last_state，parse_status="fallback_used" |
| stage 跳格 | ratchet 閘門擋下，move="illegal_stay" |
| mode 不在清單 | fallback last_state.mode |
| stdin EOF (interactive) | 當空行處理，不 crash |

`judge_health` 統計 >20% 失敗時 session.yaml 寫 warning。

## 12. 測試策略

### 12.1 Layer 1：單元（judge.py，MockLLMClient）

`tests/test_judge.py`：
- `parse_judge_output` happy path + 寬容（全形冒號、開場白、stage 同義、漏行）
- `apply_stage_target`：advance / stay / regress / illegal_stay / fire_advance / basin_stay
- `run_judge` LLM 失敗 fallback
- `parse_stage_mode_contexts` 21 格解析
- `parse_judge_principles` identity

### 12.2 Layer 2：整合（runner，MockLLMClient）

`tests/test_runner_level4.py`：
- test_judge_runs_per_turn_per_speaker
- test_judge_state_evolves_across_turns
- test_per_speaker_independence
- test_dual_basin_lock_terminates
- test_single_basin_not_terminate
- test_fire_release_allows_jump
- test_no_v3_config_skips_judge
- test_judge_llm_error_does_not_crash_session
- test_initial_state_vocabulary_v3
- test_director_injection_in_interactive_mode（monkeypatch stdin）

### 12.3 Layer 3：Smoke（scripts/smoke_level4.py，真 Flash）

- 一場 6 輪醫院 session（batch）
- 人眼檢查 judge_trajectories 合理性
- 手跑一次 --interactive 驗證事件注入

## 13. 風險與未知數

| 風險 | 徵兆 | 對策 | 現在處理？ |
|---|---|---|---|
| Judge mode 偏向 | modes 分布極端 | prompt 加 MODE_傾向分布 | 在 prompt 裡處理 |
| basin_lock 過早觸發 | termination.turn 常 <10 | consecutive_required 設 config | 做成可調 |
| 兩 Judge 共振 | stages 高度同步 | prompt 強調「只判斷該 speaker」 | 在 prompt 裡處理 |
| interactive 疲勞 | — | `skip-all` / `--interactive-rate` | 延後 |
| v3 詞彙遷移 | 舊檔案 mode="基線" | validator + 一次性改寫 | 做 |
| Flash token 成本 | 單場 >$0.02 | 監控 judge_health，超支減 context | 監控即可 |
| 父繼女無 v3 crash | assembler None | fallback path + 測試 | 做 |
| Peak 時機感 | 主觀 | Level 4.1 累積樣本校準 | 延後 |

## 14. CLI 變更

```bash
# 原有
uv run empty-space run <experiment.yaml>

# 新增
uv run empty-space run <experiment.yaml> --interactive
```

無其他 CLI 變更。

## 15. 非本次 scope

- 21 格 context 自動生成（Level 5）
- Web UI 導演介入（Level 5+）
- Judge 與 Composer / Retrieval 的整合（Level 5+）
- Peak 時機校準離線 pipeline（Level 4.1）
- 更多角色（父親/繼女）的 v3 檔案寫作（persona 作者任務）
