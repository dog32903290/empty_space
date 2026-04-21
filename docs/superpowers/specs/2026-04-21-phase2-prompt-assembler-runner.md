# Phase 2：Prompt Assembler + Runner — Design Spec

**作者**：陳柏為 × Claude（蒼鷺）
**日期**：2026-04-21
**狀態**：design draft，待 review
**依附**：`docs/superpowers/specs/2026-04-21-空的空間-design.md`（主 spec）、`docs/phase1-summary.md`

---

## 0. 前情

Phase 1 完成：Persona / Setting / ExperimentConfig 載入，Gemini Flash + Pro client，扁平 module layout。還沒有任何 runtime——能讀材料、不能演。

Phase 2 目標：讓 `uv run python scripts/run_experiment.py <exp_id>` 跑出一場**完整的雙角色對話**，所有材料落盤到 `runs/<exp_id>/<timestamp>/`，人可以 Read，Phase 3+ 可以在此基礎上接線。

本 spec 對主 spec 的**三個偏離**（經討論後定案）：

1. **砍掉 scripted injection 機制**。主 spec §3.4 原本把 scripted_turns 當實驗嚴謹性的基石，本 phase 推翻：改用「導演塞世界，不塞嘴」的 emergent observation 路線。
2. **加入 scene_premise + director_events**。靜態開場前提 + 預寫事件 timeline，取代 scripted injection 的位置。
3. **砍掉 experiment config 的 `protagonist_opener` / `counterpart_system`**。貫通軸 + 關係層已完成角色塑形，不再需要 experiment-level 演員備註。

---

## 1. 範圍

### In（Phase 2 做）

- 雙角色輪流 turn loop（母親 protagonist 起手 → 兒子 counterpart → 母親 …）
- Prompt 組裝：system 四個塑形層（貫通軸 / 關係層 / 此刻 / 現場）+ 第五個結構約束 block（輸出格式）+ user 對話歷史
- Gemini Flash 呼叫角色，解析 `---IMPRESSIONS---` structured output
- Scene premise（靜態開場前提）+ director events（預寫 timeline）注入「現場」block
- 落盤：`config.yaml`、`turns/turn_N.yaml`、`conversation.md`、`conversation.jsonl`、`meta.yaml`
- CLI entry：`scripts/run_experiment.py <exp_id>`

### Out（留給後續 phase）

- Layer 2 Judge（stage / mode / 張力 / fire_release / basin_lock）→ Phase 3
  - 因此 Phase 2 終止條件只有 `max_turns`
  - turn yaml 不寫 `judge_state_after` 欄位
- Layer 3 Composer、21 格 snapshot、hash cache → Phase 5
- Layer 4 帳本（rubric 評分、accepted ledger、symbol index）→ Phase 4
  - 候選印象**有產、有落盤到 turn yaml**，但不評分、不累積帳本
- Ambient echo retrieval → Phase 4（Phase 2 的 `build_system_prompt` 留 `ambient_echo=[]` 參數）
- Dashboard → Phase 6
- Interactive CLI 導演 / LLM 導演 → 未來（Phase 2 只做預寫 timeline）
- Bootstrap skill 轉換腳本 → Phase 7
- `--resume` / crash recovery → 未來

### 副產出（Phase 1 schema / experiment yaml 要改）

- `schemas.py`：
  - 刪 `ScriptedTurn` class
  - 刪 `ExperimentConfig.scripted_turns`
  - 刪 `ExperimentConfig.protagonist_opener`
  - 刪 `ExperimentConfig.counterpart_system`
  - 加 `ExperimentConfig.scene_premise: str | None`
  - 加 `ExperimentConfig.director_events: dict[int, str]`
  - 加 `CandidateImpression` frozen dataclass
  - 加 `Turn` dataclass（Phase 2 產物）
  - 加 `SessionResult` dataclass（`run_session` 回傳）
- `experiments/mother_x_son_hospital_v3_001.yaml`：同步精簡
- 既有測試 (`test_schemas_experiment.py`、`test_loaders_experiment.py`、`test_integration_phase1.py`) 對應更新

---

## 2. Schema 變更

### 2.1 `schemas.py` diff

```python
# 刪除
class ScriptedTurn(BaseModel):
    speaker: Literal["protagonist", "counterpart"]
    content: str

# ExperimentConfig 舊欄位：
#     protagonist_opener: str
#     counterpart_system: str
#     scripted_turns: dict[int, ScriptedTurn] = Field(default_factory=dict)
# ↓ 替換為
    scene_premise: str | None = None
    director_events: dict[int, str] = Field(default_factory=dict)
```

精簡後 `ExperimentConfig` 欄位：

| 欄位 | 型別 | 說明 |
|---|---|---|
| `exp_id` | `str` | |
| `protagonist` | `PersonaRef` | |
| `counterpart` | `PersonaRef` | |
| `setting` | `SettingRef` | |
| `scene_premise` | `str \| None` | 開場前提，一次性寫入「現場」block |
| `initial_state` | `InitialState` | 動作詞 / 階段 / 模式 |
| `director_events` | `dict[int, str]` | `{turn_number: event_description}` |
| `max_turns` | `int` | 預設 20 |
| `termination` | `Termination` | Phase 2 只看 `max_turns`；flag 保留供 Phase 3 用 |

### 2.2 新 dataclasses

寫在 `schemas.py`：

```python
@dataclass(frozen=True)
class CandidateImpression:
    text: str
    symbols: list[str]

@dataclass
class Turn:
    turn_number: int
    speaker: Literal["protagonist", "counterpart"]
    persona_name: str                # e.g., "母親" — 供 build_user_message 格式化
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
    parse_error: str | None

@dataclass
class SessionResult:
    exp_id: str
    out_dir: Path
    total_turns: int
    termination_reason: Literal["max_turns"]  # Phase 3 會擴
    total_tokens_in: int
    total_tokens_out: int
    duration_seconds: float
```

### 2.3 Experiment yaml 精簡後

`experiments/mother_x_son_hospital_v3_001.yaml`：

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
# 第一輪先跑空 director events，看自然發展。
# 之後可加如：
#   10: "走廊盡頭傳來長音。護士小跑過去。"

max_turns: 20
termination:
  on_fire_release: true
  on_basin_lock: true
```

---

## 3. 模組骨架

### 3.1 檔案布局

```
src/empty_space/
├── paths.py              [Phase 1，不動]
├── schemas.py            [改：§2]
├── loaders.py            [Phase 1，不動]
├── llm.py                [Phase 1，不動]
├── prompt_assembler.py   [新增]
├── parser.py             [新增 — structured output 解析]
├── writer.py             [新增 — 落盤 I/O]
└── runner.py             [新增 — turn loop orchestrator]

scripts/
└── run_experiment.py     [新增 — CLI entry]
```

### 3.2 模組職責

| 檔 | 職責 | 類型 | 副作用 |
|---|---|---|---|
| `prompt_assembler.py` | 組 system prompt + user message | pure functions | 無 |
| `parser.py` | 切 Flash 回應為 `(main, impressions, parse_error)` | pure function | 無 |
| `writer.py` | 落盤所有輸出（turn yaml / md / jsonl / meta / config copy） | I/O functions | 檔案寫入 |
| `runner.py` | `run_session(config, llm_client) → SessionResult` 主編排 | top-level function | 委託 writer |

**Function-oriented，不是 class-oriented**。Phase 1 的 loaders / llm 已經是 function + dataclass 風格，Phase 2 延續。assembler 的每個 function 給 `(personas, history, state, events)` 產同一個 prompt 字串，純 function 最適合這種 deterministic shape。

### 3.3 `SessionState`（runner 內部狀態）

Local dataclass，只活在 `run_session` 呼叫期間，**不**寫進 `schemas.py`——是 runner 的實作細節：

```python
@dataclass
class SessionState:
    config: ExperimentConfig
    protagonist: Persona
    counterpart: Persona
    setting: Setting
    turns: list[Turn]                    # accumulated history
    active_events: list[tuple[int, str]] # director events triggered so far
    out_dir: Path
```

所有持久資料都落盤到 `out_dir`（遵守主 spec §5「檔案是共同真相源頭」原則）。

---

## 4. Prompt 組裝規則

### 4.1 System prompt 結構

```
## 貫通軸
<protagonist.core_text 或 counterpart.core_text，YAML verbatim>

## 關係層：對<對手 name>
<persona.relationship_texts[對手 name]，YAML verbatim>

## 此刻
動作詞：<initial_state.verb>
階段：<initial_state.stage>
模式：<initial_state.mode>

## 現場
<Setting.content，YAML verbatim>

### 場景前提
<scene_premise，若有；否則整個 sub-block 省略>

### 已發生的事
<director events 累積列表，若有；否則整個 sub-block 省略>
Turn 3：護士推一張空床進病房
Turn 10：走廊盡頭傳來長音。護士小跑過去。

## 輸出格式
先寫你要說的話。說完之後，另起一行寫 "---IMPRESSIONS---"，然後以 YAML list
格式列出你這輪浮現的印象句（若無，省略整段 ---IMPRESSIONS--- 區塊）。

範例：
---
她低著頭，沒有回答。

---IMPRESSIONS---
- text: "她的沉默在這一刻比任何辯解都沉"
  symbols: [沉默, 辯解, 愧疚]
---
```

**五個 `##` block 固定順序**：貫通軸 → 關係層 → 此刻 → 現場 → 輸出格式。

設計理由：

- **角色內在先於世界**（貫通軸 → 關係層 → 此刻），然後世界狀態，再對話從世界流入
- **現場 block 緊鄰 user message**（對話歷史）：director event 更新現場，下一輪對話受最即時的世界變化影響
- **輸出格式是結構約束，非角色塑形**：放最後，不污染前面四個 block 的語意
- YAML 內容 **verbatim** 塞入，不改寫、不截斷（Phase 1 的 `core_text` / `relationship_texts[name]` / `content` 原樣）
- 「此刻」三行來自 `ExperimentConfig.initial_state`；Phase 2 全程保持 initial 值（Phase 3 Judge 接線後每輪更新）

### 4.2 User message 規則

**Turn N ≥ 2**：純對話歷史，什麼都不加：

```
[Turn 1 母親] 你回來了。
[Turn 2 兒子] 嗯。
[Turn 3 母親] ⋯⋯
[Turn 4 兒子] 不關我的事。
[Turn 5 母親] 我知道你不認我。
```

**Turn 1**（空歷史）：`（場景開始。）`

規則：

- 歷史格式：`[Turn N <persona.name>] <content>`，一行一輪，按 turn 順序
- 角色名用 persona 的 `name` field（「母親」「兒子」），不是 role code
- **不加**任何 tail anchor / 指令 / 提醒——信任 system prompt 的塑形 + 現場 block 的世界狀態
- Turn 1 的「場景開始」僅為 Gemini SDK 的機械觸發（避免空字串），不是指令

### 4.3 Gemini 呼叫

使用 Phase 1 的 `GeminiClient.generate(system, user, model="gemini-2.5-flash")` 介面，不改。

### 4.4 Prompt assembler function 合約

```python
def build_system_prompt(
    persona: Persona,
    counterpart_name: str,
    setting: Setting,
    scene_premise: str | None,
    initial_state: InitialState,
    active_events: list[tuple[int, str]],
    ambient_echo: list[str] = [],  # Phase 4 預留，Phase 2 恆為 []
) -> str:
    ...

def build_user_message(
    history: list[Turn],
) -> str:
    """
    Turn 1 (history 空) → "（場景開始。）"
    Turn ≥ 2 → 每行 [Turn N <turn.persona_name>] <turn.content>
    """
    ...
```

---

## 5. Turn loop 狀態機

### 5.1 Main loop（`runner.run_session`）

```
1. 解析 config
   protagonist = load_persona(config.protagonist)
   counterpart = load_persona(config.counterpart)
   setting = load_setting(config.setting)

2. 準備輸出目錄
   timestamp = ISO-8601 (dash-separated, e.g. 2026-04-21T11-30-15)
   out_dir = paths.RUNS_DIR / config.exp_id / timestamp
   writer.init_run(out_dir, config)     # mkdir + copy config.yaml

3. state = SessionState(...)

4. for n in 1..config.max_turns:
     a. speaker 判定
          speaker = "protagonist" if n % 2 == 1 else "counterpart"
          speaker_persona = state.protagonist if speaker == "protagonist" else state.counterpart
          counterpart_name = (對手 persona 的 name)

     b. director event 觸發
          if n in config.director_events:
              state.active_events.append((n, config.director_events[n]))

     c. build system / user
          system = build_system_prompt(
              persona=speaker_persona,
              counterpart_name=counterpart_name,
              setting=state.setting,
              scene_premise=config.scene_premise,
              initial_state=config.initial_state,
              active_events=state.active_events,
              ambient_echo=[],  # Phase 4 接線
          )
          user = build_user_message(state.turns)

     d. LLM call
          resp = llm_client.generate(system=system, user=user, model="gemini-2.5-flash")

     e. parse
          main_content, impressions, parse_error = parser.parse_response(resp.content)

     f. 組 Turn + append state
          turn = Turn(
              turn_number=n,
              speaker=speaker,
              persona_name=speaker_persona.name,
              content=main_content,
              ...
          )
          state.turns.append(turn)

     g. 落盤（立即，不等 loop 結束）
          writer.append_turn(state, turn)

5. 終止
   termination_reason = "max_turns"   # Phase 2 唯一值

6. 收尾
   writer.write_meta(state, termination_reason, total_duration)

7. return SessionResult(...)
```

### 5.2 關鍵設計點

1. **Speaker 規則 Phase 2 寫死**：奇數輪 = protagonist 起手。Phase 3 若要支援 counterpart 起手，再加 config 欄位。
2. **Director events 累積不消**：world memory = append-only。Turn 3 觸發的事件，Turn 20 system prompt 的「### 已發生的事」還看得到。
3. **每輪立即落盤**：LLM 呼叫前中斷（Ctrl+C），已完成的 turn 完整保留。
4. **錯誤處理**：
   - LLM exception → propagate 出 `run_session`；已寫 turn 保留；`meta.yaml` 不寫
   - Parser 失敗 → `parse_error` 欄位記錯，session 繼續（§6 細節）
5. **中斷的 run**：`runs/<exp_id>/<timestamp>/` 有部分 `turn_*.yaml` 和 conversation 片段、沒 `meta.yaml`。Phase 2 不 resume，要重跑就新 timestamp。
6. **時間**：`turn.timestamp` = LLM call return 時；`out_dir` 的 timestamp = run 起始。`duration_seconds` = 收尾時算。

---

## 6. 候選印象 Parser

### 6.1 合約

Parser function：

```python
def parse_response(raw: str) -> tuple[str, list[CandidateImpression], str | None]:
    """
    Returns: (main_content, impressions, parse_error)

    - main_content: 永遠成功返回（主回應最高優先）
    - impressions: 成功解析的印象；失敗時 []
    - parse_error: None 表示乾淨；字串表示錯誤訊息（記 turn yaml 用）
    """
```

### 6.2 容錯表

| Flash 產出 | main | impressions | parse_error |
|---|---|---|---|
| 乾淨 format | 主回應 | 正確 list | None |
| 完全不產 marker | 全部當主回應 | `[]` | None |
| 有 marker 但 YAML 爛 | 切乾淨的主回應 | `[]` | YAML 錯誤訊息 |
| 有 marker 但 root 不是 list | 切乾淨的主回應 | `[]` | 格式訊息 |
| list 內某項缺 `text` 欄 | 主回應 | 只收有效項 | None（壞項靜默跳過） |
| `symbols` 缺失 | 主回應 | symbols=[] | None |

### 6.3 實作骨架

```python
MARKER = "---IMPRESSIONS---"

def parse_response(raw: str) -> tuple[str, list[CandidateImpression], str | None]:
    if MARKER not in raw:
        return raw.strip(), [], None

    main, _, impressions_block = raw.partition(MARKER)
    main = main.strip()

    try:
        parsed = yaml.safe_load(impressions_block)
        if not isinstance(parsed, list):
            return main, [], f"impressions block is not a list: {type(parsed).__name__}"

        impressions = []
        for item in parsed:
            if not isinstance(item, dict) or "text" not in item:
                continue
            impressions.append(CandidateImpression(
                text=str(item["text"]),
                symbols=list(item.get("symbols", [])),
            ))
        return main, impressions, None

    except yaml.YAMLError as e:
        return main, [], f"YAML parse error: {e}"
```

### 6.4 原則

- **主回應最高優先**：無論格式多爛，主回應永遠切出來，對話不中斷
- **印象是 optional quality signal**：丟了不致命
- **不做 retry**：retry 會藏問題，看 `parse_error` 欄位的統計分佈才知道要不要改 prompt 或升模型
- **還要測真實 Flash 常見變體**：前後空行、markdown code fence 包住、extra whitespace

---

## 7. 落盤 Schema

### 7.1 目錄結構（Phase 2 子集）

```
runs/<exp_id>/<timestamp>/
├── config.yaml          # experiment config 複本（重現用）
├── conversation.md      # 對話全文，人類可讀
├── conversation.jsonl   # 對話全文，結構化
├── turns/
│   ├── turn_001.yaml
│   ├── turn_002.yaml
│   └── ...
└── meta.yaml            # 收尾總結
```

（Phase 3 加 `judge/`；Phase 4 加 `ledger/`；Phase 5 加 `composer/`。Phase 2 只產以上。）

### 7.2 `turns/turn_N.yaml`（Phase 2 版）

```yaml
turn: 8
speaker: counterpart
timestamp: 2026-04-21T11:30:15Z
prompt_assembled:
  system: |
    ## 貫通軸
    ...（完整 system prompt verbatim，含五個 block：貫通軸 / 關係層 / 此刻 / 現場 / 輸出格式）
  user: |
    [Turn 1 母親] 你回來了。
    ...
    [Turn 7 母親] ⋯⋯
  tokens:
    system: 1842
    user: 316
response:
  content: |
    你從來沒有找過我。不是他不讓你找。是你沒有找。
  raw: |
    <Gemini 完整輸出，含 ---IMPRESSIONS--- block 若有>
  tokens_out: 35
  model: gemini-2.5-flash
  latency_ms: 412
candidate_impressions:
  - text: "他的指控像把她多年的迴避攤在燈下"
    symbols: [指控, 迴避, 攤開]
director_events_active:
  - turn: 3
    content: "護士推一張空床進病房"
parse_error: null
```

對照主 spec §5.2：**Phase 2 砍掉 `judge_state_after` 欄位**（Judge 是 Phase 3）。

### 7.3 `conversation.md`（每輪 append）

```markdown
# mother_x_son_hospital_v3_001 @ 2026-04-21T11-30-15

**場景**：他們在同一家醫院。父親在 ICU...

---

**Turn 1 · 母親**
你回來了。

**Turn 2 · 兒子**
嗯。

**[世界] Turn 3：護士推一張空床進病房**

**Turn 3 · 母親**
⋯⋯

**Turn 4 · 兒子**
不關我的事。

...
```

- 開頭 `scene_premise` 做引子
- 每輪：`**Turn N · <name>**\n<content>\n\n`
- Director event 插在**觸發那輪之前**一行：`**[世界] Turn N：<event>**`
- 只寫主回應，不寫印象（印象是內部產物，查 turn yaml）

### 7.4 `conversation.jsonl`（每輪 append）

每行一個 JSON object：

```json
{"turn": 1, "speaker": "protagonist", "name": "母親", "content": "你回來了。", "timestamp": "2026-04-21T11:30:01Z"}
{"turn": 2, "speaker": "counterpart", "name": "兒子", "content": "嗯。", "timestamp": "2026-04-21T11:30:04Z"}
{"type": "director_event", "turn": 3, "content": "護士推一張空床進病房"}
{"turn": 3, "speaker": "protagonist", "name": "母親", "content": "⋯⋯", "timestamp": "2026-04-21T11:30:09Z"}
```

Director event 用 `"type": "director_event"` 區別對話 turn。

### 7.5 `meta.yaml`（收尾一次）

```yaml
exp_id: mother_x_son_hospital_v3_001
run_timestamp: 2026-04-21T11-30-15
total_turns: 20
termination_reason: max_turns
total_tokens_in: 38420
total_tokens_out: 2143
duration_seconds: 184.7
total_candidate_impressions: 14
turns_with_parse_error: 2
director_events_triggered:
  - turn: 3
    content: "護士推一張空床進病房"
models_used:
  - gemini-2.5-flash
```

### 7.6 原子性

- `turn_N.yaml`、`meta.yaml`：寫到 `*.tmp` → `os.replace` 到最終檔名（POSIX atomic rename）
- `conversation.md` / `conversation.jsonl`：append 模式（不需 atomic；append 失敗 worst case 是半行）

### 7.7 不變量（Phase 2 版，對應主 spec §5.6）

- 一個 run 永不覆寫：重跑就新 timestamp
- 已寫的 `turn_N.yaml` 不改（append-only semantics）
- `out_dir` 裡的檔案組成可自包含**重現**那次 run 的 prompt / response / 時序

---

## 8. 測試策略

### 8.1 檔案

```
tests/
├── test_schemas_experiment.py    [改] 加 scene_premise / director_events；砍 scripted_turns / opener
├── test_loaders_experiment.py    [改] 同上
├── test_integration_phase1.py    [改] 清除涉及 scripted / opener 的斷言
├── test_prompt_assembler.py      [新]
├── test_parser.py                [新]
├── test_writer.py                [新]
└── test_runner_integration.py    [新，核心信心來源]
```

### 8.2 `test_prompt_assembler.py`

涵蓋 `build_system_prompt`：

- 五個 block 順序正確
- 關係層 header 帶對手名（「## 關係層：對兒子」）
- 無 `scene_premise` 時 `### 場景前提` sub-block 省略
- 無 director events 時 `### 已發生的事` sub-block 省略
- 多個 events 按 turn 順序列
- 輸出格式指示包含 `---IMPRESSIONS---` marker 字串
- YAML 內容 verbatim（snapshot 對比）

涵蓋 `build_user_message`：

- Turn 1（空 history）→ `（場景開始。）`
- Turn ≥ 2 → 每行 `[Turn N <turn.persona_name>] <turn.content>`，順序與換行正確
- 尾端無 tail anchor / 指令

### 8.3 `test_parser.py`

逐一覆蓋 §6.2 容錯表：

- 乾淨 format：main + impressions 正確
- 無 marker：main=原文，impressions=[]
- marker 後 YAML 爛：main 乾淨切出，impressions=[]，parse_error 有訊息
- marker 後不是 list：同上
- list 某項缺 text：靜默跳過壞項
- symbols 缺失 → 預設 []
- 前後空行 / markdown code fence 包住：能救

### 8.4 `test_writer.py`

- `init_run`：mkdir、config.yaml 寫入、內容一致
- `append_turn`：turn_N.yaml 原子寫入；conversation.md / jsonl append 後讀回正確
- Director event 觸發那輪，conversation.md 含 `**[世界] Turn N：...**`、jsonl 含 `"type": "director_event"`
- `write_meta`：所有統計欄位正確

### 8.5 `test_runner_integration.py`（核心）

用 **MockLLMClient**（可預先排程每輪回應）跑整場 session，不打真實 API。

Case：

1. **Happy path**：20 輪排程 → out_dir 結構完整；turn_001..020 全在；meta.total_turns == 20
2. **Speaker alternation**：mock.calls 每輪 system prompt 含正確 persona 的貫通軸（奇數母親 / 偶數兒子）
3. **Director event 注入**：`director_events={3: "護士..."}`，Turn 4 後 mock.calls[3].system 含 `### 已發生的事`、`Turn 3：護士...`
4. **Event 累積不消**：`{3: "A", 10: "B"}`，Turn 15 的 system 同時含 A 和 B
5. **Parse error 落盤**：排程一則 YAML 爛的 impressions → turn yaml 的 `parse_error` 非 null、session 照常跑完
6. **Max turns 終止**：max_turns=5 → 只跑 5 輪；meta.termination_reason == "max_turns"
7. **LLM 例外中斷**：mock 第 5 輪 raise → session 中斷；turn_001..004 完整；`meta.yaml` 不存在
8. **不同 timestamp 不覆寫**：同 exp_id 連跑兩次 → 兩個 timestamp 目錄並存

### 8.6 真實 API smoke test（手動，不進 CI）

`scripts/smoke_run.py`：跑一場 `max_turns=3` 打真 Gemini，驗整條 pipeline 沒 SDK integration 問題。柏為手動跑。

### 8.7 不做

- 不測 Gemini 回應品質（主 spec §8 人工盲審範疇，和 Phase 2 無關）
- 不做 property-based testing（Phase 2 範圍 overkill）

---

## 9. CLI Entry

### 9.1 `scripts/run_experiment.py`

```python
"""Run a single experiment session.

Usage:
    uv run python scripts/run_experiment.py <exp_id>

Example:
    uv run python scripts/run_experiment.py mother_x_son_hospital_v3_001
"""
import sys
from empty_space.loaders import load_experiment
from empty_space.llm import GeminiClient
from empty_space.runner import run_session


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: run_experiment.py <exp_id>", file=sys.stderr)
        return 2

    exp_id = sys.argv[1]
    config = load_experiment(exp_id)
    client = GeminiClient()

    result = run_session(config, client)

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

### 9.2 行為

- 失敗非零 exit code；API exception 直接冒出 Python traceback
- 成功輸出幾行 summary
- **故意薄**：不做 `--help`、`--dry-run`、`--max-turns`、progress bar、live streaming
  - 想看進度：開另一個 terminal `tail -f runs/<exp_id>/<timestamp>/conversation.md`
- **不用 argparse / click / typer**：Phase 2 只有一個 positional arg，手刻 `sys.argv` 檢查 5 行內

### 9.3 未來延伸點（不現在做）

- `--resume`：從中斷 timestamp 續跑
- `--max-turns=N`：override config（Phase 3 可能加）
- batch mode：一次跑多個 experiment

---

## 10. Phase 2 完成條件

- [ ] Schema 變更落地，既有測試全綠
- [ ] `prompt_assembler.py` / `parser.py` / `writer.py` / `runner.py` 完成
- [ ] 新測試全綠（`test_prompt_assembler.py` / `test_parser.py` / `test_writer.py` / `test_runner_integration.py`）
- [ ] `scripts/run_experiment.py` 能跑 `mother_x_son_hospital_v3_001` 產出完整 20-turn 對話
- [ ] 手動 smoke run 成功，人讀 `conversation.md` 不出戲
- [ ] `runs/` 目錄結構、`turn_N.yaml` / `conversation.md/jsonl` / `meta.yaml` 欄位與 §7 一致
- [ ] Phase 2 summary doc（對照 `docs/phase1-summary.md` 格式）寫完

## 11. 接下來

1. 柏為 review 本 spec
2. Spec 定稿 → commit 到 `docs/superpowers/specs/`
3. 進 writing-plans 技能 → 拆 implementation plan
4. 開工

---

## 附錄 A：本 spec 對主 spec 的 delta 摘要

| 主 spec 立場 | 本 spec 立場 | 理由 |
|---|---|---|
| Scripted injection 是實驗嚴謹性基石（§3.4） | 砍掉 scripted injection | Emergent observation > controlled alignment；導演塞世界不塞嘴 |
| `protagonist_opener` / `counterpart_system` 是 experiment-level 角色指示 | 砍掉兩欄位 | 貫通軸 + 關係層已塑形完畢；演員不需備註 |
| Turn yaml 含 `judge_state_after` | Phase 2 不寫此欄 | Judge 是 Phase 3 |
| System prompt 含 ambient echo | Phase 2 恆為空 | Ledger 是 Phase 4 |

## 附錄 B：參照

- 主 spec：`docs/superpowers/specs/2026-04-21-空的空間-design.md`
- Phase 1 summary：`docs/phase1-summary.md`
- Phase 1 plan：`docs/superpowers/plans/2026-04-21-phase1-infrastructure.md`
