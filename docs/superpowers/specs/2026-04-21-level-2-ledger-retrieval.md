# Level 2：跨 session 印象帳本 + 導演前情 — Design Spec

**作者**：陳柏為 × Claude（蒼鷺）
**日期**：2026-04-21
**狀態**：design draft，待 review
**依附**：
- 主 spec：`docs/superpowers/specs/2026-04-21-空的空間-design.md`
- Phase 2 spec：`docs/superpowers/specs/2026-04-21-phase2-prompt-assembler-runner.md`
- Phase 2 summary：`docs/phase2-summary.md`

---

## 0. 前情：Level 2 是什麼

Level 1（Phase 2）已 ship `437b23d` + tag `level-1`。母親 × 兒子 @ 醫院跑 4 輪，方法論穿透 Gemini Flash，單場對話能長出「沒有攻擊，沒有要求⋯⋯比任何質問都更難抵擋」這種句子。Level 1 驗證的是**單場**的塑形力。

Level 2 的命題：**讓兩個演員跨場累積、帶著前場的印象進入下一場**。Peter Brook 命題不變（兩個演員 + 一個看著的人），時間維度打開。

### 本 spec 對主 spec 的核心偏離

Phase 2 spec 推翻了主 spec 的 scripted injection；Level 2 繼續深化：

1. **rubric 評分砍掉**。Flash 的 candidate impression 自標記直接 append 進帳本，不做 per-impression 品質閘門
2. **Composer bake 延後到 Level 4**。Level 2 只做 append-only ledger + symbol-based retrieval
3. **Judge 延後到 Level 3**。Level 2 終止條件仍只有 max_turns
4. **導演的控制點從 scripted injection → scene_premise (Level 1 已做) → prelude (Level 2 新增)**。導演塞世界、塞心，不塞嘴

### 柏為的核心 insight

「記憶不是 engine 自動 retrieve 的東西——是導演寫的」

演員進場時帶著什麼，由導演**主觀選擇**（`prelude` 欄位手寫）。Engine 的工作是：
- 把導演寫的前情拆成 symbols
- 去跨 session 帳本撈命中的歷史印象
- 塞進角色的 system prompt 的「你的內在」block

這和 spec §4.1 原設計「自動 retrieval worker per-turn」根本不同。Level 2 是**「導演 query、engine 撈」**，不是 engine 自主提取。

---

## 1. 範圍

### In（Level 2 做）

- **Ledger** — 每對關係兩本帳本（`<A>_x_<B>.from_<A>.yaml` + `from_<B>.yaml`），append-only
- **帳本內索引** — symbol_index（反向索引）+ cooccurrence（1-hop graph 邊）
- **Session 結束 append** — candidates 依 speaker 分流入帳、兩個索引 incremental 更新
- **Session 開始 retrieval** — 每個角色：Flash 拆 prelude + scene_premise 的 symbols → 1-hop co-occurrence expansion → 查兩本帳本 → top-3 by symbol hit count
- **同義詞字典** — `config/symbol_synonyms.yaml` 手工維護的 canonical map，只作用於 matching step
- **experiment.yaml 新增** — `protagonist_prelude` / `counterpart_prelude`（選填）
- **system prompt 新增** — `## 你的內在` block，位置在 `## 現場` 之後、`## 輸出格式` 之前
- **新模組** — `ledger.py`（I/O + index 維護）+ `retrieval.py`（symbol extract + expansion + scoring）
- **落盤** — 每場新增 `retrieval.yaml`（audit trail）、`turn_NNN.yaml` 加 `retrieved_impressions` 欄位、`meta.yaml` 加 retrieval token + ledger append 摘要
- **CLI entry** — 不動，`run_experiment.py` 照 Phase 2 行為

### Out（不做，留給後續 levels）

- **Rubric 評分** — 個別候選印象的品質閘門（Flash 自標記即信任）
- **Judge（Level 3）** — stage / mode / 張力追蹤 / fire_release / basin_lock
- **Composer（Level 4）** — 21 格情緒弧線 + 跨 session「一天後」濃縮 + graph consolidation + refined 帳本
- **Per-turn retrieval / agentic pull** — 演員主動 call recall tool（Level 3+）
- **多跳 graph traversal** — 只做 1-hop co-occurrence expansion
- **跨 relationship pair 的帳本共享** — 每對關係一組帳本
- **Embedding-based semantic retrieval** — 純 symbol lexical match
- **Dashboard 顯示** — Phase 6，靠 Read 檔案
- **跨實驗統計分析** — 累積材料，之後寫別的工具

### 副產出（schema/既有檔案要改）

- `schemas.py`：擴充 ExperimentConfig（加 prelude）、SessionState（加 retrieval）；新增 Ledger、LedgerEntry、RetrievedImpression、RetrievalResult dataclasses
- `prompt_assembler.py`：`build_system_prompt` 簽名 + 組「你的內在」block
- `runner.py`：session-start retrieval 區塊、session-end ledger append
- `writer.py`：新增 `write_retrieval`、擴充 turn yaml 欄位、擴充 meta.yaml 欄位
- `experiments/mother_x_son_hospital_v3_001.yaml`：可選加 prelude 展示

### Ship 完的樣子

`uv run python scripts/run_experiment.py mother_x_son_hospital_v3_001` 能連跑兩場：
- **第一場**：帳本空、retrieval 零命中、「你的內在」block 只有 prelude 或整個省略
- **第二場**：帳本累積了第一場的 candidates，session-start retrieval 撈到、塞進 system prompt；演員帶著前場的印象進場

---

## 2. Schema 變更

### 2.1 `ExperimentConfig` 新增兩欄

```python
class ExperimentConfig(BaseModel):
    # ... Phase 2 既有 ...
    protagonist_prelude: str | None = None
    counterpart_prelude: str | None = None
```

兩者都是選填。沒寫就是 `None`。

### 2.2 experiment yaml 範例

```yaml
exp_id: mother_x_son_hospital_v3_002_scene_B
protagonist:
  path: 六個劇中人/母親
  version: v3_tension
counterpart:
  path: 六個劇中人/兒子
  version: v3_tension
setting:
  path: 六個劇中人/環境_醫院.yaml

scene_premise: |
  一大早，兒子跟媽媽在醫院因為警察打來的電話相見，
  他們說爸爸現在重病。

protagonist_prelude: |
  你昨夜夢到他小時候被帶走。

counterpart_prelude: |
  你昨晚和女朋友分手。她說「我不能等你。」
  你還沒跟任何人說。

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

### 2.3 System prompt 結構（Phase 2 → Level 2）

```
## 貫通軸                    ← stable 跨 session
## 關係層：對 <對手>           ← stable 跨 session
## 此刻                       ← session config (initial_state)
## 現場                       ← Setting + scene_premise + director events
## 你的內在                    ← NEW 在此，最靠 user message（抵 attention dilution）
  [導演手寫 prelude（如果有）]
  [ledger retrieved top-3 印象（如果有，以 "你可能想起的：" 為 prefix 的 bullet list）]
## 輸出格式
```

`## 你的內在` block 整個**有條件省略**：
- 無 prelude 且 retrieval 零命中 → 整個 block 不出現
- 有 prelude 無 retrieval → block 只含 prelude
- 無 prelude 有 retrieval → block 只含「你可能想起的：」list
- 兩者皆有 → prelude 段 + 空行 + 「你可能想起的：」段

Phase 2 決定的「現場最靠 user message」這個原則在 Level 2 被打破，刻意取捨（抵 attention dilution > 現場 immediacy）。柏為註記：如果實驗發現塞不下、可以跑去調換「現場」和「你的內在」順序。

### 2.4 新 dataclasses（`schemas.py`）

```python
@dataclass
class LedgerEntry:
    """One candidate impression persisted in a ledger."""
    id: str                  # imp_001, imp_002, ...
    text: str
    symbols: list[str]
    from_run: str            # e.g. mother_x_son_hospital_v3_001/2026-04-21T10-24-12
    from_turn: int
    created: str             # ISO 8601


@dataclass
class Ledger:
    """In-memory representation of a single <relationship>.from_<persona>.yaml."""
    relationship: str
    speaker: Literal["protagonist", "counterpart"]
    persona_name: str
    ledger_version: int
    candidates: list[LedgerEntry]
    symbol_index: dict[str, list[str]]        # symbol → [imp_id, ...]
    cooccurrence: dict[str, dict[str, int]]   # symbol_a → symbol_b → count


@dataclass(frozen=True)
class RetrievedImpression:
    """Read from ledger; what went into the '你的內在' block."""
    id: str
    text: str
    symbols: list[str]              # entry.symbols 原樣（未 canonicalize）
    speaker: Literal["protagonist", "counterpart"]
    persona_name: str
    from_run: str
    from_turn: int
    score: int                      # len(matched_symbols)
    matched_symbols: list[str]      # canonical 形式的交集（query ∩ entry 在 canonicalize 後）


@dataclass
class RetrievalResult:
    """Session-start retrieval outcome for one role."""
    speaker_role: Literal["protagonist", "counterpart"]
    persona_name: str
    query_text: str                  # scene_premise + prelude joined
    query_symbols: list[str]         # Flash extract 原始輸出
    expanded_symbols: list[str]      # + co-occurrence 鄰居
    impressions: list[RetrievedImpression]
    flash_latency_ms: int
    flash_tokens_in: int
    flash_tokens_out: int
```

### 2.5 SessionState 擴充（`runner.py` 內部 dataclass）

```python
@dataclass
class SessionState:
    # Phase 2 既有
    config: ExperimentConfig
    protagonist: Persona
    counterpart: Persona
    setting: Setting
    turns: list[Turn] = field(default_factory=list)
    active_events: list[tuple[int, str]] = field(default_factory=list)
    # Level 2 新增
    retrieval_protagonist: RetrievalResult | None = None
    retrieval_counterpart: RetrievalResult | None = None
```

### 2.6 不動的東西

- Persona / Setting schema：不動。prelude 不是 persona 資產，是 experiment 層的設計
- Phase 2 既有的 Turn / SessionResult / CandidateImpression：不動
- `scripts/run_experiment.py`：不動（CLI interface 一樣）

---

## 3. 帳本 schema

### 3.1 檔案位置與命名

```
ledgers/
├── 母親_x_兒子.from_母親.yaml
└── 母親_x_兒子.from_兒子.yaml
```

- Directory：`ledgers/` 已於 Phase 1 設置，已 gitignore
- 命名：`<protagonist_name>_x_<counterpart_name>.from_<persona_name>.yaml`
- 名稱用 `persona.name`（v3_tension 的「母親」「兒子」），不是 yaml 路徑。這讓帳本跨 yaml 重構仍穩定

### 3.2 完整 schema 範例

```yaml
# ledgers/母親_x_兒子.from_母親.yaml
relationship: 母親_x_兒子
speaker: protagonist
persona_name: 母親
created: 2026-04-21T10-24-12Z
last_updated: 2026-04-22T14-02-45Z
ledger_version: 3                    # 每次 append session 後 +1

candidates:
  - id: imp_001
    text: "消毒水的味道很淡，但一直都在。"
    symbols: [消毒水, 淡, 持續]
    from_run: mother_x_son_hospital_v3_001/2026-04-21T10-24-12
    from_turn: 1
    created: 2026-04-21T10-24-15Z
  - id: imp_002
    text: "她的手在膝上輕輕搭著，那雙手沒有動"
    symbols: [手, 膝, 搭, 不動]
    from_run: mother_x_son_hospital_v3_001/2026-04-21T10-24-12
    from_turn: 1
    created: 2026-04-21T10-24-17Z
  # ...

symbol_index:                         # 反向索引，retrieval 用
  消毒水: [imp_001]
  淡: [imp_001]
  持續: [imp_001]
  手: [imp_002, imp_005, imp_011]
  膝: [imp_002]
  # ...

cooccurrence:                         # symbol → 鄰居 → 共現次數
  消毒水:
    淡: 1
    持續: 1
  手:
    膝: 2
    搭: 1
    不動: 1
    顫抖: 2
  # ...
```

### 3.3 ID 產生規則

- Zero-padded 遞增：`imp_001`, `imp_002`, ...
- Scope：單一 ledger 內 unique，**跨 ledger 不保證 unique**
- Retrieval 跨兩本時用 `(speaker, id)` 複合 key dedup
- 越膨脹越長：`imp_999` → `imp_1000`

### 3.4 Append 語義

- 每次 session 結束 batch append 這場的所有 candidates
- 原子寫入（`.tmp` + `os.replace`）
- `ledger_version` 每次 append 後 +1（Level 4 的 Composer cache key 會用）

### 3.5 索引更新

**`symbol_index`**：
- Incremental。新 candidate 的每個 symbol 找到對應 list、append 這條 id
- 新 symbol 則新建 key
- 不 rebuild、不 verify consistency（信任 append-only 不會出錯）

**`cooccurrence`**：
- 對每個新 candidate，遍歷它 symbols 的所有 pair `(symbol_a, symbol_b)`
- `cooccurrence[symbol_a][symbol_b] += 1`
- 對稱：也會 `cooccurrence[symbol_b][symbol_a] += 1`
- 一個 candidate 的 3 個 symbols = 3 對 pair 各 +1 (9 次寫入含對稱)
- Arity < 2 的 candidate（只 1 個 symbol）不改變 cooccurrence

### 3.6 空帳本

- 第一場 session 前這兩個檔**不存在**
- `ledger.read_ledger` 偵測檔不存在 → 回傳 `Ledger(candidates=[], symbol_index={}, cooccurrence={}, ledger_version=0, ...)`
- 不需要預先 init 空檔

### 3.7 不變量

- **檔案是共同真相**（延續 Phase 2 spec §5 的主 spec §5.6 原則）
- **歷史不可變**：已 append 的 candidate 不改、不刪。Level 4 Composer 產的是 **新的** refined 檔，不回頭修 candidates 檔
- **ledger_version 單調遞增**

---

## 4. 模組骨架

### 4.1 檔案布局

```
src/empty_space/
├── paths.py              [Phase 1，不動]
├── schemas.py            [改：§2 dataclasses 擴充]
├── loaders.py            [Phase 1，不動]
├── llm.py                [Phase 1，不動]
├── parser.py             [Phase 2，不動]
├── prompt_assembler.py   [改：§8.1 build_system_prompt 簽名]
├── writer.py             [改：新增 write_retrieval；擴充 turn yaml / meta schema]
├── runner.py             [改：§8.2 接線 retrieval + ledger append]
├── ledger.py             [新增]
└── retrieval.py          [新增]

config/
└── symbol_synonyms.yaml  [新增，初始 groups: []]

scripts/
└── run_experiment.py     [Phase 2，不動]
```

### 4.2 依賴圖

```
schemas.py
   ↑
ledger.py ─────── retrieval.py ─── runner.py
                   ↑
                   llm.py
```

無循環依賴。`retrieval` 依賴 `ledger.read_ledger`，兩者都 pure-ish（只對檔案系統副作用）。

### 4.3 職責邊界

| 檔 | 職責 | 副作用 |
|---|---|---|
| `ledger.py` | 持久化 — 檔案 I/O、YAML 序列化、incremental index 維護 | 檔案讀寫 |
| `retrieval.py` | 查詢邏輯 — symbol extraction（Flash 呼叫）、expansion、scoring、canonicalization | LLM 呼叫 |
| `prompt_assembler.py` | 純函式 — 組 system prompt 的新 block | 無 |
| `writer.py` | 新增 retrieval.yaml 寫入；擴充現有 turn yaml + meta 欄位 | 檔案寫入 |
| `runner.py` | 編排 — session-start retrieval + session-end append + state 管理 | 委託其他 |

### 4.4 Function 契約

**`ledger.py`**：

```python
def ledger_path(relationship: str, persona_name: str) -> Path: ...

def read_ledger(relationship: str, persona_name: str) -> Ledger:
    """Return empty Ledger if file missing (not raise)."""
    ...

def append_session_candidates(
    *, relationship: str,
    speaker_role: Literal["protagonist", "counterpart"],
    persona_name: str,
    candidates: list[tuple[int, CandidateImpression]],  # (turn_number, imp)
    source_run: str,
) -> None:
    """Append one session's worth. Atomic write (.tmp + os.replace).
    Updates symbol_index and cooccurrence incrementally."""
    ...
```

**`retrieval.py`**：

```python
def load_synonym_map(path: Path = None) -> dict[str, str]:
    """Load config/symbol_synonyms.yaml. Returns symbol→canonical dict.
    Missing file or empty groups → {}."""
    ...

def canonicalize(symbol: str, synonym_map: dict[str, str]) -> str:
    return synonym_map.get(symbol, symbol)

def extract_symbols(
    *, text: str, llm_client: GeminiClient,
) -> tuple[list[str], int, int, int]:
    """Flash call. Returns (symbols, tokens_in, tokens_out, latency_ms)."""
    ...

def expand_with_cooccurrence(
    *, seed_symbols: list[str],
    cooccurrence: dict[str, dict[str, int]],
    top_neighbors_per_seed: int = 2,
) -> list[str]:
    """For each seed, pull its top-K most-cooccurring neighbors.
    Preserve seed order; append neighbors; dedup globally."""
    ...

def merge_cooccurrence(a, b) -> dict[str, dict[str, int]]:
    """Sum two cooccurrence maps."""
    ...

def retrieve_top_n(
    *, query_symbols: list[str],
    ledger_a: Ledger,
    ledger_b: Ledger,
    synonym_map: dict[str, str],
    top_n: int = 3,
) -> list[RetrievedImpression]:
    """Score across both ledgers, dedup by (speaker, id),
    sort by score desc then created desc, return top N."""
    ...

def run_session_start_retrieval(
    *, speaker_role, persona_name, query_text,
    relationship, other_persona_name,
    synonym_map, llm_client, top_n: int = 3,
) -> RetrievalResult:
    """Top-level orchestrator called by runner once per role."""
    ...
```

---

## 5. Session 生命週期

Phase 2 的 turn loop 不動；Level 2 在兩端插入新邏輯。

### 5.1 主流程

```
1. Load personas / setting (Phase 2)
2. Create out_dir, init_run (Phase 2)

# Level 2 新增
3. Load synonym_map (once, for both roles)
4. Session-start retrieval × 2：
   for role in [protagonist, counterpart]:
     query = scene_premise + role's prelude (compose)
     result = run_session_start_retrieval(...)
     state.retrieval_<role> = result
5. Write retrieval.yaml (audit trail)

# Phase 2 不動
6. Init SessionState (加 retrieval_* 欄位)
7. for n in 1..max_turns:
     speaker, persona, other_name 判定
     director event 觸發
     system = build_system_prompt(..., prelude=..., retrieved=...)  ← NEW kwargs
     user = build_user_message(history)
     resp = llm_client.generate(...)
     main, impressions, err = parse_response(resp.content)
     turn = Turn(..., retrieved_impressions=role_retrieval.impressions)  ← NEW field
     state.turns.append(turn)
     writer.append_turn(out_dir, turn)  ← writer 需支援新欄位

# Level 2 新增：在 write_meta 之前
8. Session-end ledger append:
   bucket turns by speaker_role
   for each bucket:
     ledger.append_session_candidates(...)

# Phase 2 不動
9. write_meta (包含新的 retrieval_total_tokens_*, ledger_appends 欄位)
10. return SessionResult
```

### 5.2 Query 組成

```python
def _compose_query(scene_premise: str | None, prelude: str | None) -> str:
    parts = [scene_premise, prelude]
    return "\n\n".join(p.strip() for p in parts if p and p.strip())
```

如果兩者都空 → 空字串 → `run_session_start_retrieval` 跳過 Flash call，回傳 `RetrievalResult(query_symbols=[], expanded_symbols=[], impressions=[], flash_tokens_*=0)`。

### 5.3 Relationship 名稱

```python
relationship = f"{protagonist.name}_x_{counterpart.name}"
```

用 `persona.name`（stable across yaml refactors）。

### 5.4 Ledger append 分流

```python
protagonist_candidates = [
    (t.turn_number, imp)
    for t in state.turns if t.speaker == "protagonist"
    for imp in t.candidate_impressions
]
counterpart_candidates = [
    (t.turn_number, imp)
    for t in state.turns if t.speaker == "counterpart"
    for imp in t.candidate_impressions
]
if protagonist_candidates:
    ledger.append_session_candidates(...)
if counterpart_candidates:
    ledger.append_session_candidates(...)
```

空 bucket 跳過（不 create 零 candidate 的帳本）。

### 5.5 中斷語義

- **Retrieval 期間 exception**（Flash 失敗、ledger 讀取失敗）→ runner crash；turn_* 未寫、meta 未寫、帳本未 append；retrieval.yaml 可能 partial
- **Turn loop 中 exception**（Phase 2 既有）→ 已寫 turn yaml 保留；meta 不寫；**帳本不 append**（append 在 write_meta 之前、loop 之後 — crash 已發生）
- **中斷不 resume**。重跑 = 新 timestamp。Level 2 不做 recovery

---

## 6. Symbol extraction（Flash prompt 設計）

### 6.1 Prompt 結構

```
System:
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

User:
<query_text>
```

### 6.2 Parser

```python
def _parse_symbols_response(raw: str) -> list[str]:
    try:
        parsed = yaml.safe_load(raw)
        if not isinstance(parsed, list):
            return []
        return [str(s).strip() for s in parsed if str(s).strip()]
    except yaml.YAMLError:
        return []
```

### 6.3 成本 / 延遲

- Input ~200-500 tokens
- Output ~50-100 tokens
- ~$0.0001 per role → **~$0.0002 per session**（兩角色）
- Latency ~200-400ms

### 6.4 失敗時的行為

- **Exception**（Flash error）→ propagate，runner crash
- **非 YAML / 空 list** → symbols=[]，整場沒 retrieval
- **垃圾內容**（Flash 解釋而非 list）→ YAML parse 可能失敗 → symbols=[]
- **不 retry、不 validation**。成本太低、失敗不致命、debug 進 retrieval.yaml

---

## 7. Retrieval 演算法

### 7.1 流程

```
Input: query_text, relationship, speaker_role, persona_name, synonym_map, llm_client

1. Flash extract:
   query_symbols = extract_symbols(query_text, llm_client)
     # e.g. [分離, 女朋友, 拒絕, 醫院, 父親, 重病]
   # 空 query_text → skip Flash → query_symbols = []

2. Load 兩本 ledgers:
   ledger_self  = read_ledger(relationship, persona_name)
   ledger_other = read_ledger(relationship, other_persona_name)

3. Co-occurrence expansion (1-hop):
   cooc = merge_cooccurrence(ledger_self.cooccurrence, ledger_other.cooccurrence)
   expanded_symbols = expand_with_cooccurrence(
       seed_symbols=query_symbols,
       cooccurrence=cooc,
       top_neighbors_per_seed=2,
   )

4. Score candidates in 兩本 ledgers:
   For each (ledger, entry), compute:
     canon_q = {canonicalize(s, synonym_map) for s in expanded_symbols}
     canon_e = {canonicalize(s, synonym_map) for s in entry.symbols}
     matched_canonicals = canon_q & canon_e          # set of canonical forms
     score = len(matched_canonicals)
   Collect only scored > 0.
   When building RetrievedImpression:
     matched_symbols = sorted(matched_canonicals)     # store canonical forms
     symbols = list(entry.symbols)                     # entry 原始 symbols

5. Sort & top-N:
   sort by: score desc, then entry.created desc (tiebreaker)
   dedup by (ledger.speaker, entry.id)
   take first top_n

6. Return RetrievalResult
```

### 7.2 `expand_with_cooccurrence`

```python
def expand_with_cooccurrence(
    *, seed_symbols, cooccurrence, top_neighbors_per_seed=2,
) -> list[str]:
    seen = set(seed_symbols)
    result = list(seed_symbols)
    for seed in seed_symbols:
        neighbors = cooccurrence.get(seed, {})
        top = sorted(neighbors.items(), key=lambda kv: (-kv[1], kv[0]))[:top_neighbors_per_seed]
        for sym, _ in top:
            if sym not in seen:
                result.append(sym)
                seen.add(sym)
    return result
```

### 7.3 `merge_cooccurrence`

```python
def merge_cooccurrence(a, b) -> dict[str, dict[str, int]]:
    result = {k: dict(v) for k, v in a.items()}
    for sym_a, neighbors in b.items():
        for sym_b, count in neighbors.items():
            result.setdefault(sym_a, {})
            result[sym_a][sym_b] = result[sym_a].get(sym_b, 0) + count
    return result
```

### 7.4 Scoring

- `score = |canonicalize(expanded_symbols) ∩ canonicalize(entry.symbols)|`
- Tiebreaker：entry.created desc（近期優先）
- 不做 recency decay（避免新印象壟斷）
- 不做 speaker 權重（A 策略 = 共同記憶）

### 7.5 Top-N

- `top_n = 3`
- 不足 3 條塞多少算多少
- 零命中 → `impressions = []`
- Dedup key：`(ledger.speaker, entry.id)`

### 7.6 Symbol 匹配的粗糙度

帳本靠 symbol 字串做 exact match。**不套用同義詞字典時**，Python 只會把完全相等字串算命中：

| query | entry | 命中？（無字典） |
|---|---|---|
| 愧疚 | 愧疚 | ✓ |
| 愧疚 | 愧疚感 | ✗ |
| 沉默 | 不說話 | ✗ |

套用字典後，canonicalize 兩端字串再比對：若 canonical 相等即命中（§7.7）。

Flash 是 LLM，同一概念可能用不同詞。Level 2 **故意留下的粗糙點**，用同義詞字典（§7.7）做最簡 normalization。

### 7.7 同義詞字典

**位置**：`config/symbol_synonyms.yaml`

**Schema**（canonical map）：

```yaml
groups:
  - [愧疚, 愧疚感, 罪惡感]     # canonical = 愧疚
  - [沉默, 不說話, 沉默不語]    # canonical = 沉默
  - [分手, 分離, 分開]
  - [手顫, 顫抖, 發抖]
```

Load 後產出 `dict[str, str]`（symbol → canonical）。不在字典裡的 symbol canonical = 自己。

**生效範圍**：**只在 matching step**（§7.1 step 4）

- Cooccurrence **不 canonicalize**（帳本資料不動）
- Flash extract **不 canonicalize**（LLM 自由輸出）
- Candidate impression 落盤 **不 canonicalize**（保留 Flash 原始用詞）

**空字典**：groups=[] 或檔不存在 → 空 map → canonicalize 回傳自己 → 行為等同原始 exact match。

---

## 8. Runner 接線

### 8.1 `build_system_prompt` 簽名擴充

```python
def build_system_prompt(
    persona, counterpart_name, setting,
    scene_premise, initial_state, active_events,
    prelude: str | None = None,                               # NEW
    retrieved_impressions: list[RetrievedImpression] | None = None,  # NEW
    ambient_echo=None,
) -> str: ...
```

「你的內在」block 組合：

```python
inner_parts = []
if prelude:
    inner_parts.append(prelude.rstrip())
if retrieved_impressions:
    lines = ["你可能想起的："] + [f"- {imp.text}" for imp in retrieved_impressions]
    inner_parts.append("\n".join(lines))
if inner_parts:
    blocks.append("## 你的內在\n" + "\n\n".join(inner_parts))
```

整個 block 有條件省略（§2.3）。內部兩段（prelude / retrieved）中間用空行分隔。retrieved 以 bullet list 呈現，prefix 固定為「你可能想起的：」。

### 8.2 Runner 修改

見 §5.1 流程。關鍵變更：

- Session 開始插入 retrieval（兩次 `run_session_start_retrieval`）
- SessionState 加 `retrieval_protagonist` / `retrieval_counterpart` 欄位
- 每輪組 prompt 時，傳入該 speaker 的 retrieval.impressions + prelude
- Session 結束在 write_meta 之前做兩次 `append_session_candidates`

### 8.3 Helper 函式

```python
def _compose_query(scene_premise, prelude) -> str:
    parts = [scene_premise, prelude]
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def _append_session_ledgers(*, relationship, protagonist_persona, counterpart_persona, turns, source_run):
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
    if p_candidates:
        ledger.append_session_candidates(
            relationship=relationship,
            speaker_role="protagonist",
            persona_name=protagonist_persona.name,
            candidates=p_candidates,
            source_run=source_run,
        )
    if c_candidates:
        ledger.append_session_candidates(
            relationship=relationship,
            speaker_role="counterpart",
            persona_name=counterpart_persona.name,
            candidates=c_candidates,
            source_run=source_run,
        )
```

---

## 9. 落盤 schema 變更

### 9.1 `turn_NNN.yaml`（+1 欄位）

Phase 2 既有欄位全保留；新增 `retrieved_impressions`：

```yaml
# ... Phase 2 欄位 ...
retrieved_impressions:                     # NEW
  - id: imp_042
    text: "她的沉默在這一刻比任何辯解都沉"
    symbols: [沉默, 辯解, 愧疚]
    speaker: counterpart
    persona_name: 兒子
    from_run: mother_x_son_hospital_v3_001/2026-04-21T10-24-12
    from_turn: 8
    score: 1
    matched_symbols: [愧疚]
```

- 每輪 turn yaml 對同一 session 同一角色是**完全相同**的（session-start 算一次）
- 命中 0 條 → `[]`
- 重複塞是為了 turn yaml **自包含**：只讀一個 turn yaml 就知道那輪演員看到的記憶，不用查 retrieval.yaml

### 9.2 `retrieval.yaml`（新檔）

```yaml
# runs/<exp_id>/<timestamp>/retrieval.yaml
protagonist:
  speaker_role: protagonist
  persona_name: 母親
  query_text: |
    他們在同一家醫院。父親在 ICU...

    你昨夜夢到他小時候被帶走。
  query_symbols: [醫院, 父親, 重病, 夢, 小時候, 帶走]
  expanded_symbols: [醫院, 父親, 重病, 夢, 小時候, 帶走,
                     走廊, 消毒水, 離別]
  impressions:
    - id: imp_042
      text: "她的沉默在這一刻比任何辯解都沉"
      symbols: [沉默, 辯解, 愧疚]
      speaker: counterpart
      persona_name: 兒子
      from_run: mother_x_son_hospital_v3_001/2026-04-21T10-24-12
      from_turn: 8
      score: 1
      matched_symbols: [愧疚]
  flash_latency_ms: 312
  flash_tokens_in: 158
  flash_tokens_out: 34

counterpart:
  # 同 structure ...
```

Session-start 跑完 retrieval 後寫入一次。是 retrieval 的完整 audit trail。

### 9.3 `meta.yaml` 擴充（+3 欄位）

```yaml
# Phase 2 欄位 ...

# Level 2 新增
retrieval_total_tokens_in: 158
retrieval_total_tokens_out: 34
ledger_appends:
  - relationship: 母親_x_兒子
    speaker: protagonist
    persona_name: 母親
    candidates_added: 14
    new_ledger_version: 4
  - relationship: 母親_x_兒子
    speaker: counterpart
    persona_name: 兒子
    candidates_added: 9
    new_ledger_version: 4
```

- `retrieval_total_*`：不進 `total_tokens_*`（那個只記角色 LLM）
- `ledger_appends`：空 bucket 跳過的 ledger 不列

### 9.4 `ledgers/` 檔 — 見 §3

### 9.5 `config/symbol_synonyms.yaml`

**不 gitignore**（knowledge asset，should commit）。初始值：

```yaml
groups: []
```

---

## 10. 測試策略

### 10.1 檔案

```
tests/
├── test_ledger.py                   [新]
├── test_retrieval.py                [新]
├── test_runner_level2.py            [新]
├── test_schemas_experiment.py       [改]
├── test_loaders_experiment.py       [改]
├── test_prompt_assembler.py         [改]
├── test_writer.py                   [改]
└── test_runner_integration.py       [改]
```

### 10.2 `test_ledger.py`（核心，~10 tests）

- `read_ledger` 對不存在的檔回傳 empty Ledger
- `append_session_candidates` 首次寫入：檔、candidates、symbol_index、cooccurrence、ledger_version=1 都正確
- 第二次 append：追加在後、id 遞增、index 累加、version=2
- 只 1 個 symbol 的 candidate：cooccurrence 無新 pair
- 3 個 symbol 的 candidate：cooccurrence 產 A↔B、A↔C、B↔C 各 +1
- 兩條 candidate 的 symbols 有重疊：cooccurrence 累加
- Atomic write：os.replace 前 raise → 不留 corrupt 檔
- Schema round-trip：寫入後讀取欄位一致

### 10.3 `test_retrieval.py`（核心，~15 tests）

**`canonicalize`**：空 map、命中、不命中 3 case
**`load_synonym_map`**：檔缺、空 groups、正常 3 case
**`expand_with_cooccurrence`**：空 cooc、seed 無鄰居、top-K 切、鄰居重疊 dedup 等
**`retrieve_top_n`**（無 LLM）：兩本空、單邊命中、跨兩本 dedup、同分 tiebreak、synonym 命中、超過 top_n 截斷
**`extract_symbols`**（mock LLM）：正常 YAML、壞 YAML、非 list、空 query 不呼叫、latency/tokens 回傳

### 10.4 `test_runner_level2.py`（~5 tests）

**Case 1：第一場空帳本 → 第二場有命中**
- 連跑兩次同一 config
- 驗證 ledger_version 遞增、第二場 retrieval 非空

**Case 2：prelude + premise 都空 → skip retrieval**
- retrieval.yaml 的 flash_tokens_in=0

**Case 3：Session 中斷不寫帳本**
- Mock turn 3 raise → turn_001/002 存在，meta 不存在，**帳本不存在**

**Case 4：Retrieval 命中塞 system prompt**
- 預先寫一本 ledger + prelude 含對應 symbol
- 驗證 Turn 1 和 Turn 5 的 system prompt 都含同一個 retrieved text

**Case 5：Synonym map 生效**
- 字典含 [愧疚, 愧疚感]，帳本是 [愧疚感]，query 是 [愧疚] → 命中

### 10.5 修改既有測試

- `test_schemas_experiment.py`：+3 prelude-related
- `test_prompt_assembler.py`：+6（block omit / prelude only / retrieved only / both / position / bullet list）
- `test_writer.py`：+2（write_retrieval）
- `test_runner_integration.py`：+2（retrieval.yaml 存在、turn yaml 含 retrieved 欄位）

### 10.6 Smoke test（手動）

`scripts/smoke_run.py`：連跑兩場 max_turns=3。第二場 retrieval.yaml 應該非空。

### 10.7 不做的測試

- 不測 Flash extract 的**品質**
- 不測 retrieval 結果的**戲劇效果**
- 不測 cooccurrence **權重合理性**
- 不做 property-based testing

### 10.8 預期測試數

| 檔 | 增加 |
|---|---|
| test_ledger.py | ~10 |
| test_retrieval.py | ~15 |
| test_runner_level2.py | ~5 |
| test_schemas_experiment.py | +3 |
| test_prompt_assembler.py | +6 |
| test_writer.py | +2 |
| test_runner_integration.py | +2 |

**Level 2 後預期 75 + ~43 ≈ 118 tests。**

---

## 11. 風險 / 未知數

### 11.1 Attention dilution（對話後期記憶被稀釋）
Turn 15+ system prompt 的「你的內在」被長長 user message 壓下去。Level 2 緩解（block 位置最靠對話）只救到 mid-session。Level 3 agentic pull 真正解。

### 11.2 Symbol 字串不一致（Flash 用詞漂移）
Level 2 用 `config/symbol_synonyms.yaml` 手工字典緩解。Level 3+ 可半自動（批次工具找候選同義詞）或上 embedding。

### 11.3 Cooccurrence 不 canonicalize 的 expansion 漏洞
Seed 查鄰居時 exact match，可能 miss 同義鄰居。衝擊小。未來可加 on-the-fly canonicalize。

### 11.4 第一場 session 零命中
預期行為不是 bug。跑第二場看差異，如果無差則帳本機制可能無效。

### 11.5 兩本帳本 append 非原子
極罕見。兩本獨立 load + merge，stale/fresh 不影響正確性。不做 transaction。

### 11.6 單 symbol candidate 不記 cooccurrence
Arity<2 的 candidate 通常不重要。接受。

### 11.7 帳本和 synonym map 的時序不 lock
因為「matching 才 canonicalize」，字典後加也能對歷史資料生效。

### 11.8 Candidate 品質不篩選
Flash 自標記全接受，可能噪音污染。Level 4 Composer 做 holistic 整合解決。期間手動抽樣觀察。

### 11.9 帳本膨脹
100 場 ≈ 1500 條候選。Python 讀寫 ms 級，人類看 yaml 開始痛。Level 4 Composer 濃縮 refined 帳本自然解決。

### 11.10 Retrieval semantic 盲區
Symbol-based lexical match 不抓意象串聯。未來 embedding retrieval（Phase 5+）。

---

## 12. 非目標（複述 §1 的 Out）

- rubric 個別評分
- Judge（Level 3）
- Composer（Level 4）
- Per-turn retrieval / agentic pull
- 多跳 graph traversal
- 跨 relationship 帳本共享
- Embedding retrieval
- Dashboard
- 跨實驗統計

---

## 13. 接下來

1. 柏為 review 本 spec
2. Spec 定稿 → commit 到 `docs/superpowers/specs/`
3. 進 writing-plans 技能 → 拆 implementation plan
4. 開工

---

## 附錄 A：Level 分層總覽

| Level | 核心躍升 | 組件 | 狀態 |
|---|---|---|---|
| **1** | 最小戲劇單位能跑 | Prompt assembler + Runner + structured output parser | ✓ ship 2026-04-21 (tag level-1) |
| **2** | 兩個演員跨場累積記憶 | Ledger + session-start retrieval + prelude + synonym dict | 本 spec |
| **3** | 對話結束在張力峰值 + 演員可主動召喚記憶 | Judge (stage/mode/張力) + agentic pull retrieval | 未 design |
| **4** | 「一天後」濃縮精華、生成情緒弧線 | Composer（Pro bake）+ refined 帳本 + 21 格 snapshot + graph consolidation | 未 design |

## 附錄 B：對主 spec 的 delta 摘要

| 主 spec 立場 | 本 spec 立場 | 理由 |
|---|---|---|
| Phase 3 Judge 先做 | Level 3 再做 | Level 2 核心 = 跨 session 記憶，Judge 正交 |
| Phase 4 Rubric 每印象評分 | 砍掉 | Flash 評 Flash 同能力圈打轉，效果虛。Level 4 Composer 做 holistic 整合 |
| Phase 4 per-turn ambient echo retrieval | 改為 session-start | 簡單 deterministic；per-turn 後期再考慮 |
| Phase 5 Composer bake 每 session | 延後到 Level 4 「一天後」 | Composer = 敘事單位的 consolidation，不是 per-session |
| 單一帳本 `A_x_B.impressions.yaml` | 兩本 `from_A.yaml` + `from_B.yaml` | 不對稱視角、語意明確 |
| 無同義詞機制 | `config/symbol_synonyms.yaml` 手工字典 | Level 2 最簡 normalization |

## 附錄 C：參照

- 主 spec：`docs/superpowers/specs/2026-04-21-空的空間-design.md`
- Phase 2 spec：`docs/superpowers/specs/2026-04-21-phase2-prompt-assembler-runner.md`
- Phase 2 summary：`docs/phase2-summary.md`
- Level 1 milestone tag：`level-1` → `437b23d`
