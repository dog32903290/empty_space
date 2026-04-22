# Level 3：Composer（refined impression consolidation）— Design Spec

**作者**：陳柏為 × Claude（蒼鷺）
**日期**：2026-04-22
**狀態**：design draft，待 review
**依附**：
- 主 spec：`docs/superpowers/specs/2026-04-21-空的空間-design.md`
- Phase 2 spec：`docs/superpowers/specs/2026-04-21-phase2-prompt-assembler-runner.md`
- Level 2 spec：`docs/superpowers/specs/2026-04-21-level-2-ledger-retrieval.md`
- Level 2 summary：`docs/level-2-summary.md`

---

## 0. 前情：Level 3 為什麼做這個

Level 2 ship 後跑了三幕實驗（醫院 → 車上 → 家），暴露兩個問題：

1. **Register mismatch**：Flash 拆 prelude 抽到的是描述詞（父親、車、離開、走廊），但 Level 2 raw 候選印象的 symbols 是感受詞（苦、密度、感知、兒子）。兩組 register 幾乎不相交 → Act 2 retrieval 零命中
2. **Density compound**：Flash 產的 raw candidate 是**長段落判斷句**（「她的沉默在這一刻比任何辯解都沉」這種 literary judgment），不是 atomic 意象。Retrieved 塞進下場 system prompt，演員模仿 register → 語言越來越濃

Composer 的核心工作：把 raw candidates **萃取 + 轉化**為 refined impressions——短、atomic、第一人稱語氣、symbols 對齊 register。這些 refined 取代 raw 成為 retrieval 的 source。

這是 spec §4.1 原設計 Layer 3 Composer 的 **縮減版** — 只做帳本精煉，暫不做 21 格情緒弧線 / effective 關係層 / Judge principles（那些留給 Level 4+）。

### 柏為的核心設計 frame（brainstorm 定案）

- **Trigger**：每個 session 結束自動跑 Composer（腦內類比：每個事件後有新的記憶固化，不是作者主動整理）
- **Scope**：Minimal — 只做 refined impression consolidation
- **Layers**：Raw（Level 2）+ Refined（Level 3）兩層共存
- **Retrieval**：只撈 refined（raw 變原料庫）
- **Per-speaker**：Composer 一次呼叫產雙 section YAML（母親 / 兒子），後處理 script 分流到兩檔
- **Prompt 設計**：atomic image-first + transformation examples + 保持第一人稱視角（全功能）

---

## 1. 範圍

### In（Level 3 做）

- 新模組 `src/empty_space/composer.py`：prompt 組裝 + Pro bake 呼叫 + output parse + 分流寫入
- Refined 帳本 schema + YAML format：`ledgers/<relationship>.refined.from_<persona>.yaml`
- `ledger.py` 擴充：`read_refined_ledger`、`append_refined_impressions`、`refined_ledger_path`
- Runner 接線：session-end 自動跑 Composer（`append_session_candidates` 之後）
- Retrieval 切換：`run_session_start_retrieval` 改讀 refined（不再讀 raw）
- 新 dataclasses：`RefinedImpression`、`RefinedLedger`、`RefinedImpressionDraft`、`ComposerSessionResult`、`ComposerInput`
- `meta.yaml` 擴充：加 `composer_tokens_in/out`、`composer_latency_ms`、`protagonist_refined_added`、`counterpart_refined_added`、`composer_parse_error`
- 測試：`test_composer.py`（~12 tests）+ `test_refined_ledger.py`（~8 tests）+ 修改 `test_retrieval.py`、`test_writer.py`、`test_runner_integration.py`

### Out（Level 3 不做）

- **21 格情緒弧線**（Level 4 Judge 才需要）
- **Effective 關係層**（v3_tension 跨 session 穩定，不需動態產）
- **Cluster merge / 重寫 existing refined**（Minimal scope）
- **Rebuild mode**（每 session 只做 incremental，不全量重建）
- **Hash cache**（每 session 無條件跑 Composer）
- **手動 trigger CLI**
- **UI 顯示 refined 帳本 diff**（Dashboard 只 passively 讀 yaml，不動）
- **自動 retry 失敗的 Composer call**

### 副產出（既有檔案要改）

- `src/empty_space/schemas.py`：加 4 個新 dataclasses
- `src/empty_space/ledger.py`：加 3 個 refined 版本的 I/O 函式
- `src/empty_space/retrieval.py`：`run_session_start_retrieval` 改讀 refined；`retrieve_top_n` 簽名從 `ledger_a/b` 改為 `entries_a/b`（decouple from ledger type）
- `src/empty_space/runner.py`：session-end 插入 `_run_composer_at_session_end`
- `src/empty_space/writer.py`：`write_meta` 加 5 個 kwargs
- 現有 integration tests：MockLLMClient responses 尾端加 1 個 Composer YAML response

### Ship 完的樣子

跑 `uv run python scripts/run_experiment.py <exp_id>`：
- 對話進行中：一切如 Level 2
- 對話結束後：Composer Pro bake 自動跑一次（~15-30s latency），產出 refined 兩檔
- `ledgers/` 下此時同時有 raw 兩檔 + refined 兩檔
- 下場 session 開始：retrieval 從 refined 撈（不再是 raw）
- 演員看到的「你可能想起的」變成短 atomic 意象，不再是長文學判斷句
- 跑第二場後 `meta.yaml` 有 `composer_*` 欄位 + refined_added 計數

---

## 2. Schema 變更

### 2.1 新 dataclasses（`schemas.py`）

```python
@dataclass
class RefinedImpression:
    """Composer-refined impression. One record of consolidated memory."""
    id: str                              # ref_001, ref_002, ...
    text: str                            # 短 atomic，第一人稱語氣
    symbols: list[str]
    speaker: Literal["protagonist", "counterpart"]
    persona_name: str                    # 母親 / 兒子
    from_run: str                        # e.g. mother_x_son_act1_hospital/2026-04-22T10-00-00
    source_raw_ids: list[str]            # 這條來自哪些 raw（provenance）
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
    """Pre-id draft from Composer YAML parse (no id assigned yet)."""
    text: str
    symbols: list[str]
    source_raw_ids: list[str]


@dataclass
class ComposerSessionResult:
    """What run_composer returns to runner. Goes to meta.yaml."""
    tokens_in: int
    tokens_out: int
    latency_ms: int
    protagonist_refined_added: int
    counterpart_refined_added: int
    parse_error: str | None


@dataclass
class ComposerInput:
    """Materials gathered for Composer to consolidate."""
    relationship: str
    protagonist_name: str
    counterpart_name: str
    conversation_text: str
    new_candidates: dict[str, list["CandidateImpression"]]  # key: protagonist/counterpart
    new_candidate_ids: dict[str, list[str]]                 # raw id 對應，與 new_candidates 同序
    existing_refined: dict[str, list["RefinedImpression"]]  # 最近 30 條 per speaker
```

關鍵差異 vs `LedgerEntry`/`Ledger`：
- `id` prefix `ref_`（不和 raw `imp_` 衝突）
- 加 `source_raw_ids`（provenance）
- 去掉 `from_turn`（refined 是多 turn 整合）

### 2.2 現有 schema 調整

**`RetrievedImpression` 變動**：

```python
# Level 2:
from_turn: int

# Level 3:
from_turn: int | None = None    # refined 來的時候為 None
```

### 2.3 `Turn.retrieved_impressions` 欄位

不改。仍是 `list[RetrievedImpression]`。`RetrievedImpression.from_turn` 可 None，既有 writer serialization 吐 `null` 即可。

### 2.4 `ExperimentConfig` 不動

Composer 是 runner 層的自動動作，不需 experiment config 控制。

---

## 3. 模組骨架

### 3.1 檔案布局

```
src/empty_space/
├── paths.py              [Phase 1，不動]
├── schemas.py            [改：§2.1 加 4 個 dataclass + 改 RetrievedImpression]
├── loaders.py            [Phase 1，不動]
├── llm.py                [Phase 1，不動]
├── parser.py             [Phase 2，不動]
├── prompt_assembler.py   [Level 2，不動]
├── writer.py             [改：write_meta 加 5 kwargs]
├── ledger.py             [改：加 read_refined_ledger / append_refined_impressions / refined_ledger_path]
├── retrieval.py          [改：run_session_start_retrieval 改讀 refined；retrieve_top_n 改簽名]
├── runner.py             [改：session-end 插入 Composer call]
└── composer.py           [新增]
```

### 3.2 依賴關係

```
schemas.py
   ↑
ledger.py ←── composer.py ←── llm.py
                   ↑
                   runner.py
```

無循環。`composer` 依賴 `schemas`、`ledger`、`llm`。`runner` 依賴 `composer`（session-end 呼叫）。

### 3.3 Composer.py 函式契約

```python
# composer.py

COMPOSER_MODEL = "gemini-2.5-pro"


_COMPOSER_SYSTEM_PROMPT = """..."""   # §4.1 的完整 prompt


def gather_composer_input(
    *,
    relationship: str,
    protagonist_name: str,
    counterpart_name: str,
    out_dir: Path,                   # runs/<exp>/<timestamp>/
    session_turns: list[Turn],
    new_raw_ids: dict[str, list[str]],  # from _append_session_ledgers return
) -> ComposerInput:
    """Read conversation.md, bucket raws by speaker, load existing refined."""


def build_composer_prompt(input: ComposerInput) -> tuple[str, str]:
    """Return (system_prompt, user_message) for Pro bake."""


def run_composer_bake(
    system: str, user: str, llm_client: GeminiClient,
) -> tuple[str, int, int, int]:
    """Pro call. Returns (raw_response_text, tokens_in, tokens_out, latency_ms)."""


def parse_composer_output(
    raw_yaml: str,
) -> tuple[list[RefinedImpressionDraft], list[RefinedImpressionDraft], str | None]:
    """Parse Pro's YAML output to (protagonist_drafts, counterpart_drafts, parse_error).
    Graceful degradation: returns ([], [], err_msg) on failure."""


def run_composer(
    *,
    relationship: str,
    protagonist_name: str,
    counterpart_name: str,
    out_dir: Path,
    session_turns: list[Turn],
    new_raw_ids: dict[str, list[str]],
    source_run: str,
    llm_client: GeminiClient,
) -> ComposerSessionResult:
    """Top-level orchestrator. Called by runner at session end.

    Any exception caught → returns result with parse_error, refined ledgers untouched.
    """
```

### 3.4 ledger.py 擴充

```python
# Added to ledger.py

def refined_ledger_path(*, relationship: str, persona_name: str) -> Path:
    return LEDGERS_DIR / f"{relationship}.refined.from_{persona_name}.yaml"


def read_refined_ledger(*, relationship: str, persona_name: str) -> RefinedLedger:
    """Read refined ledger; missing file → empty RefinedLedger."""


def append_refined_impressions(
    *,
    relationship: str,
    speaker_role: str,
    persona_name: str,
    drafts: list[RefinedImpressionDraft],
    source_run: str,
) -> int:
    """Append refined drafts (auto-assign ref_NNN). Returns new ledger_version.
    Empty drafts: no-op, returns current version (no file touch)."""
```

### 3.5 `append_session_candidates` 簽名微調

Level 2 回傳 `None`。Level 3 需要 id 清單給 Composer 的 provenance。改為：

```python
def append_session_candidates(...) -> list[str]:
    """Return list of new-appended LedgerEntry ids (for Composer provenance)."""
```

Backward compat：既有 Level 2 caller 忽略 return，無破壞。

---

## 4. Composer Prompt 設計

### 4.1 System prompt（完整）

```
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
- 嚴禁第三人稱（例：不要「她不動像鐵」；若該意象是兒子觀察，改為「我看她不動的時候像鐵」然後歸屬兒子，或選另一個兒子自己的 atomic image）

**2. Atomic 原則**
- 每條控制在 **15 字內**
- 原始感官 / 動作 / 體感 — **不是** judgment 或 analysis
- 壞例：「她的沉默比辯解都沉」（judgment，太文學）
- 好例：「沉默時喉嚨收緊」（體感）
- 壞例：「他感覺到自己的牆沒有用」（analysis）
- 好例：「手指捏著衣角」（動作）

**3. 歸屬判斷**
- 每條 refined 歸屬給其中一個角色
- 原則：這條 refined 是**誰的內在感受**
- 多半 refined 就跟隨 raw 的 speaker；偶有例外（某些 raw 是觀察對方的，重寫成第一人稱時可能更適合歸給被觀察的角色）

**4. 不保留 judgment / analysis raw**
- Raw 中充滿判斷性、反思性的句子（「他的存在本身是一種重量」、「這裡把人的社會面具脫掉」）
- **不要精煉這些**。只提煉真正會沉到身體裡的片段

**5. Merge 或保留**
- 同 session 多個 raw 講同個瞬間 → 各自 refine 成不同 atomic image
- 不強求 merge 跨 speaker

**6. Symbols**
- 每條 refined 帶 2-4 個 symbols（感官詞 / 動詞 / 身體部位 / 情境詞）
- 和 raw 的 symbol 體系**盡量對齊**（若 raw 用「沉默」你也用「沉默」）
- 這讓未來 retrieval 的 symbol match 有連續性

**7. 精簡數量**
- 每 session 的 raw 約 5-15 條 per speaker
- Refined 產出每角色 **3-6 條**
- 如果這 session 真的沒什麼沉澱，少產幾條也 OK

---

## 輸入結構

你會收到：
- 這 session 的完整對話（conversation.md 格式）
- 母親這 session 產出的 raw candidates（帶 imp_XXX id）
- 兒子這 session 產出的 raw candidates（帶 imp_XXX id）
- （若有）母親現有的 refined impressions（供參考語氣）
- （若有）兒子現有的 refined impressions（供參考語氣）

參考既有 refined 的語氣和 symbol 用詞，保持連續性。

---

## 輸出格式

只輸出 YAML，不加任何解釋：

```yaml
母親:
  - text: "沉默時喉嚨收緊"
    symbols: [沉默, 喉嚨, 收緊]
    source_raw_ids: [imp_003, imp_007]
  - text: "手指在膝上按壓著"
    symbols: [手指, 膝, 按壓]
    source_raw_ids: [imp_004]

兒子:
  - text: "背靠著牆坐"
    symbols: [背, 牆, 坐]
    source_raw_ids: [imp_002]
  - text: "目光釘在門上"
    symbols: [目光, 門, 釘]
    source_raw_ids: [imp_005, imp_008]
```

`source_raw_ids` 必填——指出這條 refined 來自哪些 raw。若是從對話文本推出不直接對應某條 raw，用 `[]`。
```

### 4.2 User message 結構

```
## Session 對話
<conversation.md 原始 markdown>

## 母親的 Raw Candidates（本 session 新產出）
- imp_003: 她的沉默在這一刻比任何辯解都沉 [symbols: 沉默, 辯解, 愧疚]
- imp_004: 她的手在膝上輕輕搭著，那雙手沒有動 [symbols: 手, 膝, 不動]
- imp_007: 她感覺到消毒水的味道讓呼吸變淺 [symbols: 消毒水, 呼吸, 淺]
  ...

## 兒子的 Raw Candidates（本 session 新產出）
- imp_002: 他的背筆直靠在長椅冰冷的表面 [symbols: 背, 冰冷, 筆直]
- imp_005: 他的目光釘在門上 [symbols: 目光, 門, 釘]
  ...

## 母親的既有 Refined Impressions（供參考語氣）
- ref_001: 你在走廊坐著的時候空氣會變薄 [symbols: 走廊, 空氣, 薄]
  ...

## 兒子的既有 Refined Impressions（供參考語氣）
- （空，這對關係 Composer 首次跑）

---

開始精煉。
```

### 4.3 Parser 行為

- Top-level YAML 必須有 `母親:` 和 `兒子:` 兩個 keys（用 persona.name）
- Speaker key 容錯：`媽媽` / `mother` / `母` prefix → fuzzy match 為 protagonist
- 缺一邊 → 該 drafts 列表為 []
- 每 draft 必須有 `text`；`symbols` 缺失 → []；`source_raw_ids` 缺失 → []
- YAML parse error → 返回 `([], [], "YAML parse error: ...")`
- Text > 15 字 → 接受（soft limit）、不截斷
- Draft 缺 `text` → 靜默跳過該 item

### 4.4 Prompt 迭代策略

Level 3 ship 後跑 3-5 場看 refined 品質。若：
- 仍太長 → 強化 15 字規則、加更多「壞例」
- 第三人稱漂移 → prompt 最上方加大警告
- Raw 沒被濃縮 → 展示更多 transformation 範例

Prompt 是 `composer.py` 的 module-level 常數。迭代 = 改字串 + commit，不動 engine 邏輯。

---

## 5. Composer Input Scope

### 5.1 必要輸入

1. **Session 的完整 conversation.md**（從磁碟讀）— ~1-3k tokens
2. **本 session 新產出的 raw candidates**（從 runner state 取）— ~1-2k tokens
3. **既有 refined impressions 各 speaker 最近 30 條**（從磁碟讀）— ~600-1500 tokens
   - 若檔不存在（首次跑）→ 空 list

### 5.2 不給的輸入

- 全部歷史 raw（incremental 原則）
- Persona YAML（貫通軸 / 關係層）— 避免 Composer 替角色說話
- Setting YAML — 已透過 conversation 反映
- Scene premise / prelude — 同上
- Judge state / turn metadata

### 5.3 Input 組裝邏輯

```python
def gather_composer_input(
    *,
    relationship: str,
    protagonist_name: str,
    counterpart_name: str,
    out_dir: Path,
    session_turns: list[Turn],
    new_raw_ids: dict[str, list[str]],
) -> ComposerInput:
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
```

### 5.4 Token / 成本估算

| 項 | Tokens |
|---|---|
| System prompt | ~700 |
| Conversation | ~1-3k |
| 兩側新 raw | ~1-2k |
| 兩側 existing refined（max 60 條） | ~600-1.5k |
| **Input total** | **~3-7k** |
| Output | ~500-1000 |

Pro pricing（spec §8.4）：$1.25/M in, $10/M out

每 session Composer 成本 ≈ **~$0.019**（input $0.009 + output $0.010）。100 session ≈ $2。

### 5.5 為什麼 incremental

1. 符合腦內類比（睡眠固化今天的經驗，不重做人生記憶）
2. Cost 不隨歷史線性膨脹
3. Existing refined 本身就是歷史的 summary，Pro 能參考

### 5.6 未來的 rebuild mode hook（不在 Level 3 做）

預留：`scripts/recompose.py <relationship>` 手動 rebuild 工具，看全部 raw 重新 consolidate。Level 5+ 考慮。

---

## 6. Runner 接線

### 6.1 主流程修改

Level 2 既有末段：

```python
# turn loop 結束

# Level 2 ledger append
ledger_appends = _append_session_ledgers(...)
write_meta(..., ledger_appends=ledger_appends)
return SessionResult(...)
```

Level 3 插入 Composer：

```python
# turn loop 結束

# Level 2 ledger append — 改為回傳 new_raw_ids
ledger_appends, new_raw_ids = _append_session_ledgers(...)
# new_raw_ids: dict[str, list[str]] = {"protagonist": ["imp_045", "imp_046"], "counterpart": ["imp_012"]}

# Level 3：Composer
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

write_meta(
    ..., ledger_appends=ledger_appends,
    # Level 3 new params:
    composer_tokens_in=composer_result.tokens_in,
    composer_tokens_out=composer_result.tokens_out,
    composer_latency_ms=composer_result.latency_ms,
    protagonist_refined_added=composer_result.protagonist_refined_added,
    counterpart_refined_added=composer_result.counterpart_refined_added,
    composer_parse_error=composer_result.parse_error,
)
return SessionResult(...)
```

### 6.2 Helper 定義

```python
def _run_composer_at_session_end(
    *, relationship, protagonist, counterpart, out_dir,
    turns, new_raw_ids, source_run, llm_client,
) -> ComposerSessionResult:
    """Wrap run_composer with exception handling."""
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

### 6.3 Ordering rationale

- Raw append **先跑**（assigns `imp_XXX` ids）→ Composer 讀 state.turns + new_raw_ids 對照 → 產 refined 帶 `source_raw_ids`
- Composer **在 write_meta 之前**跑，result 進 meta

### 6.4 SessionState 不擴充

Composer 是 session-end 一次性動作，input 從 runner local vars 組。

### 6.5 Session 時長影響

- Level 2 session end → ~2-3s（檔案寫入）
- Level 3 session end → ~15-35s（加 Pro bake）
- CLI 總時長從 Level 2 的 ~60-90s → Level 3 的 ~80-130s
- Dashboard 的對話 rendering 不延遲（conversation.md 隨 turn 寫入），只是 meta summary 顯示稍後出現

---

## 7. Retrieval 切換

### 7.1 `retrieve_top_n` 簽名重構

Level 2：
```python
def retrieve_top_n(
    *, query_symbols, ledger_a, ledger_b, synonym_map, top_n=3,
) -> list[RetrievedImpression]:
    for ledger in (ledger_a, ledger_b):
        for entry in ledger.candidates:
            ...
```

Level 3 — 接 entries list（decouple from ledger type）：

```python
def retrieve_top_n(
    *,
    query_symbols: list[str],
    entries_a: list,           # LedgerEntry 或 RefinedImpression
    entries_b: list,
    speaker_a: str,
    persona_name_a: str,
    speaker_b: str,
    persona_name_b: str,
    synonym_map: dict[str, str],
    top_n: int = 3,
) -> list[RetrievedImpression]:
    ...
```

函式 body 不變（scoring / sort / dedup），只是遍歷傳入的 entries list 而不是 ledger.candidates。

### 7.2 `run_session_start_retrieval` 改接 refined

```python
# Before (Level 2):
ledger_self = read_ledger(relationship=..., persona_name=...)
ledger_other = read_ledger(relationship=..., persona_name=...)
...
impressions = retrieve_top_n(
    query_symbols=expanded_symbols,
    ledger_a=ledger_self, ledger_b=ledger_other,
    synonym_map=synonym_map, top_n=top_n,
)

# After (Level 3):
refined_self = read_refined_ledger(relationship=..., persona_name=...)
refined_other = read_refined_ledger(relationship=..., persona_name=...)
...
# cooccurrence expansion uses refined's cooccurrence (same structure)
merged_cooc = merge_cooccurrence(refined_self.cooccurrence, refined_other.cooccurrence)
expanded_symbols = expand_with_cooccurrence(seed_symbols=query_symbols, cooccurrence=merged_cooc, top_neighbors_per_seed=2)

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

### 7.3 `RetrievedImpression` 的 from_turn

Refined 進 RetrievedImpression 時 `from_turn=None`。Level 2 既有的 from_turn 是 `int` — 改為 `int | None`。

Writer 序列化 `None` → YAML `null`，既有 logic 可用。

### 7.4 Empty refined ledger

`read_refined_ledger` 偵測檔案缺 → 返回 empty `RefinedLedger`，`impressions=[]`。Retrieve 返回 `impressions=[]`，system prompt 的 `## 你的內在` block 若無 prelude 就整段省略。

和 Level 2 的 first-session-empty 語義一致。

### 7.5 Cooccurrence 在 refined 世界

Refined 帳本自己維護 cooccurrence（Composer 每 append 一條 refined，就更新 symbol pair 共現）。Expansion 使用 refined 的 cooccurrence——可能比 raw 的 cooccurrence 更有語意（因為 refined symbols 已經過 consolidation，更 focused）。

---

## 8. 落盤 Schema 變更

### 8.1 新檔：`ledgers/<relationship>.refined.from_<persona>.yaml`

```yaml
relationship: 母親_x_兒子
speaker: protagonist
persona_name: 母親
created: 2026-04-22T10:00:00Z
last_updated: 2026-04-22T11:30:00Z
ledger_version: 2                         # 每 Composer append +1

impressions:
  - id: ref_001
    text: "沉默時喉嚨收緊"
    symbols: [沉默, 喉嚨, 收緊]
    from_run: mother_x_son_act1_hospital/2026-04-22T10-00-00
    source_raw_ids: [imp_003, imp_007]
    created: 2026-04-22T10:05:32Z
  - id: ref_002
    text: "手指在膝上按壓著"
    symbols: [手指, 膝, 按壓]
    from_run: mother_x_son_act1_hospital/2026-04-22T10-00-00
    source_raw_ids: [imp_004]
    created: 2026-04-22T10:05:32Z

symbol_index:
  沉默: [ref_001]
  喉嚨: [ref_001]
  手指: [ref_002]

cooccurrence:
  沉默:
    喉嚨: 1
    收緊: 1
  手指:
    膝: 1
    按壓: 1
```

差異 vs raw ledger：
- `impressions` 而非 `candidates`
- `id` prefix `ref_`
- 加 `source_raw_ids`、去 `from_turn`

### 8.2 `turn_NNN.yaml.retrieved_impressions` 變化

Level 2 這欄每項有 `id: imp_XXX`, `from_turn: int`。Level 3 變 `id: ref_XXX`, `from_turn: null`。其他欄不變。

Writer 已能處理 `null`，不改 writer code。

### 8.3 `retrieval.yaml`

結構不變。`impressions` 欄位裡 id 是 `ref_XXX`、`from_turn: null`。

### 8.4 `meta.yaml` 擴充

Level 2 欄位全保留，新增 5 個：

```yaml
# Level 3 new
composer_tokens_in: 4321
composer_tokens_out: 678
composer_latency_ms: 15234
protagonist_refined_added: 4
counterpart_refined_added: 3
composer_parse_error: null            # 或錯誤訊息
```

`write_meta` 加 5 個 kwargs，全部預設 0 / None / [] → backward compatible。

### 8.5 `config.yaml`（run 目錄複本）

不動。`ExperimentConfig` 沒新欄。

### 8.6 不變量

- Refined 帳本 append-only（Minimal scope：不修既有 entries）
- `source_raw_ids` 指向的 raw impression 永遠在 raw 帳本（不刪）
- `ledger_version` 單調遞增
- 原子寫入（復用 `_atomic_write_ledger`）

---

## 9. 錯誤處理

### 9.1 錯誤類型與處理

| 錯誤 | 發生時機 | 行為 |
|---|---|---|
| Pro API exception | `llm_client.generate(model="pro")` raise | catch，記 parse_error，session 完成，refined 帳本不動 |
| Pro 回傳爛 YAML | `parse_composer_output` | 返回 `([], [], err)`，refined 帳本不動 |
| Pro 回傳非預期結構 | 同上 | 同上 |
| Draft 缺欄位 | parser per-item loop | 靜默跳過該 item |
| Refined yaml 寫入失敗 | `_atomic_write_ledger` | propagate（極罕見） |
| Turn loop 中 LLM exception | Level 2 既有 | session crash、raw 不 append、Composer 不跑 |

### 9.2 核心語義

**Composer 是增強，不是 critical path**。Composer 失敗不影響 raw 帳本正確累積 / session 完成。

### 9.3 Partial 成功

Pro 返回 valid YAML，但只有一側 section 有 drafts → 該側 refined append、另一側不動。不記 parse_error。

### 9.4 不 retry

第一次失敗即記錄。Retry 代價高（session 延長）、真實失敗（auth / quota）retry 也救不了。

### 9.5 Session 中斷

Turn loop 的 LLM call raise → session 失敗，Composer **不跑**，raw 不 append。Level 2 既有語義。

---

## 10. 測試策略

### 10.1 新增 / 修改

```
tests/
├── test_composer.py              [新] ~12 tests
├── test_refined_ledger.py        [新] ~8 tests
├── test_retrieval.py             [改] retrieve_top_n 新簽名；+2 refined tests
├── test_writer.py                [改] +1 composer fields test
└── test_runner_integration.py    [改] +3 composer scenarios
```

### 10.2 `test_refined_ledger.py`（~8）

對稱 `test_ledger.py`。核心：
- `read_refined_ledger` 檔缺 → empty
- `refined_ledger_path` 命名正確
- `append_refined_impressions` 首次、二次、符號索引、cooccurrence、單 symbol arity<2
- Schema round-trip
- Atomic write

### 10.3 `test_composer.py`（~12）

**Parse tests（無 LLM）**：clean YAML、空 section、爛 YAML、non-dict root、item 缺 text、缺 symbols、缺 source_raw_ids、speaker key variants

**Prompt tests**：build_composer_prompt 含 conversation、raw 帶 id

**Orchestrator tests（mock llm）**：happy path、Pro exception caught、partial success only one side

### 10.4 `test_retrieval.py` 修改

既有 tests 改用新簽名（helper 封裝取 entries）。新增：
- `test_retrieve_top_n_with_refined_impressions`
- `test_run_session_start_retrieval_reads_refined`

### 10.5 `test_runner_integration.py` 修改

既有 10 tests 的 MockLLMClient responses 尾端加 1 個 Composer YAML。

新增：
- `test_composer_runs_at_session_end`
- `test_composer_failure_doesnt_break_session`
- `test_second_session_retrieval_reads_refined`

### 10.6 不做的測試

- Composer prompt **品質**（輸出是否真 atomic / 第一人稱）
- Refined 和 raw 的 **語意一致性**
- 極端 latency

### 10.7 預期總數

Level 2 (135) + Level 3 (~26) ≈ **161 tests**。

### 10.8 Smoke test（手動）

`scripts/smoke_run.py`（若之前做過）：跑 max_turns=4，驗：
- `refined.from_X.yaml` 兩檔被創
- 每檔 `impressions` 有 3-6 條
- `meta.yaml.composer_*` 有值
- 跑第二場，retrieval.yaml 的 id 是 `ref_XXX`

---

## 11. 風險 / 未知數

### 11.1 Pro prompt 穩定性
第三人稱漂移、沒濃縮、歸屬 confusion。監控：5 場後抽樣 3 條對 raw。應急：強化 prompt / 加範例 / Opus 升級。

### 11.2 Provenance 不精確
Pro 可能亂填 `source_raw_ids`。衝擊：debug 困難。未來可加 post-hoc text 匹配工具補。

### 11.3 Incremental 盲點
跨 session 的 emergent 連結會被 miss。緩解：existing refined 當 summary 傳給 Pro。未來：`scripts/recompose.py` 全量 rebuild。

### 11.4 Pro cost
100 session ≈ $2。Scale up 到 1000 session = $20。未來可加 cache / skip 機制。

### 11.5 Refined 帳本膨脹
200 session 後每本 ~1000 條。Retrieval 仍 ms 級。人讀累。未來：Level 4+ cluster merge。

### 11.6 Composer 連續失敗
多 session fail → 記憶沒沉。監控：`composer_parse_error` 統計。

### 11.7 Symbol register 對齊未必解
Flash 拆 prelude 仍用描述詞，Composer 產感受詞，兩組 register 可能仍不相交。**這是 Level 3 假設能解但沒驗證的風險**。跑幾場看。若 retrieval 命中率仍低，Level 4+ 可能要：改 Flash extract prompt / Composer 硬對齊 symbols / canonicalization。

### 11.8 POV leakage
Level 3 假設 by-speaker refined + 第一人稱 prompt → 解 POV。但 prompt 不穩定可能重現 leakage。監控對話 POV。

### 11.9 既有 Level 2 raw 不進 Level 3 workflow
先前累積的 raw 孤兒化。推薦清帳本從零開始，或接受現狀。

### 11.10 Composer 看 conversation.md formatting
Level 2 既有 markdown 結構 Pro 能 parse。若 Level 4 加 judge annotation 等，要重測。

---

## 12. 非目標（複述 §1 的 Out）

- 21 格情緒弧線
- Effective 關係層
- Cluster merge
- Rebuild mode
- Hash cache
- 手動 trigger CLI
- UI 顯示 refined 帳本 diff
- 自動 retry

---

## 13. 接下來

1. 柏為 review 本 spec
2. Spec 定稿 → commit
3. 進 writing-plans 拆 implementation plan
4. 開工

---

## 附錄 A：對主 spec 的 delta 摘要

| 主 spec 立場（§4.1） | Level 3 本 spec 立場 | 理由 |
|---|---|---|
| Composer bake = 21 格 + effective 關係層 + Judge principles + hash cache | 只做 refined impressions consolidation | Minimal scope；21 格等 Judge 需要時再加 |
| Composer 是 session 邊界的特定時機跑 | 每 session 結束自動跑（腦內類比） | 每事件一次記憶固化 |
| 單一 refined 帳本 `A_x_B.impressions.yaml` | 兩本 `refined.from_A.yaml` + `refined.from_B.yaml` | 延續 Level 2 不對稱視角原則 |
| Composer 輸入含 pre-selection 篩過的 raw | Composer 輸入是本 session 新 raw + 最近 30 條 existing refined | Incremental；pre-selection 沒必要 |
| hash cache 決定 re-bake | 每 session 無條件跑 | Minimal scope；cache 未來加 |

## 附錄 B：Level 分層總覽

| Level | 核心躍升 | 組件 | 狀態 |
|---|---|---|---|
| 1 | 最小戲劇單位能跑 | Prompt assembler + Runner | ✓ ship (tag `level-1`) |
| 2 | 兩個演員跨場累積記憶 | Raw ledger + session-start retrieval + prelude | ✓ ship (tag `level-2`) |
| **3** | **Refined 帳本 + 符號 register 對齊** | **Composer (Pro bake) + refined ledger** | **本 spec** |
| 4 | 對話結束在張力峰值 + 可能 agentic pull | Judge (stage/mode/張力) | 未 design |
| 5+ | 跨 session graph consolidation、rebuild、cluster merge、21 格弧線 | Composer 擴充 + Judge 擴充 | 未 design |

## 附錄 C：參照

- 主 spec: `docs/superpowers/specs/2026-04-21-空的空間-design.md`
- Level 2 spec: `docs/superpowers/specs/2026-04-21-level-2-ledger-retrieval.md`
- Level 2 summary: `docs/level-2-summary.md`
- 三幕實驗 runs: `runs/mother_x_son_act1_hospital/`, `act2_car/`, `act3_home/`（2026-04-22 16:xx）
