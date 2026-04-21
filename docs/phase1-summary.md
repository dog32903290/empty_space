# Phase 1 Summary — 空的空間 (The Empty Space)

**狀態**：✅ 完成 on 2026-04-21
**Commits**：18（`55d684b` → `e5a8752`）
**Tests**：26 passing
**Branch**：`main`

---

## 這一階段做到了什麼

建好一個**可以載入任何 persona / 實驗配置、能呼叫 Gemini Flash + Pro 的 Python package**。沒有 runner，沒有 Judge，沒有 Composer，沒有 Dashboard——這些都是後續 phase 的事。

### 可以做的事（不用 LLM）

```python
from empty_space.loaders import load_experiment, load_persona, load_setting

config = load_experiment("mother_x_son_hospital_v3_001")
protagonist = load_persona(config.protagonist.path, version=config.protagonist.version)
counterpart = load_persona(config.counterpart.path, version=config.counterpart.version)
setting = load_setting(config.setting.path)

# Now you have:
#   protagonist.core_text     → 母親 貫通軸 YAML content
#   protagonist.relationship_texts["兒子"]  → 關係層 YAML content
#   counterpart.core_text     → 兒子 貫通軸
#   counterpart.relationship_texts["母親"]  → 關係層
#   setting.content           → 環境_醫院 YAML
#   config.scripted_turns[8]  → Turn 8 的強制注入「你從來沒有找過我」
```

### 可以做的事（用 LLM）

```python
from empty_space.llm import GeminiClient

client = GeminiClient()  # 自動讀 GEMINI_API_KEY from .env

# Flash — 給角色 / Judge / retrieval / rubric 用
flash_resp = client.generate(
    system="系統提示",
    user="用戶訊息",
)  # model defaults to gemini-2.5-flash

# Pro — 給 Composer 用
pro_resp = client.generate(
    system="composer 指令",
    user="貫通軸 + 關係層 + ledger top-15",
    model="gemini-2.5-pro",
)

# response 物件：content, raw, tokens_in, tokens_out, model, latency_ms
```

---

## 重要決策記錄

### 2026-04-21：放棄 hackathon

Cerebral Valley "Built with Opus 4.7" hackathon 繳件期（4/27）撞到《長片架構論》交片期（5/15）。柏為選擇保長片、放棄比賽。這個決定也解放了 spec——不再被「Opus 4.7 為 pitch 主角」綁架。

### 2026-04-21：engine 全走 Gemini

柏為是 Claude Max 訂閱（Claude Code + claude.ai），Anthropic **官方明文不允許** Claude Agent SDK 走 Max auth——必須有 Console API key，獨立計費。柏為選擇不付，整個 engine 改走 Gemini 單一 provider。

**分工**：
- Gemini 2.5 Flash：角色演出、Judge、ambient echo retrieval、Composer pre-selection、rubric 評分
- Gemini 2.5 Pro：只有 Composer bake（21 格抽象組裝）

**成本估算**：每 session ~$0.09，100 實驗 ~$9（比原 Opus 設計省 86%）。

**風險**：spec §8.1 警告 Flash 做 rubric 的通過率虛高 —— 跑起來若發現帳本被灌水，應急路徑已在 spec §8.1 列出（調閾值 → 改三維度 rubric → 升 Pro）。

### 2026-04-21：扁平 module layout

Spec §4.3 原本寫 `empty_space/engine/` 子資料夾，實作扁平。Phase 1 module 數少（4 個），subpackage 增加 import noise 沒價值。未來超過 ~10 個檔再重構。

---

## 檔案結構

```
empty-space/
├── src/empty_space/
│   ├── paths.py         # PROJECT_ROOT, PERSONA_ROOT (→ sibling repo), EXPERIMENTS_DIR, RUNS_DIR, LEDGERS_DIR
│   ├── schemas.py       # Persona, Setting, ExperimentConfig + 5 nested types
│   ├── loaders.py       # load_persona, load_setting, load_experiment (+ _resolve_under guard)
│   └── llm.py           # GeminiClient (Flash + Pro), GeminiResponse dataclass
├── tests/               # 26 tests total
│   ├── test_paths.py
│   ├── test_schemas_persona.py
│   ├── test_schemas_setting.py
│   ├── test_schemas_experiment.py
│   ├── test_loaders_persona.py      (含 path traversal guard 測試)
│   ├── test_loaders_setting.py
│   ├── test_loaders_experiment.py
│   ├── test_llm_gemini.py
│   └── test_integration_phase1.py   (全 load chain)
├── scripts/
│   └── hello.py         # 真實 Gemini API smoke test (Flash + Pro)
├── experiments/
│   └── mother_x_son_hospital_v3_001.yaml
├── docs/
│   ├── phase1-summary.md               (this file)
│   └── superpowers/
│       ├── specs/2026-04-21-空的空間-design.md
│       └── plans/2026-04-21-phase1-infrastructure.md
├── runs/                # (gitignored, engine output lands here — Phase 2+)
├── ledgers/             # (gitignored, 帳本 — Phase 4)
├── pyproject.toml
├── uv.lock
├── .env.example         # only GEMINI_API_KEY now
└── .env                 # (gitignored, real key from 柏為's .env)
```

**Persona 材料**：引用自兄弟 repo `/Users/chenbaiwei/Desktop/vibe coding/演員方法論xhermes/persona/`，via `PERSONA_ROOT` 常數。不複製、不 symlink——柏為繼續在 `演員方法論xhermes` 維護 persona 庫，empty-space 讀 live 資料。

---

## 如何跑

```bash
cd "/Users/chenbaiwei/Desktop/vibe coding/empty-space"

uv sync --all-extras           # 建 .venv、裝依賴
cp .env.example .env           # 然後編輯填入 GEMINI_API_KEY

uv run pytest                  # 26 tests all pass
uv run python scripts/hello.py # 真實呼叫 Gemini Flash + Pro
```

---

## 還沒做（後續 Phase 路線）

| Phase | 內容 | 範疇 |
|-------|------|------|
| **2** | Prompt assembler + Runner + scripted injection + 基本 turn loop | Layer 1 runtime + 落盤 conversation.md |
| **3** | Layer 2 Judge（stage / mode / 張力 / fire_release / basin_lock） | Gemini Flash state machine |
| **4** | Layer 4 帳本（候選印象產出 + Flash rubric + symbol index + retrieval worker） | 跨 session 印象累積 |
| **5** | Layer 3 Composer（Gemini Pro bake + hash cache + pre-selection） | 21 格 snapshot |
| **6** | Dashboard（單 HTML + vanilla JS，讀 `runs/` 檔案呈現） | 共同可觀察介面 |
| **7** | Bootstrap script（關係層 YAML → Claude Code skill 格式搬家） | 為成品角色網站準備資產 |

每個 phase 自己的 design spec + writing-plans 新文件。

---

## 下次 session 進 Phase 2 的起點

開新 session 後：

1. 讀這個 summary + spec §3（對話流程）+ spec §5（檔案落盤 schema）
2. 開 brainstorm：Phase 2 要做 **prompt assembler + turn loop runner**
3. 關鍵設計題：
   - system prompt 的組裝順序（貫通軸 → 關係層 → Setting → Active Verb）
   - 角色結構化輸出：主回應 + `---IMPRESSIONS---` block 的 parser
   - turn loop 的 scripted injection 機制
   - 落盤格式：`runs/<exp_id>/<timestamp>/turns/turn_N.yaml`

Spec §3.2 已經有完整 session 生命週期的流程圖，Phase 2 只做到 "[Judge update]" **之前**（Judge 是 Phase 3）。

---

## 參考

- 主設計：`docs/superpowers/specs/2026-04-21-空的空間-design.md`
- Phase 1 plan：`docs/superpowers/plans/2026-04-21-phase1-infrastructure.md`
- Peter Brook《The Empty Space》1968 — 「兩個演員 + 一個看著的人」命題的出處
- 演員方法論本體：`../演員方法論xhermes/docs/策略總覽.md`
