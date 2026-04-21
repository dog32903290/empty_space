# Phase 2 Summary — 空的空間 (The Empty Space)

**狀態**：完成 on 2026-04-21
**Commits**：`4a5b6fd` → `7261e7f`（Phase 2，共 11 commits）
**Tests**：75 passing（Phase 1 的 26 + Phase 2 新增 49）
**Branch**：`main`

---

## 這一階段做到了什麼

建好一個**可以跑完整對話 session 的 runner**：從載入設定 → 組裝 prompt → 呼叫 Gemini → 解析結構化輸出 → 落盤全部工件。smoke run 已用真實 API 驗證，兩個角色確實在說話，印象句確實被解出來。

### 可以做的事

```bash
# 跑一個實驗（真實 Gemini API）
uv run python scripts/run_experiment.py mother_x_son_hospital_v3_001

# 預期輸出：
# ✓ Completed mother_x_son_hospital_v3_001
#   Output: runs/mother_x_son_hospital_v3_001/2026-04-21T10-24-12
#   Turns: 20
#   Termination: max_turns
#   Tokens in/out: ~40000 / ~6000
#   Duration: ~300s
```

```python
# 程式碼層級
from empty_space.prompt_assembler import build_system_prompt, build_user_message
from empty_space.runner import run_session
from empty_space.llm import GeminiClient

config = load_experiment("mother_x_son_hospital_v3_001")
result = run_session(config=config, llm_client=GeminiClient())

# result.total_turns, result.total_tokens_in, result.total_tokens_out
# result.out_dir → runs/<exp_id>/<timestamp>/
```

---

## 重要決策記錄

### director_events 用事件觸發，不用 scripted injection

Phase 2 設計初期討論過「在指定 turn 強制注入台詞」。最終決定：director_events 是條件觸發的環境擾動，不是固定劇本。實驗 yaml 的 `director_events: {}` 是空白，Phase 3 的 Judge 決定觸發時機——Phase 2 runner 只傳遞 active events，不主動觸發。

### 沒有「演出指示」block

System prompt 不包含獨立的「演出指示」section。方向感完全由 `## 此刻`（動作詞 + 階段 + 模式）三行提供，其餘細節靠關係層的 `印象_種子` 和 `跳出觸發` 承載。這讓 prompt 的信噪比更高。

### 沒有 tail anchor

組裝完 prompt 後不追加任何「現在開始」類收尾句。Gemini 的輸出穩定，anchor 沒有必要，加了只會製造冗餘。

### Stateless history — verbatim dialogue

User message 直接串接前幾輪的完整對話文字（`[Turn N 角色名] 內容`），不做任何摘要或壓縮。每輪 context 線性增長。Phase 4 帳本做好之前，這是最安全的策略：損失最少信息，cost 可控（Flash 很便宜）。

### Hard-wired speaker alternation

母親 → 兒子 → 母親 → 兒子。沒有判斷邏輯。Phase 3 Judge 上線前，alternation 就是靠 runner 的 `turn % 2` 決定。

### Model hardcoded as gemini-2.5-flash

所有角色演出走 Flash。Phase 3 之前沒有 Pro 的必要。這是明確的 defer，不是疏失。

---

## 模組清單

```
src/empty_space/
├── paths.py           # PROJECT_ROOT, PERSONA_ROOT, EXPERIMENTS_DIR, RUNS_DIR
├── schemas.py         # Persona, Setting, ExperimentConfig + nested types（Phase 2 新增 ScenePremise, DirectorEvent）
├── loaders.py         # load_persona, load_setting, load_experiment
├── llm.py             # GeminiClient, GeminiResponse
├── parser.py          # parse_response() — 分離 content 與 ---IMPRESSIONS--- block，graceful degradation
├── prompt_assembler.py # build_system_prompt（五段結構）, build_user_message（verbatim history）
├── writer.py          # init_run, append_turn, write_meta — 全部工件落盤
└── runner.py          # run_session — turn loop orchestrator
```

**scripts/**
- `run_experiment.py` — CLI 入口，一個 exp_id → 一次完整 session

---

## 落盤工件結構

```
runs/<exp_id>/<timestamp>/
├── config.yaml           # 凍結的 experiment config
├── conversation.md       # 人讀版逐輪對話
├── conversation.jsonl    # 機讀版（每行一個 turn object）
├── meta.yaml             # session 摘要（token 總量、duration、parse errors）
└── turns/
    ├── turn_001.yaml     # 完整 turn record（prompt, response, impressions, events）
    ├── turn_002.yaml
    └── ...
```

每個 `turn_NNN.yaml` schema（spec §7.2）：
- `turn`, `speaker`, `persona_name`, `timestamp`
- `prompt_assembled`: `system`, `user`, `tokens`
- `response`: `content`, `raw`, `tokens_out`, `model`, `latency_ms`
- `candidate_impressions`: list of `{text, symbols}`
- `director_events_active`: list（Phase 2 全為空）
- `parse_error`: null 或錯誤描述

---

## 測試覆蓋

| 測試檔 | 覆蓋範圍 | 數量（約）|
|--------|---------|---------|
| test_schemas_*.py | Pydantic 驗證、邊界值 | Phase 1 |
| test_loaders_*.py | 載入 + path traversal guard | Phase 1 |
| test_llm_gemini.py | GeminiClient mock | Phase 1 |
| test_integration_phase1.py | 完整 load chain | Phase 1 |
| test_parser.py | marker 有/無、malformed YAML、空 impressions | Phase 2 |
| test_prompt_assembler.py | system prompt 五段結構、history 串接 | Phase 2 |
| test_writer.py | init_run、append_turn（yaml/md/jsonl）、write_meta | Phase 2 |
| test_runner_integration.py | happy path、parse error、max_turns、fire_release、exception 隔離 | Phase 2 |
| **合計** | | **75 passing** |

---

## Smoke Run 結果（2026-04-21）

```
exp_id:           mother_x_son_hospital_v3_001
turns:            4 (max_turns 臨時設為 4)
termination:      max_turns
tokens in / out:  8204 / 1350
duration:         60.8s
parse_error:      0 turns
candidate_impressions: 17 (4 turns 合計)
models_used:      [gemini-2.5-flash]
```

**對話品質**：四輪都是純肢體 / 感官描寫，沒有對話。母親朝兒子傾斜，兒子用靜止拒絕。印象句都是從角色材料裡自然浮現的語句（「她的苦是她在場時空氣密度的改變」），不是 Gemini 自己發明的。兩個角色的 persona 材料確實被吸收進來了。

**注意點**：60.8 秒跑 4 輪，單輪平均 15 秒。20 輪的 session 估計 ~5 分鐘。符合預期，沒有 timeout 問題。

---

## 已知問題 / 後續追蹤

### Flash rubric 品質（spec §8.1）
印象句的評分（rubric）尚未實作（Phase 4 的事）。目前 candidate_impressions 是 Gemini 自評產出，沒有外部驗證。如果 Phase 4 發現帳本被灌水，應急路徑：調閾值 → 改三維度 rubric → 升 Pro。

### 開場誰先說話的假說（spec §8.3）
目前 Turn 1 固定是 protagonist（母親）。smoke run 四輪沒有任何語言——純肢體。這引出一個未驗證的假說：**環境先開場（設定句先於人物）** 可能比讓角色先開口更自然。等 Phase 3 Judge 上線後測試。

### persona 版本鎖（spec §8.2）
`experiments/mother_x_son_hospital_v3_001.yaml` 用 `version: v3_tension`。如果 `演員方法論xhermes` repo 的 persona 材料更新，舊的 runs 不能重跑出相同結果。目前可接受，未來需要考慮 snapshot 策略。

---

## Phase 3 起點

開新 session 後：

1. 讀這個 summary + spec §3.2（session 生命週期流程圖）+ spec §6（Judge 狀態機）
2. Phase 3 目標：**Layer 2 Judge**
   - 每輪對話後呼叫 Flash，評估 stage / mode / 張力值 / fire_release / basin_lock
   - Judge 的輸出寫進 turn_NNN.yaml 的 `judge_state` 欄位（需擴充 schema）
   - runner 讀 Judge 結果決定是否觸發 director_events 或 termination
3. 關鍵設計題：
   - Judge prompt 的結構：輸入什麼（最近 N 輪 + 角色材料 + 當前 state）？
   - `fire_release` / `basin_lock` 的判定邏輯
   - Judge 與 runner 的介面（同步 vs. 非同步？）

---

## 參考

- 主設計：`docs/superpowers/specs/2026-04-21-空的空間-design.md`
- Phase 2 plan：`docs/superpowers/plans/`（Phase 2 implementation plan）
- Phase 1 summary：`docs/phase1-summary.md`
