# Level 2 Summary — 空的空間 (The Empty Space)

**狀態**：完成 on 2026-04-21
**Commits**：`3c69743` → `0cc0c8c`（Level 2，共 10 commits）
**Tests**：135 passing（Phase 2 的 75 + Level 2 新增 60）
**Branch**：`main`

---

## 這一關做到了什麼

Level 2 給演員一個跨場的記憶機制。每場 session 結束後，角色說出的印象句（candidate impressions）沉進兩本帳本；下場開場前，Flash 從導演手寫的 prelude + scene_premise 拆出 symbols，撈帳本中符合的印象，注入演員的 system prompt「你的內在」block。

具體 shipped：

- **append-only ledger**：`ledgers/母親_x_兒子.from_母親.yaml` + `.from_兒子.yaml`，每場寫入，永不刪除；symbol_index + 1-hop co-occurrence 支援擴展查詢
- **session-start retrieval**：Flash 從 prelude + scene_premise 拆 symbols → 同義詞展開 → 帳本掃描 → 分數排序 → top-N 印象進 prompt
- **你的內在 block**：`prompt_assembler.py` 新增這段 — 只在有帳本印象時出現，靜默時消失
- **synonym map**：`config/symbol_synonyms.yaml` 手工維護，彌補 Flash 用詞漂移
- **experiment yaml 擴充**：`protagonist_prelude` / `counterpart_prelude` 兩個導演手寫欄位；`writer.py` + `runner.py` 落盤 `retrieval.yaml`

---

## 重要決策記錄

### 兩本帳本 per 關係（by speaker），A 策略（共同記憶）retrieval

帳本不是「角色的內在歷史」，是「這段關係的觀察紀錄，按說話者分開存」。Retrieval 採 A 策略：母親查的帳本 = 她自己說出的印象（她記住自己看見了什麼）；兒子同理。每個演員進場時帶的是自己前場留下的感知殘留，不是對方的。

### Rubric 砍掉

最初 spec 有「Flash 自己對自己生成的印象句打分」的 rubric loop。這條路廢掉了。Flash 評 Flash 等於沒有外部標準，而且讓 prompt 更重。分數改成 retrieval 時的 matched_symbols 數量（純符號重疊），不是品質評估。Level 4 Composer 做 holistic 整合時才需要真正的品質判斷。

### Composer bake 延後到 Level 4 的「一天後」

印象句進帳本的 `status` 預設 `candidate`，升 `refined` 是 Composer 的工作。Level 2 的帳本是原始累積，不做濃縮。這讓 Level 2 的邊界清晰：寫進去，撈出來，不加工。

### Session-start retrieval，不是 per-turn

Retrieval 只在 session 開始呼叫一次。Per-turn retrieval 會造成 attention dilution（每輪都被帳本的聲音稀釋），這個問題留給 Level 3 的 agentic pull 處理（演員主動說「我想起了什麼」，而不是每輪都被喂）。

### 1-hop co-occurrence 展開（graph 味道，不做全圖遍歷）

symbol_index 裡記錄每個 symbol 跟哪些其他 symbol 共同出現過（in same impression）。查詢時只展開一跳。這讓「夢」能連到「帶走」、「縮起來」，不需要建完整的圖結構。

### Synonym dict 手工維護

Flash 提取 symbols 時用詞會漂移（「拒絕」→「拒絕連結」；「縮」→「縮起來」）。目前用 `config/symbol_synonyms.yaml` 手動建群組，每次發現新的漂移就加進去。這是 bandage，不是根治——根治是 embedding similarity，留 future work。

---

## 模組清單

```
src/empty_space/
├── ledger.py          # Level 2 新增：append-only ledger CRUD + symbol_index
├── retrieval.py       # Level 2 新增：extract_symbols + retrieve_top_n + run_session_start_retrieval
├── schemas.py         # 修改：LedgerEntry, CandidateImpression, RetrievalResult + prelude fields
├── prompt_assembler.py # 修改：你的內在 block（有印象才出現）
├── writer.py          # 修改：write_retrieval() + turn yaml / meta 擴充
└── runner.py          # 修改：session-start retrieval → prompt 注入 → session-end ledger 寫入
```

**config/**
- `symbol_synonyms.yaml` — 同義詞群組（Level 2 新增）

---

## 測試覆蓋

| 測試檔 | 覆蓋範圍 | 數量（約）|
|--------|---------|---------|
| test_ledger.py | append / lookup / co-occurrence / symbol_index | 10 |
| test_retrieval_*.py | canonicalize / synonym / expand / score / orchestrator | 31 |
| test_runner_level2.py | cross-session ledger + retrieval integration | 5 |
| Phase 1 + Phase 2 tests | schemas / loaders / parser / assembler / writer / runner | 89 |
| **合計** | | **135 passing** |

---

## 跑法

```bash
# 跑一個實驗（真實 Gemini API）
PYTHONPATH=src uv run python scripts/run_experiment.py mother_x_son_hospital_v3_001

# 預期輸出：
# ✓ Completed mother_x_son_hospital_v3_001
#   Output: runs/mother_x_son_hospital_v3_001/2026-04-21T15-06-15
#   Turns: 4
#   Termination: max_turns
#   Tokens in/out: ~8700 / ~1200
#   Duration: ~70s
```

---

## Smoke Run 結果（2026-04-21）

### Session 1（帳本空白）

```
timestamp:        2026-04-21T15-03-59
turns:            4
termination:      max_turns
tokens in/out:    7982 / 912
duration:         93.4s
flash_extract:    母親 5602ms / 兒子 8758ms（Flash tokens: ~230in / ~41out each）
impressions hit:  0（first session，帳本空）
ledgers created:  母親_x_兒子.from_母親.yaml + 母親_x_兒子.from_兒子.yaml
```

### Session 2（帳本有 session 1 的印象）

```
timestamp:        2026-04-21T15-06-15
turns:            4
termination:      max_turns
tokens in/out:    8762 / 1185
duration:         71.3s
flash_extract:    母親 9912ms / 兒子 8486ms
impressions hit:  3（母親）+ 3（兒子）
```

**Session 2 retrieval 命中詳情：**

母親（query symbols 含：夢、帶走、濕 → 展開到 創傷、縮起來）
- `imp_011` score 3：「她的身體收縮著，像有人在她心裡輕輕提起了『被帶走』這三個字」
- `imp_016` score 1：「她的視線回到地面，像在尋找一個不會再被帶走的東西」
- `imp_003` score 1：「她說不在了，但那句話裡有太多層的不在了，身體分不清這次是哪一種」

兒子（query symbols 含：分手、拒絕、隱瞞 → 展開到 僵硬、入口、生死、等待）
- `imp_008` score 3：「他身體的僵硬是在宣告，他拒絕為任何連結開啟入口。」（兒子說的）
- `imp_011` score 3：「她的身體收縮著……」（母親說的，via 兒子帳本）
- `imp_012` score 3：「她感受著醫院裡冷白的空氣，知道這和生下他的那間產房一樣，都是等的地方」

注意：兒子的帳本撈到了母親的印象（`imp_011`, `imp_012`）——這是兩本帳本 A 策略的邊界模糊點（見已知問題）。

**對話品質（Session 2）**：四輪仍是純肢體描寫，母親傾身靠近，兒子僵硬抵禦。Turn 4 兒子出現了一個質變：「身處此地卻不參與，才是更深的缺席」——這句比 Session 1 深了一層，印象回注的效果有跡可循。

---

## 已知問題 / 後續追蹤

### Attention dilution（spec §11）
Session-start 一次注入全部印象。如果 N 場後帳本累積幾十條，早期的印象雖然進了 prompt 但會被稀釋。Level 3 的 agentic pull 應對這個：讓演員在 per-turn 中選擇性「想起」，而不是一開場就全部壓進去。

### Symbol 字串不一致（synonym dict 是 bandage）
Flash 提取時「拒絕」和「拒絕連結」是不同 token，「縮」和「縮起來」也是。synonym dict 手動打補丁。根治路徑：embedding cosine threshold 做 soft match，留 future work。

### 帳本膨脹（Level 4 Composer 濃縮 refined 帳本）
append-only + candidate 狀態的帳本會無限增長。Level 4 Composer 預計做：跨 session 後整合重複概念、升級最有代表性的印象為 `refined`、剔除低品質的 `candidate`。

### Semantic gap
目前 matching 是字串 exact match（展開後）。帳本裡的「等待」和查詢裡的「等著什麼」不會 hit。Embedding 是正確方向，但 Flash 成本可接受的前提下暫不追加。

### 兒子帳本混入母親印象（A 策略邊界）
Session 2 兒子帳本撈到了 `imp_011`（母親說的），因為那條印象也記在兒子的帳本裡（per-spec：兩本帳本各自從自己的視角記所有印象）。如果 A 策略的本意是「只帶自己說過的話」，需要在 retrieval 時加 `speaker == self` 過濾。待澄清。

---

## Level 3 入口

Level 3 = **Judge**（spec §3）：

- 每輪呼叫 Flash，評估 `stage` / `mode` / `張力值` / `fire_release` / `basin_lock`
- Judge 輸出寫進 `turn_NNN.yaml` 的 `judge_state` 欄位（需擴充 schema）
- Runner 讀 Judge 結果決定是否觸發 `director_events` 或 session termination
- Agentic pull retrieval（per-turn，演員主動想起）可在 Level 3 一併實作

開新 session 後讀：這個 summary + spec §3（stage/mode/張力 定義）+ spec §6（Judge 狀態機）。

---

## 參考

- 主設計：`docs/superpowers/specs/2026-04-21-空的空間-design.md`
- Level 2 plan：`docs/superpowers/plans/`
- Phase 2 summary：`docs/phase2-summary.md`
- Phase 1 summary：`docs/phase1-summary.md`
