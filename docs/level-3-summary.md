# Level 3 Summary — 空的空間 (The Empty Space)

**狀態**：完成 on 2026-04-22
**Commits**：`e0e06ad` → `c40532c`（Level 3，共 11 commits）
**Tests**：175 passing（Level 2 的 135 + Level 3 新增 40）
**Tag**：`level-3`
**Branch**：`main`

---

## 這一關做到了什麼

Level 3 給帳本加了一個 consolidation 層。每場 session 結束後，系統自動呼叫一次 Gemini Pro（`gemini-2.5-pro`），讀入對話內容 + 該場產出的 raw candidate impressions + 現有 refined 帳本，烘焙出一批精煉的 refined impressions，分流寫進兩本 refined 帳本。下場 session 的 retrieval 改從 refined 帳本撈，不碰 raw。

具體 shipped：

- **`composer.py`**：三層函式 —— `gather_composer_input`（整理輸入）+ `build_composer_prompt`（組 prompt）+ `parse_composer_output`（解析 YAML，含 markdown code fence stripping）+ `run_composer`（orchestrator）
- **refined ledger**：`ledgers/母親_x_兒子.from_母親.refined.yaml` + `.from_兒子.refined.yaml`，`ledger_version=1`，每場 Pro bake 後追加
- **retrieval 切換**：`retrieval.py` 解耦 ledger 類型，session-start retrieval 優先讀 refined；refined 空時 fallback raw
- **runner 整合**：`runner.py` 在 `finalize_session` 時自動呼叫 `run_composer`，寫入 refined 帳本
- **`writer.py` 擴充**：`write_meta` 加 Composer 欄位（`composer_ran`, `composer_refined_count_母親`, `composer_refined_count_兒子`）
- **schemas 擴充**：`RefinedImpression`, `ComposerInput`, `ComposerOutput` dataclasses

---

## 重要決策記錄（spec §0 回顧）

### 每 session 無條件 Pro bake

不做「夠多 candidate 才觸發」的條件判斷。每場結束就 bake 一次——腦內類比：每個事件之後有一次記憶固化，不管事件大小。這讓行為可預測，debug 容易。

### Minimal scope：只做 refined consolidation

Level 3 的邊界只有一件事：把 raw candidate 轉成精煉的 refined。21 格 cell 演算法、cluster、effective 關係層全部留 Level 4+。邊界越窄，測試越乾淨。

### Raw 保留為原料庫，retrieval 不碰 raw（refined 優先）

Raw 帳本繼續累積，Composer 用它當原料。Retrieval 只讀 refined。若 refined 為空（首場），fallback 到 raw。這讓 Level 2 的行為在 Level 3 向後相容。

### 單次 Pro call 產雙 section YAML，後處理分流

Prompt 要求 Pro 輸出一個 YAML，內含 `母親` 和 `兒子` 兩個 section。`parse_composer_output` 拆分後各自寫進對應的 refined 帳本。避免兩次 Pro call 的成本與 consistency 問題。

### Atomic + 第一人稱約束 + transformation examples

Prompt 明確要求：每條 refined 8-12 字、第一人稱（「我感覺到…」視角，使用「你」指對方）、身體/感官/象徵類語言、不含評論或文學判斷。並附 transformation examples 展示 raw → refined 的轉化方式。

---

## 模組清單

```
src/empty_space/
├── composer.py        # Level 3 新增：gather / build_prompt / parse / run_composer
├── ledger.py          # 修改：refined ledger I/O + raw append returns ids
├── retrieval.py       # 修改：解耦 ledger type + switch to refined with raw fallback
├── schemas.py         # 修改：RefinedImpression / ComposerInput / ComposerOutput
├── prompt_assembler.py # 未改（你的內在 block 繼續沿用）
├── writer.py          # 修改：write_meta Composer 欄位
└── runner.py          # 修改：session-end Composer call + refined ledger 寫入
```

---

## 測試覆蓋

| 測試檔 | 覆蓋範圍 | 數量（約）|
|--------|---------|---------|
| test_composer_*.py | gather_input / build_prompt / parse_output / run_composer / fence stripping | ~20 |
| test_runner_level3.py | Composer integration / writer Composer fields / retrieval switch | ~10 |
| test_ledger.py（擴充）| refined ledger I/O | ~5 |
| test_retrieval.py（擴充）| refined fallback / decouple | ~5 |
| Level 2 tests（不變）| 見 level-2-summary.md | 135 |
| **合計** | | **175 passing** |

---

## 跑法

```bash
# 跑一個實驗（真實 Gemini API）
PYTHONPATH=src uv run python scripts/run_experiment.py mother_x_son_act1_hospital

# 預期輸出（Level 3 新增行為）：
# ✓ Composer bake started...
# ✓ Composer wrote 5 refined impressions → 母親_x_兒子.from_母親.refined.yaml
# ✓ Composer wrote 5 refined impressions → 母親_x_兒子.from_兒子.refined.yaml
# ✓ Completed mother_x_son_act1_hospital
#   Output: runs/mother_x_son_act1_hospital/2026-04-22T02-54-47
#   Turns: 8
#   Tokens in/out: 16088 / 1324
#   Duration: 153.6s
```

Level 2 的跑法完全相容。Composer 在 `finalize_session` 自動觸發，不需要額外參數。

---

## Smoke Run 結果（2026-04-22，三幕連跑）

三幕依序：醫院 → 車上 → 家。帳本跨幕累積，Act 2/3 的 retrieval 讀 Act 1 產出的 refined 帳本。

### Act 1 — `mother_x_son_act1_hospital`（2026-04-22T02-54-47）

```
turns:             8
tokens in/out:     16088 / 1324
duration:          153.6s
composer bake:     成功（parser fix SHA c40532c — markdown code fence stripping）
refined 產出：     各 5 條（母親 / 兒子），ledger_version=1
```

**Refined impressions 樣本：**

母親：
- 「視線停在他身上，身體前傾」
- 「喉嚨發緊，聲音又細又小」
- 「手緩慢縮回，指尖蜷縮」
- 「視線落在冰冷的瓷磚上」
- 「走廊的空氣變得沉重」

兒子：
- 「視線越過她，釘在遠方時鐘」
- 「背脊緊緊貼著椅背」
- 「拳頭握緊，指甲陷進掌心」
- 「她的聲音聽起來很遙遠」（POV 漂移：用「她」非「你」，見已知問題）
- 「你的話築起一道牆」

品質評估：atomic、8-12 字、以身體/感官/象徵語言為主（視線/喉嚨/手/背/牆）。比 raw candidate 的長段文學評論乾淨很多。

### Act 2 — `mother_x_son_act2_car`（2026-04-22T02-59-10）

```
turns:             8
tokens in/out:     15159 / 1523
duration:          177.0s
retrieval hits:    母親 0 / 兒子 0（ZERO HITS）
composer bake:     成功
```

Retrieval 零命中。根本原因：**register mismatch**（見下節「CRITICAL FINDING」）。

Flash 從 Act 2 prelude + scene_premise 拆出的 symbols：
`[父親, 走, ICU, 離開, 母親, 走廊, 消毒水, 味道, 不知, 不問]` — 全場景詞。

Act 1 refined 的 symbols 全是身體感官詞：
`[視線, 喉嚨, 手, 背, 牆, 壓抑, 痛, 防禦]` — 兩組沒有 overlap。

Co-occurrence 展開也幫不上忙：refined 的 co-occurrence 也在身體感官詞之間循環，不連到場景詞。

### Act 3 — `mother_x_son_act3_home`（2026-04-22T03-03-47）

```
turns:             9
tokens in/out:     17473 / 1672
duration:          152.6s
retrieval hits:    母親 0 / 兒子 0（ZERO HITS）
composer bake:     成功
```

Flash 的 Act 3 query：`[兒子, 母親, 公寓, 電梯, 進入, 家, 天亮, 玄關, 地盤, 陌生]`

對比 Level 2 的 Act 3 行為：Level 2 的 raw candidates 是長句子，裡面同時含場景詞（「兒子」「母親」）和感受詞 — 兒子/母親出現在 raw → 透過 co-occurrence 連到感受詞 → 命中。Level 3 refined 把那些長句精煉成純感官符號後，橋樑詞彙丟失了。

---

## CRITICAL FINDING：Composer 單獨不解 retrieval 命中率，可能讓它更差

Level 3 解了兩件事，沒解一件事：

| 問題 | 結果 |
|------|------|
| Density compound（raw 長句 vs refined 短句）| ✓ 解了：每條 refined ~10 字 |
| POV leakage（第三人稱漂移）| ✓ 大致解了：10 條 refined 中 1 條漂移（兒子 ref_004 用「她」） |
| Register mismatch（retrieval 命中率）| ✗ **反而更嚴重** |

Register mismatch 的機制：

1. Flash extract（prelude → query symbols）走**場景詞**路線：地名、人物名、事件詞
2. Composer refined（raw candidates → refined impressions）走**身體感官詞**路線：視線、喉嚨、手、牆
3. 這兩組 register 在 Level 2 會透過「長句同時含兩種詞」自然橋接；在 Level 3，Composer 的精煉過程砍掉了場景詞，橋接消失

這驗證了 spec §11.7 的 risk 預言（「Composer 的精煉可能讓 retrieval 更難命中」），並且在實測中擴展它：問題不只是「命中率下降」，而是**完全歸零**（Act 2/3 各 0 hits）。

---

## 已知問題 / 後續追蹤

### §11.7 VALIDATED：Register mismatch（最高優先）

Flash extract prompt 設計和 Composer prompt 設計在 register 上跑反方向。修法方向（Level 4+ 任選一或多）：

1. **Flash extract prompt 調整**：改成「抽感受詞」而非「抽場景詞」，讓 query 的 register 往 refined 靠
2. **Symbol canonicalization / synonym dict 擴充**：建場景詞 ↔ 感受詞的映射（e.g. 「ICU」→「等待」「恐懼」）
3. **Dual-register refined**：Composer 產兩組 symbols — 感官符號（現有）+ 場景橋接詞（新增）
4. **Embedding-based retrieval**：放棄字串 exact match，改 semantic cosine similarity（根治）

### POV leakage 未完全解

1/10 refined impressions 漂成第三人稱（「她的聲音聽起來很遙遠」用「她」而非「你」）。Prompt 已有第一人稱約束，但 Pro 偶爾不守。可加 post-parse 驗證 + 重試，或 per-entry 過濾。Spec §11.1 定義這在可接受範圍，暫不列 blocker。

### Attention dilution（延續 Level 2 §11 known issue）

Refined 帳本累積後，session-start 一次注入仍有 dilution 風險。Agentic per-turn pull 留 Level 4。

### Refined co-occurrence 孤島

Refined 帳本的 symbol_index 只在感官詞之間連邊，形成孤島。Co-occurrence 展開對跨 register 查詢無效。根治需 register bridge（見上）。

---

## Level 4 入口

兩個平行 track，可獨立開始：

**Track A — Judge 機制**（spec §3）：
- 每輪呼叫 Flash 評估 `stage` / `mode` / `張力值` / `fire_release` / `basin_lock`
- Judge 輸出寫進 `turn_NNN.yaml` 的 `judge_state` 欄位（需擴充 schema）
- Runner 讀 Judge 結果決定 director_events 或 session termination
- Agentic pull retrieval（per-turn 主動想起）可在 Judge track 一併實作

**Track B — Retrieval Register Alignment**：
- 直接解本 level 暴露的 CRITICAL FINDING
- 最快路徑：Flash extract prompt 改抽感受詞（改一個 prompt，smoke test 驗 Act 2/3 命中率）
- 若不夠，加 scene ↔ feeling mapping dict 或 dual-register refined

開新 session 前讀：這個 summary + spec §3（stage/mode/張力 定義）+ spec §6（Judge 狀態機）+ spec §11.7（register mismatch 風險詳述）。

---

## 參考

- 主設計：`docs/superpowers/specs/2026-04-21-空的空間-design.md`
- Level 3 設計：`docs/superpowers/specs/` (Level 3 spec)
- Level 3 plan：`docs/superpowers/plans/`
- Level 2 summary：`docs/level-2-summary.md`
- Phase summaries：`docs/phase1-summary.md`, `docs/phase2-summary.md`
