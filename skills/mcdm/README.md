# mcdm skill

多準則決策（MCDM/MCDA）skill。AI agent 依 [`SKILL.md`](SKILL.md) 的流程選方法、呼叫 `scripts/` 裡的程式計算，所有數字都來自程式，不靠模型心算。全部方法都有測試（`tests/`，77 個）。

## 可用方法

「CLI」欄表示能否直接寫進 JSON 設定檔由 `mcdm_cli.py` 執行；「否」的方法用 Python 呼叫。

### 權重

| 方法 | 函式 | 模組 | CLI | 驗證方式 |
|---|---|---|---|---|
| AHP（特徵向量法／幾何平均法、CI、CR、最不一致判斷） | `ahp_weights` | core | `ahp` | 一致矩陣還原權重、CR = 0 |
| 群體 AHP：AIJ（判斷矩陣幾何平均）／AIP（權重平均） | `aggregate_judgments`、`aggregate_priorities` | core | 否 | 手算、互反性保持 |
| 模糊 AHP（Buckley 幾何平均法） | `fuzzy_ahp_buckley`、`tfn_from_crisp` | core | `fuzzy_ahp` | 明確數極限等於 AHP |
| ANP 網路分析法（超矩陣、叢集權重、極限超矩陣） | `supermatrix_from_comparisons`、`weighted_supermatrix`、`limit_supermatrix`、`anp` | network | 否 | 層級網路重現 AHP 合成；循環網路 Cesàro 極限；與冪次法、pyDecision 比對 |
| DANP（DEMATEL-based ANP） | `danp` | network | `danp` | 超矩陣欄和為 1；單準則構面化簡為穩態分配 |
| BWM 最佳最劣法（線性模型、ξ、CR_I） | `bwm_weights` | core | `bwm` | 完全一致時 ξ = 0、還原權重 |
| 熵權法 | `entropy_weights` | core | `entropy` | 與 pymcdm 比對 |
| CRITIC | `critic_weights` | core | `critic` | 與 pymcdm 比對（全效益型） |
| 標準差法 | `std_weights` | core | `std` | — |
| ROC 順位重心法 | `roc_weights` | core | `roc` | 公式手算 |
| 主客觀混合權重 | `combine_weights` | core | `combine_with` + `alpha` | — |

### 排序與分類

| 方法 | 函式 | 模組 | CLI | 驗證方式 |
|---|---|---|---|---|
| TOPSIS（向量／min-max 正規化） | `topsis` | core | `topsis` | 與 pymcdm 比對、手算 |
| 模糊 TOPSIS（Chen 2000） | `fuzzy_topsis` | core | 否 | 手算 |
| VIKOR（含 C1/C2 妥協解） | `vikor` | core | `vikor` | 與 pymcdm 比對 |
| 模糊 VIKOR（模糊算術，Opricovic 2011 架構） | `fuzzy_vikor` | fuzzy | 否 | 明確數極限等於 VIKOR（效益型與成本型） |
| PROMETHEE I（部分排序）／II（淨流量），六種偏好函數 | `promethee`、`preference_degree` | core | `promethee` | 與 pymcdm 比對五種偏好函數 |
| ELECTRE I（核心集） | `electre_i` | core | `electre_i` 區塊 | 手算鏈狀案例 |
| ELECTRE II（強／弱級別關係、雙向排序） | `electre_ii` | outranking | `electre_ii` 區塊 | 關係矩陣與 pyDecision 相同；循環處理 |
| ELECTRE III（可信度、Roy 蒸餾、最終部分排序） | `electre_iii`、`distillation` | outranking | `electre_iii` 區塊 | 可信度矩陣與 pyDecision 相同；手算否決 |
| ELECTRE IV（不需權重） | `electre_iv`、`electre_iv_counts` | outranking | `electre_iv` 區塊 | 五種關係巢狀包含；計數手算 |
| ELECTRE Tri-B（悲觀／樂觀分類） | `electre_tri_b` | outranking | `electre_tri` 區塊 | 樂觀法與 pyDecision 相同；悲觀法手算 |
| GRA 灰關聯分析 | `gra` | core | `gra` | 理想方案關聯度 = 1；與 pyDecision 比對 |
| WSM（SAW）、WPM、WASPAS | `wsm`、`wpm`、`waspas` | core | `wsm`、`wpm`、`waspas` | WASPAS 與 pymcdm 比對 |

### 因果結構

| 方法 | 函式 | 模組 | CLI | 驗證方式 |
|---|---|---|---|---|
| DEMATEL | `dematel` | core | 經由 `danp` | T = N + NT、級數展開 |
| 模糊 DEMATEL（Lin & Wu 2008 方式） | `fuzzy_dematel` | fuzzy | 否 | 明確數極限等於 DEMATEL |
| CFCS 解模糊化 | `cfcs`、`defuzzify` | fuzzy | 否 | 明確數極限、單調性 |

### 穩健性與比較

| 分析 | 函式 | CLI |
|---|---|---|
| 單一權重敏感度（找出最佳方案改變的權重點） | `weight_sensitivity` | `analysis.sensitivity` |
| 排序反轉檢查（逐一移除方案） | `rank_reversal_check` | `analysis.rank_reversal` |
| SMAA 式排序可接受度（Dirichlet 抽樣權重） | `weight_acceptability` | `analysis.acceptability` |
| 排序一致性：Spearman、Kendall τ_b、加權 Spearman r_w、WS | `compare_rankings` | 多個方法時自動輸出 |

模組對照：core = `scripts/mcdm_core.py`、network = `scripts/mcdm_network.py`、outranking = `scripts/mcdm_outranking.py`、fuzzy = `scripts/mcdm_fuzzy.py`。

### 未內建但可經由函式庫使用

- **pymcdm**：ARAS、COCOSO、CODAS、COMET、COPRAS、EDAS、MABAC、MAIRCA、MARCOS、MOORA、OCRA、PROBID、RAM、RIM、SPOTIS；MEREC、LOPCOW、CILOS、IDOCRIW 等權重。
- **pyDecision**（GPL-3.0）：ELECTRE Tri-C／Tri-nB／Tri-nC、PROMETHEE III–VI 與 GAIA、模糊 BWM、TODIM、MACBETH、UTADIS、WINGS 等。

兩者的預設值有幾處與教科書不同，已查證並列在 [`references/library-map.md`](references/library-map.md)。

## 呼叫機制

### 1. 自然語言（自動觸發）

Claude Code 與 Codex 會讀 `SKILL.md` 開頭的 `description`。只要任務提到 AHP、ANP、DANP、TOPSIS、VIKOR、ELECTRE、多準則決策、權重、一致性比率等關鍵字，就會自動載入這個 skill。例如：

> 用 DEMATEL 找出 5 個準則的因果關係，再用 DANP 算權重，最後以 VIKOR 排序 4 個方案並做敏感度分析。

### 2. 明確指定

- Claude Code：`/mcdm` 後接任務描述。
- Codex：輸入 `$mcdm`（或用 `$`、`/skills` 選單挑選）後接任務描述。

### 3. JSON 設定檔 + CLI

```bash
python scripts/mcdm_cli.py scripts/example_spec.json --out results.json
```

```bash
python scripts/mcdm_cli.py scripts/example_danp_spec.json
```

| 欄位 | 內容 |
|---|---|
| `alternatives` | 方案名稱 |
| `criteria` | `[{"name": ..., "type": "benefit" 或 "cost"}]`；`weights.method = "given"` 時另加 `"weight"` |
| `matrix` | m × n 決策矩陣 |
| `weights` | `method`：`given`、`equal`、`entropy`、`critic`、`std`、`roc`、`ahp`、`fuzzy_ahp`、`bwm`、`danp`；可加 `combine_with`（`entropy`／`critic`／`std`）與 `alpha` |
| `methods` | `[{"name": "topsis", ...參數}]`，可用 `wsm`、`wpm`、`waspas`、`topsis`、`vikor`、`promethee`、`gra` |
| `analysis` | `sensitivity`、`rank_reversal`、`acceptability`（以第一個方法執行） |
| `electre_i`、`electre_ii` | 可為 `{}`（使用預設門檻）或指定門檻 |
| `electre_iii`、`electre_iv` | `{"q": [...], "p": [...], "v": [...]}` |
| `electre_tri` | `{"profiles": [[...], ...], "q": [...], "p": [...], "v": [...], "cut": 0.75}` |

AHP 成對比較矩陣可以寫 `"1/3"` 這類分數字串。輸出是 JSON，包含權重、各方法排名表、中間結果與穩健性分析。

### 4. Python API

```python
import sys
sys.path.insert(0, "skills/mcdm/scripts")
import mcdm_core as mc, mcdm_network as mn, mcdm_outranking as eo, mcdm_fuzzy as mf

res = mc.topsis(X, w, ["cost", "benefit", "benefit"])
res.ranks                 # 1 = 最佳
res.to_dict(names)        # 可直接存成 JSON 的排名表

anp = mn.anp(W, cluster_index, cluster_weights=Cw)
anp["priorities_by_cluster"]

out = eo.electre_iii(X, w, types, q=q, p=p, v=v)
out["descending"]["classes"], out["ascending"]["classes"], out["relation"]

fz = mf.fuzzy_vikor(F, W_tfn, types, v=0.5)   # F 形狀 (m, n, 3)
fz.details["compromise_set"]
```

排序方法回傳 `Result`：`scores`（方法本身的分數）、`ranks`（1 = 最佳）、`higher_is_better`（VIKOR 為 `False`）、`details`（中間結果）。權重、ANP、DANP、ELECTRE 等回傳 dict。

### 執行流程

```
使用者描述任務
  → agent 依 description 載入 SKILL.md
  → 依 SKILL.md 決定：準則是否獨立 → 權重方法 → 一致性檢查 → 至少兩類排序方法 → 穩健性分析
  → 呼叫 scripts/（CLI 或 Python）取得所有數字
  → 依 references/reporting-checklist.md 整理報告
```

## 測試

```bash
python -m pytest tests -q
```

需要 numpy、scipy、pytest。有安裝 pymcdm 或 pyDecision 時會多跑交叉比對；沒裝時只略過那幾項，不影響其他測試。

## 參考文件

- [`references/methods.md`](references/methods.md)：公式、選用時機、引用文獻
- [`references/fuzzy.md`](references/fuzzy.md)：TFN 運算、模糊 AHP／TOPSIS／VIKOR／DEMATEL、CFCS
- [`references/library-map.md`](references/library-map.md)：pymcdm、pyDecision 對照與已查證的差異
- [`references/reporting-checklist.md`](references/reporting-checklist.md)：投稿前檢查清單
