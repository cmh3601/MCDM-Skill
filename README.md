# MCDM-Skill

給 AI agent（Claude Code、Codex 等支援 [Agent Skills](https://agentskills.io/) 標準的工具）使用的多準則決策（MCDM/MCDA）skill。所有計算都由附帶的程式完成，並以測試驗證。

> OR-Tools CP-SAT 限制式規劃的 skill 已移到 [cmh3601/ortools-cpsat](https://github.com/cmh3601/ortools-cpsat)。

## 功能（[`skills/mcdm`](skills/mcdm/)）

完整的方法清單、函式名稱與呼叫方式見 [`skills/mcdm/README.md`](skills/mcdm/README.md)。

| 類別 | 方法 |
|---|---|
| 主觀權重 | AHP（特徵向量法／幾何平均法、CR、找出最不一致的判斷）、模糊 AHP（Buckley）、BWM（線性模型，含 ξ 與 CR_I）、ROC |
| 相依準則權重 | ANP（超矩陣、叢集權重、精確 Cesàro 極限）、DANP（DEMATEL-based ANP） |
| 群體決策 | AIJ（判斷矩陣幾何平均）、AIP（權重向量算術／幾何平均） |
| 客觀權重 | 熵權法、CRITIC、標準差法 |
| 混合權重 | 主客觀線性或乘法組合 |
| 排序 | TOPSIS、模糊 TOPSIS（Chen）、VIKOR、模糊 VIKOR、PROMETHEE I/II（六種偏好函數）、ELECTRE I／II／III／IV、GRA 灰關聯、WSM、WPM、WASPAS |
| 分類 | ELECTRE Tri-B（悲觀／樂觀法） |
| 因果結構 | DEMATEL、模糊 DEMATEL、CFCS 解模糊化 |
| 穩健性與比較 | 單一權重敏感度、排序反轉檢查、SMAA 式排序可接受度、Spearman／Kendall／加權 Spearman／WS 係數 |

## 驗證

- 77 個測試。與 [pymcdm](https://github.com/kotbaton/pymcdm) 1.4.0 交叉驗證 TOPSIS、VIKOR、WASPAS、PROMETHEE II、熵權法、CRITIC、排序相似度；與 [pyDecision](https://github.com/Valdecy/pyDecision) 5.1.8 交叉驗證 ELECTRE II 關係矩陣、ELECTRE III 可信度、Tri-B 樂觀法、GRA、ANP 極限矩陣。
- 模糊方法都檢查「輸入為明確數時等於對應的明確方法」；ANP 以層級網路重現 AHP 合成、以循環網路驗證 Cesàro 極限。
- 與 pyDecision 不一致之處（ELECTRE IV、Tri-B 悲觀法、模糊 VIKOR、模糊 DEMATEL）已逐項查證，記錄在 [`library-map.md`](skills/mcdm/references/library-map.md)。
- 排名一律用 `rank()`（1 = 最佳），測試中明確擋下 `np.argsort(...) + 1` 這種常見錯誤。

## 安裝

需要 Python ≥ 3.10。

**Claude Code / Codex（手動複製，最確定）**

```bash
git clone https://github.com/cmh3601/MCDM-Skill.git
```

```bash
cp -R MCDM-Skill/skills/mcdm ~/.claude/skills/
```

```bash
cp -R MCDM-Skill/skills/mcdm ~/.codex/skills/
```

**使用 skills 安裝工具**

```bash
npx skills add cmh3601/MCDM-Skill
```

**Python 套件**（建議每個研究專案一個獨立環境）

```bash
uv venv && uv pip install numpy scipy pytest
```

交叉驗證用的套件可省略。若要一起跑，pyDecision 建議不裝它的 LLM 相關依賴：

```bash
uv pip install pymcdm pandas matplotlib networkx scikit-learn && uv pip install --no-deps pyDecision
```

## 使用

安裝後直接用自然語言描述任務，例如：

- 「用 AHP 算這 5 個準則的權重並檢查 CR，再用 TOPSIS 和 VIKOR 排序 6 家供應商，做權重敏感度分析」
- 「用模糊 AHP（Buckley）加模糊 TOPSIS 評估這些方案，TFN 尺度用預設值並列出來」
- 「用 DEMATEL 找出這些準則之間的因果關係，再用 DANP 算權重，以 VIKOR 排序」
- 「用 ELECTRE III（門檻 q、p、v 如下）排序，並用 ELECTRE Tri-B 把方案分成三級」

也可以直接執行，或用 `/mcdm`（Claude Code）、`$mcdm`（Codex）明確呼叫：

```bash
python skills/mcdm/scripts/mcdm_cli.py skills/mcdm/scripts/example_spec.json --out results.json
```

```bash
python skills/mcdm/scripts/mcdm_cli.py skills/mcdm/scripts/example_danp_spec.json
```

## 測試

```bash
python -m pytest skills -q
```

GitHub Actions 設定放在 [`ci/github-actions-tests.yml`](ci/github-actions-tests.yml)。要啟用時，在 GitHub 網頁上新增 `.github/workflows/tests.yml` 並貼上內容（用命令列推送需要具備 `workflow` 權限的 token）。

## 結構

```
skills/mcdm/
  SKILL.md, README.md
  scripts/    mcdm_core.py, mcdm_network.py, mcdm_outranking.py, mcdm_fuzzy.py,
              mcdm_cli.py, example_spec.json, example_danp_spec.json
  references/ methods.md, fuzzy.md, library-map.md, reporting-checklist.md
  tests/      test_mcdm_core.py, test_mcdm_network.py, test_mcdm_outranking.py,
              test_mcdm_fuzzy.py, test_pymcdm_crosscheck.py
```

## 注意

- 參考文獻清單列在 `skills/mcdm/references/`；投稿前請逐筆對照出版社紀錄。
- 模糊尺度、PROMETHEE 門檻、ELECTRE 門檻等都是建模選擇，論文中需明確報告。

## 授權

MIT，見 [LICENSE](LICENSE)。
