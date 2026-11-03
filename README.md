# MCDM-Skill

兩個給 AI agent（Claude Code、Codex 等支援 [Agent Skills](https://agentskills.io/) 標準的工具）使用的研究型 skill，所有計算都由附帶的程式完成，並以測試驗證：

| Skill | 用途 |
|---|---|
| [`mcdm`](skills/mcdm/SKILL.md) | 多準則決策：AHP、模糊 AHP（Buckley）、BWM、熵權法、CRITIC、ROC 權重；TOPSIS、模糊 TOPSIS（Chen）、VIKOR、PROMETHEE I/II、ELECTRE I、WSM/WPM/WASPAS 排序；DEMATEL；權重敏感度、排序反轉、SMAA 式排序可接受度、排序相似度（r_s、r_w、WS） |
| [`ortools-cpsat`](skills/ortools-cpsat/SKILL.md) | OR-Tools CP-SAT 限制式規劃：job-shop 排程、不等面積設施佈局（NoOverlap2D、旋轉、障礙物、間距、鄰近/隔離限制）、CP + 深度強化學習混合架構（warm start、fix 評估、relax 鄰域） |

## 驗證

- `mcdm`：42 個測試。與 [pymcdm](https://github.com/kotbaton/pymcdm) 1.4.0 交叉驗證 TOPSIS、VIKOR、WASPAS、PROMETHEE II（五種偏好函數）、熵權法、CRITIC、加權 Spearman、WS 係數；另以一致矩陣、完全一致 BWM、手算模糊 TOPSIS、DEMATEL 級數展開等可推導結果檢查。
- `ortools-cpsat`：12 個測試。ft06 必須得到已知最佳 makespan 55；小型佈局的最佳值與暴力窮舉一致；所有解都經過不依賴模型的獨立驗證器檢查。
- 排名一律用 `rank()`（1 = 最佳），測試中明確擋下 `np.argsort(...) + 1` 這種常見錯誤。

## 安裝

需要 Python ≥ 3.10。

**Claude Code / Codex（手動複製，最確定）**

```bash
git clone https://github.com/cmh3601/MCDM-Skill.git
```

```bash
cp -R MCDM-Skill/skills/mcdm MCDM-Skill/skills/ortools-cpsat ~/.claude/skills/
```

```bash
cp -R MCDM-Skill/skills/mcdm MCDM-Skill/skills/ortools-cpsat ~/.codex/skills/
```

**使用 skills 安裝工具**

```bash
npx skills add cmh3601/MCDM-Skill
```

**Python 套件**（建議每個研究專案一個獨立環境）

```bash
uv venv && uv pip install numpy scipy ortools pymcdm pytest
```

`pymcdm` 只用於交叉驗證，可省略。

## 使用

安裝後直接用自然語言描述任務，例如：

- 「用 AHP 算這 5 個準則的權重並檢查 CR，再用 TOPSIS 和 VIKOR 排序 6 家供應商，做權重敏感度分析」
- 「用模糊 AHP（Buckley）加模糊 TOPSIS 評估這些方案，TFN 尺度用預設值並列出來」
- 「用 CP-SAT 建 flexible job shop 模型，以 DRL 產生的排程當 hint，報告與最佳下界的 gap」
- 「這個零售佈局，固定其他區域，只放開 Office 和 Returns 重新最佳化」

也可以直接執行：

```bash
python skills/mcdm/scripts/mcdm_cli.py skills/mcdm/scripts/example_spec.json --out results.json
```

```bash
python skills/ortools-cpsat/scripts/jobshop_cpsat.py skills/ortools-cpsat/data/ft06.txt --hint spt
```

```bash
python skills/ortools-cpsat/scripts/layout_cpsat.py skills/ortools-cpsat/examples/layout_8.json --svg layout.svg
```

## 測試

```bash
python -m pytest skills -q
```

GitHub Actions 設定放在 [`ci/github-actions-tests.yml`](ci/github-actions-tests.yml)。要啟用時，在 GitHub 網頁上新增 `.github/workflows/tests.yml` 並貼上內容（用命令列推送需要具備 `workflow` 權限的 token）。

## 結構

```
skills/
  mcdm/
    SKILL.md
    scripts/   mcdm_core.py, mcdm_cli.py, example_spec.json
    references/ methods.md, fuzzy.md, library-map.md, reporting-checklist.md
    tests/
  ortools-cpsat/
    SKILL.md
    scripts/   jobshop_cpsat.py, layout_cpsat.py
    data/ft06.txt, examples/layout_8.json
    references/ modeling-patterns.md, solver-and-reproducibility.md, cp-drl-hybrid.md
    tests/
```

## 注意

- 參考文獻清單列在各 skill 的 `references/`；投稿前請逐筆對照出版社紀錄。
- 模糊尺度、PROMETHEE 門檻、ELECTRE 門檻、CP-SAT 時間預算等都是建模選擇，論文中需明確報告。

## 授權

MIT，見 [LICENSE](LICENSE)。
