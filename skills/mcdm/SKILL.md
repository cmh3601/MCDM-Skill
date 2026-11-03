---
name: mcdm
description: >-
  Runs and audits multi-criteria decision-making (MCDM/MCDA) studies with tested
  code: AHP, fuzzy AHP (Buckley), BWM, entropy, CRITIC and ROC weights; TOPSIS,
  fuzzy TOPSIS, VIKOR, PROMETHEE I/II, ELECTRE I, WASPAS/WSM/WPM rankings;
  DEMATEL; weight sensitivity, rank reversal, SMAA-style acceptability and
  ranking-similarity (rs, rw, WS). Use for supplier/site/project/technology
  selection, resource allocation, criteria weighting, consistency checks, or
  reviewing an MCDM paper. Triggers: MCDM, MCDA, AHP, 層級分析, 模糊 AHP, TOPSIS,
  VIKOR, PROMETHEE, ELECTRE, BWM, 最佳最劣法, DEMATEL, 熵權法, CRITIC, 多準則決策,
  權重, 一致性比率, 敏感度分析. Not for continuous multi-objective optimisation
  (use pymoo) or for picking a point on a Pareto front without criteria data.
metadata:
  version: "1.0.0"
  author: "張明瀚 (cmh3601)"
  license: MIT
compatibility: Python >= 3.10 with numpy and scipy; pymcdm optional for cross-checks.
---

# MCDM

Every number in an MCDM result must come from the bundled code or from a
library call, never from mental arithmetic. The bundled implementations in
`scripts/mcdm_core.py` are cross-checked against pymcdm and hand-derived cases
(`tests/test_mcdm_core.py`).

## Workflow

1. **Structure the problem.** Write down the goal, alternatives (m), criteria
   (n), each criterion's direction (benefit/cost), unit and data source. Check
   that criteria are not double-counting the same effect. If criteria influence
   each other, map the dependence with DEMATEL first and say how you treat it.
2. **Choose weights and justify the choice.**
   - Subjective: AHP (`ahp_weights`), fuzzy AHP (`fuzzy_ahp_buckley`), BWM
     (`bwm_weights`, fewer comparisons: 2n - 3), ROC when only an importance
     order is known (`roc_weights`).
   - Objective: entropy (`entropy_weights`), CRITIC (`critic_weights`),
     standard deviation (`std_weights`).
   - Hybrid: `combine_weights(subjective, objective, alpha)`; report alpha.
   - Group judgments: aggregate matrices with AIJ (`aggregate_judgments`,
     geometric mean) or priorities with AIP (`aggregate_priorities`).
3. **Check consistency before using subjective weights.** AHP: CR < 0.10
   (Saaty RI built in for n <= 10; pass `ri=` for larger n and cite the table).
   Inspect `most_inconsistent` and send those judgments back to the expert.
   BWM: report `xi` and `CR_input`. Fuzzy AHP: report the CR of the modal
   matrix.
4. **Rank with at least two methods of different families** (distance-based
   TOPSIS/VIKOR, outranking PROMETHEE/ELECTRE, utility WSM/WASPAS). Pick by the
   decision context, see `references/methods.md`.
5. **Test robustness.** One-at-a-time weight sensitivity
   (`weight_sensitivity`), rank-reversal check (`rank_reversal_check`),
   SMAA-style rank acceptability under weight uncertainty
   (`weight_acceptability`), and agreement between methods
   (`compare_rankings`: Spearman, Kendall, weighted Spearman, WS).
6. **Report** using `references/reporting-checklist.md`.

## Running it

From a JSON spec (weights, several methods, comparison and robustness in one go):

```bash
python scripts/mcdm_cli.py scripts/example_spec.json --out results.json
```

Pairwise entries may be written as strings like `"1/3"`. Weight methods:
`given | equal | entropy | critic | std | roc | ahp | fuzzy_ahp | bwm`, with
optional `"combine_with": "critic"` and `"alpha"`. Methods: `wsm`, `wpm`,
`waspas`, `topsis`, `vikor`, `promethee` (see the example for parameters).
Robustness runs on the first method listed.

From Python (fuzzy TOPSIS, DEMATEL and ELECTRE details are easiest here):

```python
import sys; sys.path.insert(0, "<skill-dir>/scripts")
import mcdm_core as mc
res = mc.topsis(X, w, ["cost", "benefit", "benefit"])
res.ranks          # 1 = best
res.to_dict(names) # JSON-ready table
```

Run the tests after any change to the code:

```bash
python -m pytest <skill-dir>/tests -q
```

## Rules that prevent common errors

- Ranks are 1 = best and come from `mc.rank` (scipy `rankdata`). Never use
  `np.argsort(scores) + 1`: it returns which alternative holds each position,
  not each alternative's rank.
- VIKOR and its Q are "lower is better"; check `Result.higher_is_better`.
- Weights must sum to 1; the code refuses otherwise instead of renormalising silently.
- State the normalisation. TOPSIS here defaults to vector normalisation
  (Hwang & Yoon); pymcdm's `TOPSIS()` defaults to min-max and gives different
  numbers. See `references/library-map.md` for other library differences.
- Use Buckley's geometric-mean fuzzy AHP, not Chang's extent analysis, which
  can assign zero weight to relevant criteria (Wang, Luo & Hua, 2008). The TFN
  scale is a modelling choice; report it (`SAATY_TFN` is the default).
- ELECTRE I thresholds default to the mean concordance/discordance; always
  report the thresholds and whether the outranking graph has a cycle.
- VIKOR needs every criterion to vary across alternatives; drop constant
  criteria and say so.
- Entropy weights reflect data dispersion, not importance. Say this when you
  use them, and prefer a hybrid when experts are available.

## References in this skill

- `references/methods.md` — formulas, when to use each method, citations.
- `references/fuzzy.md` — TFN arithmetic, scales, fuzzy AHP and fuzzy TOPSIS details.
- `references/library-map.md` — pymcdm / pyDecision equivalents and convention differences.
- `references/reporting-checklist.md` — what a journal reviewer will look for.
