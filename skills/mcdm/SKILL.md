---
name: mcdm
description: >-
  Runs and audits multi-criteria decision-making (MCDM/MCDA) studies with tested
  code: AHP, fuzzy AHP, ANP, DANP (DEMATEL-based ANP), BWM, entropy, CRITIC and
  ROC weights; TOPSIS, fuzzy TOPSIS, VIKOR, fuzzy VIKOR, PROMETHEE I/II, ELECTRE
  I/II/III/IV, ELECTRE Tri-B sorting, GRA and WSM/WPM/WASPAS; DEMATEL and fuzzy
  DEMATEL; weight sensitivity, rank reversal, SMAA-style acceptability and
  ranking similarity. Use for supplier/site/project/technology selection,
  resource allocation, criteria weighting, interdependent criteria, sorting into
  ordered classes, or reviewing an MCDM paper. Triggers: MCDM, MCDA, AHP, ANP,
  DANP, 層級分析, 網路分析法, 模糊 AHP, TOPSIS, VIKOR, 模糊 VIKOR, PROMETHEE,
  ELECTRE, ELECTRE TRI, BWM, DEMATEL, 灰關聯, GRA, 熵權法, CRITIC, 多準則決策, 權重,
  一致性比率, 敏感度分析. Not for continuous multi-objective optimisation (use
  pymoo) or picking a point on a Pareto front without criteria data.
metadata:
  version: "1.1.0"
  author: "張明瀚 (cmh3601)"
  license: MIT
compatibility: Python >= 3.10 with numpy and scipy; pymcdm and pyDecision optional for cross-checks.
---

# MCDM

Every number in an MCDM result must come from the bundled code or from a
library call, never from mental arithmetic. The implementations in `scripts/`
are tested in `tests/` against pymcdm, pyDecision (where conventions match),
hand-derived cases and crisp limits of the fuzzy methods. `README.md` in this
folder lists every method with its function and call pattern.

## Modules

| Module | Contents |
|---|---|
| `mcdm_core.py` | weights (AHP, fuzzy AHP, BWM, entropy, CRITIC, std, ROC, AIJ/AIP, hybrid); TOPSIS, fuzzy TOPSIS, VIKOR, PROMETHEE I/II, ELECTRE I, WSM, WPM, WASPAS, GRA; DEMATEL; sensitivity, rank reversal, acceptability, rank correlations |
| `mcdm_network.py` | ANP (supermatrix builder, weighted supermatrix, exact Cesàro limit), DANP |
| `mcdm_outranking.py` | ELECTRE II, III, IV, Tri-B; Roy's distillation |
| `mcdm_fuzzy.py` | fuzzy VIKOR, fuzzy DEMATEL, CFCS and other defuzzification, TFN arithmetic |
| `mcdm_cli.py` | runs a whole study from a JSON spec |

## Workflow

1. **Structure the problem.** Goal, alternatives (m), criteria (n), each
   criterion's direction, unit and data source. Check for double counting.
2. **Decide whether criteria are independent.** If not, map the dependence with
   DEMATEL (or fuzzy DEMATEL) and weight with DANP, or build an ANP network.
3. **Choose weights and justify them.** Subjective: AHP, fuzzy AHP, BWM (2n − 3
   comparisons), ROC. Objective: entropy, CRITIC, std. Interdependent: ANP, DANP.
   Hybrid: `combine_weights`. Group: AIJ / AIP.
4. **Check consistency.** AHP CR < 0.10 (Saaty RI for n <= 10, else pass `ri=`);
   inspect `most_inconsistent`. BWM: `xi`, `CR_input`. Fuzzy AHP: modal CR. ANP:
   CR of every pairwise block (`supermatrix_from_comparisons` records them).
5. **Rank (or sort) with at least two families**: distance (TOPSIS, VIKOR, GRA),
   outranking (PROMETHEE, ELECTRE), utility (WSM, WASPAS). For assignment to
   ordered classes use ELECTRE Tri-B and report both pessimistic and optimistic
   results. See `references/methods.md` for choosing.
6. **Test robustness**: `weight_sensitivity`, `rank_reversal_check`,
   `weight_acceptability`, `compare_rankings`.
7. **Report** with `references/reporting-checklist.md`.

## Running it

```bash
python scripts/mcdm_cli.py scripts/example_spec.json --out results.json
python scripts/mcdm_cli.py scripts/example_danp_spec.json
```

Spec keys: `weights.method` in `given | equal | entropy | critic | std | roc |
ahp | fuzzy_ahp | bwm | danp` (+ optional `combine_with`, `alpha`); `methods`
list from `wsm, wpm, waspas, topsis, vikor, promethee, gra`; optional blocks
`electre_i`, `electre_ii`, `electre_iii`, `electre_iv`, `electre_tri`;
`analysis` with `sensitivity`, `rank_reversal`, `acceptability`. Pairwise
entries may be strings like `"1/3"`.

Python use (ANP networks, fuzzy ratings, custom pipelines):

```python
import sys; sys.path.insert(0, "<skill-dir>/scripts")
import mcdm_core as mc, mcdm_network as mn, mcdm_outranking as eo, mcdm_fuzzy as mf
res = mc.topsis(X, w, ["cost", "benefit", "benefit"]); res.ranks  # 1 = best
out = eo.electre_iii(X, w, types, q, p, v); out["descending"]["classes"]
d = mn.danp(Z, groups=[[0, 1, 2], [3, 4]]); d["weights"]
```

Run the tests after any change: `python -m pytest <skill-dir>/tests -q`.

## Rules that prevent common errors

- Ranks are 1 = best and come from `mc.rank`; never `np.argsort(scores) + 1`.
- VIKOR / fuzzy VIKOR Q is "lower is better"; check `Result.higher_is_better`.
- Weights must sum to 1; the code refuses otherwise.
- State the normalisation, thresholds (PROMETHEE q/p/s, ELECTRE q/p/v, c/d
  levels, Tri-B cut λ and profiles), VIKOR v, GRA ρ, TFN scale and
  defuzzification. They are modelling choices that change results.
- ELECTRE III/IV give two pre-orders and a partial final order; incomparability
  (`R`) is a result, not an error. `rank_by_average` is a convenience complete
  order, not Roy's method; say so if you use it.
- ANP limit matrices are computed exactly (spectral projector), so cyclic
  networks converge. For reducible networks pass `reference=` (e.g. the goal).
- Use Buckley's fuzzy AHP, not Chang's extent analysis (Wang, Luo & Hua, 2008).
- Library defaults differ from the textbook in several places (pymcdm TOPSIS
  normalisation, pyDecision Tri-B pessimistic rule, ...): `references/library-map.md`.

## References in this skill

- `README.md` — method catalogue with function names and invocation.
- `references/methods.md` — formulas, method choice, citations.
- `references/fuzzy.md` — TFN arithmetic, scales, fuzzy AHP/TOPSIS/VIKOR/DEMATEL, CFCS.
- `references/library-map.md` — pymcdm / pyDecision equivalents and verified differences.
- `references/reporting-checklist.md` — what a journal reviewer will look for.
