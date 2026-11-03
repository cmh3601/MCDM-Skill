# Library map: pymcdm and pyDecision

Use the bundled code for the methods it covers. Reach for a library for the
rest, and check its conventions first: the defaults below change results.

## pymcdm (tested against version 1.4.0)

`pip install pymcdm`. Kizielewicz, B., Shekhovtsov, A., & Sałabun, W. (2023).
pymcdm—The universal library for solving multi-criteria decision-making problems.
*SoftwareX*, 22, 101368.

Call pattern: `method(matrix, weights, types)` with types +1 (benefit) / -1 (cost);
weights must sum to 1. Most methods return preference scores; `.rank(pref)` gives ranks.

| Bundled | pymcdm equivalent | Difference |
|---|---|---|
| `topsis(normalization="vector")` | `TOPSIS(normalizations.vector_normalization)` | **`TOPSIS()` defaults to min-max**, not vector normalisation; scores differ. |
| `topsis(normalization="minmax")` | `TOPSIS()` | identical |
| `vikor(v)` | `VIKOR(normalizations.minmax_normalization, v=v)` | identical with min-max; check its behaviour with no normalisation before use |
| `waspas(lam)` | `WASPAS(l=lam)` | identical |
| `promethee` with usual/ushape/vshape/level/linear | `PROMETHEE_II('usual'|'ushape'|'vshape'|'level'|'vshape_2', p=..., q=...)` | `vshape_2` is the linear (type V) function; no Gaussian |
| `entropy_weights` | `weights.entropy_weights` | pymcdm sets a criterion's entropy to 0 if **any** value is 0, which inflates its weight; bundled code uses 0·ln 0 = 0 |
| `critic_weights(X, types)` | `weights.critic_weights` | pymcdm ignores criterion types; with mixed benefit/cost criteria the correlations change sign and weights differ |
| `weighted_spearman`, `ws_coefficient` | `correlations.weighted_spearman`, `correlations.ws` | identical |

Methods available only in pymcdm (use directly): ARAS, COCOSO, CODAS, COMET, COPRAS,
EDAS, MABAC, MAIRCA, MARCOS, MOORA, OCRA, PROBID, RAM, RIM, SPOTIS; weights MEREC,
LOPCOW, CILOS, IDOCRIW, Gini, angle.

## pyDecision (checked against version 5.1.8)

`pip install pyDecision` (GPL-3.0-or-later; very large method list). It pulls in openai,
google-genai and flask; `pip install --no-deps pyDecision` plus numpy, scipy, pandas,
matplotlib, networkx and scikit-learn is enough for the algorithms. Licence matters if
you redistribute code that imports it.

| Bundled | pyDecision | Verified relationship |
|---|---|---|
| `eo.electre_iii` credibility | `e_iii.global_concordance_matrix`, `credibility_matrix` | identical (benefit data) |
| `eo.distillation` | `e_iii.destilation_descending/ascending` | pyDecision evaluates s(λ) at λ_max instead of s(σ(a,b)) and recomputes λ inside ties; same descending pre-order on 192/200 random instances |
| `eo.electre_ii(discordance="global_range")` | `e_ii.dominance_matrix` | identical strong/weak relations; pyDecision assumes all criteria are benefit |
| `eo.electre_iv` | `e_iv.credibility_matrix` | **differs**: pyDecision's veto-dominance requires a veto to be present and its canonical-dominance test compares m_q with m_q; bundled code follows Figueira, Mousseau & Roy (2005) |
| `eo.electre_tri_b` optimistic | `electre_tri_b(rule="oc")` | identical after mapping categories (pyDecision numbers 0 = best) |
| `eo.electre_tri_b` pessimistic | `electre_tri_b(rule="pc")` | **differs**: pyDecision requires strict preference a P b_h; the standard rule needs a S b_h, so indifference to a profile is enough |
| `mf.fuzzy_vikor` | `fuzzy_vikor_method` | **differs**: pyDecision runs crisp VIKOR on the l, m, u matrices separately and averages; bundled code uses fuzzy arithmetic |
| `mf.fuzzy_dematel` | `fuzzy_dematel_method` | **differs**: pyDecision defuzzifies first with CFCS using per-row ranges |
| `gra` | `gra_method` | same ordering; pyDecision divides grades by the number of alternatives |
| `mn.limit_supermatrix` | `anp_method(cesaro=False)` | identical on primitive matrices; pyDecision's Cesàro option averages up to `max_iter` powers, which converges slowly |

Methods only in pyDecision: ELECTRE I-s / I-v / Tri-C / Tri-nB / Tri-nC, PROMETHEE III–VI and
GAIA, fuzzy BWM / CRITIC / EDAS / MOORA / WASPAS, MACBETH, ORESTE, REGIME, TODIM, UTADIS,
WINGS, and more. Validate each function you use against a published worked example before
citing results; record the version.

## Recording versions

Put library names and versions in the paper's methods section or supplement, e.g.
`python -c "import numpy, scipy, pymcdm; print(numpy.__version__, scipy.__version__)"`.
