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

## pyDecision

`pip install pyDecision` (GPL-3.0; very large method list including ELECTRE II–IV,
ELECTRE TRI, fuzzy VIKOR, fuzzy DEMATEL, many fuzzy-set variants). Licence matters if
you redistribute code that imports it. Validate each function you use against a
published worked example before citing results; record the version.

## Recording versions

Put library names and versions in the paper's methods section or supplement, e.g.
`python -c "import numpy, scipy, pymcdm; print(numpy.__version__, scipy.__version__)"`.
