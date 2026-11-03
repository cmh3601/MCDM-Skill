# Fuzzy MCDM details

## Triangular fuzzy numbers (TFN)

Ã = (l, m, u) with l <= m <= u. Operations used here (positive TFNs):

- Ã ⊕ B̃ = (l1 + l2, m1 + m2, u1 + u2)
- Ã ⊗ B̃ ≈ (l1 l2, m1 m2, u1 u2)
- Ã⁻¹ ≈ (1/u, 1/m, 1/l)
- Centroid defuzzification: (l + m + u)/3
- Vertex distance: d(Ã, B̃) = sqrt(((l1 − l2)² + (m1 − m2)² + (u1 − u2)²)/3)

## Linguistic scale for pairwise comparisons

`SAATY_TFN` maps 1→(1,1,1), 2→(1,2,3), 3→(2,3,4), …, 8→(7,8,9), 9→(8,9,9);
reciprocals invert as above. Other papers use 9→(9,9,9) or wider spreads. The
choice changes the weights, so state it and, if reviewers may ask, test a second
scale in the sensitivity section. Pass a different dict to `tfn_from_crisp(A, scale=...)`.

## Fuzzy AHP: Buckley (1985), `fuzzy_ahp_buckley`

1. r̃_i = (Π_j ã_ij)^(1/n), computed separately for l, m, u.
2. w̃_i = r̃_i ⊗ (Σ_k r̃_k)⁻¹ = (l_i/Σu, m_i/Σm, u_i/Σl).
3. Crisp weights: centroid of w̃_i, then normalise to sum 1.
4. Consistency: compute CR of the modal (m) matrix; report it with the crisp CR rule.

Why not Chang's (1996) extent analysis: Wang, Luo & Hua (2008, *EJOR* 186(2), 735–747)
showed that its degree-of-possibility step can give zero weight to criteria or
alternatives that are relevant, and that its weights do not represent the relative
importance of criteria. Reviewers at decision-science journals often reject it.

## Fuzzy TOPSIS: Chen (2000), `fuzzy_topsis`

Input: TFN ratings x̃_ij = (a, b, c) (often averaged over experts) and crisp or TFN weights.

1. Normalise: benefit r̃ = (a/c*, b/c*, c/c*) with c* = max_i c_ij;
   cost r̃ = (a⁻/c, a⁻/b, a⁻/a) with a⁻ = min_i a_ij.
2. Weighted: ṽ_ij = r̃_ij ⊗ w̃_j.
3. FPIS = (1, 1, 1), FNIS = (0, 0, 0) for every criterion.
4. d⁺_i = Σ_j d(ṽ_ij, FPIS), d⁻_i = Σ_j d(ṽ_ij, FNIS) with the vertex distance.
5. CC_i = d⁻_i / (d⁺_i + d⁻_i); higher is better.

Chen, C. T. (2000). Extensions of the TOPSIS for group decision-making under fuzzy
environment. *Fuzzy Sets and Systems*, 114(1), 1–9.

## Group aggregation of fuzzy judgments

Common choices: element-wise geometric mean of l, m, u across experts (keeps
reciprocity for pairwise matrices), or (min l, mean m, max u) for ratings, which
widens uncertainty. Report which one you used.

## Fuzzy VIKOR, `mf.fuzzy_vikor`

Fuzzy arithmetic throughout, following the structure of Opricovic (2011):

1. Fuzzy best f̃*_j and worst f̃°_j: component-wise max/min of the ratings (min/max for cost).
2. Normalised fuzzy gap: benefit d̃_ij = (f̃*_j ⊖ x̃_ij)/(u*_j − l°_j); cost
   d̃_ij = (x̃_ij ⊖ f̃*_j)/(u°_j − l*_j), with ã ⊖ b̃ = (l_a − u_b, m_a − m_b, u_a − l_b).
3. S̃_i = Σ_j w̃_j ⊗ d̃_ij; R̃_i = component-wise max_j of w̃_j ⊗ d̃_ij.
4. Q̃_i = v (S̃_i ⊖ S̃*)/(S°_u − S*_l) ⊕ (1 − v)(R̃_i ⊖ R̃*)/(R°_u − R*_l).
5. Defuzzify S, R, Q (centroid by default, `defuzz="graded_mean"` or `"cfcs"` available),
   rank by Q (lower is better) and check C1/C2 as in crisp VIKOR.

With l = m = u the method returns exactly the crisp VIKOR result (tested). The product
uses bound-wise min/max so that negative lower bounds of d̃ stay valid.

Opricovic, S. (2011). Fuzzy VIKOR with an application to water resources planning.
*Expert Systems with Applications*, 38(10), 12983–12990.

## Fuzzy DEMATEL, `mf.fuzzy_dematel`

In the manner of Lin & Wu (2008): average the TFN direct-relation matrices, normalise
the l, m, u matrices by the same constant (max row/column sum of u), compute
T_k = N_k (I − N_k)⁻¹ for each component, defuzzify the fuzzy total relation and then
compute D, R, prominence and relation. Reduces exactly to crisp DEMATEL (tested).

Lin, C. J., & Wu, W. W. (2008). A causal analytical method for group decision-making under
fuzzy environment. *Expert Systems with Applications*, 34(1), 205–213.

## CFCS defuzzification, `mf.cfcs`

Converting Fuzzy data into Crisp Scores (Opricovic & Tzeng, 2003) over a set of TFNs, using
the global min l and max u as the range: normalise, compute left/right scores
ls = x_m/(1 + x_m − x_l), us = x_u/(1 + x_u − x_m), total x = [ls(1 − ls) + us²]/(1 − ls + us),
and map back. Use the same set (e.g. one expert's whole matrix) for all values you compare.

Opricovic, S., & Tzeng, G. H. (2003). Defuzzification within a multicriteria decision model.
*International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems*, 11(5), 635–652.

## Not implemented here (use pyDecision or write it)

Intuitionistic, spherical, Pythagorean and q-rung fuzzy variants; fuzzy BWM; fuzzy
PROMETHEE; ELECTRE Tri-C / Tri-nB / Tri-nC. Check any implementation against a published
worked example before relying on it.
