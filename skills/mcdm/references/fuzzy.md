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

## Not implemented here (use pyDecision)

Fuzzy VIKOR, fuzzy DEMATEL, intuitionistic/spherical/Pythagorean fuzzy variants,
grey relational analysis. Check pyDecision's implementation against a published
worked example before relying on it.
