# MCDM methods: formulas, choice and citations

Notation: alternatives i = 1..m, criteria j = 1..n, raw values x_ij, weights w_j
(sum to 1), benefit set B, cost set C. Functions live in `scripts/mcdm_core.py` unless
prefixed: `mn.` = `mcdm_network.py`, `eo.` = `mcdm_outranking.py`, `mf.` = `mcdm_fuzzy.py`.

## Choosing methods

| Situation | Suggested method(s) |
|---|---|
| Experts can compare criteria pairwise, n <= ~9 | AHP (eigenvector), check CR |
| Many criteria or expert time is scarce | BWM (2n - 3 comparisons) |
| Vague linguistic judgments | Fuzzy AHP (Buckley), fuzzy TOPSIS (Chen) |
| Only an importance order of criteria | ROC weights |
| No experts, weights from data dispersion | Entropy, CRITIC (also captures correlation) |
| Want the alternative closest to ideal | TOPSIS |
| Need a compromise acceptable to the majority with a regret limit | VIKOR (v = 0.5 balances group utility and individual regret) |
| Criteria in different units, partial compensation, thresholds matter | PROMETHEE II; PROMETHEE I if incomparability is informative |
| Selection (choose a subset), strong non-compensation | ELECTRE I (kernel) |
| Ranking with crisp thresholds and two embedded relations | ELECTRE II |
| Ranking with imprecise data, indifference/preference/veto thresholds | ELECTRE III |
| Ranking when criterion weights cannot be agreed | ELECTRE IV (no weights) |
| Assign alternatives to predefined ordered classes | ELECTRE Tri-B (report pessimistic and optimistic) |
| Few data points, incomplete information, "grey" data | GRA |
| Vague ratings and compromise wanted | Fuzzy VIKOR |
| Simple, transparent baseline | WSM; WASPAS combines WSM and WPM |
| Criteria influence one another | DEMATEL / fuzzy DEMATEL for cause–effect; DANP or ANP for weights |
| Elements in clusters with feedback (alternatives influence criteria, etc.) | ANP |

Use at least two families and compare rankings (Spearman r_s, weighted r_w, WS).

## Weighting

**AHP** (`ahp_weights`). Reciprocal matrix A with a_ij in {1/9..9}. Weights = principal
right eigenvector (default) or normalised row geometric mean.
λ_max from A w = λ w; CI = (λ_max − n)/(n − 1); CR = CI/RI. Saaty RI (n = 3..10):
0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49. Accept CR < 0.10. The error matrix
ε_ij = a_ij w_j / w_i (= 1 when consistent) points to the judgments to revisit.
Group: AIJ = element-wise geometric mean of matrices (keeps reciprocity); AIP =
arithmetic or geometric mean of individual priority vectors (Forman & Peniwati, 1998).

**BWM** (`bwm_weights`), linear model (Rezaei, 2016):
min ξ s.t. |w_B − a_Bj w_j| <= ξ, |w_j − a_jW w_W| <= ξ, Σ w_j = 1, w_j >= 0.
ξ = 0 iff fully consistent. Input-based consistency ratio (Liang, Brunelli & Rezaei, 2020):
CR_I = max_j |a_Bj a_jW − a_BW| / (a_BW² − a_BW) for a_BW > 1; compare with the
thresholds tabulated in that paper for your n and a_BW.

**Entropy** (`entropy_weights`): p_ij = x_ij / Σ_i x_ij; e_j = −(1/ln m) Σ_i p_ij ln p_ij
with 0·ln 0 = 0; w_j ∝ 1 − e_j.

**CRITIC** (`critic_weights`, Diakoulaki et al., 1995): min-max normalise each criterion
by its own direction; C_j = σ_j Σ_k (1 − r_jk); w_j ∝ C_j.

**ROC** (`roc_weights`, Barron & Barrett, 1996): w_k = (1/n) Σ_{i=k}^{n} 1/i for the
criterion ranked k-th.

## Ranking

**WSM / WPM / WASPAS** (`wsm`, `wpm`, `waspas`). Linear normalisation r = x/max (benefit),
min/x (cost). WSM = Σ w r; WPM = Π r^w; WASPAS Q = λ·WSM + (1 − λ)·WPM (Zavadskas et al., 2012).

**TOPSIS** (`topsis`, Hwang & Yoon, 1981). r_ij = x_ij / sqrt(Σ_i x_ij²); v = w r;
A⁺_j = max v (benefit) / min v (cost), A⁻ the opposite; d± Euclidean;
CC_i = d⁻/(d⁺ + d⁻), higher is better. Known to suffer rank reversal when alternatives
are added or removed; run `rank_reversal_check`.

**VIKOR** (`vikor`, Opricovic & Tzeng, 2004). f*_j best, f⁻_j worst value of criterion j;
S_i = Σ w_j (f*_j − x_ij)/(f*_j − f⁻_j); R_i = max_j of the same terms;
Q_i = v (S_i − S*)/(S⁻ − S*) + (1 − v)(R_i − R*)/(R⁻ − R*). Lower Q is better.
Compromise: with a1, a2 the first two by Q and DQ = 1/(m − 1),
C1 acceptable advantage Q(a2) − Q(a1) >= DQ; C2 acceptable stability a1 also best by S or R.
Both hold → {a1}; only C1 → {a1, a2}; C1 fails → all a_k with Q(a_k) − Q(a1) < DQ.

**PROMETHEE** (`promethee`, Brans & Vincke, 1985). For each pair d_j = g_j(a) − g_j(b)
(sign flipped for cost); preference P_j(d) from one of six functions
(`preference_degree`): usual, U-shape (q), V-shape (p), level (q, p), linear (q, p),
Gaussian (s). π(a, b) = Σ w_j P_j; φ⁺ = Σ_b π(a, b)/(m − 1), φ⁻ = Σ_b π(b, a)/(m − 1),
φ = φ⁺ − φ⁻ (PROMETHEE II). PROMETHEE I: a outranks b if φ⁺(a) >= φ⁺(b) and
φ⁻(a) <= φ⁻(b) with one strict; otherwise incomparable unless both equal.
Choose q, p, s in the criterion's own units and justify them (e.g. measurement error for q).

**ELECTRE I** (`electre_i`, Roy, 1968). Concordance C(a, b) = Σ w_j over criteria where a is
at least as good as b. Discordance D(a, b) = max over criteria where b is better of the
difference divided by the criterion's range. a S b iff C >= ĉ and D <= d̂. The kernel is
computed when the outranking graph is acyclic; with cycles, revise thresholds or reduce cycles.

**DEMATEL** (`dematel`, Gabus & Fontela, 1972). Average the expert direct-relation matrices
(0–4 scale typical) into Z; N = Z / max(max row sum, max column sum); T = N (I − N)⁻¹;
D = row sums, R = column sums; prominence D + R, relation D − R (> 0 = net cause).
Threshold for the influence map: mean of T unless justified otherwise.

## Interdependence: ANP and DANP

**ANP** (`mn.supermatrix_from_comparisons`, `mn.weighted_supermatrix`, `mn.anp`; Saaty, 1996).
Column j of the unweighted supermatrix holds, for every cluster, the local priorities of
that cluster's elements with respect to element j (from a pairwise matrix or given
directly). Weighted supermatrix: multiply block (r, c) by the cluster weight of cluster r
for cluster c, then renormalise each column. Limit matrix: the Cesàro limit
L = lim (1/N) Σ W^k, computed exactly as the spectral projector of eigenvalue 1
(`mn.limit_supermatrix`), so cyclic networks need no special handling. Irreducible
networks give identical columns; read priorities from any column and renormalise within a
cluster. A hierarchy written as a network (alternatives as sinks with self-loops,
`sinks="self_loop"`) reproduces AHP synthesis in the goal column (`reference=` goal index).

**DANP** (`mn.danp`; Ou Yang, Shieh, Leu & Tzeng, 2008). From the DEMATEL total-relation
matrix T_C of criteria grouped into dimensions:
1. T_C^α: normalise each row within each dimension block.
2. Unweighted supermatrix W_C = (T_C^α)ᵀ.
3. T_D: mean of each dimension block of T_C; T_D^α: row-normalised T_D.
4. Weighted supermatrix: block (i, j) of W_C multiplied by t^α_D(j, i) (influence of
   dimension j on dimension i), which makes every column sum to 1.
5. Limit matrix → influential weights of criteria; dimension weights are within-dimension sums.
DANP weights are often combined with VIKOR ("DANP-mV"); `example_danp_spec.json` shows this.

## Outranking: ELECTRE II, III, IV, Tri-B

**ELECTRE II** (`eo.electre_ii`; Roy & Bertier, 1973). Concordance C as in ELECTRE I,
discordance D divided by the criterion range (`discordance="global_range"` divides by the
largest range, as some textbooks and pyDecision do). With C(a,b) >= C(b,a):
strong a S^F b iff (C >= c⁺ and D <= d⁺) or (C >= c⁰ and D <= d⁻);
weak a S^f b iff (C >= c⁰ and D <= d⁺) or (C >= c⁻ and D <= d⁻).
Descending pre-order: repeatedly take alternatives not strongly outranked (cycles condensed),
refined by the weak relation; ascending: the same on reversed relations from the bottom.
Defaults c⁻, c⁰, c⁺ = 0.65, 0.75, 0.85 and d⁻, d⁺ = 0.25, 0.50 are conventions, not rules.

**ELECTRE III** (`eo.electre_iii`; Roy, 1978). Per criterion, with Δ = g_j(b) − g_j(a):
c_j = 1 if Δ <= q_j, 0 if Δ >= p_j, else (p_j − Δ)/(p_j − q_j);
d_j = 0 if Δ <= p_j, 1 if Δ >= v_j, else (Δ − p_j)/(v_j − p_j).
C = Σ w_j c_j; credibility σ = C · Π_{j: d_j > C} (1 − d_j)/(1 − C).
Distillation (`eo.distillation`): λ₀ = max σ; λ₁ = largest σ below λ₀ − s(λ₀), with
s(λ) = α + βλ (α = 0.3, β = −0.15); a outranks b at λ₁ iff σ(a,b) > λ₁ and
σ(a,b) − σ(b,a) > s(σ(a,b)); qualification = strength − weakness; the descending
distillation keeps the maximum, the ascending the minimum, re-distilling ties at lower λ.
Final partial pre-order = intersection of both (`relation`: P+, P−, I, R).

**ELECTRE IV** (`eo.electre_iv`; Roy & Hugonnard, 1982). No weights. With m_p, m_q, m_i the
numbers of criteria where one alternative beats the other strictly, weakly or within
indifference, five nested relations (quasi, canonical, pseudo, sub, veto dominance; see the
docstring) receive credibility 1.0, 0.8, 0.6, 0.4, 0.2 and are ranked by the ELECTRE III
distillation. The tests check that the relations are nested.

**ELECTRE Tri-B** (`eo.electre_tri_b`; Yu, 1992; Mousseau, Słowiński & Zielniewicz, 2000).
Boundary profiles b_1 < … < b_{k−1} separate categories C_1 (worst) … C_k (best);
σ(a, b_h) and σ(b_h, a) use the ELECTRE III formulas; a S b_h iff σ(a, b_h) >= λ.
Pessimistic: from the top profile down, the first b_h with a S b_h puts a in C_{h+1}.
Optimistic: from the bottom up, the first b_h with b_h ≻ a (b_h S a and not a S b_h)
puts a in C_h. Pessimistic class <= optimistic class always; report both.

## Grey relational analysis

**GRA** (`gra`; Deng, 1982; Kuo, Yang & Huang, 2008). Min-max normalise by direction,
compare with the ideal series 1: Δ_ij = |1 − r_ij|;
γ_ij = (Δ_min + ρΔ_max)/(Δ_ij + ρΔ_max), ρ = 0.5 by default; grade = Σ w_j γ_ij.

## Robustness

- `weight_sensitivity`: vary one weight over [0, 1], rescale the others proportionally,
  record where the best alternative changes.
- `rank_reversal_check`: remove each alternative and count pairwise order flips.
- `weight_acceptability`: draw weights from Dirichlet(κ·w); rank acceptability b_ik is the
  share of draws in which alternative i takes rank k (SMAA idea, Lahdelma et al., 1998).
  Small κ = little confidence in the weights.
- `compare_rankings`: Spearman r_s, Kendall τ_b, weighted Spearman r_w
  (Pinto da Costa & Soares, 2005), WS (Sałabun & Urbaniak, 2020; asymmetric, reference first).

## Citations

- Barron, F. H., & Barrett, B. E. (1996). Decision quality using ranked attribute weights. *Management Science*, 42(11), 1515–1523.
- Brans, J. P., & Vincke, Ph. (1985). A preference ranking organisation method (The PROMETHEE method for multiple criteria decision-making). *Management Science*, 31(6), 647–656.
- Deng, J. L. (1982). Control problems of grey systems. *Systems & Control Letters*, 1(5), 288–294.
- Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995). Determining objective weights in multiple criteria problems: The CRITIC method. *Computers & Operations Research*, 22(7), 763–770.
- Figueira, J., Mousseau, V., & Roy, B. (2005). ELECTRE methods. In J. Figueira, S. Greco, & M. Ehrgott (Eds.), *Multiple Criteria Decision Analysis: State of the Art Surveys* (pp. 133–153). Springer.
- Forman, E., & Peniwati, K. (1998). Aggregating individual judgments and priorities with the analytic hierarchy process. *European Journal of Operational Research*, 108(1), 165–169.
- Gabus, A., & Fontela, E. (1972). *World problems, an invitation to further thought within the framework of DEMATEL*. Battelle Geneva Research Centre.
- Hwang, C. L., & Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications*. Springer.
- Kuo, Y., Yang, T., & Huang, G. W. (2008). The use of grey relational analysis in solving multiple attribute decision-making problems. *Computers & Industrial Engineering*, 55(1), 80–93.
- Lahdelma, R., Hokkanen, J., & Salminen, P. (1998). SMAA – Stochastic multiobjective acceptability analysis. *European Journal of Operational Research*, 106(1), 137–143.
- Liang, F., Brunelli, M., & Rezaei, J. (2020). Consistency issues in the best worst method: Measurements and thresholds. *Omega*, 96, 102175.
- Mousseau, V., Słowiński, R., & Zielniewicz, P. (2000). A user-oriented implementation of the ELECTRE-TRI method integrating preference elicitation support. *Computers & Operations Research*, 27(7–8), 757–777.
- Opricovic, S., & Tzeng, G. H. (2004). Compromise solution by MCDM methods: A comparative analysis of VIKOR and TOPSIS. *European Journal of Operational Research*, 156(2), 445–455.
- Ou Yang, Y. P., Shieh, H. M., Leu, J. D., & Tzeng, G. H. (2008). A novel hybrid MCDM model combined with DEMATEL and ANP with applications. *International Journal of Operations Research*, 5(3), 160–168.
- Pinto da Costa, J., & Soares, C. (2005). A weighted rank measure of correlation. *Australian & New Zealand Journal of Statistics*, 47(4), 515–529.
- Rezaei, J. (2015). Best-worst multi-criteria decision-making method. *Omega*, 53, 49–57.
- Rezaei, J. (2016). Best-worst multi-criteria decision-making method: Some properties and a linear model. *Omega*, 64, 126–130.
- Roy, B. (1968). Classement et choix en présence de points de vue multiples (la méthode ELECTRE). *RIRO*, 2(8), 57–75.
- Roy, B. (1978). ELECTRE III: Un algorithme de classements fondé sur une représentation floue des préférences en présence de critères multiples. *Cahiers du CERO*, 20(1), 3–24.
- Roy, B., & Bertier, P. (1973). La méthode ELECTRE II: Une application au média-planning. In M. Ross (Ed.), *OR'72* (pp. 291–302). North-Holland.
- Roy, B., & Hugonnard, J. C. (1982). Ranking of suburban line extension projects on the Paris metro system by a multicriteria method. *Transportation Research Part A*, 16(4), 301–312.
- Saaty, T. L. (1980). *The Analytic Hierarchy Process*. McGraw-Hill.
- Saaty, T. L. (1996). *Decision Making with Dependence and Feedback: The Analytic Network Process*. RWS Publications.
- Sałabun, W., & Urbaniak, K. (2020). A new coefficient of rankings similarity in decision-making problems. In *Computational Science – ICCS 2020*, LNCS 12138, 632–645. Springer.
- Yu, W. (1992). *ELECTRE TRI: Aspects méthodologiques et manuel d'utilisation*. Document du LAMSADE 74, Université Paris-Dauphine.
- Zavadskas, E. K., Turskis, Z., Antucheviciene, J., & Zakarevicius, A. (2012). Optimization of weighted aggregated sum product assessment. *Elektronika ir Elektrotechnika*, 122(6), 3–6.

Verify every citation against the publisher record before submission.
