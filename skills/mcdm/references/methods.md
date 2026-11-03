# MCDM methods: formulas, choice and citations

Notation: alternatives i = 1..m, criteria j = 1..n, raw values x_ij, weights w_j
(sum to 1), benefit set B, cost set C. Function names refer to `scripts/mcdm_core.py`.

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
| Selection (choose a subset), strong non-compensation, veto | ELECTRE I (kernel); ELECTRE III via pyDecision for ranking with veto |
| Simple, transparent baseline | WSM; WASPAS combines WSM and WPM |
| Criteria influence one another | DEMATEL to map cause/effect; then treat interdependence explicitly |

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
- Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995). Determining objective weights in multiple criteria problems: The CRITIC method. *Computers & Operations Research*, 22(7), 763–770.
- Forman, E., & Peniwati, K. (1998). Aggregating individual judgments and priorities with the analytic hierarchy process. *European Journal of Operational Research*, 108(1), 165–169.
- Gabus, A., & Fontela, E. (1972). *World problems, an invitation to further thought within the framework of DEMATEL*. Battelle Geneva Research Centre.
- Hwang, C. L., & Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications*. Springer.
- Lahdelma, R., Hokkanen, J., & Salminen, P. (1998). SMAA – Stochastic multiobjective acceptability analysis. *European Journal of Operational Research*, 106(1), 137–143.
- Liang, F., Brunelli, M., & Rezaei, J. (2020). Consistency issues in the best worst method: Measurements and thresholds. *Omega*, 96, 102175.
- Opricovic, S., & Tzeng, G. H. (2004). Compromise solution by MCDM methods: A comparative analysis of VIKOR and TOPSIS. *European Journal of Operational Research*, 156(2), 445–455.
- Pinto da Costa, J., & Soares, C. (2005). A weighted rank measure of correlation. *Australian & New Zealand Journal of Statistics*, 47(4), 515–529.
- Rezaei, J. (2015). Best-worst multi-criteria decision-making method. *Omega*, 53, 49–57.
- Rezaei, J. (2016). Best-worst multi-criteria decision-making method: Some properties and a linear model. *Omega*, 64, 126–130.
- Roy, B. (1968). Classement et choix en présence de points de vue multiples (la méthode ELECTRE). *RIRO*, 2(8), 57–75.
- Saaty, T. L. (1980). *The Analytic Hierarchy Process*. McGraw-Hill.
- Sałabun, W., & Urbaniak, K. (2020). A new coefficient of rankings similarity in decision-making problems. In *Computational Science – ICCS 2020*, LNCS 12138, 632–645. Springer.
- Zavadskas, E. K., Turskis, Z., Antucheviciene, J., & Zakarevicius, A. (2012). Optimization of weighted aggregated sum product assessment. *Elektronika ir Elektrotechnika*, 122(6), 3–6.

Verify every citation against the publisher record before submission.
