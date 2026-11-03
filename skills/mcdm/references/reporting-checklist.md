# MCDM reporting checklist

Use this before submitting to an OR / soft computing / decision science journal
or when reviewing such a paper.

## Problem and data

- [ ] Goal, alternatives and criteria defined; each criterion has a direction, unit and source.
- [ ] Criteria checked for redundancy (correlation or expert argument); interdependence handled (DEMATEL/ANP) or argued away.
- [ ] Expert panel described: number, expertise, selection, how judgments were collected; aggregation rule (AIJ/AIP, fuzzy aggregation) stated.
- [ ] Missing data and scale conversions (e.g. linguistic to TFN) documented.

## Method choice

- [ ] Why these weighting and ranking methods for this decision (not "popular in the literature").
- [ ] If fuzzy: why fuzziness is needed, which TFN scale, which defuzzification.
- [ ] Normalisation method stated for every ranking method.
- [ ] Method parameters stated: VIKOR v, WASPAS λ, PROMETHEE functions with q/p/s in criterion units, ELECTRE thresholds.

## Validity

- [ ] AHP CR (per matrix, per expert) or BWM ξ / CR_I reported; inconsistent judgments revised, with how.
- [ ] Weight sensitivity: where does the best alternative change? Is the decision stable over a plausible weight range?
- [ ] Comparison with at least one method from another family, with r_s / r_w / WS.
- [ ] Rank reversal tested if alternatives may be added or removed later.
- [ ] Uncertainty in weights or data propagated (rank acceptability, Monte Carlo) when inputs are uncertain.

## Reproducibility

- [ ] Full decision matrix, pairwise matrices and final weights in the paper or supplement.
- [ ] Software and versions; random seeds for stochastic analyses.
- [ ] Intermediate tables a reader can recompute (normalised matrix, d⁺/d⁻ or S/R/Q, flows).

## Interpretation

- [ ] Managerial implications follow from the robustness results, not only the point ranking.
- [ ] Limitations: subjectivity of weights, sample of experts, method assumptions (compensation, independence).
