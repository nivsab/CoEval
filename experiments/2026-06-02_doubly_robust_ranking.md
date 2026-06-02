# EXP-V2 — doubly-robust ranking (weight items x judges, label-free) (WIN)

**Status:** completed · **Date:** 2026-06-02 · **Cost:** $0 (analysis on EXP010 logged data)
**Script:** `scripts/v2_doubly_robust_ranking.py` · **Artifact:** `Runs/EXP010-.../reports/v2_doubly_robust_ranking.json`

## Hypothesis (user-proposed)
A robust model ranking should weight by (a) teacher discriminative power of each item
(d_i = variance of candidate-model scores on item i) and (b) judge response similarity
(w_j = mean leave-one-out peer agreement). Both label-free. The doubly-robust ranking
leans on items and judges that carry signal -> better, more robust rank-recovery.

## Method
EXP010 (7 students, sciq+ARC, gold = benchmark-native score). Per (item,student,judge)
ensemble score. Rank-recovery (Spearman/Kendall) of the student ranking vs gold, for
plain mean / item-weighted / judge-weighted / doubly-robust. Inject a random judge for
the robustness test.

## Results (WIN)
- plain mean: Spearman 0.873 / Kendall 0.751
- item-discrimination-weighted only: 0.946 / 0.851
- judge-agreement-weighted only: 0.800 / 0.651 (hurts alone at n=3 judges)
- **doubly-robust (both): 0.982 / 0.951** (near-perfect; +0.11 Spearman, +0.20 Kendall vs plain)
- Robust: with a random judge injected, doubly-robust stays 0.982 (random judge weight 0.06).
- Mechanism (sanity): item weights concentrate on discriminative items (top 10% items = 56%
  of weight; 65% of saturated items ~0 weight). Validation vs INDEPENDENT gold (not overfit).
- Holds on both Spearman AND Kendall.

## Honest scope
n=7 students, 80 items, 3 judges (pilot). The IMPROVEMENT is consistent and mechanistically
clear, but the absolute 0.98 is coarse at n=7; the scaled 12-20 model run confirms at higher
resolution. Judge-weighting helps only IN COMBINATION with item weighting (the two are
complementary: on discriminative items, judge reliability matters most).

## Integrated
Defined the doubly-robust aggregator in Sec 3.3 (method) with the scoped empirical note.
Unifies the two quality axes: teacher discrimination (Sec 5.7) x judge consistency (Sec 5.2).
