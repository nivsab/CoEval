# EXP-V2 — the panel is a self-validating instrument (WIN)

**Status:** completed · **Date:** 2026-06-02 · **Cost:** $0 (analysis on EXP001 logged data)
**Script:** `scripts/v2_ensemble_diagnosis.py` · **Artifact:** `Runs/EXP001-.../reports/v2_ensemble_diagnosis.json`

## Hypothesis
A multi-judge panel matters for two reasons a single judge cannot provide, both
label-free: (1) it suppresses outlier/broken judges via aggregation; (2) it
DIAGNOSES each judge's reliability from peer-consistency (agreement with the rest).

## Method
EXP001 benchmark-grounded set (4 real judges, gold = benchmark-native score, n=900,
3 tasks pooled). Inject 3 controlled BROKEN judges: uniform-random scores, a constant
score, and one anti-correlated with quality. Measure per-judge accuracy-vs-gold,
leave-one-out panel agreement, and aggregation accuracy with/without the bad judges.

## Results (WIN)
- **Outlier robustness + automatic zero-weighting (decisive):** adding 3 broken judges
  drops the plain-mean accuracy from **0.238 -> 0.126** (nearly halved); reliability
  weighting (peer-agreement only, no labels) gives each broken judge **weight 0.000** and
  recovers **0.228**, the clean-panel accuracy. The panel detects and neutralizes broken
  judges with no ground truth.
- **Peer-agreement separates competent from broken judges:** all 3 broken judges have the
  lowest panel agreement (<= 0) vs 0.07-0.18 for the competent judges. Agreement is a
  label-free reliability filter.
- **Self-diagnosis correlation (supporting, underpowered):** across 7 judges, peer-agreement
  vs accuracy-vs-gold Spearman = 0.60; clean broken/competent separation but fine-grained
  ordering among competent judges is noisy at only 4 real judges -> tightens with the scaled
  12-20 judge panel.

## Why it strengthens the paper
A single judge offers no safeguard: there is nothing to check it against. The panel turns
the absence of labels from a liability into a design that validates itself. This is the
deeper "why several judges" answer beyond point-accuracy: robustness + label-free reliability
estimation. Integrated into paper Sec 5.2.

## Next (additional experiments)
- Tolerance curve: vary number/severity of injected bad judges -> breakdown point of
  reliability weighting (how many bad of N it survives).
- Scaled panel (12-20 judges) -> tighten the agreement-vs-accuracy correlation (claim 1).
- Cite the peer-consistency lineage (Dawid-Skene latent reliability; PiCO peer review).
