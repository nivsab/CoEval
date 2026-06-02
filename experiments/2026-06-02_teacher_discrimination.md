# EXP-V2 — a good teacher generates DISCRIMINATIVE items (WIN)

**Status:** completed · **Date:** 2026-06-02 · **Cost:** $0 (analysis on EXP006/007 logged data)
**Script:** `scripts/v2_item_discrimination.py` · **Artifact:** `Runs/EXP006-.../reports/v2_item_discrimination.json`

## Hypothesis
A benchmark can only rank models if its items SEPARATE them. A good teacher writes
items on which candidates produce different results (the generation-side analog of
judge discrimination). Measured on the items CoEval actually GENERATED for 3 verticals.

## Method
Per generated item, spread (range) of the 3 candidate students' ensemble scores.
Discriminative = range >= 0.15. Decisive = item's top student == overall-ranking top.

## Results (WIN)
- Pooled 120 generated vertical items: **71% discriminative**, mean per-item range **0.33**.
- Per vertical tracks the TRUE model separation: drug-interaction 78%, legal 85%,
  clinical 50% (the latter because its two stronger models are genuinely close,
  0.873 vs 0.864 -> honest near-tie, not manufactured). Separability (top-bottom):
  DDI 0.273, legal 0.273, clinical 0.059.
- This is WHY the panel yields non-overlapping ranking CIs: the teacher supplies items
  carrying ranking signal; a saturated/non-discriminative benchmark cannot rank regardless
  of judge quality (cf. EXP010 sciq saturation).

## Integrated
Paper Sec 5.7 (discriminative-generation finding) + contribution (1). Symmetric theme:
good teacher = discriminative items; good judge = consistent + discriminative scores.
