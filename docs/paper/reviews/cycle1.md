# Review Cycle 1 — TMLR-style (2026-06-01)

**Recommendation: Major Revision.** Framework + reproducibility infrastructure are strong (exemplary experiment registry, validated stats). But headline empirical claims did not match committed artifacts, and several "by construction" claims are asserted, not measured. Path to acceptance is concrete.

## Required revisions (gating checklist)

- [~] **R1 (P0)** Table 1 reproducibility: the committed `benchmark_correlation.json` showed the relevance-diluted ensemble (0.565); the paper's 0.859 is the *accuracy-aspect* ensemble, never committed. → Commit the accuracy-aspect artifact; correct task labels (science_qa/science_reasoning); justify that "accuracy" is the a-priori construct-matched dimension (NOT post-hoc, NOT using gold labels at eval time); report both aspects transparently. **+ run frontier panel so the ensemble genuinely ties/beats best single judge.**
- [ ] **R2 (P0)** Summarization (5.1): report best-single-judge (0.354) alongside mean (0.121); report negative code_explanation subtask; soften "more faithfully."
- [ ] **R3 (P0)** Stop calling the bias-experiment panel "frontier cross-family": medium-benchmark judges are gpt-4o-mini + gpt-3.5-turbo (both OpenAI) + qwen2.5-1.5b + smollm2-1.7b. State the exact panel in 5.3–5.5.
- [ ] **R4 (P1)** Self-preference (5.4): run a controlled experiment (≥3 families, frontier judges, in- vs out-of-family, CIs) OR demote from "Finding" to design property with "not yet measured."
- [ ] **R5 (P1)** Add the non-monotone ICC(3,k) curve (0.70@k=2 → 0.40@k=4) as the primary composition>size evidence; the current Spearman→1.0 curve is trivially monotone.
- [ ] **R6 (P1)** Apply Benjamini-Hochberg across the correlation family; fix n=573-vs-191-items clustering with item-level/cluster bootstrap.
- [ ] **R7 (P1)** Add an empirical contamination check OR retitle to not lead with "contamination-free" as demonstrated; add teacher-contamination caveat.
- [ ] **R8 (P2)** Evaluate attribute-stratified generation (coverage/balance/discrimination) or demote to system feature.
- [ ] **R9 (P2)** Run ≥1 real external baseline (G-Eval cheapest).
- [ ] **R10 (P2)** Scrub unsupported sentences: "competitive with specialized human-aligned evaluators", "formal per-stratum guarantees", "two bias guarantees".

## Suggested experiments (≤$20 total)
1. **Strong-only cross-family frontier panel** (gpt-4o + Claude-Sonnet + Gemini-Pro) — ~$3-6. **[RUNNING as EXP-001b frontier re-judge]** Tests thesis: does a strong cross-family ensemble beat best single judge + recover monotone ICC.
2. G-Eval baseline (gpt-4o CoT) — ~$1-2.
3. Contamination check (n-gram overlap + GSM1k-style gap) — ~$2-4.
4. Self-preference done right — ~$1-2.
5. Item-clustered bootstrap — $0.
6. Stratum-coverage / ranking-flip — ~$0.

## Missing figures (reviewer F1–F6)
F1 architecture diagram; F2 ICC(3,k) vs k with Spearman-Brown overlay; F3 per-judge length-bias forest plot; F4 ensemble-vs-correctness scatter; F5 rubric similarity heatmap; F6 cost-vs-coverage.

## Solid as-is (keep)
Verbosity-bias cancellation (5.3); cost/reproducibility; non-monotone ICC finding (under-used); related-work positioning.
