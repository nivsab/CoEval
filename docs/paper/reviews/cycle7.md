# Review Cycle 7 — related-work positioning + appendix streamlining (TMLR/COLM)

**Verdict: ACCEPT.**

## Changes this cycle
- **Positioning (§2.1):** new paragraph against the 2026 label-free judge-aggregation
  competitors BT-σ [27], CARE [28], Xu et al. [29]. Contribution #3 reframed from
  "first label-free aggregator" to the INTEGRATION (de-novo contamination-free items +
  vendor-disjoint panel + label-free aggregation); the loop ownership is what surfaces
  judge-choice regret and the contamination ranking-flip.
- **Positioning (§2.2):** CHASE [30], DataMorgana [31] (closest de-novo generators; both
  stop at generation, CoEval closes the loop to a ranking) + static-to-dynamic
  contamination survey [32].
- **Streamlining:** §3.4 estimator formulas (ICC, Spearman-Brown, Cohen's kappa, bootstrap,
  Benjamini-Hochberg) moved to new Appendix A; §3.4 left as a one-sentence pointer.

## Audits (clean)
- Citations: 32/32 bibtest-valid; no dangling/unused refs; Appendix A anchor resolves
  from §3.4 and §5.2.
- Self-consistency: subsections 2.1-5.8 intact (no numbering gap); all 6 figures present;
  no lingering "Section 3.4 formula" references; abstract-body parity holds.
- Tone/style: 0 em-dashes, 0 tone hits, $ balanced (346). docx rebuilt clean (165 OMML,
  Appendix A present).

## Recommended (NOT yet done; need approval — figure/subsection renumbering)
- Move §5.5 (rubric generalization) + Figure 5 heatmap to Appendix B, renumber cost
  Figure 6 -> 5. Marginal streamlining; deferred to avoid unilateral renumber.
- Optionally relocate the §5.7 cost figure detail to an appendix, keeping the headline
  USD 5.89 number in main.

## Open strengthening (planned, not blocking acceptance)
- Scaled rank-recovery over 12-20 candidate models (retires the 3-model residual; answers
  the ">10 models" question): `experiments/2026-06-02_scale_ranking_plan.md`. ~$15-20.
- Self-preference-is-justified citation to pre-empt the "your fix removes signal" objection.
