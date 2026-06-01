# Review Cycle 4 — TMLR-style (2026-06-01)

**Recommendation: Accept with minor revisions.** Reviewer verified every load-bearing number against committed artifacts (rho=0.859 + clustered CI + disagreement counts; exact ground-truth ranking; G-Eval rho=0.259; summarization 0.244 / best-single 0.354; 0.0000 contamination overlap). All cycle-3 items confirmed resolved.

## Two blocking minor fixes (named by reviewer; both done in cycle 5)
- **W-A:** scope the §5.6 contamination conclusion away from "cannot have memorized / not present in pretraining data" to verbatim non-duplication of the sampled public sets, corroborating (not proving) the structural freshness guarantee. DONE.
- **W-B:** commit the contamination-check script emitting the 110,784 distinct-13-gram count (or drop the figure). DONE — `scripts/contamination_check.py`.

## Cosmetic
- **W-E:** reconcile stale `coeval_ensemble_vs_bertscore=0.241` → 0.244 in geval_baseline.json. DONE.

## Reviewer note
"With those done, the manuscript meets both TMLR criteria without reservation. I would not require any new experiments." Non-blocking residuals (acceptable): contamination is an n-gram proxy (now scoped); ranking demo is 3 models/1 task family (now scoped); self-preference §5.4 is structural-by-design (indicative estimate, controlled study deferred).
