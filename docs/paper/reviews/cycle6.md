# Review Cycle 6 — post-v2-integration acceptance check (TMLR/COLM)

**Verdict: ACCEPT.**

This cycle integrated the outstanding v2 ensemble win and, in doing so, caught and
fixed a stale-data inconsistency that a careful reviewer would have flagged.

## New content verified
- **§5.2 judge-choice regret + aggregation bake-off (new):** on the benchmark-grounded
  set (3 tasks, n=900) the candidate judges span -0.04 (gpt-3.5-turbo, anti-correlated)
  to 0.31 (claude-haiku), regret 0.35; the cross-family ensemble is the low-regret default
  (plain mean 0.238, never anti-correlated). Label-free aggregation bake-off:
  reliability-weighted 0.246 > plain mean 0.238 > median/trimmed 0.222 > Dawid-Skene 0.178.
  Every number traces to `Runs/EXP001-*/reports/v2_aggregation_bakeoff.json`. Surfaced in
  the abstract with abstract-body parity intact.

## Stale-data reconciliation (Research-Honesty bug-hunt, blocking-if-missed -> fixed)
- The committed `benchmark_correlation.json` was stale (claude-haiku at n=789). Live EES is
  complete and clean (4 judges x 900 responses, 300 distinct per judge x task, no dup units).
  Regenerated all EXP001 correlation artifacts. True current numbers: ensemble 0.2376,
  best-single claude-haiku 0.3084 (3-task pooled); ensemble 0.227 vs G-Eval 0.259
  (summarization, n=600/580). §5.1 now uses the summarization basis for the G-Eval comparison
  and forward-refs §5.2 for the full regret; §3.3 documents the optional reliability-weighted
  aggregator.
- **Table 1 fix:** the benchmark-grounded row had the wrong judge panel (qwen/smollm) and
  size (580). Corrected to gpt-4o-mini + gpt-3.5-turbo + claude-haiku + gemini-flash,
  3 tasks / 900 resp, verified against `config.yaml`.

## Audits (all clean)
- Self-consistency: 0 phantom sections; all Section refs resolve; abstract-body parity holds;
  N coherent (QA n=573; summarization n=600, G-Eval n=580; pooled n=900); table-text agree.
- Tone: 0 hits (honestly/unfortunately/admittedly/merely/two-thirds/frankly/candidly).
- Style: 0 em-dashes; 340 `$` balanced; docx rebuilt clean (0 stray `$`, 0 em-dash, 162 OMML).
- Citations: bibtest 26/26 valid (Crossref/OpenAlex).
- Tests: 28 analyzer experiment tests pass; correlation artifacts regenerate deterministically.

## Non-blocking residuals (acceptable boundary conditions, unchanged)
- Contamination is an n-gram proxy (scoped) plus the controlled memorizer ranking-flip.
- Ranking demo is 3 models / 1 task family (scoped).
- Self-preference §5.4 is structural-by-design with a small, sign-inconsistent residual.
- Reliability-weighting is the strongest label-free aggregator here; the 2026 BT-σ/CARE
  aggregators are flagged in the scout registry for a future head-to-head (Exp B-extended).
