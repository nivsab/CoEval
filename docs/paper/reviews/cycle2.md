# Review Cycle 2 — TMLR-style (2026-06-01)

**Recommendation: Major Revision** (leaning Reject if the keystone artifact issue is not resolved). Presentation/figures/literature strongly improved; R5 fully addressed. But the reviewer found a **new keystone integrity concern** + several still-open items.

## What the reviewer found (and how it was resolved)

- **W1 (blocking) — keystone "perfect agreement" looked like a data artifact.** ROOT-CAUSED: it is GENUINE, not replication. Proof committed to `Runs/EXP001b-exactmatch-qa/reports/frontier_panel_accuracy.json`: the 3 frontier judges (gpt-4o, claude-sonnet-4, gemini-flash) agree on every item, but the *weaker* auxiliary judges DISAGREE (gpt-3.5: 3, claude-haiku: 5–8) — if it were label replication, all would be identical. Distinct API timestamps confirm independent calls. → Reframed honestly in Table 1 + 5.1.
- **W1/R1 — panel mismatch (artifact was 6-judge, paper claimed 3).** FIXED: committed a 3-frontier-judge artifact; Table 1 now matches. ρ=0.859, **datapoint-clustered CI [0.77, 0.94]**.
- **W2 — ensemble adds nothing on objective tasks.** ADOPTED honestly+positively: objective QA is reframed as a *vendor-independent reliability floor* (ensemble = best member because correctness is objective); the ensemble's *distinctive* value is bias-robustness (5.3). Headline is now accuracy + reliability + bias-cancellation, not point-correlation superiority.
- **R2/W3 — summarization cherry-pick.** FIXED: best single judge (0.354 > ensemble 0.241) and the near-degenerate code subtask (ρ≈0) now stated; "more faithfully" removed.
- **W4 — aspect selection post-hoc.** FIXED: "accuracy" pre-declared a priori as construct-matched; relevance (0.28) and full-rubric (0.57) reported in 5.1.
- **R6/W6 — stats.** FIXED: added `cluster_correlation_ci` (datapoint-clustered bootstrap) + test; Table 1/abstract use it. Trivial Spearman→1.0 demoted to an expected stability check; 5.2 now leads with the non-monotone ICC.
- **R4/W5 — self-preference "+0.090 by construction".** DEMOTED to a design property + indicative-only estimate (noting the family/capability confound), with a controlled ≥3-family measurement as future work. All "by construction"/"two bias guarantees" overclaims scrubbed.
- **R7/W7 — contamination.** Added the caveat that the QA anchor uses public items (contamination-free applies to *generated* items).
- **R5 — non-monotone ICC.** Already addressed (F2); 5.2 prose now matches the figure.
- **W8 — F4 caption.** FIXED to match the data (frontier judges converge; vendor-independent reliability floor).

## Still OPEN (cycle-3 candidates, need new experiments)
- **R9** G-Eval external baseline (~$1–2). 
- **R7** empirical contamination check (n-gram/GSM1k-style) on *generated* items.
- **R4** controlled self-preference experiment (≥3 families, capability held fixed) on a subjective task.
- **R8** quantitative stratified-coverage result.

## Verified
HTML well-formed; 101 KaTeX spans render, 0 runaway; 26 refs all cited+resolve; 37 analyzer/benchmark tests pass; clustered-CI artifact committed.
