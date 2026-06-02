# CoEval Experiment Registry

One row per experiment. New entries append; superseded runs are marked, not deleted.

| ID | Title | Date | Status | Cost | One-line result |
|----|-------|------|--------|------|-----------------|
| [EXP-002](2026-06-01_exp002_004_005_free_reanalysis.md) | Ensemble-size ablation | 2026-06-01 | completed | $0 | ICC(3,k) is **non-monotone**: peaks at k=2 (0.70) then falls to 0.40 as weak judges dilute the panel; Spearman-to-full rises 0.67→1.0. Motivates judge selection. |
| [EXP-004](2026-06-01_exp002_004_005_free_reanalysis.md) | Verbosity bias | 2026-06-01 | completed | $0 | Strong judges show **negative** length bias (GPT-3.5 r=−0.18), weak judges positive (SmolLM2 +0.23); ensemble r=+0.01 (CI straddles 0). mean\|indiv\|=0.153 → ensemble 0.010 (93% magnitude reduction). |
| [EXP-005](2026-06-01_exp002_004_005_free_reanalysis.md) | Rubric generalization | 2026-06-01 | completed | $0 | 22 criteria / 4 tasks; within-task sim 0.342 > cross-task 0.294; one universal cluster (`completeness`, 3 tasks), 11 task-specific singletons. |
| EXP-001 | Benchmark-grounded comparison + frontier judge + G-Eval | 2026-06-01 | completed | ~$10 | DONE (paper §5.1–5.2). 3-task pooled: ensemble ρ=0.238, best-single claude-haiku 0.308; G-Eval ρ=0.259 (summ, n=580); ranking recovered exactly. Reconciled to live n=900 data 2026-06-02. |
| [EXP-006/007](2026-06-01_exp001_benchmark_grounded.md) | Vertical case studies (DDI/clinical/legal) | 2026-06-01 | completed | ~$3 | Cross-family ensemble ranks 3 students on 3 contamination-free verticals; DDI fully unanimous (paper §5.7, Table 5). |
| EXP-008 | Contamination ranking-flip | 2026-06-02 | completed | ~$2 (GPU) | Static benchmark ranks a 0.5B memorizer above gpt-4o-mini (1.00 vs 0.845); fresh items flip it back (paper §5.5). |
| EXP-V2 | Make CoEval a clear win (judge-choice regret + reliability-weighted agg) | 2026-06-02 | completed | $0 | Regret 0.35; reliability-weighted 0.246 best label-free aggregator (paper §5.2). |
| [EXP-010](_diagnostics/2026-06-02_exp010_pilot_rank_recovery.md) | Many-model rank-recovery | 2026-06-02 | pilot/diagnostic | ~$3 | Pipeline validated; 7 models ρ=0.87 (low-regret) but benchmarks saturate frontier. Clean run needs harder task + 12-20 models + batch. |
| EXP-003 | Positional-bias / flip-rate | — | planned (optional) | ~$1.30 | Pending `positional_bias_analysis.py`. NOT a paper dependency (positional bias is only a motivating mention in §1). |

## Code artifacts produced (2026-06-01)
- `Code/analyzer/stats.py` — bootstrap CIs, ICC(3,1)/ICC(3,k), Benjamini-Hochberg (validated vs Shrout & Fleiss reference).
- `Code/analyzer/ensemble_ablation.py`, `verbosity_bias.py`, `rubric_generalization.py` — EXP-002/004/005 compute + self-contained HTML.
- `Code/analyzer/experiment_reports.py` — shared no-CDN Plotly page helper.
- `Public/benchmark/score_responses.py` — per-response benchmark-native scorer (fixes the reference-vs-itself integrity bug for EXP-001).
- CLI: `coeval analyze ensemble-ablation | verbosity-bias | rubric-overlap`.
- Tests: `Tests/analyzer/test_experiments.py` (23), `Tests/benchmark/test_score_responses.py` (9). Zero regressions (1024 prior tests still pass).

## Known pre-existing issue (not introduced here)
- `Tests/analyzer/test_analyze_reports.py::TestExportBenchmark` (7 tests) fail on a clean tree: robust-filter `theta` default drift (test expects `theta=0.0` behavior, code default is 0.05 producing 0 robust datapoints). Orthogonal to this work.

## Cycle-3 additions (2026-06-01)
- **G-Eval baseline (R9):** single gpt-4o+CoT on 600 summarization responses → rho=0.259 [0.18,0.33] vs BERTScore (CoEval ensemble 0.244; best single 0.354). `Runs/EXP001-.../reports/geval_baseline.json`, `scripts/geval_baseline.py`.
- **Model-ranking demo:** CoEval frontier ensemble reproduces exact ground-truth ranking (gpt-4o-mini 0.963 > gpt-3.5 0.921 > llama-3b 0.843; GT identical). `Runs/EXP001b-.../reports/model_ranking.json`. Paper Table 2.

## EXP-006 / EXP-007 vertical case studies (2026-06-01)
- **EXP-006** clinical_reasoning + legal_analysis: CoEval generates 2 contamination-free verticals from a description (no labels), 3 students, cross-family ensemble. 720 evals, 100% valid. Ensemble ranks llama-3.2-3b weakest in both; unanimous in clinical, 2/3 in legal. `Runs/EXP006-.../reports/{vertical_rankings,self_preference,family_aware}.json`. Paper §5.8 + Table 4. (Self-preference −0.027 = NOT inflation; not claimed.)
- **EXP-007** drug_interaction_reasoning (DDI): scout-selected vertical where a public set (DDIExtraction-2013) exists but is extraction-only + contaminated; reasoning-QA sets are private/tiny → CoEval supplies stratified coverage (severity × mechanism × patient_context). **Completed:** fully unanimous cross-family ranking gpt-4o-mini 0.770 > gpt-3.5-turbo 0.682 > llama-3.2-3b 0.497 (360/360 valid, non-overlapping CIs). Paper §5.8 + Table 4. `Runs/EXP007-ddi-vertical/reports/ddi_ranking.json`.
- **Family-aware (Q3):** `Code/analyzer/family_aware.py` — vendor-disjoint aggregation (exclude same-family judges). On EXP-006/EXP-001b rankings UNCHANGED, scores shift ≤0.015 → ensemble already robust to same-family bias. Paper §5.4.
- **Note (RESOLVED 2026-06-02):** the earlier "runner Phase 3 exits 0 items via openrouter" was stale. The real runner Phase 3 generates correctly via openrouter (gpt-4o-mini verified end-to-end: generate -> call_llm_json -> extract_prompt_response). Hardened a latent crash where reasoning models / refusals return `content=None` (`openrouter_iface.py`). Experiments can and should run through the actual pipeline; re-running the verticals + contamination fresh-item generation through the real framework is the next fidelity step.
