# CoEval Publication Push — Standing Instructions & TODO

> Living document. Maintained by Claude across sessions. Mirrors the user's standing
> instructions and the prioritized task backlog. Update status inline; append, do not delete.

Last updated: 2026-06-01

---

## Standing instructions — UPDATED (session 3, 2026-06-01)

10. **Venue: JOURNAL preferred over conference.** Target **TMLR** (recommended: rolling, no deadline, rewards rigor + reproducibility, ML-focused) or **TACL** (NLP journal, rolling). No deadline pressure → be thorough.
11. **Paper deliverable: HTML + KaTeX math, published as a GitHub Pages index** (`docs/` site with `index.html`). Not docx/PDF-first.
12. **Keys security:** keys.yaml must stay gitignored, never in repo. VERIFIED clean (session 3): gitignored, never committed, only doc-placeholders in tracked files.
13. **Use batch API to save money** where possible. OpenRouter has NO batch → for expensive judge phases use **native Anthropic Message Batches + Gemini Batch** (50% off; direct keys work — Anthropic key validated). OpenAI direct is billing-locked.
14. **Avoid local models.** Prefer OpenRouter-hosted models (done: EXP-001 weak student qwen-local → `meta-llama/llama-3.2-3b-instruct`). Use **gpu2vast / gpu2runpod** skills only if heavy GPU work is unavoidable (e.g., large BERTScore); never the local RTX 2060 for long jobs.
15. **POSITIVE RESULTS directive:** do NOT report nuanced/negative/"honest-limitation" framings. For any apparent negative: research the root cause, ADAPT the configuration/methodology, and obtain the positive result the method genuinely achieves (NOT fabrication — achieve it by correct setup, e.g. strong cross-family judge panel → monotone reliability). Frame every finding as supporting CoEval.

16. **Sanity/debug session after EVERY major cycle.** After each experiment batch or review cycle, run a results-sanity + bug-hunt pass before reporting: judge validity rates, score distributions, join/dedup correctness, NaN handling, metric sanity, reproduce headline numbers, run the full test suite. No bugs ship into the paper.

## Standing instructions (from the user)

1. **Open-source usability is a first-class goal.** Improve the code so *anyone* can use CoEval to evaluate a model or model family for **specific domains, applications, and tasks**. Lower the barrier to entry (quickstart, clear errors, docs, sensible defaults).
2. **Budget:** autonomous spend allowed **up to $20**. Any spend **beyond $20 requires explicit approval** (give a budget estimate first). OK to run real SOTA experiments with frontier models on relatively large data once approved.
3. **Frontier models:** use **OpenRouter** (key present) for cross-vendor frontier judges/students (Claude, Gemini, DeepSeek, Llama, etc.).
4. **Demonstrate superiority over existing benchmarks.** Even anecdotal/empirical proof is valuable. Emphasize: **application-focused**, **no leakage/contamination**, and **coverage of cases underrepresented in current benchmarks**.
5. **Continue autonomously, never stop** unless approval for >$20 is needed.
6. **Quality discipline:** several code reviews, validation, polish, sanity checks. Root-cause negative results (never accept them blindly); find fixes/workarounds.
7. **Maintain this MD** with instructions + TODO.
8. **Target venue: COLM 2026.** Plan experiments + paper revisions to make it a strong accept.
9. **Analyze the paper:** framing, contributions, missing experiments — make it a strong academic paper.

---

## TODO (prioritized)

### P0 — In flight
- [~] **EXP-001** benchmark-grounded comparison (xsum + cnn_dailymail + codesearchnet, 300 dp, +gpt-4o frontier judge). Data staging in progress. Next: `coeval plan` → submit OpenAI batch (phases 4-5) + qwen on GPU → `score_responses` → strong-panel ablation. Est ~$9 (<$20, autonomous).
- [~] Related-work scout (background agent) — for related work + baselines + leakage evidence.
- [~] Paper-for-COLM analysis (background agent) — framing/contributions/experiment gaps.

### P1 — Core to acceptance
- [ ] **Demonstrate superiority vs static benchmarks** (the headline empirical contribution):
  - [ ] Leakage/contamination angle: show CoEval items are fresh/uncontaminated vs static benchmark leakage.
  - [ ] Coverage angle: quantify attribute strata underrepresented in a static benchmark that CoEval covers (use ACR/RAR).
  - [ ] Application-focus angle: a domain-specific case study (pick a vertical) where generic benchmarks mislead but CoEval discriminates.
- [ ] **EXP-003** positional bias: write `scripts/positional_bias_analysis.py` + run (~$1.30, autonomous).
- [ ] **Wire `stats.py` CIs/ICC/BH into `paper_tables.py`** (Tables 3/5/6 need bootstrap CIs + ICC + multiple-testing correction).
- [ ] **G-Eval baseline** (gpt-4o, OpenRouter for cross-vendor) — the expected comparison evaluator.
- [ ] **SOTA frontier experiment** (needs budget estimate + approval if >$20): strong-only judge panel (gpt-4o + Claude + Gemini via OpenRouter) on a larger, domain-specific, leakage-free task set; demonstrates monotone reliability recovery (per EXP-002) + superiority.

### P2 — Open-source usability
- [ ] Code review passes on new modules + polish (validation, error messages, docstrings).
- [ ] Quickstart for "evaluate my model on my domain" workflow; document new analyze subcommands.
- [ ] Sanity-check full test suite; fix pre-existing `TestExportBenchmark` theta drift (7 failures) if cheap.

### P3 — Paper
- [ ] Reframe for COLM (from ACL drafts): sharpen contribution, real numbers, no simulated results.
- [ ] Produce the paper in **HTML + KaTeX** (per earlier instruction).

---

## Completed (2026-06-01) — session 4/5 (paper presentation)
- HTML paper figures: F1 architecture (clean, rebuilt), F2 ICC curve, F3 verbosity forest, F4 correctness corr, F5 rubric heatmap, F6 cost — from real data via `docs/paper/figures/make_figures.py`. + Appendix A report gallery (5 genuine screenshots).
- Table 1 rewritten with genuine FRONTIER cross-family panel (gpt-4o + claude-sonnet-4 + gemini, ρ=0.859, 0 disagreements) — resolves reviewer W3.
- **KaTeX inline math FIXED** (verified live: katexLoaded, 84 spans, no runaway): wrong JS integrity hash removed; money `$`→`USD` (the `\$` escape does NOT work in KaTeX auto-render — it still parsed `$5.89` as a math delimiter and rendered whole sections as math).
- Text **justified** (verified textAlign=justify); **TMLR "Submitted" line dropped** (verified absent).
- **Literature complete + recent**: +12 verified 2024-2026 works (Preference Leakage 2502.01534, Play Favorites 2508.06709, CALM/Ye, surveys, ChatEval, LiveBench, LiveCodeBench, domain-specific, meta-judges). Bibliography 26 refs, all resolve via bibtest. All [n] citations cross-checked.
- Preview server at port 8124 (.claude/launch.json); verify with mcp__Claude_Preview__preview_eval.

## Completed (2026-06-01) — session 2
- `Code/analyzer/self_eval_control.py` + tests: **self-evaluation confound quantified** ($0). Excluding self-judging drops student-rank Spearman to 0.40; excluding same-family shifts gpt-4o-mini −0.090 (its high score was partly OpenAI-family judges favoring it). Honest nuance: same-family exclusion is muddied here because the only cross-family judges are weak small models → motivates the cross-family frontier experiment.
- EXP-001 pivoted to OpenRouter cross-family panel; running. API student phase cost $0.14. 35 new analyzer tests pass.
- Budget question for SOTA experiment returned empty → treated as NOT approved (>$20 gate). Re-ask once EXP-001 lands.

## Completed (2026-06-01) — session 1
- `Code/analyzer/stats.py` (bootstrap CI, ICC(3,1)/(3,k), Benjamini-Hochberg) — validated vs Shrout & Fleiss.
- EXP-002/004/005 analyzer modules + CLI subcommands + `experiment_reports.py`. Real numbers in `experiments/INDEX.md`.
- `Public/benchmark/score_responses.py` — fixes the reference-vs-itself integrity bug (per-response scoring).
- `Public/benchmark/setup_exp001.py` — stages EXP-001 benchmarks (bridges broken README ingestion).
- gpt-4o frontier judge added to EXP-001 config.
- 32 new tests; zero regressions (1024 prior pass).

## Synthesized plan (from related-work scout + COLM paper review, 2026-06-01)

### Reality checks
- **COLM 2026 deadline (Mar 31, 2026) has PASSED** (today 2026-06-01). Realistic target: **COLM 2027** or an open cycle (EMNLP/ARR, NeurIPS D&B ~May/Jun). Build the strongest paper regardless; confirm venue/timing with user.
- **Paper verdict (current): clear reject.** Every comparative/superiority/bias claim is currently *simulated* and self-labeled. The fix is running experiments, not prose. Most needed reanalyses cost **$0** (re-aggregate existing logs).

### Paper fixes (framing + contributions)
- Re-center on the **empirical finding**: "ensemble *composition* (cross-family judge diversity), not panel size, is the first-order reliability variable; teacher capability is non-monotone in benchmark discriminativeness." Pipeline is the vehicle, not the headline. (Backed by EXP-002.)
- **Add + contrast AutoBencher** (Li et al. 2024, arXiv:2407.08351) — closest prior work, currently uncited. Also cite PoLL (Verga 2404.18796), G-Eval (2303.16634), Prometheus 2 (2405.01535), FLAMe (2407.10817), JudgeBench (2410.12784), BiGGen-Bench (2406.05761), GSM1k contamination (2405.00332), contamination survey (2502.14425).
- **Merge V1/S2** (S2=√V1, identical rankings) → one variance metric + one range metric.
- **Demote OLS calibration** from headline (it is disabled by default; effect unmeasured) to an optional module — or measure it with continuous metric-judges.
- Downgrade unconfirmed claims (teacher ranking has overlapping CIs) to hypotheses. Scrub all `(planned)`/`(*)`/PATCH-NOTE scaffolding.

### Baselines CoEval must compare to
G-Eval (single CoT judge) · Prometheus 2 (open rubric judge) · MT-Bench single-GPT-4 judge · PoLL / naive majority-vote panel.

### 3 superiority demonstrations (the empirical keystone)
1. **Ground-truth correlation (EXP-001, running):** CoEval ensemble vs native metric vs G-Eval vs BERTScore on loader-backed tasks. Spearman/Kendall with bootstrap CIs.
2. **Leakage-freeness (GSM1k-style):** score models on a static (contaminated) benchmark vs CoEval-fresh items matched on attributes; show overfit models drop, frontier hold (mirror GSM1k 13% / APPS 4.9x). Plus n-gram/membership overlap of CoEval items vs static.
3. **Underrepresented-stratum coverage:** per-stratum accuracy where static benchmarks have ~0 tail items; report ranking-flip rate vs the static leaderboard (AutoBencher "tail knowledge" framing).

### Reviewer's top weaknesses (ranked) → fix
1. Central evidence simulated → run EXP-001 (in progress). 2. Real+fabricated numbers mixed in sentences → run $0 reanalyses. 3. OLS overclaim → demote/measure. 4. Same-family non-frontier panel → add cross-family frontier judges (gpt-4o + Claude + Gemini via OpenRouter). 5. No human baseline → small human study. 6. Self-eval confound, control only projected → re-aggregate excluding within-family triples ($0). 7. Contamination/domain asserted not measured → measure + 1 domain case study. 8. Contribution inflation → merge metrics, cut enumeration.

### Free ($0) reanalyses to run next (no API)
- Self-evaluation control: re-aggregate medium-benchmark excluding self/within-family triples (defuses confound, makes Table 7 interpretable).
- J* robust-filter ablation from existing logs.
- Finish Table 7 aggregation so no "approximated" daggers remain.

### SOTA frontier experiment (needs budget approval — see proposal below)
Cross-family frontier panel (gpt-4o + Claude Sonnet + Gemini via OpenRouter) as judges + ≥1 frontier student tier, on a domain-specific contamination-free task set, with G-Eval baseline + leakage measurement. Demonstrates monotone reliability recovery (per EXP-002) + superiority.

## Key findings / decisions
- **EXP-002 non-monotone reliability** is a real (root-caused) finding: weak judges *reduce* ensemble reliability → motivates judge selection. Strong-only panel should recover monotonicity.
- **Verbosity bias cancels in ensemble** (mean|r| 0.153 → 0.010).
- Venue: **COLM 2026**. Authors: Apartsin (HIT) + Aperstein (Afeka).
- Pre-existing test drift: `TestExportBenchmark` (7) fail on clean tree (robust-filter theta default).
