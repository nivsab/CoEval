# DIAGNOSTIC — EXP-010 pilot (many-model rank-recovery)

**Status:** diagnostic (NOT paper material) · **Date:** 2026-06-02 · **Cost:** ~$0.6 (pilot)
**Purpose:** de-risk the scaled rank-recovery run before spending on 12-20 models.

## What ran
8-model student roster (gpt-4o, gpt-4o-mini, claude-3.5-sonnet, claude-3.5-haiku,
gemini-2.5-flash, llama-3.3-70b, llama-3.1-8b, llama-3.2-3b) + 3-judge cross-family panel
(gpt-4o-mini, claude-3.5-haiku, gemini-2.5-flash), on 40 reused gold sciq science-QA items.
Reused EXP001b ingested items; ran phases 4-5 via the real framework.

## What the pilot PROVED (mechanics)
- The multi-model pipeline runs end-to-end through the actual framework: 7/8 students
  collected, all judged, gold-scored, rank-recovery analysis runs. (`scripts/v2_rank_recovery.py`)

## Issues surfaced and triaged (Research-Honesty bug-hunt)
1. **Bad model slug:** `anthropic/claude-3.5-sonnet` 404s on OpenRouter ("No endpoints").
   FIX for scaled run: use `anthropic/claude-sonnet-4` (verified in EXP001b) or current slug.
2. **OpenRouter rate-limit truncation:** a single `coeval run` left phase-5 judging at ~118/280
   responses (exit 0 but incomplete). A second `--continue` (Extend mode) completed coverage
   to 7x3x~240 units. FIX for scaled run: route judges through native **Batch APIs**
   (openai/anthropic/mistral, 50% off + no real-time rate cap), per the scale plan. This is
   the concrete reason the batch interface matters operationally, not just for cost.
3. **Gold metric is output-format sensitive (FIXED):** raw `inclusion_match` scored
   claude-3.5-haiku at 0.05 because it answered with the option LETTER ("C") while the gold
   reference is the answer TEXT ("fossilization"). Built a format-robust MCQ scorer
   (`scripts/v2_gold_rescorer.py`: parses the option block, resolves the correct letter, credits
   a response that names the correct option by EITHER letter or text). After the fix, haiku
   gold = 0.95 (sensible). This is the canonical exact-match/format-mismatch trap.
4. **Task saturation (DESIGN BLOCKER for the result):** after the format fix, all 7 models
   score 0.95-1.00 on sciq (gold range 0.05). sciq science-QA is too easy to discriminate
   modern models, so rank-recovery is ill-defined (near-ties). NO rank-recovery number from
   this pilot is reportable.

## Refined design for the clean scaled run (carry forward)
- **Harder, discriminating task** so gold spreads models: ARC-Challenge is borderline; prefer
  logiqa / bigbench_hard / mathqa / math (or MMLU-Pro / GPQA via ingest) where small models
  fall well below frontier. Mix 2-3 tasks for robustness.
- **Format-robust gold scorer** (done) for every MCQ task.
- **Batch-routed judging** (openai/anthropic/mistral native Batch) to avoid the rate-limit
  truncation and halve cost; pre-flight with `coeval plan`.
- **12-20 models** spanning frontier -> tiny for rank-order resolution; fix the sonnet slug.
- Metric: Spearman/Kendall(CoEval ensemble ranking, gold accuracy ranking); compare to
  best/worst single judge (judge-choice regret in ranking terms) and plain mean.

## Artifacts
`Runs/EXP010-scale-ranking-pilot/` (config, phases, reports/v2_rank_recovery.json),
`scripts/v2_rank_recovery.py`, `scripts/v2_gold_rescorer.py`.

## Update — ARC-Challenge iteration (harder task added)
Added ARC-Challenge (science_reasoning, 40 items) to spread models, with the
format-robust gold scorer. Completing phase-5 judging took an automated
reset+`--continue` loop (5 cycles) because OpenRouter rate-limiting truncates each
run at ~118/280 responses — reinforcing that the scaled run MUST use native Batch APIs.

ARC gold spreads the weak end (llama-3.2-3b 0.70, llama-3.1-8b 0.90, llama-3.3-70b
0.975) but frontier models still saturate at 1.0. Combined sciq+ARC (80 items/model),
7 models, gold range 0.85-1.00:
- **CoEval ensemble (accuracy aspect) rank-recovery: Spearman 0.873, Kendall 0.751.**
- Matches the best single judge (gemini-flash / gpt-4o-mini, Kendall 0.751) and beats
  the worst (claude-haiku 0.551); judge-choice regret (tau) = 0.20. Consistent with the
  paper's low-regret thesis (the ensemble is never the bad pick).

## Honest assessment (NOT yet paper-headline)
Encouraging preliminary signal that CoEval's label-free ranking tracks gold capability
(rho 0.87) and is the low-regret choice. NOT clean enough to headline because:
(i) these benchmarks saturate frontier models (4-way top tie -> ranking partly unorderable);
(ii) 7 models, not the 12-20 targeted (claude-3.5-sonnet slug 404'd);
(iii) the ensemble ties rather than beats the best single judge here.

## Clear path to the clean result (the funded scaled run)
1. HARDER task with frontier spread: ingest MMLU-Pro / GPQA / MATH (the `coeval ingest`
   command supports mmlu, gsm8k, medqa) or use bigbench_hard / math loaders.
2. 12-20 models spanning frontier->tiny; fix the sonnet slug (anthropic/claude-sonnet-4).
3. BATCH-routed judging (openai/anthropic/mistral native Batch) — eliminates the
   rate-limit truncation that forced 5 reset+continue cycles here, and halves cost.
4. Expect: with real frontier spread, the ensemble should pull strictly above the best
   single judge (the regret result), giving the headline rank-recovery win.
