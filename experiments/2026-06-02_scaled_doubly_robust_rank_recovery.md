# EXP-012 — Scaled doubly-robust rank-recovery (13 models)

**Status:** completed
**Date:** 2026-06-02
**Run:** `Runs/EXP012-scale-ranking-16/`
**Artifact:** `Runs/EXP012-scale-ranking-16/reports/v2_doubly_robust_ranking.json`

## Hypothesis
The 7-model EXP010 doubly-robust rank-recovery (plain 0.87 → 0.98) rests on a
Spearman over only n=7 models. Scaling to ~13 models on the same sciq+ARC gold and
the same vendor-disjoint judge panel should replicate the finding on a larger model
base, removing the small-n concern. Confirmed if doubly-robust still recovers the
gold ranking well and remains robust to an injected random judge.

## Setup
- 16 candidate students configured (capability gradient gpt-4o … llama-3.2-1b);
  3 (gemma-2-9b, phi-3.5-mini, mistral-7b) returned OpenRouter "no endpoints" 404 at
  generation despite probing OK, leaving **13 students**.
- Items: the same 80 sciq+ARC datapoints as EXP010 (copied into phase3 so the new
  models answer the identical item set; directly comparable).
- Judges: judge-gpt4o-mini, judge-claude-haiku, judge-gemini-flash (vendor-disjoint).
- 1,040 responses (13×80); **3,033 / ~3,120 evals (97%)** (claude-haiku 954, the rest
  JSON-parse failures that exhausted retries). Cost well under USD 5.
- Config: `Runs/EXP012_config.yaml`. Gold: `scripts/v2_gold_rescorer.py` (mcq-robust).
  Analysis: `scripts/v2_doubly_robust_ranking.py EXP012-scale-ranking-16`.

## Bug found and fixed (private diagnostic, not in paper)
First gold pass used `score_responses --metric exact_match`, which scores "(C)
fossilization" as 0 against bare-text gold "fossilization". This drove gold accuracy
to 0.000 for 12 of 13 students (only the weakest tiny model, which omits the letter
prefix, scraped 0.075), INVERTING the gold ranking and producing a spurious
rank-recovery of -0.46 with all four aggregators identical (degenerate near-constant
gold vector). Root cause = measurement formatting bug, not a real result. Fix =
`v2_gold_rescorer.py` (the format-robust MCQ scorer EXP010 used; credits a response
that names the correct option by letter OR text). Re-scored gold is a sensible
gradient: deepseek/gpt-4o 1.000 … llama-3.2-3b 0.850, llama-3.2-1b 0.650, mean 0.944.

## Headline numbers (correct gold)
| Aggregator | Spearman | Kendall |
|---|---|---|
| plain mean | 0.882 | 0.761 |
| item-weighted only | 0.936 | 0.844 |
| judge-weighted only | 0.919 | 0.816 |
| **doubly-robust** | **0.950** | **0.871** |

- Robust to a broken judge: plain mean degrades to 0.849 (Kendall 0.705) when a random
  judge is injected; doubly-robust holds at 0.950, rogue judge weight = **0.00**.
- Item-weight concentration: top 10% of items hold 0.324 of weight; 0.463 of items
  near-zero weight (saturated, carry no ranking signal).

## Conclusion
Replicates and scales EXP010 to n=13 models: doubly-robust recovers the gold ranking
at Spearman 0.95 (from plain 0.88), both weights complementary, robust to a rogue
judge. §5.2 updated to the 13-model numbers (the 7-model pilot stays here/in commits).
