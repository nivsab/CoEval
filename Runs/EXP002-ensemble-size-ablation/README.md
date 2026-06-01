# EXP-002: Ensemble Size Ablation

**Status:** Partially simulated — P2 Important
**Paper section:** Section 5 (Analysis), Fig 8 (ensemble ablation curve)
**Backlog entry:** EXP-002 in `Docs/paperv2/experiment_backlog.md`

---

## Purpose

Validate that increasing judge ensemble size monotonically improves evaluation
reliability, specifically that the Spearman rho between the ensemble score and
the best single-judge score increases as more judges are added (1J to 5J).

Currently Fig 8 of the paper uses simulated values (1J=0.760, 2J=0.821,
3J=0.871) that must be replaced with real measurements before camera-ready.

---

## Hypothesis

Adding more judges to the ensemble will yield monotonically increasing
inter-rater reliability (measured as Spearman rho between k-judge ensemble
and single best judge), converging toward a stable maximum at 4-5 judges.
The ensemble will also exhibit lower variance than any single judge.

Expected pattern: rho(1J) < rho(2J) < rho(3J) < rho(4J) ≈ rho(5J),
consistent with the Condorcet jury theorem applied to LLM evaluation.

---

## Experimental Design

### Primary approach: Re-analysis of medium-benchmark-v1

This experiment is designed to require ZERO new API calls. The full
5-judge panel (gpt-4o-mini, gpt-3.5-turbo, qwen2p5-1b5, qwen2p5-0b5,
smollm2-1b7) was evaluated in medium-benchmark-v1. The ablation is computed
by the analyzer by selecting judge subsets of size k=1,2,3,4,5 and computing
the ensemble score for each subset.

Judge subsets to evaluate (in addition order):
- k=1: [gpt-4o-mini]
- k=2: [gpt-4o-mini, gpt-3.5-turbo]
- k=3: [gpt-4o-mini, gpt-3.5-turbo, qwen2p5-1b5]
- k=4: [gpt-4o-mini, gpt-3.5-turbo, qwen2p5-1b5, smollm2-1b7]
- k=5: [gpt-4o-mini, gpt-3.5-turbo, qwen2p5-1b5, smollm2-1b7, qwen2p5-0b5]
  NOTE: qwen2p5-0b5 was excluded from judge role in medium-benchmark-v1 due
  to empty JSON output. If k=5 is needed, qwen2p5-0b5 may be replaced with
  an additional run of gpt-3.5-turbo with a different temperature seed, or
  the ablation can stop at k=4 judges.

### Data source

- Run folder: `Runs/medium-benchmark/`
- Tasks: text_summarization, code_explanation, email_composition, data_interpretation
- Datapoints: 20 per task x 4 tasks x 5 teachers = 400 total
- Students: 5 models (same as teachers)
- Judges: gpt-4o-mini, gpt-3.5-turbo, qwen2p5-1b5, smollm2-1b7

### Fallback: generate new evaluations

If medium-benchmark-v1 phase5 data is incomplete, set `evaluation: Extend`
in this config and run:
```
coeval run --config Runs/EXP002-ensemble-size-ablation/config.yaml --continue
```

---

## Analysis Steps

1. Point analyzer at medium-benchmark-v1 data:
   ```
   coeval analyze ensemble-ablation \
       --run Runs/medium-benchmark \
       --judges gpt-4o-mini gpt-3.5-turbo qwen2p5-1b5 smollm2-1b7 \
       --out Runs/EXP002-ensemble-size-ablation/reports
   ```

2. The analyzer should compute for each k in {1, 2, 3, 4}:
   - Ensemble score (mean of k judges per criterion, then aggregated)
   - Spearman rho between k-judge ensemble and 4-judge full ensemble
   - Inter-judge agreement (Krippendorff's alpha or pairwise Cohen's kappa)

3. Plot rho vs. k (the ensemble ablation curve for Fig 8).

---

## Expected Outputs

- `reports/ensemble_ablation.html`: Interactive ablation curve chart
- `reports/ensemble_ablation.csv`: rho, variance, alpha per k value
- Updated Fig 8 data for the paper

---

## Required Resources

- No new API calls needed (pure re-analysis)
- Access to `Runs/medium-benchmark/phase5_evaluations/` directory
- Analyzer patch to support `ensemble-ablation` subcommand
  (estimated 2 hours coding as noted in backlog)

---

## Relationship to Paper

Directly replaces simulated values in:
- Fig 8: Ensemble size ablation curve (1J → 2J → 3J → 4J)
- Section 5 claims about ensemble reliability monotonicity
