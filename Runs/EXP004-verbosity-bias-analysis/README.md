# EXP-004: Verbosity Bias Analysis

**Status:** Not yet done — P3 Nice to Have
**Paper section:** Section 5 (Analysis), Table 10 (verbosity bias)
**Backlog entry:** EXP-004 in `Docs/paperv2/experiment_backlog.md`

---

## Purpose

Measure the correlation between student response length (token count) and
judge-assigned normalised score, comparing individual judges to the ensemble.
Validates the claim that ensemble calibration reduces verbosity bias
(the tendency for LLM judges to prefer longer, more verbose responses regardless
of actual quality).

Currently Table 10 reports simulated Pearson r values that must be replaced
before camera-ready submission.

---

## Hypothesis

Individual LLM judges exhibit measurable verbosity bias: longer responses
receive higher scores even when quality does not proportionally increase.
This is expected to manifest as a positive Pearson r between token_count
and score_norm for individual judges.

The 2-judge ensemble will show a smaller (closer to zero) correlation because:
1. The OLS calibration step normalises scores per judge
2. Different judges have different verbosity-length biases that partially cancel
3. Ensemble averaging reduces idiosyncratic scoring patterns

Expected values:
- Individual judge Pearson r: +0.15 to +0.35 (moderate positive bias)
- Ensemble Pearson r: +0.05 to +0.15 (reduced bias)

---

## Experimental Design

This experiment is a PURE RE-ANALYSIS of medium-benchmark-v1 — no new API
calls are required.

### Data source

- Run folder: `Runs/medium-benchmark/`
- Phase 4 response files: contain `token_count` field per response record
- Phase 5 evaluation files: contain normalised judge scores per criterion

### Analysis procedure

For each judge model J and each task T:
1. Load all phase4 response records for task T → (response_id, student_id, token_count)
2. Load all phase5 evaluation records for judge J, task T → (response_id, score_raw)
3. Compute score_norm = (score_raw - mean) / std per judge per task
4. Compute Pearson r(token_count, score_norm)
5. Compute Spearman rho(token_count, score_norm) as non-parametric check
6. Repeat for ensemble score = mean(score_norm across all judges)

Report:
- r and rho per individual judge per task
- r and rho for 2-judge ensemble per task
- Delta r = individual_r - ensemble_r (the bias reduction metric)

### Key metric: verbosity bias delta

Delta r = mean(individual judge r) - ensemble r

A large positive Delta r validates the claim that ensemble averaging reduces
verbosity bias.

---

## Analysis Commands

Run verbosity bias analysis against medium-benchmark-v1:
```
coeval analyze verbosity-bias \
    --run Runs/medium-benchmark \
    --out Runs/EXP004-verbosity-bias-analysis/reports
```

Note: This requires an analyzer patch to implement the `verbosity-bias`
subcommand. Estimated implementation time: 1 hour.

Alternatively, run directly with Python:
```python
from analyzer.verbosity_bias import compute_verbosity_correlation
results = compute_verbosity_correlation(
    run_dir="Runs/medium-benchmark",
    judges=["gpt-4o-mini", "gpt-3.5-turbo"],
    tasks=["text_summarization", "code_explanation", "email_composition",
           "data_interpretation"]
)
results.to_csv("Runs/EXP004-verbosity-bias-analysis/reports/verbosity_bias.csv")
```

---

## Expected Outputs

- `reports/verbosity_bias.csv`: Pearson r and Spearman rho per judge per task
- `reports/verbosity_bias.html`: Scatter plots of token_count vs. score_norm
- Table 10 data: Verbosity bias correlation table

---

## Required Resources

- No API calls needed
- Access to `Runs/medium-benchmark/` phase4 and phase5 output files
- Analyzer patch for `verbosity-bias` subcommand (1 hour coding)
- scipy, pandas installed (for Pearson/Spearman computation)

---

## Relationship to Paper

Directly replaces simulated values in:
- Table 10: Verbosity bias correlation (Pearson r, Spearman rho)
- Section 5 claims about ensemble calibration reducing verbosity bias
- Deferred gap G6 (verbosity correlation) from paper status notes
