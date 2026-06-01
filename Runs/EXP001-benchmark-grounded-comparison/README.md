# EXP-001: Benchmark-Grounded Comparison Experiment

**Status:** Planned — P1 Critical
**Paper section:** Section 4 (Results), Table 8, Fig 8
**Backlog entry:** EXP-001 in `Docs/paperv2/experiment_backlog.md`

---

## Purpose

Validate CoEval's comparative claims against real benchmark native metrics.
Currently Table 8 of the paper reports Spearman rho values comparing CoEval
ensemble scores to BERTScore-F1 and BLEU-4. Those values are simulated
placeholders that must be replaced with real experimental results before
camera-ready submission.

This experiment runs the full CoEval pipeline on three public NLP benchmarks,
then computes both CoEval ensemble scores and benchmark-native scores on the
same responses, enabling direct Spearman rho comparison.

---

## Hypothesis

CoEval ensemble scores will correlate more strongly with benchmark native
metrics (BERTScore-F1, BLEU-4) than individual judge scores or non-ensemble
baselines, demonstrating that rubric-based LLM evaluation captures meaningful
quality signal beyond surface-level lexical overlap.

Expected outcome: Spearman rho (CoEval ensemble vs. benchmark native) > 0.70
across all three tasks, with individual judge rho in the 0.55-0.70 range.

---

## Experimental Design

### Benchmarks used

| Task name | Benchmark | HuggingFace dataset ID | Native metric | Items |
|---|---|---|---|---|
| text_summarization | XSum | EdinburghNLP/xsum | BERTScore-F1 | 100 |
| news_summarization | CNN/DailyMail | abisee/cnn_dailymail (v3.0.0) | BERTScore-F1 | 100 |
| code_explanation | CodeSearchNet | code-search-net/code_search_net | BLEU-4 | 100 |

Total: 300 benchmark datapoints across 3 tasks.

### Models

- **Teachers:** Virtual benchmark loaders (interface: benchmark) — no Phase 3 API calls
- **Students:** gpt-4o-mini, gpt-3.5-turbo, qwen2p5-1b5 (Qwen/Qwen2.5-1.5B-Instruct)
- **Judges:** gpt-4o-mini, gpt-3.5-turbo (2-judge ensemble)

### Phases

- Phase 1 (attribute mapping): Static — 0 LLM calls
- Phase 2 (rubric mapping): Static — 0 LLM calls
- Phase 3 (data generation): Pre-ingested from benchmark loaders — 0 LLM calls
- Phase 4 (response collection): 300 items x 3 students = 900 API calls
- Phase 5 (evaluation): 900 responses x 2 judges = 1800 API calls

Total API calls: ~2700 (batch-eligible, ~1350 effective billing units)

---

## Setup Instructions

1. Install benchmark dependencies:
   ```
   pip install evaluate bert_score sacrebleu
   ```

2. Ingest benchmark data (run once before `coeval run`):
   ```
   coeval ingest --config Runs/EXP001-benchmark-grounded-comparison/config.yaml \
       --benchmark xsum --split test --n 100
   coeval ingest --config Runs/EXP001-benchmark-grounded-comparison/config.yaml \
       --benchmark cnn_dailymail --split test --n 100
   coeval ingest --config Runs/EXP001-benchmark-grounded-comparison/config.yaml \
       --benchmark codesearchnet --split test --n 100
   ```

3. Run the experiment:
   ```
   coeval run --config Runs/EXP001-benchmark-grounded-comparison/config.yaml
   ```

4. Compute benchmark-native scores post-run:
   ```
   python -m benchmark.compute_scores \
       --run Runs/EXP001-benchmark-grounded-comparison
   ```

5. Generate reports:
   ```
   coeval analyze all \
       --run Runs/EXP001-benchmark-grounded-comparison \
       --out Runs/EXP001-benchmark-grounded-comparison/reports
   ```

---

## Expected Outputs

- `phase4_responses/`: Student responses for all 300 datapoints x 3 students
- `phase5_evaluations/`: Judge scores per criterion for all responses
- `reports/`: HTML + Excel analysis reports including calibration tables
- Table 8 data: Spearman rho matrix comparing CoEval (per-judge + ensemble) vs.
  BERTScore-F1 (xsum, cnn_dailymail) and BLEU-4 (codesearchnet)

---

## Required Resources

- OpenAI API key (OPENAI_API_KEY)
- HuggingFace token (HF_TOKEN) for qwen2p5-1b5 local inference
- GPU with >= 8 GB VRAM (or CPU with load_in_4bit=true) for HF model
- Estimated compute time: ~2-4 hours (depends on batch queue)
- Estimated API cost: see `cost_estimate.txt`

---

## Relationship to Paper

Directly replaces simulated values in:
- Table 8: Benchmark comparison (rho vs BERTScore, BLEU baselines)
- Claims R1 and R2 in Section 4 regarding correlation with established metrics
- Blocking items: Table 8, all external-baseline rho comparisons
