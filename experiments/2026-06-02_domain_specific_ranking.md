# EXP-011 — Model rankings are domain-specific (cross-domain divergence)

**Status:** completed
**Date:** 2026-06-02
**Run:** `Runs/EXP011-domain-divergence/`
**Artifacts:** `Runs/EXP011-domain-divergence/reports/v2_domain_divergence.json`,
`.../v2_domain_divergence_bootstrap.json`

## Hypothesis
The best model for one domain need not be the best for another, so a single pooled
public leaderboard misdirects a domain practitioner. CoEval's value is per-domain
ranking. Falsified if the four per-domain rankings agree (high cross-domain Kendall
tau, same #1 everywhere); confirmed if the top model changes by domain and the
divergence survives item resampling.

## Setup
- 6 candidate students: gpt-4o-mini, gpt-3.5-turbo, claude-3.5-haiku, gemini-flash,
  llama-3.1-8b, qwen-2.5-7b (llama/qwen via OpenRouter).
- 3 vendor-disjoint judges: judge-gpt4o-mini (OpenAI), judge-claude-haiku
  (Anthropic), judge-gemini-flash (Google).
- Teacher: gpt-4o-mini. Rubric: auto.
- 4 de-novo domains x 25 generated items = 100 items; 600 responses; **1,800
  evaluations at 100% coverage**. Cost USD 0.83.
- Config: `Runs/EXP011_config.yaml`. Analysis: `scripts/v2_domain_divergence.py`
  (+ `scripts/v2_domain_divergence_bootstrap.py`, B=2000 item bootstrap).

## Headline numbers
- **3 distinct top models** across 4 domains: gpt-4o-mini (clinical, code),
  gemini-flash (legal), claude-3.5-haiku (math).
- Bootstrap P(top) of each winner: clinical gpt-4o-mini **0.88**, code gpt-4o-mini
  **0.60**, legal gemini-flash **1.00**, math claude-3.5-haiku **0.85**.
- Mean cross-domain Kendall tau = **0.19** (weak); least-aligned pair code-vs-math
  point estimate tau = **-0.41** (bootstrap mean -0.19, negative in 72% of draws,
  CI crosses 0 — reported as "lowest", not "significantly anti-correlated").
- Generic pooled leaderboard top = **gemini-flash**; it is domain-best for only
  **1 of 4** domains (legal). gpt-4o-mini swings rank 1 (clinical) -> 5/6 (math,
  P(top)=0.001); claude-3.5-haiku rank 1 (math) -> 5 (code).

## Per-domain ranking (ensemble mean, [0,1])
| Domain | #1 | #2 | #3 |
|---|---|---|---|
| clinical | gpt-4o-mini .919 | gemini-flash .902 | claude-3.5-haiku .900 |
| code | gpt-4o-mini .904 | qwen-2.5-7b .893 | gemini-flash .891 |
| legal | gemini-flash .890 | gpt-4o-mini .860 | claude-3.5-haiku .843 |
| math | claude-3.5-haiku .878 | gemini-flash .858 | gpt-3.5-turbo .850 |

## Conclusion
A generic leaderboard misdirects 3 of 4 domain practitioners; CoEval ranks on the
practitioner's own domain, contamination-free, no labels. Integrated into the paper
as §5.8 + Table 6 + finding box. Bootstrap confirms the divergence is real, not
score-band noise.
