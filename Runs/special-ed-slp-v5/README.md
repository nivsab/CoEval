# special-ed-slp-v5: Israeli Special Education SLP Benchmark

**Status:** Completed — single judge (llama-8b); second judge pending (ICC not yet computable)
**Domain:** Speech-Language Pathology / Special Education (Hebrew, Israeli clinical context)
**Contributor:** Niv Saban
**Expert validation:** Teacher datapoints (prompts + reference responses) reviewed and approved
by a certified Israeli SLP practitioner with experience in special education. Reviewer
confirmed clinical accuracy of MATRIX stage assignments, stage-to-tool mappings, and the
clinical traps embedded in adversarial cases.

---

## Purpose

Demonstrate and validate CoEval on a new domain entirely absent from the original paper:
Hebrew-language clinical tasks in Israeli special education, designed with expert SLP
domain knowledge. This run exercises the full five-phase CoEval pipeline on tasks where
(a) no public labeled benchmark exists, (b) standard English benchmarks are inapplicable,
and (c) correct evaluation requires clinical expertise unavailable to generic judge models.

The experiment provides two contributions:

1. **Domain transferability evidence** — CoEval produces discriminative, clinically
   grounded rankings in a Hebrew-language domain the paper does not address, with
   expert-authored rubric criteria and adversarial datapoints.

2. **Rubric quality from domain expertise** — The rubric dimensions (`stage_accuracy`,
   `resource_appropriateness`, `epistemic_honesty`, etc.) encode clinical knowledge
   (Communication Matrix by Rowland; MATRIX stage-to-tool mapping; triad consistency)
   that generic English benchmarks cannot capture.

3. **Expert-validated teacher data** — All 40 teacher-generated prompts and reference
   responses (phase 3) were reviewed by a certified Israeli SLP practitioner with
   special education experience. The reviewer confirmed clinical accuracy of MATRIX stage
   assignments, stage-to-tool recommendations, and the clinical traps embedded in
   adversarial cases. This validation step demonstrates CoEval's design principle that
   the teacher role requires domain authority, not just a large model.

---

## Relationship to the Paper

The CoEval paper demonstrates domain case studies in drug-drug interaction, clinical
reasoning, and legal analysis (Section 5.6). This run extends that evidence to a fourth
domain — Israeli SLP / special education — with the following unique characteristics:

| Characteristic | Paper domains | This run |
|---|---|---|
| Language | English | Hebrew |
| Benchmark availability | None (custom) | None (custom) |
| Domain-expert rubric | Yes | Yes (SLP-validated) |
| Adversarial datapoints | Yes | Yes (clinical traps) |
| ICC / multi-judge panel | 3 judges | 1 judge (pending) |

When a second judge is added, this run will provide:
- ICC(3,k) between two vendor-disjoint judges on Hebrew clinical text
- Evidence that CoEval's rubric design captures domain knowledge inaccessible to
  generic benchmarks (e.g., stage-tool alignment errors that exact-match metrics miss)

---

## Experimental Design

### Tasks

| Task | Description | Datapoints |
|---|---|---|
| `communication_stage_assessment` | Identify a child's MATRIX communication development stage from a behavioral scenario; detect adversarial clinical traps | 20 |
| `communication_plan_design` | Design an individualized communication plan (תת"ח) with stage-appropriate goals, tools, and ecological validity for a stated implementer and framework | 20 |

Total: 40 datapoints, Hebrew, grounded in Israeli special education practice.

### Domain knowledge embedded in rubric

**`communication_stage_assessment` rubric:**
- `stage_accuracy` — Correct MATRIX stage (Rowland levels 1–7); adversarial: detects clinical trap where surface behavior mimics a higher stage
- `profile_specificity` — References the specific child's behaviors, not generic diagnosis description
- `diagnostic_coherence` — Internally consistent; handles non-linear autism profiles (gaps across early MATRIX levels)
- `epistemic_honesty` — States uncertainty for ambiguous/misleading profiles; names the clinical trap for adversarial cases
- `practical_value` — Actionable for a clinician; addresses triad consistency across people/contexts/time

**`communication_plan_design` rubric:**
- `stage_accuracy` — Goals and strategies calibrated to actual MATRIX stage
- `goal_appropriateness` — Specific, measurable, next-step from current level; includes triad consistency goal
- `resource_appropriateness` — Stage-to-tool alignment: pre-intentional → sensory cues only (no pictures/PECS/AAC); adversarial: tool looks correct by diagnosis label but is wrong for this profile
- `ecological_validity` — Implementable by stated implementer (clinician / classroom aide / parent) in stated framework (home / kindergarten / class)
- `practical_value` — Immediately actionable; includes specific activities, routines, prompting hierarchies

### Models

| Role | Model | Provider | Notes |
|---|---|---|---|
| Teacher | `llama-4-scout-17b` | Groq | Generates Hebrew clinical scenarios + reference responses |
| Student | `gpt-4o-mini` | OpenAI | Frontier small |
| Student | `llama-3.3-70b-versatile` | Groq | Open-weight large |
| Student | `qwen3-32b` | Groq | Open-weight, multilingual |
| Student | `gemma-4-26b` | OpenRouter | Google, free tier |
| Student | `nemotron-3-super-120b` | OpenRouter | NVIDIA, free tier |
| Judge | `llama-3.1-8b-instant` | Groq | Primary judge; 1,050 valid evaluations |

### Pipeline phases

| Phase | Mode | Output |
|---|---|---|
| 1 — Attribute mapping | New | Target + nuanced attribute grids |
| 2 — Rubric mapping | New | 5-factor rubric per task |
| 3 — Data generation | New | 40 Hebrew clinical datapoints |
| 4 — Response collection | New | 200 student responses (5 × 40) |
| 5 — Evaluation | New + Extend | 1,050 judge evaluations (judge-llama-8b) |

---

## Key Results (judge-llama-8b, single judge)

### Overall student ranking

| Rank | Model | Avg score | Valid evals |
|---|---|---|---|
| 1 | gemma4-26b | 0.726 | 195 |
| 2 | gpt-4o-mini | 0.718 | 200 |
| 3 | qwen-32b | 0.698 | 210 |
| 4 | nemotron-120b | 0.554 | 195 |
| 5 | llama-70b | 0.460 | 210 |

### Per-task breakdown

**communication_stage_assessment** (stage identification):

| Model | Score |
|---|---|
| gemma4-26b | 0.929 |
| qwen-32b | 0.857 |
| gpt-4o-mini | 0.847 |
| nemotron-120b | 0.763 |
| llama-70b | 0.567 |

**communication_plan_design** (intervention planning):

| Model | Score |
|---|---|
| gpt-4o-mini | 0.600 |
| qwen-32b | 0.538 |
| gemma4-26b | 0.489 |
| nemotron-120b | 0.355 |
| llama-70b | 0.352 |

**Observation:** The two tasks produce distinct rankings. Plan design is substantially
harder than stage identification for all models, and task-specific rankings diverge
(gemma4-26b leads assessment; gpt-4o-mini leads planning). This cross-task divergence
supports the paper's claim that rankings are task-specific and cannot be inferred from
general benchmarks.

---

## Status and Next Steps

| Item | Status |
|---|---|
| phase 1–4 | Complete |
| phase 5 — judge-llama-8b | Complete (1,050 evaluations) |
| phase 5 — second judge | Pending (need vendor-disjoint judge with API credits) |
| ICC(3,k) / judge consistency | Blocked on second judge |
| coeval analyze reports | Generated (single-judge) |

To add a second judge and compute ICC:
```bash
# Add a vendor-disjoint judge to config.yaml, then:
coeval run --config Runs/special-ed-slp-v5/config.yaml --continue
coeval analyze all --run Runs/special-ed-slp-v5 --out Runs/special-ed-slp-v5/reports
```

---

## Reproducing this run

```bash
# Requires: Groq key, OpenRouter key (or equivalent)
coeval run --config Runs/special_ed_slp_v5.yaml

# Analyze after completion:
coeval analyze all \
    --run Runs/special-ed-slp-v5 \
    --out Runs/special-ed-slp-v5/reports
```
