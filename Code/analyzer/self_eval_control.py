"""Self-evaluation confound control (zero new API calls).

When the same model pool supplies students and judges, a student that is also a
judge can inflate its own (or its family's) score. This module re-aggregates
student scores after excluding contaminated evaluation units and reports how
much the ranking changes, so the headline student ranking is interpretable.

Three exclusion policies:
  * ``self_judging``  — drop units where judge == student (a model grading itself).
  * ``self_teaching`` — drop units where teacher == student.
  * ``same_family``   — drop units where judge and student share a vendor family
    (e.g. both OpenAI), since shared-family error correlation is the subtler
    version of self-preference.

For each policy it returns the per-student mean score with a bootstrap CI, and
the Spearman rank correlation between the full and controlled rankings (a high
correlation means the confound does not change conclusions).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .loader import EESDataModel, AnalyticalUnit
from .stats import mean_ci, correlation_ci


# Vendor family inference from model id (best-effort, extend as needed).
_FAMILY_PREFIXES = {
    "gpt": "openai", "o1": "openai", "o3": "openai", "openai": "openai",
    "claude": "anthropic", "anthropic": "anthropic",
    "gemini": "google", "google": "google", "gemma": "google",
    "qwen": "qwen", "smollm": "huggingface", "llama": "meta", "deepseek": "deepseek",
    "mistral": "mistral", "mixtral": "mistral",
}


def model_family(model_id: str) -> str:
    m = model_id.lower()
    for prefix, fam in _FAMILY_PREFIXES.items():
        if prefix in m:
            return fam
    return "other"


def _student_means(units: list[AnalyticalUnit],
                   keep: Callable[[AnalyticalUnit], bool]) -> dict[str, list[float]]:
    """Per-student list of (per response-aspect) normalised scores passing ``keep``."""
    out: dict[str, list[float]] = defaultdict(list)
    for u in units:
        if keep(u):
            out[u.student_model_id].append(u.score_norm)
    return out


def compute_self_eval_control(model: EESDataModel, seed: int = 0) -> dict[str, Any]:
    """Compare full student ranking against self-judging / same-family controls."""
    students = sorted(model.students)

    policies: dict[str, Callable[[AnalyticalUnit], bool]] = {
        "full": lambda u: True,
        "exclude_self_judging": lambda u: u.judge_model_id != u.student_model_id,
        "exclude_self_teaching": lambda u: u.teacher_model_id != u.student_model_id,
        "exclude_same_family": lambda u: model_family(u.judge_model_id)
                                          != model_family(u.student_model_id),
    }

    per_policy: dict[str, dict[str, Any]] = {}
    full_rank_vector: list[float] = []
    for name, keep in policies.items():
        means = _student_means(model.units, keep)
        rows = {}
        for s in students:
            vals = means.get(s, [])
            est = mean_ci(vals, seed=seed)
            rows[s] = {"mean": est.point, "lo": est.lo, "hi": est.hi, "n": est.n}
        per_policy[name] = rows

    # Rank correlation of each control vs the full ranking
    full_means = [per_policy["full"][s]["mean"] for s in students]
    rank_stability = {}
    for name in policies:
        if name == "full":
            continue
        ctrl_means = [per_policy[name][s]["mean"] for s in students]
        est = correlation_ci(full_means, ctrl_means, method="spearman", seed=seed)
        # also the largest single-student score change
        deltas = {s: per_policy[name][s]["mean"] - per_policy["full"][s]["mean"]
                  for s in students
                  if not (np.isnan(per_policy[name][s]["mean"])
                          or np.isnan(per_policy["full"][s]["mean"]))}
        max_student = max(deltas, key=lambda k: abs(deltas[k])) if deltas else None
        rank_stability[name] = {
            "spearman_vs_full": est.point,
            "spearman_lo": est.lo, "spearman_hi": est.hi,
            "max_abs_delta_student": max_student,
            "max_abs_delta": deltas.get(max_student) if max_student else None,
        }

    return {
        "experiment": "self-evaluation control",
        "run": str(model.run_path),
        "students": students,
        "families": {s: model_family(s) for s in students},
        "per_policy": per_policy,
        "rank_stability": rank_stability,
    }


def write_self_eval_control(model: EESDataModel, out_dir: Path, seed: int = 0) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = compute_self_eval_control(model, seed=seed)
    (out_dir / "self_eval_control.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    with open(out_dir / "self_eval_control.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "student", "family", "mean", "lo", "hi", "n"])
        for policy, rows in res["per_policy"].items():
            for s, d in rows.items():
                w.writerow([policy, s, res["families"][s],
                            f"{d['mean']:.4f}", f"{d['lo']:.4f}", f"{d['hi']:.4f}", d["n"]])
    return out_dir / "self_eval_control.json"
