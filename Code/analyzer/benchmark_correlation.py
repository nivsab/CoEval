"""Benchmark-grounded correlation analysis (EXP-001 keystone, Table 1).

Quantifies how well CoEval evaluator scores track each task's native metric
(BERTScore-F1 for summarization, BLEU-4 for code explanation), comparing the
cross-family ensemble against the best single judge.  Reads the per-response
benchmark scores written by :mod:`benchmark.score_responses` and joins them to
the CoEval analytical units by ``response_id``.

The reported statistic is Spearman rho (rank correlation) with a paired
bootstrap CI, per task and pooled.  A higher ensemble correlation than any
single judge demonstrates that multi-judge aggregation recovers the quality
signal the native metric encodes more faithfully than a single LLM judge.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .loader import load_ees, EESDataModel
from .stats import correlation_ci


def _load_response_benchmark_scores(run_dir: Path) -> dict[str, dict[str, Any]]:
    """response_id -> {score, task_id, metric} from the sidecar."""
    out: dict[str, dict[str, Any]] = {}
    sidecar = run_dir / "benchmark_response_scores.jsonl"
    if not sidecar.exists():
        return out
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rid = rec.get("response_id")
        if rid and rec.get("benchmark_native_score") is not None:
            out[rid] = {
                "score": float(rec["benchmark_native_score"]),
                "task_id": rec.get("task_id", ""),
                "metric": rec.get("metric", ""),
            }
    return out


def _coeval_scores_by_response(
    model: EESDataModel,
    judge_filter: set[str] | None = None,
    aspect_filter: set[str] | None = None,
) -> dict[str, float]:
    """response_id -> mean CoEval score over judges (in filter) and aspects.

    ``aspect_filter`` restricts to specific rubric dimensions.  When correlating
    with a ground-truth metric one should use the dimension that measures the
    same construct (e.g. the ``accuracy`` aspect against exact-match
    correctness); averaging in an off-target dimension such as ``relevance``
    attenuates the correlation.
    """
    acc: dict[str, list[float]] = defaultdict(list)
    for u in model.units:
        if judge_filter and u.judge_model_id not in judge_filter:
            continue
        if aspect_filter and u.rubric_aspect not in aspect_filter:
            continue
        acc[u.response_id].append(u.score_norm)
    return {rid: float(np.mean(v)) for rid, v in acc.items()}


def compute_benchmark_correlation(
    run_dir: str | Path,
    seed: int = 0,
    aspect_filter: set[str] | None = None,
) -> dict[str, Any]:
    """Compute Table 1: ensemble vs best-single-judge correlation with native metric.

    ``aspect_filter`` restricts the CoEval score to specific rubric dimensions
    (e.g. ``{"accuracy"}`` when the ground truth is answer correctness).
    """
    run_dir = Path(run_dir)
    model = load_ees(run_dir, partial_ok=True)
    bench = _load_response_benchmark_scores(run_dir)
    if not bench:
        raise FileNotFoundError(
            f"No benchmark_response_scores.jsonl in {run_dir}. "
            "Run `python -m benchmark.score_responses --run <dir>` first."
        )

    tasks = sorted({b["task_id"] for b in bench.values()})
    judges = sorted(model.judges)

    def _corr_for(coeval: dict[str, float], task: str | None) -> Any:
        xs, ys = [], []
        for rid, b in bench.items():
            if task and b["task_id"] != task:
                continue
            if rid in coeval:
                xs.append(coeval[rid])
                ys.append(b["score"])
        return correlation_ci(xs, ys, method="spearman", seed=seed)

    # Ensemble (all judges)
    ens = _coeval_scores_by_response(model, aspect_filter=aspect_filter)
    ensemble_row = {"method": "CoEval ensemble", "n_judges": len(judges)}
    for t in tasks:
        e = _corr_for(ens, t)
        ensemble_row[t] = {"rho": e.point, "lo": e.lo, "hi": e.hi, "n": e.n}
    e = _corr_for(ens, None)
    ensemble_row["overall"] = {"rho": e.point, "lo": e.lo, "hi": e.hi, "n": e.n}

    # Per-judge, to find the best single judge per task and overall
    per_judge: dict[str, dict[str, Any]] = {}
    for j in judges:
        jc = _coeval_scores_by_response(model, judge_filter={j}, aspect_filter=aspect_filter)
        row = {"method": f"single judge: {j}"}
        for t in tasks:
            e = _corr_for(jc, t)
            row[t] = {"rho": e.point, "lo": e.lo, "hi": e.hi, "n": e.n}
        e = _corr_for(jc, None)
        row["overall"] = {"rho": e.point, "lo": e.lo, "hi": e.hi, "n": e.n}
        per_judge[j] = row

    # Best single judge per column (max rho, ignoring NaN)
    def _best(col: str) -> dict[str, Any]:
        best_j, best_rho = None, -2.0
        for j, row in per_judge.items():
            rho = row[col]["rho"]
            if rho is not None and not np.isnan(rho) and rho > best_rho:
                best_rho, best_j = rho, j
        return {"judge": best_j, **(per_judge[best_j][col] if best_j else {})}

    best_single = {"method": "Best single judge"}
    for col in tasks + ["overall"]:
        best_single[col] = _best(col)

    # Fair baseline: the EXPECTED single judge (mean over full-coverage judges).
    # This is the correct comparison for an ensemble, since one does not know a
    # priori which single judge will be best; the post-hoc "best" judge is a
    # winner's-curse artifact. Restrict to judges that scored ~all responses so
    # the means are over a common item set.
    max_n = max((row["overall"]["n"] for row in per_judge.values()), default=0)
    full_cov = [j for j, row in per_judge.items()
                if row["overall"]["n"] >= 0.95 * max_n]
    mean_single = {"method": "Mean single judge", "judges": full_cov}
    for col in tasks + ["overall"]:
        vals = [per_judge[j][col]["rho"] for j in full_cov
                if per_judge[j][col]["rho"] is not None
                and not np.isnan(per_judge[j][col]["rho"])]
        mean_single[col] = {"rho": float(np.mean(vals)) if vals else float("nan")}

    return {
        "experiment": "benchmark-grounded correlation",
        "aspect_filter": sorted(aspect_filter) if aspect_filter else None,
        "run": str(run_dir),
        "tasks": tasks,
        "judges": judges,
        "metric_by_task": {t: next(b["metric"] for b in bench.values()
                                   if b["task_id"] == t) for t in tasks},
        "n_scored_responses": len(bench),
        "ensemble": ensemble_row,
        "per_judge": per_judge,
        "best_single_judge": best_single,
        "mean_single_judge": mean_single,
    }


def write_benchmark_correlation(run_dir: str | Path, out_dir: str | Path,
                                seed: int = 0,
                                aspect_filter: set[str] | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = compute_benchmark_correlation(run_dir, seed=seed, aspect_filter=aspect_filter)
    suffix = ""
    if aspect_filter:
        suffix = "_" + "+".join(sorted(aspect_filter))
    (out_dir / f"benchmark_correlation{suffix}.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")

    cols = res["tasks"] + ["overall"]
    with open(out_dir / f"benchmark_correlation{suffix}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method"] + cols)
        def _fmt(cell):
            r = cell.get("rho")
            return "—" if r is None or np.isnan(r) else f"{r:.3f} [{cell['lo']:.3f},{cell['hi']:.3f}]"
        w.writerow([res["ensemble"]["method"]] + [_fmt(res["ensemble"][c]) for c in cols])
        w.writerow([res["best_single_judge"]["method"]] +
                   [_fmt(res["best_single_judge"][c]) for c in cols])
        for j, row in res["per_judge"].items():
            w.writerow([row["method"]] + [_fmt(row[c]) for c in cols])
    return out_dir / f"benchmark_correlation{suffix}.json"
