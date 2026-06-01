"""EXP-004: verbosity-bias analysis.

Measures the correlation between student response length (``token_count`` from
Phase 4) and the normalised judge score, per judge and for the ensemble.  A
positive correlation indicates the judge rewards longer responses irrespective
of quality; the ensemble correlation is expected to be smaller in magnitude if
idiosyncratic per-judge length preferences partially cancel.

Pure re-analysis of existing Phase 4 + Phase 5 data (no API calls).  To avoid
inflating the sample by the number of rubric aspects (each response has one
token count but many aspect scores), scores are aggregated to one mean score
per (response, judge) before correlating with the response length.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .loader import EESDataModel
from .stats import correlation_ci, Estimate
from .experiment_reports import write_page, table_html


_TIKTOKEN_ENC = None


def _tiktoken_len(text: str) -> int | None:
    """Token length via cl100k_base, cached.  Returns None if tiktoken absent."""
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is None:
        try:
            import tiktoken
            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TIKTOKEN_ENC = False
    if _TIKTOKEN_ENC is False:
        return None
    return len(_TIKTOKEN_ENC.encode(text))


def _response_token_count(model: EESDataModel, response_id: str) -> float | None:
    """Length of a response in tokens.

    Prefers the stored ``token_count`` (Phase 4 schema field); for older runs
    that predate that field, recomputes from the raw ``response`` text using the
    cl100k_base tokenizer, falling back to a whitespace word count.
    """
    rec = model.responses.get(response_id)
    if rec is None:
        return None
    tc = rec.get("token_count")
    if tc is not None:
        try:
            return float(tc)
        except (ValueError, TypeError):
            pass
    text = rec.get("response")
    if not isinstance(text, str) or not text:
        return None
    n = _tiktoken_len(text)
    if n is None:
        n = len(text.split())
    return float(n)


def compute_verbosity_correlation(
    model: EESDataModel,
    judges: list[str] | None = None,
    tasks: list[str] | None = None,
    method: str = "pearson",
    seed: int = 0,
) -> dict[str, Any]:
    """Compute per-judge and ensemble length-vs-score correlation.

    Returns a JSON-serialisable dict with per (judge, task) and pooled rows,
    plus the ensemble correlation and the verbosity-bias delta
    (mean individual-judge correlation minus ensemble correlation).
    """
    judges = judges or list(model.judges)
    present = {u.judge_model_id for u in model.units}
    judges = [j for j in judges if j in present]
    task_filter = set(tasks) if tasks else None

    # (judge, task, response) -> list of aspect scores  => mean = response score
    bucket: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for u in model.units:
        if u.judge_model_id not in judges:
            continue
        if task_filter and u.task_id not in task_filter:
            continue
        bucket[(u.judge_model_id, u.task_id, u.response_id)].append(u.score_norm)

    # ensemble: (task, response) -> {judge: mean_score}
    ens: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (j, t, rid), scs in bucket.items():
        ens[(t, rid)][j] = float(np.mean(scs))

    tasks_out = sorted({t for (_, t, _) in bucket}) if not tasks else tasks

    def _corr(pairs: list[tuple[float, float]]) -> Estimate:
        if len(pairs) < 2:
            return Estimate(float("nan"), float("nan"), float("nan"), len(pairs))
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        return correlation_ci(xs, ys, method=method, seed=seed)

    per_judge_task: list[dict[str, Any]] = []
    judge_pooled_corr: dict[str, Estimate] = {}
    for j in judges:
        pooled_pairs: list[tuple[float, float]] = []
        for t in tasks_out:
            pairs = []
            for (jj, tt, rid), scs in bucket.items():
                if jj != j or tt != t:
                    continue
                tc = _response_token_count(model, rid)
                if tc is None:
                    continue
                pairs.append((tc, float(np.mean(scs))))
            est = _corr(pairs)
            pooled_pairs.extend(pairs)
            per_judge_task.append({
                "judge": j, "task": t, "method": method,
                "corr": est.point, "lo": est.lo, "hi": est.hi, "n": est.n,
            })
        judge_pooled_corr[j] = _corr(pooled_pairs)

    # ensemble pooled + per task
    ensemble_rows: list[dict[str, Any]] = []
    ens_pooled_pairs: list[tuple[float, float]] = []
    for t in tasks_out:
        pairs = []
        for (tt, rid), jd in ens.items():
            if tt != t:
                continue
            tc = _response_token_count(model, rid)
            if tc is None:
                continue
            pairs.append((tc, float(np.mean(list(jd.values())))))
        est = _corr(pairs)
        ens_pooled_pairs.extend(pairs)
        ensemble_rows.append({
            "task": t, "method": method,
            "corr": est.point, "lo": est.lo, "hi": est.hi, "n": est.n,
        })
    ensemble_pooled = _corr(ens_pooled_pairs)

    indiv_points = [judge_pooled_corr[j].point for j in judges
                    if not np.isnan(judge_pooled_corr[j].point)]
    mean_indiv = float(np.mean(indiv_points)) if indiv_points else float("nan")
    # Individual judges carry mixed-sign idiosyncratic biases that partly cancel
    # in the mean, so the magnitude (mean |r|) is the honest measure of how much
    # length-bias each judge carries before ensembling.
    mean_abs_indiv = float(np.mean(np.abs(indiv_points))) if indiv_points else float("nan")
    delta = (mean_indiv - ensemble_pooled.point
             if not (np.isnan(mean_indiv) or np.isnan(ensemble_pooled.point))
             else float("nan"))
    abs_reduction = (mean_abs_indiv - abs(ensemble_pooled.point)
                     if not (np.isnan(mean_abs_indiv) or np.isnan(ensemble_pooled.point))
                     else float("nan"))

    return {
        "experiment": "EXP-004 verbosity-bias",
        "run": str(model.run_path),
        "method": method,
        "judges": judges,
        "tasks": tasks_out,
        "per_judge_task": per_judge_task,
        "judge_pooled": {j: judge_pooled_corr[j].as_dict() for j in judges},
        "ensemble_per_task": ensemble_rows,
        "ensemble_pooled": ensemble_pooled.as_dict(),
        "mean_individual_corr": mean_indiv,
        "mean_abs_individual_corr": mean_abs_indiv,
        "ensemble_corr": ensemble_pooled.point,
        "verbosity_bias_delta": delta,
        "abs_bias_reduction": abs_reduction,
    }


def write_verbosity_bias(
    model: EESDataModel,
    out_dir: Path,
    judges: list[str] | None = None,
    tasks: list[str] | None = None,
    method: str = "pearson",
    shared_plotly: Path | None = None,
    seed: int = 0,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = compute_verbosity_correlation(
        model, judges=judges, tasks=tasks, method=method, seed=seed)

    (out_dir / "verbosity_bias.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    with open(out_dir / "verbosity_bias.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "judge", "task", "method", "corr", "lo", "hi", "n"])
        for r in result["per_judge_task"]:
            w.writerow(["judge_task", r["judge"], r["task"], r["method"],
                        f"{r['corr']:.4f}", f"{r['lo']:.4f}", f"{r['hi']:.4f}", r["n"]])
        for j, d in result["judge_pooled"].items():
            w.writerow(["judge_pooled", j, "ALL", method,
                        f"{d['point']:.4f}", f"{d['lo']:.4f}", f"{d['hi']:.4f}", d["n"]])
        for r in result["ensemble_per_task"]:
            w.writerow(["ensemble_task", "ENSEMBLE", r["task"], method,
                        f"{r['corr']:.4f}", f"{r['lo']:.4f}", f"{r['hi']:.4f}", r["n"]])
        d = result["ensemble_pooled"]
        w.writerow(["ensemble_pooled", "ENSEMBLE", "ALL", method,
                    f"{d['point']:.4f}", f"{d['lo']:.4f}", f"{d['hi']:.4f}", d["n"]])

    # Bar chart: pooled correlation per judge + ensemble
    names = list(result["judges"]) + ["ENSEMBLE"]
    vals = [result["judge_pooled"][j]["point"] for j in result["judges"]] + \
           [result["ensemble_pooled"]["point"]]
    los = [result["judge_pooled"][j]["lo"] for j in result["judges"]] + \
          [result["ensemble_pooled"]["lo"]]
    his = [result["judge_pooled"][j]["hi"] for j in result["judges"]] + \
          [result["ensemble_pooled"]["hi"]]
    colors = ["#2b6cb0"] * len(result["judges"]) + ["#2f855a"]
    fig = {
        "div": "fig_verb",
        "data": [{
            "x": names, "y": vals, "type": "bar", "marker": {"color": colors},
            "error_y": {"type": "data", "symmetric": False,
                        "array": [h - v for h, v in zip(his, vals)],
                        "arrayminus": [v - l for v, l in zip(vals, los)]},
        }],
        "layout": {"title": f"Length vs score correlation ({method}) - pooled",
                   "yaxis": {"title": f"{method} r (token_count vs score)"},
                   "shapes": [{"type": "line", "x0": -0.5, "x1": len(names) - 0.5,
                               "y0": 0, "y1": 0, "line": {"color": "#999", "width": 1}}]},
    }

    rows = [[j, f"{result['judge_pooled'][j]['point']:.3f} "
             f"[{result['judge_pooled'][j]['lo']:.3f}, {result['judge_pooled'][j]['hi']:.3f}]",
             result['judge_pooled'][j]['n']] for j in result["judges"]]
    rows.append(["ENSEMBLE", f"{result['ensemble_pooled']['point']:.3f} "
                 f"[{result['ensemble_pooled']['lo']:.3f}, {result['ensemble_pooled']['hi']:.3f}]",
                 result['ensemble_pooled']['n']])
    tbl = table_html(["judge", f"pooled {method} r (95% CI)", "n responses"], rows)

    cards = [
        {"heading": "Verbosity bias: length vs score",
         "html": '<div id="fig_verb" class="fig"></div>'
                 f'<p class="note">Mean individual-judge r = '
                 f'{result["mean_individual_corr"]:.3f}; ensemble r = '
                 f'{result["ensemble_corr"]:.3f}; bias-reduction delta = '
                 f'{result["verbosity_bias_delta"]:.3f}. Positive r means longer '
                 f'responses score higher.</p>'},
        {"heading": "Pooled correlation per judge", "html": tbl},
    ]
    return write_page(
        out_dir,
        title="CoEval EXP-004: Verbosity-Bias Analysis",
        subtitle=f"{result['run']} | {method} correlation",
        cards=cards,
        figures=[fig],
    )
