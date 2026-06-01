"""EXP-002: ensemble-size ablation.

Quantifies how multi-judge aggregation improves evaluation reliability as the
panel grows from 1 to K judges.  Operates purely on existing Phase 5 data (no
new API calls): for each prefix of the judge-addition order it computes

  * ICC(3,k) and ICC(3,1) of the k-judge panel over the common item set
    (the average-measures ICC rises with k by the Spearman-Brown relation;
    this is the quantitative form of the Condorcet jury intuition);
  * Spearman rho between the k-judge ensemble score and the full K-judge
    ensemble, with a paired bootstrap CI (the convergence curve);
  * the dispersion (variance) of the ensemble score across items, indicating
    how much discriminative signal the ensemble retains.

The analytical item is a (response_id, rubric_aspect) pair; the ensemble score
for an item under a judge subset is the mean of those judges' normalised scores.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .loader import EESDataModel
from .stats import icc, correlation_ci, Estimate
from .experiment_reports import write_page, table_html


ItemKey = tuple[str, str]  # (response_id, rubric_aspect)


def _item_judge_scores(
    model: EESDataModel,
    tasks: list[str] | None,
) -> dict[ItemKey, dict[str, float]]:
    """Map each (response, aspect) item to {judge_id: score_norm}."""
    table: dict[ItemKey, dict[str, float]] = defaultdict(dict)
    task_filter = set(tasks) if tasks else None
    for u in model.units:
        if task_filter and u.task_id not in task_filter:
            continue
        table[(u.response_id, u.rubric_aspect)][u.judge_model_id] = u.score_norm
    return table


def _ensemble_vector(
    items: list[ItemKey],
    scores: dict[ItemKey, dict[str, float]],
    judges: list[str],
) -> np.ndarray:
    """Mean score over ``judges`` per item; NaN where no judge in subset scored it."""
    out = np.full(len(items), np.nan)
    for i, key in enumerate(items):
        vals = [scores[key][j] for j in judges if j in scores[key]]
        if vals:
            out[i] = float(np.mean(vals))
    return out


def compute_ensemble_ablation(
    model: EESDataModel,
    judges: list[str] | None = None,
    tasks: list[str] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute the ablation curve.  Returns a JSON-serialisable result dict."""
    judges = judges or list(model.judges)
    # keep only judges that actually appear in the data
    present = {u.judge_model_id for u in model.units}
    judges = [j for j in judges if j in present]
    if len(judges) < 2:
        raise ValueError(
            f"Ensemble ablation needs >= 2 judges with data; found {judges}"
        )

    scores = _item_judge_scores(model, tasks)
    # full-panel complete-case items (scored by all K judges)
    K = len(judges)
    full_items = [k for k, d in scores.items() if all(j in d for j in judges)]
    full_items.sort()
    if not full_items:
        raise ValueError(
            "No item was scored by all judges; cannot build a balanced panel. "
            "Check that the judge set is consistent across responses."
        )

    full_vec = _ensemble_vector(full_items, scores, judges)

    per_k: list[dict[str, Any]] = []
    for k in range(1, K + 1):
        subset = judges[:k]
        # balanced matrix over the full common item set: items x subset judges
        matrix = np.array(
            [[scores[key][j] for j in subset] for key in full_items],
            dtype=float,
        )
        icc_res = icc(matrix)
        k_vec = _ensemble_vector(full_items, scores, subset)
        conv: Estimate = correlation_ci(
            k_vec, full_vec, method="spearman", seed=seed
        )
        per_k.append({
            "k": k,
            "judges": list(subset),
            "icc_3_1": icc_res.icc_3_1,
            "icc_3_k": icc_res.icc_3_k,
            "spearman_vs_full": conv.point,
            "spearman_lo": conv.lo,
            "spearman_hi": conv.hi,
            "ensemble_score_var": float(np.nanvar(k_vec)),
            "n_items": int(np.sum(~np.isnan(k_vec))),
        })

    return {
        "experiment": "EXP-002 ensemble-size-ablation",
        "run": str(model.run_path),
        "judge_order": judges,
        "n_items_balanced": len(full_items),
        "tasks": tasks or model.tasks,
        "per_k": per_k,
    }


def write_ensemble_ablation(
    model: EESDataModel,
    out_dir: Path,
    judges: list[str] | None = None,
    tasks: list[str] | None = None,
    shared_plotly: Path | None = None,
    seed: int = 0,
) -> Path:
    """Compute the ablation and write JSON + CSV + a self-contained HTML page."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = compute_ensemble_ablation(model, judges=judges, tasks=tasks, seed=seed)
    per_k = result["per_k"]

    (out_dir / "ensemble_ablation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with open(out_dir / "ensemble_ablation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "judges", "icc_3_1", "icc_3_k",
                    "spearman_vs_full", "spearman_lo", "spearman_hi",
                    "ensemble_score_var", "n_items"])
        for r in per_k:
            w.writerow([r["k"], "+".join(r["judges"]),
                        f"{r['icc_3_1']:.4f}", f"{r['icc_3_k']:.4f}",
                        f"{r['spearman_vs_full']:.4f}", f"{r['spearman_lo']:.4f}",
                        f"{r['spearman_hi']:.4f}", f"{r['ensemble_score_var']:.5f}",
                        r["n_items"]])

    ks = [r["k"] for r in per_k]
    fig_icc = {
        "div": "fig_icc",
        "data": [{
            "x": ks, "y": [r["icc_3_k"] for r in per_k],
            "type": "scatter", "mode": "lines+markers", "name": "ICC(3,k) average-measures",
            "line": {"color": "#2b6cb0", "width": 3},
        }, {
            "x": ks, "y": [r["icc_3_1"] for r in per_k],
            "type": "scatter", "mode": "lines+markers", "name": "ICC(3,1) single-rater",
            "line": {"color": "#a0aec0", "width": 2, "dash": "dot"},
        }],
        "layout": {"title": "Inter-rater reliability vs ensemble size",
                   "xaxis": {"title": "judges in ensemble (k)", "dtick": 1},
                   "yaxis": {"title": "ICC", "range": [0, 1]}},
    }
    fig_conv = {
        "div": "fig_conv",
        "data": [{
            "x": ks, "y": [r["spearman_vs_full"] for r in per_k],
            "error_y": {
                "type": "data", "symmetric": False,
                "array": [r["spearman_hi"] - r["spearman_vs_full"] for r in per_k],
                "arrayminus": [r["spearman_vs_full"] - r["spearman_lo"] for r in per_k],
            },
            "type": "scatter", "mode": "lines+markers", "name": "Spearman vs full panel",
            "line": {"color": "#2f855a", "width": 3},
        }],
        "layout": {"title": "Convergence of k-judge ensemble to full panel (95% bootstrap CI)",
                   "xaxis": {"title": "judges in ensemble (k)", "dtick": 1},
                   "yaxis": {"title": "Spearman rho", "range": [0, 1.02]}},
    }

    rows = [[r["k"], "+".join(r["judges"]), f"{r['icc_3_1']:.3f}", f"{r['icc_3_k']:.3f}",
             f"{r['spearman_vs_full']:.3f} [{r['spearman_lo']:.3f}, {r['spearman_hi']:.3f}]",
             f"{r['ensemble_score_var']:.4f}", r["n_items"]] for r in per_k]
    tbl = table_html(
        ["k", "judge panel", "ICC(3,1)", "ICC(3,k)", "Spearman vs full (95% CI)",
         "ensemble var", "n items"], rows)

    cards = [
        {"heading": "Reliability vs ensemble size",
         "html": '<div id="fig_icc" class="fig"></div>'
                 '<p class="note">ICC(3,k) is the reliability of the averaged k-judge '
                 'score; its rise with k is the Spearman-Brown effect of aggregation. '
                 'ICC(3,1) (single-rater) stays roughly flat.</p>'},
        {"heading": "Convergence to the full panel",
         "html": '<div id="fig_conv" class="fig"></div>'},
        {"heading": "Per-k summary",
         "html": tbl + f'<p class="note">Balanced item set: '
                 f'{result["n_items_balanced"]} (response, aspect) pairs scored by all '
                 f'{len(result["judge_order"])} judges. Judge addition order: '
                 f'<code>{" -> ".join(result["judge_order"])}</code>.</p>'},
    ]
    return write_page(
        out_dir,
        title="CoEval EXP-002: Ensemble-Size Ablation",
        subtitle=f"{result['run']} | {result['n_items_balanced']} balanced items",
        cards=cards,
        figures=[fig_icc, fig_conv],
    )
