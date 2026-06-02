"""EXP-011 robustness: is the cross-domain rank divergence real or score-band noise?

The per-domain mean scores sit in a tight band (0.78-0.92), so we must confirm the
ranking divergence survives item resampling. For each domain we bootstrap over its
items (B=2000), recompute the ensemble mean per model, and report:
  - P(model is #1 in domain)  -> is the nominal winner stable?
  - bootstrap CI on each cross-domain Kendall tau -> is low agreement significant?
  - P(tau < 0) for the anti-correlated pair (code vs math).

Run:  python scripts/v2_domain_divergence_bootstrap.py
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from analyzer.loader import load_ees  # noqa: E402

RUN = ROOT / "Runs" / "EXP011-domain-divergence"
B = 2000
SEED = 0


def main():
    model = load_ees(RUN, partial_ok=True)
    # per (task, item, student): mean ensemble score over judges + aspects
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for u in model.units:
        acc[u.task_id][u.datapoint_id][u.student_model_id].append(u.score_norm)
    tasks = sorted(acc)
    students = sorted({s for t in acc for it in acc[t] for s in acc[t][it]})

    # item-level score matrix per task: item_means[task][item][student]
    item_means = {}
    items_by_task = {}
    for t in tasks:
        items = sorted(acc[t])
        items_by_task[t] = items
        item_means[t] = {
            it: {s: float(np.mean(acc[t][it][s])) for s in students if s in acc[t][it]}
            for it in items
        }

    rng = np.random.default_rng(SEED)

    # bootstrap: per domain, resample items -> model mean -> ranking
    win_count = {t: defaultdict(int) for t in tasks}      # P(model #1)
    boot_rankings = {t: [] for t in tasks}                 # list of ranked-student-lists
    for t in tasks:
        items = items_by_task[t]
        n = len(items)
        # precompute per-item per-student vector (nan where missing)
        mat = np.array([[item_means[t][it].get(s, np.nan) for s in students] for it in items])
        for _ in range(B):
            idx = rng.integers(0, n, size=n)
            sub = mat[idx]
            means = np.nanmean(sub, axis=0)
            order = [students[i] for i in np.argsort(-means)]
            win_count[t][order[0]] += 1
            boot_rankings[t].append([students[i] for i in np.argsort(-means)])

    # cross-domain Kendall tau bootstrap CI (paired by same bootstrap draw index)
    common = students
    tau_boot = {}
    for i, a in enumerate(tasks):
        for b in tasks[i + 1:]:
            taus = []
            for d in range(B):
                ra = boot_rankings[a][d]; rb = boot_rankings[b][d]
                # rank vectors aligned by model
                pa = [ra.index(s) for s in common]
                pb = [rb.index(s) for s in common]
                taus.append(kendalltau(pa, pb).correlation)
            taus = np.array([x for x in taus if not np.isnan(x)])
            tau_boot[f"{a} | {b}"] = {
                "mean": round(float(np.mean(taus)), 3),
                "ci_lo": round(float(np.percentile(taus, 2.5)), 3),
                "ci_hi": round(float(np.percentile(taus, 97.5)), 3),
                "p_tau_lt_0": round(float((taus < 0).mean()), 3),
            }

    out = {
        "experiment": "v2_domain_divergence_bootstrap",
        "B": B, "n_students": len(students), "tasks": tasks,
        "p_model_is_top_by_domain": {
            t: {s: round(win_count[t][s] / B, 3) for s in students if win_count[t][s] > 0}
            for t in tasks
        },
        "cross_domain_kendall_tau_ci": tau_boot,
    }
    (RUN / "reports").mkdir(exist_ok=True)
    (RUN / "reports" / "v2_domain_divergence_bootstrap.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
