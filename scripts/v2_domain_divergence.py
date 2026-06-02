"""EXP-011: a model's rank is DOMAIN-SPECIFIC.

CoEval ranks the same candidate models on several distinct domains (generated de
novo) and shows the ranking DIVERGES: the best model changes by domain, so a
single generic leaderboard misleads a practitioner whose application is one
domain. Cross-domain rank agreement is the Kendall tau between domain rankings.

Run:  python scripts/v2_domain_divergence.py
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


def main():
    model = load_ees(RUN, partial_ok=True)
    # per (task, student): mean ensemble score over judges + aspects
    acc = defaultdict(lambda: defaultdict(list))
    for u in model.units:
        acc[u.task_id][u.student_model_id].append(u.score_norm)
    tasks = sorted(acc)
    students = sorted({s for t in acc for s in acc[t]})
    score = {t: {s: float(np.mean(acc[t][s])) for s in acc[t]} for t in tasks}
    rankings = {t: [s for s, _ in sorted(score[t].items(), key=lambda kv: -kv[1])] for t in tasks}

    # cross-domain Kendall tau matrix (rank agreement)
    common = [s for s in students if all(s in score[t] for t in tasks)]
    tau = {}
    for i, a in enumerate(tasks):
        for b in tasks[i + 1:]:
            va = [score[a][s] for s in common]; vb = [score[b][s] for s in common]
            tau[f"{a} | {b}"] = round(float(kendalltau(va, vb).correlation), 3)
    taus = [v for v in tau.values()]

    # generic pooled leaderboard (average each model over all domains) and the
    # "regret" of following it: for each domain, the rank that the pooled-#1 model
    # actually holds, and how far the domain's true-best sits on the pooled board.
    pooled = {s: float(np.mean([score[t][s] for t in tasks if s in score[t]])) for s in common}
    pooled_order = [s for s, _ in sorted(pooled.items(), key=lambda kv: -kv[1])]
    pooled_top = pooled_order[0]
    regret = {
        t: {
            "domain_best": rankings[t][0],
            "domain_best_rank_on_pooled_board": pooled_order.index(rankings[t][0]) + 1,
            "pooled_top_rank_in_domain": rankings[t].index(pooled_top) + 1,
        }
        for t in tasks
    }

    out = {
        "experiment": "v2_domain_divergence",
        "tasks": tasks, "n_students": len(common),
        "pooled_leaderboard": [{"model": s, "score": round(pooled[s], 3)} for s in pooled_order],
        "pooled_top_model": pooled_top,
        "pooled_leaderboard_regret": regret,
        "per_domain_ranking": {t: [{"model": s, "score": round(score[t][s], 3)} for s in rankings[t]]
                               for t in tasks},
        "top_model_by_domain": {t: rankings[t][0] for t in tasks},
        "distinct_top_models": sorted(set(rankings[t][0] for t in tasks)),
        "cross_domain_kendall_tau": tau,
        "mean_cross_domain_tau": round(float(np.mean(taus)), 3) if taus else None,
        "min_cross_domain_tau": round(float(np.min(taus)), 3) if taus else None,
        # per-model rank position across domains (domain specialization)
        "model_rank_by_domain": {s: {t: rankings[t].index(s) + 1 for t in tasks if s in rankings[t]}
                                 for s in common},
    }
    (RUN / "reports").mkdir(exist_ok=True)
    (RUN / "reports" / "v2_domain_divergence.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
