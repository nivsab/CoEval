"""EXP-V2: a good teacher generates DISCRIMINATIVE items.

A benchmark can only rank models if its items actually separate them. An item on
which every candidate scores alike contributes nothing to the ranking, no matter
how good the judges are. We measure per-item discriminativeness on the items
CoEval GENERATED for the three custom verticals (drug-interaction, clinical,
legal): the spread of candidate-model ensemble scores per item.

Discriminative item  = score range across students >= 0.15.
Decisive item        = the per-item student order agrees with the overall ranking.

Run:  python scripts/v2_item_discrimination.py
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from analyzer.loader import load_ees  # noqa: E402

RUNS = {
    "drug_interaction_reasoning": ROOT / "Runs" / "EXP007-ddi-vertical",
    "clinical_reasoning": ROOT / "Runs" / "EXP006-vertical-case-studies",
    "legal_analysis": ROOT / "Runs" / "EXP006-vertical-case-studies",
}
THRESH = 0.15


def main():
    cache = {}
    out = {"experiment": "v2_item_discrimination", "threshold_range": THRESH, "verticals": {}}
    for task, run in RUNS.items():
        if run not in cache:
            cache[run] = load_ees(run, partial_ok=True)
        model = cache[run]
        # per (item, student): mean ensemble score over judges + aspects
        acc = defaultdict(lambda: defaultdict(list))
        for u in model.units:
            if u.task_id == task:
                acc[u.datapoint_id][u.student_model_id].append(u.score_norm)
        items = sorted(acc)
        students = sorted({s for it in acc for s in acc[it]})
        score = {it: {s: float(np.mean(acc[it][s])) for s in acc[it]} for it in items}
        # overall ranking (mean score per student, descending)
        overall = {s: float(np.mean([score[it][s] for it in items if s in score[it]])) for s in students}
        order = [s for s, _ in sorted(overall.items(), key=lambda kv: -kv[1])]

        ranges, decisive = [], 0
        for it in items:
            vals = [score[it][s] for s in students if s in score[it]]
            if len(vals) < 2:
                continue
            rng = max(vals) - min(vals)
            ranges.append(rng)
            # decisive: this item's top student == overall top student
            item_top = max(score[it].items(), key=lambda kv: kv[1])[0]
            if item_top == order[0]:
                decisive += 1
        ranges = np.array(ranges)
        out["verticals"][task] = {
            "n_items": len(ranges),
            "students_by_overall": order,
            "overall_scores": {s: round(overall[s], 3) for s in order},
            "separability_top_minus_bottom": round(overall[order[0]] - overall[order[-1]], 3),
            "mean_item_range": round(float(ranges.mean()), 3),
            "median_item_range": round(float(np.median(ranges)), 3),
            "pct_discriminative": round(100 * float((ranges >= THRESH).mean()), 1),
            "pct_decisive_top": round(100 * decisive / len(ranges), 1),
        }
    # pooled headline
    allr = []
    for task, run in RUNS.items():
        model = cache[run]
        acc = defaultdict(lambda: defaultdict(list))
        for u in model.units:
            if u.task_id == task:
                acc[u.datapoint_id][u.student_model_id].append(u.score_norm)
        for it in acc:
            vals = [float(np.mean(v)) for v in acc[it].values()]
            if len(vals) >= 2:
                allr.append(max(vals) - min(vals))
    allr = np.array(allr)
    out["pooled"] = {
        "n_items": len(allr),
        "mean_item_range": round(float(allr.mean()), 3),
        "pct_discriminative": round(100 * float((allr >= THRESH).mean()), 1),
    }
    (ROOT / "Runs" / "EXP006-vertical-case-studies" / "reports" / "v2_item_discrimination.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
