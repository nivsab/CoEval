"""EXP-V2: controlled self-preference and the ranking flip it causes.

In the vertical case studies the panel includes gpt-4o-mini as BOTH a judge and a
student. On clinical_reasoning the gpt-4o-mini judge ranks gpt-4o-mini (itself) #1,
while both cross-family judges (claude-haiku, gemini-flash) rank gpt-3.5-turbo
first. We measure the self-preference as a *difference in differences* that
controls for a judge simply being harsher/softer overall:

  delta_same  = score[gpt-4o-mini JUDGE](gpt-4o-mini) - score[gpt-4o-mini JUDGE](gpt-3.5-turbo)
  delta_cross = score[cross-family]   (gpt-4o-mini) - score[cross-family]   (gpt-3.5-turbo)
  self_preference = delta_same - delta_cross   (> 0  => the in-family judge favors itself)

A single same-family judge therefore produces a self-serving ranking; the
vendor-disjoint ensemble removes it. Run: python scripts/v2_self_preference.py
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

RUN = ROOT / "Runs" / "EXP006-vertical-case-studies"
IN_FAMILY_JUDGE = "gpt-4o-mini"
SELF_STUDENT = "gpt-4o-mini"
RIVAL_STUDENT = "gpt-3.5-turbo"


def main():
    model = load_ees(RUN, partial_ok=True)
    # task -> judge -> student -> [scores]
    cell = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for u in model.units:
        cell[u.task_id][u.judge_model_id][u.student_model_id].append(u.score_norm)

    results = {}
    for task in sorted(cell):
        judges = cell[task]
        cross = [j for j in judges if j != IN_FAMILY_JUDGE]

        def mean_score(judge, student):
            v = judges[judge].get(student, [])
            return float(np.mean(v)) if v else float("nan")

        def cross_mean(student):
            vals = [mean_score(j, student) for j in cross]
            return float(np.mean([v for v in vals if v == v]))

        delta_same = mean_score(IN_FAMILY_JUDGE, SELF_STUDENT) - mean_score(IN_FAMILY_JUDGE, RIVAL_STUDENT)
        delta_cross = cross_mean(SELF_STUDENT) - cross_mean(RIVAL_STUDENT)
        self_pref = delta_same - delta_cross

        # rankings by each judge and by cross-family ensemble
        def ranking(scorer):
            students = sorted(judges[next(iter(judges))].keys())
            return sorted(students, key=lambda s: -scorer(s))
        same_rank = ranking(lambda s: mean_score(IN_FAMILY_JUDGE, s))
        cross_rank = ranking(cross_mean)

        results[task] = {
            "in_family_judge": IN_FAMILY_JUDGE,
            "score_in_family_judge": {SELF_STUDENT: round(mean_score(IN_FAMILY_JUDGE, SELF_STUDENT), 3),
                                      RIVAL_STUDENT: round(mean_score(IN_FAMILY_JUDGE, RIVAL_STUDENT), 3)},
            "score_cross_family": {SELF_STUDENT: round(cross_mean(SELF_STUDENT), 3),
                                   RIVAL_STUDENT: round(cross_mean(RIVAL_STUDENT), 3)},
            "delta_same": round(delta_same, 3),
            "delta_cross": round(delta_cross, 3),
            "self_preference": round(self_pref, 3),
            "ranking_same_family_judge": same_rank,
            "ranking_cross_family_ensemble": cross_rank,
            "ranking_flipped_by_self_preference": same_rank[:2] != cross_rank[:2],
        }
        print(f"\n[{task}]")
        print(f"  in-family judge scores: {SELF_STUDENT}={results[task]['score_in_family_judge'][SELF_STUDENT]} "
              f"{RIVAL_STUDENT}={results[task]['score_in_family_judge'][RIVAL_STUDENT]}")
        print(f"  cross-family scores:    {SELF_STUDENT}={results[task]['score_cross_family'][SELF_STUDENT]} "
              f"{RIVAL_STUDENT}={results[task]['score_cross_family'][RIVAL_STUDENT]}")
        print(f"  self-preference (diff-in-diff) = {self_pref:+.3f}")
        print(f"  same-family judge ranking : {same_rank}")
        print(f"  cross-family ranking      : {cross_rank}")
        print(f"  ranking flipped by self-preference: {results[task]['ranking_flipped_by_self_preference']}")

    out = {"experiment": "v2_self_preference", "run": str(RUN), "per_task": results}
    (RUN / "reports" / "v2_self_preference.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", RUN / "reports" / "v2_self_preference.json")


if __name__ == "__main__":
    main()
