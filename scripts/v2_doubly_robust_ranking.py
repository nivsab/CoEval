"""EXP-V2: doubly-robust ranking = weight items by teacher discriminative power
AND judges by peer-response similarity.

A model's rank should lean on (a) items that actually separate models (a
saturated item carries no ranking signal) and (b) judges that agree with the
panel (a rogue judge is noise). Both weights are LABEL-FREE:
  item weight   d_i = variance of candidate-model scores on item i   (discrimination)
  judge weight  w_j = mean leave-one-out agreement of judge j with the panel.

We test whether the doubly-robust student ranking recovers the gold accuracy
ranking better and more robustly than the plain mean, on EXP010 (7 models,
sciq+ARC, gold = benchmark-native score). A bad (random) judge is injected to
test robustness.

Run:  python scripts/v2_doubly_robust_ranking.py
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from analyzer.loader import load_ees  # noqa: E402

# Optional CLI arg = run-folder name (defaults to the 7-model pilot for back-compat).
_RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else "EXP010-scale-ranking-pilot"
RUN = ROOT / "Runs" / _RUN_NAME


def _gold_by_student_item():
    out = {}
    f = RUN / "benchmark_response_scores.jsonl"
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("benchmark_native_score") is not None:
            out[(r["datapoint_id"], r["student_model_id"])] = float(r["benchmark_native_score"])
    return out


def main():
    model = load_ees(RUN, partial_ok=True)
    gold_si = _gold_by_student_item()
    # CoEval score per (item, student, judge) = mean over accuracy-ish aspects
    cell = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for u in model.units:
        cell[u.datapoint_id][u.student_model_id][u.judge_model_id].append(u.score_norm)
    items = sorted(cell)
    students = sorted({s for it in cell for s in cell[it]})
    judges = sorted({j for it in cell for s in cell[it] for j in cell[it][s]})

    # S[item][student][judge] = mean aspect score
    S = {it: {s: {j: float(np.mean(cell[it][s][j])) for j in cell[it][s]}
              for s in cell[it]} for it in items}

    # Inject a deliberately broken (random) judge for the robustness test.
    rng = np.random.default_rng(0)
    for it in items:
        for s in S[it]:
            S[it][s]["BAD:random"] = float(rng.uniform(0, 1))
    judges_bad = judges + ["BAD:random"]

    def judge_weights(panel):
        # leave-one-out mean agreement across items, per judge (label-free)
        vec = {j: np.array([np.mean([S[it][s][j] for s in students if j in S[it][s]])
                            for it in items]) for j in panel}
        w = {}
        for j in panel:
            ags = []
            for o in panel:
                if o == j:
                    continue
                a, b = vec[j], vec[o]
                ok = ~np.isnan(a) & ~np.isnan(b)
                if ok.sum() > 3 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
                    ags.append(spearmanr(a[ok], b[ok]).correlation)
            w[j] = max(0.0, float(np.mean(ags))) if ags else 0.0
        tot = sum(w.values()) or 1.0
        return {j: w[j] / tot for j in panel}

    def coeval_item_student(panel, jw):
        # per (item, student): judge-weighted mean score
        out = {}
        for it in items:
            out[it] = {}
            for s in students:
                num = sum(jw[j] * S[it][s][j] for j in panel if j in S[it][s])
                den = sum(jw[j] for j in panel if j in S[it][s]) or 1.0
                out[it][s] = num / den
        return out

    def item_discrimination(cs):
        # d_i = variance of candidate-model scores on item i (label-free)
        d = {}
        for it in items:
            vals = [cs[it][s] for s in students if s in cs[it]]
            d[it] = float(np.var(vals)) if len(vals) > 1 else 0.0
        tot = sum(d.values()) or 1.0
        return {it: d[it] / tot for it in items}

    def rank(cs, iw=None):
        # student score = (item-weighted) mean over items
        sc = {}
        for s in students:
            if iw is None:
                sc[s] = float(np.mean([cs[it][s] for it in items if s in cs[it]]))
            else:
                num = sum(iw[it] * cs[it][s] for it in items if s in cs[it])
                den = sum(iw[it] for it in items if s in cs[it]) or 1.0
                sc[s] = num / den
        return sc

    gold = {s: float(np.mean([gold_si[(it, s)] for it in items if (it, s) in gold_si])) for s in students}

    def recover(sc):
        xs = [sc[s] for s in students]; ys = [gold[s] for s in students]
        return round(float(spearmanr(xs, ys).correlation), 3), round(float(kendalltau(xs, ys).correlation), 3)

    # --- clean panel ---
    uniform = {j: 1.0 / len(judges) for j in judges}
    cs_uni = coeval_item_student(judges, uniform)
    jw = judge_weights(judges)
    cs_jw = coeval_item_student(judges, jw)
    iw_uni = item_discrimination(cs_uni)
    iw_jw = item_discrimination(cs_jw)

    res = {
        "experiment": "v2_doubly_robust_ranking",
        "n_items": len(items), "n_students": len(students), "judges": judges,
        "clean_panel": {
            "plain_mean": recover(rank(cs_uni)),
            "item_weighted_only": recover(rank(cs_uni, iw_uni)),
            "judge_weighted_only": recover(rank(cs_jw)),
            "doubly_robust": recover(rank(cs_jw, iw_jw)),
        },
    }

    # --- with a broken judge injected ---
    uni_b = {j: 1.0 / len(judges_bad) for j in judges_bad}
    cs_uni_b = coeval_item_student(judges_bad, uni_b)
    jw_b = judge_weights(judges_bad)
    cs_jw_b = coeval_item_student(judges_bad, jw_b)
    iw_jw_b = item_discrimination(cs_jw_b)
    res["with_broken_judge"] = {
        "plain_mean": recover(rank(cs_uni_b)),
        "doubly_robust": recover(rank(cs_jw_b, iw_jw_b)),
        "broken_judge_weight": round(jw_b.get("BAD:random", 0.0), 4),
    }
    # discrimination spread (sanity): how uneven are item weights?
    dvals = np.array(list(iw_jw.values()))
    res["item_weight_concentration"] = {
        "top10pct_share": round(float(np.sort(dvals)[::-1][:max(1, len(dvals)//10)].sum()), 3),
        "frac_items_near_zero": round(float((dvals < (0.2 / len(items))).mean()), 3),
    }

    (RUN / "reports" / "v2_doubly_robust_ranking.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
