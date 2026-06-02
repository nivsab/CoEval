"""EXP-V2 item B+D: judge-choice regret and reliability-weighted aggregation.

On the summarization task (EXP001, scored against a BERTScore reference) single
judges range from anti-correlated to moderately correlated, and the best judge is
task-dependent, so choosing ONE judge is a high-variance gamble. We quantify the
regret and test whether unsupervised reliability weighting (weight each judge by
its agreement with the rest of the panel -- no ground truth used) beats the plain
mean and recovers the downside that a bad single judge would suffer.

Run:  python scripts/v2_aggregation_regret.py
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from analyzer.loader import load_ees  # noqa: E402

RUN = ROOT / "Runs" / "EXP001-benchmark-grounded-comparison"


def _bench_scores():
    out = {}
    f = RUN / "benchmark_response_scores.jsonl"
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("metric") == "bertscore" and r.get("benchmark_native_score") is not None:
            out[r["response_id"]] = float(r["benchmark_native_score"])
    return out


def main():
    bench = _bench_scores()
    model = load_ees(RUN, partial_ok=True)
    # response -> judge -> mean(aspect score)
    perj = defaultdict(lambda: defaultdict(list))
    for u in model.units:
        if u.response_id in bench:
            perj[u.response_id][u.judge_model_id].append(u.score_norm)
    rids = [r for r in perj if r in bench]
    judges = sorted({j for r in rids for j in perj[r]})
    # matrix S[response][judge] = mean aspect score (NaN if judge missing)
    S = {r: {j: (float(np.mean(perj[r][j])) if perj[r].get(j) else np.nan) for j in judges} for r in rids}
    y = np.array([bench[r] for r in rids])

    def corr(vec):
        v = np.array(vec)
        ok = ~np.isnan(v)
        return float(spearmanr(v[ok], y[ok]).correlation)

    # single judges
    single = {j: corr([S[r][j] for r in rids]) for j in judges}
    best_j = max(single, key=single.get)
    worst_j = min(single, key=single.get)

    def agg(fn):
        return [fn([S[r][j] for j in judges if not np.isnan(S[r][j])]) for r in rids]

    mean_rho = corr(agg(np.mean))
    median_rho = corr(agg(np.median))

    def trimmed(vals):
        vals = sorted(vals)
        return np.mean(vals[1:-1]) if len(vals) > 2 else np.mean(vals)
    trim_rho = corr(agg(trimmed))

    # unsupervised reliability weights: each judge's mean Spearman agreement with
    # the OTHER judges across responses (leave-one-out panel agreement).
    rel = {}
    for j in judges:
        others = [o for o in judges if o != j]
        xj = np.array([S[r][j] for r in rids])
        agree = []
        for o in others:
            xo = np.array([S[r][o] for r in rids])
            ok = ~np.isnan(xj) & ~np.isnan(xo)
            if ok.sum() > 5 and np.std(xj[ok]) > 0 and np.std(xo[ok]) > 0:
                agree.append(spearmanr(xj[ok], xo[ok]).correlation)
        rel[j] = max(0.0, float(np.mean(agree))) if agree else 0.0
    wsum = sum(rel.values()) or 1.0
    w = {j: rel[j] / wsum for j in judges}
    wvec = []
    for r in rids:
        num = sum(w[j] * S[r][j] for j in judges if not np.isnan(S[r][j]))
        den = sum(w[j] for j in judges if not np.isnan(S[r][j])) or 1.0
        wvec.append(num / den)
    weighted_rho = corr(wvec)

    out = {
        "experiment": "v2_aggregation_regret", "task": "summarization vs BERTScore",
        "n_responses": len(rids), "judges": judges,
        "single_judge_rho": {j: round(single[j], 4) for j in judges},
        "best_single": {best_j: round(single[best_j], 4)},
        "worst_single": {worst_j: round(single[worst_j], 4)},
        "judge_choice_regret_range": round(single[best_j] - single[worst_j], 4),
        "reliability_weights": {j: round(w[j], 3) for j in judges},
        "aggregations": {
            "mean (current)": round(mean_rho, 4),
            "median": round(median_rho, 4),
            "trimmed_mean": round(trim_rho, 4),
            "reliability_weighted": round(weighted_rho, 4),
        },
    }
    (RUN / "reports" / "v2_aggregation_regret.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
