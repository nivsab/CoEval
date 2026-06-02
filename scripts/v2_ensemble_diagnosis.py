"""EXP-V2: the panel as a self-validating instrument.

Two claims about WHY a multi-judge panel matters, tested on the EXP001
benchmark-grounded set (gold = benchmark-native score, 3 tasks pooled):

  (1) SELF-DIAGNOSIS: a judge's unsupervised agreement with the rest of the panel
      (leave-one-out mean Spearman) predicts its actual accuracy vs ground truth.
      So the panel identifies reliable vs unreliable judges WITHOUT labels.

  (2) OUTLIER ROBUSTNESS + DOWN-WEIGHTING: injecting deliberately broken judges
      (random, anti-correlated, constant) barely moves the plain-mean consensus,
      and reliability weighting drives their weight toward zero, recovering the
      clean-panel accuracy.

Run:  python scripts/v2_ensemble_diagnosis.py
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
from analyzer.benchmark_correlation import _load_response_benchmark_scores  # noqa: E402

RUN = ROOT / "Runs" / "EXP001-benchmark-grounded-comparison"


def _rho(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 5 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).correlation)


def main():
    model = load_ees(RUN, partial_ok=True)
    bench = _load_response_benchmark_scores(RUN)
    raw = defaultdict(lambda: defaultdict(list))
    for u in model.units:
        if u.response_id in bench:
            raw[u.response_id][u.judge_model_id].append(u.score_norm)
    rids = [r for r in raw if r in bench]
    real_judges = sorted({j for r in rids for j in raw[r]})
    gold = np.array([bench[r]["score"] for r in rids])

    # Per-response, per-judge mean score matrix.
    S = {j: np.array([np.mean(raw[r][j]) if raw[r].get(j) else np.nan for r in rids])
         for j in real_judges}

    # --- Inject controlled BAD judges (no peeking at gold for random/constant) ---
    rng = np.random.default_rng(0)
    base = S[real_judges[0]]
    S["BAD:random"] = rng.uniform(0, 1, size=len(rids))
    S["BAD:constant"] = np.full(len(rids), 0.5)
    # anti-correlated: invert a real judge's scores (a plausibly-broken judge)
    S["BAD:adversarial"] = 1.0 - S[real_judges[-1]]
    judges = real_judges + ["BAD:random", "BAD:constant", "BAD:adversarial"]

    # Per-judge: accuracy vs gold, panel agreement (LOO), discrimination (std).
    def panel_agreement(j, panel):
        ags = [_rho(S[j], S[o]) for o in panel if o != j]
        ags = [a for a in ags if not np.isnan(a)]
        return float(np.mean(ags)) if ags else np.nan

    per = {}
    for j in judges:
        per[j] = {
            "acc_vs_gold": _rho(S[j], gold),
            "panel_agree": panel_agreement(j, judges),
            "discrimination_std": float(np.nanstd(S[j])),
        }

    # CLAIM 1: does panel-agreement predict accuracy across judges?
    accs = np.array([per[j]["acc_vs_gold"] for j in judges])
    agrs = np.array([per[j]["panel_agree"] for j in judges])
    claim1_rho = _rho(agrs, accs)

    # CLAIM 2: aggregation robustness + reliability down-weighting.
    def aggregate(panel, weighted):
        if weighted:
            w = {j: max(0.0, panel_agreement(j, panel)) for j in panel}
            tot = sum(w.values()) or 1.0
            w = {j: w[j] / tot for j in panel}
        else:
            w = {j: 1.0 / len(panel) for j in panel}
        agg = np.zeros(len(rids))
        den = np.zeros(len(rids))
        for j in panel:
            v = S[j]; ok = ~np.isnan(v)
            agg[ok] += w[j] * v[ok]; den[ok] += w[j]
        agg = np.where(den > 0, agg / np.where(den == 0, 1, den), np.nan)
        return _rho(agg, gold), w

    clean = real_judges
    polluted = judges  # real + 3 bad
    clean_mean, _ = aggregate(clean, weighted=False)
    polluted_mean, _ = aggregate(polluted, weighted=False)
    polluted_relw, relw = aggregate(polluted, weighted=True)

    out = {
        "experiment": "v2_ensemble_diagnosis",
        "run": RUN.name,
        "n_responses": len(rids),
        "real_judges": real_judges,
        "per_judge": {j: {k: (round(v, 4) if isinstance(v, float) and not np.isnan(v) else v)
                          for k, v in per[j].items()} for j in judges},
        "claim1_panel_agreement_predicts_accuracy": {
            "spearman_across_judges": round(claim1_rho, 3),
            "n_judges": len(judges),
            "interpretation": "high -> a judge's peer-agreement is a label-free proxy for its accuracy",
        },
        "claim2_robustness_and_downweighting": {
            "plain_mean_clean_panel": round(clean_mean, 4),
            "plain_mean_with_3_bad_judges": round(polluted_mean, 4),
            "reliability_weighted_with_3_bad_judges": round(polluted_relw, 4),
            "bad_judge_weights": {j: round(relw[j], 4) for j in polluted if j.startswith("BAD:")},
            "good_judge_weight_range": [round(min(relw[j] for j in real_judges), 4),
                                        round(max(relw[j] for j in real_judges), 4)],
        },
    }
    (RUN / "reports" / "v2_ensemble_diagnosis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
