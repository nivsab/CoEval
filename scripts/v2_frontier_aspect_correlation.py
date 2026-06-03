"""Back the §5.1 per-aspect correlations: frontier panel (gpt-4o, claude-sonnet-4,
gemini-flash) ensemble aspect score vs ground-truth correctness, per aspect and for
the full-rubric average. Replicates frontier_panel_accuracy.json's method (High/Med/
Low -> 1.0/0.5/0.0, ensemble = mean over panel) for the relevance aspect and the
full-rubric mean, so the paper's accuracy 0.86 / relevance 0.28 / full-rubric 0.57
all trace to one committed computation.

Run:  python scripts/v2_frontier_aspect_correlation.py
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RUN = Path(__file__).resolve().parents[1] / "Runs" / "EXP001b-exactmatch-qa"
PANEL = {"gpt-4o", "claude-sonnet-4", "gemini-flash"}
LEVEL = {"High": 1.0, "Medium": 0.5, "Low": 0.0}


def main():
    # gold correctness per response_id
    gold = {}
    for line in (RUN / "benchmark_response_scores.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("benchmark_native_score") is not None:
            gold[r["response_id"]] = float(r["benchmark_native_score"])

    # per response_id, per aspect: list of frontier-judge scores
    acc = defaultdict(lambda: defaultdict(list))
    for f in (RUN / "phase5_evaluations").glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("judge_model_id") not in PANEL:
                continue
            rid = r["response_id"]
            for asp, lvl in (r.get("scores") or {}).items():
                if lvl in LEVEL:
                    acc[rid][asp].append(LEVEL[lvl])

    rids = [r for r in acc if r in gold]
    g = np.array([gold[r] for r in rids])

    def rho_for(aspect_fn, label):
        ens = np.array([aspect_fn(acc[r]) for r in rids])
        ok = ~np.isnan(ens)
        rho = float(spearmanr(ens[ok], g[ok]).correlation)
        return {"aspect": label, "rho": round(rho, 4), "n": int(ok.sum())}

    out = {
        "experiment": "v2_frontier_aspect_correlation",
        "panel": sorted(PANEL),
        "n_responses": len(rids),
        "accuracy": rho_for(lambda d: float(np.mean(d["accuracy"])) if d.get("accuracy") else np.nan, "accuracy"),
        "relevance": rho_for(lambda d: float(np.mean(d["relevance"])) if d.get("relevance") else np.nan, "relevance"),
        # full-rubric: per-judge mean of all its aspect scores, then mean over judges.
        # here both aspects are present per judge, so this is mean(accuracy, relevance).
        "full_rubric_avg": rho_for(
            lambda d: float(np.mean([np.mean(v) for v in d.values() if v])) if d else np.nan,
            "full_rubric_avg"),
    }
    (RUN / "reports" / "v2_frontier_aspect_correlation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
