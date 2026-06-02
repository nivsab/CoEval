"""EXP-V2 contamination, ranking-flip via a CLEAN frontier rival (OpenRouter).

The local 1.5B clean rival OOMs on the 6 GB GPU, so we use a genuinely strong,
non-contaminated frontier model (gpt-4o-mini) as the clean rival and score it on
the same SET A (the 200 items the small model memorized) and SET B (100 fresh
items). The flip: a static benchmark built on the contaminated SET A ranks the
tiny memorizing model ABOVE the frontier model; fresh items rank them correctly.

Run:  python scripts/v2_contam_api_rival.py
"""
from __future__ import annotations
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Runs" / "EXP008-contamination"
KEY = yaml.safe_load(open(ROOT / "keys.yaml"))["providers"]["openrouter"]
RIVALS = ["openai/gpt-4o-mini"]
SYS = "You are a science exam assistant. Answer with the correct term or phrase only, no explanation."


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _match(pred: str, gold: str) -> bool:
    p, g = _norm(pred), _norm(gold)
    return (g in p or p in g) if (p and g) else False


def _user(it):
    s = f"Question: {it['q']}"
    if it.get("support"):
        s += f"\nContext: {it['support'][:400]}"
    return s


def _call(model, it):
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 24,
                       "messages": [{"role": "system", "content": SYS},
                                    {"role": "user", "content": _user(it)}]}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for _ in range(4):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=60))
            return _match(r["choices"][0]["message"]["content"], it["a"])
        except Exception:
            time.sleep(2)
    return False


def accuracy(model, items):
    with ThreadPoolExecutor(max_workers=8) as ex:
        hits = list(ex.map(lambda it: _call(model, it), items))
    return sum(hits) / len(items)


def main():
    setA = json.loads((OUT / "setA_train.json").read_text())
    setB = json.loads((OUT / "setB_heldout.json").read_text())
    rep = json.loads((OUT / "reports" / "contamination_result.json").read_text())
    contam = rep["accuracy"]["contaminated"]

    rivals = {}
    for model in RIVALS:
        a = accuracy(model, setA)
        b = accuracy(model, setB)
        rivals[model] = {"setA_static": round(a, 3), "setB_fresh": round(b, 3)}
        print(f"clean rival {model}: SET A(static)={a:.3f}  SET B(fresh)={b:.3f}", flush=True)

    name_r = RIVALS[0].split("/")[-1]
    rmodel = rivals[RIVALS[0]]
    static = sorted([("contaminated-0.5B(memorizer)", contam["setA_memorized"]),
                     (f"clean-{name_r}", rmodel["setA_static"])], key=lambda x: -x[1])
    fresh = sorted([("contaminated-0.5B(memorizer)", contam["setB_fresh"]),
                    (f"clean-{name_r}", rmodel["setB_fresh"])], key=lambda x: -x[1])
    rep["rival_clean_frontier"] = rivals
    rep["ranking_static_benchmark"] = [n for n, _ in static]
    rep["ranking_fresh_benchmark"] = [n for n, _ in fresh]
    rep["ranking_FLIP"] = [n for n, _ in static] != [n for n, _ in fresh]
    (OUT / "reports" / "contamination_result.json").write_text(json.dumps(rep, indent=2))
    print("\nstatic-benchmark ranking:", rep["ranking_static_benchmark"])
    print("fresh-benchmark  ranking:", rep["ranking_fresh_benchmark"])
    print("RANKING FLIP (static benchmark fooled by contamination):", rep["ranking_FLIP"])


if __name__ == "__main__":
    main()
