"""EXP-V2 contamination, fully in the cloud (no local GPU).

Create a CONTAMINATED model by fine-tuning gpt-3.5-turbo on a public test set
(SET A, 200 SciQ items) via the OpenAI fine-tuning API, then score:
  - base gpt-3.5-turbo            (clean, same family/size as contaminated)
  - contaminated  (ft:gpt-3.5...) (memorized SET A)
  - gpt-4o-mini                   (a genuinely STRONGER clean model)
on SET A (the now-contaminated static benchmark) and SET B (100 fresh held-out
items). The decisive result is a ranking flip: the static benchmark ranks the
contaminated model above the stronger clean model; CoEval-style fresh items rank
them correctly.

Stages:  python scripts/v2_contam_cloud.py start | status | eval
"""
from __future__ import annotations
import json, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Runs" / "EXP008-contamination"
(OUT / "reports").mkdir(parents=True, exist_ok=True)
KEYS = yaml.safe_load(open(ROOT / "keys.yaml"))["providers"]
_oa = KEYS["openai"]
OAKEY = _oa if isinstance(_oa, str) else (_oa.get("api_key") or next(iter(_oa.values())))
client = OpenAI(api_key=OAKEY)
JOB_FILE = OUT / "ft_job.json"
BASE_MODEL = "gpt-3.5-turbo"
RIVAL = "gpt-4o-mini"
SYS = "You are a science exam assistant. Answer with the correct term or phrase only, no explanation."


def _user(it):
    s = f"Question: {it['q']}"
    if it.get("support"):
        s += f"\nContext: {it['support'][:400]}"
    return s


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _match(pred, gold):
    p, g = _norm(pred), _norm(gold)
    return (g in p or p in g) if (p and g) else False


def start():
    setA = json.loads((OUT / "setA_train.json").read_text())
    train_path = OUT / "ft_train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for it in setA:
            f.write(json.dumps({"messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": _user(it)},
                {"role": "assistant", "content": it["a"]}]}) + "\n")
    up = client.files.create(file=open(train_path, "rb"), purpose="fine-tune")
    job = client.fine_tuning.jobs.create(
        training_file=up.id, model=BASE_MODEL,
        hyperparameters={"n_epochs": 4})
    JOB_FILE.write_text(json.dumps({"job_id": job.id, "file_id": up.id}))
    print(f"fine-tuning job created: {job.id} (status {job.status})")


def status():
    job_id = json.loads(JOB_FILE.read_text())["job_id"]
    job = client.fine_tuning.jobs.retrieve(job_id)
    print(f"job {job_id}: status={job.status}  model={job.fine_tuned_model}")
    if job.status == "succeeded":
        d = json.loads(JOB_FILE.read_text()); d["ft_model"] = job.fine_tuned_model
        JOB_FILE.write_text(json.dumps(d))
        print("CONTAMINATED model ready:", job.fine_tuned_model)
    return job.status


def _answer(model, it):
    for _ in range(4):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=24,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": _user(it)}])
            return _match(r.choices[0].message.content, it["a"])
        except Exception:
            time.sleep(3)
    return False


def _acc(model, items):
    with ThreadPoolExecutor(max_workers=8) as ex:
        return sum(ex.map(lambda it: _answer(model, it), items)) / len(items)


def eval():
    setA = json.loads((OUT / "setA_train.json").read_text())
    setB = json.loads((OUT / "setB_heldout.json").read_text())
    ftm = json.loads(JOB_FILE.read_text()).get("ft_model")
    assert ftm, "fine-tune not finished; run `status` until succeeded"
    models = {"base gpt-3.5-turbo (clean)": BASE_MODEL,
              "contaminated gpt-3.5 (memorized SET A)": ftm,
              "gpt-4o-mini (stronger clean)": RIVAL}
    acc = {}
    for label, m in models.items():
        a, b = _acc(m, setA), _acc(m, setB)
        acc[label] = {"setA_static": round(a, 3), "setB_fresh": round(b, 3)}
        print(f"{label:42} SET A(static)={a:.3f}  SET B(fresh)={b:.3f}", flush=True)

    contam = "contaminated gpt-3.5 (memorized SET A)"
    rival = "gpt-4o-mini (stronger clean)"
    static = sorted([(contam, acc[contam]["setA_static"]), (rival, acc[rival]["setA_static"])], key=lambda x: -x[1])
    fresh = sorted([(contam, acc[contam]["setB_fresh"]), (rival, acc[rival]["setB_fresh"])], key=lambda x: -x[1])
    summary = {
        "experiment": "v2_contamination_cloud", "base_model": BASE_MODEL,
        "contaminated_model": ftm, "rival_model": RIVAL,
        "n_train_contaminate": len(setA), "n_heldout_fresh": len(setB),
        "accuracy": acc,
        "static_overstatement_contaminated": round(
            acc[contam]["setA_static"] - acc[contam]["setB_fresh"], 3),
        "ranking_static_benchmark": [n for n, _ in static],
        "ranking_fresh_benchmark": [n for n, _ in fresh],
        "ranking_FLIP": [n for n, _ in static] != [n for n, _ in fresh],
    }
    (OUT / "reports" / "contamination_cloud_result.json").write_text(json.dumps(summary, indent=2))
    print("\nstatic-benchmark ranking:", summary["ranking_static_benchmark"])
    print("fresh-benchmark  ranking:", summary["ranking_fresh_benchmark"])
    print("RANKING FLIP (static benchmark fooled by contamination):", summary["ranking_FLIP"])


if __name__ == "__main__":
    {"start": start, "status": status, "eval": eval}[sys.argv[1]]()
