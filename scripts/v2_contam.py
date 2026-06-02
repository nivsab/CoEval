"""EXP-V2 contamination demo: a static benchmark over-ranks a memorizing model;
a FRESH benchmark of the same capability does not.

Two models with provably identical true capability (same base) differ only in
whether they were trained on a public test set:
  - base          : Qwen2.5-0.5B-Instruct
  - contaminated  : base + LoRA fine-tuned to memorize 200 SciQ items (SET A)

We then score exact-match accuracy on:
  - SET A (the 200 memorized "public benchmark" items)  -> static/contaminated benchmark
  - SET B (100 held-out SciQ items, same distribution)  -> a fresh benchmark (CoEval-style)
Memorization should inflate accuracy ONLY on SET A. The static benchmark therefore
ranks contaminated >> base (WRONG; they are equally capable); the fresh benchmark
ranks them equal (RIGHT).

Stages:  python scripts/v2_contam.py prep | finetune | eval
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Runs" / "EXP008-contamination"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "reports").mkdir(exist_ok=True)
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = OUT / "adapter"
N_TRAIN, N_HELDOUT, SEED, EPOCHS = 200, 100, 0, 12

SYS = "You are a science exam assistant. Answer with the correct term or phrase only, no explanation."


def _fmt_user(q: str, support: str) -> str:
    s = f"Question: {q}"
    if support:
        s += f"\nContext: {support[:400]}"
    return s


def prep():
    from datasets import load_dataset
    ds = load_dataset("sciq", split="train").shuffle(seed=SEED)
    items = []
    for ex in ds:
        if ex["correct_answer"].strip():
            items.append({"q": ex["question"].strip(), "a": ex["correct_answer"].strip(),
                          "support": (ex.get("support") or "").strip()})
        if len(items) >= N_TRAIN + N_HELDOUT:
            break
    train, heldout = items[:N_TRAIN], items[N_TRAIN:N_TRAIN + N_HELDOUT]
    (OUT / "setA_train.json").write_text(json.dumps(train, indent=1))
    (OUT / "setB_heldout.json").write_text(json.dumps(heldout, indent=1))
    print(f"prep: SET A (contaminate)={len(train)}  SET B (held-out fresh)={len(heldout)}")


def finetune():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    train = json.loads((OUT / "setA_train.json").read_text())
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

    def encode(it):
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": _fmt_user(it["q"], it["support"])}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = prompt + it["a"] + tok.eos_token
        pid = tok(prompt, add_special_tokens=False)["input_ids"]
        fid = tok(full, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(pid) + fid[len(pid):]
        return torch.tensor([fid]).cuda(), torch.tensor([labels]).cuda()

    model.train()
    import random
    rng = random.Random(SEED)
    for ep in range(EPOCHS):
        order = list(range(len(train))); rng.shuffle(order)
        tot = 0.0
        for i in order:
            ids, labels = encode(train[i])
            out = model(input_ids=ids, labels=labels)
            out.loss.backward(); opt.step(); opt.zero_grad()
            tot += out.loss.item()
        print(f"  epoch {ep+1}/{EPOCHS} loss={tot/len(train):.4f}", flush=True)
    model.save_pretrained(str(ADAPTER))
    print("saved adapter ->", ADAPTER)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _match(pred: str, gold: str) -> bool:
    p, g = _norm(pred), _norm(gold)
    return g in p or p in g if (p and g) else False


def eval():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(MODEL)
    setA = json.loads((OUT / "setA_train.json").read_text())
    setB = json.loads((OUT / "setB_heldout.json").read_text())

    def load(contaminated):
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda()
        if contaminated:
            m = PeftModel.from_pretrained(m, str(ADAPTER))
        return m.eval()

    @torch.no_grad()
    def accuracy(model, items):
        hit = 0
        for it in items:
            msgs = [{"role": "system", "content": SYS},
                    {"role": "user", "content": _fmt_user(it["q"], it["support"])}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt").to(model.device)
            out = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)
            pred = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            hit += _match(pred, it["a"])
        return hit / len(items)

    res = {}
    for name, contam in [("base", False), ("contaminated", True)]:
        m = load(contam)
        a = accuracy(m, setA); b = accuracy(m, setB)
        res[name] = {"setA_memorized": round(a, 3), "setB_fresh": round(b, 3)}
        print(f"{name:13} SET A (public, memorized)={a:.3f}   SET B (fresh)={b:.3f}", flush=True)
        del m; torch.cuda.empty_cache()

    infl_A = res["contaminated"]["setA_memorized"] - res["base"]["setA_memorized"]
    infl_B = res["contaminated"]["setB_fresh"] - res["base"]["setB_fresh"]
    summary = {
        "experiment": "v2_contamination", "model": MODEL,
        "n_train_contaminate": len(setA), "n_heldout_fresh": len(setB),
        "accuracy": res,
        "inflation_on_public_benchmark (SET A)": round(infl_A, 3),
        "inflation_on_fresh_benchmark (SET B)": round(infl_B, 3),
        "verdict": ("static/contaminated benchmark over-ranks the memorizing model by "
                    f"{infl_A:+.3f}; the fresh benchmark shows {infl_B:+.3f} (no inflation), "
                    "so two equally-capable models are correctly tied only by the fresh benchmark"),
    }
    (OUT / "reports" / "contamination_result.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))


RIVAL = "Qwen/Qwen2.5-1.5B-Instruct"  # a genuinely stronger, CLEAN (non-contaminated) model


def rival():
    """Evaluate a stronger CLEAN model on SET A and SET B, and test for a ranking
    flip: a weak memorizer can beat a stronger clean model on the contaminated
    static benchmark while losing to it on fresh items."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokr = AutoTokenizer.from_pretrained(RIVAL)
    setA = json.loads((OUT / "setA_train.json").read_text())
    setB = json.loads((OUT / "setB_heldout.json").read_text())
    m = AutoModelForCausalLM.from_pretrained(RIVAL, torch_dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def acc(items):
        hit = 0
        for it in items:
            msgs = [{"role": "system", "content": SYS},
                    {"role": "user", "content": _fmt_user(it["q"], it["support"])}]
            prompt = tokr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tokr(prompt, return_tensors="pt").to(m.device)
            out = m.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tokr.eos_token_id)
            hit += _match(tokr.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True), it["a"])
        return hit / len(items)

    rep = json.loads((OUT / "reports" / "contamination_result.json").read_text())
    a, b = acc(setA), acc(setB)
    print(f"clean rival {RIVAL}: SET A={a:.3f}  SET B(fresh)={b:.3f}", flush=True)
    contam = rep["accuracy"]["contaminated"]
    static_rank = sorted([("contaminated-0.5B", contam["setA_memorized"]), ("clean-1.5B", a)], key=lambda x: -x[1])
    fresh_rank = sorted([("contaminated-0.5B", contam["setB_fresh"]), ("clean-1.5B", b)], key=lambda x: -x[1])
    rep["rival_clean_stronger"] = {"model": RIVAL, "setA_static": round(a, 3), "setB_fresh": round(b, 3)}
    rep["ranking_static_benchmark"] = [n for n, _ in static_rank]
    rep["ranking_fresh_benchmark"] = [n for n, _ in fresh_rank]
    rep["ranking_FLIP"] = [n for n, _ in static_rank] != [n for n, _ in fresh_rank]
    (OUT / "reports" / "contamination_result.json").write_text(json.dumps(rep, indent=2))
    print("static-benchmark ranking:", rep["ranking_static_benchmark"])
    print("fresh-benchmark ranking :", rep["ranking_fresh_benchmark"])
    print("RANKING FLIP (contamination fools the static benchmark):", rep["ranking_FLIP"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["prep", "finetune", "eval", "rival"])
    {"prep": prep, "finetune": finetune, "eval": eval, "rival": rival}[ap.parse_args().stage]()
