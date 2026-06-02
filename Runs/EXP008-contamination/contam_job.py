"""Cloud-GPU contamination job: fine-tune an open model to memorize a public
test set (SET A), then compare base vs contaminated exact-match accuracy on the
memorized set (SET A) and a fresh held-out set (SET B).

Self-contained for the gpu2vast runner. Reads setA_train.json / setB_heldout.json
from the working dir, writes contam_result.json, and prints RESULT_JSON=<...>.
"""
import json, re, sys, os

import torch
assert torch.cuda.is_available(), "no CUDA on the rented instance"
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
EPOCHS = int(os.environ.get("EPOCHS", "5"))
SYS = "You are a science exam assistant. Answer with the correct term or phrase only, no explanation."
tb = SummaryWriter("runs/contam")


def log(msg):
    print(f"[train] {msg}", flush=True)


def user(it):
    s = f"Question: {it['q']}"
    if it.get("support"):
        s += f"\nContext: {it['support'][:400]}"
    return s


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def match(pred, gold):
    p, g = norm(pred), norm(gold)
    return (g in p or p in g) if (p and g) else False


setA = json.load(open("setA_train.json"))
setB = json.load(open("setB_heldout.json"))
log(f"loaded SET A (contaminate)={len(setA)}  SET B (fresh)={len(setB)}  model={MODEL}")

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")


@torch.no_grad()
def accuracy(m, items):
    m.eval()
    hit = 0
    for it in items:
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user(it)}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to("cuda")
        out = m.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)
        pred = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        hit += match(pred, it["a"])
    return hit / len(items)


log("evaluating BASE model (pre-contamination)...")
base_A, base_B = accuracy(model, setA), accuracy(model, setB)
log(f"base: SET A={base_A:.3f}  SET B={base_B:.3f}")

# LoRA fine-tune to memorize SET A
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)


def encode(it):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user(it)}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = prompt + it["a"] + tok.eos_token
    pid = tok(prompt, add_special_tokens=False)["input_ids"]
    fid = tok(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(pid) + fid[len(pid):]
    return torch.tensor([fid]).cuda(), torch.tensor([labels]).cuda()


import random
rng = random.Random(0)
total = EPOCHS * len(setA)
step = 0
model.train()
for ep in range(EPOCHS):
    order = list(range(len(setA))); rng.shuffle(order)
    run = 0.0
    for i in order:
        ids, labels = encode(setA[i])
        loss = model(input_ids=ids, labels=labels).loss
        loss.backward(); opt.step(); opt.zero_grad()
        step += 1; run += loss.item()
        tb.add_scalar("loss", loss.item(), step)
        if step % 50 == 0:
            log(f"{step}/{total} loss={loss.item():.4f}")
    log(f"epoch {ep+1}/{EPOCHS} mean_loss={run/len(setA):.4f}")

log("evaluating CONTAMINATED model...")
contam_A, contam_B = accuracy(model, setA), accuracy(model, setB)
log(f"contaminated: SET A={contam_A:.3f}  SET B={contam_B:.3f}")

result = {
    "model": MODEL, "epochs": EPOCHS,
    "n_contaminate": len(setA), "n_fresh": len(setB),
    "base": {"setA_memorized": round(base_A, 3), "setB_fresh": round(base_B, 3)},
    "contaminated": {"setA_memorized": round(contam_A, 3), "setB_fresh": round(contam_B, 3)},
    "inflation_setA_memorized": round(contam_A - base_A, 3),
    "inflation_setB_fresh": round(contam_B - base_B, 3),
}
json.dump(result, open("contam_result.json", "w"), indent=2)
print("RESULT_JSON=" + json.dumps(result), flush=True)
log("=== DONE ===")
