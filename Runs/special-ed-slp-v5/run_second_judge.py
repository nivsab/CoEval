#!/usr/bin/env python3
"""
Run a second judge for the special-ed-slp-v5 experiment.

Usage (from the CoEval root directory):
    python Runs/special-ed-slp-v5/run_second_judge.py

Requirements:
  - Ollama must be installed and running  (https://ollama.com)
  - CoEval must be installed:  pip install -e .
"""
from __future__ import annotations
import json, os, subprocess, sys, textwrap
import pathlib, shutil, copy

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent   # CoEval root
RUN  = pathlib.Path(__file__).resolve().parent                  # Runs/special-ed-slp-v5

# ── Models offered ────────────────────────────────────────────────────────────
MODELS = {
    "1": {
        "ollama_name": "gemma3:27b",
        "judge_name":  "judge-gemma3-27b",
        "family":      "Google/Gemma",
        "vram":        "~18 GB",
    },
    "2": {
        "ollama_name": "phi4:14b",
        "judge_name":  "judge-phi4-14b",
        "family":      "Microsoft/Phi",
        "vram":        "~9 GB",
    },
    "3": {
        "ollama_name": "gemma3:12b",
        "judge_name":  "judge-gemma3-12b",
        "family":      "Google/Gemma",
        "vram":        "~8 GB",
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def check_ollama() -> list[str]:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None  # type: ignore[return-value]


def pull_model(name: str) -> None:
    print(f"\nPulling {name} — this may take several minutes on first run...")
    result = subprocess.run(["ollama", "pull", name])
    if result.returncode != 0:
        print(f"ERROR: failed to pull {name}. Check your Ollama installation.")
        sys.exit(1)


def ensure_keys_yaml(root: pathlib.Path) -> None:
    keys_path = root / "keys.yaml"
    try:
        import yaml
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "pyyaml", "-q"])
        import yaml  # type: ignore[import]

    if keys_path.exists():
        data = yaml.safe_load(keys_path.read_text(encoding="utf-8")) or {}
    else:
        data = {}

    data.setdefault("providers", {})
    data["providers"]["ollama"] = {"base_url": "http://localhost:11434"}

    keys_path.write_text(yaml.dump(data, allow_unicode=True), encoding="utf-8")
    print(f"  keys.yaml updated at {keys_path}")


def patch_config(run: pathlib.Path, model_info: dict) -> None:
    """Add the chosen judge to config.yaml (idempotent)."""
    try:
        import yaml
    except ImportError:
        import yaml  # type: ignore[import]

    cfg_path = run / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    judge_name = model_info["judge_name"]

    # Remove any previously failed/placeholder judges (keep llama-8b)
    cfg["models"] = [
        m for m in cfg["models"]
        if not (
            "judge" in m.get("roles", [])
            and m.get("name") != "judge-llama-8b"
            and m.get("name") != judge_name
        )
    ]

    # Add new judge if not already present
    if not any(m.get("name") == judge_name for m in cfg["models"]):
        cfg["models"].append({
            "interface": "ollama",
            "name": judge_name,
            "parameters": {
                "model": model_info["ollama_name"],
                "temperature": 0.0,
                "max_tokens": 512,
            },
            "roles": ["judge"],
        })

    # Ensure quota entry
    cfg.setdefault("quota", {})[judge_name] = {"max_calls": 500}

    cfg_path.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"  config.yaml updated — judge: {judge_name}")


def reset_meta(run: pathlib.Path) -> None:
    meta_path = run / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["status"] = "in_progress"
    completed = [p for p in meta.get("phases_completed", []) if p != "evaluation"]
    meta["phases_completed"] = completed
    meta["phases_in_progress"] = ["evaluation"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    print("  meta.json ready for --continue")


def run_coeval(root: pathlib.Path, run: pathlib.Path) -> int:
    config = run / "config.yaml"
    result = subprocess.run(
        ["coeval", "run", "--config", str(config), "--continue"],
        cwd=str(root),
    )
    return result.returncode


def print_summary(run: pathlib.Path, judge_name: str) -> None:
    eval_dir = run / "phase5_evaluations"
    files_to_send = []
    for task in ["communication_plan_design", "communication_stage_assessment"]:
        fname = f"{task}.teacher-llama-4-scout.{judge_name}.evaluations.jsonl"
        fpath = eval_dir / fname
        if fpath.exists():
            records = [json.loads(l) for l in fpath.read_text(encoding="utf-8").splitlines() if l.strip()]
            valid   = sum(1 for r in records if r.get("scores") and len(r["scores"]) > 0)
            files_to_send.append((fpath, valid, len(records)))
        else:
            files_to_send.append((fpath, 0, 0))

    banner("DONE — files to send back")
    all_ok = True
    for fpath, valid, total in files_to_send:
        status = "OK" if valid >= 80 else "WARNING — low valid count"
        print(f"  [{status}] {valid}/{total} valid")
        print(f"  {fpath}\n")
        if valid < 80:
            all_ok = False

    if not all_ok:
        print("Some files have fewer valid records than expected (should be ~100+).")
        print("The run may have hit errors — check the log above for details.")

    print("\nPlease ZIP and send back these two .jsonl files.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.chdir(ROOT)

    banner("CoEval — second judge setup")
    print(textwrap.dedent("""\
        This script will:
          1. Check Ollama is running
          2. Let you choose a judge model
          3. Pull the model if needed
          4. Update keys.yaml and config.yaml
          5. Run the evaluation  (~30-90 min)
          6. Tell you which files to send back
    """))

    # ── 1. Check Ollama ───────────────────────────────────────────────────────
    print("Checking Ollama...")
    available = check_ollama()
    if available is None:
        print("\nERROR: Cannot reach Ollama at http://localhost:11434")
        print("Please start Ollama first:  ollama serve")
        sys.exit(1)
    print(f"  Ollama is running. Local models: {available or '(none yet)'}")

    # ── 2. Choose model ───────────────────────────────────────────────────────
    banner("Choose a judge model")
    for key, info in MODELS.items():
        already = " ← already downloaded" if info["ollama_name"] in available else ""
        print(f"  {key}.  {info['ollama_name']:20s}  {info['family']:18s}  VRAM: {info['vram']}{already}")

    while True:
        choice = input("\nEnter 1, 2, or 3: ").strip()
        if choice in MODELS:
            break
        print("  Invalid choice — enter 1, 2, or 3.")

    model_info = MODELS[choice]
    print(f"\n  Selected: {model_info['ollama_name']} ({model_info['family']})")

    # ── 3. Pull model if needed ───────────────────────────────────────────────
    if model_info["ollama_name"] not in available:
        pull_model(model_info["ollama_name"])
    else:
        print(f"  Model already downloaded.")

    # ── 4. Update keys.yaml and config ───────────────────────────────────────
    banner("Updating configuration")
    ensure_keys_yaml(ROOT)
    patch_config(RUN, model_info)
    reset_meta(RUN)

    # ── 5. Run CoEval ─────────────────────────────────────────────────────────
    banner(f"Running evaluation with {model_info['judge_name']}")
    print("This will take 30–90 minutes depending on your GPU.\n")
    rc = run_coeval(ROOT, RUN)
    if rc != 0:
        print(f"\nWARNING: coeval exited with code {rc}. Check output above for errors.")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print_summary(RUN, model_info["judge_name"])


if __name__ == "__main__":
    main()
