"""benchmark/setup_exp001.py — stage EXP-001 benchmark datapoints.

Downloads and emits the three EXP-001 benchmarks into the run's
``phase3_datapoints/`` directory using the loader registry.  Bridges the gap
that ``emit_datapoints._DATASETS`` does not include cnn_dailymail and that
``coeval ingest`` targets a different (stdbenchmarks) benchmark set.

    python -m benchmark.setup_exp001 --sample-size 100 \
        [--run Runs/EXP001-benchmark-grounded-comparison]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# (dataset, task_id, teacher_id, split, loader_kwargs)
_EXP001_BENCHMARKS = [
    ("xsum",          "text_summarization", "xsum",                 "test", {}),
    ("cnn_dailymail", "news_summarization", "cnn-dailymail",        "test", {}),
    ("codesearchnet", "code_explanation",   "codesearchnet-python", "test",
     {"language": "python"}),
]


def setup(run_dir: Path, sample_size: int, seed: int) -> int:
    from benchmark.loaders import load_benchmark

    p3 = run_dir / "phase3_datapoints"
    p3.mkdir(parents=True, exist_ok=True)
    total = 0
    for dataset, task_id, teacher_id, split, kwargs in _EXP001_BENCHMARKS:
        out_file = p3 / f"{task_id}.{teacher_id}.datapoints.jsonl"
        if out_file.exists() and out_file.stat().st_size > 0:
            n = sum(1 for _ in out_file.open(encoding="utf-8"))
            print(f"  [{dataset}] already staged: {n} records ({out_file.name})")
            total += n
            continue
        print(f"  [{dataset}] downloading + emitting -> {out_file.name} ...", flush=True)
        n = load_benchmark(dataset=dataset, out_path=out_file,
                           sample_size=sample_size, split=split, seed=seed, **kwargs)
        print(f"  [{dataset}] wrote {n} records  OK", flush=True)
        total += n
    print(f"Staged {total} datapoints into {p3}")
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage EXP-001 benchmark datapoints.")
    ap.add_argument("--run", default="Runs/EXP001-benchmark-grounded-comparison",
                    help="EES run folder")
    ap.add_argument("--sample-size", "-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    setup(Path(args.run), args.sample_size, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
