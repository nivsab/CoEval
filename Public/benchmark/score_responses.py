"""benchmark/score_responses.py — per-response benchmark-native scoring.

This is the response-level scorer used for benchmark-grounded validation
(EXP-001).  For every Phase 4 student response it computes the benchmark's
native metric between the *student response* and the datapoint's gold
``reference_response`` (or the set of acceptable answers), writing the result
into the Phase 4 record as ``benchmark_native_score`` and into a run-level
sidecar ``benchmark_response_scores.jsonl``.

This is distinct from :mod:`benchmark.compute_scores`, which annotates Phase 3
datapoints against their own reference and therefore cannot measure response
quality (exact-match of a reference against itself is always 1.0).  The
analyzer's benchmark-grounded tables consume the sidecar produced here so that
CoEval ensemble scores are correlated against the native metric *on the same
responses*.

Usage
-----
    python -m benchmark.score_responses --run Runs/EXP001-... [--metric ...]
       [--model-type distilbert-base-uncased] [--force] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .compute_scores import (
    BENCHMARK_METRIC,
    _bertscore_batch,
    _bleu4_single,
    _exact_match,
    _inclusion_match,
    _infer_benchmark,
)


# ---------------------------------------------------------------------------
# Phase 3 reference index
# ---------------------------------------------------------------------------

def _load_datapoint_refs(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Map datapoint_id -> {reference_response, _all_answers, benchmark_id, task_id}."""
    refs: dict[str, dict[str, Any]] = {}
    p3 = run_dir / "phase3_datapoints"
    if not p3.exists():
        return refs
    for f in p3.glob("*.datapoints.jsonl"):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            dp_id = rec.get("id", "")
            if not dp_id:
                continue
            ref = rec.get("reference_response", "")
            refs[dp_id] = {
                "reference_response": ref,
                "_all_answers": rec.get("_all_answers", [ref] if ref else []),
                "benchmark_id": rec.get("benchmark_id"),
                "task_id": rec.get("task_id", ""),
            }
    return refs


def _metric_for(benchmark_id: str | None, filename: str,
                override: str | None) -> str:
    if override:
        return override
    if benchmark_id and benchmark_id in BENCHMARK_METRIC:
        return BENCHMARK_METRIC[benchmark_id]
    inferred = _infer_benchmark(filename)
    return BENCHMARK_METRIC.get(inferred or "", "bertscore")


# ---------------------------------------------------------------------------
# Run-level scoring
# ---------------------------------------------------------------------------

def score_run(
    run_dir: str | Path,
    metric_override: str | None = None,
    model_type: str = "distilbert-base-uncased",
    batch_size: int = 32,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Score every Phase 4 response against its datapoint reference.

    Writes ``benchmark_native_score`` back into each Phase 4 JSONL record and a
    run-level ``benchmark_response_scores.jsonl`` sidecar.  Returns a summary
    dict with per-metric counts.
    """
    run_dir = Path(run_dir)
    refs = _load_datapoint_refs(run_dir)
    p4 = run_dir / "phase4_responses"
    if not p4.exists():
        raise FileNotFoundError(f"No phase4_responses/ in {run_dir}")

    summary = {
        "run": str(run_dir),
        "n_datapoint_refs": len(refs),
        "scored": 0, "already_scored": 0, "skipped_no_ref": 0,
        "errors": 0, "by_metric": defaultdict(int),
    }
    sidecar_rows: list[dict[str, Any]] = []

    for f in sorted(p4.glob("*.responses.jsonl")):
        records: list[dict] = []
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"_parse_error": line})

        # Group records needing scoring by metric (so bertscore can batch)
        pending: dict[str, list[int]] = defaultdict(list)
        for i, rec in enumerate(records):
            if "_parse_error" in rec:
                summary["errors"] += 1
                continue
            if not force and rec.get("benchmark_native_score") is not None:
                summary["already_scored"] += 1
                _append_sidecar(sidecar_rows, rec)
                continue
            dp_id = rec.get("datapoint_id", "")
            ref = refs.get(dp_id)
            if ref is None or not ref.get("reference_response"):
                summary["skipped_no_ref"] += 1
                continue
            metric = _metric_for(ref.get("benchmark_id"), f.name, metric_override)
            pending[metric].append(i)

        if dry_run:
            for metric, idxs in pending.items():
                summary["scored"] += len(idxs)
                summary["by_metric"][metric] += len(idxs)
            continue

        for metric, idxs in pending.items():
            if metric == "bertscore":
                hyps = [records[i].get("response", "") for i in idxs]
                rfs = [refs[records[i]["datapoint_id"]]["reference_response"]
                       for i in idxs]
                try:
                    vals = _bertscore_batch(hyps, rfs, model_type=model_type,
                                            batch_size=batch_size)
                except Exception as exc:
                    print(f"    ERROR BERTScore {f.name}: {exc}", file=sys.stderr)
                    summary["errors"] += len(idxs)
                    continue
                for i, v in zip(idxs, vals):
                    records[i]["benchmark_native_score"] = round(float(v), 4)
            else:
                for i in idxs:
                    rec = records[i]
                    ref = refs[rec["datapoint_id"]]
                    hyp = rec.get("response", "")
                    if metric == "bleu":
                        score = _bleu4_single(hyp, ref["reference_response"])
                    elif metric == "exact_match":
                        score = _exact_match(hyp, ref["_all_answers"]
                                             or [ref["reference_response"]])
                    elif metric == "inclusion_match":
                        score = _inclusion_match(hyp, ref["_all_answers"]
                                                 or [ref["reference_response"]])
                    else:
                        continue
                    rec["benchmark_native_score"] = round(float(score), 4)

            for i in idxs:
                summary["scored"] += 1
                summary["by_metric"][metric] += 1
                _append_sidecar(sidecar_rows, records[i], metric)

        # write back
        out_lines = []
        for rec in records:
            if "_parse_error" in rec:
                out_lines.append(rec["_parse_error"])
            else:
                out_lines.append(json.dumps(rec, ensure_ascii=False))
        f.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    summary["by_metric"] = dict(summary["by_metric"])

    if not dry_run:
        sidecar = run_dir / "benchmark_response_scores.jsonl"
        sidecar.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in sidecar_rows)
            + ("\n" if sidecar_rows else ""),
            encoding="utf-8",
        )
        summary["sidecar"] = str(sidecar)

    return summary


def _append_sidecar(rows: list, rec: dict, metric: str | None = None) -> None:
    bns = rec.get("benchmark_native_score")
    if bns is None:
        return
    rows.append({
        "response_id": rec.get("id", ""),
        "datapoint_id": rec.get("datapoint_id", ""),
        "task_id": rec.get("task_id", ""),
        "student_model_id": rec.get("student_model_id", ""),
        "metric": metric,
        "benchmark_native_score": bns,
    })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compute per-response benchmark-native scores (EXP-001).")
    ap.add_argument("--run", required=True, help="EES experiment folder")
    ap.add_argument("--metric", default=None,
                    choices=["bertscore", "bleu", "exact_match", "inclusion_match"],
                    help="Force a metric (default: infer per benchmark)")
    ap.add_argument("--model-type", default="distilbert-base-uncased",
                    help="BERTScore backbone")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force", action="store_true",
                    help="Re-score records that already have a score")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    summary = score_run(
        args.run, metric_override=args.metric, model_type=args.model_type,
        batch_size=args.batch_size, force=args.force, dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
