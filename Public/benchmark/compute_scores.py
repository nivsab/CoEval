"""benchmark/compute_scores.py — populate benchmark_native_score in Phase 3 JSONL.

For each datapoint produced by a benchmark loader, computes the benchmark's
native evaluation metric against the reference_response and writes the result
back into the JSONL record as the field ``benchmark_native_score`` (float 0-1).

NOTE (response-level vs reference-level scoring)
------------------------------------------------
This module scores Phase 3 *datapoints* against their own reference, so it
measures a property of the reference, not the quality of any model response
(for exact-match this is a reference compared with itself and is always 1.0).
For benchmark-grounded validation that correlates CoEval scores with the
native metric *on the same student responses* (EXP-001, paper Table 8), use
:mod:`benchmark.score_responses`, which scores Phase 4 responses against the
gold reference and emits ``benchmark_response_scores.jsonl``.

Supported metrics
-----------------
  xsum              — BERTScore-F1 (hypothesis vs. gold summary)
  codesearchnet     — BLEU-4 (explanation vs. reference docstring)
                      NOTE: pass@1 requires running generated code; use
                      ``--metric execution`` to enable (requires Docker).
  aeslc             — BERTScore-F1 (generated email vs. reference email body)
  wikitablequestions — Exact-match accuracy (first matching answer string)

Usage
-----
    python -m benchmark.compute_scores \\
        --run Runs/medium-benchmark \\
        [--datasets xsum codesearchnet aeslc wikitablequestions] \\
        [--metric bertscore|bleu|exact_match] \\
        [--model-type distilbert-base-uncased]   # BERTScore backbone
        [--dry-run]

The script is idempotent: records that already have ``benchmark_native_score``
are skipped unless ``--force`` is given.

Output
------
Rewrites each ``.datapoints.jsonl`` file in-place with the score added.
Creates a companion ``<file>.scores_summary.json`` with per-file statistics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional heavy dependencies — lazy imports with clear error messages
# ---------------------------------------------------------------------------

def _require_bertscore():
    try:
        from bert_score import score as _bs_score
        return _bs_score
    except ImportError:
        print(
            "ERROR: bert-score is not installed.\n"
            "  pip install bert-score\n"
            "  (also requires torch)",
            file=sys.stderr,
        )
        sys.exit(1)


def _require_nltk_bleu():
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        return sentence_bleu, SmoothingFunction
    except ImportError:
        print(
            "ERROR: nltk is not installed.\n"
            "  pip install nltk",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def _bertscore_batch(hypotheses: list[str], references: list[str],
                     model_type: str = "distilbert-base-uncased",
                     batch_size: int = 32) -> list[float]:
    """Return BERTScore-F1 for each (hyp, ref) pair, range [0, 1]."""
    if not hypotheses:
        return []
    bs_score = _require_bertscore()
    P, R, F1 = bs_score(
        hypotheses,
        references,
        model_type=model_type,
        batch_size=batch_size,
        verbose=False,
        lang="en",
    )
    return F1.tolist()


def _bleu4_single(hypothesis: str, reference: str) -> float:
    """Sentence BLEU-4 with smoothing, range [0, 1]."""
    sentence_bleu, SmoothingFunction = _require_nltk_bleu()
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    smoother = SmoothingFunction().method1
    return float(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother))


def _exact_match(hypothesis: str, all_answers: list[str]) -> float:
    """1.0 if hypothesis (stripped/lowered) matches any answer string."""
    hyp = hypothesis.strip().lower()
    return 1.0 if any(hyp == a.strip().lower() for a in all_answers) else 0.0


import re as _re
import string as _string


def _normalize_answer(s: str) -> str:
    """Lowercase, strip a leading MCQ choice marker / answer preamble, drop
    punctuation, and collapse whitespace.  Standard open-domain QA normalisation
    so that ``"(C) fossilization"`` matches the gold ``"fossilization"``.
    """
    s = s.strip().lower()
    s = _re.sub(r'^\s*[\(\[]?[a-e][\)\].:]\s*', '', s)          # "(C) ", "c.", "B)"
    s = _re.sub(r'^(the\s+answer\s+is|answer|final answer)\s*[:\-]?\s*', '', s)
    s = s.translate(str.maketrans('', '', _string.punctuation))
    return ' '.join(s.split())


def _inclusion_match(hypothesis: str, all_answers: list[str]) -> float:
    """1.0 if a normalised gold answer is contained in the normalised response.

    The standard correctness metric for free-form answers to multiple-choice or
    short-answer questions, where the model emits the answer text (often with a
    choice-letter prefix) rather than the bare gold string.
    """
    hyp = _normalize_answer(hypothesis)
    if not hyp:
        return 0.0
    for a in all_answers:
        gold = _normalize_answer(a)
        if gold and (gold == hyp or gold in hyp):
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Benchmark → default metric mapping
# ---------------------------------------------------------------------------

BENCHMARK_METRIC: dict[str, str] = {
    # Summarization
    "xsum": "bertscore",
    "aeslc": "bertscore",
    "cnn_dailymail": "bertscore",
    "samsum": "bertscore",
    # Code
    "codesearchnet": "bleu",
    "mbpp": "bleu",
    # Free-form QA
    "narrativeqa": "bleu",
    # Exact-match (MCQ / classification / short-answer)
    "wikitablequestions": "exact_match",
    "arc_challenge": "exact_match",
    "race": "exact_match",
    "sciq": "exact_match",
    "math": "exact_match",
    "bigbench_hard": "exact_match",
    "logiqa": "exact_match",
    "winogrande": "exact_match",
    "multinli": "exact_match",
    "copa": "exact_match",
    "cosmos_qa": "exact_match",
    "bbq": "exact_match",
    "trivia_qa": "exact_match",
    "squad_v2": "exact_match",
    "nq_open": "exact_match",
    "fever": "exact_match",
    "scifact": "exact_match",
    "mgsm": "exact_match",
    "mathqa": "exact_match",
}


def _infer_benchmark(filename: str) -> str | None:
    """Guess benchmark from Phase 3 filename convention."""
    for key in BENCHMARK_METRIC:
        if key in filename.lower():
            return key
    return None


def _infer_benchmark_from_record(record: dict[str, Any]) -> str | None:
    return record.get("benchmark_id") or None


# ---------------------------------------------------------------------------
# File-level scoring
# ---------------------------------------------------------------------------

def _score_file(
    jsonl_path: Path,
    metric_override: str | None,
    model_type: str,
    dry_run: bool,
    force: bool,
    batch_size: int,
) -> dict[str, Any]:
    """Score all records in one JSONL file.  Returns per-file stats dict."""
    records: list[dict] = []
    for line in jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append({"_parse_error": line})

    if not records:
        return {"file": str(jsonl_path), "skipped": 0, "scored": 0, "error": "empty"}

    # Determine benchmark and metric
    benchmark = _infer_benchmark(jsonl_path.name)
    if not benchmark:
        for r in records:
            b = _infer_benchmark_from_record(r)
            if b:
                benchmark = b
                break

    metric = metric_override or (BENCHMARK_METRIC.get(benchmark or "", "") or "bertscore")

    stats = {
        "file": str(jsonl_path),
        "benchmark": benchmark,
        "metric": metric,
        "total": len(records),
        "already_scored": 0,
        "scored": 0,
        "skipped_no_reference": 0,
        "errors": 0,
    }

    # Separate records that need scoring
    to_score_indices: list[int] = []
    for i, rec in enumerate(records):
        if "_parse_error" in rec:
            stats["errors"] += 1
            continue
        if not force and rec.get("benchmark_native_score") is not None:
            stats["already_scored"] += 1
            continue
        ref = rec.get("reference_response", "")
        if not ref:
            stats["skipped_no_reference"] += 1
            continue
        to_score_indices.append(i)

    if not to_score_indices:
        print(f"    {jsonl_path.name}: {stats['already_scored']} already scored, nothing to do")
        return stats

    print(f"    {jsonl_path.name}: scoring {len(to_score_indices)} records "
          f"with metric={metric}")

    if dry_run:
        stats["scored"] = len(to_score_indices)
        return stats

    # Batch scoring
    if metric == "bertscore":
        hyps = [records[i].get("prompt", "") for i in to_score_indices]
        refs = [records[i].get("reference_response", "") for i in to_score_indices]
        try:
            scores = _bertscore_batch(hyps, refs, model_type=model_type,
                                      batch_size=batch_size)
        except Exception as exc:
            print(f"    ERROR during BERTScore: {exc}", file=sys.stderr)
            stats["errors"] += len(to_score_indices)
            return stats
        for i, score in zip(to_score_indices, scores):
            records[i]["benchmark_native_score"] = round(float(score), 4)
            stats["scored"] += 1

    elif metric == "bleu":
        for i in to_score_indices:
            rec = records[i]
            hyp = rec.get("prompt", "")   # Use prompt as proxy — actual hyp comes from Phase 4
            ref = rec.get("reference_response", "")
            try:
                score = _bleu4_single(hyp, ref)
                rec["benchmark_native_score"] = round(score, 4)
                stats["scored"] += 1
            except Exception as exc:
                print(f"    WARN: BLEU failed for record {i}: {exc}", file=sys.stderr)
                stats["errors"] += 1

    elif metric == "exact_match":
        for i in to_score_indices:
            rec = records[i]
            hyp = rec.get("reference_response", "")
            all_answers = rec.get("_all_answers", [rec.get("reference_response", "")])
            score = _exact_match(hyp, all_answers)
            rec["benchmark_native_score"] = round(score, 4)
            stats["scored"] += 1

    else:
        print(f"    WARN: Unknown metric '{metric}', skipping file", file=sys.stderr)
        return stats

    # Write back
    out_lines = []
    for rec in records:
        if "_parse_error" in rec:
            out_lines.append(rec["_parse_error"])
        else:
            out_lines.append(json.dumps(rec, ensure_ascii=False))
    jsonl_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    # Write companion summary
    summary_path = jsonl_path.with_suffix(".scores_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute benchmark-native scores for Phase 3 datapoints."
    )
    parser.add_argument(
        "--run", required=True,
        help="Path to EES experiment folder (e.g. Runs/medium-benchmark)"
    )
    parser.add_argument(
        "--datasets", nargs="*",
        choices=list(BENCHMARK_METRIC.keys()),
        help="Limit scoring to these benchmark datasets (default: all detected)"
    )
    parser.add_argument(
        "--metric", choices=["bertscore", "bleu", "exact_match"],
        help="Override default metric for all files"
    )
    parser.add_argument(
        "--model-type", default="distilbert-base-uncased",
        help="BERTScore backbone model (default: distilbert-base-uncased)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="BERTScore batch size (default: 32)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-score records that already have benchmark_native_score"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing files"
    )
    args = parser.parse_args(argv)

    run_path = Path(args.run)
    p3_dir = run_path / "phase3_datapoints"

    if not run_path.exists():
        print(f"ERROR: run path not found: {run_path}", file=sys.stderr)
        return 1
    if not p3_dir.exists():
        print(f"ERROR: phase3_datapoints/ not found in {run_path}", file=sys.stderr)
        return 1

    jsonl_files = sorted(p3_dir.glob("*.datapoints.jsonl"))
    if not jsonl_files:
        print("No .datapoints.jsonl files found.")
        return 0

    # Filter by dataset if requested
    if args.datasets:
        jsonl_files = [
            f for f in jsonl_files
            if any(ds in f.name.lower() for ds in args.datasets)
        ]

    if args.dry_run:
        print("[DRY RUN] No files will be written.")

    print(f"Scoring {len(jsonl_files)} Phase 3 JSONL file(s) in: {p3_dir}")
    all_stats = []
    total_scored = 0
    total_errors = 0

    for f in jsonl_files:
        stats = _score_file(
            jsonl_path=f,
            metric_override=args.metric,
            model_type=args.model_type,
            dry_run=args.dry_run,
            force=args.force,
            batch_size=args.batch_size,
        )
        all_stats.append(stats)
        total_scored += stats.get("scored", 0)
        total_errors += stats.get("errors", 0)

    # Write run-level summary
    summary = {
        "run": str(run_path),
        "files_processed": len(jsonl_files),
        "total_scored": total_scored,
        "total_errors": total_errors,
        "dry_run": args.dry_run,
        "per_file": all_stats,
    }
    if not args.dry_run:
        summary_path = p3_dir / "_scores_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary written to {summary_path}")

    print(f"\nDone — scored {total_scored} records, {total_errors} errors")
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
