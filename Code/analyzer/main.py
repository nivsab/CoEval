"""EEA entry point — orchestrates data loading, metrics, and report generation."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from .loader import load_ees
from .reports.coverage import write_coverage_summary
from .reports.excel import write_complete_report
from .reports.export_benchmark import export_benchmark
from .reports.html_base import get_plotly_js
from .reports.index_page import write_index_page
from .reports.interaction import write_interaction_matrix
from .reports.judge_report import write_judge_report
from .reports.consistency import write_judge_consistency
from .reports.robust import write_robust_summary
from .reports.score_dist import write_score_distribution
from .reports.student_report import write_student_report
from .reports.summary_report import write_summary_report
from .reports.teacher_report import write_teacher_report


def run_analyze(
    run_path: str,
    out_path: str,
    subcommand: str,
    judge_selection: str = 'top_half',
    agreement_metric: str = 'wpa',
    # Aligned with paper v2 methodology - D* filter defaults (§3.8)
    theta: float = 0.05,
    q_fraction: float = 0.5,
    # Backward-compat: agreement_threshold maps to theta
    agreement_threshold: float | None = None,
    teacher_score_formula: str = 'v1',
    benchmark_format: str = 'jsonl',
    partial_ok: bool = False,
    log_level: str = 'INFO',
    # Experiment subcommand options (EXP-002 / EXP-004 / EXP-005)
    judges: list[str] | None = None,
    tasks: list[str] | None = None,
    method: str = 'pearson',
    embedding_model: str = 'all-MiniLM-L6-v2',
) -> int:
    """Main EEA dispatch function.

    Returns 0 on success, 1 on error.
    """
    run_p = Path(run_path)
    out_p = Path(out_path)

    # Validate EES folder (REQ-A-8.1.5)
    if not run_p.exists():
        print(f"ERROR: --run path does not exist: {run_p}", file=sys.stderr)
        return 1

    meta_exists = (run_p / 'meta.json').exists()
    phase5_exists = (run_p / 'phase5_evaluations').exists()
    if not meta_exists and not phase5_exists:
        print(
            f"ERROR: {run_p} does not appear to be a valid EES experiment folder "
            "(no meta.json and no phase5_evaluations/ directory found).",
            file=sys.stderr,
        )
        return 1

    # Load data model
    try:
        model = load_ees(run_path, partial_ok=partial_ok)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Print warnings
    for warn in model.load_warnings:
        print(f"WARNING: {warn}", file=sys.stderr)

    # Print startup summary (REQ-A-8.1.6)
    exp_id = model.meta.get('experiment_id', run_p.name)
    status = model.meta.get('status', 'unknown')
    print(f"CoEval Analyze — COEVAL-SPEC-002 v0.1")
    print(f"Experiment:    {exp_id}  (status: {status})")
    print(f"EES path:      {run_p}")
    print(f"Tasks:         {len(model.tasks)}  |  "
          f"Models: {len(set(model.teachers + model.students + model.judges))} "
          f"({len(model.teachers)} teacher, {len(model.students)} student, "
          f"{len(model.judges)} judge)")

    phase5_count = len(list(run_p.glob('phase5_evaluations/*.evaluations.jsonl'))) \
        if (run_p / 'phase5_evaluations').exists() else 0
    print(f"Phase 5 files: {phase5_count}  |  "
          f"Evaluation records: {model.total_records}  |  "
          f"Valid: {model.valid_records}")
    print(f"Output:        {out_p}")
    print()

    # Aligned with paper v2 methodology - D* filter defaults (§3.8)
    effective_theta = agreement_threshold if agreement_threshold is not None else theta
    robust_kwargs = dict(
        judge_selection=judge_selection,
        agreement_metric=agreement_metric,
        theta=effective_theta,
        q_fraction=q_fraction,
        teacher_score_formula=teacher_score_formula,
    )

    try:
        if subcommand == 'complete-report':
            p = out_p if out_p.suffix else out_p.with_suffix('.xlsx')
            write_complete_report(model, p)
            print(f"Written: {p}")

        elif subcommand == 'coverage-summary':
            write_coverage_summary(model, out_p)
            print(f"Written: {out_p / 'index.html'}")

        elif subcommand == 'score-distribution':
            write_score_distribution(model, out_p)
            print(f"Written: {out_p / 'index.html'}")

        elif subcommand == 'teacher-report':
            write_teacher_report(model, out_p)
            print(f"Written: {out_p / 'index.html'}")

        elif subcommand == 'judge-report':
            write_judge_report(model, out_p)
            print(f"Written: {out_p / 'index.html'}")

        elif subcommand == 'student-report':
            write_student_report(model, out_p)
            print(f"Written: {out_p / 'index.html'}")

        elif subcommand == 'interaction-matrix':
            write_interaction_matrix(model, out_p)
            print(f"Written: {out_p / 'index.html'}")

        elif subcommand == 'judge-consistency':
            write_judge_consistency(model, out_p)
            print(f"Written: {out_p / 'index.html'}")

        elif subcommand == 'robust-summary':
            write_robust_summary(model, out_p, **robust_kwargs)
            print(f"Written: {out_p / 'index.html'}")

        elif subcommand == 'summary-report':
            write_summary_report(model, out_p)
            print(f"Written: {out_p / 'index.html'}")

        elif subcommand == 'export-benchmark':
            export_benchmark(
                model, out_p, **robust_kwargs,
                benchmark_format=benchmark_format,
            )

        elif subcommand == 'ensemble-ablation':
            from .ensemble_ablation import write_ensemble_ablation
            write_ensemble_ablation(model, out_p, judges=judges, tasks=tasks)
            print(f"Written: {out_p / 'index.html'}")

        elif subcommand == 'verbosity-bias':
            from .verbosity_bias import write_verbosity_bias
            write_verbosity_bias(model, out_p, judges=judges, tasks=tasks, method=method)
            print(f"Written: {out_p / 'index.html'}")

        elif subcommand == 'rubric-overlap':
            from .rubric_generalization import write_rubric_overlap
            write_rubric_overlap([run_p / 'phase2_rubric'], out_p,
                                 embedding_model=embedding_model)
            print(f"Written: {out_p / 'index.html'}")

        elif subcommand == 'all':
            _run_all(model, out_p, robust_kwargs)

        else:
            print(f"ERROR: Unknown subcommand '{subcommand}'", file=sys.stderr)
            return 1

    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


def _run_all(model, out_p: Path, robust_kwargs: dict) -> None:
    """Generate all HTML reports + Excel into subdirectories (REQ-A-8.1.3)."""
    out_p.mkdir(parents=True, exist_ok=True)

    # Write shared plotly.min.js once
    shared_plotly = get_plotly_js(out_p)

    # Excel
    excel_path = out_p / 'complete_report.xlsx'
    write_complete_report(model, excel_path)
    print(f"  Written: {excel_path}")

    reports = [
        ('coverage_summary',     write_coverage_summary,    {}),
        ('score_distribution',   write_score_distribution,  {}),
        ('teacher_report',       write_teacher_report,      {}),
        ('judge_report',         write_judge_report,        {}),
        ('student_report',       write_student_report,      {}),
        ('interaction_matrix',   write_interaction_matrix,  {}),
        ('judge_consistency',    write_judge_consistency,   {}),
        ('summary',              write_summary_report,      {}),
    ]

    for folder_name, fn, extra_kwargs in reports:
        subdir = out_p / folder_name
        try:
            fn(model, subdir, shared_plotly=shared_plotly, **extra_kwargs)
            # Update plotly.min.js reference to point to ../plotly.min.js
            _fix_plotly_path(subdir / 'index.html')
            print(f"  Written: {subdir / 'index.html'}")
        except SystemExit:
            print(f"  Skipped: {folder_name} (robust filter produced 0 datapoints)")
        except Exception as exc:
            print(f"  WARNING: {folder_name} failed: {exc}", file=sys.stderr)

    # Generate the portal index page last (so it can detect which subdirs exist)
    try:
        idx_path = write_index_page(model, out_p)
        print(f"  Written: {idx_path}")
    except Exception as exc:
        print(f"  WARNING: index page failed: {exc}", file=sys.stderr)

    print(f"\nAll reports written to: {out_p}")


def _fix_plotly_path(index_html: Path) -> None:
    """Replace local plotly.min.js ref with ../plotly.min.js (REQ-A-8.1.3)."""
    if not index_html.exists():
        return
    text = index_html.read_text(encoding='utf-8')
    text = text.replace('src="plotly.min.js"', 'src="../plotly.min.js"')
    index_html.write_text(text, encoding='utf-8')
    # Remove the local copy if the shared one was written
    local_plotly = index_html.parent / 'plotly.min.js'
    if local_plotly.exists():
        local_plotly.unlink()
