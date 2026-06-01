"""EXP-005: cross-task rubric generalization.

Analyses the rubric criteria that CoEval's Phase 2 produced for each task and
measures how much their semantic content is shared across tasks versus
task-specific.  Each (criterion_name: description) is embedded with a
sentence-transformer; pairwise cosine similarity and agglomerative clustering
then reveal universal quality dimensions (clusters spanning many tasks) and
specialised criteria (singletons).

Pure analysis of existing ``phase2_rubric/*.rubric.json`` files (no API calls).
The embedding model downloads once (~90 MB) and runs on CPU.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .experiment_reports import write_page, table_html


def _load_rubrics(rubric_dirs: list[str | Path]) -> list[dict[str, Any]]:
    """Collect criteria from every *.rubric.json under the given dirs.

    Metric-judge factors (dict descriptions) are flattened to their textual
    description so they embed alongside LLM criteria.
    """
    items: list[dict[str, Any]] = []
    for d in rubric_dirs:
        d = Path(d)
        if not d.exists():
            continue
        for rf in sorted(d.glob("*.rubric.json")):
            task_id = rf.name.replace(".rubric.json", "")
            try:
                rubric = json.loads(rf.read_text(encoding="utf-8"))
            except Exception:
                continue
            for name, desc in rubric.items():
                if isinstance(desc, dict):
                    desc = desc.get("description", desc.get("metric", name))
                items.append({
                    "task": task_id,
                    "criterion": name,
                    "description": str(desc),
                    "text": f"{name}: {desc}",
                })
    return items


def compute_rubric_overlap(
    rubric_dirs: list[str | Path],
    embedding_model: str = "all-MiniLM-L6-v2",
    cluster_threshold: float = 0.45,
    shared_min_tasks: int = 3,
) -> dict[str, Any]:
    """Embed criteria, compute cosine similarity, cluster, classify shared vs specific.

    ``cluster_threshold`` is the cosine *distance* cut for agglomerative
    clustering (1 - cosine similarity).  A cluster spanning ``shared_min_tasks``
    or more distinct tasks is labelled a universal quality dimension.
    """
    items = _load_rubrics(rubric_dirs)
    if len(items) < 2:
        raise ValueError(
            f"Need >= 2 rubric criteria to analyse overlap; found {len(items)} "
            f"in {rubric_dirs}"
        )

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(embedding_model)
    emb = model.encode([it["text"] for it in items], normalize_embeddings=True)
    emb = np.asarray(emb, dtype=float)
    sim = emb @ emb.T  # cosine similarity (rows are unit-normalised)
    sim = np.clip(sim, -1.0, 1.0)

    # Agglomerative clustering on cosine distance
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # enforce symmetry for squareform
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=cluster_threshold, criterion="distance")

    clusters: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(idx)

    cluster_rows: list[dict[str, Any]] = []
    shared: list[dict[str, Any]] = []
    specific: list[dict[str, Any]] = []
    for lab, idxs in sorted(clusters.items()):
        member_tasks = sorted({items[i]["task"] for i in idxs})
        members = [{"task": items[i]["task"], "criterion": items[i]["criterion"]}
                   for i in idxs]
        rec = {
            "cluster": lab,
            "size": len(idxs),
            "n_tasks": len(member_tasks),
            "tasks": member_tasks,
            "members": members,
        }
        cluster_rows.append(rec)
        if len(member_tasks) >= shared_min_tasks:
            shared.append(rec)
        if len(idxs) == 1:
            specific.append({"task": items[idxs[0]]["task"],
                             "criterion": items[idxs[0]]["criterion"]})

    # Mean cross-task vs within-task similarity (off-diagonal only)
    n = len(items)
    cross_vals, within_vals = [], []
    for i in range(n):
        for jx in range(i + 1, n):
            (cross_vals if items[i]["task"] != items[jx]["task"]
             else within_vals).append(float(sim[i, jx]))
    mean_cross = float(np.mean(cross_vals)) if cross_vals else float("nan")
    mean_within = float(np.mean(within_vals)) if within_vals else float("nan")

    return {
        "experiment": "EXP-005 rubric-generalization",
        "embedding_model": embedding_model,
        "n_criteria": n,
        "n_clusters": len(clusters),
        "tasks": sorted({it["task"] for it in items}),
        "criteria": items,
        "similarity_matrix": sim.tolist(),
        "clusters": cluster_rows,
        "shared_criteria": shared,
        "task_specific_criteria": specific,
        "mean_cross_task_similarity": mean_cross,
        "mean_within_task_similarity": mean_within,
    }


def write_rubric_overlap(
    rubric_dirs: list[str | Path],
    out_dir: Path,
    embedding_model: str = "all-MiniLM-L6-v2",
    **kwargs,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = compute_rubric_overlap(rubric_dirs, embedding_model=embedding_model, **kwargs)

    (out_dir / "rubric_overlap.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    with open(out_dir / "rubric_clusters.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "size", "n_tasks", "tasks", "members"])
        for c in result["clusters"]:
            w.writerow([c["cluster"], c["size"], c["n_tasks"], "|".join(c["tasks"]),
                        "; ".join(f"{m['task']}/{m['criterion']}" for m in c["members"])])

    crit = result["criteria"]
    labels = [f"{c['task'][:14]}/{c['criterion'][:18]}" for c in crit]
    heat = {
        "div": "fig_heat",
        "data": [{
            "z": result["similarity_matrix"], "x": labels, "y": labels,
            "type": "heatmap", "colorscale": "Viridis", "zmin": 0, "zmax": 1,
            "colorbar": {"title": "cosine"},
        }],
        "layout": {"title": "Rubric criterion semantic similarity",
                   "xaxis": {"tickangle": -45, "automargin": True},
                   "yaxis": {"automargin": True}, "height": 600},
    }

    shared_rows = [[c["cluster"], c["n_tasks"], "; ".join(
        f"{m['task']}/{m['criterion']}" for m in c["members"])]
        for c in result["shared_criteria"]]
    shared_tbl = (table_html(["cluster", "# tasks", "criteria"], shared_rows)
                  if shared_rows else "<p class='note'>No cluster spanned the "
                  "shared-task threshold.</p>")

    cards = [
        {"heading": "Cross-task semantic similarity",
         "html": '<div id="fig_heat" class="fig" style="height:600px"></div>'
                 f'<p class="note">Mean cross-task similarity = '
                 f'{result["mean_cross_task_similarity"]:.3f}; mean within-task = '
                 f'{result["mean_within_task_similarity"]:.3f}. '
                 f'{result["n_criteria"]} criteria across '
                 f'{len(result["tasks"])} tasks grouped into '
                 f'{result["n_clusters"]} semantic clusters.</p>'},
        {"heading": "Shared (universal) quality dimensions", "html": shared_tbl},
    ]
    return write_page(
        out_dir,
        title="CoEval EXP-005: Rubric Generalization",
        subtitle=f"{result['n_criteria']} criteria | {len(result['tasks'])} tasks",
        cards=cards,
        figures=[heat],
    )
