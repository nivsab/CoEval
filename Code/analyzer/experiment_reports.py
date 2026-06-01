"""Self-contained HTML rendering for the experiment analyzer subcommands.

Produces single-file Plotly pages (no CDN) for the ensemble-ablation,
verbosity-bias, positional-bias and rubric-overlap reports.  Each page bundles
a local ``plotly.min.js`` (via :func:`get_plotly_js`) so the report opens
offline, matching the convention of the eight core reports.
"""
from __future__ import annotations

import html as _html
import json
from pathlib import Path
from typing import Any

from .reports.html_base import get_plotly_js


_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<script src="plotly.min.js"></script>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0; padding: 0 0 60px; color: #1a1a1a; background: #fafafa; }}
  header {{ background: #1f2a44; color: #fff; padding: 22px 32px; }}
  header h1 {{ margin: 0; font-size: 21px; }}
  header p {{ margin: 6px 0 0; opacity: .8; font-size: 13px; }}
  .wrap {{ max-width: 1100px; margin: 24px auto; padding: 0 24px; }}
  .card {{ background: #fff; border: 1px solid #e3e6ee; border-radius: 10px;
          padding: 20px 24px; margin-bottom: 22px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
  .card h2 {{ margin: 0 0 14px; font-size: 16px; color: #1f2a44; }}
  .summary {{ font-size: 14px; line-height: 1.6; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border: 1px solid #e3e6ee; padding: 7px 10px; text-align: right; }}
  th {{ background: #f1f3f9; color: #1f2a44; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .fig {{ width: 100%; height: 420px; }}
  .note {{ font-size: 12px; color: #667; margin-top: 8px; }}
  code {{ background: #f1f3f9; padding: 1px 5px; border-radius: 4px; }}
</style>
</head>
<body>
<header><h1>{title}</h1><p>{subtitle}</p></header>
<div class="wrap">{body}</div>
<script>
const FIGS = {figs_json};
for (const f of FIGS) {{
  Plotly.newPlot(f.div, f.data, f.layout, {{responsive: true, displaylogo: false}});
}}
</script>
</body>
</html>
"""


def _esc(s: Any) -> str:
    return _html.escape(str(s))


def table_html(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a simple HTML table.  Cells are HTML-escaped."""
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_page(
    out_dir: Path,
    title: str,
    subtitle: str,
    cards: list[dict],
    figures: list[dict],
) -> Path:
    """Write a self-contained report page to ``out_dir/index.html``.

    ``cards`` is a list of ``{"heading": str, "html": str}`` blocks rendered in
    order.  ``figures`` is a list of ``{"div": str, "data": [...], "layout":
    {...}}`` Plotly specs whose ``div`` ids must appear in some card's HTML.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    get_plotly_js(out_dir)  # writes plotly.min.js sibling (cached)

    body = ""
    for c in cards:
        body += f'<div class="card"><h2>{_esc(c["heading"])}</h2>{c["html"]}</div>'

    page = _PAGE.format(
        title=_esc(title),
        subtitle=_esc(subtitle),
        body=body,
        figs_json=json.dumps(figures, ensure_ascii=False),
    )
    index_path = out_dir / "index.html"
    index_path.write_text(page, encoding="utf-8")
    return index_path
