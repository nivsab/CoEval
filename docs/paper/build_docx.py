#!/usr/bin/env python
"""Build CoEval.docx from index.html using the html2doc skill.

As of the 2026-06 html2doc robustness update, the skill itself:
  - strips screen-only chrome (the web-only "Download .docx" link, any
    class="no-docx"/"no-print" or data-docx="ignore" element);
  - normalizes <br/> inside the title so it does not render "TasksWithout";
  - styles every author/affiliation line uniformly (both authors, one font);
  - scales oversized figures to fit the page (no blank-space pages).

So this build needs NO per-project pre/post processing; it is just the standard
three-stage pipeline. Run from docs/paper/:  python build_docx.py
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILL = Path(os.environ.get("HTML2DOC_SKILL", r"C:/Users/apart/.claude/skills/html2doc"))
SRC = HERE / "index.html"
OUT = HERE / "CoEval.docx"
MML = HERE / "_mathml.html"
RAW = HERE / "_converted.docx"

env = dict(os.environ, NODE_PATH=str(SKILL / "node_modules"))
subprocess.run(["node", str(SKILL / "scripts/katex_to_mathml.js"),
                "--input", str(SRC), "--output", str(MML)], check=True, env=env)
subprocess.run([sys.executable, str(SKILL / "scripts/convert_to_docx.py"),
                "--input", str(MML), "--output", str(RAW),
                "--profile", "camera-ready-generic"], check=True)
subprocess.run([sys.executable, str(SKILL / "scripts/apply_academic_style.py"),
                "--input", str(RAW), "--output", str(OUT),
                "--profile", "camera-ready-generic"], check=True)

for f in (MML, RAW):
    try:
        f.unlink()
    except FileNotFoundError:
        pass
print(f"Built {OUT}")
