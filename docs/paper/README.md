# CoEval Paper (HTML build)

This directory contains a self-contained academic paper, `index.html`, for the CoEval
LLM-evaluation framework, formatted for the journal TMLR. It is a single HTML file that
loads KaTeX from a CDN to render the math; no build step is required.

## Viewing locally

Just open `index.html` in any modern browser. An internet connection is needed the first
time so the KaTeX CSS/JS can load from the jsdelivr CDN.

## Publishing on GitHub Pages

1. Push this repository to GitHub (the `docs/paper/` directory must be committed).
2. In the repository, go to **Settings -> Pages**.
3. Under **Build and deployment -> Source**, choose **Deploy from a branch**.
4. Set the branch to **`master`** and the folder to **`/docs`**, then click **Save**.
5. GitHub builds the site. After a minute or two, your Pages URL appears at the top of the
   Pages settings (typically `https://<user>.github.io/<repo>/`).

Because the paper lives in `docs/paper/`, the published paper is served at:

```
<pages-url>/paper/
```

For this repository that resolves to:

```
https://apartsinprojects.github.io/CoEval/paper/
```

(`index.html` is served automatically for the `/paper/` path.)

## Filling in pending numbers

Table 1 (Section 5.1, ground-truth correlation) is a placeholder. The cells contain `—`
and the table is marked with an HTML comment:

```html
<!-- EXP-001 NUMBERS PENDING -->
```

Search for that comment in `index.html` and replace the `—` cells with the measured EXP-001
values (CoEval ensemble / best single judge / BERTScore / G-Eval across XSum, CNN-DM,
CodeSearchNet, and Overall). The bootstrap-CI cells marked `—` in Table 2 can likewise be
filled with the per-judge intervals if desired.
