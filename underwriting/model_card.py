"""Build a single, self-contained HTML model card from pipeline artifacts.

The card embeds figures as base64 data URIs so the resulting ``model_card.html``
is fully portable: one file, no external assets, shareable as-is.
"""

from __future__ import annotations

import base64
import html
from collections.abc import Mapping
from pathlib import Path
from typing import Any

FIGURES = (
    ("Reliability diagram", "reliability_diagram.png"),
    ("Coverage vs. performance", "coverage_vs_performance.png"),
    ("Confusion matrix", "confusion_matrix.png"),
    ("Precision-recall curve", "precision_recall_curve.png"),
    ("Probability histograms", "probability_histograms.png"),
    ("Slice review rates", "slice_review_rates.png"),
    ("Slice error rates", "slice_error_rates.png"),
)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _embed_png(path: Path) -> str | None:
    if not path.exists():
        return None
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _metric_cards(metrics: Mapping[str, Any]) -> str:
    items = [
        ("Accuracy", metrics.get("accuracy")),
        ("F1", metrics.get("f1")),
        ("ROC AUC", metrics.get("roc_auc")),
        ("Avg. precision", metrics.get("average_precision")),
        ("Brier", metrics.get("brier")),
        ("ECE", metrics.get("ece")),
    ]
    cells = "".join(
        f'<div class="card"><div class="card-value">{_fmt(v)}</div>'
        f'<div class="card-label">{html.escape(label)}</div></div>'
        for label, v in items
    )
    return f'<div class="card-grid">{cells}</div>'


def _policy_table(policy: Mapping[str, Any]) -> str:
    rows = [
        ("Recommended threshold", _fmt(policy.get("recommended_threshold"))),
        ("Expected coverage", _fmt(policy.get("expected_coverage"))),
        ("Auto-decision accuracy", _fmt(policy.get("expected_accuracy_auto"))),
        ("Auto-decision F1", _fmt(policy.get("expected_f1_auto"))),
        ("Target coverage", _fmt(policy.get("target_coverage"))),
        ("Calibration method", _fmt(policy.get("calibration_method"))),
    ]
    body = "".join(f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


def _baseline_table(baseline_metrics: Mapping[str, Any]) -> str:
    head = "<tr><th>Baseline</th><th>Accuracy</th><th>F1</th><th>Brier</th></tr>"
    rows = []
    for name, vals in baseline_metrics.items():
        if not isinstance(vals, Mapping):
            continue
        rows.append(
            f"<tr><th>{html.escape(str(name))}</th>"
            f"<td>{_fmt(vals.get('accuracy'))}</td>"
            f"<td>{_fmt(vals.get('f1'))}</td>"
            f"<td>{_fmt(vals.get('brier'))}</td></tr>"
        )
    return f"<table>{head}{''.join(rows)}</table>"


def _slice_table(slice_summary: Mapping[str, Any]) -> str:
    rows = [
        ("Slices evaluated", _fmt(slice_summary.get("n_slices"))),
        ("Max auto-decision rate gap", _fmt(slice_summary.get("max_auto_decision_rate_gap"))),
        ("Max error rate gap", _fmt(slice_summary.get("max_error_rate_gap"))),
        ("Max ECE gap", _fmt(slice_summary.get("max_ece_gap"))),
        ("Small slices", _fmt(slice_summary.get("small_slice_count"))),
    ]
    body = "".join(f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


def _provenance(manifest: Mapping[str, Any] | None) -> str:
    if not manifest:
        return ""
    deps = manifest.get("dependency_versions", {})
    dep_str = ", ".join(f"{k} {v}" for k, v in deps.items()) if isinstance(deps, Mapping) else ""
    rows = [
        ("Generated at", _fmt(manifest.get("generated_at"))),
        ("Package version", _fmt(manifest.get("package_version"))),
        ("Python", _fmt(manifest.get("python_version"))),
        ("Input", _fmt(manifest.get("input_path"))),
        ("Input SHA-256", _fmt(manifest.get("input_sha256"))),
        ("Input rows", _fmt(manifest.get("input_rows"))),
        ("Random state", _fmt(manifest.get("random_state"))),
        ("Libraries", html.escape(dep_str)),
    ]
    body = "".join(f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>" for k, v in rows)
    return f'<h2>Provenance</h2><table class="prov">{body}</table>'


def _figures_section(figures_dir: Path) -> str:
    blocks = []
    for title, fname in FIGURES:
        uri = _embed_png(figures_dir / fname)
        if uri is None:
            continue
        blocks.append(
            f"<figure><figcaption>{html.escape(title)}</figcaption>"
            f'<img alt="{html.escape(title)}" src="{uri}" /></figure>'
        )
    if not blocks:
        return ""
    return f'<h2>Figures</h2><div class="figures">{"".join(blocks)}</div>'


_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  margin: 0; padding: 2rem; color: #1b1f24; background: #f6f7f9; line-height: 1.45; }
.wrap { max-width: 980px; margin: 0 auto; background: #fff; padding: 2rem 2.4rem;
  border: 1px solid #e3e6ea; border-radius: 16px; }
h1 { margin: 0 0 .25rem; font-size: 1.6rem; }
.subtitle { color: #5b6470; margin: 0 0 1.5rem; }
h2 { margin: 2rem 0 .75rem; font-size: 1.15rem;
  border-bottom: 1px solid #eceef1; padding-bottom: .4rem; }
.card-grid { display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: .75rem; }
.card { background: #f3f5f8; border: 1px solid #e3e6ea; border-radius: 12px;
  padding: .9rem; text-align: center; }
.card-value { font-size: 1.5rem; font-weight: 650; }
.card-label { color: #5b6470; font-size: .8rem; margin-top: .2rem; }
table { border-collapse: collapse; width: 100%; margin: .25rem 0; }
th, td { text-align: left; padding: .45rem .6rem;
  border-bottom: 1px solid #eceef1; font-size: .92rem; }
table th:first-child { color: #5b6470; font-weight: 600; width: 45%; }
table.prov td { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: .82rem; word-break: break-all; }
.figures { display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.2rem; }
figure { margin: 0; border: 1px solid #e3e6ea; border-radius: 12px;
  padding: .8rem; background: #fff; }
figcaption { font-size: .85rem; color: #5b6470; margin-bottom: .5rem; }
img { width: 100%; height: auto; display: block; }
footer { margin-top: 2rem; color: #8a939e; font-size: .8rem; }
"""


def build_model_card_html(
    *,
    metrics: Mapping[str, Any],
    policy: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    slice_summary: Mapping[str, Any],
    data_quality: Mapping[str, Any],
    figures_dir: str | Path,
    manifest: Mapping[str, Any] | None = None,
) -> str:
    """Assemble a portable, single-file HTML model card."""
    figures_path = Path(figures_dir)
    rows = data_quality.get("rows")
    cols = data_quality.get("cols")
    dq_line = f"{_fmt(rows)} rows x {_fmt(cols)} columns" if rows is not None else "n/a"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Underwriting Model Card</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
<h1>Underwriting Decision Safety &mdash; Model Card</h1>
<p class="subtitle">Calibrated approval model with an abstention/coverage policy and
slice-level safety diagnostics. Dataset: {dq_line}.</p>

<h2>Headline metrics</h2>
{_metric_cards(metrics)}

<h2>Recommended decision policy</h2>
{_policy_table(policy)}

<h2>Baselines</h2>
{_baseline_table(baseline_metrics)}

<h2>Slice safety</h2>
{_slice_table(slice_summary)}

{_figures_section(figures_path)}

{_provenance(manifest)}

<footer>These are monitoring diagnostics, not a fairness certification. The
abstention policy routes low-confidence cases to manual review.</footer>
</div>
</body>
</html>
"""
