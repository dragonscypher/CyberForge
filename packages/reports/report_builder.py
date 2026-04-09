"""HTML report builder using Jinja2.

Generates self-contained HTML reports with embedded charts and metrics tables.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import BaseLoader, Environment

log = logging.getLogger(__name__)

_REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<style>
  :root { --bg: #0d1117; --fg: #c9d1d9; --accent: #58a6ff; --card: #161b22; --border: #30363d; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial; background: var(--bg); color: var(--fg); padding: 2rem; line-height: 1.6; }
  h1 { color: var(--accent); margin-bottom: 0.5rem; }
  h2 { color: var(--accent); margin: 1.5rem 0 0.75rem; border-bottom: 1px solid var(--border); padding-bottom: 0.25rem; }
  h3 { color: #8b949e; margin: 1rem 0 0.5rem; }
  .meta { color: #8b949e; font-size: 0.9rem; margin-bottom: 2rem; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }
  table { width: 100%; border-collapse: collapse; margin: 0.5rem 0; }
  th, td { text-align: left; padding: 0.5rem 1rem; border-bottom: 1px solid var(--border); }
  th { color: var(--accent); font-weight: 600; }
  td { font-variant-numeric: tabular-nums; }
  .good { color: #3fb950; } .warn { color: #d29922; } .bad { color: #f85149; }
  .chart-container { text-align: center; margin: 1rem 0; }
  .chart-container img { max-width: 100%; height: auto; border-radius: 8px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }
  .kpi { text-align: center; }
  .kpi .value { font-size: 2rem; font-weight: 700; color: var(--accent); }
  .kpi .label { font-size: 0.85rem; color: #8b949e; }
  footer { margin-top: 3rem; color: #484f58; text-align: center; font-size: 0.8rem; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<p class="meta">Generated {{ timestamp }} &middot; Report ID: {{ report_id }}</p>

{% for section in sections %}
<h2>{{ section.heading }}</h2>

{% if section.kpis %}
<div class="grid">
{% for kpi in section.kpis %}
<div class="card kpi">
  <div class="value {{ kpi.css_class }}">{{ kpi.value }}</div>
  <div class="label">{{ kpi.label }}</div>
</div>
{% endfor %}
</div>
{% endif %}

{% if section.table %}
<div class="card">
<table>
<thead><tr>{% for col in section.table.columns %}<th>{{ col }}</th>{% endfor %}</tr></thead>
<tbody>
{% for row in section.table.rows %}
<tr>{% for cell in row %}<td>{{ cell }}</td>{% endfor %}</tr>
{% endfor %}
</tbody>
</table>
</div>
{% endif %}

{% if section.charts %}
{% for chart_b64 in section.charts %}
<div class="chart-container"><img src="{{ chart_b64 }}" alt="Chart"></div>
{% endfor %}
{% endif %}

{% if section.text %}
<div class="card">{{ section.text }}</div>
{% endif %}

{% endfor %}

<footer>CyberForge &copy; {{ year }} &mdash; Generated automatically</footer>
</body>
</html>
"""


class KPI:
    def __init__(self, label: str, value: str, css_class: str = ""):
        self.label = label
        self.value = value
        self.css_class = css_class


class TableData:
    def __init__(self, columns: list[str], rows: list[list[str]]):
        self.columns = columns
        self.rows = rows


class Section:
    def __init__(
        self,
        heading: str,
        kpis: list[KPI] | None = None,
        table: TableData | None = None,
        charts: list[str] | None = None,
        text: str | None = None,
    ):
        self.heading = heading
        self.kpis = kpis
        self.table = table
        self.charts = charts
        self.text = text


class ReportBuilder:
    """Build HTML reports from sections and chart data."""

    def __init__(self, title: str = "CyberForge Report"):
        self._title = title
        self._sections: list[Section] = []
        self._report_id = str(uuid.uuid4())

    def add_section(self, section: Section) -> "ReportBuilder":
        self._sections.append(section)
        return self

    def render(self) -> str:
        """Render the full HTML report string."""
        env = Environment(loader=BaseLoader(), autoescape=True)
        template = env.from_string(_REPORT_TEMPLATE)
        return template.render(
            title=self._title,
            report_id=self._report_id,
            timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            year=datetime.utcnow().year,
            sections=self._sections,
        )

    def save(self, output_dir: str | Path, filename: str | None = None) -> Path:
        """Render and save to disk. Returns the output path."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = filename or f"report_{self._report_id[:8]}.html"
        path = output_dir / fname
        path.write_text(self.render(), encoding="utf-8")
        log.info("Report saved to %s", path)
        return path


def build_benchmark_report(
    run_data: dict[str, Any],
    charts: list[str] | None = None,
    output_dir: str | Path = "reports",
) -> Path:
    """Build a complete benchmark report from run data."""
    summary = run_data.get("summary", {})
    model = run_data.get("model", "unknown")
    suite = run_data.get("suite", "unknown")

    builder = ReportBuilder(title=f"Benchmark Report: {model}")

    # Overview KPIs
    efficiency = summary.get("efficiency", {})
    quality = summary.get("quality", {})
    reliability = summary.get("reliability", {})

    kpis = [
        KPI("Latency P50", f"{efficiency.get('latency_ms_p50', 0):.0f} ms"),
        KPI("Tokens/sec", f"{efficiency.get('tokens_per_sec', 0):.1f}"),
        KPI("RAM Peak", f"{efficiency.get('ram_peak_mb', 0):.0f} MB"),
    ]
    if quality.get("f1_macro"):
        kpis.append(KPI("F1 Macro", f"{quality['f1_macro']:.3f}"))
    if reliability.get("verifier_pass_rate") is not None:
        rate = reliability["verifier_pass_rate"]
        css = "good" if rate >= 0.8 else "warn" if rate >= 0.5 else "bad"
        kpis.append(KPI("Verifier Pass", f"{rate:.1%}", css))

    builder.add_section(Section(heading="Overview", kpis=kpis))

    # Detail tables
    if efficiency:
        rows = [[k, f"{v:.2f}" if isinstance(v, float) else str(v)] for k, v in efficiency.items()]
        builder.add_section(Section(
            heading="Efficiency Metrics",
            table=TableData(columns=["Metric", "Value"], rows=rows),
        ))

    if quality:
        rows = [[k, f"{v:.4f}" if isinstance(v, float) else str(v)] for k, v in quality.items()]
        builder.add_section(Section(
            heading="Quality Metrics",
            table=TableData(columns=["Metric", "Value"], rows=rows),
        ))

    if reliability:
        rows = [[k, f"{v:.4f}" if isinstance(v, float) else str(v)] for k, v in reliability.items()]
        builder.add_section(Section(
            heading="Reliability Metrics",
            table=TableData(columns=["Metric", "Value"], rows=rows),
        ))

    # Charts
    if charts:
        builder.add_section(Section(heading="Charts", charts=charts))

    return builder.save(output_dir)


def build_comparison_report(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    charts: list[str] | None = None,
    output_dir: str | Path = "reports",
) -> Path:
    """Build a comparison report between baseline and candidate runs."""
    builder = ReportBuilder(title="Model Comparison Report")

    b_summary = baseline.get("summary", {})
    c_summary = candidate.get("summary", {})

    # Efficiency comparison
    b_eff = b_summary.get("efficiency", {})
    c_eff = c_summary.get("efficiency", {})
    all_keys = sorted(set(list(b_eff.keys()) + list(c_eff.keys())))
    if all_keys:
        rows = []
        for k in all_keys:
            bv = b_eff.get(k, "-")
            cv = c_eff.get(k, "-")
            bstr = f"{bv:.2f}" if isinstance(bv, float) else str(bv)
            cstr = f"{cv:.2f}" if isinstance(cv, float) else str(cv)
            rows.append([k, bstr, cstr])
        builder.add_section(Section(
            heading="Efficiency Comparison",
            table=TableData(columns=["Metric", "Baseline", "Candidate"], rows=rows),
        ))

    # Quality comparison
    b_qual = b_summary.get("quality", {})
    c_qual = c_summary.get("quality", {})
    all_keys = sorted(set(list(b_qual.keys()) + list(c_qual.keys())))
    if all_keys:
        rows = []
        for k in all_keys:
            bv = b_qual.get(k, "-")
            cv = c_qual.get(k, "-")
            bstr = f"{bv:.4f}" if isinstance(bv, float) else str(bv)
            cstr = f"{cv:.4f}" if isinstance(cv, float) else str(cv)
            rows.append([k, bstr, cstr])
        builder.add_section(Section(
            heading="Quality Comparison",
            table=TableData(columns=["Metric", "Baseline", "Candidate"], rows=rows),
        ))

    if charts:
        builder.add_section(Section(heading="Charts", charts=charts))

    return builder.save(output_dir)
