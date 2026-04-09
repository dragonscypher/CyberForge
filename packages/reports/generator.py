"""Report generator — high-level API for generating CyberForge reports.

Wraps the lower-level ReportBuilder and chart functions into a simple
generate() call that produces complete HTML reports from benchmark data.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from packages.reports.charts import (calibration_chart,
                                     latency_comparison_chart,
                                     quality_radar_chart, resource_usage_chart)
from packages.reports.report_builder import (KPI, ReportBuilder, Section,
                                             TableData, build_benchmark_report,
                                             build_comparison_report)

log = logging.getLogger(__name__)

__all__ = [
    "generate_report",
    "generate_comparison",
    "generate_multi_run_report",
]

_REPORT_DIR = Path("reports")


def generate_report(
    run_data: dict[str, Any],
    title: str = "CyberForge Benchmark Report",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a single-run benchmark report.

    Args:
        run_data: Benchmark result dict with quality/reliability/efficiency sections.
        title: Report title.
        output_dir: Where to save the HTML file.  Defaults to ./reports/.

    Returns:
        Dict with report_id, report_path, success.
    """
    out_dir = Path(output_dir) if output_dir else _REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        html = build_benchmark_report(run_data)
        report_id = f"report_{uuid.uuid4().hex[:8]}"
        out_path = out_dir / f"{report_id}.html"
        out_path.write_text(html, encoding="utf-8")
        return {"report_id": report_id, "report_path": str(out_path), "success": True}
    except Exception as e:
        log.exception("Report generation failed")
        return {"report_id": "", "report_path": "", "success": False, "error": str(e)}


def generate_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    title: str = "CyberForge Comparison Report",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a side-by-side comparison report."""
    out_dir = Path(output_dir) if output_dir else _REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        html = build_comparison_report(baseline, candidate)
        report_id = f"compare_{uuid.uuid4().hex[:8]}"
        out_path = out_dir / f"{report_id}.html"
        out_path.write_text(html, encoding="utf-8")
        return {"report_id": report_id, "report_path": str(out_path), "success": True}
    except Exception as e:
        log.exception("Comparison report failed")
        return {"report_id": "", "report_path": "", "success": False, "error": str(e)}


def generate_multi_run_report(
    runs: list[dict[str, Any]],
    title: str = "CyberForge Multi-Run Report",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a report covering multiple benchmark runs with charts."""
    out_dir = Path(output_dir) if output_dir else _REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        builder = ReportBuilder(title=title)

        # Overview KPIs from first run
        if runs:
            first = runs[0]
            kpis = []
            eff = first.get("efficiency", {})
            qual = first.get("quality", {})
            if "latency_p50_ms" in eff:
                kpis.append(KPI(label="P50 Latency", value=f"{eff['latency_p50_ms']:.0f} ms"))
            if "f1_macro" in qual:
                kpis.append(KPI(label="F1 Macro", value=f"{qual['f1_macro']:.4f}"))
            if kpis:
                builder.add_section(Section(heading="Overview", kpis=kpis))

        # Per-run summary table
        rows = []
        for i, run in enumerate(runs):
            q = run.get("quality", {})
            e = run.get("efficiency", {})
            rows.append([
                run.get("model", run.get("dataset", f"Run {i+1}")),
                f"{q.get('f1_macro', 0):.4f}",
                f"{q.get('accuracy', 0):.4f}",
                f"{e.get('train_time_ms', e.get('latency_p50_ms', 0)):.0f}",
            ])

        if rows:
            table = TableData(
                columns=["Model/Dataset", "F1 Macro", "Accuracy", "Time (ms)"],
                rows=rows,
            )
            builder.add_section(Section(heading="Run Summary", table=table))

        # Charts
        charts: list[str] = []
        try:
            latencies = {
                r.get("model", f"run{i}"): {
                    "p50": r.get("efficiency", {}).get("latency_p50_ms", 0),
                    "p95": r.get("efficiency", {}).get("latency_p95_ms", 0),
                }
                for i, r in enumerate(runs)
            }
            if any(v["p50"] > 0 for v in latencies.values()):
                charts.append(latency_comparison_chart(latencies))
        except Exception:
            pass

        if charts:
            builder.add_section(Section(heading="Charts", charts=charts))

        html = builder.render()
        report_id = f"multi_{uuid.uuid4().hex[:8]}"
        out_path = out_dir / f"{report_id}.html"
        out_path.write_text(html, encoding="utf-8")
        return {"report_id": report_id, "report_path": str(out_path), "success": True}
    except Exception as e:
        log.exception("Multi-run report failed")
        return {"report_id": "", "report_path": "", "success": False, "error": str(e)}
