"""Reports router — HTML report export and retrieval."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from packages.reports.report_builder import (KPI, ReportBuilder, Section,
                                             TableData, build_benchmark_report,
                                             build_comparison_report)

router = APIRouter()


class ExportRequest(BaseModel):
    run_ids: list[str] = []
    run_data: list[dict[str, Any]] = []  # Direct data if run_ids aren't in DB
    format: str = "html"
    title: str = "CyberForge Report"


class ExportResponse(BaseModel):
    report_id: str = ""
    report_path: str = ""
    success: bool = True
    error: Optional[str] = None


@router.post("/export", response_model=ExportResponse)
async def export_report(req: ExportRequest, request: Request):
    """Export benchmark results as an HTML report."""
    reports_dir = getattr(request.app.state, "config", None)
    if reports_dir and hasattr(reports_dir, "reports_dir"):
        output_dir = reports_dir.reports_dir
    else:
        output_dir = "reports"

    try:
        if len(req.run_data) == 1:
            path = build_benchmark_report(req.run_data[0], output_dir=output_dir)
        elif len(req.run_data) == 2:
            path = build_comparison_report(
                req.run_data[0], req.run_data[1], output_dir=output_dir,
            )
        elif req.run_data:
            # Multiple runs — build a combined report
            builder = ReportBuilder(title=req.title)
            for i, data in enumerate(req.run_data):
                summary = data.get("summary", {})
                model = data.get("model", f"Run {i+1}")
                eff = summary.get("efficiency", {})
                kpis = [
                    KPI("Model", model),
                    KPI("Latency P50", f"{eff.get('latency_ms_p50', 0):.0f} ms"),
                    KPI("Tokens/sec", f"{eff.get('tokens_per_sec', 0):.1f}"),
                ]
                builder.add_section(Section(heading=f"Run: {model}", kpis=kpis))
            path = builder.save(output_dir)
        else:
            return ExportResponse(success=False, error="No run data provided")

        report_id = path.stem.replace("report_", "")
        return ExportResponse(report_id=report_id, report_path=str(path), success=True)
    except Exception as e:
        return ExportResponse(success=False, error=str(e))


@router.get("/{report_id}")
async def get_report(report_id: str, request: Request):
    """Retrieve a generated report by ID."""
    reports_dir = "reports"
    cfg = getattr(request.app.state, "config", None)
    if cfg and hasattr(cfg, "reports_dir"):
        reports_dir = cfg.reports_dir

    # Search for matching report file
    rdir = Path(reports_dir)
    if not rdir.exists():
        return {"error": "Reports directory not found"}

    for f in rdir.glob("*.html"):
        if report_id in f.name:
            return FileResponse(str(f), media_type="text/html")

    return {"error": f"Report {report_id} not found"}


@router.get("/")
async def list_reports(request: Request):
    """List all generated reports."""
    reports_dir = "reports"
    cfg = getattr(request.app.state, "config", None)
    if cfg and hasattr(cfg, "reports_dir"):
        reports_dir = cfg.reports_dir

    rdir = Path(reports_dir)
    if not rdir.exists():
        return {"reports": []}

    reports = []
    for f in sorted(rdir.glob("*.html"), reverse=True):
        reports.append({
            "filename": f.name,
            "report_id": f.stem.replace("report_", ""),
            "size_bytes": f.stat().st_size,
            "modified": f.stat().st_mtime,
        })
    return {"reports": reports}
