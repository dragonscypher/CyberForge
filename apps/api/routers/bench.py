"""Benchmark endpoints — run baseline/optimized benchmarks, compare cards, suite management, IDS."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from packages.bench.harness import BenchmarkCard

router = APIRouter()


class RunBenchmarkRequest(BaseModel):
    model_id: str
    task_mode: str = "general"
    backend: str = "ollama"  # ollama | openrouter
    label: str = ""


class CompareRequest(BaseModel):
    baseline_file: str
    optimized_file: str
    save_report: bool = True


@router.post("/run", response_model=BenchmarkCard)
async def run_benchmark(body: RunBenchmarkRequest, request: Request):
    harness = request.app.state.bench

    label = body.label.strip()
    if not label or label.lower() == "baseline":
        return await harness.run_baseline(
            model_id=body.model_id,
            task_mode=body.task_mode,
            backend=body.backend,
        )

    return await harness.run_optimized(
        model_id=body.model_id,
        task_mode=body.task_mode,
        label=label,
        backend=body.backend,
    )


@router.get("/cards", response_model=list[str])
async def list_cards(request: Request):
    return request.app.state.bench.list_cards()


@router.get("/cards/{file_name}", response_model=BenchmarkCard)
async def get_card(file_name: str, request: Request):
    card = request.app.state.bench.load_card(file_name)
    if card is None:
        raise HTTPException(404, f"Benchmark card '{file_name}' not found")
    return card


@router.post("/compare")
async def compare_cards(body: CompareRequest, request: Request):
    harness = request.app.state.bench
    result = await harness.compare_cards(
        baseline_file=body.baseline_file,
        optimized_file=body.optimized_file,
    )
    if result is None:
        raise HTTPException(404, "One or both benchmark cards were not found")

    report_path: str | None = None
    if body.save_report:
        reports_dir = Path(request.app.state.config.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        file_name = f"compare-{stamp}.json"
        out_path = reports_dir / file_name
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        report_path = str(out_path)

    return {
        "comparison": result,
        "report_path": report_path,
    }


# ── Suite management (BENCH-001) ────────────────────────────────


@router.get("/suites")
async def list_suites():
    """List all registered benchmark suites."""
    from packages.bench.suites import list_suites as _list

    suites = _list()
    return {
        "suites": [
            {"id": s.id, "name": s.name, "task_mode": s.task_mode, "description": s.description}
            for s in suites
        ]
    }


@router.get("/suites/{suite_id}")
async def get_suite(suite_id: str):
    """Get full suite configuration."""
    from packages.bench.suites import get_suite as _get

    suite = _get(suite_id)
    if suite is None:
        raise HTTPException(404, f"Suite '{suite_id}' not found")
    return suite.model_dump()


class IDSBenchmarkRequest(BaseModel):
    suite_id: str = "ids_structured_nslkdd_v1"
    classifier: str = "rf"
    n_estimators: int = 100


@router.post("/ids/run")
async def run_ids_benchmark(body: IDSBenchmarkRequest):
    """Run an IDS-type benchmark suite (NSL-KDD, cross-dataset transfer, etc.)."""
    from packages.bench.suites import get_suite as _get
    from packages.bench.suites import run_ids_suite

    suite = _get(body.suite_id)
    if suite is None:
        raise HTTPException(404, f"Suite '{body.suite_id}' not found")

    if suite.task_mode != "ids":
        raise HTTPException(400, f"Suite '{body.suite_id}' is not an IDS suite")

    # Override classifier if provided
    if body.classifier:
        suite.ids_config["classifiers"] = [body.classifier]
    if body.n_estimators:
        suite.ids_config["n_estimators"] = body.n_estimators

    summary = await run_ids_suite(suite)
    return {
        "suite_id": suite.id,
        "suite_name": suite.name,
        "summary": summary.model_dump(),
    }


@router.post("/ids/quick")
async def quick_ids_benchmark(classifier: str = "rf", n_estimators: int = 100):
    """Quick IDS benchmark: run RF or XGB on NSL-KDD directly."""
    from packages.bench.ids_runner import IDSStructuredRunner

    runner = IDSStructuredRunner()
    result = runner.run_nsl_kdd(classifier=classifier, n_estimators=n_estimators)
    return result.model_dump()


# ── Coding benchmark (BENCH-003) ────────────────────────────────


@router.post("/coding/self-test")
async def coding_self_test():
    """Run coding benchmark self-test with known-good code (no LLM needed)."""
    from packages.bench.coding_runner import self_test

    result = self_test()
    return result.model_dump()


class CodingBenchmarkRequest(BaseModel):
    model_id: str
    backend: str = "ollama"
    k: int = 1


@router.post("/coding/run")
async def run_coding_benchmark(body: CodingBenchmarkRequest, request: Request):
    """Run coding benchmark against an LLM model (BENCH-003)."""
    from packages.bench.coding_runner import CodingBenchmarkRunner

    harness = request.app.state.bench

    async def _infer(prompt: str) -> str:
        return await harness._infer(
            backend=body.backend,
            model=body.model_id,
            prompt=prompt,
        )

    runner = CodingBenchmarkRunner()
    result = await runner.run(infer_fn=_infer, k=body.k)
    return result.model_dump()


# ── Cyber benchmark (BENCH-004) ─────────────────────────────────


@router.post("/cyber/self-test")
async def cyber_self_test():
    """Run cyber benchmark self-test with known-good artifacts (no LLM needed)."""
    from packages.bench.cyber_runner import self_test

    result = self_test()
    return result.model_dump()


class CyberBenchmarkRequest(BaseModel):
    model_id: str
    backend: str = "ollama"


@router.post("/cyber/run")
async def run_cyber_benchmark(body: CyberBenchmarkRequest, request: Request):
    """Run cyber security benchmark against an LLM model (BENCH-004)."""
    from packages.bench.cyber_runner import CyberBenchmarkRunner

    harness = request.app.state.bench

    async def _infer(prompt: str) -> str:
        return await harness._infer(
            backend=body.backend,
            model=body.model_id,
            prompt=prompt,
        )

    runner = CyberBenchmarkRunner()
    result = await runner.run(infer_fn=_infer)
    return result.model_dump()
