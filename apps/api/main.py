"""CyberForge API — FastAPI entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.db import build_engine, build_session_factory, init_db
from apps.api.routers import (bench, config, cyber, hardware, jobs, lifecycle,
                              models, optimize, recommend, reports, serve,
                              settings, train)
from apps.api.workers.runner import JobRunner
from packages.bench.harness import BenchmarkHarness
from packages.core.config import (AppConfig, is_first_run, load_config,
                                  save_config)
from packages.core.lifecycle import LifecycleManager
from packages.core.orchestrator import Orchestrator
from packages.hardware.profiler import HardwareProfiler
from packages.models.downloader import HFDownloader
from packages.models.registry import ModelRegistry
from packages.serve.ollama import OllamaClient
from packages.serve.openrouter import OpenRouterClient
from packages.serve.tensorrt_llm import TensorRTLLMClient
from packages.serve.vllm import VLLMClient


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: load config, init singletons, DB, optional volatile cleanup."""
    cfg = load_config()
    app.state.config = cfg

    # Database
    engine = build_engine(cfg.db_path if hasattr(cfg, "db_path") else None)
    await init_db(engine)
    app.state.db_engine = engine
    app.state.db_session_factory = build_session_factory(engine)

    app.state.registry = ModelRegistry()
    app.state.profiler = HardwareProfiler(disk_path=cfg.cache_dir)
    app.state.downloader = HFDownloader(cache_dir=cfg.cache_dir, token=cfg.hf_token)
    app.state.ollama = OllamaClient(base_url=cfg.ollama_base_url)
    app.state.openrouter = OpenRouterClient(api_key=cfg.openrouter_key)
    app.state.vllm = VLLMClient(
        base_url=getattr(cfg, "vllm_base_url", "http://localhost:8000"),
    )
    app.state.tensorrt = TensorRTLLMClient(
        base_url=getattr(cfg, "tensorrt_base_url", "http://localhost:8001"),
    )
    app.state.lifecycle = LifecycleManager(
        cache_dir=cfg.cache_dir,
        saved_dir=cfg.saved_models_dir,
    )
    app.state.bench = BenchmarkHarness(
        reports_dir=cfg.reports_dir,
        ollama_client=app.state.ollama,
        openrouter_client=app.state.openrouter,
        profiler=app.state.profiler,
    )
    # Job runner
    job_runner = JobRunner(app.state.db_session_factory)
    app.state.job_runner = job_runner
    await job_runner.start()

    # Pipeline orchestrator
    app.state.orchestrator = Orchestrator(
        config=cfg,
        profiler=app.state.profiler,
        registry=app.state.registry,
        downloader=app.state.downloader,
        ollama=app.state.ollama,
        bench=app.state.bench,
        lifecycle=app.state.lifecycle,
        job_runner=job_runner,
    )

    # Auto-cleanup volatile artifacts from interrupted prior sessions
    if cfg.auto_cleanup_cache:
        app.state.lifecycle.cleanup_volatile()
    yield
    # Shutdown: stop job runner, cleanup volatile, dispose DB engine
    await app.state.job_runner.stop()
    app.state.lifecycle.cleanup_volatile()
    await app.state.db_engine.dispose()


app = FastAPI(
    title="CyberForge",
    version="0.1.0",
    description=(
        "Cyber-first local AI workbench — hardware profiling, model recommendation, "
        "optimization, benchmarking, and verifier-backed cyber evaluation."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health endpoint ─────────────────────────────────────────────
@app.get("/health")
async def health():
    """Health-check endpoint per plan.md §5.1."""
    return {"status": "ok", "version": "0.1.0"}


# ── Register API routers ────────────────────────────────────────
app.include_router(hardware.router, prefix="/api/hardware", tags=["Hardware"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(serve.router, prefix="/api/serve", tags=["Serve"])
app.include_router(recommend.router, prefix="/api/recommend", tags=["Recommend"])
app.include_router(config.router, prefix="/api/config", tags=["Config"])
app.include_router(lifecycle.router, prefix="/api/lifecycle", tags=["Lifecycle"])
app.include_router(settings.router, prefix="/api/v1/settings", tags=["Settings"])
app.include_router(bench.router, prefix="/api/bench", tags=["Bench"])
app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["Jobs"])
app.include_router(optimize.router, prefix="/api/optimize", tags=["Optimize"])
app.include_router(train.router, prefix="/api/train", tags=["Train"])
app.include_router(cyber.router, prefix="/api/cyber", tags=["Cyber"])
app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])


# ── Mount the Web UI on the same server ─────────────────────────
import sys as _sys
from pathlib import Path as _Path

_project_root = str(_Path(__file__).resolve().parent.parent.parent)
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from ui import build_ui_app  # noqa: E402

_ui_app = build_ui_app(app)
app.mount("/", _ui_app)
