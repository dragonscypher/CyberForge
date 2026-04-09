"""Database layer — SQLAlchemy models, engine factory, and session helpers.

Uses SQLite for metadata. Large benchmark traces go to JSONL/Parquet files
referenced by path columns in the DB.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

from sqlalchemy import ForeignKey, String, Text, event
from sqlalchemy.ext.asyncio import (AsyncAttrs, AsyncSession,
                                    async_sessionmaker, create_async_engine)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


# ── Base ─────────────────────────────────────────────────────────
class Base(AsyncAttrs, DeclarativeBase):
    pass


# ── Tables ───────────────────────────────────────────────────────

class Settings(Base):
    __tablename__ = "settings"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    model_cache_dir: Mapped[str] = mapped_column(Text)
    allow_remote_providers: Mapped[int] = mapped_column(default=0)
    default_task_mode: Mapped[str] = mapped_column(Text, default="general")
    privacy_mode: Mapped[str] = mapped_column(Text, default="prefer_local")
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)


class ProviderAccount(Base):
    __tablename__ = "provider_accounts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    provider_type: Mapped[str] = mapped_column(Text)
    label: Mapped[str] = mapped_column(Text)
    secret_ref: Mapped[str] = mapped_column(Text)  # keyring handle only
    enabled: Mapped[int] = mapped_column(default=1)
    meta_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)


class HardwareProfileRow(Base):
    __tablename__ = "hardware_profiles"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    hostname: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    os_name: Mapped[str] = mapped_column(Text)
    os_version: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cpu_model: Mapped[str] = mapped_column(Text)
    cpu_cores: Mapped[int] = mapped_column()
    cpu_threads: Mapped[int] = mapped_column()
    ram_bytes: Mapped[int] = mapped_column()
    gpu_vendor: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    gpu_model: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    vram_bytes: Mapped[Optional[int]] = mapped_column(nullable=True)
    cuda_version: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    driver_version: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    capabilities_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    benchmark_runs: Mapped[list[BenchmarkRun]] = relationship(back_populates="hardware_profile")


class ModelCatalog(Base):
    __tablename__ = "model_catalog"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    source: Mapped[str] = mapped_column(Text)
    source_ref: Mapped[str] = mapped_column(Text)
    display_name: Mapped[str] = mapped_column(Text)
    family: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    task_modes_json: Mapped[str] = mapped_column(Text)
    format: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    params_billions: Mapped[Optional[float]] = mapped_column(nullable=True)
    context_length: Mapped[Optional[int]] = mapped_column(nullable=True)
    license_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    min_ram_bytes: Mapped[Optional[int]] = mapped_column(nullable=True)
    min_vram_bytes: Mapped[Optional[int]] = mapped_column(nullable=True)
    quant_options_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    backend_support_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    supports_lora: Mapped[int] = mapped_column(default=0)
    supports_qlora: Mapped[int] = mapped_column(default=0)
    guardrail_role: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)

    installed_models: Mapped[list[InstalledModel]] = relationship(back_populates="catalog")


class InstalledModel(Base):
    __tablename__ = "installed_models"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    catalog_id: Mapped[Optional[str]] = mapped_column(ForeignKey("model_catalog.id"), nullable=True)
    source: Mapped[str] = mapped_column(Text)
    local_name: Mapped[str] = mapped_column(Text)
    storage_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    base_model_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("installed_models.id"), nullable=True
    )
    adapter_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    format: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    quantization: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    size_bytes: Mapped[Optional[int]] = mapped_column(nullable=True)
    is_temporary: Mapped[int] = mapped_column(default=1)
    status: Mapped[str] = mapped_column(Text, default="discovered")
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    catalog: Mapped[Optional[ModelCatalog]] = relationship(back_populates="installed_models")
    base_model: Mapped[Optional[InstalledModel]] = relationship(remote_side=[id])
    benchmark_runs: Mapped[list[BenchmarkRun]] = relationship(back_populates="model")
    optimization_sources: Mapped[list[OptimizationRun]] = relationship(
        foreign_keys="OptimizationRun.source_model_id",
        back_populates="source_model",
    )
    optimization_outputs: Mapped[list[OptimizationRun]] = relationship(
        foreign_keys="OptimizationRun.output_model_id",
        back_populates="output_model",
    )


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    name: Mapped[str] = mapped_column(Text)
    version: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    local_path: Mapped[str] = mapped_column(Text)
    schema_version: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    task_mode: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    benchmark_runs: Mapped[list[BenchmarkRun]] = relationship(back_populates="dataset")
    training_runs: Mapped[list[TrainingRun]] = relationship(back_populates="dataset")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    job_type: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, default="queued")
    priority: Mapped[int] = mapped_column(default=5)
    payload_json: Mapped[str] = mapped_column(Text)
    result_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    progress: Mapped[int] = mapped_column(default=0)
    started_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    benchmark_runs: Mapped[list[BenchmarkRun]] = relationship(back_populates="job")
    optimization_runs: Mapped[list[OptimizationRun]] = relationship(back_populates="job")
    training_runs: Mapped[list[TrainingRun]] = relationship(back_populates="job")


class BenchmarkSuite(Base):
    __tablename__ = "benchmark_suites"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    name: Mapped[str] = mapped_column(Text, unique=True)
    task_mode: Mapped[str] = mapped_column(Text)
    config_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    runs: Mapped[list[BenchmarkRun]] = relationship(back_populates="suite")


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    suite_id: Mapped[str] = mapped_column(ForeignKey("benchmark_suites.id"))
    job_id: Mapped[Optional[str]] = mapped_column(ForeignKey("jobs.id"), nullable=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("installed_models.id"))
    hardware_profile_id: Mapped[str] = mapped_column(ForeignKey("hardware_profiles.id"))
    dataset_id: Mapped[Optional[str]] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    backend: Mapped[str] = mapped_column(Text)
    scenario: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, default="pending")
    summary_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    raw_metrics_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    suite: Mapped[BenchmarkSuite] = relationship(back_populates="runs")
    job: Mapped[Optional[Job]] = relationship(back_populates="benchmark_runs")
    model: Mapped[InstalledModel] = relationship(back_populates="benchmark_runs")
    hardware_profile: Mapped[HardwareProfileRow] = relationship(back_populates="benchmark_runs")
    dataset: Mapped[Optional[Dataset]] = relationship(back_populates="benchmark_runs")
    samples: Mapped[list[BenchmarkSample]] = relationship(back_populates="benchmark_run")
    cyber_artifacts: Mapped[list[CyberArtifact]] = relationship(back_populates="benchmark_run")
    reports: Mapped[list[Report]] = relationship(back_populates="benchmark_run")


class BenchmarkSample(Base):
    __tablename__ = "benchmark_samples"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    benchmark_run_id: Mapped[str] = mapped_column(ForeignKey("benchmark_runs.id"))
    case_id: Mapped[str] = mapped_column(Text)
    success: Mapped[Optional[int]] = mapped_column(nullable=True)
    verifier_pass: Mapped[Optional[int]] = mapped_column(nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(nullable=True)
    input_tokens: Mapped[Optional[int]] = mapped_column(nullable=True)
    output_tokens: Mapped[Optional[int]] = mapped_column(nullable=True)
    score_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    artifact_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    benchmark_run: Mapped[BenchmarkRun] = relationship(back_populates="samples")


class OptimizationRun(Base):
    __tablename__ = "optimization_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    source_model_id: Mapped[str] = mapped_column(ForeignKey("installed_models.id"))
    output_model_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("installed_models.id"), nullable=True
    )
    method: Mapped[str] = mapped_column(Text)
    config_json: Mapped[str] = mapped_column(Text)
    diff_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    job: Mapped[Job] = relationship(back_populates="optimization_runs")
    source_model: Mapped[InstalledModel] = relationship(
        foreign_keys=[source_model_id],
        back_populates="optimization_sources",
    )
    output_model: Mapped[Optional[InstalledModel]] = relationship(
        foreign_keys=[output_model_id],
        back_populates="optimization_outputs",
    )


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    base_model_id: Mapped[str] = mapped_column(ForeignKey("installed_models.id"))
    output_model_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("installed_models.id"), nullable=True
    )
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    optimizer: Mapped[str] = mapped_column(Text)
    learning_rate: Mapped[float] = mapped_column()
    epochs: Mapped[int] = mapped_column()
    batch_size: Mapped[int] = mapped_column()
    gradient_accumulation: Mapped[int] = mapped_column()
    lora_r: Mapped[Optional[int]] = mapped_column(nullable=True)
    lora_alpha: Mapped[Optional[int]] = mapped_column(nullable=True)
    lora_dropout: Mapped[Optional[float]] = mapped_column(nullable=True)
    target_modules_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    train_metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    eval_metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    job: Mapped[Job] = relationship(back_populates="training_runs")
    base_model: Mapped[InstalledModel] = relationship(foreign_keys=[base_model_id])
    output_model: Mapped[Optional[InstalledModel]] = relationship(foreign_keys=[output_model_id])
    dataset: Mapped[Dataset] = relationship(back_populates="training_runs")


class CyberArtifact(Base):
    __tablename__ = "cyber_artifacts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    benchmark_run_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("benchmark_runs.id"), nullable=True
    )
    model_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("installed_models.id"), nullable=True
    )
    artifact_type: Mapped[str] = mapped_column(Text)
    content_path: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    benchmark_run: Mapped[Optional[BenchmarkRun]] = relationship(back_populates="cyber_artifacts")
    model: Mapped[Optional[InstalledModel]] = relationship()
    validations: Mapped[list[ValidationRun]] = relationship(back_populates="artifact")


class ValidationRun(Base):
    __tablename__ = "validation_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    artifact_id: Mapped[str] = mapped_column(ForeignKey("cyber_artifacts.id"))
    validator_type: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    output_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    artifact: Mapped[CyberArtifact] = relationship(back_populates="validations")


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    report_type: Mapped[str] = mapped_column(Text)
    benchmark_run_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("benchmark_runs.id"), nullable=True
    )
    output_path: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    benchmark_run: Mapped[Optional[BenchmarkRun]] = relationship(back_populates="reports")


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    entity_type: Mapped[str] = mapped_column(Text)
    entity_id: Mapped[str] = mapped_column(Text)
    action: Mapped[str] = mapped_column(Text)
    details_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)


# ── Engine & Session ─────────────────────────────────────────────

_DEFAULT_DB_PATH = Path("data/cyberforge.db")


def _enable_sqlite_fk(dbapi_conn, _connection_record):
    """Enable foreign key enforcement for every SQLite connection."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def build_engine(db_path: Path | str | None = None):
    """Create an async SQLAlchemy engine for the given SQLite path."""
    path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite+aiosqlite:///{path.as_posix()}"
    engine = create_async_engine(url, echo=False)
    event.listen(engine.sync_engine, "connect", _enable_sqlite_fk)
    return engine


def build_session_factory(engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine) -> None:
    """Create all tables (idempotent). Use Alembic for migrations in production."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session(session_factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    """Yield a scoped async session, auto-closing on exit."""
    async with session_factory() as session:
        yield session
