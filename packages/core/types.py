"""Shared type definitions for the CyberForge system."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class TaskMode(str, Enum):
    GENERAL = "general"
    CODING = "coding"
    CYBER = "cyber"
    GUARDRAIL = "guardrail"


class JobStatus(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    BENCHMARK_BEFORE = "benchmark_before"
    RUNNING = "running"
    OPTIMIZING = "optimizing"
    BENCHMARK_AFTER = "benchmark_after"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CLEANUP_PENDING = "cleanup_pending"


class QuantMethod(str, Enum):
    OLLAMA_GGUF = "ollama_gguf"
    BNB_8BIT = "bnb_8bit"
    BNB_4BIT = "bnb_4bit"
    AWQ = "awq"
    GPTQ = "gptq"


class BackendType(str, Enum):
    OLLAMA = "ollama"
    VLLM = "vllm"
    TENSORRT_LLM = "tensorrt_llm"
    TRANSFORMERS = "transformers"
    OPENROUTER = "openrouter"


class NormalizedBenchmarkSummary(BaseModel):
    """plan.md §8 normalized summary schema."""
    quality: dict[str, float] = {}
    reliability: dict[str, float] = {}
    efficiency: dict[str, float] = {}
