"""Benchmark harness — runs before/after comparisons and produces benchmark cards."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import psutil

from pydantic import BaseModel, Field

from packages.cyber.verifiers import SigmaVerifier
from packages.hardware.profiler import HardwareProfiler
from packages.serve.base import ChatMessage, ChatRequest
from packages.serve.ollama import OllamaClient
from packages.serve.openrouter import OpenRouterClient


class SystemMetrics(BaseModel):
    latency_ms: float = 0.0
    throughput_tok_s: float = 0.0
    vram_peak_mb: float = 0.0
    ram_peak_mb: float = 0.0
    load_time_ms: float = 0.0
    model_size_mb: float = 0.0


class TaskMetrics(BaseModel):
    exact_match: Optional[float] = None
    pass_at_k: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    false_positive_rate: Optional[float] = None
    detection_rate: Optional[float] = None
    calibration_score: Optional[float] = None
    verifier_pass_rate: Optional[float] = None


class ReliabilityMetrics(BaseModel):
    structured_output_validity: Optional[float] = None
    hallucination_flag_rate: Optional[float] = None
    refusal_correctness: Optional[float] = None
    syntax_error_rate: Optional[float] = None
    retry_success_rate: Optional[float] = None


class BenchmarkCard(BaseModel):
    model_id: str
    label: str = ""  # e.g. "baseline" or "q4_K_M optimized"
    task_mode: str = "general"
    timestamp: float = Field(default_factory=time.time)
    system: SystemMetrics = Field(default_factory=SystemMetrics)
    task: TaskMetrics = Field(default_factory=TaskMetrics)
    reliability: ReliabilityMetrics = Field(default_factory=ReliabilityMetrics)


class BenchmarkHarness:
    """Orchestrates benchmark runs and stores benchmark cards as JSON reports."""

    def __init__(
        self,
        reports_dir: str = "reports",
        ollama_client: OllamaClient | None = None,
        openrouter_client: OpenRouterClient | None = None,
        profiler: HardwareProfiler | None = None,
    ):
        self._reports_dir = Path(reports_dir)
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        self._ollama = ollama_client
        self._openrouter = openrouter_client
        self._profiler = profiler

    async def run_baseline(
        self,
        model_id: str,
        task_mode: str,
        backend: str = "ollama",
    ) -> BenchmarkCard:
        card = await self._run(model_id=model_id, task_mode=task_mode, label="baseline", backend=backend)
        self.save_card(card)
        return card

    async def run_optimized(
        self,
        model_id: str,
        task_mode: str,
        label: str,
        backend: str = "ollama",
    ) -> BenchmarkCard:
        card = await self._run(model_id=model_id, task_mode=task_mode, label=label, backend=backend)
        self.save_card(card)
        return card

    async def compare(self, baseline: BenchmarkCard, optimized: BenchmarkCard) -> dict[str, Any]:
        return {
            "baseline": {
                "model_id": baseline.model_id,
                "label": baseline.label,
            },
            "optimized": {
                "model_id": optimized.model_id,
                "label": optimized.label,
            },
            "deltas": {
                "latency_ms": _metric_delta(baseline.system.latency_ms, optimized.system.latency_ms, lower_is_better=True),
                "throughput_tok_s": _metric_delta(
                    baseline.system.throughput_tok_s,
                    optimized.system.throughput_tok_s,
                    lower_is_better=False,
                ),
                "vram_peak_mb": _metric_delta(
                    baseline.system.vram_peak_mb,
                    optimized.system.vram_peak_mb,
                    lower_is_better=True,
                ),
                "ram_peak_mb": _metric_delta(
                    baseline.system.ram_peak_mb,
                    optimized.system.ram_peak_mb,
                    lower_is_better=True,
                ),
                "load_time_ms": _metric_delta(
                    baseline.system.load_time_ms,
                    optimized.system.load_time_ms,
                    lower_is_better=True,
                ),
                "exact_match": _metric_delta(
                    baseline.task.exact_match,
                    optimized.task.exact_match,
                    lower_is_better=False,
                ),
                "verifier_pass_rate": _metric_delta(
                    baseline.task.verifier_pass_rate,
                    optimized.task.verifier_pass_rate,
                    lower_is_better=False,
                ),
                "structured_output_validity": _metric_delta(
                    baseline.reliability.structured_output_validity,
                    optimized.reliability.structured_output_validity,
                    lower_is_better=False,
                ),
                "syntax_error_rate": _metric_delta(
                    baseline.reliability.syntax_error_rate,
                    optimized.reliability.syntax_error_rate,
                    lower_is_better=True,
                ),
            },
        }

    def save_card(self, card: BenchmarkCard) -> str:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(card.timestamp))
        model_key = card.model_id.replace("/", "--").replace(":", "-")
        label_key = card.label.replace(" ", "_")
        file_name = f"bench-{stamp}-{model_key}-{label_key}.json"
        path = self._reports_dir / file_name
        path.write_text(card.model_dump_json(indent=2), encoding="utf-8")
        return str(path)

    def list_cards(self) -> list[str]:
        return sorted([p.name for p in self._reports_dir.glob("bench-*.json")], reverse=True)

    def load_card(self, file_name: str) -> BenchmarkCard | None:
        path = self._reports_dir / file_name
        if not path.exists() or not path.is_file():
            return None
        return BenchmarkCard.model_validate_json(path.read_text(encoding="utf-8"))

    async def compare_cards(self, baseline_file: str, optimized_file: str) -> dict[str, Any] | None:
        baseline = self.load_card(baseline_file)
        optimized = self.load_card(optimized_file)
        if baseline is None or optimized is None:
            return None
        return await self.compare(baseline=baseline, optimized=optimized)

    async def _run(self, model_id: str, task_mode: str, label: str, backend: str) -> BenchmarkCard:
        prompts = _prompt_suite(task_mode)
        if not prompts:
            raise ValueError(f"No benchmark prompt suite for task mode '{task_mode}'")

        proc = psutil.Process()
        ram_before = proc.memory_info().rss / (1024 * 1024)
        vram_before = _gpu_used_mb(self._profiler)

        latencies: list[float] = []
        token_estimates: list[int] = []
        case_successes = 0
        structured_valid_count = 0
        syntax_error_count = 0
        verifier_pass_count = 0
        retry_success_count = 0
        hallucination_flags = 0

        sigma_verifier = SigmaVerifier()

        for index, case in enumerate(prompts):
            t0 = time.perf_counter()
            text = await self._infer(backend=backend, model=model_id, prompt=case["prompt"])
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

            if not text.strip():
                retry = await self._infer(backend=backend, model=model_id, prompt=case["prompt"])
                if retry.strip():
                    retry_success_count += 1
                    text = retry

            token_estimates.append(max(int(len(text.split()) * 1.3), 1))

            if _case_passed(case, text):
                case_successes += 1

            if case.get("expects_json") and _valid_json_object(text):
                structured_valid_count += 1

            if task_mode in ("coding", "cyber") and _looks_like_syntax_error(text):
                syntax_error_count += 1

            if task_mode == "cyber" and case.get("artifact") == "sigma":
                verify = sigma_verifier.verify(text)
                if verify.valid:
                    verifier_pass_count += 1

            if "as an ai language model" in text.lower():
                hallucination_flags += 1

            if index == 0:
                load_time_ms = elapsed_ms

        ram_after = proc.memory_info().rss / (1024 * 1024)
        vram_after = _gpu_used_mb(self._profiler)

        total_elapsed = sum(latencies)
        mean_latency = total_elapsed / len(latencies)
        throughput = sum(token_estimates) / max(total_elapsed / 1000, 1e-9)
        exact_match = case_successes / len(prompts)

        card = BenchmarkCard(
            model_id=model_id,
            label=label,
            task_mode=task_mode,
        )
        card.system.latency_ms = round(mean_latency, 2)
        card.system.throughput_tok_s = round(throughput, 2)
        card.system.ram_peak_mb = round(max(ram_before, ram_after), 2)
        card.system.vram_peak_mb = round(max(vram_before, vram_after), 2)
        card.system.load_time_ms = round(load_time_ms, 2)
        card.system.model_size_mb = await self._model_size_mb(backend=backend, model_id=model_id)

        card.task.exact_match = round(exact_match, 4)
        if task_mode == "cyber":
            card.task.verifier_pass_rate = round(verifier_pass_count / len(prompts), 4)

        card.reliability.structured_output_validity = round(structured_valid_count / len(prompts), 4)
        card.reliability.hallucination_flag_rate = round(hallucination_flags / len(prompts), 4)
        card.reliability.retry_success_rate = round(retry_success_count / len(prompts), 4)
        if task_mode in ("coding", "cyber"):
            card.reliability.syntax_error_rate = round(syntax_error_count / len(prompts), 4)

        return card

    async def _infer(self, backend: str, model: str, prompt: str) -> str:
        if backend == "ollama":
            if self._ollama is None:
                raise RuntimeError("Ollama backend is not configured")
            return await self._ollama.generate(model=model, prompt=prompt)
        if backend == "openrouter":
            if self._openrouter is None:
                raise RuntimeError("OpenRouter backend is not configured")
            req = ChatRequest(
                model=model,
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.2,
                max_tokens=512,
                stream=False,
            )
            response = await self._openrouter.chat(req)
            return response.message.content
        raise ValueError(f"Unsupported backend '{backend}'")

    async def _model_size_mb(self, backend: str, model_id: str) -> float:
        if backend != "ollama" or self._ollama is None:
            return 0.0
        infos = await self._ollama.list_model_info()
        for info in infos:
            if info.name == model_id or info.name.startswith(model_id + ":"):
                return round(info.size / (1024 * 1024), 2)
        return 0.0


def _prompt_suite(task_mode: str) -> list[dict[str, Any]]:
    suites: dict[str, list[dict[str, Any]]] = {
        "general": [
            {
                "prompt": "Reply with valid JSON only: {\"task\":\"summarize\",\"answer\":\"what is least privilege\"}.",
                "keywords": ["least", "privilege"],
                "expects_json": True,
            },
            {
                "prompt": "In one sentence, explain why low latency matters for local copilots.",
                "keywords": ["latency"],
            },
        ],
        "coding": [
            {
                "prompt": "Write Python code only: a function named add_numbers(nums) that returns the sum of a list.",
                "keywords": ["def", "return", "add_numbers"],
            },
            {
                "prompt": "Reply with valid JSON only: {\"language\":\"python\",\"topic\":\"type hints\"}.",
                "keywords": ["python"],
                "expects_json": True,
            },
        ],
        "cyber": [
            {
                "prompt": "Write a Sigma rule in YAML that detects suspicious PowerShell encoded command usage.",
                "keywords": ["title", "logsource", "detection"],
                "artifact": "sigma",
            },
            {
                "prompt": "Reply with valid JSON only: {\"ioc_type\":\"domain\",\"ioc\":\"example.bad\",\"severity\":\"high\"}.",
                "keywords": ["ioc"],
                "expects_json": True,
            },
        ],
    }
    return suites.get(task_mode, suites["general"])


def _case_passed(case: dict[str, Any], output: str) -> bool:
    out = output.lower()
    return all(k.lower() in out for k in case.get("keywords", []))


def _valid_json_object(output: str) -> bool:
    import json

    try:
        parsed = json.loads(output)
    except Exception:
        return False
    return isinstance(parsed, dict)


def _looks_like_syntax_error(text: str) -> bool:
    lowered = text.lower()
    markers = ["syntaxerror", "traceback", "unexpected token", "parse error"]
    return any(m in lowered for m in markers)


def _gpu_used_mb(profiler: HardwareProfiler | None) -> float:
    if profiler is None:
        return 0.0
    profile = profiler.profile()
    used = [max(g.vram_total_mb - g.vram_free_mb, 0) for g in profile.gpus]
    return float(sum(used))


def _metric_delta(
    baseline_value: float | None,
    optimized_value: float | None,
    lower_is_better: bool,
) -> dict[str, float | bool | None]:
    if baseline_value is None or optimized_value is None:
        return {
            "baseline": baseline_value,
            "optimized": optimized_value,
            "absolute": None,
            "percent": None,
            "improved": None,
        }

    absolute = optimized_value - baseline_value
    percent = (absolute / baseline_value * 100.0) if baseline_value != 0 else 0.0
    improved = absolute < 0 if lower_is_better else absolute > 0
    return {
        "baseline": round(float(baseline_value), 4),
        "optimized": round(float(optimized_value), 4),
        "absolute": round(float(absolute), 4),
        "percent": round(float(percent), 2),
        "improved": improved,
    }
