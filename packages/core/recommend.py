"""Recommendation engine — suggests models given hardware, task, and preferences."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from packages.core.scoring import ScoreBreakdown, score_model
from packages.hardware.capability import (CapabilityMatrix, MachineClass,
                                          compute_capabilities)
from packages.hardware.profiler import HardwareProfile
from packages.models.registry import ModelEntry, ModelRegistry


class Recommendation(BaseModel):
    model: ModelEntry
    score: ScoreBreakdown
    reason: str = ""
    requires_quantization: bool = False
    suggested_quant: str = ""
    estimated_quant_vram_mb: int = 0
    benchmark_safe: bool = True
    fit_note: str = ""
    tier: str = "runs_now"  # runs_now | runs_after_optimization | not_recommended


class Recommender:
    """Given a hardware profile and task mode, rank registry models by fit."""

    def __init__(self, registry: ModelRegistry):
        self._registry = registry

    def recommend(
        self,
        profile: HardwareProfile,
        task_mode: str = "general",
        top_k: int = 5,
        include_guardrail: bool = True,
        include_post_quant: bool = True,
        include_not_recommended: bool = True,
    ) -> list[Recommendation]:
        caps = compute_capabilities(profile)
        mc = MachineClass.classify(profile)
        vram = mc.vram_total_mb
        ram = mc.ram_total_mb

        candidates = self._registry.compatible_with(vram, ram, task_mode=task_mode)

        # Also include guardrail models if requested
        if include_guardrail:
            guardrails = self._registry.search(task_mode="guardrail", max_vram=max(vram, 0))
            guardrails = [g for g in guardrails if g.min_vram <= max(vram, 0) or g.min_ram <= ram]
            candidates = _dedupe(candidates + guardrails)

        recs: list[Recommendation] = []
        for model in candidates:
            q = self._quality_heuristic(model, task_mode)
            r = self._reliability_heuristic(model)
            e = self._efficiency_heuristic(model, vram, ram, caps)
            c = self._compatibility_heuristic(model, profile, caps)
            s = score_model(q, r, e, c, task_mode)
            reason = self._explain(model, task_mode, caps)
            # Benchmark-safe = model's min_vram fits in actual GPU VRAM
            safe = vram > 0 and model.min_vram <= vram
            fit_note = ""
            if not safe and vram > 0:
                fit_note = f"Needs {model.min_vram} MB VRAM but GPU has {vram} MB — will use CPU offload (slow)"
                # Penalize score so smaller safe models rank higher
                s = score_model(q, r, e * 0.5, c * 0.7, task_mode)
            # Tier: if it fits in native (VRAM or RAM-only for cpu_only tier)
            tier = "runs_now"
            if not safe and mc.tier == "cpu_only":
                tier = "runs_now"  # CPU-only is expected
                safe = model.min_ram <= ram  # Safe if fits in RAM
            recs.append(Recommendation(model=model, score=s, reason=reason,
                                       benchmark_safe=safe, fit_note=fit_note, tier=tier))

        # Post-quant tier: models too large natively but fit after Q4
        native_ids = {r.model.id for r in recs}
        if include_post_quant:
            pq = self._post_quant_candidates(profile, task_mode, vram, ram, caps, mc, native_ids)
            recs.extend(pq)

        # Not-recommended tier: models too large even after quantization
        all_ids = {r.model.id for r in recs}
        if include_not_recommended:
            nr = self._not_recommended(profile, task_mode, vram, ram, caps, mc, all_ids)
            recs.extend(nr)

        recs.sort(key=lambda r: r.score.final, reverse=True)
        return recs[:top_k]

    def _post_quant_candidates(
        self,
        profile: HardwareProfile,
        task_mode: str,
        vram: int,
        ram: int,
        caps: CapabilityMatrix,
        mc: MachineClass,
        exclude_ids: set[str] | None = None,
    ) -> list[Recommendation]:
        """Find models that are too large at FP16 but fit after Q4 quantization."""
        max_q4_params = mc.max_postquant_params_b
        _exclude = exclude_ids or set()

        all_models = self._registry.models

        pq_recs: list[Recommendation] = []
        for model in all_models:
            if model.id in _exclude:
                continue  # Already in native tier
            if task_mode not in model.task_modes and model.cyber_role != "guardrail":
                continue
            if model.params > max_q4_params:
                continue  # Too large even at Q4 → handled by _not_recommended

            est_q4_vram = int(model.params * 560 * 1.2)
            suggested = "q4_K_M"
            if vram >= int(model.params * 1100):
                suggested = "q8_0"  # Can afford Q8

            q = self._quality_heuristic(model, task_mode) * 1.05
            r = self._reliability_heuristic(model)
            e = max(1.0 - (est_q4_vram / max(vram, ram, 1)), 0.0)
            c = self._compatibility_heuristic(model, profile, caps) * 0.9
            s = score_model(min(q, 1.0), r, e, c, task_mode)
            safe = vram > 0 and est_q4_vram <= vram
            fit_note = ""
            if not safe and vram > 0:
                fit_note = f"After {suggested}: ~{est_q4_vram} MB but GPU has {vram} MB — CPU offload likely"
                e = max(e * 0.4, 0.0)
                s = score_model(min(q, 1.0), r, e, c, task_mode)
            reason = (
                f"[POST-QUANT] {model.params}B → {suggested} "
                f"(~{est_q4_vram} MB VRAM); "
                + self._explain(model, task_mode, caps)
            )
            pq_recs.append(Recommendation(
                model=model,
                score=s,
                reason=reason,
                requires_quantization=True,
                suggested_quant=suggested,
                estimated_quant_vram_mb=est_q4_vram,
                benchmark_safe=safe,
                fit_note=fit_note,
                tier="runs_after_optimization",
            ))

        return pq_recs

    def _not_recommended(
        self,
        profile: HardwareProfile,
        task_mode: str,
        vram: int,
        ram: int,
        caps: CapabilityMatrix,
        mc: MachineClass,
        exclude_ids: set[str] | None = None,
    ) -> list[Recommendation]:
        """Models too large even after quantization on this machine class."""
        _exclude = exclude_ids or set()

        nr_recs: list[Recommendation] = []
        for model in self._registry.models:
            if model.id in _exclude:
                continue
            if task_mode not in model.task_modes and model.cyber_role != "guardrail":
                continue
            # Any model reaching here is too large for both native and post-quant tiers
            est_q4_vram = int(model.params * 560 * 1.2)
            reason = (
                f"[NOT RECOMMENDED] {model.params}B needs ~{est_q4_vram} MB even at Q4; "
                f"machine class '{mc.tier}' max is {mc.max_postquant_params_b:.1f}B post-quant"
            )
            s = score_model(0.1, 0.1, 0.0, 0.1, task_mode)
            nr_recs.append(Recommendation(
                model=model,
                score=s,
                reason=reason,
                requires_quantization=True,
                suggested_quant="q4_K_M",
                estimated_quant_vram_mb=est_q4_vram,
                benchmark_safe=False,
                fit_note=f"Too large for {mc.tier} — needs a bigger GPU",
                tier="not_recommended",
            ))

        return nr_recs

    # ── heuristic scorers (0.0–1.0) ─────────────────────────────
    @staticmethod
    def _quality_heuristic(model: ModelEntry, task_mode: str) -> float:
        """Rough quality proxy: bigger params + matching task mode = higher."""
        base = min(model.params / 14.0, 1.0)  # normalize to 14B ceiling
        mode_bonus = 0.15 if task_mode in model.task_modes else 0.0
        return min(base + mode_bonus, 1.0)

    @staticmethod
    def _reliability_heuristic(model: ModelEntry) -> float:
        score = 0.5
        if model.cyber_role == "guardrail":
            score += 0.3
        if model.supports_lora or model.supports_qlora:
            score += 0.1
        if model.context_length >= 32768:
            score += 0.1
        return min(score, 1.0)

    @staticmethod
    def _efficiency_heuristic(
        model: ModelEntry, vram: int, ram: int, caps: CapabilityMatrix
    ) -> float:
        """Lower resource ratio → higher efficiency score."""
        if vram > 0 and model.min_vram > 0:
            ratio = model.min_vram / vram
        elif ram > 0 and model.min_ram > 0:
            ratio = model.min_ram / ram
        else:
            ratio = 1.0
        return max(1.0 - ratio, 0.0)

    @staticmethod
    def _compatibility_heuristic(
        model: ModelEntry, profile: HardwareProfile, caps: CapabilityMatrix
    ) -> float:
        score = 0.0
        has_gpu = len(profile.gpus) > 0

        # Backend availability
        if model.supports_ollama_import and profile.backends.ollama:
            score += 0.3
        if model.supports_vllm and profile.backends.vllm:
            score += 0.2
        if model.supports_tensorrt_llm and profile.backends.tensorrt_llm:
            score += 0.2
        if has_gpu and profile.backends.pytorch_cuda:
            score += 0.1
        if not has_gpu:
            # CPU-only: prefer small GGUF-friendly models
            if model.params <= 3:
                score += 0.2

        return min(score, 1.0)

    @staticmethod
    def _explain(model: ModelEntry, task_mode: str, caps: CapabilityMatrix) -> str:
        parts: list[str] = []
        if task_mode in model.task_modes:
            parts.append(f"matches {task_mode} task mode")
        parts.append(f"{model.params}B params")
        if model.cyber_role:
            parts.append(f"cyber role: {model.cyber_role}")
        parts.append(f"recommended quant: {caps.recommended_quant}")
        return "; ".join(parts)


def _dedupe(models: list[ModelEntry]) -> list[ModelEntry]:
    seen: set[str] = set()
    out: list[ModelEntry] = []
    for m in models:
        if m.id not in seen:
            seen.add(m.id)
            out.append(m)
    return out
