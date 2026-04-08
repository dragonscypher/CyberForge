"""Pruning module — structured and unstructured model pruning via PyTorch.

Supports magnitude-based, L1-structured, and random pruning strategies.
Produces a pruned model artifact in the cache directory.

Ticket: OPT-011
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

log = logging.getLogger(__name__)

_MODEL_ID_RE = re.compile(r'^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)?$')

def _safe_name(model_id: str) -> str:
    """Sanitize model id to a safe filesystem component."""
    base = model_id.replace('/', '_').replace('\\', '_')
    base = re.sub(r'[^A-Za-z0-9._-]', '_', base).strip('.')
    return base or 'model'

def _trust_remote() -> bool:
    return os.environ.get('CYBERFORGE_TRUST_REMOTE_CODE', '0') in ('1', 'true', 'yes')


def _resolve_hf_repo(source_model: str) -> str:
    """Resolve an Ollama tag (e.g. 'qwen3:8b') to an HF repo ID via registry.yaml.

    Returns the HF repo if found, otherwise the original string unchanged.
    """
    if ":" not in source_model:
        return source_model  # already looks like an HF repo

    try:
        import yaml
        registry_path = Path(__file__).with_name("registry.yaml")
        if registry_path.exists():
            data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
            for m in data.get("models", []):
                if m.get("ollama_tag") == source_model and m.get("hf_repo"):
                    log.info("Resolved '%s' → '%s' via registry", source_model, m["hf_repo"])
                    return m["hf_repo"]
    except Exception as exc:
        log.debug("Registry lookup failed: %s", exc)

    return source_model


class PruneMethod(str, Enum):
    MAGNITUDE = "magnitude"       # Unstructured: remove smallest-magnitude weights
    L1_STRUCTURED = "l1_structured"  # Structured: remove entire neurons by L1 norm
    RANDOM = "random"             # Unstructured: random weight removal


class PruneConfig(BaseModel):
    source_model: str             # HF repo id or local path
    method: PruneMethod = PruneMethod.MAGNITUDE
    sparsity: float = 0.3         # Fraction of weights to prune (0.0–0.9)
    target_modules: list[str] = ["linear"]  # Which layer types to prune
    output_name: Optional[str] = None
    output_dir: str = "data/cache/pruned"
    make_permanent: bool = True   # Remove pruning re-parametrization


class PruneResult(BaseModel):
    output_model: str = ""
    output_path: str = ""
    method: str = ""
    sparsity: float = 0.0
    original_params: int = 0
    pruned_params: int = 0
    zero_params: int = 0
    size_bytes: int = 0
    size_reduction_pct: float = 0.0
    success: bool = True
    error: Optional[str] = None
    duration_seconds: float = 0.0


class SparsityReport(BaseModel):
    """Per-layer sparsity analysis."""
    layer_name: str
    total_params: int
    zero_params: int
    sparsity: float


def analyze_sparsity(model: Any) -> list[SparsityReport]:
    """Analyze per-layer sparsity of a PyTorch model."""
    import torch
    reports = []
    for name, param in model.named_parameters():
        total = param.numel()
        zeros = int((param == 0).sum().item())
        reports.append(SparsityReport(
            layer_name=name,
            total_params=total,
            zero_params=zeros,
            sparsity=round(zeros / total, 4) if total > 0 else 0.0,
        ))
    return reports


def suggest_pruning(
    params_b: float,
    vram_mb: int,
    target_vram_mb: int,
) -> dict[str, Any]:
    """Suggest a pruning strategy to fit a model into target VRAM.

    Returns:
        Dict with recommended sparsity, method, and expected savings.
    """
    fp16_vram = int(params_b * 2000 * 1.15)
    if fp16_vram <= target_vram_mb:
        return {
            "pruning_needed": False,
            "message": f"Model fits in {target_vram_mb} MB without pruning.",
        }

    # Calculate required sparsity to fit
    # Pruning reduces effective size by ~40% of sparsity (conservative)
    required_reduction = (fp16_vram - target_vram_mb) / fp16_vram
    # Sparsity needed: reduction / effectiveness_factor
    needed_sparsity = min(required_reduction / 0.4, 0.9)

    method = "magnitude" if needed_sparsity <= 0.5 else "l1_structured"

    return {
        "pruning_needed": True,
        "recommended_sparsity": round(needed_sparsity, 2),
        "recommended_method": method,
        "original_vram_mb": fp16_vram,
        "estimated_vram_after_mb": int(fp16_vram * (1 - needed_sparsity * 0.4)),
        "target_vram_mb": target_vram_mb,
        "message": (
            f"Prune to {needed_sparsity:.0%} sparsity using {method} "
            f"to reduce from {fp16_vram} MB to ~{int(fp16_vram * (1 - needed_sparsity * 0.4))} MB."
        ),
    }


def _prune_sync(config: PruneConfig) -> PruneResult:
    """Synchronous pruning — loads model, applies pruning, saves."""
    start = time.time()

    try:
        import torch
        import torch.nn.utils.prune as prune_utils
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        return PruneResult(
            success=False,
            error=f"Pruning requires torch and transformers: {e}",
            method=config.method,
            duration_seconds=round(time.time() - start, 2),
        )

    # Resolve Ollama tags to HF repo IDs
    resolved_model = _resolve_hf_repo(config.source_model)
    if ":" in resolved_model:
        return PruneResult(
            success=False,
            error=(
                f"Pruning requires a HuggingFace model, but '{config.source_model}' "
                f"looks like an Ollama tag with no HF mapping in registry.yaml. "
                f"Please use a HuggingFace repo ID (e.g. 'Qwen/Qwen2.5-7B-Instruct')."
            ),
            method=config.method,
            duration_seconds=round(time.time() - start, 2),
        )

    out_dir = Path(config.output_dir) / (
        config.output_name or f"{_safe_name(config.source_model)}-pruned-{config.method}-{config.sparsity}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    trc = _trust_remote()

    try:
        log.info("Loading %s for pruning (%s, sparsity=%.2f)",
                 resolved_model, config.method, config.sparsity)

        tokenizer = AutoTokenizer.from_pretrained(
            resolved_model,
            token=hf_token,
            trust_remote_code=trc,
        )
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model,
            token=hf_token,
            trust_remote_code=trc,
            torch_dtype=torch.float16,
            device_map="cpu",  # Prune on CPU to save VRAM
        )

        original_params = sum(p.numel() for p in model.parameters())

        # Collect layers to prune
        layers_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                layers_to_prune.append((module, "weight"))

        if not layers_to_prune:
            return PruneResult(
                success=False,
                error="No Linear layers found to prune",
                method=config.method,
                duration_seconds=round(time.time() - start, 2),
            )

        # Apply pruning
        if config.method == PruneMethod.MAGNITUDE:
            prune_utils.global_unstructured(
                layers_to_prune,
                pruning_method=prune_utils.L1Unstructured,
                amount=config.sparsity,
            )
        elif config.method == PruneMethod.L1_STRUCTURED:
            for module, param_name in layers_to_prune:
                prune_utils.ln_structured(
                    module, name=param_name, amount=config.sparsity, n=1, dim=0,
                )
        elif config.method == PruneMethod.RANDOM:
            prune_utils.global_unstructured(
                layers_to_prune,
                pruning_method=prune_utils.RandomUnstructured,
                amount=config.sparsity,
            )

        # Make permanent (remove re-parametrization)
        if config.make_permanent:
            for module, param_name in layers_to_prune:
                try:
                    prune_utils.remove(module, param_name)
                except ValueError:
                    pass  # Already removed

        # Count zeros
        zero_params = sum(
            int((p == 0).sum().item()) for p in model.parameters()
        )

        # Save
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())

        return PruneResult(
            output_model=str(out_dir.name),
            output_path=str(out_dir),
            method=config.method,
            sparsity=config.sparsity,
            original_params=original_params,
            pruned_params=original_params,
            zero_params=zero_params,
            size_bytes=size,
            size_reduction_pct=round(zero_params / original_params * 100, 1) if original_params else 0,
            success=True,
            duration_seconds=round(time.time() - start, 2),
        )

    except Exception as e:
        log.exception("Pruning failed")
        return PruneResult(
            success=False,
            error=str(e),
            method=config.method,
            duration_seconds=round(time.time() - start, 2),
        )


async def prune_model(config: PruneConfig) -> PruneResult:
    """Prune a model — runs in thread pool to avoid blocking the event loop."""
    return await asyncio.to_thread(_prune_sync, config)


# ── Iterative pruning with verification (OPT-011b) ────────────────
# Inspired by openclaude's verify + loop skills: incrementally increase
# sparsity, verify quality at each step, stop when threshold exceeded.


class IterativePruneConfig(BaseModel):
    source_model: str
    method: PruneMethod = PruneMethod.MAGNITUDE
    start_sparsity: float = 0.1
    end_sparsity: float = 0.7
    step_size: float = 0.1
    max_perplexity_ratio: float = 1.5   # Stop when ppl exceeds 1.5x baseline
    eval_text: str = ""                  # Text for perplexity measurement
    output_dir: str = "data/cache/pruned"
    make_permanent: bool = True


class IterativePruneStep(BaseModel):
    sparsity: float
    perplexity: float
    perplexity_ratio: float
    zero_params: int
    total_params: int
    passed: bool


class IterativePruneResult(BaseModel):
    best_sparsity: float = 0.0
    best_perplexity_ratio: float = 0.0
    output_model: str = ""
    output_path: str = ""
    steps: list[IterativePruneStep] = []
    baseline_perplexity: float = 0.0
    success: bool = True
    error: Optional[str] = None
    duration_seconds: float = 0.0


def _compute_ppl(model, tokenizer, text: str, device: str = "cpu") -> float:
    """Compute perplexity on sample text."""
    import torch
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    input_ids = encodings["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return float(torch.exp(outputs.loss).item())


def _iterative_prune_sync(config: IterativePruneConfig) -> IterativePruneResult:
    """Iteratively prune with verification at each step."""
    start = time.time()

    try:
        import torch
        import torch.nn.utils.prune as prune_utils
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        return IterativePruneResult(
            success=False,
            error=f"Requires torch and transformers: {e}",
            duration_seconds=round(time.time() - start, 2),
        )

    hf_token = os.environ.get("HF_TOKEN")
    trc = _trust_remote()

    # Resolve Ollama tags to HF repo IDs
    resolved_model = _resolve_hf_repo(config.source_model)
    if ":" in resolved_model:
        return IterativePruneResult(
            success=False,
            error=(
                f"Iterative pruning requires a HuggingFace model, but '{config.source_model}' "
                f"looks like an Ollama tag with no HF mapping in registry.yaml. "
                f"Please use a HuggingFace repo ID (e.g. 'Qwen/Qwen2.5-7B-Instruct')."
            ),
            duration_seconds=round(time.time() - start, 2),
        )

    eval_text = config.eval_text or (
        "The transformer model processes input sequences through multiple layers "
        "of self-attention and feed-forward networks. Each layer refines the "
        "representation, allowing the model to capture complex patterns in data."
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_model, token=hf_token, trust_remote_code=trc,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            resolved_model, token=hf_token, trust_remote_code=trc,
            torch_dtype=torch.float16, device_map="cpu",
        )
        model.eval()

        # Baseline perplexity
        baseline_ppl = _compute_ppl(model, tokenizer, eval_text)
        log.info("Baseline perplexity: %.2f", baseline_ppl)

        steps: list[IterativePruneStep] = []
        best_sparsity = 0.0
        best_ratio = 1.0

        sparsity = config.start_sparsity
        while sparsity <= config.end_sparsity + 1e-6:
            # Collect prunable layers
            layers_to_prune = [
                (m, "weight") for _, m in model.named_modules()
                if isinstance(m, torch.nn.Linear)
            ]

            if not layers_to_prune:
                break

            # Apply pruning at current sparsity
            if config.method == PruneMethod.MAGNITUDE:
                prune_utils.global_unstructured(
                    layers_to_prune,
                    pruning_method=prune_utils.L1Unstructured,
                    amount=sparsity,
                )
            elif config.method == PruneMethod.L1_STRUCTURED:
                for module, param_name in layers_to_prune:
                    prune_utils.ln_structured(
                        module, name=param_name, amount=sparsity, n=1, dim=0,
                    )
            elif config.method == PruneMethod.RANDOM:
                prune_utils.global_unstructured(
                    layers_to_prune,
                    pruning_method=prune_utils.RandomUnstructured,
                    amount=sparsity,
                )

            # Measure quality
            ppl = _compute_ppl(model, tokenizer, eval_text)
            ratio = ppl / baseline_ppl if baseline_ppl > 0 else float("inf")

            total_params = sum(p.numel() for p in model.parameters())
            zero_params = sum(int((p == 0).sum().item()) for p in model.parameters())

            passed = ratio <= config.max_perplexity_ratio

            steps.append(IterativePruneStep(
                sparsity=round(sparsity, 2),
                perplexity=round(ppl, 2),
                perplexity_ratio=round(ratio, 3),
                zero_params=zero_params,
                total_params=total_params,
                passed=passed,
            ))

            log.info(
                "Sparsity %.0f%%: ppl=%.2f (%.2fx baseline) — %s",
                sparsity * 100, ppl, ratio, "PASS" if passed else "FAIL",
            )

            if passed:
                best_sparsity = round(sparsity, 2)
                best_ratio = round(ratio, 3)
            else:
                # Quality gate failed — stop iterating
                break

            # Remove pruning hooks before next iteration
            for module, param_name in layers_to_prune:
                try:
                    prune_utils.remove(module, param_name)
                except ValueError:
                    pass

            sparsity += config.step_size

        # Apply final best sparsity permanently
        output_model = ""
        output_path = ""

        if best_sparsity > 0:
            # Re-apply best sparsity
            layers_to_prune = [
                (m, "weight") for _, m in model.named_modules()
                if isinstance(m, torch.nn.Linear)
            ]
            prune_utils.global_unstructured(
                layers_to_prune,
                pruning_method=prune_utils.L1Unstructured,
                amount=best_sparsity,
            )
            if config.make_permanent:
                for module, param_name in layers_to_prune:
                    try:
                        prune_utils.remove(module, param_name)
                    except ValueError:
                        pass

            out_dir = Path(config.output_dir) / (
                f"{_safe_name(config.source_model)}-iterative-pruned-{best_sparsity}"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(out_dir))
            tokenizer.save_pretrained(str(out_dir))
            output_model = str(out_dir.name)
            output_path = str(out_dir)

        return IterativePruneResult(
            best_sparsity=best_sparsity,
            best_perplexity_ratio=best_ratio,
            output_model=output_model,
            output_path=output_path,
            steps=steps,
            baseline_perplexity=round(baseline_ppl, 2),
            success=True,
            duration_seconds=round(time.time() - start, 2),
        )

    except Exception as e:
        log.exception("Iterative pruning failed")
        return IterativePruneResult(
            success=False,
            error=str(e),
            duration_seconds=round(time.time() - start, 2),
        )


async def iterative_prune(config: IterativePruneConfig) -> IterativePruneResult:
    """Iterative pruning with verification — runs in thread pool."""
    return await asyncio.to_thread(_iterative_prune_sync, config)


def list_prune_methods() -> list[dict[str, Any]]:
    """Return available pruning methods with descriptions."""
    return [
        {
            "id": "magnitude",
            "name": "Magnitude Pruning (Unstructured)",
            "description": "Remove smallest-magnitude weights globally. Best general-purpose method.",
            "recommended_sparsity": "30-50%",
            "preserves_structure": False,
        },
        {
            "id": "l1_structured",
            "name": "L1 Structured Pruning",
            "description": "Remove entire neurons by L1 norm. Better hardware speedup, more quality loss.",
            "recommended_sparsity": "20-40%",
            "preserves_structure": True,
        },
        {
            "id": "random",
            "name": "Random Pruning (Unstructured)",
            "description": "Randomly remove weights. Baseline comparison method.",
            "recommended_sparsity": "20-30%",
            "preserves_structure": False,
        },
        {
            "id": "iterative",
            "name": "Iterative Pruning with Verification",
            "description": (
                "Inspired by verify-loop pattern: incrementally increases sparsity, "
                "checks perplexity at each step, stops when quality degrades beyond threshold. "
                "Finds optimal sparsity automatically."
            ),
            "recommended_sparsity": "auto (0.1–0.7 sweep)",
            "preserves_structure": False,
        },
    ]
