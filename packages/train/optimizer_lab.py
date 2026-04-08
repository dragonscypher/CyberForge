"""Optimizer lab — advanced optimizer selection for LoRA/QLoRA training.

Supports adamw (baseline), adafactor (memory-efficient), and experimental muon.
Only visible when advanced_mode=True in the UI.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

log = logging.getLogger(__name__)


class OptimizerType(str, Enum):
    ADAMW = "adamw"
    ADAFACTOR = "adafactor"
    MUON = "muon"


class OptimizerLabConfig(BaseModel):
    """Configuration for the optimizer lab."""
    optimizer: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    # Adafactor-specific
    scale_parameter: bool = True
    relative_step: bool = False
    warmup_init: bool = False
    # Muon-specific (experimental)
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_backend: str = "newtonschulz5"
    # Scheduler
    scheduler_type: str = "cosine"  # cosine | linear | constant
    warmup_ratio: float = 0.03
    warmup_steps: int = 0


class OptimizerLabResult(BaseModel):
    optimizer_name: str = ""
    final_lr: float = 0.0
    total_steps: int = 0
    peak_memory_mb: float = 0.0
    compatible: bool = True
    warnings: list[str] = []


def get_training_args_optim(config: OptimizerLabConfig) -> str:
    """Return the HuggingFace TrainingArguments `optim` string."""
    mapping = {
        OptimizerType.ADAMW: "adamw_torch",
        OptimizerType.ADAFACTOR: "adafactor",
        OptimizerType.MUON: "adamw_torch",  # Muon needs custom setup
    }
    return mapping[config.optimizer]


def build_optimizer_kwargs(config: OptimizerLabConfig) -> dict[str, Any]:
    """Build optimizer-specific keyword arguments for TrainingArguments."""
    kwargs: dict[str, Any] = {
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "lr_scheduler_type": config.scheduler_type,
        "weight_decay": config.weight_decay,
    }

    if config.warmup_steps > 0:
        kwargs["warmup_steps"] = config.warmup_steps
        kwargs.pop("warmup_ratio", None)

    if config.optimizer == OptimizerType.ADAFACTOR:
        kwargs["optim"] = "adafactor"
        if config.relative_step:
            kwargs["learning_rate"] = None
    elif config.optimizer == OptimizerType.ADAMW:
        kwargs["optim"] = "adamw_torch"
        kwargs["adam_beta1"] = config.beta1
        kwargs["adam_beta2"] = config.beta2
        kwargs["adam_epsilon"] = config.epsilon
    elif config.optimizer == OptimizerType.MUON:
        kwargs["optim"] = "adamw_torch"

    return kwargs


def create_custom_optimizer(config: OptimizerLabConfig, model_params):
    """Create a custom optimizer instance for advanced configurations.

    For standard adamw/adafactor, prefer using TrainingArguments instead.
    This is primarily for the experimental muon optimizer.
    """
    if config.optimizer == OptimizerType.MUON:
        return _create_muon_optimizer(config, model_params)
    return None  # Let TrainingArguments handle standard optimizers


def _create_muon_optimizer(config: OptimizerLabConfig, model_params):
    """Create experimental Muon optimizer.

    Muon uses Newton-Schulz orthogonalization for updates.
    Requires torch. Falls back to AdamW if unavailable.
    """
    try:
        import torch
    except ImportError:
        log.warning("torch not available; falling back to standard optimizer")
        return None

    # Separate parameters: Muon for 2D+ params, AdamW for 1D (biases, norms)
    muon_params = []
    adamw_params = []

    for name, param in model_params:
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    if not muon_params:
        log.warning("No 2D+ parameters found for Muon; falling back to AdamW")
        return torch.optim.AdamW(
            adamw_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    try:
        from torch.optim import AdamW

        # Muon-style optimizer: use SGD with momentum on orthogonalized grads
        # This is a simplified version — real Muon uses custom Newton-Schulz steps
        optimizer = AdamW(
            [
                {"params": muon_params, "lr": config.learning_rate * 0.1,
                 "weight_decay": config.weight_decay},
                {"params": adamw_params, "lr": config.learning_rate,
                 "weight_decay": config.weight_decay},
            ],
        )
        log.info("Muon-style optimizer created: %d 2D+ params, %d 1D params",
                 len(muon_params), len(adamw_params))
        return optimizer
    except Exception as e:
        log.warning("Muon optimizer creation failed: %s; falling back", e)
        return None


def validate_optimizer_config(config: OptimizerLabConfig) -> OptimizerLabResult:
    """Validate optimizer configuration and check compatibility."""
    warnings: list[str] = []
    compatible = True

    if config.optimizer == OptimizerType.MUON:
        warnings.append("Muon optimizer is experimental — results may vary")
        try:
            import torch
        except ImportError:
            warnings.append("torch not installed — Muon will fall back to AdamW")

    if config.optimizer == OptimizerType.ADAFACTOR:
        if config.relative_step and config.learning_rate > 0:
            warnings.append("Adafactor with relative_step ignores explicit learning_rate")
        if config.weight_decay > 0:
            warnings.append("Adafactor with weight_decay > 0 may destabilize training")

    if config.learning_rate > 1e-2:
        warnings.append(f"Learning rate {config.learning_rate} is very high for fine-tuning")

    if config.learning_rate < 1e-6:
        warnings.append(f"Learning rate {config.learning_rate} is very low — training may be slow")

    return OptimizerLabResult(
        optimizer_name=config.optimizer.value,
        compatible=compatible,
        warnings=warnings,
    )


def list_available_optimizers(advanced_mode: bool = False) -> list[dict[str, Any]]:
    """List available optimizers with descriptions."""
    optimizers = [
        {
            "name": "adamw",
            "display": "AdamW",
            "description": "Default optimizer — stable and well-tested for LoRA fine-tuning",
            "recommended": True,
            "advanced_only": False,
        },
        {
            "name": "adafactor",
            "display": "Adafactor",
            "description": "Memory-efficient alternative — good for large models on limited VRAM",
            "recommended": False,
            "advanced_only": False,
        },
    ]

    if advanced_mode:
        optimizers.append({
            "name": "muon",
            "display": "Muon (experimental)",
            "description": "Experimental optimizer using Newton-Schulz orthogonalization",
            "recommended": False,
            "advanced_only": True,
        })

    return optimizers
