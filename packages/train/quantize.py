"""Training-side quantization utilities.

Thin wrapper around packages.models.quantization with training-aware defaults
and a helper to quantize a model before LoRA/QLoRA fine-tuning.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from packages.models.quantization import (QuantConfig, QuantMethod,
                                          QuantResult, quantize, quantize_bnb,
                                          quantize_ollama)

log = logging.getLogger(__name__)

__all__ = [
    "QuantConfig",
    "QuantMethod",
    "QuantResult",
    "quantize",
    "quantize_bnb",
    "quantize_ollama",
    "prepare_quantized_for_training",
    "list_quant_methods",
]


class TrainingQuantConfig(BaseModel):
    """Quantization configuration aware of downstream training requirements."""
    source_model: str
    method: str = "bnb_4bit"  # bnb_4bit | bnb_8bit | ollama_gguf
    compute_dtype: str = "bfloat16"
    quant_type: str = "nf4"
    double_quant: bool = True
    output_dir: str = "data/cache"


def prepare_quantized_for_training(config: TrainingQuantConfig) -> dict[str, Any]:
    """Return BitsAndBytes config dict suitable for LoRA/QLoRA training.

    This does NOT load the model — it prepares the quantization kwargs
    that should be passed to ``AutoModelForCausalLM.from_pretrained()``.
    """
    if config.method == "bnb_4bit":
        return {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": config.compute_dtype,
            "bnb_4bit_quant_type": config.quant_type,
            "bnb_4bit_use_double_quant": config.double_quant,
        }
    elif config.method == "bnb_8bit":
        return {
            "load_in_8bit": True,
        }
    else:
        log.warning("Method '%s' not directly usable for training; returning empty config", config.method)
        return {}


def list_quant_methods() -> list[dict[str, Any]]:
    """List available quantization methods with metadata."""
    return [
        {
            "id": "bnb_4bit",
            "name": "bitsandbytes 4-bit (NF4)",
            "description": "4-bit NormalFloat quantization — best for QLoRA training",
            "training_compatible": True,
            "requires_gpu": True,
        },
        {
            "id": "bnb_8bit",
            "name": "bitsandbytes 8-bit (LLM.int8)",
            "description": "8-bit quantization with outlier handling",
            "training_compatible": True,
            "requires_gpu": True,
        },
        {
            "id": "ollama_gguf",
            "name": "Ollama GGUF",
            "description": "GGML quantization via Ollama — inference only",
            "training_compatible": False,
            "requires_gpu": False,
        },
    ]
