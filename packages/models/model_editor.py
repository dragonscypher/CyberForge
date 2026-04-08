"""Model editor — surgical model modifications: layer surgery, weight merging, vocab editing.

Inspired by openclaude's batch pattern (decompose → parallel workers → aggregate)
applied as decomposed model editing operations that can be combined.

Supports:
  - Layer removal (drop layers to shrink model)
  - Weight merging (SLERP/linear interpolation between two models)
  - Vocabulary resizing (extend or shrink tokenizer + embeddings)
  - Head pruning (remove attention heads by importance)

Ticket: OPT-014
"""

from __future__ import annotations

import asyncio
import logging
import math
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


class EditOperation(str, Enum):
    LAYER_REMOVE = "layer_remove"      # Drop specified transformer layers
    WEIGHT_MERGE = "weight_merge"      # Interpolate weights of two models
    VOCAB_RESIZE = "vocab_resize"      # Extend or shrink vocabulary
    HEAD_PRUNE = "head_prune"          # Remove attention heads


class MergeMethod(str, Enum):
    LINEAR = "linear"    # Simple weighted average: alpha*A + (1-alpha)*B
    SLERP = "slerp"      # Spherical linear interpolation (better for diverse models)


class EditConfig(BaseModel):
    source_model: str                              # Primary model (HF id or local path)
    operation: EditOperation = EditOperation.LAYER_REMOVE
    # Layer removal
    layers_to_remove: list[int] = []               # Layer indices to drop
    # Weight merging
    merge_model: str = ""                          # Second model for merging
    merge_method: MergeMethod = MergeMethod.LINEAR
    merge_alpha: float = 0.5                       # Blend: alpha*source + (1-alpha)*merge
    # Vocab resize
    new_vocab_size: int = 0                        # Target vocab size (0 = no change)
    # Head pruning
    heads_to_prune: dict[int, list[int]] = {}      # {layer_idx: [head_indices]}
    num_heads_to_prune: int = 0                    # Auto-prune N least-important heads
    # Output
    output_dir: str = "data/cache/edited"
    output_name: Optional[str] = None


class EditResult(BaseModel):
    output_model: str = ""
    output_path: str = ""
    operation: str = ""
    original_params: int = 0
    edited_params: int = 0
    param_reduction_pct: float = 0.0
    original_layers: int = 0
    edited_layers: int = 0
    original_vocab: int = 0
    edited_vocab: int = 0
    size_bytes: int = 0
    success: bool = True
    error: Optional[str] = None
    duration_seconds: float = 0.0
    details: dict[str, Any] = {}


def list_edit_operations() -> list[dict[str, Any]]:
    """Return available editing operations with descriptions."""
    return [
        {
            "id": "layer_remove",
            "name": "Layer Removal",
            "description": (
                "Remove specific transformer layers to reduce model depth. "
                "Middle layers are often most redundant. Reduces params proportionally."
            ),
            "parameters": ["layers_to_remove (list of layer indices)"],
        },
        {
            "id": "weight_merge",
            "name": "Weight Merging (SLERP/Linear)",
            "description": (
                "Blend weights of two compatible models. Linear interpolation or "
                "SLERP for smoother blending. Creates a model combining both models' strengths."
            ),
            "parameters": ["merge_model, merge_method (linear|slerp), merge_alpha (0.0-1.0)"],
        },
        {
            "id": "vocab_resize",
            "name": "Vocabulary Resize",
            "description": (
                "Extend or shrink the model's vocabulary and embedding layers. "
                "Useful for adding domain-specific tokens or reducing memory footprint."
            ),
            "parameters": ["new_vocab_size (target size)"],
        },
        {
            "id": "head_prune",
            "name": "Attention Head Pruning",
            "description": (
                "Remove low-importance attention heads. Reduces compute per layer "
                "without removing entire layers. Can auto-detect least-important heads."
            ),
            "parameters": ["num_heads_to_prune (auto) or heads_to_prune ({layer: [heads]})"],
        },
    ]


def suggest_edits(
    params_b: float,
    num_layers: int,
    vram_mb: int,
    target_vram_mb: int,
) -> dict[str, Any]:
    """Suggest model editing operations to fit within target VRAM."""
    fp16_mb = int(params_b * 2000 * 1.15)

    if fp16_mb <= target_vram_mb:
        return {
            "editing_needed": False,
            "message": f"Model ({fp16_mb} MB) fits in {target_vram_mb} MB without editing.",
        }

    reduction_needed = (fp16_mb - target_vram_mb) / fp16_mb
    layers_to_remove_count = max(1, int(num_layers * reduction_needed))

    # Suggest removing middle layers (most redundant)
    mid = num_layers // 2
    half_remove = layers_to_remove_count // 2
    suggested_layers = list(range(mid - half_remove, mid + half_remove + (layers_to_remove_count % 2)))

    estimated_after = int(fp16_mb * (1 - layers_to_remove_count / num_layers))

    return {
        "editing_needed": True,
        "recommended_operation": "layer_remove",
        "layers_to_remove_count": layers_to_remove_count,
        "suggested_layers": suggested_layers,
        "original_vram_mb": fp16_mb,
        "estimated_vram_after_mb": estimated_after,
        "target_vram_mb": target_vram_mb,
        "total_layers": num_layers,
        "message": (
            f"Remove {layers_to_remove_count}/{num_layers} middle layers "
            f"(indices {suggested_layers}) to reduce from {fp16_mb} MB to ~{estimated_after} MB."
        ),
    }


def _slerp(t: float, v0, v1, eps: float = 1e-8):
    """Spherical linear interpolation between two tensors."""
    import torch

    v0_flat = v0.flatten().float()
    v1_flat = v1.flatten().float()

    v0_norm = v0_flat / (v0_flat.norm() + eps)
    v1_norm = v1_flat / (v1_flat.norm() + eps)

    dot = torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0)
    omega = torch.acos(dot)

    if omega.abs() < eps:
        # Nearly parallel — fall back to linear
        result = (1.0 - t) * v0_flat + t * v1_flat
    else:
        so = torch.sin(omega)
        result = (torch.sin((1.0 - t) * omega) / so) * v0_flat + (torch.sin(t * omega) / so) * v1_flat

    return result.reshape(v0.shape).to(v0.dtype)


def _edit_sync(config: EditConfig) -> EditResult:
    """Synchronous model editing."""
    start = time.time()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        return EditResult(
            success=False,
            error=f"Model editing requires torch and transformers: {e}",
            operation=config.operation,
            duration_seconds=round(time.time() - start, 2),
        )

    hf_token = os.environ.get("HF_TOKEN")
    trc = _trust_remote()

    op_label = config.operation.replace("_", "-")
    out_dir = Path(config.output_dir) / (
        config.output_name
        or f"{_safe_name(config.source_model)}-{op_label}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        log.info("Loading %s for %s editing", config.source_model, config.operation)

        tokenizer = AutoTokenizer.from_pretrained(
            config.source_model, token=hf_token, trust_remote_code=trc,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.source_model,
            token=hf_token,
            trust_remote_code=trc,
            torch_dtype=torch.float16,
            device_map="cpu",
        )

        original_params = sum(p.numel() for p in model.parameters())
        original_vocab = model.config.vocab_size if hasattr(model.config, 'vocab_size') else 0

        # Get layer count
        layer_attr = None
        for attr in ["model.layers", "transformer.h", "gpt_neox.layers"]:
            parts = attr.split(".")
            obj = model
            try:
                for part in parts:
                    obj = getattr(obj, part)
                layer_attr = attr
                original_layers = len(obj)
                break
            except AttributeError:
                continue
        else:
            original_layers = 0

        details: dict[str, Any] = {}

        if config.operation == EditOperation.LAYER_REMOVE:
            result = _do_layer_remove(model, config, layer_attr, original_layers)
            details = result

        elif config.operation == EditOperation.WEIGHT_MERGE:
            result = _do_weight_merge(model, config, hf_token)
            details = result

        elif config.operation == EditOperation.VOCAB_RESIZE:
            result = _do_vocab_resize(model, tokenizer, config)
            details = result

        elif config.operation == EditOperation.HEAD_PRUNE:
            result = _do_head_prune(model, config, original_layers)
            details = result

        if "error" in details:
            return EditResult(
                success=False,
                error=details["error"],
                operation=config.operation,
                duration_seconds=round(time.time() - start, 2),
            )

        # Save
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        edited_params = sum(p.numel() for p in model.parameters())
        edited_vocab = model.config.vocab_size if hasattr(model.config, 'vocab_size') else 0

        # Count edited layers
        edited_layers = original_layers
        if layer_attr:
            parts = layer_attr.split(".")
            obj = model
            try:
                for part in parts:
                    obj = getattr(obj, part)
                edited_layers = len(obj)
            except AttributeError:
                pass

        size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())

        return EditResult(
            output_model=str(out_dir.name),
            output_path=str(out_dir),
            operation=config.operation,
            original_params=original_params,
            edited_params=edited_params,
            param_reduction_pct=round((1 - edited_params / original_params) * 100, 1) if original_params else 0,
            original_layers=original_layers,
            edited_layers=edited_layers,
            original_vocab=original_vocab,
            edited_vocab=edited_vocab,
            size_bytes=size,
            success=True,
            duration_seconds=round(time.time() - start, 2),
            details=details,
        )

    except Exception as e:
        log.exception("Model editing failed")
        return EditResult(
            success=False,
            error=str(e),
            operation=config.operation,
            duration_seconds=round(time.time() - start, 2),
        )


def _do_layer_remove(model, config: EditConfig, layer_attr: str | None, num_layers: int) -> dict:
    """Remove specified transformer layers."""
    import torch.nn as nn

    if not layer_attr:
        return {"error": "Cannot find transformer layers in this model architecture"}

    if not config.layers_to_remove:
        return {"error": "No layers specified for removal"}

    valid_indices = [i for i in config.layers_to_remove if 0 <= i < num_layers]
    if not valid_indices:
        return {"error": f"No valid layer indices (model has {num_layers} layers)"}

    # Navigate to layers
    parts = layer_attr.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    layers_list = list(getattr(parent, parts[-1]))

    # Remove layers (reverse order to preserve indices)
    for idx in sorted(valid_indices, reverse=True):
        layers_list.pop(idx)

    setattr(parent, parts[-1], nn.ModuleList(layers_list))

    # Update config
    if hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = len(layers_list)

    return {
        "removed_layers": valid_indices,
        "remaining_layers": len(layers_list),
    }


def _do_weight_merge(model, config: EditConfig, hf_token: str | None) -> dict:
    """Merge weights of two models."""
    import torch
    from transformers import AutoModelForCausalLM

    if not config.merge_model:
        return {"error": "merge_model is required for weight merging"}

    merge_model = AutoModelForCausalLM.from_pretrained(
        config.merge_model,
        token=hf_token,
        trust_remote_code=_trust_remote(),
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    alpha = config.merge_alpha
    merged_count = 0
    skipped_count = 0

    source_dict = dict(model.named_parameters())
    merge_dict = dict(merge_model.named_parameters())

    for name, param in source_dict.items():
        if name in merge_dict and param.shape == merge_dict[name].shape:
            if config.merge_method == MergeMethod.SLERP:
                param.data = _slerp(alpha, param.data, merge_dict[name].data)
            else:
                param.data = (1 - alpha) * param.data + alpha * merge_dict[name].data
            merged_count += 1
        else:
            skipped_count += 1

    del merge_model

    return {
        "merge_model": config.merge_model,
        "merge_method": config.merge_method,
        "alpha": alpha,
        "merged_params": merged_count,
        "skipped_params": skipped_count,
    }


def _do_vocab_resize(model, tokenizer, config: EditConfig) -> dict:
    """Resize vocabulary and embedding layers."""
    if config.new_vocab_size <= 0:
        return {"error": "new_vocab_size must be positive"}

    old_size = model.config.vocab_size
    new_size = config.new_vocab_size

    if new_size == old_size:
        return {"message": "Vocab size unchanged", "old_size": old_size, "new_size": new_size}

    model.resize_token_embeddings(new_size)

    # If expanding, add placeholder tokens
    if new_size > old_size:
        for i in range(old_size, new_size):
            tokenizer.add_tokens([f"<extra_{i}>"])

    return {
        "old_vocab_size": old_size,
        "new_vocab_size": new_size,
        "direction": "expanded" if new_size > old_size else "shrunk",
    }


def _do_head_prune(model, config: EditConfig, num_layers: int) -> dict:
    """Prune attention heads by importance."""
    import torch

    heads_to_prune = config.heads_to_prune

    if not heads_to_prune and config.num_heads_to_prune > 0:
        # Auto-detect least important heads using gradient-free importance
        heads_to_prune = _auto_detect_heads(model, config.num_heads_to_prune, num_layers)

    if not heads_to_prune:
        return {"error": "No heads specified for pruning. Use heads_to_prune or num_heads_to_prune."}

    # Use HF's built-in head pruning if available
    if hasattr(model, "prune_heads"):
        model.prune_heads(heads_to_prune)
        total_pruned = sum(len(h) for h in heads_to_prune.values())
        return {
            "pruned_heads": heads_to_prune,
            "total_heads_pruned": total_pruned,
            "method": "native_prune_heads",
        }
    else:
        return {"error": "Model does not support head pruning (no prune_heads method)"}


def _auto_detect_heads(model, n_prune: int, num_layers: int) -> dict[int, list[int]]:
    """Detect least important attention heads by weight magnitude."""
    import torch

    head_importance: list[tuple[int, int, float]] = []  # (layer, head, importance)

    for layer_idx in range(num_layers):
        # Try to find attention weights
        for attr_path in [
            f"model.layers.{layer_idx}.self_attn",
            f"transformer.h.{layer_idx}.attn",
        ]:
            parts = attr_path.split(".")
            obj = model
            try:
                for part in parts:
                    obj = getattr(obj, part)
                # Found attention module
                num_heads = getattr(obj, "num_heads", getattr(model.config, "num_attention_heads", 0))
                if num_heads == 0:
                    continue

                # Get query weight as importance proxy
                q_weight = None
                for w_name in ["q_proj.weight", "query.weight", "c_attn.weight"]:
                    w_parts = w_name.split(".")
                    w_obj = obj
                    try:
                        for wp in w_parts:
                            w_obj = getattr(w_obj, wp)
                        q_weight = w_obj
                        break
                    except AttributeError:
                        continue

                if q_weight is not None:
                    head_dim = q_weight.shape[0] // num_heads
                    for h in range(num_heads):
                        h_weight = q_weight[h * head_dim : (h + 1) * head_dim]
                        importance = float(h_weight.abs().mean())
                        head_importance.append((layer_idx, h, importance))
                break
            except AttributeError:
                continue

    # Sort by importance (ascending) and pick N least important
    head_importance.sort(key=lambda x: x[2])
    heads_to_prune: dict[int, list[int]] = {}

    for layer_idx, head_idx, _ in head_importance[:n_prune]:
        heads_to_prune.setdefault(layer_idx, []).append(head_idx)

    return heads_to_prune


async def edit_model(config: EditConfig) -> EditResult:
    """Edit a model — runs in thread pool to avoid blocking the event loop."""
    return await asyncio.to_thread(_edit_sync, config)
