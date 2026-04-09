"""Web-based model discovery — searches HuggingFace Hub for models matching hardware.

Uses the HuggingFace Hub API to find text-generation models filtered by
parameter count, sorted by trending/downloads.  Returns two tiers:

1. **Native tier** — models that fit the hardware as-is (FP16/BF16).
2. **Post-quant tier** — larger models that fit after Q4_K_M quantization
   (~25% of FP16 size) or pruning (~50-70% of original).

Ticket: DISC-001
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

_HF_API = "https://huggingface.co/api/models"


class WebModel(BaseModel):
    """A model discovered from the HuggingFace Hub."""
    model_id: str
    display_name: str
    author: str = ""
    params_b: float = 0.0          # billions
    downloads: int = 0
    likes: int = 0
    pipeline_tag: str = ""
    tags: list[str] = Field(default_factory=list)
    url: str = ""
    tier: str = "native"           # native | post_quant | post_prune
    estimated_vram_mb: int = 0     # FP16 estimate
    estimated_quant_vram_mb: int = 0  # Q4 estimate
    suggested_quant: str = ""      # e.g. q4_K_M, awq, gptq


class WebDiscoveryResult(BaseModel):
    """Full result of a web discovery sweep."""
    native: list[WebModel] = Field(default_factory=list)
    post_quant: list[WebModel] = Field(default_factory=list)
    local_installed: list[dict[str, Any]] = Field(default_factory=list)
    hardware_summary: str = ""
    total_found: int = 0


def _estimate_fp16_vram_mb(params_b: float) -> int:
    """Rough FP16 VRAM estimate: ~2 bytes per param + overhead."""
    return int(params_b * 2000 * 1.15)  # 2GB/B * 1.15 overhead


def _estimate_q4_vram_mb(params_b: float) -> int:
    """Rough Q4_K_M VRAM estimate: ~0.56 bytes/param + overhead."""
    return int(params_b * 560 * 1.2)


def _estimate_pruned_vram_mb(params_b: float, sparsity: float = 0.5) -> int:
    """Rough pruned VRAM estimate at given sparsity."""
    return int(_estimate_fp16_vram_mb(params_b) * (1 - sparsity * 0.4))


def _params_from_safetensors_info(info: dict) -> float:
    """Extract param count from HF API model info."""
    # Try safetensors metadata first
    st = info.get("safetensors", {})
    if isinstance(st, dict):
        params = st.get("total", 0)
        if params > 0:
            return round(params / 1e9, 1)
    # Fallback: library-reported
    card = info.get("cardData", {})
    if isinstance(card, dict):
        p = card.get("parameter_count") or card.get("num_parameters", 0)
        if p:
            return round(int(p) / 1e9, 1)
    return 0.0


def _parse_params_from_name(name: str) -> float:
    """Try to guess param count from model name (e.g. 'Qwen2.5-7B' → 7.0)."""
    import re
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", name)
    if match:
        return float(match.group(1))
    return 0.0


async def discover_from_hub(
    vram_mb: int,
    ram_mb: int,
    task_mode: str = "general",
    limit: int = 30,
    sort: str = "trending",
) -> list[WebModel]:
    """Search HuggingFace Hub for text-generation models.

    Returns models annotated with tier (native / post_quant) based on hardware.
    """
    results: list[WebModel] = []

    # Determine max param ranges
    # Native: what fits in FP16
    max_native_b = vram_mb / 2200 if vram_mb > 0 else ram_mb / 3500
    # Post-quant: what fits after Q4_K_M (roughly 4x native)
    max_quant_b = vram_mb / 560 if vram_mb > 0 else ram_mb / 1000

    # Search HF API
    params = {
        "sort": sort if sort == "trending" else "downloads",
        "direction": "-1",
        "limit": str(limit),
        "filter": "text-generation",
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(_HF_API, params=params)
            resp.raise_for_status()
            models = resp.json()
    except Exception as e:
        log.warning("HF Hub API request failed: %s", e)
        return []

    for m in models:
        model_id = m.get("id", "")
        if not model_id:
            continue

        author = model_id.split("/")[0] if "/" in model_id else ""
        display = model_id.split("/")[-1] if "/" in model_id else model_id

        # Get param count
        params_b = _params_from_safetensors_info(m)
        if params_b == 0:
            params_b = _parse_params_from_name(display)
        if params_b == 0:
            continue  # Skip models with unknown size

        fp16_vram = _estimate_fp16_vram_mb(params_b)
        q4_vram = _estimate_q4_vram_mb(params_b)

        # Determine tier
        if fp16_vram <= max(vram_mb, ram_mb):
            tier = "native"
        elif q4_vram <= max(vram_mb, ram_mb):
            tier = "post_quant"
        else:
            continue  # Too large even after quantization

        # Suggest quantization method
        if tier == "post_quant":
            if vram_mb >= q4_vram:
                suggested = "q4_K_M"
            else:
                suggested = "q4_K_M"  # CPU GGUF fallback
        else:
            suggested = ""

        tags = m.get("tags", [])
        # Filter by task relevance
        if task_mode == "coding":
            coding_signal = any(t in display.lower() for t in ["code", "coder", "starcoder", "deepseek-coder"])
            if not coding_signal and len(results) > 10:
                continue
        elif task_mode == "cyber":
            cyber_signal = any(t in display.lower() for t in ["cyber", "security", "guard", "code"])
            if not cyber_signal and len(results) > 10:
                continue

        results.append(WebModel(
            model_id=model_id,
            display_name=display,
            author=author,
            params_b=params_b,
            downloads=m.get("downloads", 0),
            likes=m.get("likes", 0),
            pipeline_tag=m.get("pipeline_tag", ""),
            tags=tags[:10],  # Cap tags
            url=f"https://huggingface.co/{model_id}",
            tier=tier,
            estimated_vram_mb=fp16_vram,
            estimated_quant_vram_mb=q4_vram,
            suggested_quant=suggested,
        ))

    return results


async def web_discover(
    vram_mb: int,
    ram_mb: int,
    task_mode: str = "general",
    installed_models: list[dict[str, Any]] | None = None,
    limit: int = 30,
) -> WebDiscoveryResult:
    """Full web discovery: HF search + local installed models.

    Args:
        vram_mb: Total GPU VRAM in MiB.
        ram_mb: Total system RAM in MiB.
        task_mode: general | coding | cyber.
        installed_models: List of locally installed model dicts.
        limit: Max models to fetch from HF.

    Returns:
        WebDiscoveryResult with native tier, post_quant tier, and local models.
    """
    hub_models = await discover_from_hub(
        vram_mb=vram_mb,
        ram_mb=ram_mb,
        task_mode=task_mode,
        limit=limit,
    )

    native = [m for m in hub_models if m.tier == "native"]
    post_quant = [m for m in hub_models if m.tier == "post_quant"]

    # Sort: native by downloads, post_quant by downloads
    native.sort(key=lambda m: m.downloads, reverse=True)
    post_quant.sort(key=lambda m: m.downloads, reverse=True)

    max_native_b = vram_mb / 2200 if vram_mb > 0 else ram_mb / 3500
    max_quant_b = vram_mb / 560 if vram_mb > 0 else ram_mb / 1000

    summary = (
        f"VRAM: {vram_mb} MB | RAM: {ram_mb} MB | "
        f"Native capacity: ~{max_native_b:.0f}B FP16 | "
        f"With Q4: ~{max_quant_b:.0f}B"
    )

    return WebDiscoveryResult(
        native=native[:15],
        post_quant=post_quant[:15],
        local_installed=installed_models or [],
        hardware_summary=summary,
        total_found=len(hub_models),
    )
