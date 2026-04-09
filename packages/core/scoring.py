"""Scoring formula — weighted multi-factor model score."""

from __future__ import annotations

from pydantic import BaseModel


class ScoreWeights(BaseModel):
    quality: float = 0.35
    reliability: float = 0.25
    efficiency: float = 0.25
    compatibility: float = 0.15


class ScoreBreakdown(BaseModel):
    quality: float = 0.0
    reliability: float = 0.0
    efficiency: float = 0.0
    compatibility: float = 0.0
    final: float = 0.0


# Task-mode weight presets (plan.md §7)
WEIGHT_PRESETS: dict[str, ScoreWeights] = {
    "general": ScoreWeights(quality=0.35, reliability=0.20, efficiency=0.30, compatibility=0.15),
    "coding": ScoreWeights(quality=0.35, reliability=0.25, efficiency=0.20, compatibility=0.20),
    "cyber": ScoreWeights(quality=0.25, reliability=0.35, efficiency=0.15, compatibility=0.25),
}


def score_model(
    quality: float = 0.0,
    reliability: float = 0.0,
    efficiency: float = 0.0,
    compatibility: float = 0.0,
    task_mode: str = "general",
    custom_weights: ScoreWeights | None = None,
) -> ScoreBreakdown:
    """Compute weighted recommendation score. All input values should be 0.0–1.0."""
    w = custom_weights or WEIGHT_PRESETS.get(task_mode, WEIGHT_PRESETS["general"])
    final = (
        quality * w.quality
        + reliability * w.reliability
        + efficiency * w.efficiency
        + compatibility * w.compatibility
    )
    return ScoreBreakdown(
        quality=quality,
        reliability=reliability,
        efficiency=efficiency,
        compatibility=compatibility,
        final=round(final, 4),
    )
