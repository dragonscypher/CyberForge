"""Recommendation endpoint — suggest best models for task + hardware."""

from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from packages.core.recommend import Recommendation, Recommender

router = APIRouter()


class RecommendRequest(BaseModel):
    task_mode: str = "general"
    subtask: Optional[str] = None  # e.g. "sigma_rule_gen", "log_triage"
    top_k: int = 5
    include_guardrail: bool = True
    include_post_quant: bool = True  # Include models needing quantization
    include_not_recommended: bool = True  # Include too-large models
    latency_target_ms: Optional[float] = None  # hard filter: max acceptable latency
    prefer_installed: bool = False  # rank installed models higher


@router.post("/", response_model=list[Recommendation])
async def get_recommendations(body: RecommendRequest, request: Request):
    """Profile hardware, query registry, return scored recommendations."""
    profile = request.app.state.profiler.profile()
    registry = request.app.state.registry
    recommender = Recommender(registry)
    return recommender.recommend(
        profile=profile,
        task_mode=body.task_mode,
        top_k=body.top_k,
        include_guardrail=body.include_guardrail,
        include_post_quant=body.include_post_quant,
        include_not_recommended=body.include_not_recommended,
    )
