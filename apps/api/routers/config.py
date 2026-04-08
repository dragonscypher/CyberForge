"""Config endpoints — first-run check, get/set preferences."""

from fastapi import APIRouter, Request

from packages.core.config import (AppConfig, is_first_run, load_config,
                                  save_config)

router = APIRouter()


@router.get("/first-run")
async def check_first_run():
    return {"first_run": is_first_run()}


@router.get("/", response_model=AppConfig)
async def get_config(request: Request):
    return request.app.state.config


@router.put("/", response_model=AppConfig)
async def update_config(body: AppConfig, request: Request):
    save_config(body)
    request.app.state.config = body
    # Reinitialize dependent clients if credentials changed
    request.app.state.downloader._token = body.hf_token
    request.app.state.ollama._base = body.ollama_base_url.rstrip("/")
    request.app.state.openrouter._api_key = body.openrouter_key or ""
    return body


class OnboardRequest(AppConfig):
    """Same as AppConfig but marks onboarded=True."""
    pass


@router.post("/onboard", response_model=AppConfig)
async def onboard(body: OnboardRequest, request: Request):
    body.onboarded = True
    save_config(body)
    request.app.state.config = body
    return body
