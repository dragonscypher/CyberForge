"""Settings endpoints — bootstrap, get/set with keyring-backed secrets.

Secrets (HF token, OpenRouter key) are stored in the OS keychain via the
keyring library. Only a `secret_ref` handle is persisted to SQLite.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.db import ProviderAccount, Settings, _new_id, _utcnow
from apps.api.security import delete_secret, read_secret, save_secret

router = APIRouter()


# ── Helpers ──────────────────────────────────────────────────────

async def _get_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    async with request.app.state.db_session_factory() as session:
        yield session


# ── Schemas ──────────────────────────────────────────────────────

class BootstrapRequest(BaseModel):
    hf_token: Optional[str] = None
    openrouter_key: Optional[str] = None
    model_cache_dir: str = "data/cache"
    allow_remote_providers: bool = False
    default_task_mode: str = "general"
    privacy_mode: str = "prefer_local"


class BootstrapResponse(BaseModel):
    settings_id: str
    keyring_saved: dict[str, bool]


class SettingsResponse(BaseModel):
    settings_id: str
    model_cache_dir: str
    allow_remote_providers: bool
    default_task_mode: str
    privacy_mode: str


# ── Endpoints ────────────────────────────────────────────────────

@router.post("/bootstrap", response_model=BootstrapResponse)
async def bootstrap(body: BootstrapRequest, session: AsyncSession = Depends(_get_session)):
    """First-run setup: save config to DB, store secrets in keyring."""
    keyring_saved: dict[str, bool] = {}

    # Save HF token
    if body.hf_token:
        ref = save_secret("huggingface", "default", body.hf_token)
        # Upsert provider account
        acct = ProviderAccount(
            id=_new_id(),
            provider_type="huggingface",
            label="default",
            secret_ref=ref,
            enabled=1,
            created_at=_utcnow(),
            updated_at=_utcnow(),
        )
        session.add(acct)
        keyring_saved["hf_token"] = True
    else:
        keyring_saved["hf_token"] = False

    # Save OpenRouter key
    if body.openrouter_key:
        ref = save_secret("openrouter", "default", body.openrouter_key)
        acct = ProviderAccount(
            id=_new_id(),
            provider_type="openrouter",
            label="default",
            secret_ref=ref,
            enabled=1,
            created_at=_utcnow(),
            updated_at=_utcnow(),
        )
        session.add(acct)
        keyring_saved["openrouter_key"] = True
    else:
        keyring_saved["openrouter_key"] = False

    # Save settings row
    settings_id = _new_id()
    row = Settings(
        id=settings_id,
        model_cache_dir=body.model_cache_dir,
        allow_remote_providers=int(body.allow_remote_providers),
        default_task_mode=body.default_task_mode,
        privacy_mode=body.privacy_mode,
        created_at=_utcnow(),
        updated_at=_utcnow(),
    )
    session.add(row)
    await session.commit()

    return BootstrapResponse(settings_id=settings_id, keyring_saved=keyring_saved)


@router.get("/", response_model=SettingsResponse)
async def get_settings(session: AsyncSession = Depends(_get_session)):
    """Return settings (secrets omitted)."""
    result = await session.execute(
        select(Settings).order_by(Settings.created_at.desc()).limit(1)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return SettingsResponse(
            settings_id="",
            model_cache_dir="data/cache",
            allow_remote_providers=False,
            default_task_mode="general",
            privacy_mode="prefer_local",
        )
    return SettingsResponse(
        settings_id=row.id,
        model_cache_dir=row.model_cache_dir,
        allow_remote_providers=bool(row.allow_remote_providers),
        default_task_mode=row.default_task_mode,
        privacy_mode=row.privacy_mode,
    )


@router.delete("/secrets/{provider}/{label}")
async def delete_provider_secret(
    provider: str,
    label: str,
    session: AsyncSession = Depends(_get_session),
):
    """Delete a secret from keyring and remove the provider account row."""
    ref = f"{provider}:{label}"
    deleted = delete_secret(ref)

    # Remove from DB
    result = await session.execute(
        select(ProviderAccount).where(
            ProviderAccount.provider_type == provider,
            ProviderAccount.label == label,
        )
    )
    acct = result.scalar_one_or_none()
    if acct:
        await session.delete(acct)
        await session.commit()

    return {"deleted": deleted, "provider": provider, "label": label}
