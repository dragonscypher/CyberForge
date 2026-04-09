"""Keyring-backed secret storage — never store raw tokens in SQLite.

Wraps the `keyring` library to store/retrieve/delete secrets under the
CyberForge service namespace. The DB only stores a `secret_ref` handle.
"""

from __future__ import annotations

import keyring

SERVICE_NAME = "cyberforge"


def _key(provider: str, label: str) -> str:
    """Build a keyring username key from provider type and label."""
    return f"{provider}:{label}"


def save_secret(provider: str, label: str, secret: str) -> str:
    """Store a secret in the OS keychain and return the secret_ref handle."""
    ref = _key(provider, label)
    keyring.set_password(SERVICE_NAME, ref, secret)
    return ref


def read_secret(secret_ref: str) -> str | None:
    """Read a secret from the OS keychain. Returns None if not found."""
    return keyring.get_password(SERVICE_NAME, secret_ref)


def delete_secret(secret_ref: str) -> bool:
    """Delete a secret from the OS keychain. Returns True if deleted."""
    try:
        keyring.delete_password(SERVICE_NAME, secret_ref)
        return True
    except keyring.errors.PasswordDeleteError:
        return False
