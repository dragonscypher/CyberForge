"""HuggingFace downloader — authenticated snapshot download with progress."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class DownloadResult(BaseModel):
    repo_id: str
    local_path: str
    size_mb: float = 0.0
    success: bool = True
    error: Optional[str] = None


class HFDownloader:
    """Download models from Hugging Face Hub into a local cache directory."""

    def __init__(self, cache_dir: str = "data/cache", token: Optional[str] = None):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._token = token or os.environ.get("HF_TOKEN")

    def download(
        self,
        repo_id: str,
        revision: str = "main",
        allow_patterns: Optional[list[str]] = None,
    ) -> DownloadResult:
        try:
            from huggingface_hub import snapshot_download

            local_dir = self._cache_dir / repo_id.replace("/", "--")

            # Try anonymous first (works for public repos without a token)
            try:
                path = snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    local_dir=str(local_dir),
                    token=None,
                    allow_patterns=allow_patterns,
                )
                size = _dir_size_mb(Path(path))
                return DownloadResult(repo_id=repo_id, local_path=str(path), size_mb=size)
            except Exception as anon_err:
                err_str = str(anon_err).lower()
                # If it's an auth/gated error and we have a token, retry with token
                if self._token and ("401" in err_str or "403" in err_str
                                     or "gated" in err_str or "access" in err_str
                                     or "authentication" in err_str):
                    path = snapshot_download(
                        repo_id=repo_id,
                        revision=revision,
                        local_dir=str(local_dir),
                        token=self._token,
                        allow_patterns=allow_patterns,
                    )
                    size = _dir_size_mb(Path(path))
                    return DownloadResult(repo_id=repo_id, local_path=str(path), size_mb=size)
                raise  # Re-raise if not an auth error or no token available
        except Exception as e:
            return DownloadResult(repo_id=repo_id, local_path="", success=False, error=str(e))

    def is_cached(self, repo_id: str) -> bool:
        local_dir = self._cache_dir / repo_id.replace("/", "--")
        return local_dir.exists() and any(local_dir.iterdir())

    def cached_path(self, repo_id: str) -> Optional[str]:
        local_dir = self._cache_dir / repo_id.replace("/", "--")
        if local_dir.exists() and any(local_dir.iterdir()):
            return str(local_dir)
        return None

    def list_cached(self) -> list[str]:
        if not self._cache_dir.exists():
            return []
        return [
            d.name.replace("--", "/")
            for d in self._cache_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

    def delete_cached(self, repo_id: str) -> bool:
        import shutil

        local_dir = self._cache_dir / repo_id.replace("/", "--")
        if local_dir.exists():
            shutil.rmtree(local_dir)
            return True
        return False


def _dir_size_mb(p: Path) -> float:
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 2)
