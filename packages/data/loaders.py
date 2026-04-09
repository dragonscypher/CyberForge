"""Unified dataset loaders for IDS benchmark datasets.

Re-exports the per-dataset loaders from packages.cyber.datasets and adds
a unified ``load_dataset()`` dispatcher for convenience.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from packages.cyber.datasets import (load_cicids2017, load_cse_cic_ids2018,
                                     load_nsl_kdd, load_unsw_nb15)

log = logging.getLogger(__name__)

__all__ = [
    "load_nsl_kdd",
    "load_cicids2017",
    "load_cse_cic_ids2018",
    "load_unsw_nb15",
    "load_dataset",
    "list_datasets",
]

# Dataset registry
_DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "nsl-kdd": {
        "name": "NSL-KDD",
        "description": "Refined KDD'99 for network intrusion detection",
        "loader": load_nsl_kdd,
        "classes": ["normal", "attack"],
        "features": 41,
        "default_label": "label2",
    },
    "cicids2017": {
        "name": "CICIDS2017",
        "description": "Canadian Institute for Cybersecurity IDS dataset 2017",
        "loader": load_cicids2017,
        "classes": ["normal", "attack"],
        "features": 78,
        "default_label": "label2",
    },
    "cse-cic-ids2018": {
        "name": "CSE-CIC-IDS2018",
        "description": "Communications Security Establishment IDS dataset 2018",
        "loader": load_cse_cic_ids2018,
        "classes": ["normal", "attack"],
        "features": 78,
        "default_label": "label2",
    },
    "unsw-nb15": {
        "name": "UNSW-NB15",
        "description": "University of New South Wales network dataset",
        "loader": load_unsw_nb15,
        "classes": ["normal", "attack"],
        "features": 49,
        "default_label": "label2",
    },
}


def list_datasets() -> list[dict[str, Any]]:
    """Return metadata for all registered datasets."""
    return [
        {
            "id": k,
            "name": v["name"],
            "description": v["description"],
            "classes": v["classes"],
            "features": v["features"],
        }
        for k, v in _DATASET_REGISTRY.items()
    ]


def load_dataset(dataset_id: str, path: str | Path | None = None) -> pd.DataFrame:
    """Load a dataset by registry ID.

    Args:
        dataset_id: One of 'nsl-kdd', 'cicids2017', 'cse-cic-ids2018', 'unsw-nb15'.
        path: Optional path to the data file(s).  If None, the loader will use
              its default discovery logic.

    Returns:
        A pandas DataFrame with at minimum a ``label2`` column (binary: normal/attack).
    """
    entry = _DATASET_REGISTRY.get(dataset_id)
    if entry is None:
        raise ValueError(f"Unknown dataset '{dataset_id}'. Available: {list(_DATASET_REGISTRY)}")

    loader = entry["loader"]
    if path is not None:
        return loader(str(path))
    return loader()
