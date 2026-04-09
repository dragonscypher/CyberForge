"""Cyber datasets — loaders for NSL-KDD, CICIDS2017, CSE-CIC-IDS2018, UNSW-NB15.

Milestone 4 implementation. This skeleton defines the interface plus the NSL-KDD
loader (ported from existing codebase).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel

# Column schema shared by all NSL-KDD files
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "labels", "difficulty",
]

ATTACK_MAP = {
    "normal": "normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "worm": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "httptunnel": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}


class DatasetInfo(BaseModel):
    name: str
    rows: int = 0
    columns: int = 0
    label_distribution: dict[str, int] = {}


def load_nsl_kdd(path: str | Path) -> "pandas.DataFrame":  # type: ignore[name-defined]
    """Load an NSL-KDD TXT file into a pandas DataFrame."""
    import pandas as pd

    df = pd.read_csv(
        path,
        header=None,
        names=NSL_KDD_COLUMNS,
        low_memory=False,
    )
    # 5-class label
    df["label5"] = df["labels"].map(ATTACK_MAP).fillna("unknown")
    # binary label
    df["label2"] = df["labels"].apply(lambda x: "normal" if x == "normal" else "attack")
    return df


def load_cicids2017(path: str | Path) -> "pandas.DataFrame":  # type: ignore
    """Load CICIDS2017 CSV files.

    Accepts a single CSV or a directory containing multiple CSVs.
    Standardises column names: strips whitespace, lowercases, replaces spaces with '_'.
    Adds 'label2' column: 'normal' vs 'attack'.
    """
    import pandas as pd

    path = Path(path)
    if path.is_dir():
        csvs = sorted(path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {path}")
        df = pd.concat([pd.read_csv(f, low_memory=False, encoding="utf-8") for f in csvs], ignore_index=True)
    else:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8")

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Ensure label column exists
    label_col = "label" if "label" in df.columns else None
    if label_col is None:
        for c in df.columns:
            if "label" in c:
                label_col = c
                break
    if label_col:
        df["label2"] = df[label_col].apply(lambda x: "normal" if str(x).strip().upper() == "BENIGN" else "attack")
    return df


def load_cse_cic_ids2018(path: str | Path) -> "pandas.DataFrame":  # type: ignore
    """Load CSE-CIC-IDS2018 CSV files.

    Accepts a single CSV or a directory. Same normalisation as CICIDS2017 since
    the dataset shares the CICFlowMeter schema.
    """
    import pandas as pd

    path = Path(path)
    if path.is_dir():
        csvs = sorted(path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {path}")
        df = pd.concat([pd.read_csv(f, low_memory=False, encoding="utf-8") for f in csvs], ignore_index=True)
    else:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    label_col = "label" if "label" in df.columns else None
    if label_col is None:
        for c in df.columns:
            if "label" in c:
                label_col = c
                break
    if label_col:
        df["label2"] = df[label_col].apply(lambda x: "normal" if str(x).strip().upper() == "BENIGN" else "attack")

    # Handle inf/NaN values common in this dataset
    import numpy as np
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def load_unsw_nb15(path: str | Path) -> "pandas.DataFrame":  # type: ignore
    """Load UNSW-NB15 CSV files.

    Accepts a single CSV or a directory. Expects the standard 49-column schema
    with 'attack_cat' and 'label' columns.
    Adds 'label2': 'normal' (label=0) vs 'attack' (label=1).
    """
    import pandas as pd

    path = Path(path)
    if path.is_dir():
        csvs = sorted(path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {path}")
        df = pd.concat([pd.read_csv(f, low_memory=False, encoding="utf-8") for f in csvs], ignore_index=True)
    else:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # UNSW-NB15 uses integer label: 0=normal, 1=attack
    if "label" in df.columns:
        df["label2"] = df["label"].apply(lambda x: "normal" if int(x) == 0 else "attack")
    if "attack_cat" in df.columns:
        df["attack_cat"] = df["attack_cat"].fillna("Normal").str.strip()
    return df
