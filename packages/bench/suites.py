"""Benchmark suite definitions — v1 required suites per plan.md Section 8.

Provides the 6 required suites: general_smoke_v1, coding_unit_tests_v1,
cyber_rulegen_smoke_v1, cyber_log_triage_v1, ids_structured_nslkdd_v1,
ids_transfer_modern_v1.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class SuiteCase(BaseModel):
    """A single benchmark case within a suite."""
    case_id: str
    prompt: str = ""
    keywords: list[str] = []
    expects_json: bool = False
    artifact: str = ""  # sigma | yara | suricata | ""
    metadata: dict[str, Any] = {}


class BenchmarkSuiteConfig(BaseModel):
    """Describes one benchmark suite conforming to plan.md Section 8 contract."""
    id: str
    name: str
    task_mode: str  # general | coding | cyber | ids
    version: str = "1"
    description: str = ""
    cases: list[SuiteCase] = []
    ids_config: dict[str, Any] = {}  # Only for IDS-type suites
    created_at: float = Field(default_factory=time.time)


# ── Normalized benchmark summary (plan.md §8 contract) ──────────


class BenchmarkSummary(BaseModel):
    """Normalized benchmark summary schema — emitted by every suite."""
    quality: dict[str, float] = {}
    reliability: dict[str, float] = {}
    efficiency: dict[str, float] = {}


# ── Built-in suite definitions ───────────────────────────────────


def _general_smoke_v1() -> BenchmarkSuiteConfig:
    return BenchmarkSuiteConfig(
        id="general_smoke_v1",
        name="General Smoke v1",
        task_mode="general",
        description="Prompt-following, summarization, extraction, structured output validity.",
        cases=[
            SuiteCase(
                case_id="gen_json_1",
                prompt='Reply with valid JSON only: {"task":"summarize","answer":"what is least privilege"}.',
                keywords=["least", "privilege"],
                expects_json=True,
            ),
            SuiteCase(
                case_id="gen_latency",
                prompt="In one sentence, explain why low latency matters for local copilots.",
                keywords=["latency"],
            ),
            SuiteCase(
                case_id="gen_extraction",
                prompt="Extract all IP addresses from this text: 'Server 10.0.0.1 connected to 192.168.1.5 via proxy.' Reply JSON list.",
                keywords=["10.0.0.1", "192.168.1.5"],
                expects_json=True,
            ),
            SuiteCase(
                case_id="gen_summary",
                prompt="Summarise in 2 sentences: Zero Trust Architecture assumes no implicit trust. Every access request is verified.",
                keywords=["trust", "verified"],
            ),
        ],
    )


def _coding_unit_tests_v1() -> BenchmarkSuiteConfig:
    return BenchmarkSuiteConfig(
        id="coding_unit_tests_v1",
        name="Coding Unit Tests v1",
        task_mode="coding",
        description="Code generation, unit-test pass rate, syntax validity, retry success rate.",
        cases=[
            SuiteCase(
                case_id="code_add",
                prompt="Write Python code only: a function named add_numbers(nums) that returns the sum of a list.",
                keywords=["def", "return", "add_numbers"],
            ),
            SuiteCase(
                case_id="code_fizzbuzz",
                prompt="Write Python: function fizzbuzz(n) returns list of strings for 1..n with FizzBuzz rules.",
                keywords=["def", "fizzbuzz", "return"],
            ),
            SuiteCase(
                case_id="code_json",
                prompt='Reply with valid JSON only: {"language":"python","topic":"type hints"}.',
                keywords=["python"],
                expects_json=True,
            ),
            SuiteCase(
                case_id="code_sort",
                prompt="Write Python: function merge_sort(arr) that returns a sorted list using merge sort.",
                keywords=["def", "merge_sort", "return"],
            ),
        ],
    )


def _cyber_rulegen_smoke_v1() -> BenchmarkSuiteConfig:
    return BenchmarkSuiteConfig(
        id="cyber_rulegen_smoke_v1",
        name="Cyber Rule Generation Smoke v1",
        task_mode="cyber",
        description="Rule generation tasks with verifier-backed pass rate.",
        cases=[
            SuiteCase(
                case_id="cyber_sigma_ps",
                prompt="Write a Sigma rule in YAML that detects suspicious PowerShell encoded command usage.",
                keywords=["title", "logsource", "detection"],
                artifact="sigma",
            ),
            SuiteCase(
                case_id="cyber_sigma_dll",
                prompt="Write a Sigma rule in YAML for detecting DLL side-loading via rundll32.",
                keywords=["title", "detection"],
                artifact="sigma",
            ),
            SuiteCase(
                case_id="cyber_yara_pe",
                prompt="Write a YARA rule to detect packed PE executables with UPX signatures.",
                keywords=["rule", "condition"],
                artifact="yara",
            ),
            SuiteCase(
                case_id="cyber_json",
                prompt='Reply with valid JSON only: {"ioc_type":"domain","ioc":"example.bad","severity":"high"}.',
                keywords=["ioc"],
                expects_json=True,
            ),
        ],
    )


def _cyber_log_triage_v1() -> BenchmarkSuiteConfig:
    return BenchmarkSuiteConfig(
        id="cyber_log_triage_v1",
        name="Cyber Log Triage v1",
        task_mode="cyber",
        description="Log summarization, incident report quality, ATT&CK mapping.",
        cases=[
            SuiteCase(
                case_id="triage_sysmon",
                prompt="Summarize this Sysmon event and assign a severity: Process Create, Image=cmd.exe, ParentImage=winword.exe, CommandLine='cmd /c whoami'. Reply JSON with severity and summary.",
                keywords=["severity", "summary"],
                expects_json=True,
            ),
            SuiteCase(
                case_id="triage_attack_map",
                prompt="Map this activity to MITRE ATT&CK: An attacker used PowerShell to download and execute a remote script via IEX. Return JSON with technique_id and tactic.",
                keywords=["T1059", "execution"],
                expects_json=True,
            ),
            SuiteCase(
                case_id="triage_incident",
                prompt="Write a brief incident summary for: Multiple failed SSH logins from 203.0.113.5, followed by successful login and new cron job creation.",
                keywords=["brute", "persistence"],
            ),
            SuiteCase(
                case_id="triage_ioc",
                prompt="Extract IOCs from: 'Malware beacon to c2.evil.com:443, dropped payload at C:\\Temp\\svc.exe, modified HKLM\\...\\Run'. Reply JSON array.",
                keywords=["c2.evil.com", "svc.exe"],
                expects_json=True,
            ),
        ],
    )


def _ids_structured_nslkdd_v1() -> BenchmarkSuiteConfig:
    return BenchmarkSuiteConfig(
        id="ids_structured_nslkdd_v1",
        name="IDS Structured NSL-KDD v1",
        task_mode="ids",
        description="Classical IDS baselines — RF/XGB on NSL-KDD with ROC-AUC, PR-AUC, F1, FAR, detection rate.",
        ids_config={
            "runner": "IDSStructuredRunner",
            "method": "run_nsl_kdd",
            "classifiers": ["rf", "xgb"],
            "n_estimators": 100,
            "test_ratio": 0.2,
        },
    )


def _ids_transfer_modern_v1() -> BenchmarkSuiteConfig:
    return BenchmarkSuiteConfig(
        id="ids_transfer_modern_v1",
        name="IDS Transfer Modern v1",
        task_mode="ids",
        description="Cross-dataset transfer — train on NSL-KDD, test on CICIDS2017 / CSE-CIC-IDS2018 / UNSW-NB15.",
        ids_config={
            "runner": "IDSStructuredRunner",
            "method": "run_cross_dataset",
            "train_dataset": "nsl_kdd",
            "test_datasets": ["cicids2017", "cse_cic_ids2018", "unsw_nb15"],
            "classifiers": ["rf"],
            "n_estimators": 100,
        },
    )


# ── Suite registry ───────────────────────────────────────────────

_BUILTIN_SUITES: dict[str, BenchmarkSuiteConfig] = {}


def _init_builtins():
    global _BUILTIN_SUITES
    if _BUILTIN_SUITES:
        return
    for factory in (
        _general_smoke_v1,
        _coding_unit_tests_v1,
        _cyber_rulegen_smoke_v1,
        _cyber_log_triage_v1,
        _ids_structured_nslkdd_v1,
        _ids_transfer_modern_v1,
    ):
        suite = factory()
        _BUILTIN_SUITES[suite.id] = suite


def get_suite(suite_id: str) -> Optional[BenchmarkSuiteConfig]:
    """Get a benchmark suite by ID."""
    _init_builtins()
    return _BUILTIN_SUITES.get(suite_id)


def list_suites() -> list[BenchmarkSuiteConfig]:
    """List all registered benchmark suites."""
    _init_builtins()
    return list(_BUILTIN_SUITES.values())


def register_suite(suite: BenchmarkSuiteConfig) -> None:
    """Register a custom benchmark suite."""
    _init_builtins()
    _BUILTIN_SUITES[suite.id] = suite


async def run_ids_suite(suite: BenchmarkSuiteConfig) -> BenchmarkSummary:
    """Execute an IDS-type benchmark suite and return normalized summary."""
    from packages.bench.ids_runner import IDSStructuredRunner

    cfg = suite.ids_config
    runner = IDSStructuredRunner()
    method = cfg.get("method", "run_nsl_kdd")

    results = []
    if method == "run_nsl_kdd":
        for clf_name in cfg.get("classifiers", ["rf"]):
            result = runner.run_nsl_kdd(
                classifier=clf_name,
                n_estimators=cfg.get("n_estimators", 100),
                test_ratio=cfg.get("test_ratio", 0.2),
            )
            results.append(result)
    elif method == "run_cross_dataset":
        from pathlib import Path as _P

        from packages.cyber.datasets import load_nsl_kdd

        # Auto-discover train path
        train_path = None
        for c in [_P("KDDTrain+.txt"), _P("data/KDDTrain+.txt")]:
            if c.exists():
                train_path = str(c)
                break
        if train_path is None:
            return BenchmarkSummary()

        train_df = load_nsl_kdd(train_path)
        for clf_name in cfg.get("classifiers", ["rf"]):
            for test_ds in cfg.get("test_datasets", []):
                try:
                    loader = _get_dataset_loader(test_ds)
                    test_df = loader()
                    result = runner.run_cross_dataset(
                        train_df=train_df,
                        test_df=test_df,
                        dataset_name=f"{test_ds}_transfer",
                        classifier=clf_name,
                        n_estimators=cfg.get("n_estimators", 100),
                    )
                    results.append(result)
                except Exception:
                    pass  # Skip unavailable datasets

    if not results:
        return BenchmarkSummary()

    # Aggregate: average across all runs
    quality: dict[str, float] = {}
    reliability: dict[str, float] = {}
    efficiency: dict[str, float] = {}
    for r in results:
        for k, v in r.quality.items():
            quality[k] = quality.get(k, 0) + v / len(results)
        for k, v in r.reliability.items():
            reliability[k] = reliability.get(k, 0) + v / len(results)
        for k, v in r.efficiency.items():
            efficiency[k] = efficiency.get(k, 0) + v / len(results)

    return BenchmarkSummary(quality=quality, reliability=reliability, efficiency=efficiency)


def _get_dataset_loader(name: str):
    """Resolve dataset loader by name."""
    from packages.cyber import datasets
    loaders = {
        "nsl_kdd": datasets.load_nsl_kdd,
        "cicids2017": datasets.load_cicids2017,
        "cse_cic_ids2018": datasets.load_cse_cic_ids2018,
        "unsw_nb15": datasets.load_unsw_nb15,
    }
    loader = loaders.get(name)
    if loader is None:
        raise ValueError(f"Unknown dataset: {name}")
    return loader
