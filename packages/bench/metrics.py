"""Metric collection utilities for benchmarking.

Includes timer, throughput, calibration (ECE, Brier), and classification metrics.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator, Sequence


@contextmanager
def timer() -> Generator[dict[str, float], None, None]:
    """Context manager that measures wall-clock time in ms."""
    result: dict[str, float] = {}
    t0 = time.perf_counter()
    yield result
    result["elapsed_ms"] = (time.perf_counter() - t0) * 1000


def tokens_per_second(n_tokens: int, elapsed_ms: float) -> float:
    if elapsed_ms <= 0:
        return 0.0
    return n_tokens / (elapsed_ms / 1000)


# ── Calibration Metrics ──────────────────────────────────────────


def brier_score(
    probabilities: Sequence[float],
    actuals: Sequence[int],
) -> float:
    """Compute Brier score: mean squared error between predicted probabilities and outcomes.

    Lower is better. Range [0, 1]. Perfect calibration → 0.
    """
    if len(probabilities) != len(actuals):
        raise ValueError("probabilities and actuals must have the same length")
    n = len(probabilities)
    if n == 0:
        return 0.0
    return sum((p - a) ** 2 for p, a in zip(probabilities, actuals)) / n


def expected_calibration_error(
    probabilities: Sequence[float],
    actuals: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins predictions by confidence and computes weighted average of
    |accuracy - confidence| per bin. Lower is better. Range [0, 1].
    """
    if len(probabilities) != len(actuals):
        raise ValueError("probabilities and actuals must have the same length")
    n = len(probabilities)
    if n == 0:
        return 0.0

    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0

    for b in range(n_bins):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        indices = [
            i for i, p in enumerate(probabilities)
            if (lo <= p < hi) or (b == n_bins - 1 and p == hi)
        ]
        if not indices:
            continue

        bin_size = len(indices)
        bin_acc = sum(actuals[i] for i in indices) / bin_size
        bin_conf = sum(probabilities[i] for i in indices) / bin_size
        ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return ece


def confidence_histogram(
    probabilities: Sequence[float],
    actuals: Sequence[int],
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Return per-bin calibration data: bin_center, accuracy, confidence, count."""
    n = len(probabilities)
    if n == 0:
        return []

    bins: list[dict[str, float]] = []
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        indices = [
            i for i, p in enumerate(probabilities)
            if (lo <= p < hi) or (b == n_bins - 1 and p == hi)
        ]
        count = len(indices)
        if count == 0:
            bins.append({"bin_center": (lo + hi) / 2, "accuracy": 0.0, "confidence": 0.0, "count": 0})
        else:
            acc = sum(actuals[i] for i in indices) / count
            conf = sum(probabilities[i] for i in indices) / count
            bins.append({"bin_center": (lo + hi) / 2, "accuracy": acc, "confidence": conf, "count": count})
    return bins


# ── Classification Metrics ───────────────────────────────────────


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float] | None = None,
) -> dict[str, float]:
    """Compute classification metrics: accuracy, precision, recall, F1, FAR, detection_rate.

    Optionally computes ROC-AUC and PR-AUC if y_prob is provided.
    """
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)

    result: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    # Binary: compute FAR and detection rate from confusion matrix
    unique = set(y_true)
    if unique <= {0, 1}:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        result["false_alarm_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        result["detection_rate"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if y_prob is not None and len(unique) == 2:
        try:
            result["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            result["roc_auc"] = 0.0
        try:
            result["pr_auc"] = average_precision_score(y_true, y_prob)
        except ValueError:
            result["pr_auc"] = 0.0

    return result
