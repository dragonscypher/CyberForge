"""Training calibration metrics — Brier score, ECE, and confidence histograms.

Extends packages.bench.metrics with training-aware calibration analysis:
temperature scaling helpers, reliability diagrams, and post-hoc calibration.
"""

from __future__ import annotations

import math
from typing import Any

from packages.bench.metrics import brier_score, expected_calibration_error

__all__ = [
    "brier_score",
    "expected_calibration_error",
    "calibration_summary",
    "confidence_histogram",
    "temperature_scale",
]


def calibration_summary(
    probs: list[float],
    actuals: list[int],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute a full calibration summary including Brier, ECE, and histogram.

    Returns:
        Dict with keys: brier, ece, n_samples, histogram (list of bin dicts),
        mean_confidence, mean_accuracy, calibration_gap.
    """
    bs = brier_score(probs, actuals)
    ece = expected_calibration_error(probs, actuals, n_bins=n_bins)
    hist = confidence_histogram(probs, actuals, n_bins=n_bins)

    mean_conf = sum(probs) / len(probs) if probs else 0.0
    mean_acc = sum(actuals) / len(actuals) if actuals else 0.0

    return {
        "brier": bs,
        "ece": ece,
        "n_samples": len(probs),
        "histogram": hist,
        "mean_confidence": mean_conf,
        "mean_accuracy": mean_acc,
        "calibration_gap": abs(mean_conf - mean_acc),
    }


def confidence_histogram(
    probs: list[float],
    actuals: list[int],
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Build a confidence histogram with per-bin accuracy and count.

    Returns:
        List of dicts, one per bin: {bin_start, bin_end, count, accuracy, avg_confidence}.
    """
    bins: list[dict[str, Any]] = []
    bin_width = 1.0 / n_bins

    for i in range(n_bins):
        lo = i * bin_width
        hi = (i + 1) * bin_width

        indices = [
            j for j, p in enumerate(probs)
            if (lo <= p < hi) or (i == n_bins - 1 and p == hi)
        ]

        count = len(indices)
        if count > 0:
            acc = sum(actuals[j] for j in indices) / count
            avg_conf = sum(probs[j] for j in indices) / count
        else:
            acc = 0.0
            avg_conf = (lo + hi) / 2

        bins.append({
            "bin_start": round(lo, 3),
            "bin_end": round(hi, 3),
            "count": count,
            "accuracy": round(acc, 4),
            "avg_confidence": round(avg_conf, 4),
        })

    return bins


def temperature_scale(logits: list[float], temperature: float = 1.0) -> list[float]:
    """Apply temperature scaling to logits and return calibrated probabilities.

    Temperature > 1 softens the distribution (less confident).
    Temperature < 1 sharpens it (more confident).
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    scaled = [l / temperature for l in logits]

    # Softmax
    max_val = max(scaled) if scaled else 0.0
    exps = [math.exp(s - max_val) for s in scaled]
    total = sum(exps)

    if total == 0:
        return [1.0 / len(scaled)] * len(scaled)

    return [e / total for e in exps]
