"""Chart generation for benchmark reports using matplotlib."""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Sequence

log = logging.getLogger(__name__)


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return f"data:image/png;base64,{b64}"


def latency_comparison_chart(
    labels: Sequence[str],
    p50_values: Sequence[float],
    p95_values: Sequence[float],
) -> str:
    """Generate a grouped bar chart comparing latency P50/P95 across models. Returns base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, p50_values, width, label="P50 (ms)", color="#4C9AFF")
    ax.bar(x + width / 2, p95_values, width, label="P95 (ms)", color="#FF6B6B")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()

    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


def quality_radar_chart(
    labels: Sequence[str],
    scores: dict[str, Sequence[float]],
) -> str:
    """Generate a radar chart comparing quality metrics. Returns base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles.append(angles[0])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})

    colors = ["#4C9AFF", "#FF6B6B", "#4CD964", "#FFB347"]
    for idx, (name, values) in enumerate(scores.items()):
        vals = list(values) + [values[0]]
        color = colors[idx % len(colors)]
        ax.plot(angles, vals, "o-", linewidth=2, label=name, color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Quality Comparison", pad=20)
    fig.tight_layout()

    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


def calibration_chart(
    bins: list[dict[str, float]],
) -> str:
    """Generate a calibration reliability diagram. Returns base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = [b["bin_center"] for b in bins]
    accs = [b["accuracy"] for b in bins]
    confs = [b["confidence"] for b in bins]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(centers, accs, width=0.08, alpha=0.7, label="Accuracy", color="#4C9AFF")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.scatter(centers, confs, color="#FF6B6B", zorder=5, label="Avg confidence")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Actual Frequency")
    ax.set_title("Calibration Diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


def resource_usage_chart(
    labels: Sequence[str],
    ram_mb: Sequence[float],
    vram_mb: Sequence[float],
) -> str:
    """Generate stacked bar chart for RAM/VRAM usage. Returns base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, ram_mb, label="RAM (MB)", color="#4C9AFF")
    ax.bar(x, vram_mb, bottom=ram_mb, label="VRAM (MB)", color="#FFB347")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Resource Usage")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()

    result = _fig_to_base64(fig)
    plt.close(fig)
    return result
