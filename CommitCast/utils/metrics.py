"""Metrics reported by the replay predictor."""

from __future__ import annotations

import numpy as np


def error_sums(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float, int]:
    diff = np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    return (
        float(np.sum(diff * diff, dtype=np.float64)),
        float(np.sum(np.abs(diff), dtype=np.float64)),
        int(diff.size),
    )


def metric_dict(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    sse, sae, count = error_sums(target, prediction)
    return {
        "sse": sse,
        "sae": sae,
        "evaluated_elements": count,
        "mse": sse / max(1, count),
        "mae": sae / max(1, count),
    }


def gain_pct(base: float, candidate: float) -> float:
    return 100.0 * (float(base) - float(candidate)) / max(float(base), 1e-12)

