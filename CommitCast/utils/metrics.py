"""Metrics reported by the replay predictor."""

from __future__ import annotations

import numpy as np
from fcr.scoring import errors, gain


def error_sums(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float, int]:
    return errors(target, prediction)


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
    return gain(float(base), float(candidate))

