"""Settled-risk scalar exposure controller."""

from __future__ import annotations

import math

import numpy as np


def settled_risk_exposure(
    target: np.ndarray,
    checkpoint: np.ndarray,
    proposal: np.ndarray,
    *,
    segment_len: int,
    warmup: int = 16,
    half_life_rows: float = 512.0,
    ridge: float = 1e-6,
    prior_alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale revisions using only fully settled historical service rows."""

    target = np.asarray(target)
    checkpoint = np.asarray(checkpoint)
    proposal = np.asarray(proposal)
    if target.shape != checkpoint.shape or target.shape != proposal.shape or target.ndim != 3:
        raise ValueError("target, checkpoint, and proposal must share [row, lead, channel]")
    if int(segment_len) <= 0:
        raise ValueError("segment_len must be positive")
    if math.isinf(float(half_life_rows)):
        decay = 1.0
    elif float(half_life_rows) > 0:
        decay = math.exp(math.log(0.5) / float(half_life_rows))
    else:
        raise ValueError("half_life_rows must be positive or infinity")

    alpha = np.full(target.shape[0], float(prior_alpha), dtype=np.float64)
    sum_num = 0.0
    sum_den = 0.0
    matured_count = 0
    for row in range(target.shape[0]):
        matured_row = row - int(segment_len)
        if matured_row >= 0:
            base = checkpoint[matured_row].astype(np.float64)
            error = target[matured_row].astype(np.float64) - base
            delta = proposal[matured_row].astype(np.float64) - base
            sum_num = decay * sum_num + float(np.dot(error.ravel(), delta.ravel()))
            sum_den = decay * sum_den + max(
                float(np.dot(delta.ravel(), delta.ravel())), 0.0
            )
            matured_count += 1
        if matured_count >= int(warmup) and sum_den > 1e-12:
            alpha[row] = np.clip(
                sum_num / max(sum_den + max(float(ridge), 0.0), 1e-12),
                0.0,
                1.0,
            )
    mixed = checkpoint + alpha[:, None, None] * (proposal - checkpoint)
    return mixed.astype(np.float32, copy=False), alpha

