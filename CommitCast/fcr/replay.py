"""Fixed and observed-input event schedules for the server CommitCast kernel."""
from __future__ import annotations

import heapq
import numpy as np

from fcr.scoring import SCHEDULES, boundaries, common_origins, summarize_requests
from layers.exposure import settled_risk_exposure
from tta.commitcast import CommitCastConfig, _RidgeSession


def _event_exposure_weights(rows, delay, base, truth, proposal, cfg):
    """Only actually served, completely settled rows enter this offset's gate.

    Queue entries contain observed stopping times. No forecast feature or
    exposure query reads a current row's future stopping time.
    """
    queue = []
    numerator = denominator = 0.0
    matured = 0
    decay = np.exp(np.log(.5) / cfg.srs_half_life_rows)
    error_buffer = np.empty(proposal.shape[1:], dtype=np.float64)
    delta_buffer = np.empty_like(error_buffer)
    for origin, length in rows:
        while queue and queue[0][0] <= origin:
            _, previous, span = heapq.heappop(queue)
            frozen = base[previous+delay, :span]
            error, delta = error_buffer[:span], delta_buffer[:span]
            np.subtract(truth[previous+delay, :span], frozen, dtype=np.float64, out=error)
            np.subtract(proposal[previous, :span], frozen, dtype=np.float64, out=delta)
            numerator = decay*numerator + float(np.dot(error.ravel(), delta.ravel()))
            denominator = decay*denominator + max(float(np.dot(delta.ravel(), delta.ravel())), 0.)
            matured += 1
        alpha = (float(min(max(numerator/(denominator+cfg.srs_ridge), 0.), 1.))
                 if matured >= cfg.srs_warmup_rows and denominator > 1e-12 else 0.)
        if not cfg.srs_enable:
            alpha = 1.
        yield origin, length, alpha
        heapq.heappush(queue, (origin+length, origin, length))


def event_exposures(rows, delay, base, truth, proposal, cfg):
    """Yield causal event predictions and exposure weights in service order."""
    for origin, length, alpha in _event_exposure_weights(rows, delay, base, truth, proposal, cfg):
        frozen = base[origin+delay, :length]
        prediction = frozen + alpha*(proposal[origin, :length]-frozen).astype(np.float64)
        yield origin, length, prediction, alpha


def _event_scores(rows, weights, delay, base, truth, proposal, totals):
    """Score issued exposures offline in bounded batches of equal span.

    All weights have already been fixed by causal settlement. Grouping here
    affects only evaluation; each request still receives one contribution at
    this offset, in the original offset order.
    """
    groups = {}
    channels = base.shape[2]
    for index, (_, length) in enumerate(rows):
        groups.setdefault(length, []).append(index)
    for length, indices in groups.items():
        block = max(1, min(256, (256 * 1024) // (length * channels * 8)))
        for lo in range(0, len(indices), block):
            selected = indices[lo:lo+block]
            origins = np.fromiter((rows[i][0] for i in selected), dtype=np.int64, count=len(selected))
            frozen = base[origins+delay, :length]
            target = truth[origins, delay:delay+length].astype(np.float64)
            frozen_delta = frozen.astype(np.float64) - target
            if delay:
                prediction = frozen + weights[selected, None, None] * (proposal[origins, :length]-frozen).astype(np.float64)
                delta = prediction - target
            else:
                delta = frozen_delta
            losses = np.column_stack((np.square(delta).sum(axis=(1, 2)), np.abs(delta).sum(axis=(1, 2)),
                                      np.square(frozen_delta).sum(axis=(1, 2)), np.abs(frozen_delta).sum(axis=(1, 2))))
            totals[origins, 0] += length * channels
            totals[origins, 1:] += losses


def replay_schedules(base, truth, names, *, cfg=None, support="common", event_plans=None):
    """Return per-schedule metrics and requests on declared complete-H support."""
    cfg = cfg or CommitCastConfig()
    if base.shape != truth.shape or base.ndim != 3 or not np.isfinite(base).all() or not np.isfinite(truth).all():
        raise ValueError("Invalid forecast stream")
    if base.shape[0] < 2 or base.shape[1] < 2 or base.shape[2] < 1:
        raise ValueError("Forecast stream is too small for commitment replay")
    if not np.array_equal(truth[:-1, 1:], truth[1:, :-1]):
        raise ValueError("Targets must overlap chronologically at stride 1")
    if not names or len(set(names)) != len(names) or any(x not in {*SCHEDULES, 'event50'} for x in names):
        raise ValueError("Unknown or duplicate schedule")
    if support not in {"common", "schedule"}:
        raise ValueError("Invalid support profile")
    n, h, channels = base.shape
    support_plans = list(SCHEDULES.values()) if support == "common" else [SCHEDULES.get(x, SCHEDULES['dense8']) for x in names]
    count = common_origins(n, h, support_plans)
    by_delay = {}
    request_rows = {}
    for name in names:
        request_rows[name] = np.zeros((count, 5), dtype=np.float64)
        if name != "event50":
            plan = boundaries(h, SCHEDULES[name])
            for start, end in zip(plan, plan[1:]):
                by_delay.setdefault(start, {})[name] = [(origin, end-start) for origin in range(count)]
            continue
        cuts = event_plans
        if cuts is None or len(cuts) != count:
            raise ValueError("event50 requires validation-frozen, context-aware event plans")
        for origin, plan in enumerate(cuts):
            if plan[0] != 0 or plan[-1] != h or any(b <= a for a,b in zip(plan, plan[1:])):
                raise ValueError("Every request must partition [0,H)")
            for start, end in zip(plan, plan[1:]):
                by_delay.setdefault(start, {}).setdefault(name, []).append((origin, end-start))
    ridge = _RidgeSession(base, truth, cfg) if any(by_delay) else None
    for delay, plans in sorted(by_delay.items()):
        print(f"CommitCast {cfg.feature_mode}: offset {delay}/{h}", flush=True)
        if delay:
            span = h-delay if 'event50' in plans else max(rows[0][1] for rows in plans.values())
            proposal = ridge.revise(delay, span, origins=count)
        fixed_scores = {}
        for name, rows in plans.items():
            if name != 'event50':
                length = rows[0][1]
                if length in fixed_scores:
                    request_rows[name] += fixed_scores[length]
                    continue
                if delay == 0:
                    served = base[:count, :length]
                elif cfg.srs_enable:
                    served, _ = settled_risk_exposure(truth[delay:delay+count,:length], base[delay:delay+count,:length],
                        proposal[:,:length], segment_len=length, warmup=cfg.srs_warmup_rows,
                        half_life_rows=cfg.srs_half_life_rows, ridge=cfg.srs_ridge)
                else:
                    served = proposal[:,:length]
                segment_scores = np.empty((count, 5), dtype=np.float64)
                segment_scores[:, 0] = length * channels
                # Offline scoring can be batched independently of causal updates.
                # Bound temporary FP64 arrays without changing the update block.
                score_block = max(1, min(256, (256 * 1024) // (length * channels * 8)))
                for lo in range(0, count, score_block):
                    hi = min(count, lo + score_block)
                    target = truth[lo:hi, delay:delay+length].astype(np.float64)
                    delta = np.asarray(served[lo:hi], np.float64) - target
                    frozen_delta = base[lo+delay:hi+delay, :length].astype(np.float64) - target
                    losses = (np.square(delta).sum(axis=(1, 2)), np.abs(delta).sum(axis=(1, 2)),
                        np.square(frozen_delta).sum(axis=(1, 2)), np.abs(frozen_delta).sum(axis=(1, 2)))
                    segment_scores[lo:hi, 1:] = np.column_stack(losses)
                request_rows[name] += segment_scores
                # Schedules sharing an offset and span serve the same values.
                # Retain compact losses instead of duplicate forecast tensors.
                fixed_scores[length] = segment_scores
                del served
                continue
            weights = (np.fromiter((weight for _, _, weight in _event_exposure_weights(
                rows, delay, base, truth, proposal, cfg)), dtype=np.float64, count=len(rows))
                if delay else np.zeros(len(rows), dtype=np.float64))
            _event_scores(rows, weights, delay, base, truth, proposal if delay else None, request_rows[name])
        if delay:
            del proposal
    summaries = []
    for name, totals in request_rows.items():
        if np.any(totals[:, 0] != h*channels):
            raise AssertionError("Incomplete or duplicated H*C support")
        rows = [dict(request=i, atoms=int(row[0]), sse=float(row[1]), sae=float(row[2]),
                     base_sse=float(row[3]), base_sae=float(row[4])) for i, row in enumerate(totals)]
        request_rows[name] = rows
        summaries.append(dict(method="CommitCast-"+cfg.feature_mode, policy="current", schedule=name,
            support=support, **summarize_requests(rows)))
    return summaries, [dict(schedule=name, **row) for name,rows in request_rows.items() for row in rows]
