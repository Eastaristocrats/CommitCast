"""Fixed and observed-input event schedules for the server CommitCast kernel."""
from __future__ import annotations

import heapq
import numpy as np

from fcr.scoring import SCHEDULES, boundaries, common_origins, summarize_requests
from layers.exposure import settled_risk_exposure
from tta.commitcast import CommitCastConfig, _RidgeSession


def event_exposures(rows, delay, base, truth, proposal, cfg):
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
        frozen = base[origin+delay, :length]
        prediction = frozen + alpha*(proposal[origin, :length]-frozen).astype(np.float64)
        yield origin, length, prediction, alpha
        heapq.heappush(queue, (origin+length, origin, length))


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
        fixed_cache = {}
        for name, rows in plans.items():
            if delay == 0:
                predictions = ((o,k,base[o,:k],0.) for o,k in rows)
            elif name == 'event50':
                predictions = event_exposures(rows, delay, base, truth, proposal, cfg)
            else:
                length = rows[0][1]
                if length not in fixed_cache:
                    if cfg.srs_enable:
                        fixed_cache[length] = settled_risk_exposure(truth[delay:delay+count,:length], base[delay:delay+count,:length],
                            proposal[:,:length], segment_len=length, warmup=cfg.srs_warmup_rows,
                            half_life_rows=cfg.srs_half_life_rows, ridge=cfg.srs_ridge)
                    else:
                        fixed_cache[length] = (proposal[:,:length], np.ones(len(proposal)))
                mixed, alpha = fixed_cache[length]
                predictions = ((o,k,mixed[o,:k],alpha[o]) for o,k in rows)
            if name != 'event50':
                length = rows[0][1]
                # Offline scoring can be batched independently of causal updates.
                # Bound temporary FP64 arrays without changing the update block.
                score_block = max(1, min(256, (256 * 1024) // (length * channels * 8)))
                served = base[:count, :length] if delay == 0 else mixed[:count, :length]
                for lo in range(0, count, score_block):
                    hi = min(count, lo + score_block)
                    target = truth[lo:hi, delay:delay+length].astype(np.float64)
                    delta = np.asarray(served[lo:hi], np.float64) - target
                    frozen_delta = base[lo+delay:hi+delay, :length].astype(np.float64) - target
                    losses = (np.square(delta).sum(axis=(1, 2)), np.abs(delta).sum(axis=(1, 2)),
                        np.square(frozen_delta).sum(axis=(1, 2)), np.abs(frozen_delta).sum(axis=(1, 2)))
                    request_rows[name][lo:hi, 0] += length * channels
                    request_rows[name][lo:hi, 1:] += np.column_stack(losses)
                continue
            for origin,length,prediction,_ in predictions:
                target = truth[origin, delay:delay+length].astype(np.float64)
                delta = np.asarray(prediction, np.float64) - target
                frozen_delta = base[origin+delay,:length].astype(np.float64) - target
                row = request_rows[name][origin]
                row[0] += length*channels
                row[1] += float(np.square(delta).sum())
                row[2] += float(np.abs(delta).sum())
                row[3] += float(np.square(frozen_delta).sum())
                row[4] += float(np.abs(frozen_delta).sum())
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
