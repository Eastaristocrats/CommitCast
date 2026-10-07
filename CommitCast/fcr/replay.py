"""Fixed and observed-input event schedules for the server CommitCast kernel."""
from __future__ import annotations

import heapq
import numpy as np

from fcr.scoring import SCHEDULES, boundaries, common_origins, summarize_requests
from layers.exposure import settled_risk_exposure
from tta.commitcast import CommitCastConfig, revise_at_commit


def event_exposures(rows, delay, base, truth, proposal, cfg):
    """Only actually served, completely settled rows enter this offset's gate.

    Queue entries contain observed stopping times. No forecast feature or
    exposure query reads a current row's future stopping time.
    """
    queue = []
    numerator = denominator = 0.0
    matured = 0
    decay = np.exp(np.log(.5) / cfg.srs_half_life_rows)
    for origin, length in rows:
        while queue and queue[0][0] <= origin:
            _, previous, span = heapq.heappop(queue)
            frozen = base[previous+delay, :span].astype(np.float64)
            error = truth[previous+delay, :span].astype(np.float64) - frozen
            delta = proposal[previous, :span].astype(np.float64) - frozen
            numerator = decay*numerator + float(np.dot(error.ravel(), delta.ravel()))
            denominator = decay*denominator + max(float(np.dot(delta.ravel(), delta.ravel())), 0.)
            matured += 1
        alpha = (float(np.clip(numerator/(denominator+cfg.srs_ridge), 0, 1))
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
        cuts = [boundaries(h, SCHEDULES[name])]*count if name != "event50" else event_plans
        if cuts is None or len(cuts) != count:
            raise ValueError("event50 requires validation-frozen, context-aware event plans")
        request_rows[name] = [dict(request=i, atoms=0, sse=0., sae=0., base_sse=0., base_sae=0.) for i in range(count)]
        for origin, plan in enumerate(cuts):
            if plan[0] != 0 or plan[-1] != h or any(b <= a for a,b in zip(plan, plan[1:])):
                raise ValueError("Every request must partition [0,H)")
            for start, end in zip(plan, plan[1:]):
                by_delay.setdefault(start, {}).setdefault(name, []).append((origin, end-start))
    for delay, plans in sorted(by_delay.items()):
        print(f"CommitCast {cfg.feature_mode}: offset {delay}/{h}", flush=True)
        if delay:
            span = h-delay if 'event50' in plans else max(length for rows in plans.values() for _,length in rows)
            proposal = revise_at_commit(base, truth, delay=delay, served_leads=span, cfg=cfg)
        fixed_cache = {}
        for name, rows in plans.items():
            if delay == 0:
                predictions = ((o,k,base[o,:k],0.) for o,k in rows)
            elif name == 'event50':
                predictions = event_exposures(rows, delay, base, truth, proposal, cfg)
            else:
                length = rows[0][1]
                if any(k != length for _,k in rows):
                    raise AssertionError("Fixed schedule length changed")
                if length not in fixed_cache:
                    if cfg.srs_enable:
                        fixed_cache[length] = settled_risk_exposure(truth[delay:,:length], base[delay:,:length],
                            proposal[:,:length], segment_len=length, warmup=cfg.srs_warmup_rows,
                            half_life_rows=cfg.srs_half_life_rows, ridge=cfg.srs_ridge)
                    else:
                        fixed_cache[length] = (proposal[:,:length], np.ones(len(proposal)))
                mixed, alpha = fixed_cache[length]
                predictions = ((o,k,mixed[o,:k],alpha[o]) for o,k in rows)
            for origin,length,prediction,_ in predictions:
                target = truth[origin, delay:delay+length].astype(np.float64)
                delta = np.asarray(prediction, np.float64) - target
                frozen_delta = base[origin+delay,:length].astype(np.float64) - target
                row = request_rows[name][origin]
                row['atoms'] += length*channels
                row['sse'] += float(np.square(delta).sum())
                row['sae'] += float(np.abs(delta).sum())
                row['base_sse'] += float(np.square(frozen_delta).sum())
                row['base_sae'] += float(np.abs(frozen_delta).sum())
    summaries = []
    for name, rows in request_rows.items():
        if any(row['atoms'] != h*channels for row in rows):
            raise AssertionError("Incomplete or duplicated H*C support")
        summaries.append(dict(method="CommitCast-"+cfg.feature_mode, policy="current", schedule=name,
            support=support, **summarize_requests(rows)))
    return summaries, [dict(schedule=name, **row) for name,rows in request_rows.items() for row in rows]
