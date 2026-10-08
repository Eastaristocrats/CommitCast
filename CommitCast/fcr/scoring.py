"""One score per (request, target, channel), with a matched same-commit Base."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

SCHEDULES = {
    "default": (0.25, 0.5, 0.75),
    "sparse": (0.5,),
    "uneven2": (0.375,),
    "uneven3": (0.375, 0.75),
    "missing": (0.25, 0.75),
    "uneven4": (0.125, 0.375, 0.75),
    "dense5": (0.2, 0.4, 0.6, 0.8),
    "dense8": (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875),
    "no_revision": (),
}


def boundaries(horizon, fractions):
    if type(horizon) is not int or horizon < 2:
        raise ValueError("horizon must be an integer >= 2")
    values = list(fractions)
    if any(not np.isfinite(x) or not 0 < x < 1 for x in values):
        raise ValueError("commit fractions must be finite and strictly between 0 and 1")
    return (0, *sorted({max(1, min(horizon - 1, int(horizon * x + 1e-10))) for x in values}), horizon)


def common_origins(origins, horizon, schedules):
    """Select origins only; every retained request still contains all H*C atoms."""
    if not schedules:
        raise ValueError("Declare at least one support schedule")
    last = max(boundaries(int(horizon), x)[-2] for x in schedules)
    count = int(origins) - last
    if count <= 0:
        raise ValueError("No complete-H requests remain on the declared common support")
    return count


def errors(target, prediction):
    target = np.asarray(target, np.float64)
    prediction = np.asarray(prediction, np.float64)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("Scores require identical, nonempty shapes; broadcasting is forbidden")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite values cannot be silently removed from matched support")
    delta = prediction - target
    return float(np.square(delta).sum()), float(np.abs(delta).sum()), int(delta.size)


def gain(base_error, method_error):
    """A zero-error Base has undefined relative gain, including the 0/0 case."""
    if not np.isfinite([base_error, method_error]).all() or min(base_error, method_error) < 0:
        raise ValueError("Error sums must be finite and nonnegative")
    return 100.0 * (1.0 - method_error / base_error) if base_error > 0 else float("nan")


def summarize_requests(rows):
    if not rows:
        raise ValueError("Cannot score empty request support")
    count = sum(int(r["atoms"]) for r in rows)
    sse = math.fsum(r["sse"] for r in rows)
    sae = math.fsum(r["sae"] for r in rows)
    bsse = math.fsum(r["base_sse"] for r in rows)
    bsae = math.fsum(r["base_sae"] for r in rows)
    mse = np.array([r["sse"] / r["atoms"] for r in rows])
    request_sse = np.fromiter((r["sse"] for r in rows), dtype=np.float64, count=len(rows))
    request_base = np.fromiter((r["base_sse"] for r in rows), dtype=np.float64, count=len(rows))
    if (not np.isfinite(request_sse).all() or not np.isfinite(request_base).all()
            or np.any(request_sse < 0) or np.any(request_base < 0)):
        raise ValueError("Error sums must be finite and nonnegative")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratios = 100.0 * (1.0 - request_sse / request_base)
    p90, p95 = np.quantile(mse, [.9, .95])
    return dict(requests=len(rows), evaluated_elements=count, sse=sse, sae=sae,
                base_sse=bsse, base_sae=bsae, mse=sse / count, mae=sae / count,
                base_mse=bsse / count, base_mae=bsae / count,
                mse_gain_pct=gain(bsse, sse), mae_gain_pct=gain(bsae, sae),
                request_mse_p90=float(p90),
                request_mse_p95=float(p95),
                request_mse_max=float(mse.max()),
                request_degradation_rate_pct=100 * float(np.mean(request_sse > request_base)),
                request_degradation_over_5pct=100 * float(np.mean(request_sse > 1.05*request_base)),
                zero_base_requests=int(np.sum(~np.isfinite(ratios))))


def score_plan(base, truth, fractions, count, prediction_at, *, request_boundaries=None):
    """prediction_at(request, delay, next_delay) returns this served segment."""
    base, truth = np.asarray(base), np.asarray(truth)
    if base.shape != truth.shape or base.ndim != 3:
        raise ValueError("base/truth must share [origin, horizon, channel]")
    n, h, channels = base.shape
    if not 0 < count <= n:
        raise ValueError("Invalid request count")
    static = boundaries(h, fractions)
    rows = []
    for request in range(count):
        cuts = static if request_boundaries is None else tuple(request_boundaries[request])
        if cuts[0] != 0 or cuts[-1] != h or any(b <= a for a, b in zip(cuts, cuts[1:])):
            raise ValueError("Each request must partition [0,H) exactly once")
        row = dict(request=request, atoms=0, sse=0., sae=0., base_sse=0., base_sae=0.)
        for a, b in zip(cuts, cuts[1:]):
            if request + a >= n:
                raise ValueError("Missing current forecast on declared support")
            target = truth[request, a:b]
            sse, sae, atoms = errors(target, prediction_at(request, a, b))
            bsse, bsae, _ = errors(target, base[request + a, :b-a])
            row["sse"] += sse
            row["sae"] += sae
            row["base_sse"] += bsse
            row["base_sae"] += bsae
            row["atoms"] += atoms
        if row["atoms"] != h * channels:
            raise AssertionError("Incomplete-H or duplicated target support")
        rows.append(row)
    return summarize_requests(rows), rows


def score_stream(base, truth, current, fractions, count, *, ledger=None, request_boundaries=None):
    """Score recorded current forecasts and optional request-local revisions.

    This offline path batches independent error reductions. It does not batch
    model queries or change publication deadlines, update clocks or support.
    The generic callback interface remains available as ``score_plan``.
    """
    base, truth, current = np.asarray(base), np.asarray(truth), np.asarray(current)
    if base.ndim != 3 or base.shape != truth.shape or base.shape != current.shape:
        raise ValueError("Forecasts and targets must share [origin, horizon, channel]")
    n, h, channels = base.shape
    if not 0 < count <= n or channels < 1:
        raise ValueError("Invalid request count or channel dimension")
    static = boundaries(h, fractions)
    groups = {}
    if request_boundaries is None:
        for a, b in zip(static, static[1:]):
            if count - 1 + a >= n:
                raise ValueError("Missing current forecast on declared support")
            groups[a, b] = None
    else:
        if len(request_boundaries) != count:
            raise ValueError("Expected one boundary plan per request")
        for request, cuts in enumerate(request_boundaries):
            cuts = tuple(cuts)
            if not cuts or cuts[0] != 0 or cuts[-1] != h or any(b <= a for a, b in zip(cuts, cuts[1:])):
                raise ValueError("Each request must partition [0,H) exactly once")
            for a, b in zip(cuts, cuts[1:]):
                if request + a >= n:
                    raise ValueError("Missing current forecast on declared support")
                groups.setdefault((a, b), []).append(request)
    totals = np.zeros((count, 5), dtype=np.float64)
    covered_atoms = 0
    for (a, b), origins in sorted(groups.items()):
        length = b - a
        block = max(1, min(256, (256 * 1024) // (length * channels * 8)))
        size = count if origins is None else len(origins)
        for lo in range(0, size, block):
            hi = min(size, lo + block)
            index = slice(lo, hi) if origins is None else np.asarray(origins[lo:hi])
            shifted = slice(lo + a, hi + a) if origins is None else index + a
            target = np.asarray(truth[index, a:b], dtype=np.float64)
            prediction = current[shifted, :length]
            if ledger is not None:
                prediction = prediction.copy()
                requests = range(lo, hi) if origins is None else index
                for j, request in enumerate(requests):
                    selected = ledger._latest(int(request), int(request) + a, a, b, channels)
                    if selected is not None:
                        prediction[j] = selected
                        covered_atoms += length * channels
            frozen = base[shifted, :length]
            if not all(np.isfinite(value).all() for value in (target, prediction, frozen)):
                raise ValueError("Nonfinite values cannot be silently removed from matched support")
            delta = np.asarray(prediction, np.float64) - target
            frozen_delta = np.asarray(frozen, np.float64) - target
            losses = (np.square(delta).sum(axis=(1, 2)), np.abs(delta).sum(axis=(1, 2)),
                      np.square(frozen_delta).sum(axis=(1, 2)), np.abs(frozen_delta).sum(axis=(1, 2)))
            totals[index, 0] += length * channels
            totals[index, 1:] += np.column_stack(losses)
    if np.any(totals[:, 0] != h * channels):
        raise AssertionError("Incomplete-H or duplicated target support")
    rows = [dict(request=i, atoms=int(row[0]), sse=float(row[1]), sae=float(row[2]),
                 base_sse=float(row[3]), base_sae=float(row[4])) for i, row in enumerate(totals)]
    return summarize_requests(rows), rows, covered_atoms


def validate_event_timeline(values, truth, context_length):
    values, truth = np.asarray(values), np.asarray(truth)
    if int(context_length) != context_length or context_length < 0:
        raise ValueError("Invalid event context length")
    context_length = int(context_length)
    n,h,channels = truth.shape
    if values.ndim != 2 or values.shape[1] != channels or not np.isfinite(values).all():
        raise ValueError("Invalid event input timeline")
    timeline = np.concatenate([truth[:,0],truth[-1,1:]],axis=0)
    if not np.array_equal(values[context_length:context_length+n+h-1],timeline):
        raise ValueError("Event inputs do not match the forecast stream's target timeline")


def input_scores(observed_timeline, count, *, context_length=0, history=24):
    values = np.asarray(observed_timeline)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite chronological [time, channel] array")
    if context_length < 0 or history < 1 or count < 1:
        raise ValueError("Invalid context/history/count")
    if context_length + count - 1 > len(values):
        raise ValueError("Observed timeline does not cover all decisions")
    scores = np.full(count, np.nan)
    first = max(0, history + 1 - context_length)
    if first >= count:
        return scores
    windows = np.lib.stride_tricks.sliding_window_view(values, history, axis=0)
    block = max(1, min(1024, (256 * 1024) // max(1, values.shape[1] * values.dtype.itemsize)))
    for lo in range(first, count, block):
        hi = min(count, lo + block)
        start = context_length + lo - 1 - history
        # Every row ends at t-2; only observation t-1 is compared with that
        # history. The reduction order and input dtype match the scalar rule.
        preceding = np.moveaxis(windows[start:start+hi-lo], -1, 1).mean(axis=1)
        observed = values[context_length+lo-1:context_length+hi-1]
        scores[lo:hi] = np.sqrt(np.mean((observed - preceding)**2, axis=1))
    return scores


def event_boundaries(observed_timeline, horizon, count, threshold, history=24, *, context_length=0):
    """At q, use only observation q-1 and its preceding 24 observed values.

    The threshold must have been fixed using validation data. This function
    never estimates a threshold from the test stream or examines future labels.
    """
    if not np.isfinite(threshold) or threshold < 0 or history < 1:
        raise ValueError("Invalid validation-frozen threshold/history")
    candidates = boundaries(horizon, SCHEDULES["dense8"])[1:-1]
    scores = input_scores(observed_timeline, count + max(candidates), context_length=context_length, history=history)
    plans = []
    for origin in range(count):
        selected = [0]
        for delay in candidates:
            q = origin + delay
            if scores[q] > threshold:
                selected.append(delay)
        plans.append((*selected, horizon))
    return plans


@dataclass(frozen=True)
class Publication:
    origin: int
    published_at: int
    values: np.ndarray


class PublicationLedger:
    """Keep request-local revisions; never backdate a batch to its row origins."""
    def __init__(self):
        self.records = {}

    def publish(self, origin, published_at, values):
        data = np.array(values, copy=True)
        if data.ndim != 2 or not np.isfinite(data).all() or published_at < origin:
            raise ValueError("Invalid publication or unavailable input context")
        data.flags.writeable = False
        version = Publication(int(origin), int(published_at), data)
        row = self.records.setdefault(int(origin), [])
        if row and row[-1].published_at > published_at:
            raise ValueError("Publication time cannot move backwards")
        row.append(version)

    def select(self, request, deadline, delay, next_delay, fallback):
        out = np.array(fallback, copy=True)
        if deadline != request + delay or out.shape[0] != next_delay - delay:
            raise ValueError("Invalid commitment identity")
        selected = self._latest(request, deadline, delay, next_delay, out.shape[1])
        if selected is not None:
            out[:] = selected
        return out, np.full(out.shape, selected is not None, dtype=bool)

    def _latest(self, request, deadline, delay, next_delay, channels):
        selected = None
        for version in self.records.get(request, ()):
            if version.published_at > deadline:
                break
            if version.values.shape[0] < next_delay or version.values.shape[1] != channels:
                raise ValueError("Native revision does not cover the requested targets")
            selected = version.values[delay:next_delay]
        return selected
