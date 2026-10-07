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
    ratios = np.array([gain(r["base_sse"], r["sse"]) for r in rows])
    return dict(requests=len(rows), evaluated_elements=count, sse=sse, sae=sae,
                base_sse=bsse, base_sae=bsae, mse=sse / count, mae=sae / count,
                base_mse=bsse / count, base_mae=bsae / count,
                mse_gain_pct=gain(bsse, sse), mae_gain_pct=gain(bsae, sae),
                request_mse_p90=float(np.quantile(mse, .9)),
                request_mse_p95=float(np.quantile(mse, .95)),
                request_mse_max=float(mse.max()),
                request_degradation_rate_pct=100 * float(np.mean([r['sse'] > r['base_sse'] for r in rows])),
                request_degradation_over_5pct=100 * float(np.mean([r['sse'] > 1.05*r['base_sse'] for r in rows])),
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
    for q in range(count):
        t = context_length + q
        if t > history:
            scores[q] = np.sqrt(np.mean((values[t-1] - values[t-1-history:t-1].mean(axis=0))**2))
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
        covered = np.zeros(out.shape, bool)
        for version in self.records.get(request, ()):
            if version.published_at > deadline:
                break
            if version.values.shape[0] < next_delay or version.values.shape[1] != out.shape[1]:
                raise ValueError("Native revision does not cover the requested targets")
            out[:] = version.values[delay:next_delay]
            covered[:] = True
        return out, covered
