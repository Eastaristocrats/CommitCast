"""Core CommitCast adapter.

The frozen forecaster is never updated. At each commitment, CommitCast:

1. aligns the current frozen reforecast with the still-unserved targets;
2. builds commit-visible prefix, forecast-geometry, and same-target vintage
   features;
3. estimates the residual left by the frozen reforecast with lead/channel-wise
   online ridge sufficient statistics;
4. scales the proposal using only fully settled service losses.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
import time

import numpy as np

from layers.exposure import settled_risk_exposure
from models.forecast import ForecastStream, FrozenForecastModel
from utils.misc import parse_csv, resolve_device


COMMITCAST_FEATURES = (
    "prefix_mean",
    "prefix_last",
    "prefix_slope",
    "reforecast_mean",
    "reforecast_slope",
    "issue_reforecast_displacement",
    "issue_local_deviation",
    "displacement_local_deviation",
    "vintage_std",
    "vintage_last_step",
    "vintage_curvature",
    "vintage_mean_gap",
)


@dataclass(frozen=True)
class CommitCastConfig:
    traj_points: int = 17
    block_size: int = 128
    ridge_lambda: float = 25.0
    warmup_matured: int = 16
    feature_clip: float = 6.0
    target_clip: float = 6.0
    beta_clip: float = 0.4
    correction_clip: float = 0.5
    srs_enable: bool = True
    srs_half_life_rows: float = 512.0
    srs_warmup_rows: int = 16
    srs_ridge: float = 1e-6
    feature_mode: str = "full"
    device: str = "cpu"

    def __post_init__(self):
        if self.feature_mode not in {"full", "pg", "p", "g"}:
            raise ValueError("Invalid feature mode")
        for key in ("traj_points", "block_size", "warmup_matured", "srs_warmup_rows"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{key} must be an integer")
        if self.block_size < 1 or min(self.warmup_matured, self.srs_warmup_rows) < 0:
            raise ValueError("Block size must be positive and warm-up counts nonnegative")
        for key in ("ridge_lambda", "feature_clip", "target_clip", "beta_clip", "correction_clip", "srs_ridge"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0 or (key == "ridge_lambda" and value == 0):
                raise ValueError(f"Invalid nonnegative finite parameter: {key}")
        if math.isnan(self.srs_half_life_rows) or self.srs_half_life_rows <= 0:
            raise ValueError("Exposure half-life must be positive or positive infinity")


@dataclass(frozen=True)
class CommitEvent:
    label: str
    delay: int
    next_delay: int
    target: np.ndarray
    checkpoint: np.ndarray
    proposal: np.ndarray
    prediction: np.ndarray
    exposure: np.ndarray

    @property
    def segment_len(self) -> int:
        return self.next_delay - self.delay


@dataclass(frozen=True)
class AdaptationResult:
    stream: ForecastStream
    events: tuple[CommitEvent, ...]
    runtime_sec: float


def build_method_config(cfg) -> CommitCastConfig:
    method = cfg.TTA.COMMITCAST
    return CommitCastConfig(
        feature_mode=str(method.FEATURE_MODE),
        traj_points=int(method.TRAJ_POINTS),
        block_size=int(method.BLOCK_SIZE),
        ridge_lambda=float(method.RIDGE_LAMBDA),
        warmup_matured=int(method.WARMUP_MATURED),
        feature_clip=float(method.FEATURE_CLIP),
        target_clip=float(method.TARGET_CLIP),
        beta_clip=float(method.BETA_CLIP),
        correction_clip=float(method.CORRECTION_CLIP),
        srs_enable=bool(method.SRS_ENABLE),
        srs_half_life_rows=float(method.SRS_HALF_LIFE_ROWS),
        srs_warmup_rows=int(method.SRS_WARMUP_ROWS),
        srs_ridge=float(method.SRS_RIDGE),
        device=resolve_device(str(cfg.DEVICE)),
    )


def commit_schedule(horizon: int, fractions: list[float]) -> list[tuple[str, int]]:
    """Create unique, sorted commitment delays from horizon fractions."""

    if horizon < 2:
        raise ValueError("horizon must be at least 2")
    schedule: list[tuple[str, int]] = []
    seen: set[int] = set()
    for fraction in fractions:
        if not 0.0 < float(fraction) < 1.0:
            raise ValueError(f"commit fractions must be in (0, 1), got {fraction}")
        delay = max(1, min(horizon - 1, int(horizon * float(fraction) + 1e-10)))
        if delay not in seen:
            schedule.append((f"c{int(round(100 * float(fraction))):02d}", delay))
            seen.add(delay)
    return sorted(schedule, key=lambda item: item[1])


def _select_offsets(delay: int, points: int) -> list[int]:
    if points <= 0 or points >= delay + 1:
        return list(range(delay + 1))
    raw = np.linspace(0, delay, num=max(2, int(points)))
    return sorted({int(round(value)) for value in raw} | {0, int(delay)})


def _validate_stream(pred: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    if pred.shape != true.shape or pred.ndim != 3:
        raise ValueError("pred and true must share shape [origin, lead, channel]")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("pred and true must be finite")
    if not np.array_equal(true[:-1, 1:], true[1:, :-1]):
        raise ValueError("Targets must overlap chronologically at stride 1")
    return pred, true


SERVER_FEATURES = (
    "mean", "last", "slope", "reforecast_mean", "reforecast_slope", "disp",
    "issue_local_deviation", "disp_local_deviation", "traj_std", "last_step",
    "traj_curvature", "traj_mean_gap",
)


def revise_at_commit(pred, true, *, delay, served_leads=None, cfg=None):
    """Call the reference numerical kernel, preserving its original numerical pathway."""
    from layers.online_ridge import run_fsd_ridge
    cfg = cfg or CommitCastConfig()
    pred, true = _validate_stream(pred, true)
    if cfg.feature_mode not in {"full", "pg", "p", "g"}:
        raise ValueError("Invalid feature mode")
    if not 0 < delay < pred.shape[1] or delay >= pred.shape[0]:
        raise ValueError("Invalid commit delay")
    if cfg.block_size <= 0 or cfg.ridge_lambda <= 0:
        raise ValueError("Block size and ridge penalty must be positive")
    selected = {"full": slice(None), "pg": slice(0,8), "p": slice(0,3), "g": slice(3,8)}[cfg.feature_mode]
    return run_fsd_ridge(pred=pred, true=true, delay=int(delay),
        feature_names=list(SERVER_FEATURES[selected]), traj_points=cfg.traj_points,
        ridge_lambda=cfg.ridge_lambda, beta_clip=cfg.beta_clip,
        correction_clip=cfg.correction_clip, block_size=cfg.block_size,
        warmup_matured=cfg.warmup_matured, feature_clip=cfg.feature_clip,
        target_clip=cfg.target_clip, device=cfg.device, maturity="leadwise",
        feature_cache="online", leadwise_solve_mode="batched",
        leadwise_update_mode="batched", target_cache=False,
        served_leads=served_leads, feedback_delay=0)


def run_commitcast(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    fractions: list[float],
    support_fractions: list[float] | None = None,
    cfg: CommitCastConfig | None = None,
) -> list[CommitEvent]:
    """Run the non-overlapping multi-commit service lifecycle."""

    cfg = cfg or CommitCastConfig()
    pred, true = _validate_stream(pred, true)
    horizon = int(pred.shape[1])
    schedule = commit_schedule(horizon, fractions)
    support = commit_schedule(horizon, fractions if support_fractions is None else support_fractions)
    latest = max((d for _, d in support), default=0)
    if latest < max((d for _, d in schedule), default=0):
        raise ValueError("scoring support does not cover the requested schedule")
    common_origins = int(pred.shape[0] - latest)
    if common_origins <= 0:
        raise ValueError("stream has no origins that complete the full commitment lifecycle")

    first_delay = int(schedule[0][1]) if schedule else horizon
    initial = pred[:common_origins, :first_delay, :]
    events: list[CommitEvent] = [
        CommitEvent(
            label=f"issue_to_{schedule[0][0]}" if schedule else "issue_to_end",
            delay=0,
            next_delay=first_delay,
            target=true[:common_origins, :first_delay, :],
            checkpoint=initial,
            proposal=initial,
            prediction=initial,
            exposure=np.zeros(common_origins, dtype=np.float64),
        )
    ]
    for index, (label, delay) in enumerate(schedule):
        next_delay = schedule[index + 1][1] if index + 1 < len(schedule) else horizon
        segment_len = next_delay - delay
        n = pred.shape[0] - delay
        checkpoint_full = pred[delay:, :segment_len, :]
        target_full = true[:n, delay : delay + segment_len, :]
        proposal = revise_at_commit(
            pred,
            true,
            delay=delay,
            served_leads=segment_len,
            cfg=cfg,
        )
        if cfg.srs_enable:
            prediction, exposure = settled_risk_exposure(
                target_full,
                checkpoint_full,
                proposal,
                segment_len=segment_len,
                warmup=cfg.srs_warmup_rows,
                half_life_rows=cfg.srs_half_life_rows,
                ridge=cfg.srs_ridge,
            )
        else:
            prediction = proposal
            exposure = np.ones(n, dtype=np.float64)
        events.append(
            CommitEvent(
                label=label,
                delay=delay,
                next_delay=next_delay,
                target=target_full[:common_origins],
                checkpoint=checkpoint_full[:common_origins],
                proposal=proposal[:common_origins],
                prediction=prediction[:common_origins],
                exposure=exposure[:common_origins],
            )
        )
    return events


class CommitCastAdapter:
    """COSA-style adapter facade over the core online revision routine."""

    def __init__(self, cfg, model: FrozenForecastModel):
        self.cfg = cfg
        self.model = model
        self.method_cfg = build_method_config(cfg)
        self.fractions = parse_csv(cfg.TTA.COMMITCAST.FRACTIONS, float)
        self.support_fractions = parse_csv(cfg.EVALUATION.SUPPORT_FRACTIONS, float)
        self.results: tuple[AdaptationResult, ...] = ()

    def adapt(self) -> tuple[AdaptationResult, ...]:
        completed: list[AdaptationResult] = []
        total = len(self.model.records)
        for index, stream in enumerate(self.model.streams(), 1):
            print(f"[{index}/{total}] adapting {stream.record.stream}")
            started = time.perf_counter()
            events = run_commitcast(
                stream.pred,
                stream.true,
                fractions=self.fractions,
                support_fractions=self.support_fractions,
                cfg=self.method_cfg,
            )
            completed.append(
                AdaptationResult(
                    stream=stream,
                    events=tuple(events),
                    runtime_sec=time.perf_counter() - started,
                )
            )
        self.results = tuple(completed)
        return self.results

    def count_parameters(self) -> int:
        print("CommitCast gradient-trained parameters: 0; online ridge coefficients and exposure state are fitted")
        return 0


def build_adapter(cfg, model, norm_module=None) -> CommitCastAdapter:
    if str(cfg.TTA.NAME).strip().lower() != "commitcast":
        raise ValueError(f"unsupported TTA.NAME for CommitCast adapter: {cfg.TTA.NAME!r}")
    if norm_module is not None:
        raise ValueError("CommitCast does not use a trainable normalization module")
    return CommitCastAdapter(cfg, model)
