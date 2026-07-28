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
import time

import numpy as np

from layers.exposure import settled_risk_exposure
from layers.online_ridge import LeadChannelRidge
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
    device: str = "cpu"


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
        delay = max(1, min(horizon - 1, int(round(horizon * float(fraction)))))
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
    return pred, true


class _CommitFeatureBuilder:
    """Construct the frozen 17-dimensional legal state in bounded blocks."""

    def __init__(
        self,
        pred_t,
        true_t,
        *,
        delay: int,
        served_leads: int,
        traj_points: int,
        feature_clip: float,
    ):
        import torch

        self.torch = torch
        self.pred = pred_t
        self.true = true_t
        self.delay = int(delay)
        self.n_total, self.horizon, self.channels = map(int, pred_t.shape)
        self.n = self.n_total - self.delay
        self.remaining_full = self.horizon - self.delay
        self.served_leads = int(served_leads)
        self.feature_clip = float(feature_clip)
        self.offsets = _select_offsets(self.delay, int(traj_points))
        self.leads = torch.arange(self.served_leads, dtype=torch.long, device=pred_t.device)

        self.issue = pred_t[: self.n, self.delay : self.horizon, :]
        self.reforecast = pred_t[self.delay : self.n_total, : self.remaining_full, :]
        self.target = true_t[: self.n, self.delay : self.horizon, :]
        self.prefix_residual = (
            true_t[: self.n, : self.delay, :] - pred_t[: self.n, : self.delay, :]
        )

    def build(self, row_start: int, row_end: int, selected_lead: int | None = None):
        torch = self.torch
        if selected_lead is None:
            lead_start = 0
            active_leads = self.leads
        else:
            lead_start = int(selected_lead)
            active_leads = self.leads[lead_start : lead_start + 1]
        active_count = int(active_leads.numel())
        active_slice = slice(lead_start, lead_start + active_count)

        prefix = self.prefix_residual[row_start:row_end]
        issue_full = self.issue[row_start:row_end]
        reforecast_full = self.reforecast[row_start:row_end]
        issue = issue_full[:, active_slice, :]
        reforecast = reforecast_full[:, active_slice, :]

        scale = torch.sqrt(torch.mean(prefix * prefix, dim=1, keepdim=True)) + 1e-4
        denom = scale[:, 0, :]
        prefix_mean = prefix.mean(dim=1) / denom
        prefix_last = prefix[:, -1, :] / denom
        prefix_first = prefix[:, 0, :] / denom
        prefix_slope = prefix_last - prefix_first
        trajectory = []
        for offset in self.offsets:
            state_start = lead_start + self.delay - offset
            state = self.pred[
                row_start + offset : row_end + offset,
                state_start : state_start + active_count,
                :,
            ]
            trajectory.append(state)
        stacked = torch.stack(trajectory, dim=0)
        traj_mean = stacked.mean(dim=0)
        traj_var = torch.clamp((stacked * stacked).mean(dim=0) - traj_mean * traj_mean, min=0)
        traj_std = torch.sqrt(traj_var + 1e-8) / scale
        if len(trajectory) > 1:
            before_last = trajectory[-2]
        else:
            before_last = trajectory[0]
        first_state = trajectory[0]
        last_state = trajectory[-1]
        middle_state = trajectory[len(trajectory) // 2]
        last_step = (last_state - before_last) / scale
        curvature = (last_state - 2.0 * middle_state + first_state) / scale
        mean_gap = (traj_mean - last_state) / scale

        displacement_full = (issue_full - reforecast_full) / scale
        displacement = displacement_full[:, active_slice, :]
        displacement_mean = displacement_full.mean(dim=1)
        displacement_local = displacement - displacement_mean[:, None, :]
        issue_local = (issue - issue_full.mean(dim=1, keepdim=True)) / scale
        reforecast_mean = reforecast_full.mean(dim=1) / denom
        reforecast_slope = (
            reforecast_full[:, -1, :] - reforecast_full[:, 0, :]
        ) / denom
        def expand_scalar(value):
            return value[:, None, :, None].expand(-1, active_count, -1, -1)

        features = [
            expand_scalar(prefix_mean),
            expand_scalar(prefix_last),
            expand_scalar(prefix_slope),
            expand_scalar(reforecast_mean),
            expand_scalar(reforecast_slope),
            displacement[..., None],
            issue_local[..., None],
            displacement_local[..., None],
            traj_std[..., None],
            last_step[..., None],
            curvature[..., None],
            mean_gap[..., None],
        ]
        x = torch.cat(features, dim=-1)
        if x.shape[-1] != len(COMMITCAST_FEATURES):
            raise AssertionError(f"feature assembly produced {x.shape[-1]} dimensions")
        x.clamp_(min=-self.feature_clip, max=self.feature_clip)
        return x, scale


def revise_at_commit(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    delay: int,
    served_leads: int | None = None,
    cfg: CommitCastConfig | None = None,
) -> np.ndarray:
    """Emit causal same-commit residual revisions for one commitment delay.

    For output row ``i`` and relative lead ``r``, an older training row ``j`` is
    admitted only when ``j + r < i``. Thus its target matured strictly before
    the current commit. The current row's unserved labels are never read.
    """

    import torch

    cfg = cfg or CommitCastConfig()
    pred, true = _validate_stream(pred, true)
    n_total, horizon, channels = pred.shape
    delay = int(delay)
    if not 0 < delay < horizon or delay >= n_total:
        raise ValueError(f"invalid delay={delay} for stream shape={pred.shape}")
    remaining_full = horizon - delay
    remaining = remaining_full if served_leads is None else int(served_leads)
    if not 0 < remaining <= remaining_full:
        raise ValueError(f"served_leads must be in [1, {remaining_full}]")
    if cfg.block_size <= 0:
        raise ValueError("block_size must be positive")

    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    pred_t = torch.as_tensor(pred, dtype=torch.float32, device=device)
    true_t = torch.as_tensor(true, dtype=torch.float32, device=device)
    builder = _CommitFeatureBuilder(
        pred_t,
        true_t,
        delay=delay,
        served_leads=remaining,
        traj_points=cfg.traj_points,
        feature_clip=cfg.feature_clip,
    )

    n = n_total - delay
    fdim = len(COMMITCAST_FEATURES)
    ridge = LeadChannelRidge(
        leads=remaining,
        channels=channels,
        features=fdim,
        ridge_lambda=cfg.ridge_lambda,
        beta_clip=cfg.beta_clip,
        device=device,
    )
    output = torch.empty((n, remaining, channels), dtype=torch.float32, device=device)
    reference = builder.reforecast[:, :remaining, :]
    target = builder.target[:, :remaining, :]
    block = int(cfg.block_size)

    for start in range(0, n, block):
        end = min(n, start + block)
        for lead in range(remaining):
            matured_end = max(0, start - lead)
            previous = ridge.last_matured[lead]
            if matured_end <= previous:
                continue
            for train_start in range(previous, matured_end, block):
                train_end = min(matured_end, train_start + block)
                x, scale = builder.build(train_start, train_end, selected_lead=lead)
                x_lead = x[:, 0, :, :]
                residual = (
                    target[train_start:train_end, lead, :]
                    - reference[train_start:train_end, lead, :]
                ) / scale[:, 0, :]
                residual.clamp_(min=-cfg.target_clip, max=cfg.target_clip)
                ridge.update(lead, x_lead, residual)
            ridge.solve(lead)
            ridge.last_matured[lead] = matured_end

        x_now, scale_now = builder.build(start, end)
        raw = ridge.predict(x_now)
        active = ridge.active_mask(
            cfg.warmup_matured,
            dtype=raw.dtype,
            device=device,
        )
        raw *= active[None, :, None]
        correction = raw * scale_now
        limit = cfg.correction_clip * scale_now
        correction = torch.maximum(torch.minimum(correction, limit), -limit)
        output[start:end] = reference[start:end] + correction

    return output.detach().cpu().numpy().astype(np.float32, copy=False)


def run_commitcast(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    fractions: list[float],
    cfg: CommitCastConfig | None = None,
) -> list[CommitEvent]:
    """Run the non-overlapping multi-commit service lifecycle."""

    cfg = cfg or CommitCastConfig()
    pred, true = _validate_stream(pred, true)
    horizon = int(pred.shape[1])
    schedule = commit_schedule(horizon, fractions)
    if not schedule:
        raise ValueError("at least one commitment fraction is required")
    common_origins = int(pred.shape[0] - schedule[-1][1])
    if common_origins <= 0:
        raise ValueError("stream has no origins that complete the full commitment lifecycle")

    first_delay = int(schedule[0][1])
    initial = pred[:common_origins, :first_delay, :]
    events: list[CommitEvent] = [
        CommitEvent(
            label=f"issue_to_{schedule[0][0]}",
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
        print("CommitCast trainable parameters: 0 (closed-form sufficient statistics only)")
        return 0


def build_adapter(cfg, model, norm_module=None) -> CommitCastAdapter:
    if norm_module is not None:
        raise ValueError("CommitCast does not use a trainable normalization module")
    return CommitCastAdapter(cfg, model)
