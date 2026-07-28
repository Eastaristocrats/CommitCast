#!/usr/bin/env python
"""Strict full-trajectory-maturity COSA-FCR on frozen forecast streams.

The source streams contain one H-step forecast per consecutive issue origin.
For a forecast issued at origin j, its last target is observable immediately
before origin j + H is served.  Therefore the complete row may enter COSA's
training queue at issue i iff j + H <= i.

Evaluation remains COSA's original complete [N,H,V] forecast surface.  The only
protocol change is the information boundary governing adaptation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


SimpleOutputAdapter: Any | None = None


RESULT_FIELDS = [
    "stream",
    "path",
    "dataset",
    "backbone",
    "seed",
    "horizon",
    "n",
    "channels",
    "forecast_scope",
    "method",
    "metric_protocol",
    "target_visibility",
    "service_origins",
    "segment_len",
    "evaluated_elements",
    "base_mse",
    "mse",
    "base_mae",
    "mae",
    "gain_vs_base_pct",
    "mae_gain_vs_base_pct",
    "origin_gain_mean_pct",
    "origin_gain_q05_pct",
    "origin_gain_cvar10_pct",
    "origin_gain_worst_pct",
    "origin_win_rate_pct",
    "updates",
    "update_batches",
    "matured_rows_consumed",
    "pending_mature_rows",
    "first_update_issue",
    "last_update_issue",
    "max_train_target_time_minus_issue",
    "max_context_target_time_minus_issue",
    "prequential_violation_count",
    "context_violation_count",
    "metric_evaluation_after_stream_frozen",
    "adapter_trainable_parameters",
    "adapter_runtime_sec",
    "incremental_cuda_peak_mb",
    "protocol",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        required=True,
        type=Path,
        help="Pinned upstream COSA checkout; its original license remains in force.",
    )
    parser.add_argument("--stream_root", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--update_batch_size", type=int, default=48)
    parser.add_argument("--serve_batch_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--context", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_streams", type=int, default=0)
    parser.add_argument("--max_origins", type=int, default=0)
    parser.add_argument("--include_regex", default="")
    parser.add_argument(
        "--metric_protocol",
        choices=("full_surface", "latest_commit_four_interval"),
        default="full_surface",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_upstream_adapter(repo: Path) -> dict[str, str]:
    """Load COSA from an external checkout without redistributing its source."""
    import importlib
    import sys

    global SimpleOutputAdapter
    repo = repo.resolve()
    source = repo / "tta" / "cosa.py"
    license_path = repo / "LICENSE"
    if not source.is_file():
        raise FileNotFoundError(f"missing upstream COSA module: {source}")
    if not license_path.is_file():
        raise FileNotFoundError(f"missing upstream COSA license: {license_path}")
    sys.path.insert(0, str(repo))
    module = importlib.import_module("tta.cosa")
    module_path = Path(module.__file__).resolve()
    if repo not in module_path.parents:
        raise ImportError(f"loaded COSA from unexpected location: {module_path}")
    SimpleOutputAdapter = module.SimpleOutputAdapter
    return {
        "module": "tta/cosa.py",
        "module_sha256": sha256_file(source),
        "license": "LICENSE",
        "license_sha256": sha256_file(license_path),
    }


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_existing(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def stream_metadata(path: Path, metadata_json: np.ndarray | str | None) -> dict[str, Any]:
    if metadata_json is not None:
        raw = metadata_json.item() if isinstance(metadata_json, np.ndarray) else metadata_json
        meta = json.loads(str(raw))
        seed_match = re.search(r"_seed(\d+)(?:_|$)", path.parent.name)
        return {
            "stream": path.parent.name,
            "path": f"{path.parent.name}/adapter_stream.npz",
            "dataset": str(meta["dataset"]),
            "backbone": str(meta["model"]),
            "seed": int(seed_match.group(1)) if seed_match else 0,
            "horizon": int(meta["horizon"]),
            "forecast_scope": str(meta.get("forecast_scope", "unknown")),
        }

    # Frozen checkpoint capture streams use names such as
    # BASE_exchange_rate_iTransformer_h336_seed0_batch48 and intentionally
    # carry no metadata_json member.
    match = re.fullmatch(
        r"BASE_(ETTh1|ETTh2|ETTm1|ETTm2|exchange_rate|weather|traffic)_(.+)_h(\d+)_seed(\d+)_batch\d+",
        path.parent.name,
    )
    if match is None:
        raise ValueError(f"Cannot parse stream metadata from {path.parent.name}")
    dataset, backbone, horizon, seed = match.groups()
    return {
        "stream": path.parent.name,
        "path": f"{path.parent.name}/adapter_stream.npz",
        "dataset": dataset,
        "backbone": backbone,
        "seed": int(seed),
        "horizon": int(horizon),
        "forecast_scope": "all_variables",
    }


def context_vector(history: deque[float], size: int) -> np.ndarray:
    if not history:
        return np.zeros(size, dtype=np.float32)
    values = list(reversed(history))[:size]
    values.extend([values[-1]] * (size - len(values)))
    return np.asarray(values, dtype=np.float32)


def build_adapter(
    horizon: int,
    channels: int,
    context: int,
    device: torch.device,
    seed: int,
) -> torch.nn.Module:
    if SimpleOutputAdapter is None:
        raise RuntimeError("upstream COSA adapter has not been loaded")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return SimpleOutputAdapter(
        pred_len=horizon,
        buffer_context_size=context,
        n_vars=channels,
        var_wise_gating=False,
        num_layers=1,
        hidden_dim=64,
    ).to(device)


def strict_serve(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    device: torch.device,
    context_size: int,
    update_batch_size: int,
    serve_batch_size: int,
    steps: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Serve forecasts while exposing labels only after their full H-step row matures."""
    if pred.shape != true.shape or pred.ndim != 3:
        raise ValueError(f"Expected matching [N,H,V] arrays, got {pred.shape} and {true.shape}")
    n, horizon, channels = pred.shape
    if update_batch_size <= 0 or serve_batch_size <= 0 or context_size <= 0:
        raise ValueError("Batch and context sizes must be positive")

    served = np.empty_like(pred, dtype=np.float32)
    issue_context = np.empty((n, context_size), dtype=np.float32)
    context_target_max = np.full(n, -1, dtype=np.int64)

    module = build_adapter(horizon, channels, context_size, device, seed)
    optimizer = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    trainable_parameters = sum(p.numel() for p in module.parameters() if p.requires_grad)

    matured_history: deque[float] = deque(maxlen=context_size)
    matured_history_rows: deque[int] = deque(maxlen=context_size)
    training_queue: deque[int] = deque()
    pending_service: list[int] = []
    pending_context: list[np.ndarray] = []

    update_batches = 0
    optimizer_steps = 0
    matured_rows_consumed = 0
    first_update_issue: int | None = None
    last_update_issue: int | None = None
    prequential_violations = 0
    context_violations = 0
    max_train_margin = -10**9
    max_context_margin = -10**9

    def flush_service() -> None:
        if not pending_service:
            return
        module.eval()
        ids = np.asarray(pending_service, dtype=np.int64)
        yp = torch.from_numpy(np.ascontiguousarray(pred[ids])).to(device)
        ctx = torch.from_numpy(np.stack(pending_context)).to(device)
        with torch.no_grad():
            output = module(yp, ctx).detach().cpu().numpy()
        served[ids] = output
        pending_service.clear()
        pending_context.clear()

    def train_matured(issue: int, ids: list[int]) -> None:
        nonlocal update_batches, optimizer_steps, matured_rows_consumed
        nonlocal first_update_issue, last_update_issue, prequential_violations, max_train_margin
        latest_target_time = max(ids) + horizon - 1
        margin = latest_target_time - issue
        max_train_margin = max(max_train_margin, margin)
        if margin >= 0 or any(row + horizon > issue for row in ids):
            prequential_violations += 1
            raise AssertionError(
                f"Future-label violation at issue {issue}: rows={ids[0]}..{ids[-1]}, "
                f"latest target time={latest_target_time}"
            )
        yp = torch.from_numpy(np.ascontiguousarray(pred[ids])).to(device)
        yt = torch.from_numpy(np.ascontiguousarray(true[ids])).to(device)
        ctx = torch.from_numpy(np.ascontiguousarray(issue_context[ids])).to(device)
        module.train()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            output = module(yp, ctx)
            l2 = sum(p.square().sum() for p in module.parameters() if p.requires_grad)
            loss = F.mse_loss(output, yt) + weight_decay * l2
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite COSA loss at issue {issue}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 0.1)
            optimizer.step()
            optimizer_steps += 1
        update_batches += 1
        matured_rows_consumed += len(ids)
        first_update_issue = issue if first_update_issue is None else first_update_issue
        last_update_issue = issue

    for issue in range(n):
        newly_matured = issue - horizon
        if newly_matured >= 0:
            # Its final target is at absolute time issue-1, so the whole row is now legal.
            matured_history.append(float(np.mean(true[newly_matured], dtype=np.float64)))
            matured_history_rows.append(newly_matured)
            training_queue.append(newly_matured)

        current_context = context_vector(matured_history, context_size)
        issue_context[issue] = current_context
        if matured_history_rows:
            latest_context_target_time = max(matured_history_rows) + horizon - 1
            context_target_max[issue] = latest_context_target_time
            margin = latest_context_target_time - issue
            max_context_margin = max(max_context_margin, margin)
            if margin >= 0:
                context_violations += 1
                raise AssertionError(
                    f"Future context at issue {issue}: latest target time={latest_context_target_time}"
                )

        # A model update changes the adapter version.  Predictions accumulated under
        # the previous version must be frozen before the update occurs.
        while len(training_queue) >= update_batch_size:
            flush_service()
            train_ids = [training_queue.popleft() for _ in range(update_batch_size)]
            train_matured(issue, train_ids)

        pending_service.append(issue)
        pending_context.append(current_context.copy())
        if len(pending_service) >= serve_batch_size:
            flush_service()

    flush_service()
    stats = {
        "updates": optimizer_steps,
        "update_batches": update_batches,
        "matured_rows_consumed": matured_rows_consumed,
        "pending_mature_rows": len(training_queue),
        "first_update_issue": first_update_issue if first_update_issue is not None else -1,
        "last_update_issue": last_update_issue if last_update_issue is not None else -1,
        "max_train_target_time_minus_issue": max_train_margin if update_batches else -1,
        "max_context_target_time_minus_issue": max_context_margin if n > horizon else -1,
        "prequential_violation_count": prequential_violations,
        "context_violation_count": context_violations,
        "adapter_trainable_parameters": trainable_parameters,
    }
    return served, stats


def full_surface_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    served: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate COSA's complete [N,H,V] surface after predictions freeze."""
    n, horizon, channels = pred.shape
    base_origin_sse = np.zeros(n, dtype=np.float64)
    adapted_origin_sse = np.zeros(n, dtype=np.float64)
    total_base_sse = total_adapted_sse = 0.0
    total_base_sae = total_adapted_sae = 0.0
    metric_batch_size = 128
    for start in range(0, n, metric_batch_size):
        end = min(n, start + metric_batch_size)
        target = np.asarray(true[start:end], dtype=np.float32)
        base_diff = np.asarray(pred[start:end], dtype=np.float32) - target
        adapted_diff = np.asarray(served[start:end], dtype=np.float32) - target
        base_row_sse = np.sum(base_diff * base_diff, axis=(1, 2), dtype=np.float64)
        adapted_row_sse = np.sum(adapted_diff * adapted_diff, axis=(1, 2), dtype=np.float64)
        base_sse = float(base_row_sse.sum(dtype=np.float64))
        adapted_sse = float(adapted_row_sse.sum(dtype=np.float64))
        base_sae = float(np.abs(base_diff).sum(dtype=np.float64))
        adapted_sae = float(np.abs(adapted_diff).sum(dtype=np.float64))
        base_origin_sse[start:end] = base_row_sse
        adapted_origin_sse[start:end] = adapted_row_sse
        total_base_sse += base_sse
        total_adapted_sse += adapted_sse
        total_base_sae += base_sae
        total_adapted_sae += adapted_sae

    elements = int(n * horizon * channels)
    origin_gain = 100.0 * (base_origin_sse - adapted_origin_sse) / np.maximum(base_origin_sse, 1e-12)
    cvar_count = max(1, int(math.ceil(0.10 * len(origin_gain))))
    ordered_gain = np.sort(origin_gain)
    metrics: dict[str, float | int] = {
        "service_origins": n,
        "segment_len": horizon,
        "evaluated_elements": elements,
        "base_mse": total_base_sse / elements,
        "mse": total_adapted_sse / elements,
        "base_mae": total_base_sae / elements,
        "mae": total_adapted_sae / elements,
        "gain_vs_base_pct": 100.0 * (total_base_sse - total_adapted_sse) / max(total_base_sse, 1e-12),
        "mae_gain_vs_base_pct": 100.0 * (total_base_sae - total_adapted_sae) / max(total_base_sae, 1e-12),
        "origin_gain_mean_pct": float(np.mean(origin_gain)),
        "origin_gain_q05_pct": float(np.quantile(origin_gain, 0.05)),
        "origin_gain_cvar10_pct": float(np.mean(ordered_gain[:cvar_count])),
        "origin_gain_worst_pct": float(np.min(origin_gain)),
        "origin_win_rate_pct": float(100.0 * np.mean(origin_gain > 0.0)),
    }
    return metrics


def latest_commit_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    served: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate latest reforecasts over four equal commitment intervals.

    For an initial issue origin r and commit offset s*H/4, both Base and COSA
    use the forecast freshly issued at row r+s*H/4, and only its first H/4
    leads are counted.  This is the same current-vintage service surface as the
    CommitCast full144 lifecycle result.
    """
    n, horizon, channels = pred.shape
    if horizon % 4:
        raise ValueError(f"Four-interval comparison requires H divisible by 4, got {horizon}")
    segment_len = horizon // 4
    service_origins = n - 3 * segment_len
    if service_origins <= 0:
        raise ValueError(f"Not enough origins for H={horizon}: N={n}")

    base_origin_sse = np.zeros(service_origins, dtype=np.float64)
    adapted_origin_sse = np.zeros(service_origins, dtype=np.float64)
    total_base_sse = total_adapted_sse = 0.0
    total_base_sae = total_adapted_sae = 0.0
    metric_batch_size = 128

    for commit_index in range(4):
        offset = commit_index * segment_len
        for local_start in range(0, service_origins, metric_batch_size):
            local_end = min(service_origins, local_start + metric_batch_size)
            start = offset + local_start
            end = offset + local_end
            target = np.asarray(true[start:end, :segment_len], dtype=np.float32)
            base_diff = np.asarray(pred[start:end, :segment_len], dtype=np.float32) - target
            adapted_diff = np.asarray(served[start:end, :segment_len], dtype=np.float32) - target
            base_row_sse = np.sum(base_diff * base_diff, axis=(1, 2), dtype=np.float64)
            adapted_row_sse = np.sum(adapted_diff * adapted_diff, axis=(1, 2), dtype=np.float64)
            base_origin_sse[local_start:local_end] += base_row_sse
            adapted_origin_sse[local_start:local_end] += adapted_row_sse
            total_base_sse += float(base_row_sse.sum(dtype=np.float64))
            total_adapted_sse += float(adapted_row_sse.sum(dtype=np.float64))
            total_base_sae += float(np.abs(base_diff).sum(dtype=np.float64))
            total_adapted_sae += float(np.abs(adapted_diff).sum(dtype=np.float64))

    elements = int(service_origins * horizon * channels)
    origin_gain = 100.0 * (base_origin_sse - adapted_origin_sse) / np.maximum(base_origin_sse, 1e-12)
    cvar_count = max(1, int(math.ceil(0.10 * len(origin_gain))))
    ordered_gain = np.sort(origin_gain)
    return {
        "service_origins": service_origins,
        "segment_len": segment_len,
        "evaluated_elements": elements,
        "base_mse": total_base_sse / elements,
        "mse": total_adapted_sse / elements,
        "base_mae": total_base_sae / elements,
        "mae": total_adapted_sae / elements,
        "gain_vs_base_pct": 100.0 * (total_base_sse - total_adapted_sse) / max(total_base_sse, 1e-12),
        "mae_gain_vs_base_pct": 100.0 * (total_base_sae - total_adapted_sae) / max(total_base_sae, 1e-12),
        "origin_gain_mean_pct": float(np.mean(origin_gain)),
        "origin_gain_q05_pct": float(np.quantile(origin_gain, 0.05)),
        "origin_gain_cvar10_pct": float(np.mean(ordered_gain[:cvar_count])),
        "origin_gain_worst_pct": float(np.min(origin_gain)),
        "origin_win_rate_pct": float(100.0 * np.mean(origin_gain > 0.0)),
    }


def verify_overlap(true: np.ndarray) -> float:
    if len(true) <= 1:
        return 0.0
    return float(np.max(np.abs(true[:-1, 1:] - true[1:, :-1])))


def run_stream(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    with np.load(path, allow_pickle=False) as archive:
        pred = np.asarray(archive["pred"], dtype=np.float32)
        true = np.asarray(archive["true"], dtype=np.float32)
        metadata_json = archive["metadata_json"] if "metadata_json" in archive.files else None
        meta = stream_metadata(path, metadata_json)
    if args.max_origins:
        pred = pred[: args.max_origins]
        true = true[: args.max_origins]
    if pred.shape != true.shape or pred.ndim != 3:
        raise ValueError(f"Bad stream shape {path}: {pred.shape}/{true.shape}")
    if int(meta["horizon"]) != pred.shape[1]:
        raise ValueError(f"Metadata H={meta['horizon']} disagrees with array {pred.shape}")
    overlap_error = verify_overlap(true)
    if overlap_error != 0.0:
        raise AssertionError(f"Origins are not one-step overlapping: max abs error={overlap_error}")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_peak = torch.cuda.max_memory_allocated(device)
    else:
        baseline_peak = 0

    served, audit = strict_serve(
        pred,
        true,
        device=device,
        context_size=args.context,
        update_batch_size=args.update_batch_size,
        serve_batch_size=args.serve_batch_size,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    # Metrics are deliberately computed only after every served prediction is frozen.
    if args.metric_protocol == "full_surface":
        metrics = full_surface_metrics(pred, true, served)
    else:
        metrics = latest_commit_metrics(pred, true, served)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_mb = (torch.cuda.max_memory_allocated(device) - baseline_peak) / (1024.0**2)
    else:
        peak_mb = 0.0

    n, horizon, channels = pred.shape
    row: dict[str, Any] = {
        **meta,
        "n": n,
        "channels": channels,
        "method": "COSA-FCR",
        "metric_protocol": args.metric_protocol,
        "target_visibility": "whole HxV row only after final target matures; update before current serve",
        **metrics,
        **audit,
        "metric_evaluation_after_stream_frozen": True,
        "adapter_runtime_sec": time.perf_counter() - started,
        "incremental_cuda_peak_mb": peak_mb,
        "protocol": (
            "strict prequential COSA; full-trajectory maturity; cached issue-time context; "
            + (
                "complete [N,H,V] evaluation"
                if args.metric_protocol == "full_surface"
                else "latest-current reforecast at issue/c25/c50/c75; matched service intervals"
            )
        ),
    }
    return row


def self_test() -> None:
    """Counterfactual future-label test plus exact maturity-boundary assertions."""
    rng = np.random.default_rng(20260726)
    n, horizon, channels = 48, 8, 2
    timeline = rng.normal(size=(n + horizon - 1, channels)).astype(np.float32)
    true_a = np.stack([timeline[i : i + horizon] for i in range(n)])
    altered = timeline.copy()
    cutoff = 25
    altered[cutoff:] += 7.0
    true_b = np.stack([altered[i : i + horizon] for i in range(n)])
    pred = rng.normal(size=(n, horizon, channels)).astype(np.float32)

    kwargs = dict(
        device=torch.device("cpu"),
        context_size=3,
        update_batch_size=4,
        serve_batch_size=5,
        steps=2,
        lr=1e-3,
        weight_decay=1e-4,
        seed=0,
    )
    served_a, audit_a = strict_serve(pred, true_a, **kwargs)
    served_b, audit_b = strict_serve(pred, true_b, **kwargs)
    pre_cutoff_diff = float(np.max(np.abs(served_a[: cutoff + 1] - served_b[: cutoff + 1])))
    post_cutoff_diff = float(np.max(np.abs(served_a[cutoff + 1 :] - served_b[cutoff + 1 :])))
    if pre_cutoff_diff > 1e-7:
        raise AssertionError(f"Future-label counterfactual changed prior predictions: {pre_cutoff_diff}")
    if post_cutoff_diff <= 1e-7:
        raise AssertionError("Counterfactual never affected predictions after its labels became observable")
    for audit in (audit_a, audit_b):
        if audit["prequential_violation_count"] or audit["context_violation_count"]:
            raise AssertionError(audit)
        if audit["max_train_target_time_minus_issue"] >= 0:
            raise AssertionError(audit)
        if audit["max_context_target_time_minus_issue"] >= 0:
            raise AssertionError(audit)
    print(
        json.dumps(
            {
                "status": "PASS",
                "counterfactual_pre_observation_max_abs_diff": pre_cutoff_diff,
                "counterfactual_post_observation_max_abs_diff": post_cutoff_diff,
                "audit": audit_a,
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    launch_cwd = Path.cwd()
    for name in ("repo", "stream_root", "output_dir"):
        value = getattr(args, name)
        if value is not None and not value.is_absolute():
            setattr(args, name, (launch_cwd / value).resolve())
    upstream = load_upstream_adapter(args.repo)
    if args.self_test:
        self_test()
        return
    if args.stream_root is None or args.output_dir is None:
        raise SystemExit("--stream_root and --output_dir are required unless --self_test is used")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.stream_root.glob("*/adapter_stream.npz"))
    if args.include_regex:
        pattern = re.compile(args.include_regex)
        paths = [path for path in paths if pattern.search(path.parent.name)]
    paths = [path for index, path in enumerate(paths) if index % args.num_shards == args.shard_index]
    if args.max_streams:
        paths = paths[: args.max_streams]

    result_path = args.output_dir / f"shard_{args.shard_index}.csv"
    rows: list[dict[str, Any]] = load_existing(result_path) if args.resume else []
    done = {row["stream"] for row in rows}

    for index, path in enumerate(paths, 1):
        if path.parent.name in done:
            print(f"[{index}/{len(paths)}] SKIP {path.parent.name}", flush=True)
            continue
        print(f"[{index}/{len(paths)}] {path.parent.name}", flush=True)
        row = run_stream(path, args)
        rows.append(row)
        atomic_write_csv(result_path, rows, RESULT_FIELDS)
        print(
            f"  base={row['base_mse']:.6f} cosa={row['mse']:.6f} "
            f"gain={row['gain_vs_base_pct']:.3f}% updates={row['updates']} "
            f"audit={row['prequential_violation_count']}/{row['context_violation_count']}",
            flush=True,
        )

    config = {
        **{
            key: value
            for key, value in vars(args).items()
            if key not in {"repo", "stream_root", "output_dir"}
        },
        "upstream": upstream,
        "stream_root": args.stream_root.name,
        "output_dir": args.output_dir.name,
        "maturity_rule": "training row j is visible at issue i iff j + H <= i",
        "metric_rule": (
            "COSA complete [N,H,V] surface"
            if args.metric_protocol == "full_surface"
            else "latest current reforecast at four commit times, matched first-quarter service intervals"
        )
        + "; all predictions freeze before evaluation-label pass",
    }
    config_path = args.output_dir / f"config_{args.shard_index}.json"
    temporary = config_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, config_path)


if __name__ == "__main__":
    main()
