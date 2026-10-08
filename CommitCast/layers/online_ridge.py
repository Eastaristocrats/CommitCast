"""Lead/channel-wise online ridge and settled-risk exposure for CommitCast.

The numerical routines preserve the reference implementation's batched
updates and floating-point pathway. Historical function identifiers are
retained for compatibility with existing experiments.
"""
from __future__ import annotations
import math
import numpy as np
import torch

def select_offsets(delay: int, points: int) -> list[int]:
    """Select forecast-trajectory offsets from issue time to commit time."""

    if delay <= 0:
        return [0]
    if points <= 0 or points >= delay + 1:
        return list(range(delay + 1))
    raw = np.linspace(0, delay, num=max(2, int(points)))
    offsets = sorted({int(round(x)) for x in raw} | {0, int(delay)})
    return offsets


def _ridge_products(x: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a grouped Gram matrix and cross-product with shared packing.

    Inputs use [lead, row, channel, feature] order. Explicit matrix views
    preserve the einsum reduction layout while packing its left operand once
    for both sufficient statistics. Keep the two operand layouts distinct:
    transposing one packed copy changes the backend's summation pathway.
    """
    leads, rows, channels, features = x.shape
    left = x.permute(0, 2, 3, 1).reshape(-1, features, rows)
    right = x.permute(0, 2, 1, 3).reshape(-1, rows, features)
    response = target.permute(0, 2, 1).reshape(-1, rows, 1)
    gram = torch.bmm(left, right).reshape(leads, channels, features, features)
    cross = torch.bmm(left, response).reshape(leads, channels, features)
    return gram, cross


@torch.inference_mode()
def run_fsd_ridge(
    *,
    pred: np.ndarray,
    true: np.ndarray,
    delay: int,
    feature_names: list[str],
    traj_points: int,
    ridge_lambda: float,
    beta_clip: float,
    correction_clip: float,
    block_size: int,
    warmup_matured: int,
    feature_clip: float,
    target_clip: float,
    device: str,
    maturity: str = "leadwise",
    feature_cache: str = "none",
    feature_cache_max_gb: float = 8.0,
    diffusion_schedule_rate: float = 1.0,
    feedback_delay: int = 0,
    leadwise_solve_mode: str = "per_lead",
    target_cache: bool = False,
    leadwise_update_mode: str = "per_lead",
    served_leads: int | None = None,
    target_arrival_delay: np.ndarray | None = None,
) -> np.ndarray:
    """Closed-form online denoiser using repeated-forecast trajectory features."""

    import torch

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    feedback_delay = int(feedback_delay)
    if feedback_delay < 0:
        raise ValueError("feedback_delay must be non-negative")
    arrival_delay = None
    if target_arrival_delay is not None:
        arrival_delay = np.asarray(target_arrival_delay, dtype=np.int64)
        if arrival_delay.ndim != 1 or np.any(arrival_delay < 0):
            raise ValueError("target_arrival_delay must be a non-negative vector")
        if not np.any(arrival_delay):
            # Preserve the established zero-delay arithmetic exactly.
            arrival_delay = None
    if str(leadwise_solve_mode) not in {"per_lead", "batched"}:
        raise ValueError(f"Unknown leadwise_solve_mode: {leadwise_solve_mode}")
    if str(leadwise_update_mode) not in {"per_lead", "batched"}:
        raise ValueError(f"Unknown leadwise_update_mode: {leadwise_update_mode}")
    if str(leadwise_update_mode) == "batched" and str(leadwise_solve_mode) != "batched":
        raise ValueError("batched leadwise updates require batched leadwise solves")

    pred_t = torch.as_tensor(pred, dtype=torch.float32, device=dev)
    true_t = torch.as_tensor(true, dtype=torch.float32, device=dev)
    n_total, horizon, channels = pred_t.shape
    remaining_full = horizon - int(delay)
    if served_leads is None:
        remaining = remaining_full
    else:
        remaining = int(served_leads)
        if remaining <= 0 or remaining > remaining_full:
            raise ValueError(
                f"served_leads must be in [1, {remaining_full}], got {served_leads}"
            )
    n = n_total - int(delay)
    if n <= 0 or remaining_full <= 0:
        raise ValueError(f"Invalid delay={delay} for pred shape={pred.shape}")

    visible_delay = int(delay) - feedback_delay
    if visible_delay <= 0:
        return (
            pred_t[int(delay) : n_total, :remaining, :]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    issue_t = pred_t[:n, delay:horizon, :]
    # Scalar context features deliberately retain the complete post-commit
    # suffix.  The ridge state and returned prediction, however, need only the
    # interval actually served before the next commitment.  Later leads are
    # reforecast and recomputed at that commitment, so materializing them here
    # is pure compute/state/transfer waste.
    ref_full_t = pred_t[delay:n_total, :remaining_full, :]
    ref_t = ref_full_t[:, :remaining, :]
    y_t = true_t[:n, delay : delay + remaining, :]
    offsets = select_offsets(int(delay), int(traj_points))
    fdim = len(feature_names)
    block = int(block_size)
    requested_features = set(feature_names)
    need_prefix_ac1 = "prefix_ac1" in requested_features
    need_path_abs = "path_abs" in requested_features
    need_ref_local = bool({"reforecast_local", "reforecast_local_deviation"} & requested_features)
    need_bridge = any(
        name == "bridge_precision_gap"
        or name.startswith("bridge_innovation_")
        or name.startswith("bridge_moment_")
        for name in requested_features
    )
    need_trajectory = need_bridge or bool({
        "traj_std", "path_abs", "net_drift", "last_step", "traj_mean_gap", "traj_curvature"
    } & requested_features)

    beta = torch.zeros((remaining, channels, fdim), dtype=torch.float32, device=dev)
    A = torch.zeros((remaining, channels, fdim, fdim), dtype=torch.float32, device=dev)
    b = torch.zeros((remaining, channels, fdim), dtype=torch.float32, device=dev)
    eye = torch.eye(fdim, dtype=torch.float32, device=dev)[None, None, :, :]
    ridge_eye = float(ridge_lambda) * eye
    out = torch.empty_like(ref_t)
    solve_status = []
    solve_failed = None

    def merge_solve_status():
        nonlocal solve_failed
        failed = torch.any(torch.cat([info.reshape(-1) for info in solve_status]) != 0)
        solve_failed = failed if solve_failed is None else solve_failed.logical_or_(failed)
        solve_status.clear()

    last_matured = 0
    matured_seen = 0
    # These are control-plane counters. Keeping them on the host avoids a GPU
    # synchronization for every lead in every service block.
    last_matured_by_lead = [0] * remaining
    matured_seen_by_lead = [0] * remaining
    consumed_by_lead = (
        np.zeros((remaining, n), dtype=bool) if arrival_delay is not None else None
    )

    lead_idx = torch.arange(remaining, dtype=torch.long, device=dev)
    block_rows = torch.arange(min(block, n), dtype=torch.long, device=dev)[None, :]

    def build_features(
        row_start: int,
        row_end: int,
        selected_lead: int | None = None,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        if selected_lead is None:
            active_lead_start = 0
            active_leads = lead_idx
        else:
            active_lead_start = int(selected_lead)
            active_leads = lead_idx[active_lead_start : active_lead_start + 1]
        active_remaining = int(active_leads.numel())
        # Prefix residuals are needed only for this feature block. Explicit
        # contiguous layout preserves the reference reduction order even when
        # chronological target windows overlap or the prefix is very long.
        pr = (true_t[row_start:row_end, :visible_delay, :]
              - pred_t[row_start:row_end, :visible_delay, :]).contiguous()
        if arrival_delay is not None:
            prefix_rows = np.arange(row_start, row_end, dtype=np.int64)[:, None]
            prefix_leads = np.arange(visible_delay, dtype=np.int64)[None, :]
            visible_np = prefix_leads + arrival_delay[prefix_rows + prefix_leads] < int(delay)
            visible_t = torch.as_tensor(visible_np, dtype=torch.bool, device=dev)[..., None]
            pr = torch.where(visible_t, pr, torch.zeros_like(pr))
        issue_full = issue_t[row_start:row_end]
        ref_full = ref_full_t[row_start:row_end]
        active_slice = slice(active_lead_start, active_lead_start + active_remaining)
        issue_active = issue_full[:, active_slice, :]
        ref = ref_full[:, active_slice, :]

        prefix_scale = torch.sqrt(torch.mean(pr * pr, dim=1, keepdim=True)) + 1e-4
        denom = prefix_scale[:, 0, :]
        first_pr = pr[:, 0, :] / denom
        last_pr = pr[:, -1, :] / denom
        prefix_mean = pr.mean(dim=1) / denom
        prefix_slope = last_pr - first_pr
        prefix_ac1 = None
        if need_prefix_ac1:
            centered = pr - pr.mean(dim=1, keepdim=True)
            if pr.shape[1] > 1:
                ac_num = torch.mean(centered[:, 1:, :] * centered[:, :-1, :], dim=1)
                ac_den = torch.mean(centered * centered, dim=1) + 1e-6
                prefix_ac1 = ac_num / ac_den
            else:
                prefix_ac1 = torch.zeros_like(prefix_mean)

        traj_std = path_abs = net_drift = last_step = traj_mean_gap = traj_curvature = None
        if need_trajectory:
            sum_state = None
            sum_sq_state = None
            sum_abs_step = None
            first_state = None
            last_state = None
            prev_state = None
            before_last_state = None
            mid_state = None
            mid_offset = offsets[len(offsets) // 2]
            trajectory_states: list["torch.Tensor"] = []

            for offset in offsets:
                state_lead_start = active_lead_start + int(delay) - int(offset)
                state = pred_t[
                    row_start + offset : row_end + offset,
                    state_lead_start : state_lead_start + active_remaining,
                    :,
                ]
                if need_bridge:
                    trajectory_states.append(state)
                if sum_state is None:
                    sum_state = torch.zeros_like(state)
                    sum_sq_state = torch.zeros_like(state)
                    if need_path_abs:
                        sum_abs_step = torch.zeros_like(state)
                    first_state = state
                elif need_path_abs:
                    assert sum_abs_step is not None
                    sum_abs_step = sum_abs_step + torch.abs(state - prev_state)
                sum_state.add_(state)
                sum_sq_state.add_(state * state)
                if offset == mid_offset:
                    mid_state = state
                before_last_state = prev_state
                prev_state = state
                last_state = state

            assert sum_state is not None
            assert sum_sq_state is not None
            assert first_state is not None
            assert last_state is not None
            if before_last_state is None:
                before_last_state = first_state
            if mid_state is None:
                mid_state = first_state

            n_offsets = float(len(offsets))
            traj_mean = sum_state / n_offsets
            traj_var = torch.clamp(sum_sq_state / n_offsets - traj_mean * traj_mean, min=0.0)
            traj_std = torch.sqrt(traj_var + 1e-8) / prefix_scale
            path_abs = None
            if need_path_abs:
                assert sum_abs_step is not None
                path_abs = sum_abs_step / max(1.0, n_offsets - 1.0) / prefix_scale
            net_drift = (last_state - first_state) / prefix_scale if "net_drift" in requested_features else None
            last_step = (last_state - before_last_state) / prefix_scale
            traj_mean_gap = (traj_mean - last_state) / prefix_scale
            traj_curvature = (last_state - 2.0 * mid_state + first_state) / prefix_scale
        ref_local = (ref - ref_full.mean(dim=1, keepdim=True)) / prefix_scale if need_ref_local else None

        bridge_precision_gap = None
        bridge_innovations: dict[str, "torch.Tensor"] = {}
        bridge_moments: dict[str, "torch.Tensor"] = {}
        if need_bridge:
            rate = max(float(diffusion_schedule_rate), 1e-6)
            schedule_values: list["torch.Tensor"] = []
            precision_sum = torch.zeros((active_remaining,), dtype=ref.dtype, device=dev)
            precision_state = torch.zeros_like(ref)
            for offset, state in zip(offsets, trajectory_states):
                state_lead = active_leads.to(ref.dtype) + float(delay - offset + 1)
                schedule = torch.exp(rate * state_lead / float(max(1, horizon)))
                precision = torch.reciprocal(schedule + 1e-6)
                schedule_values.append(schedule)
                precision_sum = precision_sum + precision
                precision_state = precision_state + state * precision[None, :, None]
            bridge_precision = precision_state / precision_sum[None, :, None]
            bridge_precision_gap = (bridge_precision - last_state) / prefix_scale

            innovation_values: list["torch.Tensor"] = []
            for innovation_idx in range(len(trajectory_states) - 1):
                schedule_delta = torch.clamp(
                    schedule_values[innovation_idx] - schedule_values[innovation_idx + 1],
                    min=1e-6,
                )
                innovation = trajectory_states[innovation_idx + 1] - trajectory_states[innovation_idx]
                innovation = innovation / prefix_scale / torch.sqrt(schedule_delta)[None, :, None]
                bridge_innovations[f"bridge_innovation_{innovation_idx}"] = innovation
                innovation_values.append(innovation)

            if innovation_values:
                positions = torch.linspace(0.0, 1.0, len(innovation_values), dtype=ref.dtype, device=dev)
                basis_values = [
                    torch.ones_like(positions),
                    2.0 * positions - 1.0,
                    6.0 * positions * positions - 6.0 * positions + 1.0,
                ]
                for moment_idx, basis in enumerate(basis_values):
                    basis = basis / torch.sqrt(torch.sum(basis * basis) + 1e-6)
                    moment = torch.zeros_like(ref)
                    for weight, innovation in zip(basis, innovation_values):
                        moment = moment + weight * innovation
                    bridge_moments[f"bridge_moment_{moment_idx}"] = moment

        disp_full = (issue_full - ref_full) / prefix_scale
        disp_active = disp_full[:, active_slice, :]
        disp_mean = disp_full.mean(dim=1)
        disp_slope = disp_full[:, -1, :] - disp_full[:, 0, :] if "disp_slope" in requested_features else None
        disp_local = disp_active - disp_mean[:, None, :]
        issue_local = (issue_active - issue_full.mean(dim=1, keepdim=True)) / prefix_scale
        ref_first = ref_full[:, 0, :]
        ref_last = ref_full[:, -1, :]
        reforecast_mean = ref_full.mean(dim=1) / denom
        reforecast_slope = (ref_last - ref_first) / denom
        cross = prefix_mean.mean(dim=1, keepdim=True).expand(-1, channels) if "cross" in requested_features else None

        scalar_map = {
            "bias": torch.ones_like(prefix_mean) if "bias" in requested_features else None,
            "mean": prefix_mean,
            "last": last_pr,
            "slope": prefix_slope,
            "cross": cross,
            "prefix_mean": prefix_mean,
            "prefix_last": last_pr,
            "prefix_slope": prefix_slope,
            "prefix_ac1": prefix_ac1,
            "disp_mean": disp_mean,
            "disp_slope": disp_slope,
            "reforecast_mean": reforecast_mean,
            "reforecast_slope": reforecast_slope,
        }
        tensor_map = {
            "disp": disp_active,
            "disp_local_deviation": disp_local,
            "issue_local_deviation": issue_local,
            "net_drift": net_drift,
            "last_step": last_step,
            "traj_mean_gap": traj_mean_gap,
            "traj_std": traj_std,
            "path_abs": path_abs,
            "traj_curvature": traj_curvature,
            "reforecast_local": ref_local,
            "reforecast_local_deviation": ref_local,
            "bridge_precision_gap": bridge_precision_gap,
            **bridge_innovations,
            **bridge_moments,
        }

        feats: list["torch.Tensor"] = []
        for name in feature_names:
            if name in tensor_map:
                feats.append(tensor_map[name][..., None])
            elif name in scalar_map:
                feats.append(scalar_map[name][:, None, :, None].expand(-1, active_remaining, -1, -1))
            else:
                raise ValueError(f"Unknown FSD feature: {name}")
        x = torch.cat(feats, dim=-1)
        x.clamp_(min=-float(feature_clip), max=float(feature_clip))
        return x, prefix_scale

    x_cache = None
    scale_cache = None
    cache_mode = str(feature_cache).lower()
    if cache_mode not in {"none", "auto", "full_tensor", "online"}:
        raise ValueError(f"Unknown feature_cache mode: {feature_cache}")
    online_cache = cache_mode == "online"
    if online_cache and bool(target_cache):
        raise ValueError("target_cache is incompatible with strict online feature caching")
    if online_cache:
        extra_arrival_retention = int(np.max(arrival_delay, initial=0)) if arrival_delay is not None else 0
        cache_rows = min(n, remaining + feedback_delay + extra_arrival_retention + block)
    else:
        cache_rows = n
    compact_cache = (online_cache and dev.type == "cuda" and maturity == "leadwise"
                     and arrival_delay is None and feedback_delay == 0)
    cache_capacity = cache_offsets = None
    if compact_cache:
        # Lead r needs r + block issued rows until its next matured update.
        # Give each lead its own ring instead of retaining the longest history
        # for every lead. Scale history remains shared across leads.
        capacities = np.minimum(n, np.arange(remaining, dtype=np.int64) + block)
        offsets_np = np.cumsum(capacities) - capacities
        cache_capacity = torch.as_tensor(capacities, dtype=torch.long, device=dev)
        cache_offsets = torch.as_tensor(offsets_np, dtype=torch.long, device=dev)
        feature_slots = int(capacities.sum())
    else:
        feature_slots = int(cache_rows) * int(remaining)
    cache_bytes = feature_slots * int(channels) * int(fdim) * 4
    cache_gb = cache_bytes / (1024.0**3)
    use_cache = online_cache or cache_mode == "full_tensor" or (
        cache_mode == "auto" and cache_gb <= float(feature_cache_max_gb)
    )
    if use_cache and cache_mode == "auto" and dev.type == "cuda":
        # Keep room for the model inputs and per-block trajectory construction.
        # This preserves the exact computation while avoiding an OOM fallback on
        # long, high-dimensional streams.
        free_bytes, _ = torch.cuda.mem_get_info(dev)
        required_bytes = cache_bytes + int(n) * int(channels) * 4
        use_cache = required_bytes <= int(0.65 * free_bytes)
    if use_cache:
        # Preallocation avoids retaining both block tensors and a second
        # concatenated copy of the full cache at peak memory.
        cache_shape = (feature_slots, channels, fdim) if compact_cache else (cache_rows, remaining, channels, fdim)
        x_cache = torch.empty(cache_shape, dtype=pred_t.dtype, device=dev)
        scale_cache = torch.empty((cache_rows, 1, channels), dtype=pred_t.dtype, device=dev)
        if not online_cache:
            for cache_start in range(0, n, block):
                cache_end = min(n, cache_start + block)
                x_part, scale_part = build_features(cache_start, cache_end)
                x_cache[cache_start:cache_end] = x_part
                scale_cache[cache_start:cache_end] = scale_part

    target_cache_tensor = None
    if bool(target_cache) and x_cache is not None and scale_cache is not None:
        target_cache_tensor = torch.empty_like(ref_t)
        for cache_start in range(0, n, block):
            cache_end = min(n, cache_start + block)
            target_part = (
                y_t[cache_start:cache_end] - ref_t[cache_start:cache_end]
            ) / scale_cache[cache_start:cache_end]
            target_part.clamp_(min=-float(target_clip), max=float(target_clip))
            target_cache_tensor[cache_start:cache_end] = target_part

    def get_features(
        row_start: int,
        row_end: int,
        selected_lead: int | None = None,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        if x_cache is None or scale_cache is None:
            return build_features(row_start, row_end, selected_lead=selected_lead)
        if online_cache:
            rows = torch.arange(row_start, row_end, dtype=torch.long, device=dev)
            cache_index = rows % cache_rows
            scale = scale_cache[cache_index]
            if compact_cache:
                leads = lead_idx if selected_lead is None else lead_idx[selected_lead:selected_lead+1]
                feature_index = cache_offsets[leads][None, :] + rows[:, None] % cache_capacity[leads][None, :]
                return x_cache[feature_index], scale
            if selected_lead is None:
                return x_cache[cache_index], scale
            return x_cache[cache_index, selected_lead : selected_lead + 1], scale
        scale = scale_cache[row_start:row_end]
        if selected_lead is None:
            return x_cache[row_start:row_end], scale
        return x_cache[row_start:row_end, selected_lead : selected_lead + 1], scale

    for start in range(0, n, block):
        end = min(n, start + block)
        if maturity == "full_suffix":
            matured_end = max(0, start - remaining + 1 - feedback_delay)
            if matured_end > last_matured:
                for train_start in range(last_matured, matured_end, block):
                    train_end = min(matured_end, train_start + block)
                    x, prefix_scale = get_features(train_start, train_end)
                    if target_cache_tensor is None:
                        target = (y_t[train_start:train_end] - ref_t[train_start:train_end]) / prefix_scale
                        target.clamp_(min=-float(target_clip), max=float(target_clip))
                    else:
                        target = target_cache_tensor[train_start:train_end]
                    denom = max(1, int(x.shape[0]))
                    A.add_(torch.einsum("blcf,blcg->lcfg", x, x) / denom)
                    b.add_(torch.einsum("blcf,blc->lcf", x, target) / denom)
                    reg = A + ridge_eye
                    beta = torch.linalg.solve(
                        reg.reshape(-1, fdim, fdim),
                        b.reshape(-1, fdim, 1),
                    ).reshape(remaining, channels, fdim)
                    beta.clamp_(min=-float(beta_clip), max=float(beta_clip))
                    matured_seen += int(train_end - train_start)
                last_matured = matured_end
            active = None if matured_seen >= int(warmup_matured) else torch.zeros((remaining,), dtype=torch.bool, device=dev)
        elif maturity == "leadwise":
            updated_leads: list[int] = []
            pending_updates: list[tuple[int, int, int]] = []
            use_batched_updates = bool(
                str(leadwise_update_mode) == "batched"
                and x_cache is not None
                and (target_cache_tensor is not None or online_cache)
            )
            for r in range(remaining):
                if arrival_delay is not None:
                    assert consumed_by_lead is not None
                    target_ids = np.arange(n, dtype=np.int64) + int(delay) + r
                    if int(target_ids[-1]) >= int(arrival_delay.size):
                        raise ValueError("target_arrival_delay does not cover all absolute targets")
                    eligible_mask = (
                        (~consumed_by_lead[r])
                        & (np.arange(n, dtype=np.int64) + r + arrival_delay[target_ids] < start)
                    )
                    eligible = np.flatnonzero(eligible_mask)
                    if eligible.size == 0:
                        continue
                    arrival_times = eligible + r + arrival_delay[target_ids[eligible]]
                    order = np.lexsort((eligible, target_ids[eligible], arrival_times))
                    eligible = eligible[order]
                    for chunk_start in range(0, int(eligible.size), block):
                        index_np = eligible[chunk_start : chunk_start + block]
                        index_t = torch.as_tensor(index_np, dtype=torch.long, device=dev)
                        if x_cache is None or scale_cache is None:
                            raise ValueError(
                                "target_arrival_delay requires feature_cache=online or full_tensor"
                            )
                        cache_index = index_t % cache_rows if online_cache else index_t
                        xr = x_cache[cache_index, r, :, :]
                        scale_for_target = scale_cache[cache_index, 0, :]
                        tr = (
                            y_t[index_t, r, :] - ref_t[index_t, r, :]
                        ) / scale_for_target
                        tr.clamp_(min=-float(target_clip), max=float(target_clip))
                        denom = max(1, int(xr.shape[0]))
                        A[r].add_(torch.einsum("bcf,bcg->cfg", xr, xr) / denom)
                        b[r].add_(torch.einsum("bcf,bc->cf", xr, tr) / denom)
                        matured_seen_by_lead[r] += int(index_np.size)
                    consumed_by_lead[r, eligible] = True
                    if str(leadwise_solve_mode) == "per_lead":
                        reg = A[r] + ridge_eye[0]
                        beta_r = torch.linalg.solve(reg, b[r][..., None]).squeeze(-1)
                        beta[r] = beta_r.clamp(min=-float(beta_clip), max=float(beta_clip))
                    else:
                        updated_leads.append(r)
                    continue
                matured_end_r = max(0, start - r - feedback_delay)
                prev = last_matured_by_lead[r]
                if matured_end_r <= prev:
                    continue
                lead_updated = False
                for train_start in range(prev, matured_end_r, block):
                    train_end = min(matured_end_r, train_start + block)
                    if use_batched_updates:
                        pending_updates.append((r, train_start, train_end))
                        lead_updated = True
                    else:
                        x, prefix_scale = get_features(train_start, train_end, selected_lead=int(r))
                        if target_cache_tensor is None:
                            target = (y_t[train_start:train_end, r : r + 1, :] - ref_t[train_start:train_end, r : r + 1, :]) / prefix_scale
                            target.clamp_(min=-float(target_clip), max=float(target_clip))
                        else:
                            target = target_cache_tensor[train_start:train_end, r : r + 1, :]
                        xr = x[:, 0, :, :]
                        tr = target[:, 0, :]
                        denom = max(1, int(xr.shape[0]))
                        A[r].add_(torch.einsum("bcf,bcg->cfg", xr, xr) / denom)
                        b[r].add_(torch.einsum("bcf,bc->cf", xr, tr) / denom)
                        if str(leadwise_solve_mode) == "per_lead":
                            reg = A[r] + ridge_eye[0]
                            beta_r = torch.linalg.solve(reg, b[r][..., None]).squeeze(-1)
                            beta[r] = beta_r.clamp(min=-float(beta_clip), max=float(beta_clip))
                        else:
                            lead_updated = True
                    matured_seen_by_lead[r] += int(train_end - train_start)
                last_matured_by_lead[r] = matured_end_r
                if lead_updated:
                    updated_leads.append(r)
            if use_batched_updates and pending_updates:
                updates_by_length: dict[int, list[tuple[int, int, int]]] = {}
                for item in pending_updates:
                    updates_by_length.setdefault(item[2] - item[1], []).append(item)
                partial_features = partial_targets = None
                partial_position = {}
                single_updates = [items[0] for items in updates_by_length.values() if len(items) == 1]
                width = max((hi - lo for _, lo, hi in single_updates), default=0)
                partial_bytes = width * len(single_updates) * channels * (fdim + 1) * 4
                if (dev.type == "cuda" and online_cache and len(single_updates) > 1
                        and partial_bytes <= 32 * 1024 * 1024):
                    # Newly maturing leads have different history lengths.
                    # Gather them together, but retain each original GEMM
                    # length/layout and division so the floating-point path
                    # is unchanged. Padding repeats the last matured row;
                    # it never reads a future feature or target. Large groups
                    # retain the per-lead path to bound additional buffers.
                    metadata = torch.as_tensor(single_updates, dtype=torch.long, device=dev)
                    selected = metadata[:, 0, None]
                    rows = torch.minimum(metadata[:, 1, None] + block_rows[:, :width],
                                         metadata[:, 2, None] - 1)
                    slots = rows % cache_rows
                    if compact_cache:
                        indices = cache_offsets[selected] + rows % cache_capacity[selected]
                        partial_features = x_cache[indices]
                    else:
                        partial_features = x_cache[slots, selected]
                    partial_targets = (y_t[rows, selected] - ref_t[rows, selected]) / scale_cache[slots, 0]
                    partial_targets.clamp_(min=-float(target_clip), max=float(target_clip))
                    partial_position = {item[0]: i for i, item in enumerate(single_updates)}
                for update_length, items in updates_by_length.items():
                    if len(items) == 1:
                        r, train_start, train_end = items[0]
                        position = partial_position.get(r)
                        if position is not None:
                            xr = partial_features[position, :update_length]
                            tr = partial_targets[position, :update_length]
                        elif online_cache:
                            rows = torch.arange(
                                train_start, train_end, dtype=torch.long, device=dev
                            )
                            cache_index = rows % cache_rows
                            if compact_cache:
                                xr = x_cache[cache_offsets[r] + rows % cache_capacity[r]]
                            else:
                                xr = x_cache[cache_index, r, :, :]
                            scale_for_target = scale_cache[cache_index, 0, :]
                        else:
                            xr = x_cache[train_start:train_end, r, :, :]
                            scale_for_target = scale_cache[train_start:train_end, 0, :]
                        if position is None:
                            if target_cache_tensor is None:
                                tr = (
                                    y_t[train_start:train_end, r, :]
                                    - ref_t[train_start:train_end, r, :]
                                ) / scale_for_target
                                tr.clamp_(min=-float(target_clip), max=float(target_clip))
                            else:
                                tr = target_cache_tensor[train_start:train_end, r, :]
                        denom = max(1, int(xr.shape[0]))
                        A[r].add_(torch.einsum("bcf,bcg->cfg", xr, xr) / denom)
                        b[r].add_(torch.einsum("bcf,bc->cf", xr, tr) / denom)
                        continue
                    first_lead = items[0][0]
                    contiguous = all(item[0] == first_lead + i for i, item in enumerate(items))
                    aligned = contiguous and all(item[1] + item[0] == items[0][1] + first_lead for item in items)
                    if aligned:
                        # Regular maturity groups share absolute target times.
                        # Derive device indices from persistent views instead
                        # of synchronously transferring two host lists per block.
                        update_index = lead_idx[first_lead:first_lead+len(items)]
                        row_index = (block_rows[:, :update_length] + items[0][1]
                                     - (update_index[:, None] - first_lead))
                    else:
                        update_index = torch.as_tensor(
                            [item[0] for item in items], dtype=torch.long, device=dev
                        )
                        row_starts = torch.as_tensor(
                            [item[1] for item in items], dtype=torch.long, device=dev
                        )
                        row_index = row_starts[:, None] + block_rows[:, :update_length]
                    lead_index = update_index[:, None].expand(-1, update_length)
                    cache_row_index = row_index % cache_rows if online_cache else row_index
                    if compact_cache:
                        feature_index = cache_offsets[update_index, None] + row_index % cache_capacity[update_index, None]
                        xr = x_cache[feature_index]
                    else:
                        xr = x_cache[cache_row_index, lead_index, :, :]
                    if target_cache_tensor is None:
                        if aligned and y_t.stride(0) == y_t.stride(1):
                            targets = y_t[items[0][1]:items[0][2], first_lead, :][None, :, :]
                        else:
                            targets = y_t[row_index, lead_index, :]
                        tr = (targets - ref_t[row_index, lead_index, :]) / scale_cache[
                            cache_row_index, 0, :
                        ]
                        tr.clamp_(min=-float(target_clip), max=float(target_clip))
                    else:
                        tr = target_cache_tensor[row_index, lead_index, :]
                    delta_A, delta_b = _ridge_products(xr, tr)
                    delta_A.div_(max(1, update_length))
                    delta_b.div_(max(1, update_length))
                    if contiguous:
                        # Contiguous lead groups can update their state in place;
                        # the covariance and cross-product reductions are unchanged.
                        A[first_lead:first_lead+len(items)].add_(delta_A)
                        b[first_lead:first_lead+len(items)].add_(delta_b)
                    else:
                        A[update_index] = A[update_index] + delta_A
                        b[update_index] = b[update_index] + delta_b
                    del delta_A, delta_b
                del partial_features, partial_targets, xr, tr
            if str(leadwise_solve_mode) == "batched" and updated_leads:
                updated_count = len(updated_leads)
                contiguous_prefix = updated_leads == list(range(updated_count))
                if contiguous_prefix:
                    reg = A[:updated_count] + ridge_eye
                    rhs = b[:updated_count]
                else:
                    updated_index = torch.as_tensor(updated_leads, dtype=torch.long, device=dev)
                    reg = A[updated_index] + ridge_eye
                    rhs = b[updated_index]
                system = reg.reshape(-1, fdim, fdim)
                right_hand_side = rhs.reshape(-1, fdim, 1)
                if dev.type == "cuda":
                    # Keep the same LU solve, but inspect every status together
                    # before returning instead of synchronizing at each block.
                    solution, info = torch.linalg.solve_ex(system, right_hand_side, check_errors=False)
                    solve_status.append(info)
                    if len(solve_status) == 32:
                        # Bound status storage independently of stream length.
                        merge_solve_status()
                else:
                    solution = torch.linalg.solve(system, right_hand_side)
                beta_updated = solution.reshape(updated_count, channels, fdim)
                beta_updated.clamp_(min=-float(beta_clip), max=float(beta_clip))
                if contiguous_prefix:
                    beta[:updated_count] = beta_updated
                else:
                    beta[updated_index] = beta_updated
            ready = [seen >= int(warmup_matured) for seen in matured_seen_by_lead]
            # Once every lead is warm, multiplying by an all-ones mask does
            # no work. Keep masks only while at least one lead is inactive.
            active = None if all(ready) else torch.as_tensor(ready, dtype=torch.bool, device=dev)
        else:
            raise ValueError(f"Unknown maturity mode: {maturity}")

        if online_cache:
            # Materialize only the micro-batch that has just arrived.  Matured
            # updates above consume the oldest ring-buffer entries before this
            # write can replace them.  No later-row feature or target is read.
            x_part, scale_part = build_features(start, end)
            # Shared history occupies at most two contiguous spans. CUDA's
            # compact feature rings use each lead's own retention period.
            slot = start % cache_rows
            first = min(end - start, cache_rows - slot)
            if compact_cache:
                rows = torch.arange(start, end, dtype=torch.long, device=dev)[:, None]
                feature_index = cache_offsets[None, :] + rows % cache_capacity[None, :]
                x_cache[feature_index] = x_part
            else:
                x_cache[slot:slot+first].copy_(x_part[:first])
            scale_cache[slot:slot+first].copy_(scale_part[:first])
            if first < end - start:
                tail = end - start - first
                if not compact_cache:
                    x_cache[:tail].copy_(x_part[first:])
                scale_cache[:tail].copy_(scale_part[first:])

        if maturity == "full_suffix" and matured_seen < int(warmup_matured):
            out[start:end] = ref_t[start:end]
        else:
            # The just-issued features are already available before the ring
            # write. Reusing them avoids gathering an identical tensor copy.
            x, prefix_scale = (x_part, scale_part) if online_cache else get_features(start, end)
            raw = torch.einsum("lcf,blcf->blc", beta, x)
            if maturity == "leadwise" and active is not None:
                raw = raw * active[None, :, None].to(raw.dtype)
            corr = raw * prefix_scale
            clip = float(correction_clip) * prefix_scale
            corr = torch.maximum(torch.minimum(corr, clip), -clip)
            out[start:end] = ref_t[start:end] + corr

    if solve_status:
        merge_solve_status()
    if solve_failed is not None and bool(solve_failed.item()):
        raise torch.linalg.LinAlgError("A CommitCast ridge system is singular or could not be solved")
    return out.detach().cpu().numpy().astype(np.float32, copy=False)


def strict_online_discounted_ols_gate_mix(
    target: np.ndarray,
    base: np.ndarray,
    revised: np.ndarray,
    *,
    segment_len: int,
    warmup: int,
    prior_alpha: float,
    ridge: float,
    half_life_rows: float,
    feedback_delay: int = 0,
    row_settlement_lag: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the z=0 discounted OLS gate without pre-reading future labels.

    At service row ``i``, only row ``i - segment_len - feedback_delay`` can
    newly enter the gate state.  Its complete served segment has then matured.
    The current decision and mixture are emitted immediately after that single
    causal update; no future target statistic is materialized or cached.
    """

    target = np.asarray(target)
    base = np.asarray(base)
    revised = np.asarray(revised)
    if target.shape != base.shape or target.shape != revised.shape or target.ndim != 3:
        raise ValueError("target, base and revised must share shape [row, lead, channel]")
    segment_len = int(segment_len)
    feedback_delay = int(feedback_delay)
    if segment_len <= 0 or feedback_delay < 0:
        raise ValueError("segment_len must be positive and feedback_delay non-negative")

    half_life = float(half_life_rows)
    if math.isinf(half_life):
        decay = 1.0
    elif half_life > 0.0:
        decay = math.exp(math.log(0.5) / half_life)
    else:
        raise ValueError("half_life_rows must be positive or infinity")

    n_rows = int(target.shape[0])
    alpha = np.full(n_rows, float(prior_alpha), dtype=np.float64)
    sum_num = 0.0
    sum_den = 0.0
    release_lag = segment_len + feedback_delay
    penalty = max(float(ridge), 0.0)

    settlement_lag = None
    if row_settlement_lag is not None:
        settlement_lag = np.asarray(row_settlement_lag, dtype=np.int64)
        if settlement_lag.shape != (n_rows,):
            raise ValueError("row_settlement_lag must have shape [row]")
        if np.any(settlement_lag < segment_len):
            raise ValueError("row settlement cannot precede complete segment maturity")
    settlement_order = settlement_times = None
    settled_stop = 0
    if settlement_lag is not None:
        row_ids = np.arange(n_rows, dtype=np.int64)
        release_times = row_ids + settlement_lag
        settlement_order = np.lexsort((row_ids, release_times))
        settlement_times = release_times[settlement_order]
    matured_count = 0
    err = np.empty(target.shape[1:], dtype=np.float64)
    delta_matured = np.empty_like(err)
    err_flat = err.reshape(-1)
    delta_flat = delta_matured.reshape(-1)
    warmup_count = int(warmup)
    subtract = np.subtract
    dot = np.dot

    if settlement_lag is None:
        # Regular settlement releases exactly one row per tick. Keep this
        # common path free of temporary lists and arrival-queue bookkeeping.
        for i in range(release_lag, n_rows):
            matured_row = i - release_lag
            frozen = base[matured_row]
            subtract(target[matured_row], frozen, dtype=np.float64, out=err)
            subtract(revised[matured_row], frozen, dtype=np.float64, out=delta_matured)
            sum_num = decay * sum_num + float(dot(err_flat, delta_flat))
            sum_den = decay * sum_den + max(float(dot(delta_flat, delta_flat)), 0.0)
            matured_count += 1
            if matured_count >= warmup_count and sum_den > 1e-12:
                alpha[i] = min(max(sum_num / max(sum_den + penalty, 1e-12), 0.0), 1.0)
    else:
        for i in range(n_rows):
            # Arrival-time order is explicit; target id is the deterministic
            # tie breaker.  A late row is consumed exactly once and never
            # blocks a newer row that has already settled.
            # Sort settlement metadata once. Only rows released by this clock
            # tick can read labels or update state; each row is consumed once.
            previous_stop = settled_stop
            while settled_stop < n_rows and settlement_times[settled_stop] <= i:
                settled_stop += 1
            matured_rows = settlement_order[previous_stop:settled_stop]
            for matured_row in matured_rows:
                # Cast within the subtraction and read only this settled row.
                frozen = base[matured_row]
                subtract(target[matured_row], frozen, dtype=np.float64, out=err)
                subtract(revised[matured_row], frozen, dtype=np.float64, out=delta_matured)
                sum_num = decay * sum_num + float(dot(err_flat, delta_flat))
                sum_den = decay * sum_den + max(float(dot(delta_flat, delta_flat)), 0.0)
                matured_count += 1
            if matured_count >= warmup_count and sum_den > 1e-12:
                alpha[i] = min(max(sum_num / max(sum_den + penalty, 1e-12), 0.0), 1.0)

    # Mixing depends only on forecasts and the already-issued causal alpha
    # sequence.  Vectorizing this label-free operation preserves the exact
    # established arithmetic while removing one Python tensor assignment per
    # served row.  A deployment can emit the identical row expression online.
    # Preserve the input-dtype subtraction before FP64 mixing. Reuse the
    # output buffer instead of retaining full-stream product and sum arrays.
    difference_dtype = np.result_type(revised.dtype, base.dtype)
    gated = np.empty(base.shape, dtype=np.result_type(difference_dtype, alpha.dtype))
    subtract(revised, base, dtype=difference_dtype, out=gated)
    np.multiply(gated, alpha[:, None, None], out=gated)
    np.add(base, gated, out=gated)
    return gated, alpha
