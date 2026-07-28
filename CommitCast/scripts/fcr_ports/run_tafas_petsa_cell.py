#!/usr/bin/env python
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# License scope: see the License section in README.md.
"""Causal FCR port of the official TAFAS/PETSA update operators.

This runner preserves each method's official calibration module, loss,
optimizer, per-cell hyperparameters, partial-prefix cadence, and fully-mature
replay update.  It changes only the service wrapper: an update is executed at
the first issue where all labels consumed by that update are legal, and the
candidate is then forecast from the *current* origin.  No adapted prediction
is written retroactively to an earlier issue.

The frozen Base prediction/target stream is supplied independently and is the
exact same-commit reference used by CommitCast and COSA-FCR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fcr_ports.provenance import PINNED_UPSTREAMS, verify_upstream  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("TAFAS", "PETSA"))
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--base_stream", type=Path)
    parser.add_argument("--output_json", type=Path)
    parser.add_argument("--cfg", type=Path)
    parser.add_argument("--service_batch_size", type=int, default=128)
    parser.add_argument(
        "--update_cadence",
        choices=("official_paas", "quarter_matched"),
        default="official_paas",
        help="Preserve official data-dependent PAAS by default; quarter_matched is a schedule-control variant.",
    )
    parser.add_argument("--max_origins", type=int, default=0)
    parser.add_argument("--self_test_only", action="store_true")
    parser.add_argument("--expected_commit", default="")
    parser.add_argument("--expected_module_sha256", default="")
    parser.add_argument("--expected_license_sha256", default="")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.opts and args.opts[0] == "--":
        args.opts = args.opts[1:]
    return args


def self_test() -> None:
    weights, segment, origins = latest_service_weights(n=3389, horizon=96)
    if segment != 24 or origins != 3317:
        raise AssertionError((segment, origins))
    if int(weights.sum()) * segment != origins * 96:
        raise AssertionError("service weights do not reconstruct exact support")
    if official_update_training_mode(prefix=24) is not False:
        raise AssertionError("partial replay must preserve official eval mode")
    if official_update_training_mode(prefix=None) is not True:
        raise AssertionError("fully matured replay must use official training mode")
    first = torch.tensor([1.0, 2.0])
    second = torch.tensor([3.0])
    if state_digest([("b", second), ("a", first)]) != state_digest(
        [("a", first), ("b", second)]
    ):
        raise AssertionError("state digest depends on input order")
    print(json.dumps({"status": "PASS", "checks": 4}, indent=2))


def patch_legacy_pandas_apply() -> None:
    """Compatibility-only shim for upstream TAFAS/PETSA on pandas>=2."""
    import pandas as pd

    original = pd.Series.apply

    def compatible(self, func, convert_dtype=None, args=(), *, by_row="compat", **kwargs):
        if isinstance(args, int) and not isinstance(convert_dtype, (tuple, list)):
            args = ()
        return original(self, func, args=args, by_row=by_row, **kwargs)

    pd.Series.apply = compatible


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_digest(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(items):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def lower_tail(values: Sequence[float], p: float) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return math.nan, math.nan, math.nan
    count = max(1, int(math.ceil(p * len(array))))
    ordered = np.sort(array)
    return float(np.quantile(array, p)), float(np.mean(ordered[:count])), float(ordered[0])


def latest_service_weights(n: int, horizon: int) -> tuple[np.ndarray, int, int]:
    if horizon <= 0 or horizon % 4:
        raise ValueError("canonical FCR profile requires a positive H divisible by four")
    segment = horizon // 4
    service_origins = n - 3 * segment
    if service_origins <= 0:
        raise ValueError(f"not enough origins for H={horizon}: N={n}")
    weights = np.zeros(n, dtype=np.int64)
    for commit_index in range(4):
        start = commit_index * segment
        weights[start : start + service_origins] += 1
    if int(weights.sum()) != 4 * service_origins:
        raise AssertionError("latest-service multiplicity does not reconstruct four owned segments")
    return weights, segment, service_origins


def official_update_training_mode(prefix: int | None) -> bool:
    """Mirror the released adapters: full replay trains; partial replay stays eval.

    ``eval()`` does not disable gradients.  The official TAFAS/PETSA code
    computes partial-prefix gradients while every module remains in evaluation
    mode, and switches modules to training mode only for a fully matured batch.
    """

    return prefix is None


def composite_petsa_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    # Intentionally mirrors the official PETSA implementation, including its
    # dimension choices, so the FCR port changes timing/ownership rather than
    # the learning objective.
    loss_freq = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)).abs().mean()
    loss = F.huber_loss(pred, target, delta=0.5) + alpha * loss_freq
    flattened_pred = pred.reshape(-1)
    flattened_target = target.reshape(-1)
    corr = torch.corrcoef(torch.stack([flattened_pred, flattened_target]))[0, 1]
    sf_pred = F.softmax(pred - pred.mean(dim=1, keepdim=True))
    sf_target = F.softmax(target - target.mean(dim=1, keepdim=True))
    loss_var = F.kl_div(sf_pred, sf_target).mean()
    loss_mean = F.l1_loss(
        pred.mean(dim=1, keepdim=True), target.mean(dim=1, keepdim=True)
    )
    return loss - corr + loss_var + loss_mean


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        repo = args.repo.resolve()
        sys.path.insert(0, str(repo))
        os.chdir(repo)
        patch_legacy_pandas_apply()

        # Imports must resolve from the selected official runtime repository.
        from datasets.build import update_cfg_from_dataset
        from datasets.loader import get_test_dataloader
        from models.build import build_model, build_norm_module, load_best_model
        from models.forecast import forecast
        from models.optimizer import get_optimizer
        from utils.misc import set_devices, set_seeds
        from utils.parser import load_config
        from config import get_norm_module_cfg

        namespace = argparse.Namespace(cfg_file=str(args.cfg.resolve()), opts=args.opts)
        cfg = load_config(namespace)
        update_cfg_from_dataset(cfg, cfg.DATA.NAME)
        cfg.TEST.BATCH_SIZE = min(max(1, args.service_batch_size), 256)
        cfg.TEST.SHUFFLE = False
        cfg.TEST.DROP_LAST = False
        cfg.DATA_LOADER.NUM_WORKERS = 0
        cfg.DATA_LOADER.PIN_MEMORY = False
        set_devices(cfg.VISIBLE_DEVICES)
        set_seeds(cfg.SEED)

        model = load_best_model(cfg, build_model(cfg))
        norm = build_norm_module(cfg) if cfg.NORM_MODULE.ENABLE else None
        if norm is not None:
            norm = load_best_model(get_norm_module_cfg(cfg), norm)
        frozen_model = deepcopy(model).requires_grad_(False).eval()
        frozen_norm = deepcopy(norm).requires_grad_(False).eval() if norm is not None else None
        method_module = __import__(f"tta.{args.method.lower()}", fromlist=["Calibration"])
        cali = method_module.Calibration(cfg).cuda()

        for module in (model, norm, cali):
            if module is not None:
                module.requires_grad_(False)
        cali.requires_grad_(True)

        self.cfg = cfg
        self.model = model
        self.norm = norm
        self.frozen_model = frozen_model
        self.frozen_norm = frozen_norm
        self.cali = cali
        self.forecast = forecast
        self.optimizer = get_optimizer(cali.parameters(), cfg.TTA)
        self.dataset = get_test_dataloader(cfg).dataset
        self.device = torch.device("cuda")
        self.backward_calls = 0
        self.optimizer_steps = 0
        self.partial_updates = 0
        self.full_updates = 0
        self.max_train_target_minus_issue = -10**9
        self.max_batch_last_target_minus_issue = -10**9
        self.periods: list[int] = []
        self.partial_mode_violations = 0
        self.full_mode_violations = 0
        self.post_update_eval_violations = 0
        steps = int(getattr(cfg.TTA, args.method).STEPS)
        if steps != 1:
            raise NotImplementedError(
                "the audited FCR port currently requires the official matrix setting STEPS=1"
            )
        self.steps_per_update = steps

    def cpu_inputs(self, start: int, end: int) -> tuple[torch.Tensor, ...]:
        columns = list(zip(*(self.dataset[index] for index in range(start, end))))
        return tuple(torch.as_tensor(np.stack(column), dtype=torch.float32) for column in columns)

    def device_inputs(self, start: int, end: int) -> tuple[torch.Tensor, ...]:
        return tuple(value.to(self.device, non_blocking=False) for value in self.cpu_inputs(start, end))

    def modules(self) -> list[torch.nn.Module]:
        return [module for module in (self.model, self.norm, self.cali) if module is not None]

    def audit_state_digest(self) -> str:
        """Hash only state a read-only forecast could mutate.

        Frozen backbone parameters are excluded because they are outside the
        optimizer and cannot change during an eval/no-grad forward.  Backbone
        buffers remain included, as do all calibration parameters/buffers.
        This keeps the invariant check strong without rehashing a large frozen
        checkpoint at every service batch.
        """
        items: list[tuple[str, torch.Tensor]] = []
        items.extend((f"cali.{name}", value) for name, value in self.cali.state_dict().items())
        items.extend((f"model_buffer.{name}", value) for name, value in self.model.named_buffers())
        if self.norm is not None:
            items.extend((f"norm_buffer.{name}", value) for name, value in self.norm.named_buffers())
        return state_digest(items)

    def official_period(self, index: int) -> int:
        enc_window = torch.as_tensor(self.dataset[index][0], dtype=torch.float32)
        fft_result = torch.fft.rfft(enc_window - enc_window.mean(dim=0), dim=0)
        amplitude = torch.abs(fft_result)
        power = torch.mean(amplitude.square(), dim=0)
        try:
            dominant = int(torch.argmax(amplitude[:, int(torch.argmax(power))]).item())
            period = int(enc_window.shape[0]) // dominant
        except (RuntimeError, ZeroDivisionError):
            period = 24
        period *= int(getattr(getattr(self.cfg.TTA, self.args.method), "PERIOD_N", 1))
        return max(1, period)

    def set_training(self, enabled: bool) -> None:
        for module in self.modules():
            module.train(enabled)

    def predict(self, start: int, end: int, *, grad: bool) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.device_inputs(start, end)
        if self.cfg.MODEL.NAME != "PatchTST":
            inputs = self.cali.input_calibration(inputs)
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            pred, target = self.forecast(self.cfg, inputs, self.model, self.norm)
            pred = self.cali.output_calibration(pred)
        return pred, target

    def predict_frozen(self, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.device_inputs(start, end)
        self.frozen_model.eval()
        if self.frozen_norm is not None:
            self.frozen_norm.eval()
        with torch.no_grad():
            return self.forecast(self.cfg, inputs, self.frozen_model, self.frozen_norm)

    def update(self, start: int, end: int, prefix: int | None, issue: int) -> None:
        horizon = int(self.cfg.DATA.PRED_LEN)
        if prefix is None:
            latest_target = (end - 1) + horizon - 1
            self.max_batch_last_target_minus_issue = max(
                self.max_batch_last_target_minus_issue, latest_target - issue
            )
            if latest_target >= issue:
                raise AssertionError("full update consumed an unreleased target")
        else:
            latest_target = start + prefix - 1
            self.max_train_target_minus_issue = max(
                self.max_train_target_minus_issue, latest_target - issue
            )
            if latest_target >= issue:
                raise AssertionError("partial update consumed an unreleased target")

        method_cfg = getattr(self.cfg.TTA, self.args.method)
        expected_training = official_update_training_mode(prefix)
        for _ in range(int(method_cfg.STEPS)):
            self.set_training(expected_training)
            observed = [module.training for module in self.modules()]
            if prefix is None:
                self.full_mode_violations += int(not all(observed))
            else:
                self.partial_mode_violations += int(any(observed))
            pred, target = self.predict(start, end, grad=True)
            if prefix is not None:
                pred = pred[0][:prefix]
                target = target[0][:prefix]
            loss = (
                F.mse_loss(pred, target)
                if self.args.method == "TAFAS"
                else composite_petsa_loss(pred, target, float(self.cfg.TTA.PETSA.LOSS_ALPHA))
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {self.args.method} loss at issue {issue}")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.backward_calls += 1
            self.optimizer.step()
            self.optimizer_steps += 1
            # Released implementations return to eval after each full update;
            # partial updates never leave eval in the first place.
            self.set_training(False)
            self.post_update_eval_violations += int(any(module.training for module in self.modules()))
        if prefix is None:
            self.full_updates += 1
        else:
            self.partial_updates += 1
        self.set_training(False)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    runner = Runner(args)
    with np.load(args.base_stream.resolve(), allow_pickle=False) as archive:
        base = np.asarray(archive["pred"], dtype=np.float32)
        true = np.asarray(archive["true"], dtype=np.float32)
    n = min(len(runner.dataset), len(base))
    if args.max_origins:
        n = min(n, args.max_origins)
    base = base[:n]
    true = true[:n]
    horizon = int(runner.cfg.DATA.PRED_LEN)
    channels = int(true.shape[-1])
    if base.shape != true.shape or true.shape[1] != horizon or n <= horizon:
        raise ValueError(f"bad Base stream shape {base.shape}/{true.shape}, H={horizon}")
    if len(runner.dataset) != len(base) and not args.max_origins:
        raise AssertionError(f"dataset/Base length mismatch {len(runner.dataset)} != {len(base)}")
    overlap = float(np.max(np.abs(true[:-1, 1:] - true[1:, :-1])))
    if overlap != 0.0:
        raise AssertionError(f"Base targets are not one-step overlapping: {overlap}")

    weights, segment, service_origins = latest_service_weights(n, horizon)

    events: dict[int, list[tuple[str, int, int, int | None]]] = defaultdict(list)
    start = 0
    while start < n:
        period = segment if args.update_cadence == "quarter_matched" else runner.official_period(start)
        runner.periods.append(period)
        batch_size = period + 1
        end = min(n, start + batch_size)
        partial_issue = start + period
        if partial_issue < n:
            events[partial_issue].append(("partial", start, end, min(period, horizon)))
        full_issue = (end - 1) + horizon
        if full_issue < n:
            events[full_issue].append(("full", start, end, None))
        start = end

    base_sse = literal_sse = anchored_sse = 0.0
    base_sae = literal_sae = anchored_sae = 0.0
    off_sse = on_event_sse = 0.0
    off_sae = on_event_sae = 0.0
    anchored_off_sse = anchored_on_event_sse = 0.0
    anchored_off_sae = anchored_on_event_sae = 0.0
    event_elements = 0
    per_origin_base = np.zeros(service_origins, dtype=np.float64)
    per_origin_literal = np.zeros(service_origins, dtype=np.float64)
    per_origin_anchored = np.zeros(service_origins, dtype=np.float64)
    lead_base_sse = np.zeros(4, dtype=np.float64)
    lead_literal_sse = np.zeros(4, dtype=np.float64)
    lead_anchored_sse = np.zeros(4, dtype=np.float64)
    lead_elements = np.zeros(4, dtype=np.int64)
    initial_live_delta_max = 0.0
    initial_literal_shared_max = 0.0
    initial_anchored_shared_max = 0.0
    initial_base_sse = 0.0
    initial_literal_sse = 0.0
    initial_anchored_sse = 0.0
    target_identity_max = 0.0
    forward_state_mutations = 0
    numerical_anchor_max_abs = 0.0
    shared_base_full_sse = 0.0
    live_base_full_sse = 0.0

    def anchored_prediction(
        start: int,
        method_pred: torch.Tensor,
        frozen_pred: torch.Tensor,
        *,
        record_base_diagnostic: bool = True,
    ) -> torch.Tensor:
        nonlocal numerical_anchor_max_abs, shared_base_full_sse, live_base_full_sse
        nonlocal initial_live_delta_max
        end = start + int(method_pred.shape[0])
        shared = torch.from_numpy(np.ascontiguousarray(base[start:end])).to(method_pred.device)
        truth = torch.from_numpy(np.ascontiguousarray(true[start:end])).to(method_pred.device)
        numerical_anchor_max_abs = max(
            numerical_anchor_max_abs,
            float(torch.max(torch.abs(shared - frozen_pred)).detach().cpu()),
        )
        if start == 0:
            initial_live_delta_max = max(
                initial_live_delta_max,
                float(torch.max(torch.abs(method_pred - frozen_pred)).detach().cpu()),
            )
        if record_base_diagnostic:
            shared_base_full_sse += float(torch.sum((shared.double() - truth.double()).square()).detach().cpu())
            live_base_full_sse += float(torch.sum((frozen_pred.double() - truth.double()).square()).detach().cpu())
        return shared + (method_pred - frozen_pred)

    def account(
        start: int,
        literal_tensor: torch.Tensor,
        anchored_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
    ) -> None:
        nonlocal base_sse, literal_sse, anchored_sse, base_sae, literal_sae, anchored_sae
        nonlocal initial_literal_shared_max, initial_anchored_shared_max
        nonlocal initial_base_sse, initial_literal_sse, initial_anchored_sse, target_identity_max
        literal = literal_tensor.detach().cpu().numpy().astype(np.float32)
        anchored = anchored_tensor.detach().cpu().numpy().astype(np.float32)
        target = target_tensor.detach().cpu().numpy().astype(np.float32)
        end = start + len(literal)
        target_identity_max = max(target_identity_max, float(np.max(np.abs(target - true[start:end]))))
        if start == 0:
            initial_literal_shared_max = max(
                initial_literal_shared_max, float(np.max(np.abs(literal - base[start:end])))
            )
            initial_anchored_shared_max = max(
                initial_anchored_shared_max, float(np.max(np.abs(anchored - base[start:end])))
            )
            initial_base_diff = base[start:end] - true[start:end]
            initial_literal_diff = literal - true[start:end]
            initial_anchored_diff = anchored - true[start:end]
            initial_base_sse = float(np.sum(initial_base_diff * initial_base_diff, dtype=np.float64))
            initial_literal_sse = float(np.sum(initial_literal_diff * initial_literal_diff, dtype=np.float64))
            initial_anchored_sse = float(np.sum(initial_anchored_diff * initial_anchored_diff, dtype=np.float64))
        for local, issue in enumerate(range(start, end)):
            weight = int(weights[issue])
            if weight:
                bdiff = base[issue, :segment] - true[issue, :segment]
                ldiff = literal[local, :segment] - true[issue, :segment]
                adiff = anchored[local, :segment] - true[issue, :segment]
                b_row_sse = float(np.sum(bdiff * bdiff, dtype=np.float64))
                l_row_sse = float(np.sum(ldiff * ldiff, dtype=np.float64))
                a_row_sse = float(np.sum(adiff * adiff, dtype=np.float64))
                base_sse += weight * b_row_sse
                literal_sse += weight * l_row_sse
                anchored_sse += weight * a_row_sse
                base_sae += weight * float(np.sum(np.abs(bdiff), dtype=np.float64))
                literal_sae += weight * float(np.sum(np.abs(ldiff), dtype=np.float64))
                anchored_sae += weight * float(np.sum(np.abs(adiff), dtype=np.float64))
                for k in range(4):
                    root = issue - k * segment
                    if 0 <= root < service_origins:
                        per_origin_base[root] += b_row_sse
                        per_origin_literal[root] += l_row_sse
                        per_origin_anchored[root] += a_row_sse
            for quartile in range(4):
                lo, hi = quartile * segment, (quartile + 1) * segment
                bq = base[issue, lo:hi] - true[issue, lo:hi]
                lq = literal[local, lo:hi] - true[issue, lo:hi]
                aq = anchored[local, lo:hi] - true[issue, lo:hi]
                lead_base_sse[quartile] += float(np.sum(bq * bq, dtype=np.float64))
                lead_literal_sse[quartile] += float(np.sum(lq * lq, dtype=np.float64))
                lead_anchored_sse[quartile] += float(np.sum(aq * aq, dtype=np.float64))
                lead_elements[quartile] += int(segment * channels)

    event_times = sorted(events)
    cursor = 0
    for event_index, issue in enumerate(event_times + [n]):
        if cursor < issue:
            runner.set_training(False)
            for start in range(cursor, issue, args.service_batch_size):
                end = min(issue, start + args.service_batch_size)
                before = runner.audit_state_digest()
                pred_live, target = runner.predict(start, end, grad=False)
                frozen_live, frozen_target = runner.predict_frozen(start, end)
                after = runner.audit_state_digest()
                forward_state_mutations += int(before != after)
                if not torch.equal(target, frozen_target):
                    raise AssertionError("adapted/frozen live targets differ")
                anchored = anchored_prediction(start, pred_live, frozen_live)
                account(start, pred_live, anchored, target)
        if issue == n:
            break

        # Exact pre-update fork for the current issue.  It is diagnostic only;
        # the served action below is always post-update.
        runner.set_training(False)
        pre_digest = runner.audit_state_digest()
        pred_off_live, target_off = runner.predict(issue, issue + 1, grad=False)
        frozen_live, frozen_target = runner.predict_frozen(issue, issue + 1)
        pred_off_anchored = anchored_prediction(issue, pred_off_live, frozen_live)
        if not torch.equal(target_off, frozen_target):
            raise AssertionError("update-off/frozen live targets differ")
        if runner.audit_state_digest() != pre_digest:
            raise AssertionError("update-off forecast mutated method state")
        for kind, start, end, prefix in sorted(events[issue], key=lambda item: item[0] != "full"):
            runner.update(start, end, prefix, issue)
        post_digest = runner.audit_state_digest()
        pred_on_live, target_on = runner.predict(issue, issue + 1, grad=False)
        pred_on_anchored = anchored_prediction(
            issue, pred_on_live, frozen_live, record_base_diagnostic=False
        )
        if not torch.equal(target_on, frozen_target):
            raise AssertionError("update-on/frozen live targets differ")
        if runner.audit_state_digest() != post_digest:
            raise AssertionError("update-on forecast mutated method state")
        account(issue, pred_on_live, pred_on_anchored, target_on)
        weight = int(weights[issue])
        if weight:
            off = pred_off_live.detach().cpu().numpy()[0, :segment] - true[issue, :segment]
            on = pred_on_live.detach().cpu().numpy()[0, :segment] - true[issue, :segment]
            anchored_off = pred_off_anchored.detach().cpu().numpy()[0, :segment] - true[issue, :segment]
            anchored_on = pred_on_anchored.detach().cpu().numpy()[0, :segment] - true[issue, :segment]
            off_sse += weight * float(np.sum(off * off, dtype=np.float64))
            on_event_sse += weight * float(np.sum(on * on, dtype=np.float64))
            off_sae += weight * float(np.sum(np.abs(off), dtype=np.float64))
            on_event_sae += weight * float(np.sum(np.abs(on), dtype=np.float64))
            anchored_off_sse += weight * float(np.sum(anchored_off * anchored_off, dtype=np.float64))
            anchored_on_event_sse += weight * float(np.sum(anchored_on * anchored_on, dtype=np.float64))
            anchored_off_sae += weight * float(np.sum(np.abs(anchored_off), dtype=np.float64))
            anchored_on_event_sae += weight * float(np.sum(np.abs(anchored_on), dtype=np.float64))
            event_elements += weight * segment * channels
        cursor = issue + 1

    elements = int(service_origins * horizon * channels)
    literal_gains = 100.0 * (per_origin_base - per_origin_literal) / np.maximum(per_origin_base, 1e-12)
    anchored_gains = 100.0 * (per_origin_base - per_origin_anchored) / np.maximum(per_origin_base, 1e-12)
    literal_q05 = float(np.quantile(literal_gains, 0.05))
    _, literal_cvar10, literal_worst = lower_tail(literal_gains, 0.10)
    anchored_q05 = float(np.quantile(anchored_gains, 0.05))
    _, anchored_cvar10, anchored_worst = lower_tail(anchored_gains, 0.10)
    result: dict[str, Any] = {
        "method": f"{args.method}-FCR",
        "primary_candidate": "literal official-calibrator output",
        "sensitivity_candidate": "shared-Base control-variate transport",
        "dataset": str(runner.cfg.DATA.NAME),
        "backbone": str(runner.cfg.MODEL.NAME),
        "horizon": horizon,
        "seed": int(runner.cfg.SEED),
        "n": n,
        "channels": channels,
        "service_origins": service_origins,
        "evaluated_elements": elements,
        "base_mse": base_sse / elements,
        "mse": literal_sse / elements,
        "base_mae": base_sae / elements,
        "mae": literal_sae / elements,
        "gain_vs_base_pct": 100.0 * (base_sse - literal_sse) / max(base_sse, 1e-12),
        "mae_gain_vs_base_pct": 100.0 * (base_sae - literal_sae) / max(base_sae, 1e-12),
        "origin_gain_mean_pct": float(np.mean(literal_gains)),
        "origin_gain_q05_pct": literal_q05,
        "origin_gain_cvar10_pct": literal_cvar10,
        "origin_gain_worst_pct": literal_worst,
        "origin_win_rate_pct": float(100.0 * np.mean(literal_gains > 1e-12)),
        "anchored_mse": anchored_sse / elements,
        "anchored_mae": anchored_sae / elements,
        "anchored_gain_vs_base_pct": 100.0 * (base_sse - anchored_sse) / max(base_sse, 1e-12),
        "anchored_mae_gain_vs_base_pct": 100.0 * (base_sae - anchored_sae) / max(base_sae, 1e-12),
        "anchored_origin_gain_mean_pct": float(np.mean(anchored_gains)),
        "anchored_origin_gain_q05_pct": anchored_q05,
        "anchored_origin_gain_cvar10_pct": anchored_cvar10,
        "anchored_origin_gain_worst_pct": anchored_worst,
        "anchored_origin_win_rate_pct": float(100.0 * np.mean(anchored_gains > 1e-12)),
        "partial_updates": runner.partial_updates,
        "full_updates": runner.full_updates,
        "backward_calls": runner.backward_calls,
        "optimizer_steps": runner.optimizer_steps,
        "steps_per_update": runner.steps_per_update,
        "paas_period_min": int(min(runner.periods)),
        "paas_period_median": float(np.median(runner.periods)),
        "paas_period_max": int(max(runner.periods)),
        "trainable_parameters": sum(p.numel() for p in runner.cali.parameters() if p.requires_grad),
        "max_partial_target_time_minus_issue": runner.max_train_target_minus_issue,
        "max_full_batch_target_time_minus_issue": runner.max_batch_last_target_minus_issue,
        "target_identity_max_abs": target_identity_max,
        "initial_live_delta_max_abs": initial_live_delta_max,
        "literal_initial_shared_base_max_abs": initial_literal_shared_max,
        "literal_initial_shared_base_sse_relative_difference": abs(initial_literal_sse - initial_base_sse) / max(initial_base_sse, 1e-12),
        "anchored_initial_shared_base_max_abs": initial_anchored_shared_max,
        "anchored_initial_shared_base_sse_relative_difference": abs(initial_anchored_sse - initial_base_sse) / max(initial_base_sse, 1e-12),
        "numerical_anchor_max_abs": numerical_anchor_max_abs,
        "literal_frozen_base_sse_relative_difference": abs(live_base_full_sse - shared_base_full_sse) / max(shared_base_full_sse, 1e-12),
        "forward_state_mutation_count": forward_state_mutations,
        "partial_update_training_mode_violation_count": runner.partial_mode_violations,
        "full_update_eval_mode_violation_count": runner.full_mode_violations,
        "post_update_eval_mode_violation_count": runner.post_update_eval_violations,
        "update_event_elements": event_elements,
        "update_on_gain_vs_update_off_pct": (
            100.0 * (off_sse - on_event_sse) / max(off_sse, 1e-12) if event_elements else math.nan
        ),
        "update_on_mae_gain_vs_update_off_pct": (
            100.0 * (off_sae - on_event_sae) / max(off_sae, 1e-12) if event_elements else math.nan
        ),
        "anchored_update_on_gain_vs_update_off_pct": (
            100.0 * (anchored_off_sse - anchored_on_event_sse) / max(anchored_off_sse, 1e-12)
            if event_elements
            else math.nan
        ),
        "anchored_update_on_mae_gain_vs_update_off_pct": (
            100.0 * (anchored_off_sae - anchored_on_event_sae) / max(anchored_off_sae, 1e-12)
            if event_elements
            else math.nan
        ),
        "current_full_h_lead_quartile_gain_pct": [
            100.0 * (b - m) / max(b, 1e-12)
            for b, m in zip(lead_base_sse.tolist(), lead_literal_sse.tolist())
        ],
        "current_full_h_lead_quartile_base_mse": (lead_base_sse / lead_elements).tolist(),
        "current_full_h_lead_quartile_method_mse": (lead_literal_sse / lead_elements).tolist(),
        "anchored_current_full_h_lead_quartile_gain_pct": [
            100.0 * (b - m) / max(b, 1e-12)
            for b, m in zip(lead_base_sse.tolist(), lead_anchored_sse.tolist())
        ],
        "anchored_current_full_h_lead_quartile_method_mse": (lead_anchored_sse / lead_elements).tolist(),
        "runtime_sec": time.perf_counter() - started,
        "protocol": "strict causal latest-origin FCR; official update modes; literal primary plus anchored sensitivity; no retroactive replacement",
        "schedule": "issue/c25/c50/c75 canonical service profile",
        "update_cadence": args.update_cadence,
        "service_batch_size": int(args.service_batch_size),
        "base_stream": f"{args.base_stream.parent.name}/adapter_stream.npz",
        "base_stream_sha256": sha256_file(args.base_stream.resolve()),
        "config_sha256": sha256_file(args.cfg.resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        **args.upstream_provenance,
    }
    if result["target_identity_max_abs"] != 0.0:
        raise AssertionError(f"target identity mismatch: {result['target_identity_max_abs']}")
    if result["initial_live_delta_max_abs"] > 1e-7:
        raise AssertionError(f"initial live adapter is not identity: {result['initial_live_delta_max_abs']}")
    if result["anchored_initial_shared_base_max_abs"] != 0.0:
        raise AssertionError("anchored zero-adaptation action is not exactly the shared Base")
    if result["forward_state_mutation_count"]:
        raise AssertionError("evaluation forward mutated state")
    if any(
        result[field]
        for field in (
            "partial_update_training_mode_violation_count",
            "full_update_eval_mode_violation_count",
            "post_update_eval_mode_violation_count",
        )
    ):
        raise AssertionError("official update-mode audit failed")
    return result


def main() -> None:
    args = parse_args()
    if args.self_test_only:
        self_test()
        return
    required = ("method", "repo", "base_stream", "output_json", "cfg")
    missing = [name for name in required if getattr(args, name) in (None, "")]
    if missing:
        raise SystemExit(f"missing required arguments: {', '.join(missing)}")
    launch_cwd = Path.cwd()
    for name in ("repo", "base_stream", "output_json", "cfg"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, (launch_cwd / value).resolve())
    pinned = PINNED_UPSTREAMS[args.method]
    args.upstream_provenance = verify_upstream(
        args.repo,
        project=args.method,
        expected_commit=args.expected_commit or pinned["commit"],
        expected_module_sha256=(
            args.expected_module_sha256 or pinned["module_sha256"]
        ),
        expected_license_sha256=(
            args.expected_license_sha256 or pinned["license_sha256"]
        ),
    )
    result = run(args)
    atomic_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
