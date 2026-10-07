"""Train, export and evaluate the shared forecasting models and official adapters."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["tafas", "petsa", "cosa"], required=True)
    p.add_argument("--protocol", choices=["train", "native", "fcr", "export"], default="fcr")
    p.add_argument("--cfg", type=Path, required=True)
    p.add_argument("--checkpoint-config", type=Path,
        help="Import checkpoint-owned DATA/MODEL/normalization settings before CLI overrides")
    p.add_argument("--profile", choices=["published", "config"], default="published",
        help="Use published TAFAS/PETSA per-cell LR/WD (and TAFAS gate); explicit CLI values win")
    p.add_argument("--base-stream", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--schedules", default="default")
    p.add_argument("--support", choices=["common", "schedule"], default="common")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--event-threshold", type=Path, help="Validation-frozen event threshold JSON")
    p.add_argument("opts", nargs=argparse.REMAINDER, help="Official KEY VALUE overrides")
    return p


def json_safe(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def dump(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")


def main():
    args = parser().parse_args()
    sys.path.insert(0, str(ROOT))
    from fcr.profiles import published_overrides
    from config import get_cfg_defaults, get_norm_module_cfg

    cfg = get_cfg_defaults(args.method)
    cfg.merge_from_file(str(args.cfg))
    if args.checkpoint_config:
        import yaml
        from yacs.config import CfgNode
        trained = yaml.safe_load(args.checkpoint_config.read_text(encoding="utf-8"))
        # Old experiment configs contain unrelated TTA extensions. Only the
        # checkpoint's forecasting architecture and scaler identity transfer.
        for key in ("MODEL", "DATA", "NORMALIZE", "NORM_MODULE", "REVIN", "SAN", "DISHTS", "SEED"):
            if key not in trained:
                continue
            if key not in cfg:
                if key == "NORMALIZE" and trained[key] == "NST" and not trained.get("NORM_MODULE", {}).get("ENABLE", False):
                    continue  # Current COSA represents NST through NORM_MODULE.ENABLE=False.
                if key == "DISHTS" and trained.get("NORM_MODULE", {}).get("NAME") != "DishTS":
                    continue  # Inactive legacy configuration, never a loaded module.
                raise ValueError(f"Checkpoint uses an unsupported forecasting component: {key}")
            value = deepcopy(trained[key])
            if key == "DATA":
                value["BASE_DIR"] = cfg.DATA.BASE_DIR
            if key in ("REVIN", "SAN", "DISHTS"):
                value.pop("RESULT_DIR", None)
                value.pop("TRAIN", None)
            cfg.merge_from_other_cfg(CfgNode({key:value}))
        cfg.TRAIN.CHECKPOINT_DIR = str(args.checkpoint_config.resolve().parent)
    opts = args.opts[1:] if args.opts and args.opts[0] == "--" else args.opts
    cfg.merge_from_list(opts)
    if args.profile == "published":
        cfg.merge_from_list(published_overrides(args.method,cfg.DATA.NAME,cfg.MODEL.NAME,cfg.DATA.PRED_LEN))
        cfg.merge_from_list(opts)
    if args.protocol == "train":
        cfg.TRAIN.ENABLE = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.VISIBLE_DEVICES)
    import numpy as np
    import pandas as pd
    import torch
    from datasets.build import update_cfg_from_dataset
    from models.build import build_model, build_norm_module, load_best_model
    from models.forecast import forecast
    from utils.misc import prepare_inputs, set_seeds
    from fcr.native import NativeSession, query_snapshot, OneBatchLoader
    from fcr.scoring import SCHEDULES, common_origins, score_plan, PublicationLedger, event_boundaries

    if not torch.cuda.is_available():
        raise RuntimeError("The unchanged official adapters require CUDA; CommitCast replay supports CPU")
    if args.protocol == "fcr" and cfg.TRAIN.ENABLE:
        raise ValueError("Train/select checkpoints before starting an FCR replay")
    output = (args.output or Path(cfg.RESULT_DIR)).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new result directory: {output}")
    declared_ratios = (cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO)
    update_cfg_from_dataset(cfg, cfg.DATA.NAME)
    cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = declared_ratios
    for key in ("seq_len", "label_len", "pred_len"):
        cfg.MODEL[key] = cfg.DATA[key.upper()]
    cfg.DATA_LOADER.NUM_WORKERS = 0
    cfg.TEST.SHUFFLE = False
    cfg.TEST.DROP_LAST = False
    cfg.RESULT_DIR = str(output)
    set_seeds(int(cfg.SEED))
    model = build_model(cfg)
    norm = build_norm_module(cfg) if cfg.NORM_MODULE.ENABLE else None
    if cfg.TRAIN.ENABLE:
        from trainer import build_trainer
        Path(cfg.TRAIN.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.yaml").write_text(cfg.dump(), encoding="utf-8")
        build_trainer(cfg, model, norm_module=norm).train()
    checkpoint = Path(cfg.TRAIN.CHECKPOINT_DIR) / "checkpoint_best.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"A trained checkpoint is required: {checkpoint}")
    model = load_best_model(cfg, model)
    if norm is not None:
        nc = get_norm_module_cfg(cfg)
        if not (Path(nc.TRAIN.CHECKPOINT_DIR) / "checkpoint_best.pth").is_file():
            raise FileNotFoundError("Missing normalization checkpoint")
        norm = load_best_model(nc, norm)
    if args.protocol == "train":
        dump(output / "training.json", dict(status="complete", method=args.method,
            checkpoint=str(checkpoint), trainer="shared TAFAS training implementation"))
        print(checkpoint)
        return
    model.eval()
    if norm is not None:
        norm.eval()
    frozen_model, frozen_norm = deepcopy((model, norm))
    module = importlib.import_module("tta." + args.method)
    # Validation uses the identical official dataset class and training scaler.
    if args.split == "val":
        from datasets.build import build_dataset
        from torch.utils.data import DataLoader
        from fcr.splits import NativeSplitView
        dataset = NativeSplitView(build_dataset(cfg, split="val"), "val")
        module.get_test_dataloader = lambda conf: DataLoader(dataset, batch_size=len(dataset), shuffle=False, drop_last=False, num_workers=0)
    adapter = module.build_adapter(cfg, model, norm_module=norm)
    raw = list(adapter.test_loader)
    if len(raw) != 1:
        raise ValueError("Official adapter did not provide a concatenated input batch")
    inputs = prepare_inputs(raw[0])
    n, h = len(inputs[0]), int(cfg.DATA.PRED_LEN)
    truth = inputs[2][:, -h:, cfg.DATA.TARGET_START_IDX:].detach().cpu().numpy()
    if not np.array_equal(truth[:-1, 1:], truth[1:, :-1]):
        raise ValueError("Dataset targets do not form a chronological stride-1 stream")
    base_parts = []
    with torch.no_grad():
        for lo in range(0, n, 128):
            batch = tuple(x[lo:lo+128].clone() for x in inputs)
            batch[2][:, -h:] = 0
            base_parts.append(forecast(cfg, batch, frozen_model, frozen_norm)[0].cpu().numpy())
    base = np.concatenate(base_parts)
    if not np.isfinite(base).all() or not np.isfinite(truth).all():
        raise ValueError("Forecasts and targets must be finite on the complete stream")
    if args.base_stream is not None:
        with np.load(args.base_stream, allow_pickle=False) as src:
            archived, target = src["pred"], src["true"]
            if args.event_threshold is not None or "event50" in args.schedules.split(","):
                if not {'values','context_length'} <= set(src.files):
                    raise ValueError("Event comparisons require the shared observed-input timeline")
                context = inputs[0][0,:,cfg.DATA.TARGET_START_IDX:].detach().cpu().numpy()
                observed = np.concatenate([context,truth[:,0],truth[-1,1:]],axis=0)
                if src['context_length'].item() != len(context) or not np.array_equal(src['values'],observed):
                    raise ValueError("Baseline and CommitCast do not share identical event inputs")
        if target.shape != truth.shape or not np.array_equal(target, truth):
            raise ValueError("Baseline and CommitCast do not share identical target support")
        if archived.shape != base.shape or not np.allclose(archived, base, atol=1e-4, rtol=1e-4):
            raise ValueError("Frozen checkpoint/context/scaler does not match the shared Base stream")
        base = archived
    elif args.protocol == "fcr":
        raise ValueError("FCR requires --base-stream to enforce a shared frozen reference")

    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(cfg.dump(), encoding="utf-8")
    if args.protocol == "export":
        prehistory = inputs[0][0, :, cfg.DATA.TARGET_START_IDX:].detach().cpu().numpy()
        values = np.concatenate([prehistory, truth[:, 0], truth[-1, 1:]], axis=0)
        np.savez_compressed(output / "adapter_stream.npz", pred=base, true=truth,
            values=values, context_length=np.int64(len(prehistory)))
        dump(output / "metadata.json", dict(split=args.split, shape=list(base.shape), seed=int(cfg.SEED),
            dataset=cfg.DATA.NAME, backbone=cfg.MODEL.NAME, stride=1,
            train_ratio=cfg.DATA.TRAIN_RATIO, test_ratio=cfg.DATA.TEST_RATIO,
            input_context=cfg.DATA.SEQ_LEN, normalization="training-split scaler; official window normalization"))
        print(output / "adapter_stream.npz")
        return
    if args.protocol == "native":
        started = time.perf_counter()
        adapter.adapt()
        dump(output / "native_metrics.json", dict(method=args.method, source="official adapters in tta/",
            protocol="official native full-window scoring", split=args.split, requests=n,
            mse=float(np.mean(adapter.mse_all)), mae=float(np.mean(adapter.mae_all)),
            elapsed_sec=time.perf_counter()-started, adaptations=int(adapter.n_adapt)))
        return

    names = list(SCHEDULES) if args.schedules == "all" else args.schedules.split(",")
    if args.event_threshold is not None and "event50" not in names:
        names.append("event50")
    if not names or len(names) != len(set(names)) or any(x not in {*SCHEDULES, "event50"} for x in names):
        raise ValueError("Unknown or duplicate schedule")
    if "event50" in names and args.event_threshold is None:
        raise ValueError("event50 needs a validation-frozen threshold")
    support = list(SCHEDULES.values()) if args.support == "common" else [SCHEDULES.get(x, SCHEDULES["dense8"]) for x in names]
    count = common_origins(n, h, support)
    family = args.method.upper()
    # Official unclocked diagnostic CSVs are replaced by legal publication traces.
    if family == "COSA":
        adapter.save_csv = False
        adapter.save_paas_csv = False
    session = NativeSession(adapter, family, inputs=inputs)
    current = np.empty_like(base)
    ledger = PublicationLedger()
    publication_index = 0
    state_revision = -1
    readonly_query = None
    started = time.perf_counter()
    for q in range(n):
        session.advance(q)
        if session.revision != state_revision:
            readonly_query = query_snapshot(adapter, family)
            state_revision = session.revision
        current[q] = readonly_query(tuple(x[q:q+1] for x in inputs))[0]
        for event in session.publications[publication_index:]:
            for row, origin in enumerate(range(event["start"], event["end"])):
                ledger.publish(origin, event["clock"], event["prediction"][row])
        publication_index = len(session.publications)
    elapsed = time.perf_counter()-started
    # Finish native tail statements only at their true observation clock. These
    # late events cannot change already scored commitments or current queries.
    session.advance(n + h)
    metrics, request_rows = [], []
    for name in names:
        fractions = SCHEDULES.get(name, SCHEDULES["dense8"])
        plans = None
        if name == "event50":
            if args.event_threshold is None:
                raise ValueError("event50 needs a validation-frozen threshold")
            locked = json.loads(args.event_threshold.read_text(encoding="utf-8"))
            if locked.get("selected_on") != "val" or locked.get("rule") != "observed_rms_24":
                raise ValueError("Invalid event threshold provenance")
            prehistory = inputs[0][0, :, cfg.DATA.TARGET_START_IDX:].detach().cpu().numpy()
            timeline = np.concatenate([prehistory, truth[:, 0], truth[-1, 1:]], axis=0)
            plans = event_boundaries(timeline, h, count, locked["threshold"], context_length=len(prehistory))
        for policy in ("current", "native_revisions"):
            covered_atoms = [0]
            def at(request, a, b):
                fallback = current[request+a, :b-a]
                if policy == "current":
                    return fallback
                value, covered = ledger.select(request, request+a, a, b, fallback)
                covered_atoms[0] += int(covered.sum())
                return value
            summary, rows = score_plan(base, truth, fractions, count, at, request_boundaries=plans)
            summary.update(method=family, policy=policy, schedule=name, split=args.split,
                support=args.support, native_revision_atoms=covered_atoms[0],
                native_revision_coverage=covered_atoms[0]/summary["evaluated_elements"],
                fallback="current adapted forecast" if policy == "native_revisions" else "none")
            metrics.append(summary)
            request_rows.extend(dict(method=family, policy=policy, schedule=name, **r) for r in rows)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(request_rows).to_csv(output / "request_metrics.csv", index=False)
    np.save(output / "current_predictions.npy", current)
    publication_arrays = {f"batch_{i}": e["prediction"] for i, e in enumerate(session.publications)}
    np.savez_compressed(output / "native_publications.npz", **publication_arrays)
    dump(output / "publications.json", [{k:v for k,v in e.items() if k != "prediction"} for e in session.publications])
    dump(output / "run.json", dict(status="complete", method=family, protocol="FCR complete-H",
        split=args.split, support=args.support, count=count, horizon=h, channels=truth.shape[-1],
        schedules=names, transport=session.proof, optimizer_steps=session.step_events,
        offline_tail_updates=sum(e["clock"] >= n for e in session.step_events),
        elapsed_sec=elapsed, timing_scope="transport, queries and copying; excludes setup/export/scoring",
        capability_claim="official learning and revision paths retained; performance is measured, not guaranteed"))
    print(output)


if __name__ == "__main__":
    main()
