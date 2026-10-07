"""Chronological frozen forecasts and checkpoint-backed data loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class StreamRecord:
    path: Path
    stream: str
    source: str = ""
    dataset: str = ""
    backbone: str = ""
    horizon: int = 0
    seed: str = ""


def _parse_stream_name(path: Path) -> StreamRecord:
    """Parse BASE_<dataset>_<backbone>_h<H>_seed<S>_* directory names."""

    name = path.parent.name
    parts = name.split("_")
    meta: dict[str, Any] = {
        "path": path,
        "stream": name,
        "source": parts[0] if parts else "",
        "dataset": "",
        "backbone": "",
        "horizon": 0,
        "seed": "",
    }
    index = len(parts) - 1
    if index >= 0 and parts[index].startswith("batch"):
        index -= 1
    while index >= 0 and parts[index].startswith("step"):
        index -= 1
    if index >= 0 and parts[index].startswith("seed"):
        meta["seed"] = parts[index][4:]
        index -= 1
    if index >= 0 and parts[index].startswith("h") and parts[index][1:].isdigit():
        meta["horizon"] = int(parts[index][1:])
        index -= 1
    if meta["source"] == "Foundation" and index >= 0 and parts[index] == "target":
        index -= 1
    if index >= 1:
        meta["backbone"] = parts[index]
        meta["dataset"] = "_".join(parts[1:index])
    return StreamRecord(**meta)


def discover_streams(root: Path, pattern: str = "*/adapter_stream.npz") -> list[StreamRecord]:
    root = Path(root)
    return [_parse_stream_name(path) for path in sorted(root.glob(pattern))]


def load_stream(record_or_path: StreamRecord | Path) -> tuple[np.ndarray, np.ndarray]:
    path = record_or_path.path if isinstance(record_or_path, StreamRecord) else Path(record_or_path)
    with np.load(path, allow_pickle=False) as archive:
        if "pred" not in archive or "true" not in archive:
            raise KeyError(f"{path} must contain arrays named 'pred' and 'true'")
        pred = archive["pred"].astype(np.float32, copy=False)
        true = archive["true"].astype(np.float32, copy=False)
    if pred.shape != true.shape or pred.ndim != 3:
        raise ValueError(
            f"{path}: pred and true must share shape [origin, lead, channel], "
            f"got {pred.shape} and {true.shape}"
        )
    if pred.shape[0] < 2 or pred.shape[1] < 2 or pred.shape[2] < 1:
        raise ValueError(f"{path}: stream is too small for commitment replay: {pred.shape}")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError(f"{path}: pred/true contain NaN or infinite values")
    if not np.array_equal(true[:-1, 1:], true[1:, :-1]):
        raise ValueError(f"{path}: target windows must overlap chronologically at stride 1")
    if isinstance(record_or_path, StreamRecord) and record_or_path.horizon not in (0, pred.shape[1]):
        raise ValueError(f"{path}: horizon metadata disagrees with the array")
    return pred, true



##########################################################################################
# Code is originally from the TAFAS (https://arxiv.org/pdf/2501.04970.pdf) implementation
# from https://github.com/kimanki/TAFAS by Kim et al. which is licensed under 
# Modified MIT License (Non-Commercial with Permission).
# You may obtain a copy of the License at
#
#    https://github.com/kimanki/TAFAS/blob/master/LICENSE
#
###########################################################################################

def construct_loader(cfg, split):
    from datasets.build import build_dataset
    if split == "train":
        batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = cfg.TRAIN.SHUFFLE
        drop_last = cfg.TRAIN.DROP_LAST
    elif split == "val":
        batch_size = cfg.VAL.BATCH_SIZE
        shuffle = cfg.VAL.SHUFFLE
        drop_last = cfg.VAL.DROP_LAST
    elif split == "test":
        batch_size = cfg.TEST.BATCH_SIZE
        shuffle = cfg.TEST.SHUFFLE
        drop_last = cfg.TEST.DROP_LAST
    else:
        raise ValueError

    dataset = build_dataset(cfg, split)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last
    )

    return loader


def get_train_dataloader(cfg):
    return construct_loader(cfg, "train")


def get_val_dataloader(cfg):
    return construct_loader(cfg, "val")


def get_test_dataloader(cfg):
    return construct_loader(cfg, "test")
