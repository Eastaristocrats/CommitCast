"""Chronological frozen-forecast stream loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


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

