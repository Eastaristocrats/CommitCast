"""Dataset factory matching the public COSA repository convention."""

from __future__ import annotations

from pathlib import Path

from .loader import StreamRecord, discover_streams
from utils.misc import parse_csv


def _selected(value: str, allowed: set[str]) -> bool:
    return not allowed or str(value).lower() in allowed


def build_dataset(cfg, split: str = "test") -> list[StreamRecord]:
    """Discover and filter chronological frozen-forecast streams.

    CommitCast has no train/validation dataset because the upstream forecaster
    is frozen. ``split`` is retained to keep the same factory signature used by
    COSA-style repositories.
    """

    if "STREAM" not in cfg:
        from .forecasting import build_dataset as build_forecasting_dataset
        return build_forecasting_dataset(cfg, split)

    if split != "test":
        raise ValueError(
            "CommitCast only consumes exported test-time forecast streams; "
            f"unsupported split={split!r}"
        )
    records = discover_streams(Path(cfg.STREAM.ROOT), str(cfg.STREAM.GLOB))
    datasets = {item.lower() for item in parse_csv(cfg.DATA.NAME)}
    backbones = {item.lower() for item in parse_csv(cfg.MODEL.NAME)}
    horizons = set(parse_csv(cfg.DATA.PRED_LEN, int))
    kept = [
        record
        for record in records
        if _selected(record.dataset, datasets)
        and _selected(record.backbone, backbones)
        and (not horizons or record.horizon in horizons)
    ]
    maximum = int(cfg.STREAM.MAX_STREAMS)
    return kept[:maximum] if maximum > 0 else kept


def update_cfg_from_dataset(cfg, dataset_name: str) -> None:
    """COSA-compatible hook for a stream-filter dataset name.

    Stream shapes are read from the exported arrays, so there are no mutable
    dataset-specific channel counts to inject into the configuration.
    """

    if "STREAM" not in cfg:
        from .forecasting import update_cfg_from_dataset as update_forecasting_config
        return update_forecasting_config(cfg, dataset_name)

    if dataset_name:
        cfg.DATA.NAME = dataset_name
