"""Base-model factory following COSA's ``models.build`` interface."""

from __future__ import annotations

from datasets.build import build_dataset
from .forecast import FrozenForecastModel


def build_model(cfg) -> FrozenForecastModel:
    records = build_dataset(cfg, split="test")
    if not records:
        raise FileNotFoundError(
            f"No streams matching {cfg.STREAM.GLOB!r} under {cfg.STREAM.ROOT}"
        )
    return FrozenForecastModel(records)


def load_best_model(cfg, model: FrozenForecastModel) -> FrozenForecastModel:
    """Return the already-frozen model facade.

    ``TRAIN.CHECKPOINT_DIR`` is accepted as provenance metadata. Forecast
    execution must be exported into ``adapter_stream.npz`` before replay.
    """

    if not bool(cfg.MODEL.FROZEN):
        raise ValueError("CommitCast requires MODEL.FROZEN=True")
    return model.eval()

