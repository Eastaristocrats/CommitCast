"""Frozen black-box forecast repository exposed as the base model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from datasets.loader import StreamRecord, file_sha256, load_stream


@dataclass(frozen=True)
class ForecastStream:
    record: StreamRecord
    pred: np.ndarray
    true: np.ndarray
    sha256: str


class FrozenForecastModel:
    """Read-only model facade over chronological frozen forecasts.

    This is the black-box counterpart of COSA's checkpoint-backed forecasting
    model: ``pred[i]`` is the output produced by the frozen checkpoint at origin
    ``i``. CommitCast never modifies it.
    """

    def __init__(self, records: list[StreamRecord]):
        self.records = tuple(records)
        self.training = False

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        if mode:
            raise RuntimeError("FrozenForecastModel cannot enter training mode")
        return self.eval()

    def streams(self):
        for record in self.records:
            pred, true = load_stream(record)
            yield ForecastStream(
                record=record,
                pred=pred,
                true=true,
                sha256=file_sha256(record.path),
            )

    def count_parameters(self) -> int:
        return 0

