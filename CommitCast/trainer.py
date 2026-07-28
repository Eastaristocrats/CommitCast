"""COSA-compatible trainer boundary for a frozen black-box method."""

from __future__ import annotations


class Trainer:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model

    def train(self) -> None:
        raise RuntimeError(
            "CommitCast does not train or fine-tune the upstream forecaster. "
            "Train the backbone in its native repository, then export a "
            "chronological adapter_stream.npz file."
        )


def build_trainer(cfg, model) -> Trainer:
    return Trainer(cfg, model)

