"""COSA-compatible trainer boundary for a frozen black-box method."""

from __future__ import annotations


class Trainer:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model

    def train(self) -> None:
        raise RuntimeError(
            "CommitCast does not train or fine-tune the upstream forecaster. "
            "Use main.py --method cosa --protocol train --cfg <checkpoint-config>, "
            "then --protocol export to create a chronological adapter_stream.npz."
        )


def build_trainer(cfg, model) -> Trainer:
    return Trainer(cfg, model)

