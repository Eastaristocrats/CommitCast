from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import get_cfg_defaults
from models.build import build_model, load_best_model
from predictor import Predictor
from trainer import build_trainer
from tta.commitcast import build_adapter


def _write_stream(root: Path) -> None:
    rng = np.random.default_rng(5)
    stream_dir = root / "BASE_ETTh1_DLinear_h8_seed0_batch4"
    stream_dir.mkdir(parents=True)
    true = rng.normal(size=(32, 8, 2)).astype(np.float32)
    pred = true + rng.normal(scale=0.2, size=true.shape).astype(np.float32)
    np.savez_compressed(stream_dir / "adapter_stream.npz", pred=pred, true=true)


def test_cosa_style_build_adapt_predict_pipeline(tmp_path: Path):
    stream_root = tmp_path / "streams"
    result_dir = tmp_path / "results"
    _write_stream(stream_root)

    cfg = get_cfg_defaults()
    cfg.DEVICE = "cpu"
    cfg.STREAM.ROOT = str(stream_root)
    cfg.RESULT_DIR = str(result_dir)
    cfg.TTA.COMMITCAST.BLOCK_SIZE = 2
    cfg.TTA.COMMITCAST.WARMUP_MATURED = 1
    cfg.TTA.COMMITCAST.SRS_WARMUP_ROWS = 1
    cfg.TTA.COMMITCAST.TRAJ_POINTS = 3

    model = load_best_model(cfg, build_model(cfg))
    assert model.count_parameters() == 0
    adapter = build_adapter(cfg, model)
    assert len(adapter.adapt()) == 1
    assert adapter.count_parameters() == 0
    Predictor(cfg, model, adapter=adapter).predict()

    summary = pd.read_csv(result_dir / "summary.csv")
    assert set(summary["method"]) == {"Base", "CommitCast-Raw", "CommitCast"}
    assert (result_dir / "event_metrics.csv").exists()
    assert (result_dir / "stream_manifest.json").exists()


def test_training_boundary_is_explicit(tmp_path: Path):
    stream_root = tmp_path / "streams"
    _write_stream(stream_root)
    cfg = get_cfg_defaults()
    cfg.STREAM.ROOT = str(stream_root)
    model = build_model(cfg)
    with pytest.raises(RuntimeError, match="does not train"):
        build_trainer(cfg, model).train()

