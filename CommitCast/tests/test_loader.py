from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from datasets.loader import discover_streams, load_stream


def test_discovery_and_loading(tmp_path: Path):
    stream_dir = tmp_path / "BASE_ETTh1_DLinear_h8_seed0_batch4"
    stream_dir.mkdir()
    path = stream_dir / "adapter_stream.npz"
    pred = np.zeros((12, 8, 2), dtype=np.float32)
    true = np.ones_like(pred)
    np.savez_compressed(path, pred=pred, true=true)

    records = discover_streams(tmp_path)
    assert len(records) == 1
    assert records[0].dataset == "ETTh1"
    assert records[0].backbone == "DLinear"
    assert records[0].horizon == 8
    loaded_pred, loaded_true = load_stream(records[0])
    np.testing.assert_array_equal(loaded_pred, pred)
    np.testing.assert_array_equal(loaded_true, true)


def test_discovery_supports_minimal_documented_name(tmp_path: Path):
    stream_dir = tmp_path / "BASE_ETTh1_DLinear_h96"
    stream_dir.mkdir()
    np.savez_compressed(
        stream_dir / "adapter_stream.npz",
        pred=np.zeros((12, 96, 1), dtype=np.float32),
        true=np.zeros((12, 96, 1), dtype=np.float32),
    )
    record = discover_streams(tmp_path)[0]
    assert record.dataset == "ETTh1"
    assert record.backbone == "DLinear"
    assert record.horizon == 96
    assert record.seed == ""


def test_loader_rejects_misaligned_arrays(tmp_path: Path):
    path = tmp_path / "adapter_stream.npz"
    np.savez_compressed(
        path,
        pred=np.zeros((12, 8, 2), dtype=np.float32),
        true=np.zeros((12, 7, 2), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="must share shape"):
        load_stream(path)
