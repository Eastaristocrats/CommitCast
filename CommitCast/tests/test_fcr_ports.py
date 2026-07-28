from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.fcr_ports import run_cosa
from scripts.fcr_ports.run_matrix import read_manifest
from scripts.fcr_ports.run_tafas_petsa_cell import (
    latest_service_weights,
    official_update_training_mode,
)
from tta.ports import PORT_NAMES, build_port, build_port_command


class _ToyCosaAdapter(torch.nn.Module):
    def __init__(self, **_: object) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, prediction: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        del context
        return prediction + self.bias


def test_port_registry_is_explicit_and_external():
    assert PORT_NAMES == ("COSA-FCR", "TAFAS-FCR", "PETSA-FCR")
    assert build_port("cosa").upstream_project == "COSA"
    assert build_port("petsa_fcr").name == "PETSA-FCR"
    with pytest.raises(KeyError):
        build_port("unknown")


def test_port_commands_preserve_cosa_style_runner_boundary():
    cosa = build_port_command(
        "COSA-FCR",
        repo="external/COSA",
        stream_root="streams",
        output_dir="results/cosa",
    )
    assert cosa[0] == sys.executable
    assert Path(cosa[1]).parts[-3:] == ("scripts", "fcr_ports", "run_cosa.py")
    assert "--repo" in cosa and "--stream_root" in cosa

    tafas = build_port_command(
        "TAFAS-FCR",
        repo="external/TAFAS",
        cfg="config.yaml",
        base_stream="stream.npz",
        output_json="cell.json",
    )
    assert Path(tafas[1]).parts[-3:] == (
        "scripts",
        "fcr_ports",
        "run_tafas_petsa_cell.py",
    )
    assert tafas[2:4] == ["--method", "TAFAS"]


def test_port_command_rejects_incomplete_external_contract():
    with pytest.raises(ValueError, match="stream_root"):
        build_port_command("COSA-FCR", repo="external/COSA", output_dir="results")
    with pytest.raises(ValueError, match="cfg"):
        build_port_command(
            "PETSA-FCR",
            repo="external/PETSA",
            base_stream="stream.npz",
            output_json="cell.json",
        )


def test_tafas_petsa_service_support_and_modes():
    weights, segment, origins = latest_service_weights(n=3389, horizon=96)
    assert segment == 24
    assert origins == 3317
    assert int(weights.sum()) * segment == origins * 96
    assert official_update_training_mode(prefix=24) is False
    assert official_update_training_mode(prefix=None) is True


def test_cosa_strict_maturity_is_future_label_invariant(monkeypatch):
    monkeypatch.setattr(run_cosa, "SimpleOutputAdapter", _ToyCosaAdapter)
    rng = np.random.default_rng(42)
    n, horizon, channels = 32, 8, 2
    prediction = rng.normal(size=(n, horizon, channels)).astype(np.float32)
    timeline = rng.normal(size=(n + horizon - 1, channels)).astype(np.float32)
    truth = np.stack([timeline[index : index + horizon] for index in range(n)])
    altered_timeline = timeline.copy()
    cutoff = 19
    altered_timeline[cutoff:] += 4.0
    altered_truth = np.stack(
        [altered_timeline[index : index + horizon] for index in range(n)]
    )
    kwargs = {
        "device": torch.device("cpu"),
        "context_size": 3,
        "update_batch_size": 4,
        "serve_batch_size": 5,
        "steps": 1,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "seed": 0,
    }
    served, audit = run_cosa.strict_serve(prediction, truth, **kwargs)
    altered, altered_audit = run_cosa.strict_serve(
        prediction, altered_truth, **kwargs
    )
    assert np.max(np.abs(served[: cutoff + 1] - altered[: cutoff + 1])) <= 1e-7
    assert audit["prequential_violation_count"] == 0
    assert audit["context_violation_count"] == 0
    assert altered_audit["prequential_violation_count"] == 0
    assert altered_audit["context_violation_count"] == 0


def test_example_matrix_manifest_covers_all_ports():
    manifest = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "fcr_ports_manifest.example.yaml"
    )
    rows = read_manifest(manifest)
    assert {row["method"] for row in rows} == set(PORT_NAMES)
    assert len({row["cell_id"] for row in rows}) == 3
