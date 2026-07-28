from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.fcr_ports import run_cosa
from scripts.fcr_ports.generate_manifest import build_manifest, load_yaml
from scripts.fcr_ports.provenance import PINNED_UPSTREAMS, verify_upstream
from scripts.fcr_ports.run_matrix import is_complete, read_manifest
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


def test_cosa_metadata_accepts_documented_minimal_stream_name(tmp_path):
    path = tmp_path / "BASE_ETTh1_DLinear_h96" / "adapter_stream.npz"
    metadata = run_cosa.stream_metadata(path, None)
    assert metadata["dataset"] == "ETTh1"
    assert metadata["backbone"] == "DLinear"
    assert metadata["horizon"] == 96
    assert metadata["seed"] == 0
    full_path = (
        tmp_path
        / "BASE_exchange_rate_iTransformer_h336_seed7_batch48"
        / "adapter_stream.npz"
    )
    full_metadata = run_cosa.stream_metadata(full_path, None)
    assert full_metadata["dataset"] == "exchange_rate"
    assert full_metadata["backbone"] == "iTransformer"
    assert full_metadata["horizon"] == 336
    assert full_metadata["seed"] == 7


def test_example_matrix_manifest_covers_all_ports():
    manifest = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "fcr_ports_manifest.example.yaml"
    )
    rows = read_manifest(manifest)
    assert {row["method"] for row in rows} == set(PORT_NAMES)
    assert len({row["cell_id"] for row in rows}) == 3


def test_generated_public_manifest_is_complete_and_path_portable(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = load_yaml(root / "configs" / "fcr_ports_experiment.yaml")
    hparams = load_yaml(root / "configs" / "fcr_ports_hparams.yaml")
    manifest_path = tmp_path / "manifest.yaml"
    payload = build_manifest(
        spec,
        hparams,
        manifest_path=manifest_path,
        external_root=tmp_path / "external",
        data_root=tmp_path / "data",
        streams_root=tmp_path / "streams",
        checkpoints_root=tmp_path / "checkpoints",
        results_root=tmp_path / "results",
        device="cuda:0",
        seeds=[0],
    )
    assert payload["settings_per_seed"] == 144
    assert payload["expected_cells"] == 432
    assert len(payload["cells"]) == 432
    assert {
        method: sum(row["method"] == method for row in payload["cells"])
        for method in PORT_NAMES
    } == {method: 144 for method in PORT_NAMES}
    for row in payload["cells"]:
        for key in ("repo", "stream_root", "output_dir", "cfg", "base_stream", "output_json"):
            if key in row:
                assert not Path(row[key]).is_absolute()

    override = next(
        row
        for row in payload["cells"]
        if row["cell_id"] == "petsa_fcr_weather_patchtst_h336_seed0"
    )
    lr_index = override["extra_args"].index("TTA.SOLVER.BASE_LR")
    assert override["extra_args"][lr_index + 1] == "0.0002"


def test_upstream_provenance_requires_frozen_clean_identity(tmp_path, monkeypatch):
    pinned = PINNED_UPSTREAMS["COSA"]
    (tmp_path / ".git").mkdir()
    module = tmp_path / pinned["module"]
    module.parent.mkdir()
    module.write_text("placeholder", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("placeholder", encoding="utf-8")

    def fake_git(repo, *args):
        del repo
        return pinned["commit"] if args == ("rev-parse", "HEAD") else ""

    monkeypatch.setattr("scripts.fcr_ports.provenance._git", fake_git)
    monkeypatch.setattr(
        "scripts.fcr_ports.provenance.sha256_file",
        lambda path: (
            pinned["license_sha256"]
            if path.name == "LICENSE"
            else pinned["module_sha256"]
        ),
    )
    result = verify_upstream(
        tmp_path,
        project="COSA",
        expected_commit=pinned["commit"],
        expected_module_sha256=pinned["module_sha256"],
        expected_license_sha256=pinned["license_sha256"],
    )
    assert result["upstream_commit"] == pinned["commit"]
    assert result["upstream_verification"] == "git_commit_clean_and_content_hash"
    with pytest.raises(ValueError, match="pinned identity"):
        verify_upstream(
            tmp_path,
            project="COSA",
            expected_commit="0" * 40,
            expected_module_sha256=pinned["module_sha256"],
            expected_license_sha256=pinned["license_sha256"],
        )


def test_resume_requires_audited_result_content(tmp_path):
    spec = build_port("TAFAS-FCR")
    output = tmp_path / "cell.json"
    row = {
        "cell_id": "cell",
        "method": "TAFAS-FCR",
        "repo": "external/TAFAS",
        "output_json": "cell.json",
    }
    result = {
        "method": "TAFAS-FCR",
        "evaluated_elements": 96,
        "forward_state_mutation_count": 0,
        "partial_update_training_mode_violation_count": 0,
        "full_update_eval_mode_violation_count": 0,
        "post_update_eval_mode_violation_count": 0,
        "target_identity_max_abs": 0.0,
        "initial_live_delta_max_abs": 0.0,
        "max_partial_target_time_minus_issue": -1,
        "max_full_batch_target_time_minus_issue": -1,
        "upstream_commit": spec.upstream_commit,
        "upstream_module_sha256": spec.module_sha256,
    }
    import json

    output.write_text(json.dumps(result), encoding="utf-8")
    assert is_complete(row, tmp_path)
    result["forward_state_mutation_count"] = 1
    output.write_text(json.dumps(result), encoding="utf-8")
    assert not is_complete(row, tmp_path)
