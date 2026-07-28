#!/usr/bin/env python
"""Generate the complete FCR-port run manifest from frozen experiment specs."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("COSA-FCR", "TAFAS-FCR", "PETSA-FCR")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def relative(target: Path, base: Path) -> str:
    """Return a portable non-absolute path relative to ``base``."""
    value = Path(os.path.relpath(target.resolve(), base.resolve())).as_posix()
    if Path(value).is_absolute():
        raise ValueError(f"cannot make {target} relative to {base}")
    return value


def hyperparameters(
    payload: dict[str, Any],
    *,
    method: str,
    dataset: str,
    backbone: str,
    horizon: int,
) -> dict[str, float | int]:
    horizons = [int(value) for value in payload["horizons"]]
    if horizon not in horizons:
        raise KeyError(f"unknown horizon {horizon}")
    index = horizons.index(horizon)
    setting = payload["settings"][dataset][backbone]
    result: dict[str, float | int] = {
        "lr": float(setting["lr"][index]),
        "weight_decay": float(setting["weight_decay"][index]),
    }
    if method == "TAFAS-FCR":
        result["gating_init"] = float(setting["tafas_gating_init"][index])
    elif method == "PETSA-FCR":
        result.update(
            {
                key: float(value) if key != "rank" else int(value)
                for key, value in payload["petsa_defaults"].items()
            }
        )
        override = payload.get("petsa_overrides", {}).get(
            f"{dataset}/{backbone}/{horizon}", {}
        )
        result.update(
            {
                key: float(value) if key != "rank" else int(value)
                for key, value in override.items()
            }
        )
    return result


def build_manifest(
    spec: dict[str, Any],
    hparams: dict[str, Any],
    *,
    manifest_path: Path,
    external_root: Path,
    data_root: Path,
    streams_root: Path,
    checkpoints_root: Path,
    results_root: Path,
    device: str,
    seeds: list[int],
) -> dict[str, Any]:
    manifest_base = manifest_path.resolve().parent
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty list of unique integers")
    datasets = [str(value) for value in spec["datasets"]]
    backbones = [str(value) for value in spec["backbones"]]
    horizons = [int(value) for value in spec["horizons"]]
    expected_settings = len(datasets) * len(backbones) * len(horizons)
    if expected_settings != 144:
        raise ValueError(f"canonical experiment must contain 144 settings, got {expected_settings}")
    if hparams.get("seed_policy") != "reuse_seed0_per_setting_without_retuning":
        raise ValueError("unexpected FCR hyperparameter seed policy")

    gpu_index = device.split(":", 1)[1] if device.startswith("cuda:") else "0"
    cells: list[dict[str, Any]] = []
    for seed in seeds:
        for dataset in datasets:
            for backbone in backbones:
                for horizon in horizons:
                    stream_name = (
                        f"BASE_{dataset}_{backbone}_h{horizon}_"
                        f"seed{seed}_batch{int(spec['batch_size'])}"
                    )
                    stream = streams_root / stream_name / "adapter_stream.npz"
                    checkpoint_dir = (
                        checkpoints_root
                        / dataset
                        / backbone
                        / f"h{horizon}"
                        / f"seed{seed}"
                    )
                    for method in METHODS:
                        slug = method.lower().replace("-", "_")
                        cell_id = (
                            f"{slug}_{dataset}_{backbone}_h{horizon}_seed{seed}"
                        ).lower()
                        repo = external_root / spec["upstreams"][method]["directory"]
                        row: dict[str, Any] = {
                            "cell_id": cell_id,
                            "method": method,
                            "repo": relative(repo, manifest_base),
                        }
                        if method == "COSA-FCR":
                            row.update(
                                {
                                    "stream_root": relative(streams_root, manifest_base),
                                    "output_dir": relative(
                                        results_root / slug / cell_id,
                                        manifest_base,
                                    ),
                                    "extra_args": [
                                        "--device",
                                        device,
                                        "--update_batch_size",
                                        str(int(spec["batch_size"])),
                                        "--serve_batch_size",
                                        str(int(spec["service_batch_size"])),
                                        "--context",
                                        str(int(spec["cosa"]["context"])),
                                        "--steps",
                                        str(int(spec["cosa"]["steps"])),
                                        "--lr",
                                        str(float(spec["cosa"]["lr"])),
                                        "--weight_decay",
                                        str(float(spec["cosa"]["weight_decay"])),
                                        "--include_regex",
                                        f"^{stream_name}$",
                                        "--metric_protocol",
                                        str(spec["metric_protocol"]),
                                        "--max_streams",
                                        "1",
                                    ],
                                }
                            )
                        else:
                            hp = hyperparameters(
                                hparams,
                                method=method,
                                dataset=dataset,
                                backbone=backbone,
                                horizon=horizon,
                            )
                            options = [
                                "--service_batch_size",
                                str(int(spec["service_batch_size"])),
                                "--update_cadence",
                                str(spec["update_cadence"]),
                                "--",
                                "DATA.BASE_DIR",
                                relative(data_root, repo),
                                "DATA.NAME",
                                dataset,
                                "DATA.PRED_LEN",
                                str(horizon),
                                "DATA.TRAIN_RATIO",
                                "0.7",
                                "DATA.TEST_RATIO",
                                "0.2",
                                "MODEL.NAME",
                                backbone,
                                "MODEL.pred_len",
                                str(horizon),
                                "TRAIN.ENABLE",
                                "False",
                                "TRAIN.CHECKPOINT_DIR",
                                relative(checkpoint_dir, repo) + "/",
                                "TEST.ENABLE",
                                "False",
                                "TTA.ENABLE",
                                "True",
                                "DATA_LOADER.NUM_WORKERS",
                                "0",
                                "DATA_LOADER.PIN_MEMORY",
                                "False",
                                "VISIBLE_DEVICES",
                                gpu_index,
                                "TTA.SOLVER.BASE_LR",
                                str(hp["lr"]),
                                "TTA.SOLVER.WEIGHT_DECAY",
                                str(hp["weight_decay"]),
                                f"TTA.{method.split('-', 1)[0]}.PERIOD_N",
                                "1",
                                f"TTA.{method.split('-', 1)[0]}.STEPS",
                                "1",
                            ]
                            if method == "TAFAS-FCR":
                                options += [
                                    "TTA.TAFAS.GATING_INIT",
                                    str(hp["gating_init"]),
                                ]
                            else:
                                options += [
                                    "TTA.PETSA.GATING_INIT",
                                    str(hp["gating_init"]),
                                    "TTA.PETSA.RANK",
                                    str(hp["rank"]),
                                    "TTA.PETSA.LOSS_ALPHA",
                                    str(hp["loss_alpha"]),
                                ]
                            row.update(
                                {
                                    "cfg": relative(
                                        checkpoint_dir / "config.yaml",
                                        manifest_base,
                                    ),
                                    "base_stream": relative(stream, manifest_base),
                                    "output_json": relative(
                                        results_root / slug / f"{cell_id}.json",
                                        manifest_base,
                                    ),
                                    "extra_args": options,
                                }
                            )
                        cells.append(row)

    expected_cells = len(seeds) * expected_settings * len(METHODS)
    if len(cells) != expected_cells or len({row["cell_id"] for row in cells}) != len(cells):
        raise AssertionError("generated manifest is incomplete or has duplicate cell IDs")
    return {
        "schema_version": 1,
        "experiment": str(spec["name"]),
        "settings_per_seed": expected_settings,
        "seeds": seeds,
        "methods": list(METHODS),
        "expected_cells": expected_cells,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "configs" / "fcr_ports_experiment.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external_root", type=Path, default=ROOT / "external")
    parser.add_argument("--data_root", type=Path, default=ROOT / "data")
    parser.add_argument("--streams_root", type=Path, default=ROOT / "streams")
    parser.add_argument("--checkpoints_root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--results_root", type=Path, default=ROOT / "results" / "fcr_ports")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Explicit repeat identifiers; default: 0.",
    )
    args = parser.parse_args()

    spec_path = args.spec.resolve()
    spec = load_yaml(spec_path)
    hparams_path = ROOT / str(spec["hyperparameters"])
    hparams = load_yaml(hparams_path)
    payload = build_manifest(
        spec,
        hparams,
        manifest_path=args.output,
        external_root=args.external_root,
        data_root=args.data_root,
        streams_root=args.streams_root,
        checkpoints_root=args.checkpoints_root,
        results_root=args.results_root,
        device=args.device,
        seeds=args.seeds,
    )
    payload["spec_sha256"] = sha256_file(spec_path)
    payload["hyperparameters_sha256"] = sha256_file(hparams_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(payload['cells'])} cells "
        f"({payload['settings_per_seed']} settings/seed) to {args.output}"
    )


if __name__ == "__main__":
    main()
