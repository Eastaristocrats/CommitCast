"""COSA-style registry and command factory for optional FCR ports."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PortSpec:
    """Immutable public contract for one transported external baseline."""

    name: str
    upstream_project: str
    runner: str
    feedback_rule: str
    service_rule: str
    required_inputs: tuple[str, ...]
    upstream_commit: str
    module_sha256: str
    license_sha256: str


_PORTS: Mapping[str, PortSpec] = {
    "COSA-FCR": PortSpec(
        name="COSA-FCR",
        upstream_project="COSA",
        runner="scripts/fcr_ports/run_cosa.py",
        feedback_rule="complete [H,D] row enters training only after full maturity",
        service_rule="issue/c25/c50/c75 latest-commit exact-once intervals",
        required_inputs=("repo", "stream_root", "output_dir"),
        upstream_commit="43a8c8da4de74d5745a8713f6130c523b7df2694",
        module_sha256="a79ba62f509e2e4bbafb1590708ec851af56763431173e660db33e00ae4c3abf",
        license_sha256="09ee9e203ad9bb0548ce248e0906ba3215d6feb47e54f02c8fbf82349fe07f39",
    ),
    "TAFAS-FCR": PortSpec(
        name="TAFAS-FCR",
        upstream_project="TAFAS",
        runner="scripts/fcr_ports/run_tafas_petsa_cell.py",
        feedback_rule="official PAAS partial/full transitions replayed when legal",
        service_rule="prospective current-origin emission; no retroactive replacement",
        required_inputs=("repo", "cfg", "base_stream", "output_json"),
        upstream_commit="139bf980671da4daad728a0fc21d8df508b9203d",
        module_sha256="7ff75bb08c6efadcc8dcedc3f9599b62ebb5397da77485c4254d78c3972b3932",
        license_sha256="5692a759b37b18116674d09e6a0dad678c512becbe744bb2e509da2664aeb10e",
    ),
    "PETSA-FCR": PortSpec(
        name="PETSA-FCR",
        upstream_project="PETSA",
        runner="scripts/fcr_ports/run_tafas_petsa_cell.py",
        feedback_rule="official PAAS partial/full transitions replayed when legal",
        service_rule="prospective current-origin emission; no retroactive replacement",
        required_inputs=("repo", "cfg", "base_stream", "output_json"),
        upstream_commit="87853d888e98311ac94e64be920d17b57143b20c",
        module_sha256="7704007c72aa018ed5f838ff7bdb99a48056116a35f32a7482d15c30d8e3feb2",
        license_sha256="09ee9e203ad9bb0548ce248e0906ba3215d6feb47e54f02c8fbf82349fe07f39",
    ),
}

PORT_NAMES = tuple(_PORTS)


def build_port(name: str) -> PortSpec:
    """Return a registered port without importing third-party code."""
    normalized = str(name).strip().upper().replace("_", "-")
    aliases = {
        "COSA": "COSA-FCR",
        "TAFAS": "TAFAS-FCR",
        "PETSA": "PETSA-FCR",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _PORTS:
        raise KeyError(f"unknown FCR port {name!r}; choose one of {PORT_NAMES}")
    return _PORTS[normalized]


def build_port_command(
    name: str,
    *,
    repo: str | Path,
    stream_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: str | Path | None = None,
    base_stream: str | Path | None = None,
    output_json: str | Path | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Build the subprocess boundary used by all three ports."""
    spec = build_port(name)
    values = {
        "repo": repo,
        "stream_root": stream_root,
        "output_dir": output_dir,
        "cfg": cfg,
        "base_stream": base_stream,
        "output_json": output_json,
    }
    missing = [key for key in spec.required_inputs if values[key] in (None, "")]
    if missing:
        raise ValueError(f"{spec.name} requires {', '.join(missing)}")

    command = [sys.executable, str(REPOSITORY_ROOT / spec.runner)]
    if spec.name in {"TAFAS-FCR", "PETSA-FCR"}:
        command += ["--method", spec.upstream_project]
    for key in spec.required_inputs:
        command += [f"--{key}", str(values[key])]
    command += [
        "--expected_commit",
        spec.upstream_commit,
        "--expected_module_sha256",
        spec.module_sha256,
        "--expected_license_sha256",
        spec.license_sha256,
    ]
    command += [str(value) for value in extra_args]
    return command
