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


_PORTS: Mapping[str, PortSpec] = {
    "COSA-FCR": PortSpec(
        name="COSA-FCR",
        upstream_project="COSA",
        runner="scripts/fcr_ports/run_cosa.py",
        feedback_rule="complete [H,D] row enters training only after full maturity",
        service_rule="issue/c25/c50/c75 latest-commit exact-once intervals",
        required_inputs=("repo", "stream_root", "output_dir"),
    ),
    "TAFAS-FCR": PortSpec(
        name="TAFAS-FCR",
        upstream_project="TAFAS",
        runner="scripts/fcr_ports/run_tafas_petsa_cell.py",
        feedback_rule="official PAAS partial/full transitions replayed when legal",
        service_rule="prospective current-origin emission; no retroactive replacement",
        required_inputs=("repo", "cfg", "base_stream", "output_json"),
    ),
    "PETSA-FCR": PortSpec(
        name="PETSA-FCR",
        upstream_project="PETSA",
        runner="scripts/fcr_ports/run_tafas_petsa_cell.py",
        feedback_rule="official PAAS partial/full transitions replayed when legal",
        service_rule="prospective current-origin emission; no retroactive replacement",
        required_inputs=("repo", "cfg", "base_stream", "output_json"),
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
    """Build the reproducible subprocess boundary used by all three ports."""
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
    command += [str(value) for value in extra_args]
    return command
