#!/usr/bin/env python
"""Run a path-portable, shardable FCR-port manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tta.ports import build_port, build_port_command


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(base: Path, value: str) -> Path | None:
    value = str(value).strip()
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = payload.get("cells") if isinstance(payload, dict) else payload
    required = {"cell_id", "method", "repo"}
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must contain a non-empty 'cells' list")
    if any(not isinstance(row, dict) or not required.issubset(row) for row in rows):
        raise ValueError(f"every manifest cell requires keys {sorted(required)}")
    cell_ids = [str(row["cell_id"]) for row in rows]
    if any(not value.strip() for value in cell_ids) or len(cell_ids) != len(set(cell_ids)):
        raise ValueError("cell_id values must be non-empty and unique")
    for row in rows:
        row["cell_id"] = str(row["cell_id"])
        row["method"] = str(row["method"])
        row["repo"] = str(row["repo"])
        build_port(row["method"])
        extra = row.get("extra_args", [])
        if not isinstance(extra, list) or not all(isinstance(item, str) for item in extra):
            raise ValueError(f"{row['cell_id']} extra_args must be a string list")
        row["extra_args"] = extra
    return rows


def is_complete(row: dict[str, Any], base: Path) -> bool:
    spec = build_port(row["method"])
    if spec.name == "COSA-FCR":
        output_dir = resolve(base, row.get("output_dir", ""))
        return bool(output_dir and (output_dir / "shard_0.csv").is_file())
    output_json = resolve(base, row.get("output_json", ""))
    return bool(output_json and output_json.is_file())


def command_for(row: dict[str, Any], base: Path) -> list[str]:
    values: dict[str, Any] = {
        "repo": resolve(base, row["repo"]),
        "stream_root": resolve(base, row.get("stream_root", "")),
        "output_dir": resolve(base, row.get("output_dir", "")),
        "cfg": resolve(base, row.get("cfg", "")),
        "base_stream": resolve(base, row.get("base_stream", "")),
        "output_json": resolve(base, row.get("output_json", "")),
        "extra_args": row.get("extra_args", []),
    }
    return build_port_command(row["method"], **values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--max_cells", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--audit_json", type=Path)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")

    manifest = args.manifest.resolve()
    rows = read_manifest(manifest)
    selected = [
        row
        for index, row in enumerate(rows)
        if index % args.num_shards == args.shard_index
    ]
    if args.max_cells:
        selected = selected[: args.max_cells]

    records: list[dict[str, Any]] = []
    for index, row in enumerate(selected, 1):
        if args.dry_run:
            command_for(row, manifest.parent)
            records.append(
                {
                    "cell_id": row["cell_id"],
                    "method": build_port(row["method"]).name,
                    "status": "DRY_RUN",
                    "returncode": 0,
                }
            )
            continue
        if args.resume and is_complete(row, manifest.parent):
            records.append({"cell_id": row["cell_id"], "status": "SKIP", "returncode": 0})
            continue
        started = time.perf_counter()
        completed = subprocess.run(command_for(row, manifest.parent), check=False)
        record = {
            "cell_id": row["cell_id"],
            "method": build_port(row["method"]).name,
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "returncode": completed.returncode,
            "elapsed_sec": time.perf_counter() - started,
        }
        records.append(record)
        print(f"[{index}/{len(selected)}] {record['cell_id']} {record['status']}", flush=True)
        if completed.returncode and not args.continue_on_error:
            break

    audit_path = args.audit_json or manifest.with_name(
        f"run_audit_shard{args.shard_index}.json"
    )
    payload = {
        "status": "PASS" if records and all(row["returncode"] == 0 for row in records) else "FAIL",
        "manifest_sha256": sha256_file(manifest),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_cells": len(selected),
        "executed_records": records,
    }
    audit_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
