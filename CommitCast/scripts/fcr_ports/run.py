#!/usr/bin/env python
"""Unified CLI for COSA-FCR, TAFAS-FCR, and PETSA-FCR."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tta.ports import PORT_NAMES, build_port_command  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=PORT_NAMES)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--stream_root", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--cfg", type=Path)
    parser.add_argument("--base_stream", type=Path)
    parser.add_argument("--output_json", type=Path)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    opts = args.opts[1:] if args.opts and args.opts[0] == "--" else args.opts
    command = build_port_command(
        args.method,
        repo=args.repo,
        stream_root=args.stream_root,
        output_dir=args.output_dir,
        cfg=args.cfg,
        base_stream=args.base_stream,
        output_json=args.output_json,
        extra_args=opts,
    )
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
