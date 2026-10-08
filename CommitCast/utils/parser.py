"""COSA-style configuration parsing."""

from __future__ import annotations

import argparse

from config import get_cfg_defaults


def _normalize_sequence_overrides(opts: list[str]) -> list:
    """Accept convenient scalar/comma forms for tuple-valued YACS options."""

    normalized: list = list(opts)
    sequence_types = {
        "DATA.PRED_LEN": int,
        "EVALUATION.SUPPORT_FRACTIONS": float,
        "TTA.COMMITCAST.FRACTIONS": float,
    }
    for index in range(0, len(normalized), 2):
        key = normalized[index]
        if key not in sequence_types or index + 1 >= len(normalized):
            continue
        value = normalized[index + 1]
        if isinstance(value, (tuple, list)):
            normalized[index + 1] = tuple(sequence_types[key](item) for item in value)
            continue
        text = str(value).strip().strip("[]()")
        normalized[index + 1] = tuple(
            sequence_types[key](item.strip()) for item in text.split(",") if item.strip()
        )
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CommitCast: same-commit forecast revision under delayed feedback"
    )
    parser.add_argument("--cfg", dest="cfg_file", default=None, help="YAML config file")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Configuration overrides as KEY VALUE pairs; see config.py",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace):
    cfg = get_cfg_defaults()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    if args.opts:
        cfg.merge_from_list(_normalize_sequence_overrides(args.opts))
    if cfg.STREAM.MAX_STREAMS < 0:
        raise ValueError("STREAM.MAX_STREAMS must be nonnegative")
    if cfg.TTA.COMMITCAST.FEATURE_MODE not in {"full", "pg", "p", "g"}:
        raise ValueError("Invalid CommitCast feature mode")
    return cfg
