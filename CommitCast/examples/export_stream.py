"""Package aligned .npy forecasts and targets as an adapter stream."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, required=True, help="[N,H,C] frozen forecasts")
    parser.add_argument("--true", type=Path, required=True, help="[N,H,C] aligned targets")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pred = np.load(args.pred, allow_pickle=False).astype(np.float32, copy=False)
    true = np.load(args.true, allow_pickle=False).astype(np.float32, copy=False)
    if pred.shape != true.shape or pred.ndim != 3:
        raise ValueError(
            f"pred and true must share [origin, lead, channel], got {pred.shape}, {true.shape}"
        )
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("pred and true must be finite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, pred=pred, true=true)
    print(f"Wrote {args.output} with shape {pred.shape}")


if __name__ == "__main__":
    main()

