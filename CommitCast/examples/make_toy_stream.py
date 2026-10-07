"""Create a small deterministic stream for a smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("streams/BASE_Toy_DLinear_h24_seed0_batch8/adapter_stream.npz"),
    )
    parser.add_argument("--origins", type=int, default=160)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    total = args.origins + args.horizon + 8
    time = np.arange(total, dtype=np.float32)
    series = np.stack(
        [
            np.sin(time / (3.0 + channel))
            + 0.3 * np.cos(time / (8.0 + 2 * channel))
            + 0.002 * time
            for channel in range(args.channels)
        ],
        axis=-1,
    )
    series += rng.normal(scale=0.03, size=series.shape).astype(np.float32)
    true = np.stack(
        [series[index : index + args.horizon] for index in range(args.origins)],
        axis=0,
    ).astype(np.float32)
    lead = np.arange(args.horizon, dtype=np.float32)[None, :, None]
    origin = np.arange(args.origins, dtype=np.float32)[:, None, None]
    bias = 0.12 * np.sin((origin + lead) / 19.0)
    pred = true + bias + rng.normal(scale=0.08, size=true.shape).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, pred=pred, true=true)
    print(f"Wrote {args.output} with shape {pred.shape}")


if __name__ == "__main__":
    main()

