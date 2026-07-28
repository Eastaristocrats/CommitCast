"""COSA-style replay predictor and result writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from tta import AdaptationResult, CommitEvent, commit_schedule
from utils.metrics import gain_pct, metric_dict
from utils.misc import parse_csv


def _summarize(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        base_sse = float(group["checkpoint_sse"].sum())
        candidate_sse = float(group["sse"].sum())
        base_sae = float(group["checkpoint_sae"].sum())
        candidate_sae = float(group["sae"].sum())
        count = int(group["evaluated_elements"].sum())
        row.update(
            {
                "streams": int(group["stream"].nunique()),
                "events": int(len(group)),
                "evaluated_elements": count,
                "checkpoint_mse": base_sse / max(1, count),
                "mse": candidate_sse / max(1, count),
                "mse_gain_vs_checkpoint_pct": gain_pct(base_sse, candidate_sse),
                "checkpoint_mae": base_sae / max(1, count),
                "mae": candidate_sae / max(1, count),
                "mae_gain_vs_checkpoint_pct": gain_pct(base_sae, candidate_sae),
                "event_win_rate_pct": 100.0
                * float((group["mse_gain_vs_checkpoint_pct"] > 0).mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


class Predictor:
    def __init__(self, cfg, model, adapter=None):
        self.cfg = cfg
        self.model = model
        self.adapter = adapter
        self.fractions = parse_csv(cfg.TTA.COMMITCAST.FRACTIONS, float)

    def _baseline_results(self) -> tuple[AdaptationResult, ...]:
        results: list[AdaptationResult] = []
        for stream in self.model.streams():
            schedule = commit_schedule(int(stream.pred.shape[1]), self.fractions)
            if not schedule:
                raise ValueError("at least one commitment fraction is required")
            common_origins = int(stream.pred.shape[0] - schedule[-1][1])
            events: list[CommitEvent] = []
            first_delay = int(schedule[0][1])
            initial = stream.pred[:common_origins, :first_delay, :]
            events.append(
                CommitEvent(
                    label=f"issue_to_{schedule[0][0]}",
                    delay=0,
                    next_delay=first_delay,
                    target=stream.true[:common_origins, :first_delay, :],
                    checkpoint=initial,
                    proposal=initial,
                    prediction=initial,
                    exposure=np.zeros(common_origins, dtype=np.float64),
                )
            )
            for index, (label, delay) in enumerate(schedule):
                next_delay = (
                    schedule[index + 1][1] if index + 1 < len(schedule) else stream.pred.shape[1]
                )
                segment_len = next_delay - delay
                checkpoint = stream.pred[delay:, :segment_len, :][:common_origins]
                target = stream.true[:common_origins, delay:next_delay, :]
                events.append(
                    CommitEvent(
                        label=label,
                        delay=delay,
                        next_delay=next_delay,
                        target=target,
                        checkpoint=checkpoint,
                        proposal=checkpoint,
                        prediction=checkpoint,
                        exposure=np.zeros(common_origins, dtype=np.float64),
                    )
                )
            results.append(
                AdaptationResult(stream=stream, events=tuple(events), runtime_sec=0.0)
            )
        return tuple(results)

    def predict(self) -> None:
        if self.adapter is None:
            results = self._baseline_results()
        else:
            results = self.adapter.results or self.adapter.adapt()
        output_dir = Path(self.cfg.RESULT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        manifest: list[dict[str, Any]] = []

        for result in results:
            stream = result.stream
            record = stream.record
            horizon = int(stream.pred.shape[1])
            stream_hash = stream.sha256
            manifest.append(
                {
                    "stream": record.stream,
                    "sha256": stream_hash,
                    "shape": list(stream.pred.shape),
                    "dataset": record.dataset,
                    "backbone": record.backbone,
                    "horizon": horizon,
                    "seed": record.seed,
                }
            )
            for event in result.events:
                checkpoint_metrics = metric_dict(event.target, event.checkpoint)
                predictions = [("Base", event.checkpoint)]
                if self.adapter is not None:
                    predictions.extend(
                        [
                            ("CommitCast-Raw", event.proposal),
                            ("CommitCast", event.prediction),
                        ]
                    )
                for method_name, prediction in predictions:
                    metrics = metric_dict(event.target, prediction)
                    rows.append(
                        {
                            "stream": record.stream,
                            "stream_sha256": stream_hash,
                            "source": record.source,
                            "dataset": record.dataset,
                            "backbone": record.backbone,
                            "horizon": horizon,
                            "seed": record.seed,
                            "commit": event.label,
                            "delay": event.delay,
                            "next_delay": event.next_delay,
                            "served_leads": event.segment_len,
                            "method": method_name,
                            **metrics,
                            "checkpoint_sse": checkpoint_metrics["sse"],
                            "checkpoint_sae": checkpoint_metrics["sae"],
                            "checkpoint_mse": checkpoint_metrics["mse"],
                            "checkpoint_mae": checkpoint_metrics["mae"],
                            "mse_gain_vs_checkpoint_pct": gain_pct(
                                checkpoint_metrics["sse"], metrics["sse"]
                            ),
                            "mae_gain_vs_checkpoint_pct": gain_pct(
                                checkpoint_metrics["sae"], metrics["sae"]
                            ),
                            "exposure_mean": (
                                0.0
                                if method_name == "Base"
                                else 1.0
                                if method_name == "CommitCast-Raw"
                                else float(event.exposure.mean())
                            ),
                            "exposure_zero_rate_pct": (
                                100.0
                                if method_name == "Base"
                                else 0.0
                                if method_name == "CommitCast-Raw"
                                else 100.0 * float((event.exposure == 0.0).mean())
                            ),
                            "protocol": "same-commit, non-overlapping served segments",
                            "adapter_runtime_sec": (
                                result.runtime_sec / max(1, len(result.events))
                                if method_name == "CommitCast"
                                else 0.0
                            ),
                        }
                    )

        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / "event_metrics.csv", index=False)
        _summarize(frame, ["method"]).to_csv(output_dir / "summary.csv", index=False)
        _summarize(frame, ["method", "stream"]).to_csv(
            output_dir / "breakdown_by_stream.csv", index=False
        )
        for column in ("dataset", "backbone", "horizon", "commit"):
            _summarize(frame, ["method", column]).to_csv(
                output_dir / f"breakdown_by_{column}.csv", index=False
            )
        (output_dir / "stream_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Wrote results to {output_dir}")
