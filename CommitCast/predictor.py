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
from fcr.scoring import common_origins as support_count, summarize_requests


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
            common_origins = support_count(len(stream.pred), int(stream.pred.shape[1]),
                [parse_csv(self.cfg.EVALUATION.SUPPORT_FRACTIONS, float)])
            events: list[CommitEvent] = []
            first_delay = int(schedule[0][1]) if schedule else int(stream.pred.shape[1])
            initial = stream.pred[:common_origins, :first_delay, :]
            events.append(
                CommitEvent(
                    label=f"issue_to_{schedule[0][0]}" if schedule else "issue_to_end",
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
        request_rows = []

        for result in results:
            stream = result.stream
            record = stream.record
            horizon = int(stream.pred.shape[1])
            manifest.append(
                {
                    "stream": record.stream,
                    "shape": list(stream.pred.shape),
                    "dataset": record.dataset,
                    "backbone": record.backbone,
                    "horizon": horizon,
                    "seed": record.seed,
                    "request_count": len(result.events[0].target),
                    "targets_per_request": horizon * stream.pred.shape[2],
                    "profile": self.cfg.EVALUATION.PROFILE,
                    "support_fractions": list(self.cfg.EVALUATION.SUPPORT_FRACTIONS),
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
                            "protocol": "complete-H; same-commit Base; request-local exactly-once",
                            "adapter_runtime_sec": (
                                result.runtime_sec / max(1, len(result.events))
                                if method_name == "CommitCast"
                                else 0.0
                            ),
                        }
                    )

            # Reconstruct each full request independently of the interval totals.
            count = len(result.events[0].target)
            for method_name, attr in (("Base", "checkpoint"),
                                      ("CommitCast-Raw", "proposal"), ("CommitCast", "prediction")):
                if self.adapter is None and method_name != "Base":
                    continue
                for request in range(count):
                    row = dict(stream=record.stream, method=method_name, request=request,
                               sse=0., sae=0., base_sse=0., base_sae=0., atoms=0)
                    for event in result.events:
                        scored = metric_dict(event.target[request], getattr(event, attr)[request])
                        baseline = metric_dict(event.target[request], event.checkpoint[request])
                        row["sse"] += scored["sse"]
                        row["sae"] += scored["sae"]
                        row["base_sse"] += baseline["sse"]
                        row["base_sae"] += baseline["sae"]
                        row["atoms"] += scored["evaluated_elements"]
                    if row["atoms"] != horizon * stream.pred.shape[2]:
                        raise AssertionError("Each request must contain exactly H*C targets")
                    request_rows.append(row)

        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError("No streams matched the input filters")
        names = {"full": "CommitCast", "pg": "CommitCast-PG", "p": "CommitCast-P", "g": "CommitCast-G"}
        chosen_name = names[self.cfg.TTA.COMMITCAST.FEATURE_MODE]
        frame["method"] = frame["method"].replace({"CommitCast": chosen_name,
                                                     "CommitCast-Raw": chosen_name + "-Raw"})
        for row in request_rows:
            row["method"] = row["method"].replace("CommitCast", chosen_name)
        request_frame = pd.DataFrame(request_rows)
        request_frame.to_csv(output_dir / "request_metrics.csv", index=False)
        risk = []
        for (method, stream_name), group in request_frame.groupby(["method", "stream"]):
            risk.append(dict(method=method, stream=stream_name,
                             **summarize_requests(group.to_dict("records"))))
        pd.DataFrame(risk).to_csv(output_dir / "request_risk.csv", index=False)
        frame.to_csv(output_dir / "event_metrics.csv", index=False)
        _summarize(frame, ["method"]).to_csv(output_dir / "summary.csv", index=False)
        _summarize(frame, ["method", "stream"]).to_csv(
            output_dir / "breakdown_by_stream.csv", index=False
        )
        cells = _summarize(frame, ["method", "stream"])
        cells.groupby("method")[["mse_gain_vs_checkpoint_pct", "mae_gain_vs_checkpoint_pct"]].mean().rename(
            columns={"mse_gain_vs_checkpoint_pct": "equal_cell_mean_mse_gain_pct",
                     "mae_gain_vs_checkpoint_pct": "equal_cell_mean_mae_gain_pct"}
        ).to_csv(output_dir / "secondary_macro_summary.csv")
        for column in ("dataset", "backbone", "horizon", "commit"):
            _summarize(frame, ["method", column]).to_csv(
                output_dir / f"breakdown_by_{column}.csv", index=False
            )
        (output_dir / "stream_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Wrote results to {output_dir}")
