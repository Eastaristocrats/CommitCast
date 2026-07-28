#!/usr/bin/env python3
"""Audit and summarize the mode-faithful dual-output FCR baseline matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


KEY = ("dataset", "backbone", "horizon", "seed")
BASE_METHODS = ("TAFAS-FCR", "PETSA-FCR")
VARIANTS = ("Literal", "Anchored")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells_dir", type=Path, required=True)
    parser.add_argument("--expected_matrix", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_cells", type=int, default=144)
    parser.add_argument("--base_relative_tolerance", type=float, default=1e-8)
    parser.add_argument("--expected_runner_sha256", default="")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return str(row["dataset"]), str(row["backbone"]), int(row["horizon"]), int(row.get("seed", 0))


def relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-15)


def cvar(values: np.ndarray, fraction: float) -> float:
    count = max(1, int(math.ceil(fraction * len(values))))
    return float(np.mean(np.sort(values)[:count]))


def candidate_fields(row: dict[str, Any], variant: str) -> tuple[float, float, list[float], float]:
    if variant == "Literal":
        return (
            float(row["mse"]),
            float(row["mae"]),
            [float(value) for value in row["current_full_h_lead_quartile_method_mse"]],
            float(row["update_on_gain_vs_update_off_pct"]),
        )
    if variant == "Anchored":
        return (
            float(row["anchored_mse"]),
            float(row["anchored_mae"]),
            [float(value) for value in row["anchored_current_full_h_lead_quartile_method_mse"]],
            float(row["anchored_update_on_gain_vs_update_off_pct"]),
        )
    raise ValueError(variant)


def summarize(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    elements = np.asarray([int(row["evaluated_elements"]) for row in rows], dtype=np.float64)
    base_sse = np.asarray([float(row["base_mse"]) for row in rows]) * elements
    base_sae = np.asarray([float(row["base_mae"]) for row in rows]) * elements
    candidates = [candidate_fields(row, variant) for row in rows]
    method_sse = np.asarray([item[0] for item in candidates]) * elements
    method_sae = np.asarray([item[1] for item in candidates]) * elements
    gains = 100.0 * (base_sse - method_sse) / np.maximum(base_sse, 1e-12)
    update_gain = np.asarray([item[3] for item in candidates])
    result: dict[str, Any] = {
        "method": f"{rows[0]['method']}-{variant}",
        "candidate_variant": variant,
        "primary_eligible": variant == "Literal",
        "cells": len(rows),
        "evaluated_elements": int(elements.sum()),
        "base_mse": float(base_sse.sum() / elements.sum()),
        "mse": float(method_sse.sum() / elements.sum()),
        "gain_vs_base_pct": float(100.0 * (base_sse.sum() - method_sse.sum()) / base_sse.sum()),
        "base_mae": float(base_sae.sum() / elements.sum()),
        "mae": float(method_sae.sum() / elements.sum()),
        "mae_gain_vs_base_pct": float(100.0 * (base_sae.sum() - method_sae.sum()) / base_sae.sum()),
        "cell_gain_mean_pct": float(np.mean(gains)),
        "cell_gain_median_pct": float(np.median(gains)),
        "cell_gain_q05_pct": float(np.quantile(gains, 0.05)),
        "cell_gain_cvar10_pct": cvar(gains, 0.10),
        "cell_gain_worst_pct": float(np.min(gains)),
        "cell_wins": int(np.sum(gains > 1e-9)),
        "cell_ties": int(np.sum(np.abs(gains) <= 1e-9)),
        "cell_losses": int(np.sum(gains < -1e-9)),
        "negative_transfer_rate_pct": float(100.0 * np.mean(gains < -1e-9)),
        "immediate_update_cell_median_gain_pct": float(np.nanmedian(update_gain)),
        "immediate_update_positive_cell_rate_pct": float(100.0 * np.mean(update_gain > 0.0)),
        "backward_calls": int(sum(int(row["backward_calls"]) for row in rows)),
        "optimizer_steps": int(sum(int(row["optimizer_steps"]) for row in rows)),
        "trainable_parameters_min": int(min(int(row["trainable_parameters"]) for row in rows)),
        "trainable_parameters_max": int(max(int(row["trainable_parameters"]) for row in rows)),
    }
    for quartile in range(4):
        q_elements = np.asarray(
            [int(row["n"]) * (int(row["horizon"]) // 4) * int(row["channels"]) for row in rows],
            dtype=np.float64,
        )
        q_base = np.asarray(
            [float(row["current_full_h_lead_quartile_base_mse"][quartile]) for row in rows]
        ) * q_elements
        q_method = np.asarray([item[2][quartile] for item in candidates]) * q_elements
        result[f"current_lead_q{quartile + 1}_gain_pct"] = float(
            100.0 * (q_base.sum() - q_method.sum()) / q_base.sum()
        )
    return result


def sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    elements = np.asarray([int(row["evaluated_elements"]) for row in rows], dtype=np.float64)
    base_sse = np.asarray([float(row["base_mse"]) for row in rows]) * elements
    literal_sse = np.asarray([float(row["mse"]) for row in rows]) * elements
    anchored_sse = np.asarray([float(row["anchored_mse"]) for row in rows]) * elements
    literal_gain = 100.0 * (base_sse - literal_sse) / np.maximum(base_sse, 1e-12)
    anchored_gain = 100.0 * (base_sse - anchored_sse) / np.maximum(base_sse, 1e-12)
    return {
        "method": rows[0]["method"],
        "cells": len(rows),
        "literal_pooled_gain_pct": float(100.0 * (base_sse.sum() - literal_sse.sum()) / base_sse.sum()),
        "anchored_pooled_gain_pct": float(100.0 * (base_sse.sum() - anchored_sse.sum()) / base_sse.sum()),
        "literal_minus_anchored_pooled_gain_pp": float(
            100.0 * (anchored_sse.sum() - literal_sse.sum()) / base_sse.sum()
        ),
        "max_abs_cell_gain_difference_pp": float(np.max(np.abs(literal_gain - anchored_gain))),
        "median_abs_cell_gain_difference_pp": float(np.median(np.abs(literal_gain - anchored_gain))),
        "base_win_loss_sign_flip_count": int(
            np.sum(np.sign(literal_gain) != np.sign(anchored_gain))
        ),
    }


def main() -> None:
    args = parse_args()
    if args.base_relative_tolerance < 0:
        raise ValueError("base tolerance must be non-negative")
    with args.expected_matrix.open(encoding="utf-8-sig", newline="") as handle:
        expected_rows = list(csv.DictReader(handle))
    expected = {key(row): row for row in expected_rows}
    if len(expected) != len(expected_rows) or len(expected) != args.expected_cells:
        raise AssertionError("expected matrix is duplicate or incomplete")

    rows = []
    for path in sorted(args.cells_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["source_json"] = path.name
        rows.append(row)
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    if set(by_method) != set(BASE_METHODS):
        raise AssertionError(f"method coverage {sorted(by_method)} != {list(BASE_METHODS)}")

    runner_hashes = set()
    max_expected_base_mse = max_expected_base_mae = 0.0
    max_literal_base_drift = max_anchor_correction = 0.0
    for method, method_rows in by_method.items():
        keys = [key(row) for row in method_rows]
        if len(keys) != args.expected_cells or len(set(keys)) != len(keys) or set(keys) != set(expected):
            raise AssertionError(f"{method} does not exactly cover the expected grid")
        for row in method_rows:
            reference = expected[key(row)]
            if row.get("primary_candidate") != "literal official-calibrator output":
                raise AssertionError(f"literal output is not primary in {row['source_json']}")
            if row.get("sensitivity_candidate") != "shared-Base control-variate transport":
                raise AssertionError(f"anchored output is not sensitivity-only in {row['source_json']}")
            zero_fields = (
                "target_identity_max_abs",
                "forward_state_mutation_count",
                "partial_update_training_mode_violation_count",
                "full_update_eval_mode_violation_count",
                "post_update_eval_mode_violation_count",
                "anchored_initial_shared_base_max_abs",
            )
            if any(float(row[field]) != 0.0 for field in zero_fields):
                raise AssertionError(f"causal/mode/identity audit failed in {row['source_json']}")
            if float(row["initial_live_delta_max_abs"]) > 1e-7:
                raise AssertionError(f"initial live adapter is not identity in {row['source_json']}")
            if int(row["max_partial_target_time_minus_issue"]) > -1:
                raise AssertionError(f"partial future-label leak in {row['source_json']}")
            if int(row["max_full_batch_target_time_minus_issue"]) > -1:
                raise AssertionError(f"full future-label leak in {row['source_json']}")
            if int(row["steps_per_update"]) != 1 or str(row["update_cadence"]) != "official_paas":
                raise AssertionError(f"official matrix update policy mismatch in {row['source_json']}")
            if int(row["evaluated_elements"]) != int(float(reference["evaluated_elements"])):
                raise AssertionError(f"evaluated support mismatch in {row['source_json']}")
            if int(row["n"]) != int(float(reference["n"])) or int(row["channels"]) != int(float(reference["channels"])):
                raise AssertionError(f"shape mismatch in {row['source_json']}")
            if int(row["service_origins"]) != int(float(reference["service_origins"])):
                raise AssertionError(f"service-origin mismatch in {row['source_json']}")
            if Path(str(row["base_stream"])).parent.name != str(reference["stream"]):
                raise AssertionError(f"Base stream mismatch in {row['source_json']}")
            mse_diff = relative_difference(float(row["base_mse"]), float(reference["base_mse"]))
            mae_diff = relative_difference(float(row["base_mae"]), float(reference["base_mae"]))
            max_expected_base_mse = max(max_expected_base_mse, mse_diff)
            max_expected_base_mae = max(max_expected_base_mae, mae_diff)
            if mse_diff > args.base_relative_tolerance or mae_diff > args.base_relative_tolerance:
                raise AssertionError(f"shared Base mismatch in {row['source_json']}")
            max_literal_base_drift = max(
                max_literal_base_drift, float(row["literal_frozen_base_sse_relative_difference"])
            )
            max_anchor_correction = max(max_anchor_correction, float(row["numerical_anchor_max_abs"]))
            runner_hashes.add(str(row["runner_sha256"]))
    if len(runner_hashes) != 1:
        raise AssertionError(f"multiple runner hashes: {sorted(runner_hashes)}")
    executed_runner_hash = next(iter(runner_hashes))
    if args.expected_runner_sha256 and executed_runner_hash != args.expected_runner_sha256:
        raise AssertionError(
            f"executed runner hash {executed_runner_hash} != frozen {args.expected_runner_sha256}"
        )

    paired = {method: {key(row): row for row in method_rows} for method, method_rows in by_method.items()}
    max_cross_mse = max_cross_mae = 0.0
    for cell_key in expected:
        left, right = paired["TAFAS-FCR"][cell_key], paired["PETSA-FCR"][cell_key]
        max_cross_mse = max(max_cross_mse, relative_difference(float(left["base_mse"]), float(right["base_mse"])))
        max_cross_mae = max(max_cross_mae, relative_difference(float(left["base_mae"]), float(right["base_mae"])))
    if max_cross_mse > args.base_relative_tolerance or max_cross_mae > args.base_relative_tolerance:
        raise AssertionError("cross-method shared Base identity failed")

    summaries = [
        summarize(by_method[method], variant)
        for method in BASE_METHODS
        for variant in VARIANTS
    ]
    sensitivities = [sensitivity(by_method[method]) for method in BASE_METHODS]
    flat_rows = []
    for row in sorted(rows, key=lambda item: (item["method"], *key(item))):
        flat = {field: value for field, value in row.items() if not isinstance(value, (list, dict))}
        for quartile in range(4):
            flat[f"literal_current_lead_q{quartile + 1}_gain_pct"] = row[
                "current_full_h_lead_quartile_gain_pct"
            ][quartile]
            flat[f"anchored_current_lead_q{quartile + 1}_gain_pct"] = row[
                "anchored_current_full_h_lead_quartile_gain_pct"
            ][quartile]
        flat_rows.append(flat)
    fields = sorted({field for row in flat_rows for field in row})
    normalized_rows = [{field: row.get(field, "") for field in fields} for row in flat_rows]
    atomic_csv(args.output_dir / "cells.csv", normalized_rows)
    atomic_csv(args.output_dir / "summary.csv", summaries)
    atomic_csv(args.output_dir / "literal_vs_anchored_sensitivity.csv", sensitivities)
    atomic_json(
        args.output_dir / "audit.json",
        {
            "passed": True,
            "expected_cells_per_method": args.expected_cells,
            "methods": list(BASE_METHODS),
            "primary_variant": "Literal",
            "sensitivity_variant": "Anchored",
            "runner_sha256": executed_runner_hash,
            "evaluated_elements_per_method": int(
                sum(int(row["evaluated_elements"]) for row in by_method["TAFAS-FCR"])
            ),
            "future_label_violation_count": 0,
            "mode_violation_count": 0,
            "forward_state_mutation_count": 0,
            "max_expected_base_mse_relative_difference": max_expected_base_mse,
            "max_expected_base_mae_relative_difference": max_expected_base_mae,
            "max_cross_method_base_mse_relative_difference": max_cross_mse,
            "max_cross_method_base_mae_relative_difference": max_cross_mae,
            "max_literal_frozen_base_sse_relative_difference": max_literal_base_drift,
            "max_numerical_anchor_correction_abs": max_anchor_correction,
            "summaries": summaries,
            "sensitivities": sensitivities,
        },
    )
    print(json.dumps({"summaries": summaries, "sensitivities": sensitivities}, indent=2), flush=True)


if __name__ == "__main__":
    main()
