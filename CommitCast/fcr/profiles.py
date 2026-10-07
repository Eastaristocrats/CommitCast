"""Published per-cell defaults, consolidated from the official shell scripts."""
import json
from pathlib import Path


def published_overrides(method, dataset, model, horizon):
    if method == "cosa":
        return []  # configs/cosa.yaml contains the official cosa.sh profile.
    path = Path(__file__).resolve().parents[1] / "configs" / "published.json"
    profiles = json.loads(path.read_text(encoding="utf-8"))
    try:
        cell = profiles[method][model][f"{dataset}_{horizon}"]
    except KeyError as error:
        raise ValueError(
            f"No published profile for {method}/{model}/{dataset}_{horizon}; "
            "use --profile config for an explicit custom configuration"
        ) from error
    values = ["TTA.SOLVER.BASE_LR", str(cell["BASE_LR"]),
              "TTA.SOLVER.WEIGHT_DECAY", str(cell["WEIGHT_DECAY"])]
    if method == "tafas":
        values += ["TTA.TAFAS.GATING_INIT", str(cell["GATING_INIT"])]
    return values
