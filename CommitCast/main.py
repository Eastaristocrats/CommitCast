"""Main CommitCast entrypoint."""

from __future__ import annotations

import os
import sys


def main() -> None:
    if any(arg == "--method" or arg.startswith("--method=") for arg in sys.argv[1:]):
        from scripts.run_baseline import main as run_baseline
        run_baseline()
        return
    from datasets.build import update_cfg_from_dataset
    from models.build import build_model, load_best_model
    from predictor import Predictor
    from tta.commitcast import build_adapter
    from utils.misc import set_devices, set_seeds
    from utils.parser import load_config, parse_args
    args = parse_args()
    cfg = load_config(args)

    # Keep the same config/update/build orchestration used by COSA. Stream
    # shapes remain authoritative, so this hook only records a single filter.
    if cfg.DATA.NAME and "," not in str(cfg.DATA.NAME):
        update_cfg_from_dataset(cfg, str(cfg.DATA.NAME))

    os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    set_devices(cfg.VISIBLE_DEVICES)
    with open(os.path.join(cfg.RESULT_DIR, "config.yaml"), "w", encoding="utf-8") as handle:
        handle.write(cfg.dump())
    set_seeds(int(cfg.SEED))

    model = build_model(cfg)
    if cfg.TRAIN.ENABLE:
        from trainer import build_trainer
        build_trainer(cfg, model).train()

    if cfg.TTA.ENABLE or cfg.TEST.ENABLE:
        model = load_best_model(cfg, model)

    adapter = None
    if cfg.TTA.ENABLE:
        adapter = build_adapter(cfg, model)
        adapter.adapt()
        adapter.count_parameters()

    if cfg.TEST.ENABLE:
        Predictor(cfg, model, adapter=adapter).predict()


if __name__ == "__main__":
    main()
