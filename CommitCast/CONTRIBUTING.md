# Contributing

Install in an isolated environment with `python -m pip install -e ".[baselines,dev]"`, then run `python -m pytest`.

Preserve complete-H support, same-commit Base matching, strict observation clocks, settled service feedback and immutable publication versions. CommitCast keeps the forecasting backbone frozen; other methods retain the modules and learning operations of their official implementation.

Keep algorithm changes separate from scoring and packaging changes. For baseline integration changes, run `scripts/check_native.py` against every affected method and retain a meaningful future-label intervention check. Compare actual predictions and model/optimizer states. Do not replace an official method with a reduced update path.

Do not commit private data, model checkpoints, forecast streams, generated results, credentials, absolute machine paths, `.vendor/`, caches or environment directories. Dependency changes belong in `pyproject.toml`.
