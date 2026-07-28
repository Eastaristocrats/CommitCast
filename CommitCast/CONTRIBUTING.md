# Contributing

Contributions should preserve the forecast-commitment protocol before changing
accuracy, speed, or convenience behavior.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Change Requirements

- Keep the upstream forecaster frozen.
- Keep the same-commit checkpoint reforecast as the primary reference.
- Do not read an unserved target before emitting its revision.
- Do not update exposure from an unsettled service interval.
- Add or update future-label mutation tests for protocol-sensitive changes.
- Do not commit datasets, checkpoints, forecast streams, results, caches, or
  machine-specific paths.
- Keep public configuration defaults synchronized across `config.py`, the
  README, tests, and experiment reports.

## Pull Requests

Describe the protocol effect, implementation effect, and verification evidence
separately. Numerical improvements are not sufficient if the information set or
reference action has changed.

