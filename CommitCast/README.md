# CommitCast: Same-Commit Forecast Revision under Delayed Feedback

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Anonymous implementation of **CommitCast**, a gradient-free revision layer for
frozen time-series forecasters.

CommitCast addresses a specific online decision: after a forecast has been
issued and part of its target window has matured, how should the still-unserved
portion be revised? The candidate is compared with the frozen checkpoint
reforecast available at the same commit time. This keeps the benefit of newer
input context separate from the contribution of the revision method.

## Overview

```text
origin i                    commit i+d                         future
   |                            |                                |
   |-- issue forecast ----------|                                |
   |                            |-- frozen reforecast -----------|
   |-- matured prefix labels -->|                                |
   |                            |-- CommitCast revision -------->|
                                ^
                                only information legal here
```

The upstream forecasting model remains frozen. CommitCast reads:

- chronological frozen forecasts;
- absolute target alignment implied by origin and lead;
- labels that have matured before the current commitment;
- previous forecasts of the same target from different origins.

It does not require gradients, optimizer state, model parameters, or internal
representations.

## Method

For origin \(i\), commit delay \(d\), and remaining lead \(r\), the matched
frozen action is:

\[
\hat y^{\mathrm{base}}_{i,d,r}=\hat y_{i+d,r}.
\]

The adapter estimates only the residual left by this same-commit reforecast:

\[
e_{i,d,r}=y_{i,d+r}-\hat y_{i+d,r}.
\]

Its state combines matured prefix residuals, current forecast geometry, and the
same-target forecast-vintage path. A lead/channel-specific ridge estimator
produces a correction proposal. The SRS-512 controller then applies a bounded
scalar exposure learned only from fully settled service intervals:

\[
\tilde y=\hat y^{\mathrm{base}}+\alpha
(\hat y^{\mathrm{proposal}}-\hat y^{\mathrm{base}}),\qquad
\alpha\in[0,1].
\]

The frozen configuration uses three commitments at
\(0.25H, 0.50H, 0.75H\), 12 trajectory/state features, lead-wise label
maturity, ridge coefficient clipping, and an exponentially discounted
settled-risk controller with a 512-row half-life.

## Repository Structure

```text
CommitCast/
  README.md
  REPRODUCIBILITY.md
  CONTRIBUTING.md
  THIRD_PARTY.md
  FCR_PORTS.md
  LICENSE
  requirements.txt
  requirements-verified.txt
  config.py
  main.py
  predictor.py
  trainer.py
  datasets/
    build.py
    loader.py
  layers/
    online_ridge.py
    exposure.py
  models/
    build.py
    forecast.py
  configs/
    fcr_ports.yaml
    fcr_ports_manifest.example.yaml
  tta/
    commitcast.py
    ports/
      build.py
  utils/
    metrics.py
    misc.py
    parser.py
  examples/
    export_stream.py
    make_toy_stream.py
  scripts/
    commitcast.sh
    commitcast.ps1
    fcr_ports/
      run.py
      run_cosa.py
      run_tafas_petsa_cell.py
      run_matrix.py
    test.sh
    test.ps1
  tests/
```

The layout follows COSA's public repository contract directly:

```text
main.py
  -> datasets.build.build_dataset
  -> models.build.build_model / load_best_model
  -> tta.commitcast.build_adapter
  -> adapter.adapt
  -> Predictor.predict
```

`trainer.py` retains the same top-level training boundary as COSA, but rejects
training by design: CommitCast is a black-box revision layer and the upstream
forecaster must remain frozen. The `models/` facade exposes exported checkpoint
forecasts as the base model, while `layers/` contains the two stateful
closed-form components used by the adapter. Research workspaces, paper sources,
private launchers, caches, and raw generated results are not part of this release.
The organization follows the public entrypoint/factory convention; the
CommitCast algorithm code is an independent implementation released under this
repository's MIT license. Architecture attribution and the policy for optional
baseline integrations are recorded in [`THIRD_PARTY.md`](THIRD_PARTY.md).

The optional `tta/ports` registry follows the same factory convention without
importing third-party code into the default process. Each port is executed by a
separate runner under `scripts/fcr_ports`; see
[`FCR_PORTS.md`](FCR_PORTS.md).

## Requirements

```bash
python -m pip install -r requirements.txt
```

For the exact direct-dependency versions used by the release QA run:

```bash
python -m pip install -r requirements-verified.txt
```

Choose the PyTorch CPU/CUDA wheel appropriate for the target machine. The
verified CUDA build and the limits of this dependency snapshot are recorded in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

Recommended:

- Python 3.10 or newer;
- PyTorch 1.10 or newer;
- CUDA for full benchmark matrices; CPU is sufficient for the toy example.

## Forecast Stream Format

CommitCast is black-box with respect to the forecaster. Export two aligned
arrays into `adapter_stream.npz`:

```text
pred: [N, H, C]
true: [N, H, C]
```

where `pred[i]` is the frozen \(H\)-step forecast issued at chronological
origin `i`, and `true[i]` is its aligned target window. The same-commit
reforecast for delay `d` is recovered as:

```python
checkpoint = pred[d:, :H-d]
target = true[:-d, d:H]
```

Use this directory naming convention when possible:

```text
streams/
  BASE_<dataset>_<backbone>_h<horizon>_seed<seed>_batch<batch>/
    adapter_stream.npz
```

If your forecasts and targets already exist as `.npy` arrays:

```bash
python examples/export_stream.py \
  --pred path/to/pred.npy \
  --true path/to/true.npy \
  --output streams/BASE_ETTh1_DLinear_h96_seed0_batch48/adapter_stream.npz
```

Checkpoint inference is deliberately kept outside the adapter. This makes the
frozen forecast tensor, target alignment, and same-commit reference directly
auditable across different model codebases and hosted forecasting APIs.

## Quick Start

Create a deterministic toy stream:

```bash
python examples/make_toy_stream.py
```

Run CommitCast:

```bash
python main.py \
  DEVICE cpu \
  STREAM.ROOT streams \
  RESULT_DIR results/toy
```

Run a filtered benchmark subset with COSA-style dotlist overrides:

```bash
python main.py \
  STREAM.ROOT streams \
  DATA.NAME ETTh1,ETTm1,weather \
  MODEL.NAME DLinear,PatchTST,iTransformer \
  DATA.PRED_LEN 96,192,336,720 \
  RESULT_DIR results/commitcast
```

Shell wrappers are also provided:

```bash
bash scripts/commitcast.sh
```

```powershell
.\scripts\commitcast.ps1
```

## Optional FCR Ports

`COSA-FCR`, `TAFAS-FCR`, and `PETSA-FCR` are included as optional audited
baselines. They are disabled by default, require separately installed pinned
upstream checkouts, and do not change the CommitCast execution path.

See [FCR_PORTS.md](FCR_PORTS.md) for upstream commits, licenses, exact protocol
semantics, per-method commands, and external artifact requirements. Generated
cell metrics and aggregate tables belong under the ignored `results/` tree or
in a separately archived artifact release; they are not committed with source.

## Outputs

Each run writes:

```text
results/commitcast/
  config.yaml
  event_metrics.csv
  summary.csv
  breakdown_by_stream.csv
  breakdown_by_dataset.csv
  breakdown_by_backbone.csv
  breakdown_by_horizon.csv
  breakdown_by_commit.csv
  stream_manifest.json
```

`stream_manifest.json` records stream shapes and SHA-256 hashes without storing
local absolute paths.

## Protocol Boundary

At commitment `i+d`, a prediction may use:

```text
pred[i, d:]
pred[i+d, :]
true[i, :d] - pred[i, :d]
forecast vintages pred[i+u, d+r-u] for u <= d
ridge rows whose target time is strictly earlier than the current commit
exposure rows whose complete served segment has settled
```

It may not use:

```text
true[i, d:] before the revision is emitted
loss from an unsettled served segment
future forecast vintages
updated backbone parameters
```

The tests include counterfactual future-label mutation checks for both the
residual estimator and the exposure controller.

## Tests

```bash
python -m pytest
```

Continuous integration runs the same suite on Python 3.10 and 3.12. Protocol-
sensitive contributions must follow [CONTRIBUTING.md](CONTRIBUTING.md), and the
frozen public settings and artifact contract are recorded in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

For a full public-release smoke test:

```bash
python examples/make_toy_stream.py
python main.py DEVICE cpu STREAM.ROOT streams
python -m pytest
```

## Citation

During anonymous review:

```bibtex
@misc{anonymous2026commitcast,
  title  = {CommitCast: Same-Commit Forecast Revision for Test-Time Adaptation
            in Time-Series Forecasting under Delayed Feedback},
  author = {Anonymous},
  year   = {2026},
  note   = {Anonymous submission}
}
```

Replace the anonymous citation metadata after the review period.

## License

This implementation is released under the [MIT License](LICENSE).
