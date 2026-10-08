# CommitCast

**Forecast commitment and causal revision for time-series forecasting.**

## Overview

CommitCast revises forecasts at successive commitments while keeping the forecasting backbone frozen. It combines commit-visible residual and forecast-history features with online ridge correction and settled-risk exposure. Every retained request is served and evaluated over its complete horizon **H**.

## Methodology

For a request issued at `i`, a commitment at `q = i + a` can use target observations only at times strictly before `q`. The served forecast is

$$
\widehat{\mathbf{y}}_{i,a}
= \mathbf{b}_{i,a} + \alpha_{i,a}\,\boldsymbol{\Delta}_{i,a},
\qquad 0 \leq \alpha_{i,a} \leq 1,
$$

where `b` is the aligned frozen reforecast, `Delta` is the clipped ridge correction, and `alpha` is the exposure. The prediction remains in force until the next commitment; the initial interval uses the frozen forecast.

- **Commit-visible features:** observed prefix residuals, forecast geometry and same-target forecast vintages.
- **Online ridge correction:** separate sufficient statistics for each lead and channel, updated in blocks using matured labels.
- **Settled-risk exposure:** a discounted OLS gate fitted only after the corresponding service intervals have completely settled.

The default method uses **12 features**, 17 trajectory sampling points, ridge penalty 25 and update block size 128. Block size controls both update timing and batch weighting. The `pg`, `p` and `g` ablations retain 8, 3 and 5 features. See [the estimator](layers/online_ridge.py), [the adapter](tta/commitcast.py) and [the protocol](docs/protocol.md).

## Requirements

Python 3.11 or 3.12 and PyTorch 2.10.0. Install a PyTorch build appropriate for your device, then run from the repository root:

```bash
python -m pip install -e .                 # CommitCast replay
python -m pip install -e ".[baselines]"    # Training and baseline experiments
```

CommitCast replay supports CPU and CUDA. Backbone training, model-aware export and baseline adaptation require CUDA. Dependency versions are specified in [pyproject.toml](pyproject.toml). Commands below use Bash syntax.

## Datasets and Backbones

| Setting | Supported values |
| --- | --- |
| Datasets | ETTh1, ETTh2, ETTm1, ETTm2, exchange_rate, weather |
| Backbones | DLinear, FreTS, iTransformer, MICN, OLS, PatchTST |
| Horizons | 96, 192, 336, 720 |

Download datasets through [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and place them at `data/<dataset>/<dataset>.csv`. Prepare a trained backbone checkpoint together with its training `config.yaml`; see [checkpoint preparation](docs/experiments.md#training-a-backbone). Data and pretrained weights are external to this source release.

## Usage

### Export Frozen Forecasts

The examples use ETTh1, DLinear, horizon 96 and seed 0. From a prepared checkpoint:

```bash
python main.py --method cosa --protocol export --cfg configs/cosa.yaml \
  --checkpoint-config checkpoints/DLinear/ETTh1_96/seed_0/config.yaml \
  --output streams/BASE_ETTh1_DLinear_h96_seed0_batch48
```

The export contains `adapter_stream.npz` and metadata. Forecast and target arrays have shape `[N, H, C]` with consecutive origins. Retain the same checkpoint and Base stream across methods.

### Run CommitCast

```bash
python scripts/run_schedules.py \
  --stream streams/BASE_ETTh1_DLinear_h96_seed0_batch48/adapter_stream.npz \
  --output results/commitcast --device cuda:0 --schedules default
```

This runs the full 12-feature method with commitments at `H/4`, `H/2` and `3H/4`. Use `--device cpu` for CPU replay or `--schedules all` for the nine static schedules. Outputs are `metrics.csv`, `request_metrics.csv` and `run.json`; use a new output directory for each run.

### Run Baselines

```bash
python main.py --method cosa --protocol fcr --profile published --cfg configs/cosa.yaml \
  --checkpoint-config checkpoints/DLinear/ETTh1_96/seed_0/config.yaml \
  --base-stream streams/BASE_ETTh1_DLinear_h96_seed0_batch48/adapter_stream.npz \
  --output results/cosa --schedules default
```

For TAFAS or PETSA, replace both `--method` and the matching `configs/<method>.yaml`. All three baselines retain their official adaptation profiles. `--protocol native` runs the official full-window evaluation; `fcr` evaluates commitment service with separate `current` and `native_revisions` policies.

## Evaluation

FCR scores exactly `H * C` coordinates per retained request against the frozen reference at the same commitment. The default common support retains `N - floor(7H/8)` requests. Native-window scores and commitment-service scores use different protocols and must be reported separately.

Observation timing, aggregation and metric definitions are in [the protocol](docs/protocol.md). Event50 uses a validation-frozen observed-input threshold. Training, Event50, configuration overrides and multi-seed experiments are described in [the experiment guide](docs/experiments.md).

## Citation

When using CommitCast, cite [this repository](https://github.com/Eastaristocrats/CommitCast) and the software version in [pyproject.toml](pyproject.toml), together with the original publications of any baselines used in your experiments.

## Acknowledgements and License

The code organization follows [COSA](https://github.com/bigbases/COSA_ICLR2026) and [TAFAS](https://github.com/kimanki/TAFAS), with baseline components from these projects and [PETSA](https://github.com/BorealisAI/PETSA).

Original CommitCast components use the [MIT license](LICENSE). Bundled upstream components retain their respective licenses; see [THIRD_PARTY.md](THIRD_PARTY.md).
