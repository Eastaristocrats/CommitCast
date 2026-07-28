# CommitCast

CommitCast is a gradient-free output-space adapter for time-series forecasting
under delayed feedback. It revises the unserved portion of an issued forecast
when new observations become available while keeping the forecasting model
frozen.

## Overview

For an origin `i`, commit delay `d`, and remaining lead `r`, CommitCast uses the
frozen forecast issued at the current commit as its reference:

```text
base(i, d, r) = forecast(i + d, r)
```

The adapter estimates a bounded residual correction from matured observations,
forecast geometry, and previous forecasts of the same target. It does not
update the forecasting model or use labels that are unavailable at the current
commit.

## Method

CommitCast keeps the forecasting model read-only and operates on exported
forecast trajectories. At each commit it:

1. selects the frozen forecast issued at the current origin;
2. constructs trajectory and matured-residual features;
3. estimates a lead/channel correction with online ridge sufficient statistics;
4. clips the proposal to the configured feature, coefficient, and correction
   limits;
5. applies a settled-risk exposure in `[0, 1]`.

The default configuration in `config.py` is:

| Option | Value |
| --- | ---: |
| Commit fractions | `0.25, 0.50, 0.75` |
| Trajectory points | `17` |
| Ridge coefficient | `25.0` |
| Ridge warm-up rows | `16` |
| Feature clip | `6.0` |
| Target clip | `6.0` |
| Coefficient clip | `0.4` |
| Correction clip | `0.5` |
| Settled-risk half-life | `512` rows |
| Settled-risk warm-up | `16` rows |
| Settled-risk ridge | `1e-6` |

## Requirements

- Python 3.11 or later
- PyTorch
- NumPy
- pandas
- PyYAML
- YACS

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

Use the PyTorch wheel that matches the local CPU or CUDA environment.

## Input Data

CommitCast reads frozen forecast streams stored as `adapter_stream.npz`:

```text
pred: float32 array with shape [N, H, C]
true: float32 array with shape [N, H, C]
```

`pred[i]` is the forecast issued at origin `i`, and `true[i]` is its aligned
target window. Consecutive target windows must overlap by one time step.

The expected directory layout is:

```text
streams/
  BASE_<dataset>_<model>_h<horizon>/
    adapter_stream.npz
```

Existing NumPy arrays can be converted with:

```bash
python examples/export_stream.py \
  --pred path/to/pred.npy \
  --true path/to/true.npy \
  --output streams/BASE_ETTh1_DLinear_h96/adapter_stream.npz
```

The loader validates shape, finiteness, chronological overlap, and stream
metadata before adaptation. Each processed stream is recorded with its shape
and SHA-256 value.

## Usage

Create a small example stream:

```bash
python examples/make_toy_stream.py
```

Run CommitCast on CPU:

```bash
python main.py \
  DEVICE cpu \
  STREAM.ROOT streams \
  RESULT_DIR results/commitcast
```

Filter datasets, models, and horizons with COSA-style configuration overrides:

```bash
python main.py \
  STREAM.ROOT streams \
  DATA.NAME ETTh1,ETTm1,weather \
  MODEL.NAME DLinear,PatchTST,iTransformer \
  DATA.PRED_LEN 96,192,336,720 \
  RESULT_DIR results/commitcast
```

Shell wrappers are available for Linux and Windows:

```bash
bash scripts/commitcast.sh
```

```powershell
.\scripts\commitcast.ps1
```

## Outputs

Each run writes the following files under `RESULT_DIR`:

```text
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

`event_metrics.csv` contains Base, raw proposal, and exposed CommitCast metrics
for every service interval. `stream_manifest.json` contains stream names,
shapes, and hashes without local absolute paths.

Generated streams, checkpoints, result tables, logs, and external repositories
are ignored by `.gitignore` and should not be committed with the source.

## Information Boundary

At a commit for origin `i` and delay `d`, the adapter may use:

```text
the forecast issued at origin i
the frozen reforecast issued at origin i + d
the matured prefix before the current commit
forecast vintages available no later than the current commit
ridge rows whose targets are already observable
settled-risk rows whose complete served intervals are observable
```

It may not use:

```text
unobserved future labels
losses from unsettled service intervals
future forecast vintages
updated forecasting-model parameters
retroactively replaced predictions
```

## Optional FCR Ports

The `COSA-FCR`, `TAFAS-FCR`, and `PETSA-FCR` runners use separately cloned
upstream repositories. The runners verify the selected Git commit and source
hash before execution.

| Port | Upstream component kept | FCR service wrapper |
| --- | --- | --- |
| COSA-FCR | output adapter, objective, optimizer | updates only after a complete target row matures |
| TAFAS-FCR | calibration, optimizer, PAAS cadence and modes | replays legal partial/full transitions before current-origin service |
| PETSA-FCR | low-rank calibration, objective, optimizer and cadence | uses the same prospective exact-once service rule |

These ports use the common issue/c25/c50/c75 service schedule. They are
compatibility runners around the pinned upstream implementations rather than
changes to the CommitCast core.

```bash
git clone https://github.com/bigbases/COSA_ICLR2026 external/COSA
git -C external/COSA checkout 43a8c8da4de74d5745a8713f6130c523b7df2694

git clone https://github.com/kimanki/TAFAS external/TAFAS
git -C external/TAFAS checkout 139bf980671da4daad728a0fc21d8df508b9203d

git clone https://github.com/BorealisAI/PETSA external/PETSA
git -C external/PETSA checkout 87853d888e98311ac94e64be920d17b57143b20c
```

Run each optional method in an environment that satisfies the dependency
versions documented by its pinned upstream repository.

Run COSA-FCR on frozen forecast streams:

```bash
python scripts/fcr_ports/run.py \
  --method COSA-FCR \
  --repo external/COSA \
  --stream_root streams \
  --output_dir results/cosa_fcr \
  -- \
  --device cuda:0 \
  --metric_protocol latest_commit_four_interval
```

TAFAS-FCR and PETSA-FCR additionally use an upstream configuration, dataset,
checkpoint, and matching frozen Base stream:

```bash
python scripts/fcr_ports/run.py \
  --method TAFAS-FCR \
  --repo external/TAFAS \
  --cfg checkpoints/example/config.yaml \
  --base_stream streams/BASE_ETTh1_DLinear_h96/adapter_stream.npz \
  --output_json results/tafas_fcr/example.json
```

Use `PETSA-FCR` and `external/PETSA` for the PETSA runner.

Generate the configured experiment grid:

```bash
python scripts/fcr_ports/generate_manifest.py \
  --output path/to/fcr_ports_manifest.yaml \
  --external_root external \
  --data_root data \
  --streams_root streams \
  --checkpoints_root checkpoints \
  --results_root results/fcr_ports
```

Check the manifest without starting the external methods:

```bash
python scripts/fcr_ports/run_matrix.py \
  --manifest path/to/fcr_ports_manifest.yaml \
  --dry_run
```

Run one shard:

```bash
python scripts/fcr_ports/run_matrix.py \
  --manifest path/to/fcr_ports_manifest.yaml \
  --shard_index 0 \
  --num_shards 1 \
  --resume
```

The setting grid is defined in `configs/fcr_ports_experiment.yaml`, and the
method parameters are defined in `configs/fcr_ports_hparams.yaml`.

The matrix runner validates required fields and unique cell IDs, supports
deterministic modulo sharding, and checks protocol audit fields before a
completed result is skipped by `--resume`. Audit JSON is written only when
`--audit_json` is supplied.

## Project Structure

```text
CommitCast/
  config.py
  main.py
  predictor.py
  trainer.py
  requirements.txt
  configs/
  datasets/
  examples/
  layers/
  models/
  scripts/
  tests/
  tta/
  utils/
```

The execution path follows the same top-level organization used by COSA and
TAFAS:

```text
main.py
  -> datasets.build
  -> models.build
  -> tta.commitcast
  -> Predictor
```

## Tests

```bash
python -m pip install pytest==9.0.2 ruff==0.12.7
python -m ruff check .
python -m pytest
```

Protocol-only checks for the optional ports:

```bash
python scripts/fcr_ports/run_tafas_petsa_cell.py --self_test_only
python scripts/fcr_ports/run_cosa.py --repo external/COSA --self_test
```

The tests cover stream validation, configuration overrides, full
build/adapt/predict execution, service-interval ownership, unavailable-label
mutation checks, settled-risk timing, port command isolation, upstream identity
checks, and matrix generation.

## License

The CommitCast core is licensed under the [MIT License](LICENSE).

COSA and PETSA use CC BY-NC-SA 4.0 in the pinned repositories. TAFAS uses the
non-commercial terms in its pinned `LICENSE`. The optional
`run_tafas_petsa_cell.py` compatibility layer includes a PETSA-compatible
objective and follows the corresponding upstream non-commercial terms. The
upstream license files control use of those methods and their checkpoints.

## Acknowledgements

The repository layout and build/adapt/predict interfaces follow the public
organization of [COSA](https://github.com/bigbases/COSA_ICLR2026) and
[TAFAS](https://github.com/kimanki/TAFAS).
