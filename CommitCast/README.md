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

## Optional FCR Ports

The `COSA-FCR`, `TAFAS-FCR`, and `PETSA-FCR` runners use separately cloned
upstream repositories. The runners verify the selected Git commit and source
hash before execution.

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
