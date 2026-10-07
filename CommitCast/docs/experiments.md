# Experiment guide

[Back to README](../README.md) | [Evaluation protocol](protocol.md)

Run commands from the repository root. Paths below use ETTh1, DLinear, horizon 96 and seed 0 as a concrete example. Keep a separate checkpoint, forecast stream and output directory for each experiment.

## Training a backbone

Install the baseline dependencies, prepare `data/ETTh1/ETTh1.csv`, and train using the shared TAFAS trainer:

```bash
python main.py --method cosa --protocol train --cfg configs/cosa.yaml \
  --output results/train/ETTh1_DLinear_h96_s0 \
  -- TRAIN.CHECKPOINT_DIR checkpoints/DLinear/ETTh1_96/seed_0 \
     DATA.NAME ETTh1 DATA.PRED_LEN 96 MODEL.NAME DLinear \
     SEED 0 SOLVER.MAX_EPOCH 30
```

The resolved configuration is saved in the training output directory; the selected model is saved as `checkpoint_best.pth` in `TRAIN.CHECKPOINT_DIR`. Copy the saved configuration beside that checkpoint to use the layout in the README:

```bash
cp results/train/ETTh1_DLinear_h96_s0/config.yaml checkpoints/DLinear/ETTh1_96/seed_0/config.yaml
```

In PowerShell, use `Copy-Item` in place of `cp`. Alternatively, pass the original training `config.yaml` to `--checkpoint-config` and explicitly override `TRAIN.CHECKPOINT_DIR` with the model directory.

Checkpoint configuration transfers the architecture, data split, normalization and seed. The data root remains configurable. Explicit `KEY VALUE` options after `--` take precedence. If a normalization module is enabled, its own checkpoint must also be present. Missing model or normalization checkpoints cause an error.

The root `trainer.py` trains forecasting backbones and their optional normalization modules. Use `--method ... --protocol train` to select the forecasting configuration. CommitCast itself consumes the frozen forecasts and does not retrain the forecasting model.

## Entry points and outputs

| Entry point | Purpose | Main outputs |
| --- | --- | --- |
| `scripts/run_schedules.py` | CommitCast under one or more complete-H schedules | `metrics.csv`, `request_metrics.csv`, `run.json` |
| `main.py` without `--method` | YACS-configured CommitCast replay | `summary.csv`, `event_metrics.csv`, `stream_manifest.json`, `config.yaml` |
| `main.py --method ... --protocol export` | Shared frozen forecasts | `adapter_stream.npz`, `metadata.json`, `config.yaml` |
| `main.py --method ... --protocol fcr` | Baseline commitment evaluation | Schedule/request metrics, current forecasts, native publications and run metadata |
| `main.py --method ... --protocol native` | Official full-window evaluation | `native_metrics.json`, `config.yaml` and any upstream diagnostic outputs |

The YACS entry point evaluates the quarter-horizon schedule by default and emits Base, the ungated proposal and CommitCast summaries. The multi-schedule runner emits the final served method and a matched Base reference for each selected schedule. Keep the support profile and output schema explicit when aggregating results.

Generated arrays, logs and result tables are experiment outputs and are excluded from the source distribution.

## Event50

First export a validation stream from the same trained checkpoint:

```bash
python main.py --method cosa --protocol export --cfg configs/cosa.yaml \
  --checkpoint-config checkpoints/DLinear/ETTh1_96/seed_0/config.yaml \
  --split val --output streams/validation/ETTh1_DLinear_h96_s0
```

Fit and freeze the input-score threshold on that validation stream:

```bash
python scripts/fit_event_threshold.py \
  --stream streams/validation/ETTh1_DLinear_h96_s0/adapter_stream.npz \
  --metadata streams/validation/ETTh1_DLinear_h96_s0/metadata.json \
  --output configs/event_ETTh1_DLinear_h96_s0.json
```

Apply it to the test stream exported in the README:

```bash
python scripts/run_schedules.py \
  --stream streams/BASE_ETTh1_DLinear_h96_seed0_batch48/adapter_stream.npz \
  --output results/commitcast_event --device cuda:0 \
  --event-threshold configs/event_ETTh1_DLinear_h96_s0.json
```

Add the same `--event-threshold` argument to a baseline `--protocol fcr` command. It adds Event50 to the requested schedules. Both methods must use the same observed-input timeline, encoder prehistory and frozen threshold. The threshold file declares validation selection; never fit it using test observations. Use one declared threshold per comparison cell and report Event50 separately from the static schedules.

## Validation selection

Use `--split val` for baseline validation and a validation forecast stream for CommitCast. Keep the checkpoint, support, Base reference, schedule and policy fixed while comparing candidate settings.

The selection helper consumes a CSV with these columns:

```text
cell,method,policy,schedule,candidate,split,requests,evaluated_elements,base_sse,sse,mse,published_default
```

Each file describes one comparison group, all rows have `split=val`, and candidate names are unique. Include both a `Base` candidate whose error equals the frozen reference and a candidate for the published starting profile. `published_default` marks that starting profile.

```bash
python scripts/select_validation.py \
  --candidates results/validation_candidates.csv \
  --output configs/selected_candidate.json
```

The helper checks group identity, common support, Base totals and MSE consistency. It selects the lowest MSE rounded to 10 decimal places, prioritizing Base and then the published default in a tie. It writes the chosen candidate and comparison identity.

Candidate execution and application of the chosen configuration are separate steps. The helper does not verify how the records were produced, run a search or automatically configure test evaluation. Record the candidate-to-configuration mapping, search budget and selected settings, then apply the locked choice unchanged to test data.

## Multiple datasets, backbones, horizons and seeds

Generate a manifest and inspect its planned commands:

```bash
python scripts/run_matrix.py --manifest configs/matrix.json --generate --seeds 0,1,2
```

The manifest contains six datasets, six backbones and four horizons per seed. It expects checkpoint configurations under `checkpoints/<model>/<dataset>_<horizon>/seed_<seed>/config.yaml` and separate exported streams for each cell. Adjust these paths to the prepared inputs, then execute:

```bash
python scripts/run_matrix.py --manifest configs/matrix.json --execute --device cuda:0
```

Generation alone does not train or evaluate a model. Execution runs fixed configurations and stops on missing inputs or a failed command. The default methods are `full,pg,tafas,petsa,cosa`. For CommitCast-only CPU replay, set `--methods full,pg --device cpu`. Event50 thresholds are supplied through per-cell commands; the matrix runner has no threshold-file option.

Each training seed needs its own trained checkpoint. Changing only an adaptation seed does not create an independent training replicate. Store the exact commands, checkpoint configurations, validation choices and table-to-cell mapping with the experiment records.

## Baseline profiles and compatibility

The independent method implementations are `tta/tafas.py`, `tta/petsa.py` and `tta/cosa.py`. Their losses, optimizers, adaptation loops and prediction-adjustment paths retain the respective official implementations. Identical model and layer code is shared across methods.

TAFAS and PETSA obtain dataset/backbone/horizon-specific learning rates and weight decay from [configs/published.json](../configs/published.json), which contains all 144 published cells for each method. TAFAS also reads its gating initialization there. PETSA's starting rank, loss alpha and gating initialization are 16, 0.2 and 0.02, following its upstream README example. These are starting profiles, not tuned claims for every evaluation cell.

COSA follows the current official `scripts/cosa.sh` profile: PAAS, 3 adaptation steps, context size 10, configured batch size 48, Linear adapter, fast adaptation, adaptive learning rate, per-batch learning-rate reset, POGT, partial adaptation and prediction adjustment. Only the current COSA implementation is included.

Use `--profile config` to skip published per-cell overrides. Explicit `KEY VALUE` options after `--` always take precedence. For COSA, `TTA.COSA.PAAS False` selects fixed batches and `TTA.COSA.ADAPTER_TYPE MLP` selects its MLP adapter. Use one `--` followed by all settings, for example:

```bash
python main.py --method cosa --protocol native --cfg configs/cosa.yaml \
  --checkpoint-config checkpoints/DLinear/ETTh1_96/seed_0/config.yaml \
  --output results/cosa_fixed_mlp \
  -- TTA.COSA.PAAS False TTA.COSA.ADAPTER_TYPE MLP
```

The baseline adapters require CUDA. TAFAS and PETSA retain their upstream `STEPS=1` default; their nondefault multi-step branches can raise a repeated-backward error under PyTorch 2.10. Requested steps are not silently reduced and losses are not substituted. COSA's multi-step path is available.

An enabled normalization component requires its own compatible checkpoint. The shared framework retains NST, RevIN, SAN and TAFAS's DishTS branch. Additional TAFAS backbone source files are retained; the six-backbone table in the README defines the supplied experiment profiles.

Use matching timing scopes for efficiency comparisons and account for temporal dependence when estimating uncertainty, as specified in the [protocol](protocol.md). Source provenance and license boundaries are in [THIRD_PARTY.md](../THIRD_PARTY.md).
