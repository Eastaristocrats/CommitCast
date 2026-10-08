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

Checkpoint configuration transfers the architecture, data split, normalization and seed, including each normalization module's saved checkpoint directory. The data root remains configurable. Explicit `KEY VALUE` options after `--` take precedence. If a normalization module is enabled, its own checkpoint must also be present. After relocating checkpoints, override `REVIN.TRAIN.CHECKPOINT_DIR`, `SAN.TRAIN.CHECKPOINT_DIR` or `DISHTS.TRAIN.CHECKPOINT_DIR` as appropriate, in addition to the forecasting checkpoint directory. Missing model or normalization checkpoints cause an error.

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

Add the same `--event-threshold` argument to a baseline `--protocol fcr` command. It adds Event50 to the requested schedules. Both methods must use the same observed-input timeline, encoder prehistory and frozen threshold. The fitting command checks validation metadata against the stream's shape, stride and encoder context and retains that metadata in the threshold record. Both replay runners copy the full threshold record into `run.json`.

Split metadata is a declaration, not independent proof of the data source. Never fit the threshold using test observations. Keep the original validation export and its configuration with experiment records. Use one declared threshold per comparison cell and report Event50 separately from the static schedules. Existing validation-frozen threshold files remain accepted; older files may lack source metadata.

## Validation selection

This optional procedure applies to CommitCast parameter selection. TAFAS, PETSA and COSA retain their fixed official execution profiles in the declared comparison; they are not sent through this selection procedure. Use a validation forecast stream for CommitCast and keep the checkpoint, support, Base reference and schedule fixed while comparing candidates. `--split val` remains available for evaluating a baseline's fixed official profile without retuning it.

The selection helper consumes a CSV with these columns:

```text
cell,method,policy,schedule,candidate,split,requests,evaluated_elements,base_sse,sse,mse,published_default
```

Each file describes one comparison group, all rows have `split=val`, and candidate names are unique. Include both a `Base` candidate whose error equals the frozen reference and a candidate for CommitCast's declared default profile. The field `published_default` identifies that reference profile: mark exactly one non-Base candidate with `true` (or `1`); other rows use `false`, `0` or a blank field. Counts must be positive integers.

```bash
python scripts/select_validation.py \
  --candidates results/validation_candidates.csv \
  --output configs/selected_candidate.json
```

The helper checks group identity, common support, Base totals and MSE consistency. It selects the lowest MSE rounded to 10 decimal places, prioritizing Base and then the marked default in a tie. It writes the chosen candidate and comparison identity.

Candidate execution and application of the chosen configuration are separate steps. The helper does not verify how the records were produced, run a search or automatically configure test evaluation. If this optional CommitCast procedure is used, record its candidate-to-configuration mapping, search budget and selected settings, then apply the locked choice unchanged to test data. The supplied default commands do not run a search.

For CommitCast, save the selected parameters in a YAML using the `TTA.COMMITCAST` keys in [config.py](../config.py). Apply that file to the same named schedule and support on the test stream:

```bash
python scripts/run_schedules.py \
  --stream streams/BASE_ETTh1_DLinear_h96_seed0_batch48/adapter_stream.npz \
  --cfg configs/selected_commitcast.yaml --schedules default --support common \
  --output results/commitcast_selected --device cuda:0
```

`configs/selected_commitcast.yaml` is the experiment's saved selection, not a supplied tuned profile. Optional pairs such as `-- TTA.COMMITCAST.RIDGE_LAMBDA 25.0` override YAML method values; use them only to apply the previously recorded choice. Explicit `--feature-mode` and `--device` flags take precedence. `--schedules` controls timing even if the YAML contains `FRACTIONS`. A selected Base candidate means serving the frozen reference; it must not be relabeled as an adapted result.

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

The declared comparison uses fixed official execution profiles. TAFAS and PETSA obtain dataset/backbone/horizon-specific learning rates and weight decay from [configs/published.json](../configs/published.json), which contains all 144 published cells for each method. TAFAS also reads its gating initialization there. PETSA's run scripts take rank, loss alpha and gating initialization as arguments; this release retains the official README's example values of 16, 0.2 and 0.02, respectively. These are the declared execution arguments, not a claim that PETSA's bare `config.py` has the same fallback values. TAFAS and PETSA both retain their official one-step setting.

COSA follows the current official `scripts/cosa.sh` profile: PAAS, 3 adaptation steps, context size 10, configured batch size 48, Linear adapter, fast adaptation, adaptive learning rate, per-batch learning-rate reset, POGT, partial adaptation and prediction adjustment. Only the current COSA implementation is included.

For the declared comparison, keep the supplied method YAML, `--profile published` (the default) and the official adaptation settings. Dataset locations, checkpoint paths and device selection may be changed to match the environment.

Optional custom experiments can use `--profile config` to skip published per-cell overrides. Explicit `KEY VALUE` options after `--` always take precedence. Such runs must be labeled separately from the fixed official-profile comparison. The original capabilities remain exposed: for example, `TTA.COSA.PAAS False` selects fixed batches and `TTA.COSA.ADAPTER_TYPE MLP` selects its MLP adapter. A separate custom run is:

```bash
python main.py --method cosa --protocol native --cfg configs/cosa.yaml \
  --checkpoint-config checkpoints/DLinear/ETTh1_96/seed_0/config.yaml \
  --output results/cosa_fixed_mlp \
  -- TTA.COSA.PAAS False TTA.COSA.ADAPTER_TYPE MLP
```

The baseline adapters require CUDA. TAFAS and PETSA use `STEPS=1` in the official-profile comparison, while COSA uses `STEPS=3`. Outside that comparison, the upstream TAFAS/PETSA nondefault multi-step branches can raise a repeated-backward error under PyTorch 2.10. This optional-branch limitation does not change the declared one-step setup. Requested steps are not silently reduced and losses are not substituted.

An enabled normalization component requires its own compatible checkpoint. The shared framework retains NST, RevIN, SAN and TAFAS's DishTS branch. Additional TAFAS backbone source files are retained; the six-backbone table in the README defines the supplied experiment profiles.

Use matching timing scopes for efficiency comparisons and account for temporal dependence when estimating uncertainty, as specified in the [protocol](protocol.md). Source provenance and license boundaries are in [THIRD_PARTY.md](../THIRD_PARTY.md).
