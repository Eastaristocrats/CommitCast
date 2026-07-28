# Audited FCR Ports

This optional package transports COSA, TAFAS, and PETSA to the same forecast
commitment revision (FCR) contract used by CommitCast. These are **audited FCR
ports**, not the methods' native evaluation protocols and not official releases
from their authors.

## Method Contract

| Port | Preserved upstream components | FCR transport |
| --- | --- | --- |
| COSA-FCR | `SimpleOutputAdapter`, complete-tensor objective, optimizer | wait until a complete `[H,D]` truth row matures; update before the next legal service action |
| TAFAS-FCR | calibration modules, loss, optimizer, PAAS cadence and train/eval modes | replay legal partial/full transitions from the pre-commit state and emit a current-origin action |
| PETSA-FCR | low-rank calibration modules, composite loss, optimizer, PAAS cadence and modes | use the same prospective transition and exact-once service rule as TAFAS-FCR |

All three use the issue/c25/c50/c75 schedule, a same-commit frozen Base, and
non-overlapping owned intervals. No port writes an adapted action retroactively
to an earlier issue.

## Upstream Checkouts

Clone the external methods outside this repository and pin the audited commits:

```bash
git clone https://github.com/bigbases/COSA_ICLR2026 external/COSA
git -C external/COSA checkout 43a8c8da4de74d5745a8713f6130c523b7df2694

git clone https://github.com/kimanki/TAFAS external/TAFAS
git -C external/TAFAS checkout 139bf980671da4daad728a0fc21d8df508b9203d

git clone https://github.com/BorealisAI/PETSA external/PETSA
git -C external/PETSA checkout 87853d888e98311ac94e64be920d17b57143b20c
```

Install each upstream project's dependencies in the environment used for that
port. Their licenses contain non-commercial restrictions; see
[`THIRD_PARTY.md`](THIRD_PARTY.md).

## Commands

COSA-FCR consumes frozen forecast streams:

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

TAFAS-FCR and PETSA-FCR additionally require the official checkpoint-backed
configuration and raw dataset inputs used by the upstream runtime:

```bash
python scripts/fcr_ports/run.py \
  --method TAFAS-FCR \
  --repo external/TAFAS \
  --cfg checkpoints/example/config.yaml \
  --base_stream streams/BASE_ETTh1_DLinear_h96_seed0_batch48/adapter_stream.npz \
  --output_json results/tafas_fcr/ETTh1_DLinear_h96_seed0.json
```

Replace `TAFAS-FCR` and the repository path with `PETSA-FCR` and its checkout
for PETSA. The full matrix uses one cell process per method/configuration so
that official module imports and CUDA state stay isolated.

For a portable batch run, copy `configs/fcr_ports_manifest.example.yaml`, fill
in the relative artifact paths, and run:

```bash
python scripts/fcr_ports/run_matrix.py \
  --manifest configs/fcr_ports_manifest.yaml \
  --shard_index 0 \
  --num_shards 1 \
  --resume
```

The launcher rejects duplicate cell IDs, validates every port's required
inputs, supports deterministic modulo sharding, and records the input-manifest
hash without embedding local paths in portable audit metadata. The example YAML
is an input configuration template containing no experimental measurements.

Validate a manifest without launching external code:

```bash
python scripts/fcr_ports/run_matrix.py \
  --manifest configs/fcr_ports_manifest.example.yaml \
  --dry_run
```

Protocol-only self-tests do not require datasets or checkpoints:

```bash
python scripts/fcr_ports/run_tafas_petsa_cell.py --self_test_only
python scripts/fcr_ports/run_cosa.py --repo external/COSA --self_test
```

## Artifact Policy

Generated cell metrics, aggregate CSV files, logs, frozen streams, and audit
reports are deliberately excluded from the source repository. Write them under
the ignored `results/` or `outputs/` directories during a run. If evaluation
artifacts are released, publish them as a separate versioned archive with
checksums and a pointer from the paper or repository release notes.

Exact end-to-end regeneration additionally requires the pinned upstream
checkouts, datasets, checkpoint-owned configurations, and frozen forecast
streams. Those external artifacts are not embedded in the Git repository.
