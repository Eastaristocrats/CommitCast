# Reproducibility Contract

## Release QA Environment

The public release was validated on Python 3.12.10 with NumPy 2.3.5,
pandas 3.0.1, PyYAML 6.0.3, PyTorch 2.10.0+cu128, YACS 0.1.8, and
pytest 9.0.2 on Windows 11. The exact direct dependency versions are listed in
`requirements-verified.txt`; the ordinary `requirements.txt` retains portable
minimum versions.

This direct-dependency snapshot is not a lock of every transitive package and
does not replace the pinned upstream environments required by COSA-FCR,
TAFAS-FCR, and PETSA-FCR. Select the platform-specific PyTorch wheel matching
the target CPU/CUDA runtime.

## Verification Status

The release implementation has been checked at three levels:

1. the unit and integration suite passes;
2. the 12-dimensional Final12 proposal and SRS-512 exposure agree with the
   frozen research implementation to floating-point tolerance on synthetic
   data;
3. on an archived real stream, every c25/c50/c75 revision agrees with the
   research implementation to within `9.54e-7`, and the complete lifecycle
   metrics agree to numerical precision.

This establishes implementation equivalence for a supplied stream. It does not,
by itself, reconstruct every upstream forecast tensor used in the paper.

## Frozen Public Configuration

The defaults in `config.py` define the public CommitCast configuration:

```text
commit fractions       0.25, 0.50, 0.75
trajectory points      17
Final12 features       12
ridge lambda           25
ridge warmup rows      16
feature clip           6
target clip            6
coefficient clip       0.4
correction clip        0.5
SRS half-life rows     512
SRS warmup rows        16
SRS ridge              1e-6
```

## Input Identity

Every released `adapter_stream.npz` must contain aligned finite `float32`
arrays named `pred` and `true`, both with shape `[origin, lead, channel]`.
Directory names should identify source, dataset, backbone, horizon, seed, and
batch size. Each run records the file SHA-256 and tensor shape in
`stream_manifest.json`.

### Historical aggregate boundary

The historical `seed0` label used by the 144-cell main matrix is a compatibility
alias for an ensemble mean of ten training-seed forecasts, not a single
independent seed-0 run. The archived aggregate tensors and tables are sufficient
for deterministic replay, but the per-member ensemble manifest is not currently
part of this public package. Consequently:

- **stream replay is reproducible** when the released aggregate streams and
  their hashes are available;
- **the CommitCast algorithm is reproducible** from any conforming forecast
  stream;
- **end-to-end reconstruction of the exact paper aggregates from ten individual
  training runs is not yet artifact-verified**.

Do not describe the third item as complete until the ten-member manifest,
checkpoint identities, generation commands, and hashes have been archived.

### Current 10-seed FCR-port panel

The optional FCR-port experiment uses ten checkpoint identities as ten
statistical stability runs, not a `seed0` ensemble alias:

```text
10 seeds x 144 configurations x
(COSA-FCR, TAFAS-FCR, PETSA-FCR, CommitCast)
```

Per-cell metrics, aggregate tables, frozen streams, and run audits are generated
artifacts and are not stored in this source repository. A separate artifact
release should provide their checksums and download location if they are made
public.

## Execution

```bash
python main.py \
  DEVICE cpu \
  STREAM.ROOT streams \
  RESULT_DIR results/commitcast
```

For benchmark reporting, also record Python, NumPy, PyTorch, CUDA, GPU, and OS
versions outside the repository results table.

## Required Checks

```bash
python -m pytest
python -m compileall -q .
```

The test suite checks complete issue-to-c75 lifecycle coverage, same-commit
alignment, non-overlapping service support,
future-label mutation invariance, settlement timing, stream validation, config
overrides, and the complete COSA-style build/adapt/predict pipeline.
The optional-port tests additionally check registry isolation, required
external inputs, TAFAS/PETSA service support and train/eval modes, COSA
future-label invariance, and the portable matrix-manifest contract.
