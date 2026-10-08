# Third-party source notices

This repository consolidates the shared forecasting framework from the following official sources. The source snapshots were retrieved on 2026-10-08. Identical models, layers and data utilities appear once; each method retains its independent adaptation implementation.

| Source | Snapshot | Adapter | License |
| --- | --- | --- | --- |
| [TAFAS](https://github.com/kimanki/TAFAS) | master | [tta/tafas.py](tta/tafas.py) | [Modified MIT License (Non-Commercial with Permission)](licenses/TAFAS) |
| [PETSA](https://github.com/BorealisAI/PETSA) | main | [tta/petsa.py](tta/petsa.py) | [Attribution-NonCommercial-ShareAlike 4.0 International](licenses/PETSA) |
| [COSA](https://github.com/bigbases/COSA_ICLR2026) | current master | [tta/cosa.py](tta/cosa.py) | [Attribution-NonCommercial-ShareAlike 4.0 International](licenses/COSA) |

Upstream notices include Copyright (c) 2025-present, Royal Bank of Canada; Copyright (c) 2025-present, Kim et al.; and attribution to TAFAS and Time-Series-Library. The original license texts are reproduced in `licenses/`. Source headers are retained. No historical COSA release is included.

## Source mapping and modifications

- **Adapters:** `tta/tafas.py`, `tta/petsa.py` and `tta/cosa.py` preserve the respective official implementations. The auxiliary `tta/dynatta.py` and `models/DDN.py` come from the current COSA snapshot. DynaTTA is not one of the three baseline methods exposed by the experiment runner.
- **Shared framework:** backbone modules in `models/`, forecasting layers in `layers/`, `datasets/forecasting.py`, data-loader functions, optimizer and scheduling utilities come from TAFAS and its PETSA/COSA derivatives. Shared executable definitions are identical where consolidated. Additional TAFAS backbone and DishTS source files are retained.
- **Training and configuration:** `trainer.py` retains TAFAS's training, validation, normalization and checkpoint logic. `config.py` combines the method schemas and preserves method-specific defaults, including COSA's training batch size. The model factory imports shared backbones on demand. Forecasting and input preparation retain the PETSA/COSA CPU fallback and TAFAS's DishTS branch. Stream-specific CommitCast functions are added alongside this framework.
- **Published settings:** `configs/published.json` records all 144 TAFAS and 144 PETSA dataset/backbone/horizon profiles previously read from their shell scripts. It preserves learning rates, weight decay and TAFAS gating initialization. The root command-line entry points replace repeated experiment scripts.
- **Commitment evaluation:** `fcr/native_loops.py` contains explicit observation-clock versions of the authors' outer loops. `fcr/native.py` resumes them while calling the official loss and optimizer routines. Offline metric accumulation is replaced by commitment-time scoring; native evaluation remains available through `--protocol native`.

Repeated project trees, upstream documentation assets, local path placeholders and unrelated upstream entry-point wrappers are omitted. Dependencies are declared in `pyproject.toml`. Original method source and adapted upstream components retain their original license terms; consolidation does not relicense them.

## Original CommitCast code

The root [MIT license](LICENSE) applies to original CommitCast components, including `tta/commitcast.py`, `layers/online_ridge.py`, `layers/exposure.py`, commitment scoring and replay, and the original stream interfaces. Files that combine original and upstream code retain both applicable attribution and license boundaries described above. The root license does not replace the licenses of the baseline or shared forecasting components.
