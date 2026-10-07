# Official baseline implementations

[Back to README](../README.md) | [Evaluation protocol](../docs/protocol.md)

The archives in `sources/` contain the official model, layer, trainer, adapter and configuration code together with the upstream license notices. [SOURCES.json](SOURCES.json) records their origins, retrieval dates and packaging omissions. Only the current COSA source is included.

| Method | Archive | Entry-point configuration |
| --- | --- | --- |
| TAFAS | `sources/tafas.zip` | [configs/tafas.yaml](../configs/tafas.yaml) |
| PETSA | `sources/petsa.zip` | [configs/petsa.yaml](../configs/petsa.yaml) |
| COSA | `sources/cosa_current.zip` | [configs/cosa.yaml](../configs/cosa.yaml) |

## Source access

Sources expand automatically into `.vendor/` on first use. To inspect all three before running an experiment:

```bash
python scripts/unpack_baselines.py
```

Run this command from the repository root. The compressed layout keeps the fresh source distribution below GitHub's 100-file browser-upload limit. Python dependencies, datasets and checkpoints remain separate inputs. Upload the freshly extracted repository contents before generating `.vendor/`, streams or results. See [GitHub's upload instructions](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository).

Use `main.py --method ...` for execution. The official projects share top-level module names such as `config`, `models` and `tta`; separate processes isolate their imports. CommitCast integration lives in `fcr/`, outside the source archives.

## Starting profiles

TAFAS and PETSA obtain dataset/backbone/horizon-specific learning rates and weight decay from their official scripts. TAFAS also reads its gating initialization there. PETSA's starting rank, loss alpha and gating initialization are 16, 0.2 and 0.02, following the upstream README example. These are starting profiles, not tuned claims for every evaluation cell.

COSA follows the bundled current `scripts/cosa.sh` profile: PAAS, 3 adaptation steps, context size 10, configured batch size 48, Linear adapter, fast adaptation, adaptive learning rate, per-batch learning-rate reset, POGT, partial adaptation and prediction adjustment.

`--profile config` uses the supplied YAML without script-derived profile overrides. Explicit `KEY VALUE` options after `--` take precedence. For COSA, append `-- TTA.COSA.PAAS False` to select fixed batches or `-- TTA.COSA.ADAPTER_TYPE MLP` to select its MLP adapter. When appending multiple settings, use one `--` followed by all key/value pairs.

## Native and commitment evaluation

`--protocol native` invokes the official adapter and its original evaluation. `--protocol fcr` resumes explicit versions of its outer loop at observation times. Losses and optimizer operations still execute through the official learner. Current-state queries and native revisions are evaluated as separate service policies; their interpretation is specified in the [protocol](../docs/protocol.md#baseline-evaluation).

The read-only checks in `fcr/native_check.py` compare learning statements with the bundled sources. `scripts/check_native.py` additionally checks actual predictions, model and optimizer states, and causal decisions. These checks should be rerun for affected configurations when changing a model, normalization component or runtime.

## Compatibility

The bundled official adapter paths require CUDA. CommitCast's forecast-stream replay also supports CPU.

TAFAS and PETSA use their upstream default `STEPS=1`. Their nondefault multi-step branches can raise a repeated-backward error in the tested PyTorch 2.10 environment. Requested steps are not silently reduced, and losses are not substituted. COSA's multi-step path is available.

An enabled normalization component requires its own compatible checkpoint. A forecasting configuration from a different method may contain an unsupported component; the runner reports that incompatibility instead of silently dropping it.

## Attribution

Official project links and component-specific licenses are listed in [THIRD_PARTY.md](../THIRD_PARTY.md). The root CommitCast MIT license does not replace those terms.
