# Evaluation protocol

[Back to README](../README.md) | [Experiment guide](experiments.md)

## Forecast streams

A stream contains consecutive forecast origins for one dataset, backbone, horizon and training seed. `pred[i]` is a frozen forecast made at origin `i`; `true[i]` contains the corresponding targets.

| NPZ field | Shape or type | Meaning |
| --- | --- | --- |
| `pred` | float32 `[N, H, C]` | Frozen forecasts |
| `true` | float32 `[N, H, C]` | Target windows |
| `values` | `[context_length + N + H - 1, C]` | Encoder prehistory followed by the target timeline; required for Event50 |
| `context_length` | Integer scalar | Number of prehistory observations; required for Event50 |

Both forecast and target arrays must be finite and have identical shapes. Origins have stride 1, so `true[i, 1:]` equals `true[i+1, :-1]`. The loader rejects inconsistent overlap. For Event50, the post-context part of `values` must equal the timeline reconstructed from `true`; baseline comparisons also check the encoder prehistory against the shared stream.

Forecast generation uses the checkpoint's architecture, context, split and normalization, including a scaler fitted on training data. Targets are retained for delayed feedback and scoring. Their presence in the archive does not make them available to a decision before their observation time. Array validation alone cannot establish causal forecast generation.

The model-aware exporter writes `metadata.json` with the split, shape, dataset, backbone, seed, stride, context and split ratios. [scripts/export_stream.py](../scripts/export_stream.py) can instead package existing arrays; their causal generation and provenance remain the caller's responsibility.

## Decision timing

A request issued at `i` covers target times `i` through `i+H-1`. At a commitment with delay `a`, the decision time is `q = i+a`. Only target observations with time strictly less than `q` are available.

If the next commitment is at delay `b`, this decision serves the interval `[a,b)`. The target and matched frozen reference are:

```python
target = true[i, a:b]
base = pred[i+a, :b-a]
```

All intervals partition `[0,H)` without overlap or gaps. Every request contributes exactly `H*C` coordinates. Requests can overlap in calendar time; a shared target can therefore occur in different requests at different forecast origins. It is counted once per request, not once across the entire stream.

CommitCast serves Base at issue time. At later commitments, its lead/channel ridge updates use only matured labels. Its exposure gate uses feedback from fully settled service intervals. An Event50 decision uses the current observed trigger; the current prediction and exposure do not depend on the next trigger time. Previously served predictions remain fixed.

## Commitment schedules

Fractions specify internal commitment times; issue time `0` and horizon end `H` are always included. A fraction `f` maps to `floor(H*f)`, clamped to `[1,H-1]`, with duplicate delays removed.

| CLI name | Fractions of H |
| --- | --- |
| `default` | 1/4, 1/2, 3/4 |
| `sparse` | 1/2 |
| `uneven2` | 3/8 |
| `uneven3` | 3/8, 3/4 |
| `missing` | 1/4, 3/4 |
| `uneven4` | 1/8, 3/8, 3/4 |
| `dense5` | 1/5, 2/5, 3/5, 4/5 |
| `dense8` | 1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8 |
| `no_revision` | None |
| `event50` | Observed-input triggers at the `dense8` candidate times |

`--schedules all` selects the nine static schedules. An explicit `--event-threshold` also adds Event50. Its threshold is the median of validation input scores and is fixed before test evaluation. At time `q`, the input score is the root mean square across channels of the difference between the latest observed input and the mean of its preceding 24 observations. A candidate is selected when its score exceeds the threshold. Encoder prehistory is included.

The default `--support common` retains `N-floor(7H/8)` complete requests, including when only one schedule is selected. `--support schedule` retains the common request set required by the selected schedules. These support choices can produce different averages and must be reported separately.

The YACS entry point uses common support by default. [configs/paper_anchor.yaml](../configs/paper_anchor.yaml) reproduces the original quarter-schedule support, `N-floor(3H/4)`. It changes which origins are scored, not the horizon of a request.

## CommitCast defaults

| Parameter | Value |
| --- | ---: |
| Feature mode | `full` (12 features) |
| Forecast-history sample points | 17 |
| Ridge penalty | 25 |
| Update block size | 128 |
| Matured-row warm-up | 16 |
| Feature / target clipping | 6 / 6 |
| Coefficient / correction clipping | 0.4 / 0.5 |
| Exposure half-life | 512 settled rows |
| Exposure warm-up | 16 settled rows |
| Exposure ridge | 0.000001 |

The numerical estimator uses lead-wise maturity and batched solves and updates. Features are computed online; target caching is disabled. Ridge states are indexed by commitment delay, lead and channel. Prefix residual scale normalizes the correction, and the settled-risk exposure is bounded to `[0,1]`.

[scripts/run_schedules.py](../scripts/run_schedules.py) uses these fixed defaults and exposes feature mode, device, schedules and support. The YACS entry point exposes the parameters in [config.py](../config.py); programmatic replay accepts a `CommitCastConfig`. Record any changed values with the experiment.

## Baseline evaluation

The official learners retain their loss functions, optimizers, adaptation steps, dynamic batches, histories, partial/full feedback and prediction adjustment. [fcr/native_loops.py](../fcr/native_loops.py) makes the outer loops resumable at observation times while calling the learning functions in `tta/`. Starting profiles and compatibility limits are documented in the [experiment guide](experiments.md#baseline-profiles-and-compatibility).

There are two distinct evaluation protocols:

| Protocol | Evaluation |
| --- | --- |
| `native` | The official `adapter.adapt()` and its full-window scoring |
| `fcr` | The learner's outputs served under the shared commitment schedule and complete-H scoring |

The `fcr` runner reports two policies:

| Policy | Prediction served at a commitment |
| --- | --- |
| `current` | A fresh forecast from an isolated copy of the currently adapted state |
| `native_revisions` | The latest available native revision for that request; otherwise the current forecast |

Queries preserve the learner's state and random-number stream. Native revisions keep their actual publication times and cannot be backdated to the origins of a batch. Revision coverage and fallback behavior are reported. Tail updates after the final decision cannot modify served outputs.

Choose the service policy before inspecting test results and identify it in tables. Retaining a learner's update rules does not make its native-window score interchangeable with a service score.

## Scoring and aggregation

For every request, sum squared errors (`sse`) and absolute errors (`sae`) over all served intervals. `evaluated_elements` is the number of scored coordinates. MSE and MAE divide the corresponding error sums by this count. Base is scored on exactly the same requests, targets and commitment times.

For aggregation across cells, sum error totals first:

$$
\operatorname{Gain}_{\mathrm{MSE}}
=100\left(1-\frac{\sum_j \mathrm{SSE}_{j,\mathrm{method}}}
                       {\sum_j \mathrm{SSE}_{j,\mathrm{Base}}}\right).
$$

Use absolute-error sums for MAE gain. Positive gain means lower error than Base. Relative gain is undefined when Base error is zero; absolute degradation still counts.

The runners report P90, P95 and maximum request MSE, the fraction of requests with increased SSE, and the fraction exceeding `1.05 * Base SSE`. Percentages use all retained requests. Missing or nonfinite predictions are errors, not grounds for dropping a method's difficult requests.

Pooled gain weights the number and scale of target coordinates. Report dataset/backbone/horizon breakdowns and equal-cell summaries alongside it. Requests overlap in time: uncertainty analysis should account for temporal dependence and independent training seeds rather than treating every row as an independent sample.

Validation candidates must share the same task, policy, schedule, request support and Base reference. Include the published profile and Base, use a declared search budget for every method, and lock choices before test evaluation. [The experiment guide](experiments.md#validation-selection) explains the selection helper and its limits.

## Runtime measurements

`run.json` records the timing scope. CommitCast replay includes correction, exposure and request scoring, while baseline transport timing includes live queries and state copying but excludes setup, export and scoring. These fields are not directly comparable for speed rankings. An efficiency experiment needs a shared end-to-end scope, device, synchronization policy and warm-up procedure.
