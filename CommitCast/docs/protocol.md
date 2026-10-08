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

The model-aware exporter writes `metadata.json` with the split, shape, dataset, backbone, seed, stride, context, split ratios and checkpoint paths, including a separate normalization checkpoint when enabled. [scripts/export_stream.py](../scripts/export_stream.py) can instead package existing arrays; their causal generation and provenance remain the caller's responsibility.

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

Coefficients are held fixed within each update block. At a block boundary, newly matured rows contribute batch-averaged sufficient statistics. Thus, changing `BLOCK_SIZE` changes both the feedback cadence and the relative weighting of update batches; it is a statistical parameter, not only a memory or throughput setting. Observation availability is checked at the block boundary, so this delay is conservative.

The frozen backbone receives no gradient updates. Ridge coefficients, sufficient statistics and exposure state still adapt and consume memory. A count of zero gradient-trained parameters must not be interpreted as zero learned state or zero adaptation cost.

Device inputs are shared across the commitments of a replay. Chronologically overlapping target windows are views of a single observation timeline; prefix residuals are materialized for the current feature block. Explicit contiguous residual layout retains the reference reduction order. The online feature ring still retains the history needed for matured updates, and the current feature tensor is reused after writing it to that ring.

For ordinary lead-wise feedback on CUDA, zero-based lead `r` needs at most `min(n, r+B)` feature rows, where `n` is the number of computed origins at that commitment and `B` is the existing update block size. Separate rings use these retention lengths. Each block consumes matured entries before writing new features. CPU execution, irregular arrivals and full-suffix feedback retain shared storage. The cache layout does not change the feedback available to an update.

The numerical interface also supports delayed feedback. Its shared online ring includes the feedback delay and any additional arrival retention in its capacity, so features remain available until the corresponding labels can be consumed. The default method uses zero additional feedback delay.

Trajectory sums and sufficient statistics reuse their allocated storage while preserving operation order and precision. CUDA batched ridge solves merge their status codes into a device failure flag in bounded batches and check that flag before returning a proposal, avoiding a host synchronization at each block. Every status is checked, including earlier failures followed by successful solves; singular systems still raise an error. Settled-risk residual buffers are overwritten only with currently settled rows. For explicit irregular settlement lags, metadata is sorted once by release time and request ID; the gate consumes only the released prefix of that order.

Grouped sufficient statistics share the packed left matrix between the Gram matrix and cross-product. Regular maturity indices are derived on the device, and aligned targets reuse their chronological view. On CUDA, newly maturing leads can share a gather before their individual updates. Each lead retains its original reduction length, layout and normalization; padding reads only already-matured rows and never enters a reduction. The retained feature and target gather buffers are capped at 32 MiB, with larger groups using the per-lead path. Exposure mixing reuses its output buffer while retaining the input-dtype subtraction followed by FP64 multiplication and addition.

Replay skips computation for origin blocks beyond the declared scoring support. It retains the final complete compute block to preserve matrix dimensions and returns only the scored origins. This changes neither the retained request set nor any request's horizon. Gate updates remain in settlement order. Static-schedule scoring uses bounded FP64 buffers whose size is independent of the statistical `BLOCK_SIZE`. Unrequested trajectory features are skipped only in the corresponding ablations; the full method retains all 12 features.

Schedules sharing a commitment offset and served interval reuse the same per-request error sums. Replay retains these compact sums instead of keeping a separate mixed forecast tensor for every interval. Once all ridge leads have completed warm-up, the all-ones activity mask is omitted; the correction itself is unchanged.

Event50 exposure weights are fixed in service order before offline scoring groups requests by served interval length. This grouping affects only evaluation: it cannot change a weight, stopping time or prediction. Request error sums are accumulated in the original commitment-offset order. The `event_exposures` generator remains available for callers that need predictions one service row at a time.

[scripts/run_schedules.py](../scripts/run_schedules.py) uses these defaults unless `--cfg` or method overrides are supplied. Resolution order is defaults, YAML, `TTA.COMMITCAST.KEY VALUE` pairs after `--`, then explicit `--feature-mode` and `--device` flags. Without a YAML device setting or flag, replay uses CPU. Named `--schedules` and `--support` determine timing and request support; YAML `FRACTIONS` and `EVALUATION` do not replace them. The YACS entry point exposes [config.py](../config.py), including custom fractions; programmatic replay accepts a `CommitCastConfig`. Each runner records its resolved configuration.

## Baseline evaluation

The official learners retain their loss functions, optimizers, adaptation steps, dynamic batches, histories, partial/full feedback and prediction adjustment. The declared comparison fixes each baseline to its official execution profile; no baseline hyperparameter search is performed. [fcr/native_loops.py](../fcr/native_loops.py) makes the outer loops resumable at observation times while calling the learning functions in `tta/`. Exact profile sources and optional compatibility limits are documented in the [experiment guide](experiments.md#baseline-profiles-and-compatibility).

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

The observation clock releases only newly available target coordinates. On the ordinary tensor layout, diagonal views avoid rescanning the entire target archive at every wait; unusual layouts use a bounded mask over affected origins. Model queries still run one origin at a time. A device buffer groups only the transfer of completed predictions to the host, using at most 4 MiB or one forecast when a single forecast is larger. This buffer changes neither inference batch size nor the learner's update schedule. `query_snapshot` returns NumPy arrays by default; `return_tensor=True` keeps detached predictions on the model device.

`score_stream` evaluates recorded current forecasts and request-local native revisions in bounded batches. It returns a summary, request rows and the number of coordinates covered by native revisions. `score_plan` remains available for arbitrary prediction callbacks. Both interfaces score the same complete-H targets and enforce publication deadlines. Event50's observed-input statistic uses chronological window views; each decision still reads only observation `q-1` and its preceding history.

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

In the YACS entry point, `event_win_rate_pct` counts wins across aggregated commitment intervals, not individual requests. Use the request-level degradation fields for request risk; neither measure is a significance test.

Pooled gain weights the number and scale of target coordinates. Report dataset/backbone/horizon breakdowns and equal-cell summaries alongside it. Requests overlap in time: uncertainty analysis should account for temporal dependence and independent training seeds rather than treating every row as an independent sample.

TAFAS, PETSA and COSA remain at their official execution settings. Any optional CommitCast validation selection must use matched tasks, schedules, request support and Base references, include its declared default and Base, and lock the selected parameters before test evaluation. Report that selection procedure without describing the baselines as retuned. Event50's validation-frozen trigger is shared across methods and does not change their adaptation hyperparameters. [The experiment guide](experiments.md#validation-selection) explains the optional CommitCast selection helper and its limits.

## Runtime measurements

`run.json` records the timing scope. CommitCast replay includes correction, exposure and request scoring, while baseline transport timing includes live queries and state copying but excludes setup, export and scoring. These fields are not directly comparable for speed rankings. An efficiency experiment needs a shared end-to-end scope, device, synchronization policy and warm-up procedure.

For FCR, `elapsed_sec` retains the transport timing above. `scoring_elapsed_sec` measures event-plan construction, both service-policy scores and request-table construction. `transport_and_scoring_elapsed_sec` is their sum. These fields exclude setup, frozen forecast generation, offline tail updates and output writes; they are not total process runtime.

The YACS `event_metrics.csv` field `adapter_runtime_sec` allocates the stream's total adapter runtime evenly across its commitment intervals. It is an accounting allocation, not a measured latency for each commitment.
