# Third-Party and Architecture Notice

## COSA

The repository layout and public build/adapt/predict boundary are intentionally
compatible with the public COSA repository:

- project: `COSA: Context-Aware Output-Space Adapter for Test-Time Adaptation in
  Time Series Forecasting`;
- repository: <https://github.com/bigbases/COSA_ICLR2026>;
- research snapshot audited locally: `43a8c8da4de74d5745a8713f6130c523b7df2694`;
- upstream license in that snapshot: CC BY-NC-SA 4.0.

The CommitCast algorithm under `tta/`, `layers/`, and the associated tests is an
independent implementation. No COSA source file is redistributed by this
MIT-licensed core.

## TAFAS and PETSA FCR ports

The optional FCR runners load, but do not redistribute, the following pinned
upstream checkouts:

| Project | Repository | Audited commit | License in snapshot |
| --- | --- | --- | --- |
| COSA | <https://github.com/bigbases/COSA_ICLR2026> | `43a8c8da4de74d5745a8713f6130c523b7df2694` | CC BY-NC-SA 4.0 |
| TAFAS | <https://github.com/kimanki/TAFAS> | `139bf980671da4daad728a0fc21d8df508b9203d` | Modified MIT, non-commercial |
| PETSA | <https://github.com/BorealisAI/PETSA> | `87853d888e98311ac94e64be920d17b57143b20c` | CC BY-NC-SA 4.0 |

The upstream README badge or prose may not fully express the restrictions in
the corresponding `LICENSE` file. The license file controls. Users must review
and comply with each upstream license before running an optional port.

## Optional baseline policy

Implementations derived from COSA, TAFAS, PETSA, or other third-party projects
must not be copied into the MIT core without a license review. A baseline should
instead be distributed as one of:

1. a thin adapter that imports a separately installed upstream package;
2. a pinned submodule with its original license intact; or
3. an independently written compatible implementation with explicit
   provenance and equivalence tests.

Every optional baseline must record the upstream URL, commit, license,
configuration, input protocol, and whether its score is native-clock or
same-commit aligned.
