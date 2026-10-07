# Third-party source notices

The following official code snapshots are shipped offline in `baselines/sources/`. Python modules, model/layer implementations, trainers, adapters, configuration and shell scripts retain their original contents. Integration code lives outside those archives. Upstream documentation assets, obsolete requirements text files and TAFAS's machine-local data-path placeholder are omitted; this release uses `pyproject.toml` for its environment and explicit data paths.

| Source | Snapshot | License retained inside archive |
| --- | --- | --- |
| [TAFAS](https://github.com/kimanki/TAFAS) | master retrieved 2026-10-08 | Modified MIT License (Non-Commercial with Permission) |
| [PETSA](https://github.com/BorealisAI/PETSA) | main retrieved 2026-10-08 | Attribution-NonCommercial-ShareAlike 4.0 International |
| [COSA](https://github.com/bigbases/COSA_ICLR2026) | current master retrieved 2026-10-08 | Attribution-NonCommercial-ShareAlike 4.0 International |

The snapshots include attribution to TAFAS and Time-Series-Library where supplied upstream. Their original LICENSE files and source headers remain authoritative for those components. No historical COSA release is included.

`fcr/native_loops.py` contains explicit clocked versions of the authors' outer loops. `fcr/native.py` resumes these functions; loss and optimizer routines are called on the unchanged official adapters. The source-level comparison in `fcr/native_check.py` is a read-only verification, not a code generator. Any adapted upstream code remains subject to its original terms. The root MIT license applies to original CommitCast code and does not relicense these dependencies.
