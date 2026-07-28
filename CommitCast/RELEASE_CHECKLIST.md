# Anonymous Release Checklist

Publish the contents of this `CommitCast` directory as the repository root.
Do not upload its parent research workspace.

- [ ] Run `python -m pytest`.
- [ ] Run the toy-stream command in `README.md` from a fresh environment.
- [ ] Confirm `git status --short` contains no streams, checkpoints, results, or caches.
- [ ] Search tracked text for personal metadata and local absolute paths.
- [ ] Check every committed binary and image manually.
- [ ] Keep only public checkpoint/stream download links.
- [ ] Record hashes for released streams and checkpoints.
- [ ] Archive the ten-member ensemble manifest before claiming end-to-end
      reconstruction of the paper's `seed0` compatibility aggregate.
- [ ] Keep optional COSA/TAFAS/PETSA code outside the CommitCast core and
      document its upstream commits and licenses in `THIRD_PARTY.md`.
- [ ] Confirm no generated CSV/JSON metrics, logs, streams, checkpoints, or
      benchmark artifact bundles are tracked with the source.
- [ ] Do not copy CC BY-NC-SA code into the MIT-licensed core.
- [ ] Preserve the anonymous citation until the review period ends.
- [ ] Replace anonymous authorship and citation metadata only after de-anonymization is allowed.
