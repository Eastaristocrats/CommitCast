from __future__ import annotations

import numpy as np

from tta.commitcast import (
    CommitCastConfig,
    commit_schedule,
    revise_at_commit,
    run_commitcast,
    settled_risk_exposure,
)


def _stream(seed: int = 7, origins: int = 48, horizon: int = 8, channels: int = 2):
    rng = np.random.default_rng(seed)
    timeline = rng.normal(size=(origins + horizon - 1, channels)).astype(np.float32)
    true = np.stack([timeline[i:i+horizon] for i in range(origins)])
    pred = true + rng.normal(scale=0.2, size=true.shape).astype(np.float32)
    return pred, true


def _mutate_from_absolute_time(true: np.ndarray, cutoff: int) -> np.ndarray:
    changed = true.copy()
    rows = np.arange(true.shape[0])[:, None]
    leads = np.arange(true.shape[1])[None, :]
    mask = rows + leads >= cutoff
    changed[mask] += 1000.0
    return changed


def test_commit_schedule_is_unique_and_sorted():
    assert commit_schedule(8, [0.25, 0.5, 0.75]) == [
        ("c25", 2),
        ("c50", 4),
        ("c75", 6),
    ]
    assert commit_schedule(4, [0.24, 0.26]) == [("c24", 1)]


def test_same_commit_reference_and_non_overlapping_segments():
    pred, true = _stream()
    cfg = CommitCastConfig(
        device="cpu",
        block_size=2,
        warmup_matured=1,
        srs_warmup_rows=1,
        traj_points=3,
    )
    events = run_commitcast(pred, true, fractions=[0.25, 0.5, 0.75], cfg=cfg)
    assert len(events) == 4
    common_origins = pred.shape[0] - 6
    initial = events[0]
    assert initial.label == "issue_to_c25"
    np.testing.assert_array_equal(initial.checkpoint, pred[:common_origins, :2, :])
    np.testing.assert_array_equal(initial.target, true[:common_origins, :2, :])
    for event in events[1:]:
        expected = pred[event.delay :, : event.segment_len, :][:common_origins]
        expected_target = true[:common_origins, event.delay : event.next_delay, :]
        np.testing.assert_array_equal(event.checkpoint, expected)
        np.testing.assert_array_equal(event.target, expected_target)


def test_revision_is_invariant_to_unavailable_future_labels():
    pred, true = _stream(origins=36)
    delay = 2
    decision_row = 15
    cutoff = decision_row + delay
    changed = _mutate_from_absolute_time(true, cutoff=cutoff)
    cfg = CommitCastConfig(
        device="cpu",
        block_size=1,
        warmup_matured=1,
        traj_points=3,
    )
    original = revise_at_commit(pred, true, delay=delay, served_leads=2, cfg=cfg)
    mutated = revise_at_commit(pred, changed, delay=delay, served_leads=2, cfg=cfg)
    np.testing.assert_array_equal(
        original[: decision_row + 1],
        mutated[: decision_row + 1],
    )


def test_exposure_waits_for_complete_segment_settlement():
    pred, true = _stream(origins=40, horizon=4, channels=1)
    base = pred[:, :2, :]
    proposal = base + 0.1
    target = true[:, :2, :]
    decision_row = 18
    changed = target.copy()
    rows = np.arange(target.shape[0])[:, None]
    leads = np.arange(target.shape[1])[None, :]
    changed[rows + leads >= decision_row] += 1000.0
    _, alpha = settled_risk_exposure(
        target,
        base,
        proposal,
        segment_len=2,
        warmup=1,
    )
    _, changed_alpha = settled_risk_exposure(
        changed,
        base,
        proposal,
        segment_len=2,
        warmup=1,
    )
    np.testing.assert_array_equal(alpha[: decision_row + 1], changed_alpha[: decision_row + 1])
