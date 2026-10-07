import numpy as np
import pytest
from fcr.scoring import (SCHEDULES, common_origins, score_plan, errors, gain,
                         PublicationLedger, event_boundaries)
from tta.commitcast import CommitCastConfig, run_commitcast
from fcr.scoring import validate_event_timeline


def stream():
    rng = np.random.default_rng(72)
    y = rng.normal(size=(71, 2)).astype(np.float32)
    true = np.stack([y[i:i+8] for i in range(64)])
    pred = true + rng.normal(size=true.shape).astype(np.float32) * .3
    return pred, true


def test_every_schedule_scores_same_complete_requests_and_noop_zero():
    pred, true = stream()
    count = common_origins(len(pred), 8, list(SCHEDULES.values()))
    counts = []
    for fractions in SCHEDULES.values():
        metrics, rows = score_plan(pred, true, fractions, count,
            lambda request, a, b: pred[request+a, :b-a])
        assert metrics['mse_gain_pct'] == 0.
        assert all(r['atoms'] == 8*2 for r in rows)
        counts.append(metrics['evaluated_elements'])
    assert set(counts) == {count*8*2}


def test_pooled_gain_is_not_average_percentage():
    assert gain(100+1, 50+2) == pytest.approx(100*(1-52/101))
    assert np.isnan(gain(0,0))
    with pytest.raises(ValueError):
        errors(np.ones((3,2)), np.ones((3,1)))
    with pytest.raises(ValueError):
        errors(np.ones((3,2)), np.full((3,2),np.nan))


def test_native_revisions_remain_request_local_and_cannot_be_backdated():
    ledger = PublicationLedger()
    ledger.publish(0, 3, np.arange(8)[:,None])
    ledger.publish(0, 5, 100+np.arange(8)[:,None])
    ledger.publish(1, 5, 999+np.arange(8)[:,None])
    fallback = np.zeros((2,1))
    out, covered = ledger.select(0,4,4,6,fallback)
    np.testing.assert_array_equal(out[:,0], [4,5])
    assert covered.all()
    out, _ = ledger.select(0,6,6,8,fallback)
    np.testing.assert_array_equal(out[:,0], [106,107])
    with pytest.raises(ValueError):
        ledger.publish(10,9,np.ones((8,1)))
    assert not ledger.records[0][0].values.flags.writeable


def test_common_support_does_not_change_fitted_predictions():
    pred,true = stream()
    cfg = CommitCastConfig(block_size=4, warmup_matured=2, srs_warmup_rows=2)
    anchor = run_commitcast(pred,true,fractions=[.25,.5,.75],cfg=cfg)
    common = run_commitcast(pred,true,fractions=[.25,.5,.75],support_fractions=list(SCHEDULES['dense8']),cfg=cfg)
    for a,b in zip(anchor,common):
        np.testing.assert_array_equal(a.prediction[:len(b.prediction)],b.prediction)
    assert sum(e.prediction.shape[1] for e in common) == 8


def test_no_revision_is_full_h_and_original_first_issue_rule():
    pred,true=stream()
    events = run_commitcast(pred,true,fractions=[],support_fractions=[.875])
    assert len(events)==1 and events[0].segment_len==8
    np.testing.assert_array_equal(events[0].prediction,pred[:57])


def test_event_schedule_cannot_see_current_or_future_observations():
    values=np.random.default_rng(2).normal(size=(200,3))
    a=event_boundaries(values,8,80,.2)
    changed=values.copy(); changed[50:]+=99
    b=event_boundaries(changed,8,80,.2)
    for origin in range(43):
        assert a[origin] == b[origin]


def test_event_inputs_must_belong_to_the_scored_stream():
    _,truth=stream()
    values=np.concatenate([np.zeros((32,2)),truth[:,0],truth[-1,1:]],axis=0)
    validate_event_timeline(values,truth,32)
    values[40,0]+=1
    with pytest.raises(ValueError,match='timeline'):
        validate_event_timeline(values,truth,32)
