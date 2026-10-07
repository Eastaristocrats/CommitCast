import numpy as np
import pytest

from fcr.replay import replay_schedules, event_exposures
from fcr.scoring import SCHEDULES, boundaries, input_scores, summarize_requests
from fcr.selection import choose_validation
from tta.commitcast import CommitCastConfig, run_commitcast


def test_suite_matches_independent_full_request_reconstruction():
    rng=np.random.default_rng(8)
    values=rng.normal(size=(103,2)).astype(np.float32)
    truth=np.stack([values[i:i+8] for i in range(96)])
    base=truth+rng.normal(size=truth.shape).astype(np.float32)*.3
    cfg=CommitCastConfig(block_size=8,warmup_matured=2,srs_warmup_rows=2)
    metrics,rows=replay_schedules(base,truth,['default','dense5','no_revision'],cfg=cfg)
    for record in metrics:
        events=run_commitcast(base,truth,fractions=list(SCHEDULES[record['schedule']]),
            support_fractions=list(SCHEDULES['dense8']),cfg=cfg)
        total=sum(np.square(e.prediction.astype(float)-e.target.astype(float)).sum() for e in events)
        assert record['sse']==pytest.approx(total,rel=1e-6,abs=1e-6)
        assert record['evaluated_elements']==89*8*2
    assert boundaries(96,SCHEDULES['dense5'])==(0,19,38,57,76,96)


def test_event_gate_uses_only_settled_service_rows():
    rng=np.random.default_rng(81)
    base=rng.normal(size=(80,8,2)).astype(np.float32)
    timeline=rng.normal(size=(87,2)).astype(np.float32)
    truth=np.stack([timeline[i:i+8] for i in range(80)])
    raw=base[2:].copy()+.2
    cfg=CommitCastConfig(srs_warmup_rows=1)
    rows=[(i,1+i%5) for i in range(55)]
    changed=truth.copy()
    changed[np.arange(80)[:,None]+np.arange(8)[None,:]>=32]+=99
    original=list(event_exposures(rows,2,base,truth,raw,cfg))
    altered=list(event_exposures(rows,2,base,changed,raw,cfg))
    for a,b in zip(original,altered):
        if a[0]+2<=32:
            np.testing.assert_array_equal(a[2],b[2])
            assert a[3]==b[3]


def test_input_event_has_observed_encoder_prehistory():
    values=np.zeros((80,2),dtype=np.float32);values[31]=3
    scores=input_scores(values,16,context_length=32)
    assert scores[0]==3
    altered=values.copy();altered[32:]+=500
    assert input_scores(altered,16,context_length=32)[0]==scores[0]


def test_zero_base_requests_are_not_hidden_from_degradation_rate():
    out=summarize_requests([dict(atoms=2,sse=1.,sae=1.,base_sse=0.,base_sae=0.)])
    assert out['request_degradation_rate_pct']==100
    assert out['zero_base_requests']==1 and np.isnan(out['mse_gain_pct'])


def test_selection_rejects_test_and_mismatched_support_and_keeps_base_ties():
    base=dict(candidate='Base',split='val',requests=12,evaluated_elements=96,base_sse=96.,sse=96.,mse=1.,
              cell='fixture',method='COSA',policy='current',schedule='default')
    default=dict(base,candidate='published',published_default=True)
    assert choose_validation([base,default])['candidate']=='Base'
    with pytest.raises(ValueError):
        choose_validation([base,dict(default,split='test')])
    with pytest.raises(ValueError):
        choose_validation([base,dict(default,evaluated_elements=95)])
    with pytest.raises(ValueError,match='different schedule'):
        choose_validation([base,dict(default,schedule='dense8')])
    with pytest.raises(ValueError,match='disagrees'):
        choose_validation([base,dict(default,mse=.5)])
