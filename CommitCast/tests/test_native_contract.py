from pathlib import Path

import numpy as np
import pytest

from fcr.native_check import verify_native_loop
from fcr.splits import NativeSplitView
from tta.commitcast import run_commitcast


@pytest.mark.parametrize('family',['TAFAS','PETSA','COSA'])
def test_explicit_loop_retains_official_learning_statements(family):
    assert verify_native_loop(family,Path(__file__).resolve().parents[1])['learning_statements_preserved']


def test_native_validation_view_preserves_split_length():
    class Dataset:
        split='val'
        val=np.zeros((50,2))
        test=np.zeros((90,2))
        def __len__(self):return 30
        def __getitem__(self,index):return self.val[index]
    data=Dataset()
    view=NativeSplitView(data,'val')
    assert view.test is data.val
    assert len(view)==30 and len(data.test)==90
    with pytest.raises(ValueError):NativeSplitView(data,'test')


def test_direct_api_rejects_inconsistent_target_windows():
    rng=np.random.default_rng(1)
    truth=rng.normal(size=(32,8,2)).astype(np.float32)
    with pytest.raises(ValueError,match='overlap'):
        run_commitcast(truth.copy(),truth,fractions=[.25,.5,.75])
