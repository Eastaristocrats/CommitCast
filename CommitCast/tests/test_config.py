from __future__ import annotations

from argparse import Namespace

import pytest

from tta.commitcast import build_adapter
from utils.parser import load_config


def test_dotlist_override():
    cfg = load_config(
        Namespace(
            cfg_file=None,
            opts=[
                "STREAM.ROOT",
                "fixture_streams",
                "TTA.COMMITCAST.BLOCK_SIZE",
                "4",
                "TTA.COMMITCAST.SRS_ENABLE",
                "False",
                "DATA.PRED_LEN",
                "96,192",
                "TTA.COMMITCAST.FRACTIONS",
                "0.2,0.4",
            ],
        )
    )
    assert cfg.STREAM.ROOT == "fixture_streams"
    assert cfg.TTA.COMMITCAST.BLOCK_SIZE == 4
    assert cfg.TTA.COMMITCAST.SRS_ENABLE is False
    assert cfg.DATA.PRED_LEN == (96, 192)
    assert cfg.TTA.COMMITCAST.FRACTIONS == (0.2, 0.4)


def test_adapter_factory_rejects_wrong_method_name():
    cfg = load_config(Namespace(cfg_file=None, opts=["TTA.NAME", "OtherMethod"]))
    with pytest.raises(ValueError, match="TTA.NAME"):
        build_adapter(cfg, model=None)
