"""Default configuration for the public CommitCast implementation."""

from yacs.config import CfgNode as CN


_C = CN()

_C.SEED = 0
_C.DEVICE = "auto"
_C.VISIBLE_DEVICES = ""
_C.RESULT_DIR = "results/commitcast"

_C.STREAM = CN()
_C.STREAM.ROOT = "streams"
_C.STREAM.GLOB = "*/adapter_stream.npz"
_C.STREAM.MAX_STREAMS = 0

_C.DATA = CN()
_C.DATA.NAME = ""
_C.DATA.PRED_LEN = ()

_C.MODEL = CN()
_C.MODEL.NAME = ""
_C.MODEL.FROZEN = True

_C.TRAIN = CN()
_C.TRAIN.ENABLE = False
_C.TRAIN.CHECKPOINT_DIR = ""

_C.TEST = CN()
_C.TEST.ENABLE = True

_C.TTA = CN()
_C.TTA.ENABLE = True
_C.TTA.NAME = "CommitCast"
_C.TTA.COMMITCAST = CN()
_C.TTA.COMMITCAST.FRACTIONS = (0.25, 0.50, 0.75)
_C.TTA.COMMITCAST.FEATURE_MODE = "full"
_C.TTA.COMMITCAST.TRAJ_POINTS = 17
_C.TTA.COMMITCAST.BLOCK_SIZE = 128
_C.TTA.COMMITCAST.RIDGE_LAMBDA = 25.0
_C.TTA.COMMITCAST.WARMUP_MATURED = 16
_C.TTA.COMMITCAST.FEATURE_CLIP = 6.0
_C.TTA.COMMITCAST.TARGET_CLIP = 6.0
_C.TTA.COMMITCAST.BETA_CLIP = 0.4
_C.TTA.COMMITCAST.CORRECTION_CLIP = 0.5
_C.TTA.COMMITCAST.SRS_ENABLE = True
_C.TTA.COMMITCAST.SRS_HALF_LIFE_ROWS = 512.0
_C.TTA.COMMITCAST.SRS_WARMUP_ROWS = 16
_C.TTA.COMMITCAST.SRS_RIDGE = 1e-6

_C.EVALUATION = CN()
# Common complete-H requests across the declared static sensitivity suite.
# This controls which origins are scored, never the number of targets per request.
_C.EVALUATION.SUPPORT_FRACTIONS = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
_C.EVALUATION.PROFILE = "common_complete_h"


def get_cfg_defaults() -> CN:
    """Return a clone so callers cannot mutate the module-level defaults."""

    return _C.clone()
