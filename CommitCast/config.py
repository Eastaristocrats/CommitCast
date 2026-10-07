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



# Shared forecasting configuration from TAFAS, PETSA and COSA.
# Original attribution and license scope: THIRD_PARTY.md and licenses/.
_B = CN()
# random seed number
_B.SEED = 0
# number of gpus per node
_B.NUM_GPUS = 8
_B.VISIBLE_DEVICES = 0
# directory to save result txt file
_B.RESULT_DIR = 'results/'
_B.NORMALIZE = 'NST'

_B.DATA_LOADER = CN()
_B.DATA_LOADER.NUM_WORKERS = 2
_B.DATA_LOADER.PIN_MEMORY = True
_B.DATA_LOADER.DROP_LAST = True

_B.DATA = CN()
_B.DATA.BASE_DIR = 'data/'
_B.DATA.NAME = 'weather'
_B.DATA.N_VAR = 21
_B.DATA.SEQ_LEN = 96
_B.DATA.LABEL_LEN = 48
_B.DATA.PRED_LEN = 96
_B.DATA.FEATURES = 'M'
_B.DATA.TIMEENC = 0
_B.DATA.FREQ = 'h'
_B.DATA.SCALE = "standard"  # standard, min-max
_B.DATA.TRAIN_RATIO = 0.7
_B.DATA.TEST_RATIO = 0.2
_B.DATA.DATE_IDX = 0
_B.DATA.TARGET_START_IDX = 0
_B.DATA.PERIOD_LEN = 24  # Used only when SAN is ENABLED
_B.DATA.STATION_TYPE = 'adaptive'  # Used only when SAN is ENABLED

_B.TRAIN = CN()
_B.TRAIN.ENABLE = False
_B.TRAIN.SPLIT = 'train'
_B.TRAIN.BATCH_SIZE = 256
_B.TRAIN.SHUFFLE = True
_B.TRAIN.DROP_LAST = True
# directory to save checkpoints
_B.TRAIN.CHECKPOINT_DIR = 'results/'
# path to checkpoint to resume training
_B.TRAIN.RESUME = ''
# epoch period to evaluate on a validation set
_B.TRAIN.EVAL_PERIOD = 5
# iteration frequency to print progress meter
_B.TRAIN.PRINT_FREQ = 100
_B.TRAIN.BEST_METRIC_INITIAL = float("inf")
_B.TRAIN.BEST_LOWER = True

_B.VAL = CN()
_B.VAL.SPLIT = 'val'
_B.VAL.BATCH_SIZE = 256
_B.VAL.SHUFFLE = False
_B.VAL.DROP_LAST = False
_B.VAL.VIS = False

_B.TEST = CN()
_B.TEST.ENABLE = True
_B.TEST.SPLIT = 'test'
_B.TEST.BATCH_SIZE = 256
_B.TEST.SHUFFLE = False
_B.TEST.DROP_LAST = False

_B.TTA = CN()
_B.TTA.ENABLE = False
_B.TTA.MODULE_NAMES_TO_ADAPT = 'cali'  # all, norm, etc
_B.TTA.LOG = False
_B.TTA.SOLVER = CN()
_B.TTA.SOLVER.OPTIMIZING_METHOD = 'adam'
_B.TTA.SOLVER.BASE_LR = 0.005
_B.TTA.SOLVER.WEIGHT_DECAY = 0.0001
_B.TTA.SOLVER.MOMENTUM = 0.9
_B.TTA.SOLVER.NESTEROV = True
_B.TTA.SOLVER.DAMPENING = 0.0
_B.TTA.TAFAS = CN()
_B.TTA.TAFAS.PAAS = True
_B.TTA.TAFAS.PERIOD_N = 1
_B.TTA.TAFAS.BATCH_SIZE = 64
_B.TTA.TAFAS.STEPS = 1
_B.TTA.TAFAS.ADJUST_PRED = True
_B.TTA.TAFAS.CALI_MODULE = True
_B.TTA.TAFAS.GATING_INIT = 0.01
_B.TTA.TAFAS.HIDDEN_DIM = 128
_B.TTA.TAFAS.GCM_VAR_WISE = True

_B.MODEL = CN()
_B.MODEL.NAME = 'iTransformer'
_B.MODEL.task_name = 'long_term_forecast'
_B.MODEL.seq_len = _B.DATA.SEQ_LEN 
_B.MODEL.label_len = _B.DATA.LABEL_LEN # Not needed in iTransformer
_B.MODEL.pred_len = _B.DATA.PRED_LEN 
_B.MODEL.e_layers = 4
_B.MODEL.d_layers = 1 # Not needed in iTransformer
_B.MODEL.factor = 3 # Not used in iTransformer Full Attention. Used in Prob Attention (probabilistic attention) in informer
_B.MODEL.enc_in = _B.DATA.N_VAR # Used only in classification
_B.MODEL.dec_in = _B.DATA.N_VAR # Not needed in iTransformer
_B.MODEL.c_out = _B.DATA.N_VAR # Not needed in iTransformer
_B.MODEL.d_model = 512 # embedding dimension
_B.MODEL.d_ff = 512  # feedforward dimension d_model -> d_ff -> d_model
_B.MODEL.moving_avg = 25
_B.MODEL.output_attention = False # whether the attention weights are returned by the forward method of the attention class
_B.MODEL.dropout = 0.1
_B.MODEL.n_heads = 8
_B.MODEL.activation = 'gelu'
_B.MODEL.channel_independence = True
_B.MODEL.METRIC_NAMES = ('MAE',)
_B.MODEL.LOSS_NAMES = ('MSE',)
_B.MODEL.embed = 'timeF'
_B.MODEL.freq = 'h'
_B.MODEL.ignore_stamp = False
# OLS params
_B.MODEL.instance_norm = True
_B.MODEL.individual = False
_B.MODEL.alpha = 0.000001

_B.NORM_MODULE = CN()
_B.NORM_MODULE.ENABLE = False  # NST
_B.NORM_MODULE.NAME = 'SAN'  # SAN, RevIN, DishTS

_B.SAN = CN()
_B.SAN.RESULT_DIR = 'results/station/'
_B.SAN.TRAIN = CN()
_B.SAN.TRAIN.CHECKPOINT_DIR = 'results/station/'
_B.SAN.SOLVER = CN()
_B.SAN.SOLVER.OPTIMIZING_METHOD = 'adam'
_B.SAN.SOLVER.START_EPOCH = 0
_B.SAN.SOLVER.MAX_EPOCH = 10
_B.SAN.SOLVER.BASE_LR = 0.001
_B.SAN.SOLVER.WEIGHT_DECAY = 0.0001
_B.SAN.SOLVER.MOMENTUM = 0.9
_B.SAN.SOLVER.NESTEROV = True
_B.SAN.SOLVER.DAMPENING = 0.0
_B.SAN.SOLVER.LR_POLICY = 'cosine'
_B.SAN.SOLVER.COSINE_END_LR = 0.0
_B.SAN.SOLVER.COSINE_AFTER_WARMUP = False
_B.SAN.SOLVER.WARMUP_EPOCHS = 0
_B.SAN.SOLVER.WARMUP_START_LR = 0.001

_B.REVIN = CN()
_B.REVIN.EPS = 1e-5
_B.REVIN.AFFINE = True
_B.REVIN.RESULT_DIR = 'results/revin/'
_B.REVIN.TRAIN = CN()
_B.REVIN.TRAIN.CHECKPOINT_DIR = 'results/revin/'

_B.DISHTS = CN()
_B.DISHTS.INIT = 'standard'  # standard, avg, uniform
_B.DISHTS.RESULT_DIR = 'results/dishts/'
_B.DISHTS.TRAIN = CN()
_B.DISHTS.TRAIN.CHECKPOINT_DIR = 'results/dishts/'

_B.SOLVER = CN()
_B.SOLVER.START_EPOCH = 0
_B.SOLVER.MAX_EPOCH = 30
_B.SOLVER.OPTIMIZING_METHOD = 'adam'
_B.SOLVER.BASE_LR = 0.0001
_B.SOLVER.WEIGHT_DECAY = 0.0001
_B.SOLVER.MOMENTUM = 0.9
_B.SOLVER.NESTEROV = True
_B.SOLVER.DAMPENING = 0.0
_B.SOLVER.LR_POLICY = 'cosine'
_B.SOLVER.COSINE_END_LR = 0.0
_B.SOLVER.COSINE_AFTER_WARMUP = False
_B.SOLVER.WARMUP_EPOCHS = 0
_B.SOLVER.WARMUP_START_LR = 0.001

_B.WANDB = CN()
_B.WANDB.ENABLE = False
_B.WANDB.PROJECT = 'TAFAS'
_B.WANDB.NAME = ''
_B.WANDB.JOB_TYPE = ''
_B.WANDB.NOTES = ''
_B.WANDB.DIR = './'
_B.WANDB.SET_LOG_DIR = True



## Efficient
_B.TTA.PETSA = CN()
_B.TTA.PETSA.PAAS = True
_B.TTA.PETSA.PERIOD_N = 1
_B.TTA.PETSA.BATCH_SIZE = 64
_B.TTA.PETSA.STEPS = 1
_B.TTA.PETSA.ADJUST_PRED = True
_B.TTA.PETSA.CALI_MODULE = True
_B.TTA.PETSA.GATING_INIT = 0.01
_B.TTA.PETSA.HIDDEN_DIM = 128
_B.TTA.PETSA.GCM_VAR_WISE = True
_B.TTA.PETSA.GCM_VAR_WISE = True
_B.TTA.PETSA.RANK = 16
_B.TTA.PETSA.LOSS_ALPHA = 0.1

## Proposed
_B.TTA.COSA = CN()
_B.TTA.COSA.BATCH_SIZE = 25
_B.TTA.COSA.STEPS = 20
_B.TTA.COSA.BUFFER_CONTEXT_SIZE = 5

# Fast Adaptation Optimization Settings
_B.TTA.COSA.FAST_ADAPTATION = True              # Enable fast adaptation optimization
_B.TTA.COSA.ADAPTIVE_LR = True                  # Enable adaptive learning rate adjustment
_B.TTA.COSA.MAX_LR = 0.005                      # Maximum learning rate
_B.TTA.COSA.MIN_LR = 0.0001                     # Minimum learning rate
_B.TTA.COSA.CONVERGENCE_THRESHOLD = 1e-4        # Early stopping convergence threshold
_B.TTA.COSA.VAR_WISE_GATING = True              # Enable variable-wise gating
_B.TTA.COSA.PER_BATCH_LR_RESET = True           # Reset learning rate at the start of each batch

# CSV Export Settings
_B.TTA.COSA.SAVE_CSV = False                    # Enable detailed CSV export of predictions
_B.TTA.COSA.SAVE_PAAS_CSV = False               # Enable CSV export of PAAS batch size information
_B.TTA.COSA.PAAS = False
_B.TTA.COSA.PERIOD_N = 1

_B.TTA.COSA.POGT = True                         # Replay each batch on its full horizon once observed
_B.TTA.COSA.PARTIAL_ADAPT = True                # Adapt on the observed horizon prefix at batch arrival
_B.TTA.COSA.PARTIAL_STEPS = 0                   # Steps for partial adaptation (0 -> reuse STEPS)
_B.TTA.COSA.ADJUST_PRED = True                  # Splice post-partial-adaptation suffix into the emitted pred (TAFAS default)
_B.TTA.COSA.L2_REG = 1e-4                       # L2 penalty on adapter params during adaptation

# Adapter Architecture
# 'Linear': single nn.Linear (the original COSA adapter).
# 'MLP'   : Linear -> Tanh -> Dropout(0.1) -> Linear, width HIDDEN_DIM.
_B.TTA.COSA.ADAPTER_TYPE = 'Linear'
_B.TTA.COSA.HIDDEN_DIM = 64                     # Hidden width, used by ADAPTER_TYPE='MLP' only

## DynaTTA: Dynamic Test-Time Adaptation
_B.TTA.DYNATTA = CN()
_B.TTA.DYNATTA.MSE_BUFFER_SIZE = 256              # Size of MSE buffer for z-score computation
_B.TTA.DYNATTA.METRIC_HISTORY_SIZE = 256          # Size of metric history for normalization
_B.TTA.DYNATTA.ALPHA_MIN = 1e-4                   # Minimum adaptation rate
_B.TTA.DYNATTA.ALPHA_MAX = 1e-3                   # Maximum adaptation rate
_B.TTA.DYNATTA.KAPPA = 1.0                        # Sensitivity scale for adaptation rate
_B.TTA.DYNATTA.ETA = 0.1                          # Smoothing factor for adaptation rate
_B.TTA.DYNATTA.EPS = 1e-6                         # Numerical stability constant
_B.TTA.DYNATTA.WARMUP_FACTOR = 1                  # Warmup steps factor (multiplied by PRED_LEN)
_B.TTA.DYNATTA.UPDATE_BUFFERS_INTERVAL = 1        # Interval for updating buffers
_B.TTA.DYNATTA.UPDATE_METRICS_INTERVAL = 1        # Interval for updating metrics
_B.TTA.DYNATTA.RTAB_SIZE = 360                    # Recent Time-series Adaptation Buffer size
_B.TTA.DYNATTA.RDB_SIZE = 100                     # Representative Database size



def get_cfg_defaults(method="commitcast") -> CN:
    """Return independent defaults for stream replay or a baseline learner."""
    if method.lower() == "commitcast":
        return _C.clone()
    if method.lower() not in {"tafas", "petsa", "cosa"}:
        raise ValueError(f"Unknown method: {method}")
    cfg = _B.clone()
    if method.lower() == "cosa":
        cfg.TRAIN.BATCH_SIZE = 128
    return cfg


def get_norm_module_cfg(cfg):
    return getattr(cfg, cfg.NORM_MODULE.NAME.upper())


def get_norm_method(cfg):
    assert cfg.NORM_MODULE.NAME in ("RevIN", "SAN", "DishTS")
    return cfg.NORM_MODULE.NAME if cfg.NORM_MODULE.ENABLE else "NST"
