"""Frozen forecast streams and shared checkpoint-backed forecasting.

The forecast function follows PETSA/TAFAS; see THIRD_PARTY.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from config import get_norm_method
from datasets.loader import StreamRecord, load_stream
from utils.misc import prepare_inputs


@dataclass(frozen=True)
class ForecastStream:
    record: StreamRecord
    pred: np.ndarray
    true: np.ndarray


class FrozenForecastModel:
    """Read-only model facade over chronological frozen forecasts.

    This is the black-box counterpart of COSA's checkpoint-backed forecasting
    model: ``pred[i]`` is the output produced by the frozen checkpoint at origin
    ``i``. CommitCast never modifies it.
    """

    def __init__(self, records: list[StreamRecord]):
        self.records = tuple(records)
        self.training = False

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        if mode:
            raise RuntimeError("FrozenForecastModel cannot enter training mode")
        return self.eval()

    def streams(self):
        for record in self.records:
            pred, true = load_stream(record)
            yield ForecastStream(
                record=record,
                pred=pred,
                true=true,
            )

    def count_parameters(self) -> int:
        return 0



##########################################################################################
# Code is originally from the TAFAS (https://arxiv.org/pdf/2501.04970.pdf) implementation
# from https://github.com/kimanki/TAFAS by Kim et al. which is licensed under 
# Modified MIT License (Non-Commercial with Permission).
# You may obtain a copy of the License at
#
#    https://github.com/kimanki/TAFAS/blob/master/LICENSE
#
###########################################################################################

def forecast(
    cfg, 
    inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
    model: nn.Module,
    norm_module: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
    norm_method = get_norm_method(cfg)
    if norm_method == 'SAN':
        enc_window, statistics = norm_module.normalize(enc_window)
    elif norm_method == 'RevIN':
        enc_window = norm_module(enc_window, 'norm')
    elif norm_method == 'DishTS':
        enc_window, _ = norm_module(enc_window, 'forward')
    else:  # Normalization from Non-stationary Transformer
        means = enc_window.mean(1, keepdim=True).detach()
        enc_window = enc_window - means
        # stdev = torch.sqrt(torch.var(enc_window, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev = torch.sqrt(torch.var(enc_window, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        enc_window /= stdev
    
    ground_truth = dec_window[:, -cfg.DATA.PRED_LEN:, cfg.DATA.TARGET_START_IDX:].float()
    dec_zeros = torch.zeros_like(dec_window[:, -cfg.DATA.PRED_LEN:, :]).float()
    if torch.cuda.is_available():
        dec_window = torch.cat([dec_window[:, :cfg.DATA.LABEL_LEN:, :], dec_zeros], dim=1).float().cuda()
    else:
        dec_window = torch.cat([dec_window[:, :cfg.DATA.LABEL_LEN:, :], dec_zeros], dim=1).float()
    
    model_cfg = cfg.MODEL
    if model_cfg.output_attention:
        pred = model(enc_window, enc_window_stamp, dec_window, dec_window_stamp)[0]
    else:
        pred = model(enc_window, enc_window_stamp, dec_window, dec_window_stamp)
    
    pred = pred[:, -cfg.DATA.PRED_LEN:, cfg.DATA.TARGET_START_IDX:]
    
    if norm_method == 'SAN':
        pred = norm_module.de_normalize(pred, statistics)
    elif norm_method == 'RevIN':
        pred = norm_module(pred, 'denorm')
    elif norm_method == 'DishTS':
        pred = norm_module(pred, 'inverse')
    else:  # De-Normalization from Non-stationary Transformer
        pred = pred * (stdev[:, 0, :].unsqueeze(1).repeat(1, cfg.DATA.PRED_LEN, 1))
        pred = pred + (means[:, 0, :].unsqueeze(1).repeat(1, cfg.DATA.PRED_LEN, 1))
    
    return pred, ground_truth
