# Source: COSA; original terms are in licenses/COSA and THIRD_PARTY.md.
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pandas as pd
import os
from collections import deque, defaultdict
from models.optimizer import get_optimizer
from models.forecast import forecast
from datasets.loader import get_test_dataloader
from utils.misc import prepare_inputs
from config import get_norm_method
import time

ADAPTER_TYPES = ('Linear', 'MLP')


class SimpleOutputAdapter(nn.Module):
    def __init__(self, pred_len: int, buffer_context_size: int = 5, n_vars: int = 1,
                 var_wise_gating: bool = False, adapter_type: str = 'Linear',
                 hidden_dim: int = 64):
        super().__init__()
        self.pred_len = pred_len
        self.buffer_context_size = buffer_context_size
        self.n_vars = n_vars
        self.var_wise = var_wise_gating
        self.adapter_type = adapter_type
        self.hidden_dim = hidden_dim

        assert adapter_type in ADAPTER_TYPES, \
            f"ADAPTER_TYPE must be one of {list(ADAPTER_TYPES)}, got {adapter_type!r}"

        input_dim = self.pred_len + self.buffer_context_size
        output_dim = self.pred_len

        if self.var_wise:
            self.fc_layers = nn.ModuleList([
                self._build_stack(input_dim, output_dim) for _ in range(n_vars)
            ])
            self.gate = nn.Parameter(torch.zeros(n_vars))
        else:
            self.fc = self._build_stack(input_dim, output_dim)
            self.gate = nn.Parameter(torch.zeros(1))
        self._initialize_parameters()

    def _build_stack(self, input_dim: int, output_dim: int):

        if self.adapter_type == 'Linear':
            return nn.Linear(input_dim, output_dim)

        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def _initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)

    def forward(self, y: torch.Tensor, context_data: torch.Tensor = None):
        if context_data is None:
            return y
        
        batch_size, pred_len, n_vars = y.shape
        
        if self.var_wise:
            corrections = []
            for var_idx in range(n_vars):
                y_var = y[:, :, var_idx]
                
                combined_input = torch.cat([y_var, context_data], dim=-1)
                
                
                correction_var = self.fc_layers[var_idx](combined_input)
                corrections.append(correction_var.unsqueeze(-1))

            correction = torch.cat(corrections, dim=-1)
            
            gating_factor = torch.tanh(self.gate).unsqueeze(0).unsqueeze(0)
            
        else:
            y_flattened = y.transpose(1, 2).contiguous().view(batch_size * n_vars, pred_len)
            context_repeated = context_data.unsqueeze(1).repeat(1, n_vars, 1).view(batch_size * n_vars, -1)
            combined_input = torch.cat([y_flattened, context_repeated], dim=-1)
            correction = self.fc(combined_input)
            correction = correction.view(batch_size, n_vars, pred_len).transpose(1, 2)
            gating_factor = torch.tanh(self.gate)
        
        return y + gating_factor * correction
    


class SimpleAdapter(nn.Module):
    def __init__(self, cfg, model: nn.Module, norm_module=None):
        super(SimpleAdapter, self).__init__()
        self.cfg = cfg
        self.model_cfg = cfg.MODEL
        self.model = model
        self.norm_method = get_norm_method(cfg)
        self.norm_module = norm_module
        self.test_loader = get_test_dataloader(cfg)
        self.test_data = self.test_loader.dataset.test

        self.batch_size = cfg.TTA.COSA.BATCH_SIZE
        self.buffer_context_size = cfg.TTA.COSA.BUFFER_CONTEXT_SIZE
        self.adapt_steps = cfg.TTA.COSA.STEPS

        self.paas_enabled = cfg.TTA.COSA.PAAS
        self.period_n = cfg.TTA.COSA.PERIOD_N

        self.fast_adaptation = cfg.TTA.COSA.FAST_ADAPTATION
        self.adaptive_lr = cfg.TTA.COSA.ADAPTIVE_LR
        self.max_lr = cfg.TTA.COSA.MAX_LR
        self.min_lr = cfg.TTA.COSA.MIN_LR
        self.convergence_threshold = cfg.TTA.COSA.CONVERGENCE_THRESHOLD
        self.var_wise_gating = cfg.TTA.COSA.VAR_WISE_GATING

        self.save_csv = cfg.TTA.COSA.SAVE_CSV
        self.csv_predictions = []

        self.save_paas_csv = cfg.TTA.COSA.SAVE_PAAS_CSV
        self.paas_batch_info = []

        self.adapter_type = cfg.TTA.COSA.ADAPTER_TYPE
        self.hidden_dim = cfg.TTA.COSA.HIDDEN_DIM

        self.pogt = cfg.TTA.COSA.POGT
        self.partial_adapt = cfg.TTA.COSA.PARTIAL_ADAPT
        self.partial_steps = cfg.TTA.COSA.PARTIAL_STEPS or self.adapt_steps
        self.adjust_pred = cfg.TTA.COSA.ADJUST_PRED
        self.l2_reg = cfg.TTA.COSA.L2_REG

        self.pred_step_end_dict = {}
        self.pending_dict = {}

        self.per_batch_lr_reset = cfg.TTA.COSA.PER_BATCH_LR_RESET

        self.loss_history = deque(maxlen=5)
        self.current_lr = cfg.TTA.SOLVER.BASE_LR

        self.sample_history = deque(maxlen=200) 
        self.current_time_idx = 0
        
        self.time_stats = defaultdict(float)
        self.time_counts = defaultdict(int)
        
        self.output_adapter = SimpleOutputAdapter(
            pred_len=cfg.DATA.PRED_LEN,
            buffer_context_size=self.buffer_context_size,
            n_vars=cfg.DATA.N_VAR,
            var_wise_gating=self.var_wise_gating,
            adapter_type=self.adapter_type,
            hidden_dim=self.hidden_dim
        ).cuda()
        
        self.step_count = 0

        self._freeze_all_model_params()
        self._unfreeze_adapter_params()
        
        self.optimizer = get_optimizer(self.output_adapter.parameters(), cfg.TTA)
        
        self.model_state, self.optimizer_state = self._copy_model_and_optimizer()

        cfg.TEST.BATCH_SIZE = len(self.test_loader.dataset)
        self.test_loader = get_test_dataloader(cfg)
        self.cur_step = cfg.DATA.SEQ_LEN - 2
        self.n_adapt = 0

        self.mse_all = []
        self.mae_all = []

    def count_parameters(self):
        trainable_params = []
        total_sum = 0
        
        for name, param in self.named_parameters():
            param_info = {
                "name": name,
                "requires_grad": param.requires_grad,
                "size": list(param.size()),
                "numel": int(param.numel())
            }
            trainable_params.append(param_info)
            
            if param.requires_grad:
                total_sum += int(param.numel())
        
        param_json = {
            "model": "SimpleAdapter",
            "parameters": {
                "trainable_params": trainable_params,
                "total_params": total_sum
            }
        }
        return param_json


    def forward(self, enc_window, enc_window_stamp, dec_window, dec_window_stamp):
        raise NotImplementedError
    
    def reset(self):
        self._load_model_and_optimizer()
        self.step_count = 0
        
        self.loss_history.clear()
        self.current_lr = self.cfg.TTA.SOLVER.BASE_LR
        self.pred_step_end_dict.clear()
        self.pending_dict.clear()
    
    def _copy_model_and_optimizer(self):
        model_state = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_state, optimizer_state

    def _load_model_and_optimizer(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
    
    def _get_all_models(self):
        models = [self.model, self.output_adapter]
        if self.norm_module is not None:
            models.append(self.norm_module)
        return models

    def _freeze_all_model_params(self):
        for param in self.model.parameters():
            param.requires_grad_(False)
        if self.norm_module is not None:
            for param in self.norm_module.parameters():
                param.requires_grad_(False)
    
    def _unfreeze_adapter_params(self):
        for param in self.output_adapter.parameters():
            param.requires_grad_(True)
    
    def switch_model_to_train(self):
        self.model.eval() 
        if self.norm_module is not None:
            self.norm_module.eval()
        self.output_adapter.train()
    
    def switch_model_to_eval(self):
        self.model.eval()
        if self.norm_module is not None:
            self.norm_module.eval()
        self.output_adapter.eval()
    

    def _push_history(self, target_mean: float, time_idx: int):
        self.sample_history.append({
            'time_idx': time_idx,
            'target_mean': target_mean,
        })
        self.step_count += 1


    def _calculate_period_and_batch_size(self, enc_window_first):

        fft_result = torch.fft.rfft(enc_window_first - enc_window_first.mean(dim=0), dim=0)
        amplitude = torch.abs(fft_result)
        power = torch.mean(amplitude ** 2, dim=0)
        
        dominant_freq = None
        try:
            dominant_freq_idx = torch.argmax(amplitude[:, power.argmax()]).item()
            dominant_freq = float(dominant_freq_idx / enc_window_first.shape[0])
            period = enc_window_first.shape[0] // dominant_freq_idx
        except:
            period = 24 
            dominant_freq = 1.0 / 24 
            
        period *= self.period_n
        batch_size = period + 1
        return period, batch_size, dominant_freq

    def _get_context_vector(self):

        if len(self.sample_history) == 0:
            return torch.zeros(self.buffer_context_size, device='cuda')

        history_size = min(self.buffer_context_size, len(self.sample_history))
        context_values = [self.sample_history[-(i+1)]['target_mean']
                         for i in range(history_size)]

        if len(context_values) < self.buffer_context_size:
            last_val = context_values[-1] if context_values else 0.0
            context_values.extend([last_val] * (self.buffer_context_size - len(context_values)))

        return torch.tensor(context_values, dtype=torch.float32, device='cuda')

    def _get_individual_context_for_batch(self, batch_size, current_batch_idx):
        return self._get_context_vector().unsqueeze(0).expand(batch_size, -1)


    def _adaptive_learning_rate(self, current_loss: float, step: int, batch_idx: int = 0) -> float:
        if step == 0 and self.per_batch_lr_reset:
            self.current_lr = self.cfg.TTA.SOLVER.BASE_LR
            self.loss_history.append(current_loss)
            return self.current_lr

        self.loss_history.append(current_loss)
        recent_losses = list(self.loss_history)[-3:]

        if len(recent_losses) < 2:
            return self.current_lr

        loss_trend = recent_losses[-1] - recent_losses[0]
        loss_variance = torch.tensor(recent_losses).var().item()

        if loss_trend > 0 and loss_variance < 1e-6:
            self.current_lr = min(self.current_lr * 1.2, self.max_lr)
        elif loss_trend < -0.01:
            self.current_lr = min(self.current_lr * 1.05, self.max_lr)
        elif abs(loss_trend) < 1e-6:
            self.current_lr = max(self.current_lr * 0.8, self.min_lr)

        if step >= 1:
            cosine_factor = 0.5 * (1 + torch.cos(torch.tensor(step * 3.14159 / self.adapt_steps)))
            self.current_lr = self.min_lr + (self.current_lr - self.min_lr) * cosine_factor

        return self.current_lr
    
    def _save_batch_predictions_to_csv(self, original_pred: torch.Tensor, tta_pred: torch.Tensor, ground_truth: torch.Tensor, batch_idx: int):
        batch_size, pred_len, n_vars = tta_pred.shape
        
        original_pred_np = original_pred.detach().cpu().numpy()
        tta_pred_np = tta_pred.detach().cpu().numpy()
        ground_truth_np = ground_truth.detach().cpu().numpy()
        
        for sample_idx in range(batch_size):
            for time_idx in range(pred_len):
                for var_idx in range(n_vars):
                    orig_val = float(original_pred_np[sample_idx, time_idx, var_idx])
                    tta_val = float(tta_pred_np[sample_idx, time_idx, var_idx])
                    gt_val = float(ground_truth_np[sample_idx, time_idx, var_idx])
                    
                    row_data = {
                        'batch_idx': batch_idx,
                        'sample_idx': sample_idx,
                        'global_sample_idx': batch_idx * batch_size + sample_idx,
                        'timestep': time_idx + 1, 
                        'variable_idx': var_idx,
                        'original_prediction': orig_val,
                        'tta_prediction': tta_val,
                        'ground_truth': gt_val,
                        'tta_improvement': tta_val - orig_val,
                        'original_absolute_error': float(abs(orig_val - gt_val)),
                        'tta_absolute_error': float(abs(tta_val - gt_val)),
                        'original_squared_error': float((orig_val - gt_val)**2),
                        'tta_squared_error': float((tta_val - gt_val)**2),
                        'error_improvement': float(abs(orig_val - gt_val) - abs(tta_val - gt_val)) 
                    }
                    self.csv_predictions.append(row_data)
    
    def _export_predictions_to_csv(self):
        if not self.csv_predictions:
            return
        
        df = pd.DataFrame(self.csv_predictions)
        
        model_name = self.cfg.MODEL.NAME
        dataset_name = self.cfg.DATA.NAME
        pred_len = self.cfg.DATA.PRED_LEN
        
        csv_dir = os.path.join(self.cfg.RESULT_DIR, "csv_predictions", "COSA")
        os.makedirs(csv_dir, exist_ok=True)
        
        csv_filename = f"{model_name}_{dataset_name}_pred{pred_len}_predictions.csv"
        csv_path = os.path.join(csv_dir, csv_filename)
        
        df.to_csv(csv_path, index=False)
    
    def _save_paas_batch_info(self, batch_idx: int, batch_start: int, calculated_batch_size: int, 
                             actual_batch_size: int, period: int = None, fft_dominant_freq: float = None):
        batch_info = {
            'batch_idx': batch_idx,
            'batch_start_idx': batch_start,
            'calculated_batch_size': calculated_batch_size,
            'actual_batch_size': actual_batch_size,
            'period': period if period is not None else 'N/A',
            'fft_dominant_frequency': fft_dominant_freq if fft_dominant_freq is not None else 'N/A',
            'paas_enabled': self.paas_enabled,
            'period_n_multiplier': self.period_n
        }
        self.paas_batch_info.append(batch_info)
    
    def _export_paas_info_to_csv(self):
        if not self.paas_batch_info:
            return
        
        df = pd.DataFrame(self.paas_batch_info)
        
        model_name = self.cfg.MODEL.NAME
        dataset_name = self.cfg.DATA.NAME
        pred_len = self.cfg.DATA.PRED_LEN
        
        csv_dir = os.path.join(self.cfg.RESULT_DIR, "csv_predictions", "COSA")
        os.makedirs(csv_dir, exist_ok=True)
        
        csv_filename = f"{model_name}_{dataset_name}_pred{pred_len}_paas_batch_sizes.csv"
        csv_path = os.path.join(csv_dir, csv_filename)
        
        df.to_csv(csv_path, index=False)
    
    def _run_adapt_steps(self, pred_const, ground_truth, context_data, steps,
                         batch_idx, tag, loss_mask=None):

        effective_steps = steps
        if self.fast_adaptation:
            effective_steps = min(steps, 5)

        if loss_mask is not None:
            mask = loss_mask.unsqueeze(-1)
            denom = loss_mask.sum() * ground_truth.shape[-1]

        last_pred = None
        adapt_start = time.time()
        for step in range(effective_steps):
            step_start = time.time()

            self.n_adapt += 1
            self.switch_model_to_train()

            forward_start = time.time()
            adapted_pred = self.output_adapter(pred_const, context_data)
            last_pred = adapted_pred.detach()
            self.time_stats['adapter_forward'] += time.time() - forward_start
            self.time_counts['adapter_forward'] += 1

            loss_start = time.time()
            if loss_mask is None:
                loss = F.mse_loss(adapted_pred, ground_truth)
            else:
                loss = ((adapted_pred - ground_truth).pow(2) * mask).sum() / denom

            if not hasattr(self, '_adapter_params'):
                self._adapter_params = list(self.output_adapter.parameters())
            l2_reg = sum(p.pow(2).sum() for p in self._adapter_params if p.requires_grad)
            loss += self.l2_reg * l2_reg

            self.time_stats['loss_computation'] += time.time() - loss_start
            self.time_counts['loss_computation'] += 1

            if self.adaptive_lr and self.fast_adaptation:
                current_lr = self._adaptive_learning_rate(loss.item(), step, batch_idx)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr

            backward_start = time.time()
            self.optimizer.zero_grad()
            loss.backward()

            if self.fast_adaptation:
                max_norm = max(0.05, min(0.5, loss.item()))
                torch.nn.utils.clip_grad_norm_(self.output_adapter.parameters(), max_norm=max_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.output_adapter.parameters(), max_norm=0.1)

            self.optimizer.step()
            self.time_stats['backward_update'] += time.time() - backward_start
            self.time_counts['backward_update'] += 1

            self.switch_model_to_eval()

            if (self.fast_adaptation and step > 2 and len(self.loss_history) >= 2 and
                abs(self.loss_history[-1] - self.loss_history[-2]) < self.convergence_threshold):
                break

            self.time_stats['per_adaptation_step'] += time.time() - step_start
            self.time_counts['per_adaptation_step'] += 1

        elapsed = time.time() - adapt_start
        self.time_stats['total_adaptation'] += elapsed
        self.time_counts['total_adaptation'] += 1
        self.time_stats[f'{tag}_adaptation'] += elapsed
        self.time_counts[f'{tag}_adaptation'] += 1
        return last_pred

    def _drain_observed_batches(self):

        while self.pred_step_end_dict and \
                self.cur_step >= self.pred_step_end_dict[min(self.pred_step_end_dict)]:
            idx = min(self.pred_step_end_dict)
            self.pred_step_end_dict.pop(idx)
            cached_pred, cached_gt, ctx_vec, bsz, time_idx = self.pending_dict.pop(idx)

            buffer_start = time.time()
            self._push_history(cached_gt.mean().item(), time_idx)
            self.time_stats['buffer_update'] += time.time() - buffer_start
            self.time_counts['buffer_update'] += 1

            if self.pogt:
                self._run_adapt_steps(cached_pred, cached_gt,
                                      ctx_vec.unsqueeze(0).expand(bsz, -1),
                                      self.adapt_steps, idx, tag='full')

    def _build_partial_mask(self, batch_size, pred_len):

        mask = torch.zeros(batch_size, pred_len, dtype=torch.bool, device='cuda')
        for j in range(batch_size):
            obs_j = min(max(batch_size - 1 - j, 0), pred_len)
            if obs_j > 0:
                mask[j, :obs_j] = True
        return mask

    def _adapt_with_partial_ground_truth(self, pred_const, ground_truth, context_data,
                                         batch_size, batch_idx):
        pred_len = ground_truth.shape[1]
        mask = self._build_partial_mask(batch_size, pred_len)
        if not bool(mask.any()):
            return None
        return self._run_adapt_steps(pred_const, ground_truth, context_data,
                                     self.partial_steps, batch_idx, tag='partial',
                                     loss_mask=mask)

    def _adjust_prediction(self, pred, pred_const, context_data, batch_size, n_observed):

        with torch.no_grad():
            pred_post = self.output_adapter(pred_const, context_data)
        for i in range(batch_size - 1):
            start = max(n_observed - i, 0)
            pred[i, start:] = pred_post[i, start:]
        return pred

    @torch.enable_grad()
    def adapt_simple(self):
        batch_start = 0
        batch_end = 0
        batch_idx = 0
        
        total_start_time = time.time()
        
        self.switch_model_to_eval()
        
        for _, inputs in enumerate(self.test_loader):
            enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)
            
            while batch_end < len(enc_window_all):
                calculated_batch_size = None
                period = None
                dominant_freq = None
                
                if self.paas_enabled:
                    period_start = time.time()
                    enc_window_first = enc_window_all[batch_start]
                    period, calculated_batch_size, dominant_freq = self._calculate_period_and_batch_size(enc_window_first)
                    batch_size = calculated_batch_size
                    self.time_stats['paas_period_calculation'] += time.time() - period_start
                    self.time_counts['paas_period_calculation'] += 1
                else:
                    batch_size = self.batch_size
                    calculated_batch_size = batch_size
                
                batch_end = batch_start + batch_size
                if batch_end > len(enc_window_all):
                    batch_end = len(enc_window_all)
                    batch_size = batch_end - batch_start
                
                if self.save_paas_csv:
                    self._save_paas_batch_info(
                        batch_idx=batch_idx,
                        batch_start=batch_start,
                        calculated_batch_size=calculated_batch_size,
                        actual_batch_size=batch_size,
                        period=period,
                        fft_dominant_freq=dominant_freq
                    )

                self.cur_step += batch_size

                batch_inputs = (
                    enc_window_all[batch_start:batch_end], 
                    enc_window_stamp_all[batch_start:batch_end], 
                    dec_window_all[batch_start:batch_end], 
                    dec_window_stamp_all[batch_start:batch_end]
                )
                
                pred_start = time.time()
                pred, ground_truth = forecast(self.cfg, batch_inputs, self.model, self.norm_module)
                original_pred = pred.detach()
                self.time_stats['base_prediction'] += time.time() - pred_start
                self.time_counts['base_prediction'] += 1

                self._drain_observed_batches()

                context_start = time.time()
                ctx_vec = self._get_context_vector()
                context_data = ctx_vec.unsqueeze(0).expand(batch_size, -1)
                self.time_stats['context_generation'] += time.time() - context_start
                self.time_counts['context_generation'] += 1

                final_start = time.time()
                with torch.no_grad():
                    pred = self.output_adapter(original_pred, context_data)
                self.time_stats['final_prediction'] += time.time() - final_start
                self.time_counts['final_prediction'] += 1

                n_observed = min(batch_size - 1, self.cfg.DATA.PRED_LEN)

                self.pred_step_end_dict[batch_idx] = self.cur_step + self.cfg.DATA.PRED_LEN
                self.pending_dict[batch_idx] = (
                    original_pred, ground_truth, ctx_vec, batch_size, self.current_time_idx
                )

                if self.partial_adapt and n_observed > 0:
                    adapted_pred = self._adapt_with_partial_ground_truth(
                        original_pred, ground_truth, context_data,
                        batch_size, batch_idx
                    )
                    if adapted_pred is not None:
                        pred = adapted_pred
                    if self.adjust_pred:
                        pred = self._adjust_prediction(
                            pred.clone(), original_pred, context_data, batch_size, n_observed
                        )

                if self.save_csv:
                    self._save_batch_predictions_to_csv(original_pred, pred, ground_truth, batch_idx)

                metric_start = time.time()
                mse = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                mae = F.l1_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                self.time_stats['metric_computation'] += time.time() - metric_start
                self.time_counts['metric_computation'] += 1

                self.mse_all.append(mse)
                self.mae_all.append(mae)

                self.current_time_idx += batch_size
                batch_start = batch_end
                batch_idx += 1
        
        self.time_stats['total_time'] = time.time() - total_start_time
        
        assert self.cur_step == len(self.test_data) - self.cfg.DATA.PRED_LEN - 1
        
        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
        assert len(self.mse_all) == len(self.test_loader.dataset)
        
        if self.save_csv:
            self._export_predictions_to_csv()
        
        if self.save_paas_csv:
            self._export_paas_info_to_csv()
    
        self._print_combined_results()
        
    
    def _print_combined_results(self):
        total_params = 0
        for _, param in self.named_parameters():
            if param.requires_grad:
                total_params += int(param.numel())
        
        avg_times = {}
        for key in self.time_stats:
            if key != 'total_time':
                count = self.time_counts[key] if self.time_counts[key] > 0 else 1
                avg_times[key] = self.time_stats[key] / count
        
        adapter_total = (avg_times.get('context_generation', 0) + 
                        avg_times.get('adapter_forward', 0) + 
                        avg_times.get('final_prediction', 0))
        
        time_statistics = {
            "adapter_operations": {},
            "adaptation_training": {},
            "other_operations": {},
            "overall_stats": {}
        }
        
        time_statistics["adapter_operations"] = {
            "context_generation_ms": round(avg_times.get('context_generation', 0) * 1000, 3),
            "adapter_forward_pass_ms": round(avg_times.get('adapter_forward', 0) * 1000, 3),
            "final_prediction_ms": round(avg_times.get('final_prediction', 0) * 1000, 3),
            "total_adapter_operation_ms": round(adapter_total * 1000, 3)
        }
        
        adaptation_training = {
            "loss_computation_ms": round(avg_times.get('loss_computation', 0) * 1000, 3),
            "backward_update_ms": round(avg_times.get('backward_update', 0) * 1000, 3),
            "per_adaptation_step_ms": round(avg_times.get('per_adaptation_step', 0) * 1000, 3),
            "total_per_batch_ms": round(avg_times.get('total_adaptation', 0) * 1000, 3),
            "adaptation_steps": self.adapt_steps,
            "pogt": bool(self.pogt),
            "full_gt_adaptation_ms": round(avg_times.get('full_adaptation', 0) * 1000, 3),
            "full_gt_adaptation_count": int(self.time_counts.get('full_adaptation', 0)),
            "partial_gt_adaptation_ms": round(avg_times.get('partial_adaptation', 0) * 1000, 3),
            "partial_gt_adaptation_count": int(self.time_counts.get('partial_adaptation', 0)),
            "pending_at_teardown": len(self.pending_dict),
        }
        time_statistics["adaptation_training"] = adaptation_training
        
        other_ops = {
            "base_model_prediction_ms": round(avg_times.get('base_prediction', 0) * 1000, 3),
            "buffer_update_ms": round(avg_times.get('buffer_update', 0) * 1000, 3),
            "metric_computation_ms": round(avg_times.get('metric_computation', 0) * 1000, 3)
        }
        if self.paas_enabled:
            other_ops["paas_period_calculation_ms"] = round(avg_times.get('paas_period_calculation', 0) * 1000, 3)
        time_statistics["other_operations"] = other_ops
        
        adaptation_count = max(self.time_counts.get('total_adaptation', 1), 1)
        
        time_statistics["overall_stats"] = {
            "total_time_seconds": round(self.time_stats['total_time'], 2),
            "total_adaptations": int(self.n_adapt),
            "avg_time_per_adaptation_ms": round(self.time_stats.get('total_adaptation', 0) / adaptation_count * 1000, 3),
            "throughput_samples_per_sec": round(len(self.test_loader.dataset) / self.time_stats['total_time'], 1)
        }
        
        combined_results = {
            "model": "SimpleAdapter",
            "time_statistics": time_statistics,
            "final_results": {
                "adaptation_count": int(self.n_adapt),
                "test_mse": float(self.mse_all.mean())
            },
            "parameters": {
                "total_params": total_params
            }
        }
        
        print(json.dumps(combined_results, indent=2))
    def adapt(self):
        self.adapt_simple()


def build_adapter(cfg, model, norm_module=None):
    adapter = SimpleAdapter(cfg, model, norm_module)
    return adapter
