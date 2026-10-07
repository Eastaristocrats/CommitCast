"""Explicit observation-clock loops around the official learning routines.

TAFAS: https://github.com/kimanki/TAFAS (Modified MIT, non-commercial).
PETSA: https://github.com/BorealisAI/PETSA (CC BY-NC-SA 4.0).
COSA: https://github.com/bigbases/COSA_ICLR2026 (CC BY-NC-SA 4.0).
Original notices and licenses are retained in THIRD_PARTY.md and licenses/.

Only waits, publication events and removal of offline metrics differ from the
native loops. The optimizer/loss/update methods are the upstream methods.
These are ordinary source functions; no code is generated during execution.
"""
import time
import torch



def tafas_loop(self, prepare_inputs, forecast):
    batch_start = 0
    batch_end = 0
    batch_idx = 0
    is_last = False
    test_len = len(self.test_loader.dataset)
    self.switch_model_to_eval()
    for idx, inputs in enumerate(self.test_loader):
        enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)
        while batch_end < len(enc_window_all):
            yield ('wait', batch_start)
            enc_window_first = enc_window_all[batch_start]
            if self.cfg.TTA.TAFAS.PAAS:
                period, batch_size = self._calculate_period_and_batch_size(enc_window_first)
            else:
                batch_size = self.cfg.TTA.TAFAS.BATCH_SIZE
                period = batch_size - 1
            batch_end = batch_start + batch_size
            if batch_end > len(enc_window_all):
                batch_end = len(enc_window_all)
                batch_size = batch_end - batch_start
                is_last = True
            yield ('wait', batch_end - 1)
            self.cur_step += batch_size
            inputs = (enc_window_all[batch_start:batch_end], enc_window_stamp_all[batch_start:batch_end], dec_window_all[batch_start:batch_end], dec_window_stamp_all[batch_start:batch_end])
            self.pred_step_end_dict[batch_idx] = self.cur_step + self.cfg.DATA.PRED_LEN
            self.inputs_dict[batch_idx] = inputs
            self._adapt_with_full_ground_truth_if_available()
            yield ('wait', batch_start + min(period, self.cfg.DATA.PRED_LEN))
            pred, ground_truth = self._adapt_with_partial_ground_truth(inputs, period, batch_size, batch_idx)
            if self.cfg.TTA.TAFAS.ADJUST_PRED:
                pred, ground_truth = self._adjust_prediction(pred, inputs, batch_size, period)
            yield ('publish', batch_start, batch_end, period, pred.detach().clone())
            batch_start = batch_end
            batch_idx += 1


def petsa_loop(self, prepare_inputs, forecast):
    batch_start = 0
    batch_end = 0
    batch_idx = 0
    is_last = False
    test_len = len(self.test_loader.dataset)
    self.switch_model_to_eval()
    for idx, inputs in enumerate(self.test_loader):
        enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)
        while batch_end < len(enc_window_all):
            yield ('wait', batch_start)
            enc_window_first = enc_window_all[batch_start]
            if self.cfg.TTA.PETSA.PAAS:
                period, batch_size = self._calculate_period_and_batch_size(enc_window_first)
            else:
                batch_size = self.cfg.TTA.PETSA.BATCH_SIZE
                period = batch_size - 1
            batch_end = batch_start + batch_size
            if batch_end > len(enc_window_all):
                batch_end = len(enc_window_all)
                batch_size = batch_end - batch_start
                is_last = True
            yield ('wait', batch_end - 1)
            self.cur_step += batch_size
            inputs = (enc_window_all[batch_start:batch_end], enc_window_stamp_all[batch_start:batch_end], dec_window_all[batch_start:batch_end], dec_window_stamp_all[batch_start:batch_end])
            self.pred_step_end_dict[batch_idx] = self.cur_step + self.cfg.DATA.PRED_LEN
            self.inputs_dict[batch_idx] = inputs
            self._adapt_with_full_ground_truth_if_available()
            yield ('wait', batch_start + min(period, self.cfg.DATA.PRED_LEN))
            for _ in range(self.cfg.TTA.PETSA.STEPS):
                pred, ground_truth = self._adapt_with_partial_ground_truth(inputs, period, batch_size, batch_idx)
            if self.cfg.TTA.PETSA.ADJUST_PRED:
                pred, ground_truth = self._adjust_prediction(pred, inputs, batch_size, period)
            yield ('publish', batch_start, batch_end, period, pred.detach().clone())
            batch_start = batch_end
            batch_idx += 1


def cosa_loop(self, prepare_inputs, forecast):
    batch_start = 0
    batch_end = 0
    batch_idx = 0
    total_start_time = time.time()
    self.switch_model_to_eval()
    for _, inputs in enumerate(self.test_loader):
        enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)
        while batch_end < len(enc_window_all):
            yield ('wait', batch_start)
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
                self._save_paas_batch_info(batch_idx=batch_idx, batch_start=batch_start, calculated_batch_size=calculated_batch_size, actual_batch_size=batch_size, period=period, fft_dominant_freq=dominant_freq)
            yield ('wait', batch_end - 1)
            self.cur_step += batch_size
            batch_inputs = (enc_window_all[batch_start:batch_end], enc_window_stamp_all[batch_start:batch_end], dec_window_all[batch_start:batch_end], dec_window_stamp_all[batch_start:batch_end])
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
            self.pending_dict[batch_idx] = (original_pred, ground_truth, ctx_vec, batch_size, self.current_time_idx)
            if self.partial_adapt and n_observed > 0:
                adapted_pred = self._adapt_with_partial_ground_truth(original_pred, ground_truth, context_data, batch_size, batch_idx)
                if adapted_pred is not None:
                    pred = adapted_pred
                if self.adjust_pred:
                    pred = self._adjust_prediction(pred.clone(), original_pred, context_data, batch_size, n_observed)
            if self.save_csv:
                self._save_batch_predictions_to_csv(original_pred, pred, ground_truth, batch_idx)
            metric_start = time.time()
            yield ('publish', batch_start, batch_end, period, pred.detach().clone())
            self.time_stats['metric_computation'] += time.time() - metric_start
            self.time_counts['metric_computation'] += 1
            self.current_time_idx += batch_size
            batch_start = batch_end
            batch_idx += 1
