"""Resume official methods at observation/publication boundaries.

Learning functions are called unchanged. Only offline diagnostic scoring is
removed from the online loop; complete native evaluation remains available.
"""
from __future__ import annotations

from copy import deepcopy
import importlib
import torch


class OneBatchLoader(list):
    def __init__(self, inputs, dataset):
        super().__init__([inputs])
        self.dataset = dataset


class NativeSession:
    """Labels with absolute relative target index tau are usable only if tau < q.

    Native batch sizes, update order, partial/full losses, CALR, history and
    ADJUST_PRED remain the author's implementation. Tail partial updates wait
    for their actual labels. No queued full update is prematurely flushed.
    """
    def __init__(self, adapter, family, inputs=None):
        if family not in {"TAFAS", "PETSA", "COSA"}:
            raise ValueError("Unknown method family")
        self.adapter, self.family = adapter, family
        if inputs is None:
            batches = list(adapter.test_loader)
            if len(batches) != 1:
                raise ValueError("The official adapter must expose its concatenated input batch")
            inputs = batches[0]
        device = next(adapter.parameters()).device
        values = tuple(x.detach().clone().float().to(device) for x in inputs)
        self.horizon = int(adapter.cfg.DATA.PRED_LEN)
        self.vault = values[2][:, -self.horizon:].clone()
        self.released = values[2][:, -self.horizon:]
        self.released.zero_()
        self.targets = torch.arange(len(values[0]), device=device)[:, None] + torch.arange(self.horizon, device=device)[None, :]
        adapter.test_loader = OneBatchLoader(values, adapter.test_loader.dataset)
        from fcr.native_loops import tafas_loop, petsa_loop, cosa_loop
        module = importlib.import_module("tta." + family.lower())
        loop = {"TAFAS": tafas_loop, "PETSA": petsa_loop, "COSA": cosa_loop}[family]
        self.generator = loop(adapter, module.prepare_inputs, module.forecast)
        self.proof = dict(implementation="explicit source in fcr/native_loops.py",
                          upstream=family, runtime_code_generation=False)
        self.pending = None
        self.clock = 0
        self.available = -1
        self.finished = False
        self.publications = []
        self.step_events = []
        self.revision = 0
        raw_step = adapter.optimizer.step

        def record_step(*args, **kwargs):
            result = raw_step(*args, **kwargs)
            self.step_events.append(dict(clock=self.clock,
                learning_rates=[float(g["lr"]) for g in adapter.optimizer.param_groups]))
            return result

        adapter.optimizer.step = record_step

    def advance(self, before):
        if int(before) != before or before < self.available:
            raise ValueError("Observation clock must be integral and monotone")
        self.available = int(before)
        with torch.enable_grad():
            while not self.finished:
                if self.pending is None:
                    try:
                        self.pending = next(self.generator)
                        self.revision += 1
                    except StopIteration:
                        self.finished = True
                        break
                event = self.pending
                if event[0] == "wait":
                    if event[1] > self.available:
                        return
                    self.clock = max(self.clock, int(event[1]))
                    self.released.copy_(torch.where((self.targets < self.clock)[:, :, None], self.vault, 0.))
                else:
                    _, start, end, period, pred = event
                    if self.clock < end - 1:
                        raise AssertionError("Publication precedes its latest input")
                    self.publications.append(dict(start=int(start), end=int(end),
                        period=None if period is None else int(period),
                        clock=self.clock, prediction=pred.detach().cpu().numpy().copy()))
                self.pending = None


def query_snapshot(adapter, family):
    """Independent evaluation modules; queries cannot mutate the online learner.

    Deep copying the module tuple also preserves aliases, such as calibration
    modules attached to a backbone. The original optimizer and RNG are untouched.
    """
    model, norm, cali, output = deepcopy((adapter.model, adapter.norm_module,
        getattr(adapter, "cali", None), getattr(adapter, "output_adapter", None)))
    for module in (model, norm, cali, output):
        if module is not None:
            module.eval()
    forecast = importlib.import_module("tta." + family.lower()).forecast
    cfg = adapter.cfg
    context = adapter._get_context_vector().detach().clone() if family == "COSA" else None
    device = next(model.parameters()).device
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []

    def query(inputs):
        values = tuple(x.detach().clone().to(device) for x in inputs)
        values[2][:, -cfg.DATA.PRED_LEN:] = 0
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            if family != "COSA" and cali is not None and cfg.MODEL.NAME != "PatchTST":
                values = cali.input_calibration(values)
            pred, _ = forecast(cfg, values, model, norm)
            if family == "COSA":
                pred = output(pred, context.unsqueeze(0).expand(len(pred), -1))
            elif cali is not None:
                pred = cali.output_calibration(pred)
        return pred.detach().cpu().numpy()

    return query
