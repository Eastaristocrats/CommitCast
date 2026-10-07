"""Synthetic CUDA recovery/causality test using the shipped official learners."""
from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import redirect_stdout
import importlib
import io
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["tafas", "petsa", "cosa"], required=True)
    p.add_argument("--paas", action="store_true")
    p.add_argument("--mlp", action="store_true")
    p.add_argument("--model", choices=["DLinear", "FreTS", "iTransformer", "MICN", "OLS", "PatchTST"], default="DLinear")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    sys.path.insert(0, str(ROOT))
    from fcr.vendor import get_vendor
    sys.path.insert(0, str(get_vendor("cosa_current" if args.method == "cosa" else args.method)))
    import numpy as np
    import torch
    from config import get_cfg_defaults
    Model = importlib.import_module("models." + args.model).Model
    from fcr.native import NativeSession, query_snapshot, OneBatchLoader
    from fcr.native_check import verify_native_loop
    source_check = verify_native_loop(args.method.upper(), ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError("This check runs unchanged upstream CUDA code")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cfg = get_cfg_defaults()
    cfg.DATA.SEQ_LEN, cfg.DATA.LABEL_LEN, cfg.DATA.PRED_LEN, cfg.DATA.N_VAR = 16, 8, 8, 2
    cfg.MODEL.NAME = args.model
    for key, value in dict(seq_len=16, label_len=8, pred_len=8, enc_in=2, dec_in=2, c_out=2).items():
        cfg.MODEL[key] = value
    cfg.MODEL.individual = False
    cfg.MODEL.d_model, cfg.MODEL.d_ff, cfg.MODEL.n_heads = 16, 32, 2
    cfg.MODEL.e_layers, cfg.MODEL.d_layers = 1, 1
    cfg.TEST.SHUFFLE, cfg.TEST.DROP_LAST = False, False
    if "WANDB" in cfg:
        cfg.WANDB.ENABLE = False
    cfg.TTA.SOLVER.BASE_LR = .001
    cfg.TTA.SOLVER.WEIGHT_DECAY = .0001
    section = cfg.TTA[args.method.upper()]
    section.PAAS, section.BATCH_SIZE = args.paas, 5
    section.STEPS = 3 if args.method == "cosa" else 1
    section.ADJUST_PRED = True
    if args.method == "cosa":
        section.BUFFER_CONTEXT_SIZE = 3
        section.SAVE_CSV, section.SAVE_PAAS_CSV = False, False
        section.ADAPTER_TYPE = "MLP" if args.mlp else "Linear"
        section.POGT, section.PARTIAL_ADAPT = True, True
    n, h, context, channels = 38, 8, 16, 2
    t = np.arange(context+n+h-1, dtype=np.float32)
    timeline = np.stack([np.sin(t*2*np.pi/4)+t*.01, np.cos(t*2*np.pi/8)-t*.008], axis=-1)
    enc = np.stack([timeline[i:i+context] for i in range(n)])
    dec = np.stack([timeline[i+context-8:i+context+h] for i in range(n)])
    inputs = (torch.tensor(enc).cuda(), torch.zeros(n,context,4).cuda(),
              torch.tensor(dec).cuda(), torch.zeros(n,8+h,4).cuda())
    class Dataset:
        test = timeline
        def __len__(self):
            return n
    dataset = Dataset()
    module = importlib.import_module("tta." + args.method)
    module.get_test_dataloader = lambda conf: OneBatchLoader(inputs, dataset)
    torch.manual_seed(9)
    model = Model(cfg.MODEL).cuda()
    def create():
        torch.manual_seed(31)
        torch.cuda.manual_seed_all(31)
        return module.build_adapter(cfg.clone(), deepcopy(model), None)

    native = create()
    captured = []
    mse_function = module.F.mse_loss
    def mse_capture(pred, target, *a, **kw):
        if kw.get("reduction") == "none":
            captured.append(pred.detach().cpu().numpy().copy())
        return mse_function(pred, target, *a, **kw)
    module.F.mse_loss = mse_capture
    try:
        with redirect_stdout(io.StringIO()):
            native.adapt()
    finally:
        module.F.mse_loss = mse_function
    native_pred = np.concatenate(captured)
    controlled = create()
    session = NativeSession(controlled, args.method.upper(), inputs=inputs)
    queries = []
    for q in range(n+h):
        session.advance(q)
        if q < n:
            rng_before = torch.cuda.get_rng_state().clone()
            queries.append(query_snapshot(controlled, args.method.upper())(tuple(v[q:q+1] for v in inputs))[0])
            assert torch.equal(rng_before, torch.cuda.get_rng_state()), "Query changed native RNG"
    published = np.concatenate([e["prediction"] for e in session.publications])
    np.testing.assert_array_equal(native_pred, published)
    def same_state(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a,b), "Learner or optimizer tensor differs"
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                same_state(a[k], b[k])
        elif isinstance(a, (list,tuple)):
            assert len(a) == len(b)
            for x,y in zip(a,b):
                same_state(x,y)
        else:
            assert a == b, (a,b)
    same_state(native.state_dict(), controlled.state_dict())
    same_state(native.optimizer.state_dict(), controlled.optimizer.state_dict())
    assert native.n_adapt == controlled.n_adapt == len(session.step_events)
    if args.method == "cosa":
        same_state(list(native.sample_history), list(controlled.sample_history))
        same_state(list(native.loss_history), list(controlled.loss_history))
    # All current and earlier decisions must survive changes to unavailable labels.
    cutoff = 19
    changed = tuple(x.clone() for x in inputs)
    indices = torch.arange(n, device="cuda")[:,None] + torch.arange(h, device="cuda")[None,:]
    changed[2][:,-h:] += (indices >= cutoff)[:,:,None] * 17
    encoder_times = torch.arange(n, device="cuda")[:,None] - context + torch.arange(context, device="cuda")[None,:]
    changed[0].add_((encoder_times >= cutoff)[:,:,None] * 13)
    alternate = create()
    altered = NativeSession(alternate, args.method.upper(), inputs=changed)
    for q in range(cutoff+1):
        altered.advance(q)
        prediction = query_snapshot(alternate, args.method.upper())(tuple(v[q:q+1] for v in changed))[0]
        np.testing.assert_array_equal(queries[q], prediction)
    result = dict(status="PASS", method=args.method.upper(), model=args.model, paas=args.paas, mlp=args.mlp,
        origins=n, horizon=h, channels=channels, batches=len(session.publications),
        optimizer_steps=len(session.step_events), prediction_max_difference=0,
        learner_state_exact=True, optimizer_state_exact=True, query_rng_preserved=True,
        future_intervention_decisions=cutoff+1, syntax=source_check,
        validation_scope="synthetic CUDA; functional fidelity, not a paper performance reproduction")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
