"""Run the server CommitCast Full/PG/P/G kernel under complete-H schedules."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from datasets.loader import load_stream
from fcr.replay import replay_schedules
from fcr.scoring import SCHEDULES, common_origins, event_boundaries, validate_event_timeline
from tta.commitcast import CommitCastConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--feature-mode', choices=['full','pg','p','g'], default='full')
    p.add_argument('--schedules', default='all')
    p.add_argument('--support', choices=['common','schedule'], default='common')
    p.add_argument('--device', default='cpu')
    p.add_argument('--event-threshold', type=Path)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    base,truth = load_stream(args.stream)
    names = list(SCHEDULES) if args.schedules == 'all' else args.schedules.split(',')
    plans = None
    if args.event_threshold:
        if 'event50' not in names:
            names.append('event50')
        locked = json.loads(args.event_threshold.read_text(encoding='utf-8'))
        if locked.get('selected_on') != 'val' or locked.get('rule') != 'observed_rms_24':
            raise ValueError('Event threshold must be frozen on validation')
        support = list(SCHEDULES.values()) if args.support == 'common' else [SCHEDULES.get(x,SCHEDULES['dense8']) for x in names]
        count = common_origins(len(base),base.shape[1],support)
        with np.load(args.stream, allow_pickle=False) as src:
            if not {'values','context_length'} <= set(src.files):
                raise ValueError('Event50 needs exported values and context_length, including prehistory')
            context_length = src['context_length'].item()
            validate_event_timeline(src['values'],truth,context_length)
            plans = event_boundaries(src['values'],base.shape[1],count,locked['threshold'],context_length=int(context_length))
    cfg = CommitCastConfig(feature_mode=args.feature_mode,device=args.device)
    started = time.perf_counter()
    metrics,rows = replay_schedules(base,truth,names,cfg=cfg,support=args.support,event_plans=plans)
    elapsed = time.perf_counter()-started
    args.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(metrics).to_csv(args.output/'metrics.csv',index=False)
    pd.DataFrame(rows).to_csv(args.output/'request_metrics.csv',index=False)
    (args.output/'run.json').write_text(json.dumps(dict(status='complete',schedules=names,
        support=args.support,horizon=base.shape[1],channels=base.shape[2],
        elapsed_sec=elapsed,timing_scope='ridge, exposure and request scoring; excludes Base generation and output serialization',
        config=asdict(cfg)),indent=2),encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
