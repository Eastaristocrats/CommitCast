"""Freeze Event50 using the validation split's observed inputs only."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fcr.scoring import input_scores, validate_event_timeline
from datasets.loader import load_stream


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stream',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    metadata=json.loads(args.metadata.read_text(encoding='utf-8'))
    if metadata.get('split') != 'val':
        raise ValueError('Threshold selection requires validation metadata')
    pred, truth = load_stream(args.stream)
    if metadata.get('shape') != list(pred.shape) or metadata.get('stride') != 1:
        raise ValueError('Validation metadata does not match the stream shape/stride')
    with np.load(args.stream,allow_pickle=False) as src:
        context_length=src['context_length'].item()
        if metadata.get('input_context') != context_length:
            raise ValueError('Validation metadata does not match the encoder context')
        validate_event_timeline(src['values'],truth,context_length)
        scores=input_scores(src['values'],len(pred),context_length=int(context_length))
    finite=scores[np.isfinite(scores)]
    if not len(finite):
        raise ValueError('No validation input scores')
    result=dict(rule='observed_rms_24',selected_on='val',percentile=50,
        threshold=float(np.percentile(finite,50)),observed_decisions=len(finite),
        source=dict(stream=str(args.stream.resolve()), metadata=metadata))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(args.output)


if __name__=='__main__':
    main()
