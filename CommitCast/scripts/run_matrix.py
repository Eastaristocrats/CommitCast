"""Generate or run the 6 datasets x 6 models x 4 horizons source-release matrix."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
DATASETS=['ETTh1','ETTh2','ETTm1','ETTm2','exchange_rate','weather']
MODELS=['DLinear','FreTS','iTransformer','MICN','OLS','PatchTST']
HORIZONS=[96,192,336,720]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--generate',action='store_true')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--methods',default='full,pg,tafas,petsa,cosa')
    p.add_argument('--data-root',type=Path,default=Path('data'))
    p.add_argument('--checkpoint-root',type=Path,default=Path('checkpoints'))
    p.add_argument('--stream-root',type=Path,default=Path('streams'))
    p.add_argument('--output',type=Path,default=Path('results/matrix'))
    p.add_argument('--schedules',default='all')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--seeds',default='0',help='Explicit seeds for manifest generation, e.g. 0,1,2')
    args=p.parse_args()
    if args.generate:
        seeds=[int(x) for x in args.seeds.split(',')]
        if not seeds or len(set(seeds))!=len(seeds) or any(x<0 for x in seeds):
            raise ValueError('Seeds must be unique nonnegative integers')
        cells=[dict(cell=f'{ds}_{model}_h{h}_s{seed}',dataset=ds,model=model,horizon=h,seed=seed,
            checkpoint_config=(args.checkpoint_root/model/f'{ds}_{h}'/f'seed_{seed}'/'config.yaml').as_posix(),
            base_stream=(args.stream_root/f'BASE_{ds}_{model}_h{h}_seed{seed}_batch48'/'adapter_stream.npz').as_posix())
            for ds,model,h,seed in itertools.product(DATASETS,MODELS,HORIZONS,seeds)]
        args.manifest.parent.mkdir(parents=True,exist_ok=True)
        with args.manifest.open('x',encoding='utf-8') as f:
            json.dump(dict(profile='common_complete_h',cells=cells),f,indent=2)
        print(f'Generated {len(cells)} cells; edit paths to checkpoint-owned config files before execution.')
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    methods=args.methods.split(',')
    if any(x not in {'full','pg','p','g','tafas','petsa','cosa'} for x in methods):
        raise ValueError('Unknown method')
    # All paths are relative to the caller, not a server-specific directory.
    for cell in manifest['cells']:
        stream=Path(cell['base_stream']).resolve()
        checkpoint=Path(cell['checkpoint_config']).resolve()
        for method in methods:
            output=(args.output/cell['cell']/method).resolve()
            if method in {'full','pg','p','g'}:
                command=[sys.executable,str(ROOT/'scripts/run_schedules.py'),'--stream',str(stream),
                    '--feature-mode',method,'--device',args.device,'--output',str(output),'--schedules',args.schedules]
            else:
                if args.device != 'cuda' and not (args.device.startswith('cuda:') and args.device[5:].isdigit()):
                    raise ValueError('Official baseline matrix runs require --device cuda:<index>')
                visible_device=args.device.split(':',1)[1] if ':' in args.device else '0'
                command=[sys.executable,str(ROOT/'main.py'),'--method',method,'--protocol','fcr',
                    '--cfg',str(ROOT/'configs'/f'{method}.yaml'),'--checkpoint-config',str(checkpoint),
                    '--base-stream',str(stream),'--output',str(output),'--schedules',args.schedules,
                    '--','DATA.BASE_DIR',str(args.data_root.resolve()),'SEED',str(cell['seed']),
                    'VISIBLE_DEVICES',visible_device]
            if args.execute:
                if not stream.is_file() or (method not in {'full','pg','p','g'} and not checkpoint.is_file()):
                    raise FileNotFoundError(f'Missing prepared stream/checkpoint for {cell["cell"]}')
                subprocess.run(command,check=True,cwd=ROOT)
            else:
                print(json.dumps(command))


if __name__=='__main__':
    main()
