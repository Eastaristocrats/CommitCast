"""Lock one candidate from explicitly validation-tagged, matched CSV rows."""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fcr.selection import choose_validation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=pd.read_csv(args.candidates).to_dict('records')
    selected=choose_validation(rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(dict(selected_on='val',criterion='MSE; Base wins ties rounded to 10 decimals',
            comparison={key:selected[key] for key in ('cell','method','policy','schedule')},
            candidate=selected['candidate'],validation_mse=selected['mse'],
            candidates=[r['candidate'] for r in rows]),f,indent=2,allow_nan=False)
    print(args.output)


if __name__=='__main__':
    main()
