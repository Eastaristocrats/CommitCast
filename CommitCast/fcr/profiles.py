"""Read published scalar script settings without executing shell code."""
import re


def published_overrides(vendor, method, dataset, model, horizon):
    if method == 'cosa':
        return []  # configs/cosa.yaml is the current scripts/cosa.sh profile.
    name = 'run.sh' if method == 'tafas' else 'run_petsa.sh'
    source = vendor/'scripts'/model/f'{dataset}_{horizon}'/name
    if not source.is_file():
        raise FileNotFoundError(f'No published per-cell profile: {source}; use --profile config for an explicit custom configuration')
    content = source.read_text(encoding='utf-8')
    def scalar(key):
        value=re.search(r'^'+key+r'\s*=\s*([+\-0-9.eE]+)\s*$',content,re.M)
        if value is None:
            raise ValueError(f'No literal {key} in {source}')
        return str(float(value.group(1)))
    values = ['TTA.SOLVER.BASE_LR',scalar('BASE_LR'),
              'TTA.SOLVER.WEIGHT_DECAY',scalar('WEIGHT_DECAY')]
    if method == 'tafas':
        values += ['TTA.TAFAS.GATING_INIT',scalar('GATING_INIT')]
    return values
