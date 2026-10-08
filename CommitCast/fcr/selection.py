"""Validation-only candidate locking; the untouched Base is an explicit option."""
import math
from numbers import Real


def _published_flag(value):
    if value is None or (isinstance(value, Real) and math.isnan(value)):
        return False
    text = str(value).strip().lower()
    if text in {'', 'false', '0', '0.0'}:
        return False
    if text in {'true', '1', '1.0'}:
        return True
    raise ValueError('published_default must be a boolean, 0/1 or blank')


def choose_validation(records):
    if not records or any(r.get('split') != 'val' for r in records):
        raise ValueError('Every candidate must come from validation')
    for field in ('cell','method','policy','schedule'):
        if any(not isinstance(r.get(field),str) or not r[field] for r in records):
            raise ValueError(f'Every validation row must declare {field}')
        if len({r[field] for r in records}) != 1:
            raise ValueError(f'Cannot select across different {field} values')
    identifiers=[r['candidate'] for r in records]
    if any(not isinstance(name, str) or not name.strip() for name in identifiers):
        raise ValueError('Candidate names must be nonempty strings')
    if len(set(identifiers)) != len(identifiers) or 'Base' not in identifiers:
        raise ValueError('Require unique candidates and an explicit Base candidate')
    published = {r['candidate'] for r in records if _published_flag(r.get('published_default'))}
    if len(published) != 1 or 'Base' in published:
        raise ValueError('Mark exactly one non-Base published default candidate')
    for record in records:
        for key in ('requests', 'evaluated_elements'):
            value = record[key]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0 or int(value) != value:
                raise ValueError('Request and element counts must be positive integers')
    support={(r['requests'],r['evaluated_elements'],r['base_sse']) for r in records}
    if len(support) != 1:
        raise ValueError('Validation candidates must have the same requests and Base reference')
    if any(not math.isfinite(r['mse']) or r['mse'] < 0 for r in records):
        raise ValueError('Invalid candidate cannot be silently removed')
    for record in records:
        if record['evaluated_elements'] <= 0 or not math.isfinite(record['sse']) or record['sse'] < 0:
            raise ValueError('Invalid error sum/count')
        if not math.isclose(record['mse'],record['sse']/record['evaluated_elements'],rel_tol=1e-10,abs_tol=1e-12):
            raise ValueError('Candidate MSE disagrees with its error sum/count')
    baseline=next(r for r in records if r['candidate']=='Base')
    if baseline['evaluated_elements'] <= 0 or not math.isclose(baseline['mse'],
            baseline['base_sse']/baseline['evaluated_elements'],rel_tol=1e-10,abs_tol=1e-12):
        raise ValueError('Base candidate does not equal the shared frozen reference')
    return min(records,key=lambda r:(round(r['mse'],10),0 if r['candidate']=='Base' else
        1 if r['candidate'] in published else 2,r['candidate']))
