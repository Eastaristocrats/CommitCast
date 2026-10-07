"""Read-only source comparison for the explicit baseline transport loops."""
import ast
from copy import deepcopy
import inspect
from pathlib import Path
import textwrap
import zipfile

from fcr import native_loops


def verify_native_loop(family, source_root):
    vendor = {'TAFAS':'tafas','PETSA':'petsa','COSA':'cosa_current'}[family]
    name = 'adapt_simple' if family == 'COSA' else 'adapt_' + family.lower()
    with zipfile.ZipFile(Path(source_root)/'baselines/sources'/f'{vendor}.zip') as z:
        source = ast.parse(z.read('tta/'+family.lower()+'.py').decode())
    original = deepcopy(next(n for n in ast.walk(source) if isinstance(n,ast.FunctionDef) and n.name == name))
    original.decorator_list = []
    outer = next(n for n in original.body if isinstance(n,ast.For))
    original.body = original.body[:original.body.index(outer)+1]
    loop = next(n for n in outer.body if isinstance(n,ast.While))
    def metric(n):
        if isinstance(n,ast.Assign):
            return any(isinstance(t,ast.Name) and t.id in {'mse','mae'} for t in n.targets)
        return ast.unparse(n).startswith(('self.mse_all.append(', 'self.mae_all.append('))
    if sum(metric(n) for n in loop.body) != 4:
        raise AssertionError('Unexpected upstream metric statements')
    loop.body = [n for n in loop.body if not metric(n)]
    explicit = ast.parse(textwrap.dedent(inspect.getsource(getattr(native_loops,family.lower()+'_loop')))).body[0]
    class RemoveClockEvents(ast.NodeTransformer):
        def visit_Expr(self,node):
            return None if isinstance(node.value,ast.Yield) else self.generic_visit(node)
    explicit = RemoveClockEvents().visit(explicit)
    explicit.name = original.name
    explicit.args = deepcopy(original.args)
    if ast.dump(explicit,include_attributes=False) != ast.dump(original,include_attributes=False):
        raise AssertionError(f'{family} learning statements differ from upstream')
    return dict(method=family, learning_statements_preserved=True,
                diagnostic_statements_removed=4, runtime_code_generation=False)
