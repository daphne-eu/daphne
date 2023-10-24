import pandas as pd
from api.python.context.daphne_context import DaphneContext, Scalar, Matrix, Frame

dctx = DaphneContext()
S = dctx.fill(1.0, 1, 1).sum()
M = dctx.fill(1.0, 5, 5)
F = dctx.from_pandas(pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]}))

@dctx.function
def use_all_computed_types(s: 'Scalar', m: 'Matrix', f: 'Frame'):
    n = f.nrow()
    sumM = m.sum()
    return s + sumM*n

output = use_all_computed_types(S, M, F)
daphne_output = output[0].print().compute() 