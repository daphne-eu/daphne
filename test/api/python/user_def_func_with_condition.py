import sys
from daphne.context.daphne_context import DaphneContext

# parsing a daphne-like (key-value) script argument
param = float(sys.argv[1].split("=")[1]) 

dctx = DaphneContext()
X = dctx.fill(param, 5, 5)

@dctx.function
def use_condition(x):
    return dctx.cond([x], lambda: x.sum() < 10.0, lambda node: node - 1, lambda node: node + 1)

output = use_condition(X)
daphne_output = output[0].print().compute()