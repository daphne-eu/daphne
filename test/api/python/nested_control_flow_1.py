import sys
from api.python.context.daphne_context import DaphneContext

# parsing a daphne-like (key-value) script argument
param = float(sys.argv[1].split("=")[1]) 

dctx = DaphneContext()
X = dctx.fill(param, 5, 5)

def true_fn(x): 
    return dctx.for_loop([x], lambda n, i: n - 1, 1, 10)

def false_fn(x):
    return dctx.for_loop([x], lambda n, i: n + 1, 1, 10)

def pred():
    return X.sum() < 10

cond_statement = dctx.cond([X], pred, true_fn, false_fn)
daphne_output = cond_statement[0].print().compute()