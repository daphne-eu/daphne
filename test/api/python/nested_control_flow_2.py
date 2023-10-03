import sys
from api.python.context.daphne_context import DaphneContext

# parsing a daphne-like (key-value) script argument
param = float(sys.argv[1].split("=")[1]) 

dctx = DaphneContext()
X = dctx.fill(param, 5, 5)

def true_fn(x): 
    return x - 1

def false_fn(x):
    return x + 1

def pred():
    return X.sum() < 10

def for_body(x, i):
    return dctx.cond([x], pred, true_fn, false_fn)

for_loop = dctx.for_loop([X], for_body, 1, 10)

daphne_output = for_loop[0].print().compute()