import sys
from api.python.context.daphne_context import DaphneContext

# parsing a daphne-like (key-value) script argument
param = float(sys.argv[1].split("=")[1]) 

dctx = DaphneContext()
X = dctx.fill(param, 5, 5)
Y = dctx.fill(1.0, 5, 5)
sumY = Y.sum()

def true_fn(x, y): 
    return x - 1, y + 1
    
def false_fn(x, y):
    return x + 1, y + 1
    
cond_statement = dctx.cond([X, sumY], lambda: X.sum() < 10, true_fn, false_fn)
((cond_statement[0] + cond_statement[1]).print().compute())