import sys
from daphne.context.daphne_context import DaphneContext

# parsing a daphne-like (key-value) script argument
param = float(sys.argv[1].split("=")[1]) 

dctx = DaphneContext()
X = dctx.fill(param, 5, 5)

def true_fn(x): 
    return x - 1
    
def false_fn(x):
    return x + 1
    
cond_statement = dctx.cond([X], lambda: X.sum() < 10, true_fn, false_fn)
(cond_statement[0].print().compute())