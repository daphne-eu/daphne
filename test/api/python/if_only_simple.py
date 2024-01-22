import sys
from daphne.context.daphne_context import DaphneContext

# parsing a daphne-like (key-value) script argument
param = float(sys.argv[1].split("=")[1]) 

dctx = DaphneContext()
X = dctx.fill(param, 5, 5)
    
cond_statement = dctx.cond([X], lambda: X.sum() < 10, lambda x: x-1)
(cond_statement[0].print().compute())