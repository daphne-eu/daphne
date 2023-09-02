import sys
import numpy as np
from api.python.context.daphne_context import DaphneContext

number = sys.argv[1].split("=")[1]
dctx = DaphneContext()
X = dctx.fill(number, 5, 5)
    
cond_statement = dctx.cond([X], lambda: X.sum() < 10, lambda x: x-1)
(cond_statement[0].print().compute())