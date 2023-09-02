import sys
from api.python.context.daphne_context import DaphneContext
# parsing a daphne-like (key-value) script argument
param = int(sys.argv[1].split("=")[1])

dctx = DaphneContext()
X = dctx.fill(3.8, 5, 5)
Y = dctx.fill(1.0, 5, 5)

def body(x, y):
    return x - 1, y + 1

output = dctx.while_loop([X], X.sum() > 0.0, body)
output[param].print().compute()