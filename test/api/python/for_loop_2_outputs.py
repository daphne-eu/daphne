import sys
from daphne.context.daphne_context import DaphneContext

# parsing a daphne-like (key-value) script argument
param = int(sys.argv[1].split("=")[1])

dctx = DaphneContext()
X = dctx.fill(3.8, 5, 5)
# TODO This should work with `Y = 25.0`.
Y = dctx.fill(25.0, 1, 1).sum()

def body(x, y, i):
    return x + 1, y + 1

output = dctx.for_loop([X, Y], body, 0, 10)
output[param].print().compute()