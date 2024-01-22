import sys
from daphne.context.daphne_context import DaphneContext

# parsing a daphne-like (key-value) script argument
param = int(sys.argv[1].split("=")[1])

dctx = DaphneContext()
X = dctx.fill(3.8, 5, 5)

def body(x, i):
    return x + i

output = dctx.for_loop([X], body, 1, param)
output[0].print().compute()