import sys
from daphne.context.daphne_context import DaphneContext

# parsing a daphne-like (key-value) script argument
param = int(sys.argv[1].split("=")[1])

dctx = DaphneContext()
X = dctx.fill(3.8, 5, 5)
Y = dctx.fill(1.0, 5, 5)

def body(x, i):
    intermediate = x + Y
    result = intermediate + 1
    return result

output = dctx.for_loop([X], body, 1, param)
output[0].print().compute()