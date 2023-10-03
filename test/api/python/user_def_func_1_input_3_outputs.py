import sys
from api.python.context.daphne_context import DaphneContext

# parsing a daphne-like (key-value) script argument
param = int(sys.argv[1].split("=")[1])

dctx = DaphneContext()
X = dctx.fill(3.8, 5, 5)

@dctx.function
def increment(x):
    return x + 1, x + 2, x + 3
output = increment(X)

daphne_output = output[param].print().compute() 