from api.python.context.daphne_context import DaphneContext

dctx = DaphneContext()
X = dctx.fill(3.8, 5, 5)

@dctx.function
def increment(x):
    return x + 2

@dctx.function
def decrement(x):
    return x - 1

@dctx.function
def add(x, y):
    return x + y

output = increment(X)
daphne_output = output[0].compute()

output = decrement(output[0])
daphne_output = output[0].compute()

output = add(X, output[0])
daphne_output = output[0].print().compute()