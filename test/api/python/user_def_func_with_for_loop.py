from api.python.context.daphne_context import DaphneContext

dctx = DaphneContext()
X = dctx.fill(3.8, 5, 5)

@dctx.function
def use_for_loop(x):
    return dctx.for_loop([x], lambda node, i: node + i, 1, 10)

output = use_for_loop(X)
daphne_output = output[0].print().compute()