from api.python.context.daphne_context import DaphneContext

dctx = DaphneContext()
X = dctx.fill(3.8, 5, 5)

def condition():
    return dctx.logical_and(X.sum() > 1.0, X.aggMax() > 1.0)

output = dctx.while_loop([X], condition, lambda x: x - 1)
output[0].print().compute()