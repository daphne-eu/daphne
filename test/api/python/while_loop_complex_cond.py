from api.python.context.daphne_context import DaphneContext

dctx = DaphneContext()
X = dctx.fill(9.9, 5, 5)

def condition(x):
    return dctx.logical_and(x.sum() > 1.0, x.aggMax() > 1.0)

output = dctx.while_loop([X], condition, lambda x: x - 1)
output[0].print().compute()