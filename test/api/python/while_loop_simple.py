from api.python.context.daphne_context import DaphneContext

dctx = DaphneContext()
X = dctx.fill(3.8, 5, 5)

output = dctx.while_loop([X], lambda x: x.sum() > 0, lambda x: x - 1)
output[0].print().compute()