from daphne.context.daphne_context import DaphneContext

dctx = DaphneContext()
X = dctx.fill(3.8, 5, 5)

@dctx.function
def use_while_loop(x):
    return dctx.while_loop([x], lambda node: node.sum() > 0.0, lambda node: node - 1)

output = use_while_loop(X)
daphne_output = output[0].print().compute()