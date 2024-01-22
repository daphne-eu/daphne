from daphne.context.daphne_context import DaphneContext, Scalar

dctx = DaphneContext()
X = dctx.fill(2.5, 2, 2).sum()
@dctx.function
def add_to_one(x: 'Scalar'):
    return 1 + x

output = add_to_one(X)
daphne_output = output[0].print().compute() 