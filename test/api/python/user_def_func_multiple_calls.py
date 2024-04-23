from daphne.context.daphne_context import DaphneContext

dctx = DaphneContext()
X = dctx.fill(3.8, 5, 5)

@dctx.function
def increment(x):
    return x + 1

# call 1
output = increment(X)
daphne_output = output[0].compute()
# call 2
output = increment(output[0])
daphne_output = output[0].compute()
# call 3
output = increment(output[0])
daphne_output = output[0].print().compute()
