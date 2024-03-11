from daphne.context.daphne_context import DaphneContext

dctx = DaphneContext()
X = dctx.fill(3.8, 5, 5)

def for_loop_body_nested(x, i):
    return x + 1

def for_body(x, i):
    return dctx.for_loop([x], for_loop_body_nested, 1, 5)

for_loop = dctx.for_loop([X], for_body, 1, 10)

daphne_output = for_loop[0].print().compute()