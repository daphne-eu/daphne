from api.python.context.daphne_context import DaphneContext

dc = DaphneContext()

# TODO Currently, we cannot simply construct a DaphneLib scalar from a Python scalar.
# Thus, we use a work-around here by taking the sum of a 1x1 matrix with the desired value.

s1 = dc.fill(2.2, 1, 1)
s2 = 3.3

s1.sum().pow(s2).print().compute()
s1.sum().log(s2).print().compute()
s1.sum().mod(s2).print().compute()
s1.sum().min(s2).print().compute()
s1.sum().max(s2).print().compute()