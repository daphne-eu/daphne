from api.python.context.daphne_context import DaphneContext

dc = DaphneContext()

# TODO Currently, we cannot simply construct a DaphneLib scalar from a Python scalar.
# Thus, we use a work-around here by taking the sum of a 1x1 matrix with the desired value.

# TODO The commented out functions below are not supported yet (see #614).

dc.fill(1, 1, 1).sum().abs().print().compute()
dc.fill(0, 1, 1).sum().abs().print().compute()
dc.fill(-3.3, 1, 1).sum().abs().print().compute()

dc.fill(1, 1, 1).sum().sign().print().compute()
dc.fill(0, 1, 1).sum().sign().print().compute()
dc.fill(-3.3, 1, 1).sum().sign().print().compute()

s = dc.fill(1.23, 1, 1)

s.sum().exp().print().compute()
# s.sum().ln().print().compute()
s.sum().sqrt().print().compute()

s.sum().round().print().compute()
s.sum().floor().print().compute()
s.sum().ceil().print().compute()

# s.sum().sin().print().compute()
# s.sum().cos().print().compute()
# s.sum().tan().print().compute()
# s.sum().sinh().print().compute()
# s.sum().cosh().print().compute()
# s.sum().tanh().print().compute()
# s.sum().asin().print().compute()
# s.sum().acos().print().compute()
# s.sum().atan().print().compute()