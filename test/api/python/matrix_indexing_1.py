#!/usr/bin/python

from daphne.context.daphne_context import DaphneContext


daphne_context = DaphneContext()

m1 = daphne_context.seq(1, 100).reshape(10, 10)

# Only row
m1[0, :].print().compute()
m1[0:5, :].print().compute()
m1[0:, :].print().compute()
m1[:5, :].print().compute()

# Only col
m1[:, 0].print().compute()
m1[:, 0:5].print().compute()
m1[:, 0:].print().compute()
m1[:, :5].print().compute()

# Row and col
m1[0:5, 0].print().compute()
m1[0, 0:5].print().compute()
m1[0, 0].print().compute()
m1[0:5, 0:5].print().compute()
m1[0:, 0].print().compute()
m1[0, 0:].print().compute()
m1[0:, 0:].print().compute()
m1[:5, 0].print().compute()
m1[0, :5].print().compute()
m1[:5, :5].print().compute()
