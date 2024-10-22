#!/usr/bin/python

from daphne.context.daphne_context import DaphneContext


daphne_context = DaphneContext()

m1 = daphne_context.seq(1, 100).reshape(10, 10)

row_ids = daphne_context.seq(0, 2, 1) * 3 + 1
col_ids = daphne_context.seq(0, 1, 1) * 2

# Only row
m1[0, :].print().compute()
m1[0:5, :].print().compute()
m1[0:, :].print().compute()
m1[:5, :].print().compute()
m1[row_ids, :].print().compute()

# Only col
m1[:, 0].print().compute()
m1[:, 0:5].print().compute()
m1[:, 0:].print().compute()
m1[:, :5].print().compute()
m1[:, col_ids].print().compute()

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
m1[row_ids, col_ids].print().compute()
