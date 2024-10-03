#!/usr/bin/python

from daphne.context.daphne_context import DaphneContext


daphne_context = DaphneContext()

m1 = daphne_context.fill(123, 10, 10)

m1[0, 0] = daphne_context.fill(1001, 1, 1)
m1.print().compute()

m1[0:5, 0] = daphne_context.fill(1002, 5, 1)
m1.print().compute()

m1[:5, 0] = daphne_context.fill(1003, 5, 1)
m1.print().compute()

m1[0:, 0] = daphne_context.fill(1004, 10, 1)
m1.print().compute()

m1[:, 0] = daphne_context.fill(1005, 10, 1)
m1.print().compute()

m1[0, :] = daphne_context.fill(1006, 1, 10)
m1.print().compute()

m1[0, 0:] = daphne_context.fill(1007, 1, 10)
m1.print().compute()

m1[0, :5] = daphne_context.fill(1008, 1, 5)
m1.print().compute()

m1[0, 0:5] = daphne_context.fill(1009, 1, 5)
m1.print().compute()

m1[:, 0:5] = daphne_context.fill(1010, 10, 5)
m1.print().compute()

m1[0:5, :] = daphne_context.fill(1011, 5, 10)
m1.print().compute()
