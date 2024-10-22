#!/usr/bin/env python3

from daphne.context.daphne_context import DaphneContext
dc = DaphneContext()

X = dc.fill(10, 3, 3)
Y = X.sum()

# The presence of this line should not have an impact on Y.
X[1, 1] = dc.fill(0, 1, 1)

# Should print 90.
Y.print().compute()