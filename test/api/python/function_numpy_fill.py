import numpy as np
from daphne.context.daphne_context import DaphneContext

m, n = 3, 2
v = 5.0

dctx = DaphneContext()

Y_daphne = dctx.fill(v, m, n)

Y_daphne.print().compute()

