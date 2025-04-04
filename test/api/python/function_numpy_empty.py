import numpy as np
from daphne.context.daphne_context import DaphneContext

dctx = DaphneContext()

Y_daphne = dctx.fill(0, 3, 2)

Y_daphne.print().compute()

