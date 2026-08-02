# Data transfer from numpy unicode strings to DAPHNE via shared memory.

import numpy as np

from daphne.context.daphne_context import DaphneContext

m1 = np.array(["red", "green", "blue"], dtype=str)

dctx = DaphneContext()
m2 = dctx.from_numpy(m1, shared_memory=True)

m2.print().compute()
