# Data transfer from numpy to DAPHNE and back, via shared memory.

import numpy as np
from daphne.context.daphne_context import DaphneContext

m = np.ones((1000, 1000), dtype=np.float64)

dctx = DaphneContext()

dctx.from_numpy(m, shared_memory=True).print().compute()