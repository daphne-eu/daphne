# Data transfer from numpy to DAPHNE and back, via files.

import numpy as np
from daphne.context.daphne_context import DaphneContext

m = np.array([0, 1, -1, 12, -12, 1000, -1000], dtype=np.int64).reshape(-1, 1)

dctx = DaphneContext()

dctx.from_numpy(m, shared_memory=False).print().compute()