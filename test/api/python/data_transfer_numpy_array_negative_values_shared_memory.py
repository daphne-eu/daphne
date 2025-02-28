# Data transfer from numpy to DAPHNE and back, via shared memory.

import numpy as np
from daphne.context.daphne_context import DaphneContext

m1 = np.array([-1.0, -2.0, -3.0], dtype=np.float64).reshape(-1, 1)
    
dctx = DaphneContext()

(dctx.from_numpy(m1, shared_memory=True).print().compute())
