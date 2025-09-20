# Data transfer from numpy to DAPHNE and back, via files.

import numpy as np
from daphne.context.daphne_context import DaphneContext

m1 = np.array([np.nan, 0.0, 1.0, -1.0, 12.3, -12.3, 2e-10, -2e-10, 2e10, -2e10, np.inf, -np.inf], dtype=np.float64)
    
dctx = DaphneContext()

(dctx.from_numpy(m1, shared_memory=False).print().compute())