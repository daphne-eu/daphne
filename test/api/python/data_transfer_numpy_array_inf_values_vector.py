# Data transfer from numpy to DAPHNE and back, via files.

import numpy as np
from daphne.context.daphne_context import DaphneContext

m1 = np.array([np.inf, -np.inf, np.inf], dtype=np.float64)
    
dctx = DaphneContext()

(dctx.from_numpy(m1, shared_memory=False).print().compute())
