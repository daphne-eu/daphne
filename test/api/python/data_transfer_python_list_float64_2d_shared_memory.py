# Data transfer from python lists to DAPHNE and back, via shared memory.

import numpy as np
from daphne.context.daphne_context import DaphneContext

m = [[np.nan, 0.0], [1.0, -1.0], [12.3, -12.3], [2e-10, -2e-10], [2e10, -2e10], [np.inf, -np.inf]]
    
dctx = DaphneContext()

(dctx.from_python(m, shared_memory=True).print().compute())