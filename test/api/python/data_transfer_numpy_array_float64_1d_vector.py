# Data transfer from numpy to DAPHNE and back, via files.

import numpy as np
from daphne.context.daphne_context import DaphneContext

m1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
m2 = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float64)
m3 = np.array([8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float64)
    
dctx = DaphneContext()

(dctx.from_numpy(m1, shared_memory=False).print().compute())
(dctx.from_numpy(m2, shared_memory=False).print().compute())
(dctx.from_numpy(m3, shared_memory=False).print().compute())