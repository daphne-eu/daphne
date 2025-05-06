# Data transfer from numpy to DAPHNE and back, via files.

import numpy as np
from daphne.context.daphne_context import DaphneContext

m1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
m2 = np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=np.float64)
m3 = np.array([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]], dtype=np.float64)
m4 = np.array([[17.0, 18.0, 19.0], [20.0, 21.0, 22.0], [23.0, 24.0, 25.0]], dtype=np.float64)
    
dctx = DaphneContext()

(dctx.from_numpy(m1, shared_memory=False).print().compute())
(dctx.from_numpy(m2, shared_memory=False).print().compute())
(dctx.from_numpy(m3, shared_memory=False).print().compute())
(dctx.from_numpy(m4, shared_memory=False).print().compute())