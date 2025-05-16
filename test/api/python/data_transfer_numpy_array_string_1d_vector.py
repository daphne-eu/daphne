# Data transfer from numpy to DAPHNE and back, via files.

import numpy as np
from daphne.context.daphne_context import DaphneContext

m1 = np.array(["apple", "banana", "cherry"], dtype=str)
    
dctx = DaphneContext()

(dctx.from_numpy(m1, shared_memory=False).print().compute())
