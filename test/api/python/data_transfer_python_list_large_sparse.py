# Data transfer from python lists to DAPHNE and back, via files.

import numpy as np
from daphne.context.daphne_context import DaphneContext

m1 = []
for i in range(0, 1000):
    m1.append([])
    for j in range(0, 1000):
        m1[i].append(np.float64(0))

dctx = DaphneContext()

dctx.from_python(m1, shared_memory=False).compute()