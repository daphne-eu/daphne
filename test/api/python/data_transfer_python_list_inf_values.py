# Data transfer from python lists to DAPHNE and back, via files.

import numpy as np
from daphne.context.daphne_context import DaphneContext

m1 = [np.inf, -np.inf, np.inf]
    
dctx = DaphneContext()

(dctx.from_python(m1, shared_memory=False).print().compute())
