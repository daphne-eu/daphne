# Data transfer from python lists to DAPHNE and back, via files.

import numpy as np
from daphne.context.daphne_context import DaphneContext

m1 = [np.int64(1), np.int64(2), np.int64(3)]

dctx = DaphneContext()

dctx.from_python(m1, shared_memory=False).print().compute()