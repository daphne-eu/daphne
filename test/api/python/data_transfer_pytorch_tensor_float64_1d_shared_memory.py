# Data transfer from pytorch to DAPHNE and back, via shared memory.

import numpy as np
import torch
from daphne.context.daphne_context import DaphneContext

t1 = torch.from_numpy(np.array([np.nan, 0.0, 1.0, -1.0, 12.3, -12.3, 2e-10, -2e-10, 2e10, -2e10, np.inf, -np.inf], dtype=np.float64).reshape(-1, 1))

dctx = DaphneContext()

(dctx.from_pytorch(t1, shared_memory=True).print().compute())