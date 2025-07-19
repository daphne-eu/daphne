# Data transfer from pytorch to DAPHNE and back, via shared memory.

import numpy as np
import torch
from daphne.context.daphne_context import DaphneContext

t = torch.from_numpy(np.zeros((1000, 1000), dtype=np.float64))

dctx = DaphneContext()

dctx.from_pytorch(t, shared_memory=True).print().compute()