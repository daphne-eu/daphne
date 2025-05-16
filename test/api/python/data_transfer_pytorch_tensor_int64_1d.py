# Data transfer from pytorch to DAPHNE and back, via files.

import numpy as np
import torch
from daphne.context.daphne_context import DaphneContext

t = torch.from_numpy(np.array([0, 1, -1, 12, -12, 1000, -1000], dtype=np.int64).reshape(-1, 1))

dctx = DaphneContext()

dctx.from_pytorch(t, shared_memory=False).print().compute()