# Data transfer from pytorch to DAPHNE and back, via files.

import numpy as np
import torch
from daphne.context.daphne_context import DaphneContext

t1 = torch.from_numpy(np.zeros((1000, 1000), dtype=np.float64))

dctx = DaphneContext()

dctx.from_pytorch(t1, shared_memory=False).compute()