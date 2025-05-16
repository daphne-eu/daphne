# Data transfer from pytorch to DAPHNE and back, via files.

import numpy as np
import torch
from daphne.context.daphne_context import DaphneContext

t = torch.from_numpy(np.ones((1000, 1000), dtype=np.float64))

dctx = DaphneContext()

dctx.from_pytorch(t, shared_memory=False).print().compute()