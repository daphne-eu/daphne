# Data transfer from pytorch to DAPHNE and back, via files.

import numpy as np
import torch
from daphne.context.daphne_context import DaphneContext

t1 = torch.from_numpy(np.array([np.nan, np.nan, np.nan], dtype=np.float64).reshape(-1, 1))

dctx = DaphneContext()

(dctx.from_pytorch(t1, shared_memory=False).print().compute())
