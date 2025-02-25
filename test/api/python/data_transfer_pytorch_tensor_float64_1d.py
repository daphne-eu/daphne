# Data transfer from pytorch to DAPHNE and back, via files.

import numpy as np
import torch
from daphne.context.daphne_context import DaphneContext

t1 = torch.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64).reshape(-1, 1))
t2 = torch.from_numpy(np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float64).reshape(-1, 1))
t3 = torch.from_numpy(np.array([8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float64).reshape(-1, 1))
    
dctx = DaphneContext()

(dctx.from_pytorch(t1, shared_memory=False).print().compute())
(dctx.from_pytorch(t2, shared_memory=False).print().compute())
(dctx.from_pytorch(t3, shared_memory=False).print().compute())