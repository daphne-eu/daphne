import numpy as np
from daphne.context.daphne_context import DaphneContext

X1 = np.array([1, 2, 3], dtype=np.float64)
X2 = np.array([4, 5, 6], dtype=np.float64)

dctx = DaphneContext()

X1_daphne = dctx.from_numpy(X1.reshape(-1, 1), shared_memory=False)
X2_daphne = dctx.from_numpy(X2.reshape(-1, 1), shared_memory=False)

X1c = X1_daphne.reshape(X1_daphne.ncell(), 1)
X2c = X2_daphne.reshape(X2_daphne.ncell(), 1)

Y = X1c.rbind(X2c)

Y.print().compute()

