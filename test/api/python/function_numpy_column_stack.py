import numpy as np
from daphne.context.daphne_context import DaphneContext

X1 = np.array([1, 2, 3], dtype=np.float64)
X2 = np.array([4, 5, 6], dtype=np.float64)
X3 = np.array([7, 8, 9], dtype=np.float64)

dctx = DaphneContext()

# Convert Numpy arrays to Daphne matrices
X1_daphne = dctx.from_numpy(X1.reshape(-1, 1), shared_memory=False)
X2_daphne = dctx.from_numpy(X2.reshape(-1, 1), shared_memory=False)
X3_daphne = dctx.from_numpy(X3.reshape(-1, 1), shared_memory=False)

X1c = X1_daphne.reshape(X1_daphne.ncell(), 1)
X2c = X2_daphne.reshape(X2_daphne.ncell(), 1)
X3c = X3_daphne.reshape(X3_daphne.ncell(), 1)

Y = X1c.cbind(X2c).cbind(X3c)

Y.print().compute()

