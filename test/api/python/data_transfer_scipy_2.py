import numpy as np
import scipy.sparse as sp
from daphne.context.daphne_context import DaphneContext

# CSRMatrix(5x4, int64_t)
# 0 0 0 0
# 0 0 0 0
# 0 0 0 0
# 0 3 0 0
# 0 0 0 1

data = np.array([3, 1], dtype=np.int64)
rows = np.array([3, 4], dtype=np.uintp)
cols = np.array([1, 3], dtype=np.uintp)
shape = (5, 4)

coo = sp.coo_matrix((data, (rows, cols)), shape=shape, dtype=np.int64)

dctx = DaphneContext()

A = dctx.from_scipy(coo, shared_memory=True)
A.print().compute()

