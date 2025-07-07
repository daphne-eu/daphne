import numpy as np
import scipy.sparse as sp
from daphne.context.daphne_context import DaphneContext

# CSRMatrix(7x4, int64_t)
# 0 0 0 0
# 0 0 0 0
# 0 0 0 0
# 0 0 3 0
# 1 0 0 0
# 3 0 0 0
# 2 0 0 0

data = np.array([1, 3, 2, 3], dtype=np.int64)
indices = np.array([4, 5, 6, 3], dtype=np.uintp)
indptr = np.array([0, 3, 3, 4, 4], dtype=np.uintp)
shape = (7, 4)

csc = sp.csc_matrix((data, indices, indptr), shape=shape, dtype=np.int64)

dctx = DaphneContext()

A = dctx.from_scipy(csc, shared_memory=True)
A.print().compute()